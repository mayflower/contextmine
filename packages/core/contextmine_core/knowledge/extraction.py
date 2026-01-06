"""LLM-based entity and relationship extraction for GraphRAG.

This module implements the entity/relationship extraction from the Microsoft
GraphRAG paper "From Local to Global: A Graph RAG Approach to Query-Focused
Summarization".

Instead of relying on syntactic symbols from tree-sitter (which can't understand
that User, Account, Member are semantically related), we use LLM to extract:

1. Semantic entities (domain concepts, not just code symbols)
2. Relationships between entities
3. Entity descriptions for later summarization

The extracted entities form the basis for Leiden community detection.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from contextmine_core.research.llm import LLMProvider
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Pydantic schemas for structured LLM output
# -----------------------------------------------------------------------------


class ExtractedEntity(BaseModel):
    """An entity extracted from code by LLM."""

    name: str = Field(
        description="Canonical name for this entity (e.g., 'User Authentication', 'Payment Processing')"
    )
    type: str = Field(
        description="Entity type: CONCEPT, COMPONENT, DATA_MODEL, PATTERN, SERVICE, or OTHER"
    )
    description: str = Field(description="Brief description of what this entity represents")
    aliases: list[str] = Field(
        default_factory=list,
        description="Alternative names/symbols that refer to this entity (e.g., ['User', 'Account', 'Member'])",
    )
    source_symbols: list[str] = Field(
        default_factory=list,
        description="Code symbols (class/function names) associated with this entity",
    )


class ExtractedRelationship(BaseModel):
    """A relationship between entities extracted by LLM."""

    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    relationship: str = Field(
        description="Type of relationship: USES, CONTAINS, DEPENDS_ON, IMPLEMENTS, VALIDATES, TRANSFORMS, CALLS, or OTHER"
    )
    description: str = Field(description="Brief description of the relationship")
    strength: float = Field(ge=0, le=1, description="Relationship strength (0-1)")


class ExtractionResult(BaseModel):
    """Complete extraction result from a code chunk."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompts
# -----------------------------------------------------------------------------

ENTITY_EXTRACTION_SYSTEM = """You are a code analyst extracting semantic entities from source code.

Your task is to identify high-level DOMAIN CONCEPTS, not just code symbols.

For example, if you see classes named User, Account, Member, AuthService, LoginController:
- Extract ONE entity called "User Authentication" or "Identity Management"
- List User, Account, Member, AuthService, LoginController as source_symbols
- List "user", "account", "member", "auth" as aliases

Entity types:
- CONCEPT: Domain concepts (authentication, payment processing, data validation)
- COMPONENT: Architectural components (API layer, cache service, message queue)
- DATA_MODEL: Core data entities (User, Order, Product - the domain model)
- PATTERN: Design patterns (Repository, Factory, Observer)
- SERVICE: External services or integrations (email service, payment gateway)
- OTHER: Anything else significant

Focus on SEMANTIC meaning, not syntactic structure. Group related symbols into single entities."""

ENTITY_EXTRACTION_PROMPT = """Analyze this code and extract semantic entities.

CODE:
```
{code}
```

FILE PATH: {file_path}

Extract entities that represent meaningful domain concepts. Group related code symbols
(classes, functions, variables) into single semantic entities.

Remember:
- User, Account, Member, Profile might all be ONE entity "User Management"
- authenticate(), login(), validatePassword() might all be ONE entity "Authentication"
- OrderService, OrderRepository, OrderController might all be ONE entity "Order Processing"

Return a JSON object with 'entities' array."""

RELATIONSHIP_EXTRACTION_SYSTEM = """You are a code analyst extracting relationships between domain entities.

Given a list of entities and the code they came from, identify how entities relate to each other.

Relationship types:
- USES: Entity A uses/calls Entity B
- CONTAINS: Entity A contains/owns Entity B
- DEPENDS_ON: Entity A depends on Entity B
- IMPLEMENTS: Entity A implements/realizes Entity B
- VALIDATES: Entity A validates Entity B
- TRANSFORMS: Entity A transforms/converts Entity B
- CALLS: Entity A invokes Entity B
- OTHER: Other significant relationship

Focus on SEMANTIC relationships between domain concepts, not just code-level function calls."""

RELATIONSHIP_EXTRACTION_PROMPT = """Given these entities extracted from code, identify relationships between them.

ENTITIES:
{entities}

CODE CONTEXT:
```
{code}
```

Identify meaningful relationships between the entities. Focus on domain-level relationships,
not just which function calls which.

Return a JSON object with 'relationships' array."""


# -----------------------------------------------------------------------------
# Dataclasses for internal use
# -----------------------------------------------------------------------------


@dataclass
class SemanticEntity:
    """A semantic entity for the knowledge graph."""

    id: str  # Hash-based ID
    name: str
    entity_type: str
    description: str
    aliases: list[str] = field(default_factory=list)
    source_symbols: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)

    @property
    def natural_key(self) -> str:
        """Natural key for deduplication."""
        return f"entity:{self.name.lower().replace(' ', '_')}"


@dataclass
class SemanticRelationship:
    """A semantic relationship between entities."""

    source_entity: str  # Entity name
    target_entity: str  # Entity name
    relationship_type: str
    description: str
    strength: float = 1.0


@dataclass
class ExtractionBatch:
    """Result of extracting from multiple code chunks."""

    entities: list[SemanticEntity] = field(default_factory=list)
    relationships: list[SemanticRelationship] = field(default_factory=list)
    chunks_processed: int = 0
    errors: list[str] = field(default_factory=list)

    # Track which entities appear in which documents (for cross-doc relationships)
    # document_uri -> list of entity names (before resolution)
    entity_occurrences: dict[str, list[str]] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Main extraction functions
# -----------------------------------------------------------------------------


async def extract_entities_from_chunk(
    llm_provider: LLMProvider,
    code: str,
    file_path: str,
) -> list[ExtractedEntity]:
    """Extract semantic entities from a code chunk using LLM.

    Args:
        llm_provider: LLM provider for extraction
        code: Source code text
        file_path: Path to the source file

    Returns:
        List of extracted entities
    """
    if not code.strip():
        return []

    # Truncate very long code
    if len(code) > 8000:
        code = code[:8000] + "\n... (truncated)"

    prompt = ENTITY_EXTRACTION_PROMPT.format(code=code, file_path=file_path)

    try:
        result = await llm_provider.generate_structured(
            system=ENTITY_EXTRACTION_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_schema=ExtractionResult,
            temperature=0,
            max_tokens=2000,
        )
        return result.entities
    except Exception as e:
        logger.warning("Entity extraction failed for %s: %s", file_path, e)
        return []


async def extract_relationships_from_chunk(
    llm_provider: LLMProvider,
    entities: list[ExtractedEntity],
    code: str,
) -> list[ExtractedRelationship]:
    """Extract relationships between entities using LLM.

    Args:
        llm_provider: LLM provider for extraction
        entities: Entities to find relationships between
        code: Source code context

    Returns:
        List of extracted relationships
    """
    if not entities or len(entities) < 2:
        return []

    # Format entities for prompt
    entities_text = "\n".join(f"- {e.name} ({e.type}): {e.description}" for e in entities)

    # Truncate code
    if len(code) > 4000:
        code = code[:4000] + "\n... (truncated)"

    prompt = RELATIONSHIP_EXTRACTION_PROMPT.format(
        entities=entities_text,
        code=code,
    )

    try:
        result = await llm_provider.generate_structured(
            system=RELATIONSHIP_EXTRACTION_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_schema=ExtractionResult,
            temperature=0,
            max_tokens=1500,
        )
        return result.relationships
    except Exception as e:
        logger.warning("Relationship extraction failed: %s", e)
        return []


async def extract_from_documents(
    session: AsyncSession,
    collection_id: UUID,
    llm_provider: LLMProvider,
    embedder: Any,
    max_chunks: int = 100,
    similarity_threshold: float = 0.85,
) -> ExtractionBatch:
    """Extract entities and relationships from all documents in a collection.

    Implements full GraphRAG extraction including:
    1. Per-document entity extraction (LLM)
    2. Per-document relationship extraction (LLM)
    3. Entity resolution via embedding similarity (merge semantically similar entities)
    4. Cross-document relationship inference (co-occurrence)

    Args:
        session: Database session
        collection_id: Collection to process
        llm_provider: LLM provider for extraction
        embedder: Embedding provider for semantic entity resolution. REQUIRED.
                  Uses embedding similarity to merge entities across languages,
                  naming conventions, and synonyms.
        max_chunks: Maximum chunks to process (for cost control)
        similarity_threshold: Cosine similarity threshold for entity merging (0.85 default)

    Returns:
        ExtractionBatch with all extracted entities and relationships

    Raises:
        ValueError: If embedder is not provided
    """
    if embedder is None:
        raise ValueError(
            "embedder is required for GraphRAG entity extraction. "
            "Entity resolution requires embedding similarity to work across "
            "languages, naming conventions, and synonyms."
        )

    from contextmine_core.models import Document, Source
    from sqlalchemy import select

    batch = ExtractionBatch()

    # Get documents via source (collection_id is on Source, not Document)
    stmt = (
        select(Document)
        .join(Source, Document.source_id == Source.id)
        .where(
            Source.collection_id == collection_id,
            Document.content_markdown.isnot(None),
        )
        .limit(max_chunks)
    )

    result = await session.execute(stmt)
    documents = result.scalars().all()

    all_entities: list[ExtractedEntity] = []

    for doc in documents:
        if not doc.content_markdown:
            continue

        doc_uri = doc.uri or str(doc.id)

        # Extract entities from this document
        entities = await extract_entities_from_chunk(
            llm_provider=llm_provider,
            code=doc.content_markdown,
            file_path=doc_uri,
        )

        if entities:
            # Track source file for each entity
            for entity in entities:
                entity.source_symbols.append(doc_uri)

            # Track which entities appear in this document (for cross-doc relationships)
            batch.entity_occurrences[doc_uri] = [e.name for e in entities]

            all_entities.extend(entities)

            # Extract relationships within this document
            relationships = await extract_relationships_from_chunk(
                llm_provider=llm_provider,
                entities=entities,
                code=doc.content_markdown,
            )

            for rel in relationships:
                batch.relationships.append(
                    SemanticRelationship(
                        source_entity=rel.source,
                        target_entity=rel.target,
                        relationship_type=rel.relationship,
                        description=rel.description,
                        strength=rel.strength,
                    )
                )

        batch.chunks_processed += 1

    # Resolve and merge entities across documents using embedding similarity
    # This handles multi-language, naming conventions, abbreviations, synonyms
    batch.entities = await _resolve_entities_with_embeddings(
        all_entities, embedder, similarity_threshold
    )
    logger.info(
        "Entity resolution: %d entities merged to %d using embedding similarity (threshold=%.2f)",
        len(all_entities),
        len(batch.entities),
        similarity_threshold,
    )

    # Create cross-document relationships based on co-occurrence
    # Entities that appear in the same document are implicitly related
    cross_doc_relationships = _extract_cross_document_relationships(
        batch.entities,
        batch.entity_occurrences,
        batch.relationships,
    )
    batch.relationships.extend(cross_doc_relationships)

    logger.info(
        "Extracted %d entities, %d relationships (%d cross-doc) from %d chunks",
        len(batch.entities),
        len(batch.relationships),
        len(cross_doc_relationships),
        batch.chunks_processed,
    )

    return batch


async def _resolve_entities_with_embeddings(
    entities: list[ExtractedEntity],
    embedder: Any,
    similarity_threshold: float = 0.85,
) -> list[SemanticEntity]:
    """Resolve and merge similar entities using embedding similarity.

    This is the proper GraphRAG entity resolution that works across:
    - Different languages (User vs Benutzer vs Usuario)
    - Different naming conventions (user_auth vs UserAuth)
    - Abbreviations (auth vs authentication)
    - Synonyms (customer vs client vs user)

    Args:
        entities: Raw extracted entities
        embedder: Embedding provider with embed_texts() method
        similarity_threshold: Cosine similarity threshold for merging (0.85 = very similar)

    Returns:
        Deduplicated and merged semantic entities
    """
    import numpy as np

    if not entities:
        return []

    # Create embedding text for each entity (name + description for context)
    embed_texts = [f"{e.name}: {e.description}" for e in entities]

    # Get embeddings
    embeddings = await embedder.embed_texts(embed_texts)
    embeddings_array = np.array(embeddings)

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized = embeddings_array / norms

    # Compute pairwise cosine similarity
    similarity_matrix = np.dot(normalized, normalized.T)

    # Use Union-Find to group similar entities
    parent = list(range(len(entities)))

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Union entities above similarity threshold
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            if similarity_matrix[i, j] >= similarity_threshold:
                union(i, j)

    # Group entities by their root parent
    groups: dict[int, list[ExtractedEntity]] = {}
    for idx, entity in enumerate(entities):
        root = find(idx)
        if root not in groups:
            groups[root] = []
        groups[root].append(entity)

    # Merge each group
    return _merge_entity_groups(groups)


def _merge_entity_groups(groups: dict[int, list[ExtractedEntity]]) -> list[SemanticEntity]:
    """Merge grouped entities into single SemanticEntity instances."""
    merged: list[SemanticEntity] = []

    for group in groups.values():
        if not group:
            continue

        # Pick the most descriptive name (longest description = most context)
        best_name = max(group, key=lambda e: len(e.description)).name

        # Merge all data from the group
        all_aliases: set[str] = set()
        all_symbols: set[str] = set()
        all_files: set[str] = set()
        descriptions: list[str] = []
        types: list[str] = []

        for entity in group:
            all_aliases.update(entity.aliases)
            all_aliases.add(entity.name)  # Original name becomes alias
            all_symbols.update(entity.source_symbols)
            descriptions.append(entity.description)
            types.append(entity.type)

        # Pick most common type
        type_counts: dict[str, int] = {}
        for t in types:
            type_counts[t] = type_counts.get(t, 0) + 1
        best_type = max(type_counts.keys(), key=lambda t: type_counts[t])

        # Merge descriptions - take the longest (most detailed)
        merged_desc = descriptions[0] if descriptions else ""
        if len(descriptions) > 1:
            merged_desc = max(descriptions, key=len)

        # Generate stable ID from canonical name
        canonical = best_name.lower().strip()
        entity_id = hashlib.sha256(canonical.encode()).hexdigest()[:16]

        merged.append(
            SemanticEntity(
                id=entity_id,
                name=best_name,
                entity_type=best_type,
                description=merged_desc,
                aliases=list(all_aliases - {best_name}),
                source_symbols=list(all_symbols),
                source_files=list(all_files),
            )
        )

    return merged


def _extract_cross_document_relationships(
    resolved_entities: list[SemanticEntity],
    entity_occurrences: dict[str, list[str]],
    existing_relationships: list[SemanticRelationship],
) -> list[SemanticRelationship]:
    """Extract relationships between entities that co-occur across documents.

    This implements the cross-document relationship inference from GraphRAG:
    - Entities appearing in the same document are implicitly related
    - Relationship strength is based on co-occurrence frequency
    - Only creates relationships not already captured by LLM extraction

    Args:
        resolved_entities: Merged semantic entities
        entity_occurrences: Map of document_uri -> list of original entity names
        existing_relationships: Already extracted relationships (to avoid duplicates)

    Returns:
        List of co-occurrence relationships
    """
    relationships: list[SemanticRelationship] = []

    # Build mapping from original names/aliases to resolved entity names
    # This handles entity resolution (e.g., "User" -> "User Management")
    name_to_resolved: dict[str, str] = {}
    for entity in resolved_entities:
        # Map the canonical name
        name_to_resolved[entity.name.lower()] = entity.name
        # Map all aliases
        for alias in entity.aliases:
            name_to_resolved[alias.lower()] = entity.name

    # Track existing relationship pairs to avoid duplicates
    existing_pairs: set[tuple[str, str]] = set()
    for rel in existing_relationships:
        # Normalize to lowercase for comparison
        src = rel.source_entity.lower()
        tgt = rel.target_entity.lower()
        existing_pairs.add((src, tgt))
        existing_pairs.add((tgt, src))  # Bidirectional check

    # Count co-occurrences across documents
    # co_occurrence[(entity_a, entity_b)] = number of documents they share
    co_occurrence: dict[tuple[str, str], int] = {}
    co_occurrence_docs: dict[tuple[str, str], list[str]] = {}

    for doc_uri, original_names in entity_occurrences.items():
        # Resolve original names to canonical entity names
        resolved_names_in_doc: set[str] = set()
        for orig_name in original_names:
            resolved = name_to_resolved.get(orig_name.lower())
            if resolved:
                resolved_names_in_doc.add(resolved)

        # Create pairs of co-occurring entities in this document
        resolved_list = sorted(resolved_names_in_doc)
        for i, entity_a in enumerate(resolved_list):
            for entity_b in resolved_list[i + 1 :]:
                # Skip self-relationships
                if entity_a == entity_b:
                    continue

                # Canonical ordering for consistent keys
                pair = (entity_a, entity_b) if entity_a < entity_b else (entity_b, entity_a)

                # Skip if already have explicit relationship
                if (entity_a.lower(), entity_b.lower()) in existing_pairs:
                    continue

                co_occurrence[pair] = co_occurrence.get(pair, 0) + 1
                if pair not in co_occurrence_docs:
                    co_occurrence_docs[pair] = []
                co_occurrence_docs[pair].append(doc_uri)

    # Create relationships for co-occurring entity pairs
    for (entity_a, entity_b), count in co_occurrence.items():
        # Calculate strength based on co-occurrence frequency
        # More co-occurrences = stronger implicit relationship
        # Use logarithmic scaling: 1 doc = 0.3, 2 docs = 0.5, 5+ docs = 0.8
        if count >= 5:
            strength = 0.8
        elif count >= 3:
            strength = 0.6
        elif count >= 2:
            strength = 0.5
        else:
            strength = 0.3

        doc_list = co_occurrence_docs.get((entity_a, entity_b), [])
        doc_summary = ", ".join(doc_list[:3])
        if len(doc_list) > 3:
            doc_summary += f" +{len(doc_list) - 3} more"

        relationships.append(
            SemanticRelationship(
                source_entity=entity_a,
                target_entity=entity_b,
                relationship_type="CO_OCCURS",
                description=f"Co-occur in {count} document(s): {doc_summary}",
                strength=strength,
            )
        )

    return relationships


# -----------------------------------------------------------------------------
# Persistence functions
# -----------------------------------------------------------------------------


async def persist_semantic_entities(
    session: AsyncSession,
    collection_id: UUID,
    batch: ExtractionBatch,
) -> dict:
    """Persist extracted entities and relationships to the knowledge graph.

    Creates SEMANTIC_ENTITY nodes, SEMANTIC_RELATIONSHIP edges, and
    FILE_MENTIONS_ENTITY edges to connect semantic entities to their
    source files (for graph connectivity in retrieval).

    Args:
        session: Database session
        collection_id: Collection UUID
        batch: Extraction batch to persist

    Returns:
        Stats dict
    """
    from contextmine_core.models import (
        KnowledgeEdge,
        KnowledgeEdgeKind,
        KnowledgeNode,
        KnowledgeNodeKind,
    )
    from sqlalchemy import delete, select

    stats = {"entities_created": 0, "relationships_created": 0, "file_edges_created": 0}

    # Delete existing semantic entities for this collection
    # (We'll rebuild from scratch each time for simplicity)
    existing = await session.execute(
        select(KnowledgeNode.id).where(
            KnowledgeNode.collection_id == collection_id,
            KnowledgeNode.kind == KnowledgeNodeKind.SEMANTIC_ENTITY,
        )
    )
    existing_ids = [row[0] for row in existing.fetchall()]

    if existing_ids:
        await session.execute(
            delete(KnowledgeEdge).where(
                KnowledgeEdge.source_node_id.in_(existing_ids)
                | KnowledgeEdge.target_node_id.in_(existing_ids)
            )
        )
        await session.execute(delete(KnowledgeNode).where(KnowledgeNode.id.in_(existing_ids)))

    # Build lookup of FILE nodes by URI for graph connectivity
    # This enables neighborhood traversal from search results to semantic entities
    file_result = await session.execute(
        select(KnowledgeNode.id, KnowledgeNode.natural_key).where(
            KnowledgeNode.collection_id == collection_id,
            KnowledgeNode.kind == KnowledgeNodeKind.FILE,
        )
    )
    file_node_by_uri: dict[str, UUID] = {}
    for node_id, natural_key in file_result.all():
        # natural_key is "file:uri" - extract the uri part
        if natural_key.startswith("file:"):
            uri = natural_key[5:]
            file_node_by_uri[uri] = node_id

    # Create entity nodes
    entity_name_to_id: dict[str, UUID] = {}
    entity_source_files: dict[str, list[str]] = {}  # entity_name -> list of source URIs

    for entity in batch.entities:
        node = KnowledgeNode(
            collection_id=collection_id,
            kind=KnowledgeNodeKind.SEMANTIC_ENTITY,
            natural_key=entity.natural_key,
            name=entity.name,
            meta={
                "type": entity.entity_type,
                "description": entity.description,
                "aliases": entity.aliases,
                "source_symbols": entity.source_symbols,
                "source_files": entity.source_files,
            },
        )
        session.add(node)
        await session.flush()
        entity_name_to_id[entity.name.lower()] = node.id
        # Track source files for this entity
        entity_source_files[entity.name.lower()] = entity.source_symbols + entity.source_files
        stats["entities_created"] += 1

    # Create semantic relationship edges
    for rel in batch.relationships:
        source_id = entity_name_to_id.get(rel.source_entity.lower())
        target_id = entity_name_to_id.get(rel.target_entity.lower())

        if not source_id or not target_id:
            continue

        edge = KnowledgeEdge(
            collection_id=collection_id,
            source_node_id=source_id,
            target_node_id=target_id,
            kind=KnowledgeEdgeKind.SEMANTIC_RELATIONSHIP,
            meta={
                "relationship_type": rel.relationship_type,
                "description": rel.description,
                "strength": rel.strength,
            },
        )
        session.add(edge)
        stats["relationships_created"] += 1

    # Create FILE_MENTIONS_ENTITY edges to connect the code graph to semantic graph
    # This is critical for GraphRAG retrieval to work properly
    created_file_edges: set[tuple[UUID, UUID]] = set()
    for entity_name, entity_id in entity_name_to_id.items():
        source_uris = entity_source_files.get(entity_name, [])
        for uri in source_uris:
            file_id = file_node_by_uri.get(uri)
            if file_id and (file_id, entity_id) not in created_file_edges:
                edge = KnowledgeEdge(
                    collection_id=collection_id,
                    source_node_id=file_id,
                    target_node_id=entity_id,
                    kind=KnowledgeEdgeKind.FILE_MENTIONS_ENTITY,
                    meta={"source_uri": uri},
                )
                session.add(edge)
                created_file_edges.add((file_id, entity_id))
                stats["file_edges_created"] += 1

    logger.info(
        "Persisted %d entities, %d relationships, %d file edges",
        stats["entities_created"],
        stats["relationships_created"],
        stats["file_edges_created"],
    )

    return stats
