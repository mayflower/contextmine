"""Community summary generation for GraphRAG.

This module generates LLM-based summaries for communities.
Summaries are stored in KnowledgeCommunity.summary and embedded for
vector similarity search in GraphRAG retrieval.

Both LLM provider and embedding provider are REQUIRED - there is no
extractive fallback (that would defeat the purpose of GraphRAG).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

from contextmine_core.models import (
    CommunityMember,
    EmbeddingTargetType,
    KnowledgeCommunity,
    KnowledgeEmbedding,
    KnowledgeEvidence,
    KnowledgeNode,
    KnowledgeNodeEvidence,
)
from pydantic import BaseModel, Field
from sqlalchemy import select, text, update

if TYPE_CHECKING:
    from contextmine_core.research.llm import LLMProvider
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class CommunitySummaryOutput(BaseModel):
    """Structured output schema for LLM-generated community summaries."""

    title: str = Field(description="Short descriptive title (3-7 words)")
    responsibilities: list[str] = Field(
        description="What this component/module does (2-5 bullet points)",
        max_length=5,
    )
    key_concepts: list[str] = Field(
        description="Main abstractions and concepts (2-5 items)",
        max_length=5,
    )
    key_dependencies: list[str] = Field(
        description="External dependencies or related modules (0-5 items)",
        max_length=5,
        default_factory=list,
    )
    key_paths: list[str] = Field(
        description="Important code paths or files (0-3 items)",
        max_length=3,
        default_factory=list,
    )
    confidence: float = Field(ge=0, le=1, description="Confidence in this summary (0-1)")


@dataclass
class SummaryStats:
    """Statistics from summary generation."""

    communities_summarized: int = 0
    communities_skipped: int = 0
    embeddings_created: int = 0
    embeddings_skipped: int = 0


async def generate_community_summaries(
    session: AsyncSession,
    collection_id: UUID,
    provider: LLMProvider,
    embed_provider: Any,
    top_k: int = 10,
) -> SummaryStats:
    """Generate summaries and embeddings for all communities.

    GraphRAG REQUIRES LLM-generated summaries. There is no extractive fallback.

    Args:
        session: Database session
        collection_id: Collection UUID
        provider: LLM provider for narrative summaries. REQUIRED.
        embed_provider: Embedding provider for vector embeddings. REQUIRED.
        top_k: Number of top members to include in summary

    Returns:
        SummaryStats with counts

    Raises:
        ValueError: If provider or embed_provider is None
    """
    if provider is None:
        raise ValueError(
            "LLM provider is required for GraphRAG community summaries. "
            "Community summaries must be LLM-generated, not extractive."
        )
    if embed_provider is None:
        raise ValueError(
            "Embedding provider is required for GraphRAG community summaries. "
            "Community summaries must be embedded for semantic retrieval."
        )

    stats = SummaryStats()

    # Get all communities for this collection
    result = await session.execute(
        select(KnowledgeCommunity).where(KnowledgeCommunity.collection_id == collection_id)
    )
    communities = result.scalars().all()

    for community in communities:
        # Get community context (members, evidence, etc.)
        context = await _gather_community_context(session, community.id, top_k)

        # Generate LLM summary (required, no fallback)
        summary = await _llm_summarize_community(provider, context)
        summary_text = _format_llm_summary(summary)

        # Update community summary
        await session.execute(
            update(KnowledgeCommunity)
            .where(KnowledgeCommunity.id == community.id)
            .values(summary=summary_text)
        )
        stats.communities_summarized += 1

        # Generate embedding (embed_provider already validated as required)
        embedding_created = await _embed_community_summary(
            session, collection_id, community.id, summary_text, embed_provider
        )
        if embedding_created:
            stats.embeddings_created += 1
        else:
            stats.embeddings_skipped += 1

    return stats


@dataclass
class CommunityContext:
    """Context gathered for a community summary.

    For GraphRAG, communities contain SEMANTIC_ENTITY nodes (LLM-extracted
    domain concepts). The context fields are:
    - entity_names: Names of semantic entities (e.g., "User Authentication")
    - entity_types: Types (CONCEPT, COMPONENT, DATA_MODEL, etc.)
    - entity_descriptions: LLM-generated descriptions of what entities represent
    - source_symbols: Code symbols associated with entities
    """

    community_id: UUID
    level: int
    member_nodes: list[dict[str, Any]] = field(default_factory=list)
    evidence_snippets: list[str] = field(default_factory=list)
    # Semantic entity fields (GraphRAG)
    entity_names: list[str] = field(default_factory=list)
    entity_types: dict[str, int] = field(default_factory=dict)
    entity_descriptions: list[str] = field(default_factory=list)
    source_symbols: list[str] = field(default_factory=list)


async def _gather_community_context(
    session: AsyncSession,
    community_id: UUID,
    top_k: int,
) -> CommunityContext:
    """Gather context for a community summary.

    GraphRAG communities contain SEMANTIC_ENTITY nodes (LLM-extracted
    domain concepts). Non-semantic nodes are skipped with a warning.

    Args:
        session: Database session
        community_id: Community UUID
        top_k: Number of top members to include

    Returns:
        CommunityContext with members, evidence, etc.
    """
    from contextmine_core.models import KnowledgeNodeKind

    # Get community info
    result = await session.execute(
        select(KnowledgeCommunity).where(KnowledgeCommunity.id == community_id)
    )
    community = result.scalar_one()

    context = CommunityContext(
        community_id=community_id,
        level=community.level,
    )

    # Get top members by score
    result = await session.execute(
        select(CommunityMember, KnowledgeNode)
        .join(KnowledgeNode, CommunityMember.node_id == KnowledgeNode.id)
        .where(CommunityMember.community_id == community_id)
        .order_by(CommunityMember.score.desc())
        .limit(top_k)
    )
    members = result.all()

    for member, node in members:
        # GraphRAG communities contain SEMANTIC_ENTITY nodes only
        if node.kind != KnowledgeNodeKind.SEMANTIC_ENTITY:
            logger.warning(
                "Unexpected node kind %s in community %s. "
                "GraphRAG communities should only contain SEMANTIC_ENTITY nodes.",
                node.kind,
                community_id,
            )
            continue

        node_info = {
            "name": node.name,
            "type": node.meta.get("type", "unknown"),
            "description": node.meta.get("description", ""),
            "aliases": node.meta.get("aliases", []),
            "source_symbols": node.meta.get("source_symbols", []),
            "score": member.score,
        }
        context.member_nodes.append(node_info)

        # Track entity names
        if node.name:
            context.entity_names.append(node.name)

        # Track entity types
        entity_type = node.meta.get("type", "unknown")
        context.entity_types[entity_type] = context.entity_types.get(entity_type, 0) + 1

        # Track descriptions
        if node.meta.get("description"):
            context.entity_descriptions.append(node.meta["description"])

        # Track source symbols
        for sym in node.meta.get("source_symbols", []):
            if sym not in context.source_symbols:
                context.source_symbols.append(sym)

        # Get evidence snippets
        ev_result = await session.execute(
            select(KnowledgeEvidence)
            .join(KnowledgeNodeEvidence, KnowledgeEvidence.id == KnowledgeNodeEvidence.evidence_id)
            .where(KnowledgeNodeEvidence.node_id == node.id)
            .limit(3)
        )
        for evidence in ev_result.scalars():
            if evidence.snippet and len(evidence.snippet) < 500:
                context.evidence_snippets.append(evidence.snippet)

    return context


async def _llm_summarize_community(
    provider: LLMProvider,
    context: CommunityContext,
) -> CommunitySummaryOutput:
    """Generate LLM summary of community.

    Args:
        provider: LLM provider
        context: Community context

    Returns:
        CommunitySummaryOutput with structured summary
    """
    # Build prompt
    prompt = _build_summary_prompt(context)

    # Call LLM with structured output
    result = await provider.generate_structured(
        system=(
            "You are a code analyst summarizing a software component. "
            "Only use information provided. Do not hallucinate."
        ),
        messages=[{"role": "user", "content": prompt}],
        output_schema=CommunitySummaryOutput,
        temperature=0,  # Deterministic
    )

    return result


def _build_summary_prompt(context: CommunityContext) -> str:
    """Build prompt for LLM community summary.

    GraphRAG communities contain semantic entities only.

    Args:
        context: Community context with semantic entities

    Returns:
        Formatted prompt string
    """
    lines = [
        "Summarize this software component based on the following facts:",
        "",
        f"Level: {context.level}",
        f"Member count: {len(context.member_nodes)}",
        "",
    ]

    # GraphRAG uses semantic entities (LLM-extracted domain concepts)
    lines.append("## Domain Concepts (Semantic Entities)")
    for node in context.member_nodes[:10]:
        entity_type = node.get("type", "unknown")
        lines.append(f"- **{node['name']}** ({entity_type})")
        if node.get("description"):
            lines.append(f"  {node['description']}")

    if context.entity_types:
        lines.extend(["", "## Entity Types"])
        for entity_type, count in sorted(context.entity_types.items(), key=lambda x: -x[1]):
            lines.append(f"- {count} {entity_type}")

    if context.source_symbols:
        lines.extend(["", "## Associated Code Symbols"])
        for sym in context.source_symbols[:10]:
            lines.append(f"- {sym}")

    if context.entity_descriptions:
        lines.extend(["", "## Entity Descriptions"])
        for desc in context.entity_descriptions[:5]:
            lines.append(f"- {desc}")

    if context.evidence_snippets:
        lines.extend(["", "## Code Snippets"])
        for snippet in context.evidence_snippets[:3]:
            lines.append(f"```\n{snippet[:200]}\n```")

    return "\n".join(lines)


def _format_llm_summary(summary: CommunitySummaryOutput) -> str:
    """Format LLM summary to text.

    Args:
        summary: Structured summary

    Returns:
        Formatted text summary
    """
    lines = [f"# {summary.title}", ""]

    if summary.responsibilities:
        lines.append("## Responsibilities")
        for r in summary.responsibilities:
            lines.append(f"- {r}")
        lines.append("")

    if summary.key_concepts:
        lines.append("## Key Concepts")
        for c in summary.key_concepts:
            lines.append(f"- {c}")
        lines.append("")

    if summary.key_dependencies:
        lines.append("## Dependencies")
        for d in summary.key_dependencies:
            lines.append(f"- {d}")
        lines.append("")

    if summary.key_paths:
        lines.append("## Key Paths")
        for p in summary.key_paths:
            lines.append(f"- {p}")
        lines.append("")

    lines.append(f"Confidence: {summary.confidence:.0%}")
    return "\n".join(lines)


async def _embed_community_summary(
    session: AsyncSession,
    collection_id: UUID,
    community_id: UUID,
    summary_text: str,
    embed_provider: Any,
) -> bool:
    """Embed community summary and store.

    Args:
        session: Database session
        collection_id: Collection UUID
        community_id: Community UUID
        summary_text: Summary text to embed
        embed_provider: Embedding provider (must have embed_batch method)

    Returns:
        True if embedding created, False if skipped (unchanged)
    """
    # Compute content hash for idempotency
    content_hash = hashlib.sha256(summary_text.encode()).hexdigest()[:32]

    # Check if embedding already exists with same hash
    result = await session.execute(
        select(KnowledgeEmbedding).where(
            KnowledgeEmbedding.collection_id == collection_id,
            KnowledgeEmbedding.target_type == EmbeddingTargetType.COMMUNITY,
            KnowledgeEmbedding.target_id == community_id,
        )
    )
    existing = result.scalar_one_or_none()

    if existing and existing.content_hash == content_hash:
        logger.debug("Skipping unchanged embedding for community %s", community_id)
        return False

    # Generate embedding using embed_batch (the standard interface)
    try:
        embed_result = await embed_provider.embed_batch([summary_text])
        embedding_vector = embed_result.embeddings[0]
    except Exception as e:
        logger.warning("Failed to embed community %s: %s", community_id, e)
        return False

    # Convert vector to pgvector format string
    embedding_str = "[" + ",".join(str(x) for x in embedding_vector) + "]"

    # Create or update embedding record
    if existing:
        # Update existing - delete and recreate for simplicity
        await session.delete(existing)
        await session.flush()

    # Get model info from provider
    model_name = getattr(embed_provider, "model_name", "unknown")
    provider_name = getattr(embed_provider, "provider", "unknown")
    if hasattr(embed_provider, "_model_name"):
        model_name = embed_provider._model_name
    if hasattr(embed_provider, "_provider"):
        provider_name = str(
            embed_provider._provider.value
            if hasattr(embed_provider._provider, "value")
            else embed_provider._provider
        )

    # Create new embedding
    embedding = KnowledgeEmbedding(
        collection_id=collection_id,
        target_type=EmbeddingTargetType.COMMUNITY,
        target_id=community_id,
        model_name=model_name,
        provider=provider_name,
        content_hash=content_hash,
    )
    session.add(embedding)
    await session.flush()  # Get embedding.id

    # Set vector via parameterized SQL (SQLAlchemy doesn't handle pgvector directly)
    await session.execute(
        text(
            "UPDATE knowledge_embeddings SET embedding = CAST(:embedding AS vector) "
            "WHERE id = :embedding_id"
        ),
        {"embedding": embedding_str, "embedding_id": str(embedding.id)},
    )

    return True
