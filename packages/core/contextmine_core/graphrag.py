"""GraphRAG retrieval service.

Implements GraphRAG as described in "From Local to Global: A Graph RAG
Approach to Query-Focused Summarization" (Microsoft Research, 2024).

Key components:
1. Leiden-based hierarchical community detection (in communities.py)
2. LLM-generated community summaries (in summaries.py)
3. Map-reduce query answering:
   - MAP: Generate partial answers from each relevant community
   - REDUCE: Combine partial answers into final response
4. Local context: Entity neighborhood expansion from search seeds
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

if TYPE_CHECKING:
    from contextmine_core.research.llm import LLMProvider
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """Evidence citation for a claim."""

    file_path: str
    start_line: int
    end_line: int
    snippet: str | None = None

    def format(self) -> str:
        """Format as human-readable string."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


@dataclass
class CommunityContext:
    """Global context from a community summary."""

    community_id: UUID
    level: int
    title: str
    summary: str
    relevance_score: float
    member_count: int


@dataclass
class EntityContext:
    """Local context from an entity node."""

    node_id: UUID
    kind: str
    natural_key: str
    name: str
    evidence: list[Citation] = field(default_factory=list)
    relevance_score: float = 0.0


@dataclass
class EdgeContext:
    """Edge in the context graph."""

    source_id: str
    target_id: str
    kind: str
    source_name: str | None = None
    target_name: str | None = None


@dataclass
class PathContext:
    """A path between entities."""

    nodes: list[str]  # node natural_keys in order
    edges: list[str]  # edge kinds in order
    description: str = ""


@dataclass
class ContextPack:
    """Complete GraphRAG context for a query."""

    query: str

    # Global context (from communities)
    communities: list[CommunityContext] = field(default_factory=list)

    # Local context (from entities)
    entities: list[EntityContext] = field(default_factory=list)

    # Graph structure
    edges: list[EdgeContext] = field(default_factory=list)
    paths: list[PathContext] = field(default_factory=list)

    # All citations
    citations: list[Citation] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Render as markdown for LLM consumption."""
        lines = [f"# GraphRAG Context: {self.query}\n"]

        lines.append(
            f"Found {len(self.communities)} communities, {len(self.entities)} entities, "
            f"{len(self.citations)} citations.\n"
        )

        if self.communities:
            lines.append("## Global Context (Community Summaries)\n")
            for comm in self.communities:
                lines.append(f"### {comm.title} (Level {comm.level})")
                lines.append(
                    f"*Relevance: {comm.relevance_score:.0%}, {comm.member_count} members*\n"
                )
                if comm.summary:
                    summary = (
                        comm.summary[:500] + "..." if len(comm.summary) > 500 else comm.summary
                    )
                    lines.append(summary)
                lines.append("")

        if self.entities:
            lines.append("## Local Context (Entities)\n")
            by_kind: dict[str, list[EntityContext]] = {}
            for entity in self.entities:
                by_kind.setdefault(entity.kind, []).append(entity)

            for kind, entities in sorted(by_kind.items()):
                lines.append(f"**{kind.upper()}** ({len(entities)}):")
                for entity in entities[:10]:
                    citation_count = len(entity.evidence)
                    citation_note = f" [{citation_count} citations]" if citation_count else ""
                    lines.append(f"- {entity.name}{citation_note}")
                if len(entities) > 10:
                    lines.append(f"  ... and {len(entities) - 10} more")
                lines.append("")

        if self.edges:
            lines.append("## Relationships\n")
            edge_counts: dict[str, int] = {}
            for edge in self.edges:
                edge_counts[edge.kind] = edge_counts.get(edge.kind, 0) + 1
            for kind, count in sorted(edge_counts.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"- {kind}: {count}")
            lines.append("")

        if self.paths:
            lines.append("## Key Paths\n")
            for path in self.paths[:3]:
                lines.append(f"- {path.description}")
            lines.append("")

        if self.citations:
            lines.append("## Source Citations\n")
            for cit in self.citations[:10]:
                lines.append(f"- `{cit.format()}`")
            if len(self.citations) > 10:
                lines.append(f"  ... and {len(self.citations) - 10} more")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "query": self.query,
            "communities": [
                {
                    "community_id": str(c.community_id),
                    "level": c.level,
                    "title": c.title,
                    "summary": c.summary,
                    "relevance_score": c.relevance_score,
                    "member_count": c.member_count,
                }
                for c in self.communities
            ],
            "entities": [
                {
                    "node_id": str(e.node_id),
                    "kind": e.kind,
                    "natural_key": e.natural_key,
                    "name": e.name,
                    "relevance_score": e.relevance_score,
                    "evidence": [
                        {
                            "file_path": ev.file_path,
                            "start_line": ev.start_line,
                            "end_line": ev.end_line,
                            "snippet": ev.snippet,
                        }
                        for ev in e.evidence
                    ],
                }
                for e in self.entities
            ],
            "edges": [
                {
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "kind": e.kind,
                }
                for e in self.edges
            ],
            "paths": [
                {
                    "nodes": p.nodes,
                    "edges": p.edges,
                    "description": p.description,
                }
                for p in self.paths
            ],
            "citations": [
                {
                    "file_path": c.file_path,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "snippet": c.snippet,
                }
                for c in self.citations
            ],
        }


# -----------------------------------------------------------------------------
# Main retrieval functions
# -----------------------------------------------------------------------------


async def graph_rag_context(
    session: AsyncSession,
    query: str,
    collection_id: UUID | None = None,
    user_id: UUID | None = None,
    max_communities: int = 5,
    max_entities: int = 20,
    max_depth: int = 2,
) -> ContextPack:
    """GraphRAG retrieval combining global (community) and local (entity) context.

    Args:
        session: Database session
        query: Natural language query
        collection_id: Optional collection filter
        user_id: Optional user for access control
        max_communities: Maximum communities to include (global)
        max_entities: Maximum entities to include (local)
        max_depth: Maximum neighborhood expansion depth

    Returns:
        ContextPack with global + local context
    """
    from contextmine_core import get_settings
    from contextmine_core.embeddings import get_embedder, parse_embedding_model_spec
    from contextmine_core.search import (
        SearchResponse,
        get_accessible_collection_ids,
        hybrid_search,
    )

    pack = ContextPack(query=query)

    # Get accessible collections
    if collection_id:
        collection_ids = [collection_id]
    else:
        collection_ids = await get_accessible_collection_ids(session, user_id)

    if not collection_ids:
        return pack

    # Get query embedding - REQUIRED for GraphRAG (no fake embedder fallback)
    settings = get_settings()
    emb_provider, emb_model = parse_embedding_model_spec(settings.default_embedding_model)
    embedder = get_embedder(emb_provider, emb_model)

    embed_result = await embedder.embed_batch([query])
    query_embedding = embed_result.embeddings[0]

    # Step 1: Global context - Community similarity
    communities = await _find_relevant_communities(
        session, query_embedding, collection_ids, max_communities
    )
    pack.communities = communities

    # Step 2: Hybrid search for local context seeds
    search_response: SearchResponse = await hybrid_search(
        query=query,
        query_embedding=query_embedding,
        user_id=user_id,
        collection_id=collection_ids[0] if collection_ids else None,
        top_k=max_entities,
    )

    # Step 3: Map search results to graph nodes
    seed_node_ids = await _map_search_to_nodes(session, search_response, collection_ids)

    # Step 4: Add community member nodes as additional seeds
    community_member_ids = await _get_community_member_nodes(
        session, [c.community_id for c in communities], limit=10
    )
    all_seed_ids = list(set(seed_node_ids + community_member_ids))

    # Step 5: Expand neighborhood from seeds
    entities, edges = await _expand_from_seeds(
        session, all_seed_ids, collection_ids, max_depth, max_entities
    )
    pack.entities = entities
    pack.edges = edges

    # Step 6: Gather citations
    pack.citations = await _gather_citations(session, [e.node_id for e in entities])

    # Attach citations to entities
    citation_by_node = await _citations_by_node(session, [e.node_id for e in entities])
    for entity in pack.entities:
        entity.evidence = citation_by_node.get(entity.node_id, [])

    return pack


async def graph_neighborhood(
    session: AsyncSession,
    node_id: UUID,
    collection_id: UUID | None = None,
    depth: int = 1,
    edge_kinds: list[str] | None = None,
    max_nodes: int = 50,
) -> ContextPack:
    """Get the neighborhood of a specific node.

    Args:
        session: Database session
        node_id: Starting node ID
        collection_id: Optional collection filter
        depth: Expansion depth (1-3)
        edge_kinds: Optional filter for edge kinds
        max_nodes: Maximum nodes to return

    Returns:
        ContextPack with neighborhood entities and edges
    """
    from contextmine_core.models import KnowledgeNode
    from sqlalchemy import select

    pack = ContextPack(query=f"Neighborhood of {node_id}")

    # Get collection IDs
    if collection_id:
        collection_ids = [collection_id]
    else:
        stmt = select(KnowledgeNode.collection_id).where(KnowledgeNode.id == node_id)
        result = await session.execute(stmt)
        coll = result.scalar_one_or_none()
        if coll:
            collection_ids = [coll]
        else:
            return pack

    # Expand from this node
    entities, edges = await _expand_from_seeds(session, [node_id], collection_ids, depth, max_nodes)

    # Filter edges by kind if specified
    if edge_kinds:
        edges = [e for e in edges if e.kind in edge_kinds]

    pack.entities = entities
    pack.edges = edges
    pack.citations = await _gather_citations(session, [e.node_id for e in entities])

    return pack


async def trace_path(
    session: AsyncSession,
    from_node_id: UUID,
    to_node_id: UUID,
    collection_id: UUID | None = None,
    max_hops: int = 6,
) -> ContextPack:
    """Find shortest path between two nodes using BFS.

    Args:
        session: Database session
        from_node_id: Starting node ID
        to_node_id: Target node ID
        collection_id: Optional collection filter
        max_hops: Maximum path length

    Returns:
        ContextPack with path entities and edges
    """
    from contextmine_core.models import KnowledgeEdge, KnowledgeNode
    from sqlalchemy import or_, select

    pack = ContextPack(query=f"Path {from_node_id} → {to_node_id}")

    # Get collection IDs
    if collection_id:
        collection_ids = [collection_id]
    else:
        stmt = select(KnowledgeNode.collection_id).where(
            KnowledgeNode.id.in_([from_node_id, to_node_id])
        )
        result = await session.execute(stmt)
        collection_ids = list({row[0] for row in result.fetchall()})

    if not collection_ids:
        return pack

    # BFS to find shortest path
    queue: deque[tuple[UUID, list[UUID], list[tuple[UUID, UUID, str]]]] = deque()
    queue.append((from_node_id, [from_node_id], []))
    visited: set[UUID] = {from_node_id}

    path_nodes: list[UUID] = []
    path_edges: list[tuple[UUID, UUID, str]] = []

    while queue and not path_nodes:
        current, node_path, edge_path = queue.popleft()

        if len(node_path) > max_hops + 1:
            break

        if current == to_node_id:
            path_nodes = node_path
            path_edges = edge_path
            break

        stmt = select(KnowledgeEdge).where(
            KnowledgeEdge.collection_id.in_(collection_ids),
            or_(
                KnowledgeEdge.source_node_id == current,
                KnowledgeEdge.target_node_id == current,
            ),
        )

        edge_result = await session.execute(stmt)
        for edge in edge_result.scalars().all():
            next_node = (
                edge.target_node_id if edge.source_node_id == current else edge.source_node_id
            )

            if next_node not in visited:
                visited.add(next_node)
                new_node_path = node_path + [next_node]
                new_edge_path = edge_path + [
                    (edge.source_node_id, edge.target_node_id, edge.kind.value)
                ]
                queue.append((next_node, new_node_path, new_edge_path))

    if not path_nodes:
        return pack

    # Fetch full node data
    stmt = select(KnowledgeNode).where(KnowledgeNode.id.in_(path_nodes))
    node_result = await session.execute(stmt)
    node_map: dict[UUID, KnowledgeNode] = {n.id: n for n in node_result.scalars().all()}

    for nid in path_nodes:
        if nid in node_map:
            node = node_map[nid]
            pack.entities.append(
                EntityContext(
                    node_id=node.id,
                    kind=node.kind.value,
                    natural_key=node.natural_key,
                    name=node.name,
                )
            )

    for src, tgt, kind in path_edges:
        pack.edges.append(EdgeContext(source_id=str(src), target_id=str(tgt), kind=kind))

    # Build path description
    if pack.entities:
        names = [e.name for e in pack.entities]
        pack.paths.append(
            PathContext(
                nodes=[e.natural_key for e in pack.entities],
                edges=[e.kind for e in pack.edges],
                description=" → ".join(names),
            )
        )

    pack.citations = await _gather_citations(session, path_nodes)

    return pack


# -----------------------------------------------------------------------------
# Internal helper functions
# -----------------------------------------------------------------------------


async def _find_relevant_communities(
    session: AsyncSession,
    query_embedding: list[float],
    collection_ids: list[UUID],
    max_communities: int,
) -> list[CommunityContext]:
    """Find communities via vector similarity on summary embeddings."""
    from contextmine_core.models import (
        EmbeddingTargetType,
        KnowledgeCommunity,
        KnowledgeEmbedding,
    )
    from sqlalchemy import bindparam, select, text

    communities: list[CommunityContext] = []
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    # Vector similarity search on community embeddings - REQUIRED for GraphRAG
    # No fallback to random selection (that defeats semantic retrieval)
    # Use bindparam for safe SQL construction (nosemgrep: avoid-sqlalchemy-text)
    similarity_expr = text(
        "1 - (knowledge_embeddings.embedding <=> :embedding::vector) as similarity"
    ).bindparams(bindparam("embedding", value=embedding_str))
    stmt = (
        select(
            KnowledgeCommunity.id,
            KnowledgeCommunity.level,
            KnowledgeCommunity.title,
            KnowledgeCommunity.summary,
            KnowledgeCommunity.meta,
            similarity_expr,
        )
        .join(
            KnowledgeEmbedding,
            (KnowledgeEmbedding.target_id == KnowledgeCommunity.id)
            & (KnowledgeEmbedding.target_type == EmbeddingTargetType.COMMUNITY),
        )
        .where(KnowledgeCommunity.collection_id.in_(collection_ids))
        .order_by(text("similarity DESC"))
        .limit(max_communities)
    )

    result = await session.execute(stmt)
    for row in result.all():
        meta = row[4] or {}
        communities.append(
            CommunityContext(
                community_id=row[0],
                level=row[1],
                title=row[2] or "Untitled",
                summary=row[3] or "",
                relevance_score=float(row[5]) if row[5] else 0.0,
                member_count=meta.get("member_count", 0),
            )
        )

    # If no communities found, the index is incomplete - don't fake it
    if not communities:
        logger.warning(
            "No community embeddings found for collections %s. "
            "Run LLM entity extraction and community summarization first.",
            collection_ids,
        )

    return communities


async def _map_search_to_nodes(
    session: AsyncSession,
    search_response: Any,
    collection_ids: list[UUID],
) -> list[UUID]:
    """Map search results to knowledge graph FILE nodes.

    GraphRAG requires proper mapping from search to graph nodes.
    No fallback to arbitrary nodes (that defeats the purpose of search).
    """
    from contextmine_core.models import KnowledgeNode, KnowledgeNodeKind
    from sqlalchemy import select

    node_ids: list[UUID] = []

    if not hasattr(search_response, "results") or not search_response.results:
        return node_ids

    uris = list({r.uri for r in search_response.results if r.uri})
    if not uris:
        return node_ids

    natural_keys = [f"file:{uri}" for uri in uris]

    stmt = select(KnowledgeNode.id).where(
        KnowledgeNode.collection_id.in_(collection_ids),
        KnowledgeNode.kind == KnowledgeNodeKind.FILE,
        KnowledgeNode.natural_key.in_(natural_keys),
    )

    result = await session.execute(stmt)
    node_ids.extend([row[0] for row in result.fetchall()])

    # No fallback to arbitrary nodes - if search doesn't map to graph, return empty
    # The caller should handle empty results appropriately

    return node_ids[:20]


async def _get_community_member_nodes(
    session: AsyncSession,
    community_ids: list[UUID],
    limit: int = 10,
) -> list[UUID]:
    """Get top member nodes from communities."""
    from contextmine_core.models import CommunityMember
    from sqlalchemy import select

    if not community_ids:
        return []

    stmt = (
        select(CommunityMember.node_id)
        .where(CommunityMember.community_id.in_(community_ids))
        .order_by(CommunityMember.score.desc())
        .limit(limit)
    )

    result = await session.execute(stmt)
    return [row[0] for row in result.fetchall()]


async def _expand_from_seeds(
    session: AsyncSession,
    seed_node_ids: list[UUID],
    collection_ids: list[UUID],
    max_depth: int,
    max_entities: int,
) -> tuple[list[EntityContext], list[EdgeContext]]:
    """Expand neighborhood from seed nodes via BFS."""
    from contextmine_core.models import KnowledgeEdge, KnowledgeNode
    from sqlalchemy import or_, select

    entities: list[EntityContext] = []
    edges: list[EdgeContext] = []
    visited: set[UUID] = set()
    current_ids = set(seed_node_ids)

    for depth in range(max_depth + 1):
        if not current_ids or len(visited) >= max_entities:
            break

        ids_to_fetch = current_ids - visited
        if not ids_to_fetch:
            break

        stmt = select(KnowledgeNode).where(
            KnowledgeNode.id.in_(list(ids_to_fetch)),
            KnowledgeNode.collection_id.in_(collection_ids),
        )

        result = await session.execute(stmt)
        node_map: dict[UUID, KnowledgeNode] = {}

        for node in result.scalars():
            if node.id not in visited and len(entities) < max_entities:
                entities.append(
                    EntityContext(
                        node_id=node.id,
                        kind=node.kind.value,
                        natural_key=node.natural_key,
                        name=node.name,
                        relevance_score=1.0 - (depth * 0.2),
                    )
                )
                visited.add(node.id)
                node_map[node.id] = node

        if depth < max_depth:
            stmt = select(KnowledgeEdge).where(
                KnowledgeEdge.collection_id.in_(collection_ids),
                or_(
                    KnowledgeEdge.source_node_id.in_(list(visited)),
                    KnowledgeEdge.target_node_id.in_(list(visited)),
                ),
            )

            result = await session.execute(stmt)
            next_ids: set[UUID] = set()

            for edge in result.scalars():
                src_node = node_map.get(edge.source_node_id)
                tgt_node = node_map.get(edge.target_node_id)
                edges.append(
                    EdgeContext(
                        source_id=str(edge.source_node_id),
                        target_id=str(edge.target_node_id),
                        kind=edge.kind.value,
                        source_name=src_node.name if src_node else None,
                        target_name=tgt_node.name if tgt_node else None,
                    )
                )

                if edge.source_node_id not in visited:
                    next_ids.add(edge.source_node_id)
                if edge.target_node_id not in visited:
                    next_ids.add(edge.target_node_id)

            current_ids = next_ids

    return entities, edges


async def _gather_citations(
    session: AsyncSession,
    node_ids: list[UUID],
) -> list[Citation]:
    """Gather citations for nodes from KnowledgeEvidence."""
    from contextmine_core.models import KnowledgeEvidence, KnowledgeNodeEvidence
    from sqlalchemy import select

    citations: list[Citation] = []
    seen: set[tuple[str, int, int]] = set()

    if not node_ids:
        return citations

    stmt = (
        select(KnowledgeEvidence)
        .join(
            KnowledgeNodeEvidence,
            KnowledgeNodeEvidence.evidence_id == KnowledgeEvidence.id,
        )
        .where(KnowledgeNodeEvidence.node_id.in_(node_ids))
    )

    result = await session.execute(stmt)
    for evidence in result.scalars():
        key = (evidence.file_path, evidence.start_line, evidence.end_line)
        if key not in seen:
            seen.add(key)
            citations.append(
                Citation(
                    file_path=evidence.file_path,
                    start_line=evidence.start_line,
                    end_line=evidence.end_line,
                    snippet=evidence.snippet,
                )
            )

    return citations[:50]


async def _citations_by_node(
    session: AsyncSession,
    node_ids: list[UUID],
) -> dict[UUID, list[Citation]]:
    """Get citations grouped by node."""
    from contextmine_core.models import KnowledgeEvidence, KnowledgeNodeEvidence
    from sqlalchemy import select

    result_map: dict[UUID, list[Citation]] = {}

    if not node_ids:
        return result_map

    stmt = (
        select(KnowledgeNodeEvidence.node_id, KnowledgeEvidence)
        .join(
            KnowledgeEvidence,
            KnowledgeNodeEvidence.evidence_id == KnowledgeEvidence.id,
        )
        .where(KnowledgeNodeEvidence.node_id.in_(node_ids))
    )

    result = await session.execute(stmt)
    for node_id, evidence in result.all():
        if node_id not in result_map:
            result_map[node_id] = []
        result_map[node_id].append(
            Citation(
                file_path=evidence.file_path,
                start_line=evidence.start_line,
                end_line=evidence.end_line,
                snippet=evidence.snippet,
            )
        )

    return result_map


# -----------------------------------------------------------------------------
# Map-Reduce Query Answering (GraphRAG Core)
# -----------------------------------------------------------------------------


@dataclass
class MapReduceResult:
    """Result of map-reduce query answering."""

    query: str
    final_answer: str
    partial_answers: list[str] = field(default_factory=list)
    communities_used: int = 0
    context: ContextPack | None = None


MAP_PROMPT_TEMPLATE = """You are analyzing a software codebase to answer a question.

QUESTION: {query}

COMMUNITY SUMMARY:
Title: {community_title}
Level: {community_level}
Members: {member_count}

{community_summary}

Based ONLY on the information in this community summary, provide a partial answer to the question.
If this community is not relevant to the question, respond with "NOT_RELEVANT".
Be concise and factual. Do not make up information not present in the summary.

PARTIAL ANSWER:"""

REDUCE_PROMPT_TEMPLATE = """You are synthesizing information about a software codebase.

QUESTION: {query}

The following partial answers were generated from different parts of the codebase:

{partial_answers}

Synthesize these partial answers into a comprehensive final answer.
- Combine overlapping information
- Resolve any contradictions by preferring more specific answers
- If all answers said NOT_RELEVANT, say "No relevant information found in the indexed codebase."
- Be concise and well-organized

FINAL ANSWER:"""


async def graph_rag_query(
    session: AsyncSession,
    query: str,
    llm_provider: LLMProvider,
    collection_id: UUID | None = None,
    user_id: UUID | None = None,
    max_communities: int = 10,
    max_entities: int = 20,
    include_local_context: bool = True,
) -> MapReduceResult:
    """Answer a query using GraphRAG map-reduce strategy.

    This is the main GraphRAG entry point that implements the full
    map-reduce query answering from the Microsoft paper:

    1. Find relevant communities via embedding similarity
    2. MAP: Generate partial answer from each community summary
    3. REDUCE: Combine partial answers into final response
    4. Optionally include local entity context

    Args:
        session: Database session
        query: Natural language query
        llm_provider: LLM provider for map/reduce generation
        collection_id: Optional collection filter
        user_id: Optional user for access control
        max_communities: Maximum communities to query
        max_entities: Maximum entities for local context
        include_local_context: Whether to include entity context

    Returns:
        MapReduceResult with final answer and metadata
    """
    result = MapReduceResult(query=query, final_answer="")

    # Get context (communities + entities)
    context = await graph_rag_context(
        session=session,
        query=query,
        collection_id=collection_id,
        user_id=user_id,
        max_communities=max_communities,
        max_entities=max_entities if include_local_context else 0,
    )
    result.context = context
    result.communities_used = len(context.communities)

    if not context.communities:
        result.final_answer = (
            "No community summaries available. The codebase may not have been fully indexed."
        )
        return result

    # MAP phase: Generate partial answers from each community
    map_tasks = []
    for comm in context.communities:
        map_tasks.append(
            _map_community(
                llm_provider=llm_provider,
                query=query,
                community=comm,
            )
        )

    partial_answers = await asyncio.gather(*map_tasks, return_exceptions=True)

    # Filter valid partial answers
    valid_partials: list[str] = []
    for i, answer in enumerate(partial_answers):
        if isinstance(answer, BaseException):
            logger.warning("Map failed for community %s: %s", context.communities[i].title, answer)
            continue
        if answer and answer.strip() and answer.strip().upper() != "NOT_RELEVANT":
            valid_partials.append(answer)

    result.partial_answers = valid_partials

    # REDUCE phase: Combine partial answers
    if not valid_partials:
        result.final_answer = "No relevant information found in the indexed codebase communities."
        return result

    if len(valid_partials) == 1:
        result.final_answer = valid_partials[0]
        return result

    # Multiple partial answers - reduce them
    result.final_answer = await _reduce_answers(
        llm_provider=llm_provider,
        query=query,
        partial_answers=valid_partials,
    )

    return result


async def _map_community(
    llm_provider: LLMProvider,
    query: str,
    community: CommunityContext,
) -> str:
    """Generate partial answer from a single community (MAP phase)."""
    prompt = MAP_PROMPT_TEMPLATE.format(
        query=query,
        community_title=community.title,
        community_level=community.level,
        member_count=community.member_count,
        community_summary=community.summary or "No summary available.",
    )

    response = await llm_provider.generate_text(
        system="You are a precise code analyst. Answer based only on provided information.",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500,
    )

    return response.strip()


async def _reduce_answers(
    llm_provider: LLMProvider,
    query: str,
    partial_answers: list[str],
) -> str:
    """Combine partial answers into final response (REDUCE phase)."""
    # Format partial answers
    formatted_partials = "\n\n".join(
        f"--- Partial Answer {i + 1} ---\n{answer}" for i, answer in enumerate(partial_answers)
    )

    prompt = REDUCE_PROMPT_TEMPLATE.format(
        query=query,
        partial_answers=formatted_partials,
    )

    response = await llm_provider.generate_text(
        system="You are a precise code analyst synthesizing information.",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1000,
    )

    return response.strip()
