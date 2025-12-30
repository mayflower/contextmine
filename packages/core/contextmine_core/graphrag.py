"""GraphRAG retrieval service.

Implements graph-augmented retrieval by:
1. Using hybrid search to find relevant documents/chunks
2. Mapping search hits to Knowledge Graph nodes
3. Expanding graph neighborhood (depth 1-2)
4. Building a compact evidence bundle with citations
5. Returning as Markdown + structured JSON
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from contextmine_core.search import SearchResponse
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """A node in the result graph."""

    id: str
    kind: str
    name: str
    natural_key: str
    meta: dict = field(default_factory=dict)


@dataclass
class GraphEdge:
    """An edge in the result graph."""

    source_id: str
    target_id: str
    kind: str
    meta: dict = field(default_factory=dict)


@dataclass
class Evidence:
    """Evidence citation."""

    file_path: str
    start_line: int
    end_line: int
    snippet: str | None = None


@dataclass
class GraphRAGResult:
    """Result from GraphRAG retrieval."""

    query: str
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    summary_markdown: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "query": self.query,
            "nodes": [
                {
                    "id": n.id,
                    "kind": n.kind,
                    "name": n.name,
                    "natural_key": n.natural_key,
                    "meta": n.meta,
                }
                for n in self.nodes
            ],
            "edges": [
                {
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "kind": e.kind,
                    "meta": e.meta,
                }
                for e in self.edges
            ],
            "evidence": [
                {
                    "file_path": e.file_path,
                    "start_line": e.start_line,
                    "end_line": e.end_line,
                    "snippet": e.snippet,
                }
                for e in self.evidence
            ],
            "summary_markdown": self.summary_markdown,
        }


async def graph_rag_bundle(
    session: AsyncSession,
    query: str,
    collection_id: UUID | None = None,
    user_id: UUID | None = None,
    max_depth: int = 2,
    max_nodes: int = 20,
    max_results: int = 10,
) -> GraphRAGResult:
    """Perform GraphRAG retrieval with full knowledge context.

    This is the main research tool that combines:
    1. Semantic search to find relevant documents/code
    2. Knowledge graph expansion to find related entities
    3. Business rules that apply to the matched content
    4. Database tables and API endpoints related to the query
    5. Architecture context from arc42 documentation

    Args:
        session: Database session
        query: Natural language query
        collection_id: Optional collection filter
        user_id: Optional user for access control
        max_depth: Maximum graph expansion depth (1-2 recommended)
        max_nodes: Maximum nodes to include in result
        max_results: Maximum search results to use as seeds

    Returns:
        GraphRAGResult with nodes, edges, evidence, and markdown summary
    """
    from contextmine_core import get_settings
    from contextmine_core.embeddings import FakeEmbedder, get_embedder, parse_embedding_model_spec
    from contextmine_core.search import (
        get_accessible_collection_ids,
        hybrid_search,
    )

    result = GraphRAGResult(query=query)

    # Get accessible collections
    if collection_id:
        collection_ids = [collection_id]
    else:
        collection_ids = await get_accessible_collection_ids(session, user_id)

    if not collection_ids:
        result.summary_markdown = "No accessible collections found."
        return result

    # Get query embedding
    settings = get_settings()
    try:
        emb_provider, emb_model = parse_embedding_model_spec(settings.default_embedding_model)
        embedder = get_embedder(emb_provider, emb_model)
    except (ValueError, Exception):
        embedder = FakeEmbedder()

    embed_result = await embedder.embed_batch([query])
    query_embedding = embed_result.embeddings[0]

    # Step 1: Hybrid search to find relevant content
    # Use first collection for search (hybrid_search takes single collection_id)
    search_response: SearchResponse = await hybrid_search(
        query=query,
        query_embedding=query_embedding,
        user_id=user_id,
        collection_id=collection_ids[0] if collection_ids else None,
        top_k=max_results,
    )

    if not search_response.results:
        result.summary_markdown = f"No results found for query: {query}"
        return result

    # Step 2: Map search results to Knowledge Graph nodes
    seed_node_ids = await _map_results_to_nodes(session, search_response, collection_ids)

    # Step 3: Find related business rules, tables, and endpoints
    related_knowledge = await _find_related_knowledge(session, query, seed_node_ids, collection_ids)

    # Add related knowledge nodes as seeds
    all_seed_ids = list(set(seed_node_ids + related_knowledge.get("node_ids", [])))

    if not all_seed_ids:
        # Fall back to search results only
        result.summary_markdown = _build_search_only_markdown(query, search_response)
        return result

    # Step 4: Expand graph neighborhood
    nodes, edges = await _expand_neighborhood(
        session, all_seed_ids, collection_ids, max_depth, max_nodes
    )

    result.nodes = nodes
    result.edges = edges

    # Step 5: Gather evidence
    result.evidence = await _gather_evidence(session, [n.id for n in nodes])

    # Step 6: Build rich markdown summary with knowledge context
    result.summary_markdown = _build_research_markdown(
        query, search_response, nodes, edges, result.evidence, related_knowledge
    )

    return result


async def _find_related_knowledge(
    session: AsyncSession,
    query: str,
    seed_node_ids: list[UUID],
    collection_ids: list[UUID],
) -> dict:
    """Find business rules, tables, endpoints related to the query.

    Returns a dict with:
    - node_ids: Additional node IDs to include
    - business_rules: Relevant business rules
    - tables: Relevant database tables
    - endpoints: Relevant API endpoints
    - architecture_context: Relevant arc42 sections
    """
    from contextmine_core.models import (
        KnowledgeArtifact,
        KnowledgeArtifactKind,
        KnowledgeNode,
        KnowledgeNodeKind,
    )
    from sqlalchemy import select

    related: dict = {
        "node_ids": [],
        "business_rules": [],
        "tables": [],
        "endpoints": [],
        "jobs": [],
        "architecture_context": None,
    }

    query_lower = query.lower()
    query_words = set(query_lower.split())

    # Find business rules that match the query
    stmt = select(KnowledgeNode).where(
        KnowledgeNode.collection_id.in_(collection_ids),
        KnowledgeNode.kind == KnowledgeNodeKind.BUSINESS_RULE,
    )
    result = await session.execute(stmt)
    for rule in result.scalars().all():
        meta = rule.meta or {}
        # Check if rule matches query by name, description, or natural language
        searchable = " ".join(
            [
                rule.name.lower(),
                meta.get("description", "").lower(),
                meta.get("natural_language", "").lower(),
                meta.get("category", "").lower(),
            ]
        )
        # Match if any query word appears in rule content
        if any(word in searchable for word in query_words if len(word) > 2):
            related["node_ids"].append(rule.id)
            related["business_rules"].append(
                {
                    "id": str(rule.id),
                    "name": rule.name,
                    "category": meta.get("category", "unknown"),
                    "severity": meta.get("severity", "unknown"),
                    "rule": meta.get("natural_language", meta.get("description", "")),
                }
            )

    # Find database tables that match the query
    stmt = select(KnowledgeNode).where(
        KnowledgeNode.collection_id.in_(collection_ids),
        KnowledgeNode.kind == KnowledgeNodeKind.DB_TABLE,
    )
    result = await session.execute(stmt)
    for table in result.scalars().all():
        if any(word in table.name.lower() for word in query_words if len(word) > 2):
            related["node_ids"].append(table.id)
            meta = table.meta or {}
            related["tables"].append(
                {
                    "id": str(table.id),
                    "name": table.name,
                    "columns": meta.get("column_count", 0),
                    "primary_key": meta.get("primary_key"),
                }
            )

    # Find API endpoints that match the query
    stmt = select(KnowledgeNode).where(
        KnowledgeNode.collection_id.in_(collection_ids),
        KnowledgeNode.kind == KnowledgeNodeKind.API_ENDPOINT,
    )
    result = await session.execute(stmt)
    for endpoint in result.scalars().all():
        meta = endpoint.meta or {}
        searchable = " ".join(
            [
                endpoint.name.lower(),
                meta.get("path", "").lower(),
                meta.get("description", "").lower(),
                " ".join(meta.get("tags", [])).lower(),
            ]
        )
        if any(word in searchable for word in query_words if len(word) > 2):
            related["node_ids"].append(endpoint.id)
            related["endpoints"].append(
                {
                    "id": str(endpoint.id),
                    "method": meta.get("method", "GET"),
                    "path": meta.get("path", endpoint.name),
                    "description": meta.get("description", "")[:100],
                }
            )

    # Find jobs that match the query
    stmt = select(KnowledgeNode).where(
        KnowledgeNode.collection_id.in_(collection_ids),
        KnowledgeNode.kind == KnowledgeNodeKind.JOB,
    )
    result = await session.execute(stmt)
    for job in result.scalars().all():
        if any(word in job.name.lower() for word in query_words if len(word) > 2):
            related["node_ids"].append(job.id)
            meta = job.meta or {}
            related["jobs"].append(
                {
                    "id": str(job.id),
                    "name": job.name,
                    "type": meta.get("job_type", "unknown"),
                    "schedule": meta.get("schedule"),
                }
            )

    # Get architecture context from arc42 artifact
    stmt = (
        select(KnowledgeArtifact)
        .where(
            KnowledgeArtifact.collection_id.in_(collection_ids),
            KnowledgeArtifact.kind == KnowledgeArtifactKind.ARC42,
        )
        .order_by(KnowledgeArtifact.created_at.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    artifact = result.scalar_one_or_none()
    if artifact and artifact.content:
        # Extract relevant section based on query
        content_lower = artifact.content.lower()
        if any(word in content_lower for word in query_words if len(word) > 3):
            # Include a summary of what's in the architecture doc
            related["architecture_context"] = _extract_relevant_arc42(artifact.content, query_words)

    # Limit to avoid overwhelming results
    related["node_ids"] = related["node_ids"][:15]
    related["business_rules"] = related["business_rules"][:5]
    related["tables"] = related["tables"][:5]
    related["endpoints"] = related["endpoints"][:5]
    related["jobs"] = related["jobs"][:3]

    return related


def _extract_relevant_arc42(content: str, query_words: set[str]) -> str:
    """Extract relevant sections from arc42 content."""
    lines = content.split("\n")
    relevant_sections: list[str] = []
    current_section = ""
    current_content: list[str] = []

    for line in lines:
        if line.startswith("## "):
            # Save previous section if relevant
            if current_section and current_content:
                section_text = "\n".join(current_content)
                if any(word in section_text.lower() for word in query_words if len(word) > 3):
                    relevant_sections.append(f"**{current_section}**: {section_text[:200]}...")
            current_section = line[3:].strip()
            current_content = []
        else:
            current_content.append(line)

    # Check last section
    if current_section and current_content:
        section_text = "\n".join(current_content)
        if any(word in section_text.lower() for word in query_words if len(word) > 3):
            relevant_sections.append(f"**{current_section}**: {section_text[:200]}...")

    return "\n".join(relevant_sections[:3]) if relevant_sections else ""


def _build_research_markdown(
    query: str,
    search_response: SearchResponse,
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    evidence: list[Evidence],
    related_knowledge: dict,
) -> str:
    """Build a rich markdown research result."""
    lines = [f"# Research Results: {query}\n"]

    # Summary
    rule_count = len(related_knowledge.get("business_rules", []))
    table_count = len(related_knowledge.get("tables", []))
    endpoint_count = len(related_knowledge.get("endpoints", []))

    lines.append(
        f"Found {len(nodes)} code entities, {rule_count} business rules, "
        f"{table_count} database tables, {endpoint_count} API endpoints.\n"
    )

    # Business Rules section
    business_rules = related_knowledge.get("business_rules", [])
    if business_rules:
        lines.append("## Relevant Business Rules\n")
        for rule in business_rules:
            lines.append(f"### {rule['name']}")
            lines.append(f"- **Category**: {rule['category']} | **Severity**: {rule['severity']}")
            if rule.get("rule"):
                lines.append(f"- **Rule**: {rule['rule']}")
            lines.append("")

    # Database Tables section
    tables = related_knowledge.get("tables", [])
    if tables:
        lines.append("## Related Database Tables\n")
        for table in tables:
            pk = f" (PK: {table['primary_key']})" if table.get("primary_key") else ""
            lines.append(f"- **{table['name']}**: {table['columns']} columns{pk}")
        lines.append("")

    # API Endpoints section
    endpoints = related_knowledge.get("endpoints", [])
    if endpoints:
        lines.append("## Related API Endpoints\n")
        for ep in endpoints:
            desc = f" - {ep['description']}" if ep.get("description") else ""
            lines.append(f"- `{ep['method']} {ep['path']}`{desc}")
        lines.append("")

    # Jobs section
    jobs = related_knowledge.get("jobs", [])
    if jobs:
        lines.append("## Related Jobs/Workflows\n")
        for job in jobs:
            schedule = f" (schedule: {job['schedule']})" if job.get("schedule") else ""
            lines.append(f"- **{job['name']}** ({job['type']}){schedule}")
        lines.append("")

    # Architecture Context
    arch_context = related_knowledge.get("architecture_context")
    if arch_context:
        lines.append("## Architecture Context\n")
        lines.append(arch_context)
        lines.append("")

    # Code entities from graph
    nodes_by_kind: dict[str, list[GraphNode]] = {}
    for node in nodes:
        if node.kind not in ("business_rule", "db_table", "api_endpoint", "job"):
            nodes_by_kind.setdefault(node.kind, []).append(node)

    if nodes_by_kind:
        lines.append("## Related Code\n")
        for kind, kind_nodes in sorted(nodes_by_kind.items()):
            lines.append(f"**{kind.upper()}**:")
            for node in kind_nodes[:5]:
                lines.append(f"- {node.name}")
            if len(kind_nodes) > 5:
                lines.append(f"  ... and {len(kind_nodes) - 5} more")
            lines.append("")

    # Evidence citations
    if evidence:
        lines.append("## Source Citations\n")
        for ev in evidence[:8]:
            lines.append(f"- `{ev.file_path}:{ev.start_line}-{ev.end_line}`")
        if len(evidence) > 8:
            lines.append(f"  ... and {len(evidence) - 8} more")
        lines.append("")

    # Search context (what docs were found)
    if search_response.results:
        lines.append("## Documents Found\n")
        for r in search_response.results[:5]:
            lines.append(f"- [{r.title}]({r.uri}) (relevance: {r.score:.0%})")
        lines.append("")

    return "\n".join(lines)


async def _map_results_to_nodes(
    session: AsyncSession,
    search_response: SearchResponse,
    collection_ids: list[UUID],
) -> list[UUID]:
    """Map search results to Knowledge Graph node IDs."""
    from contextmine_core.models import KnowledgeNode, KnowledgeNodeKind
    from sqlalchemy import select

    node_ids: list[UUID] = []

    # Get unique document IDs from search results
    document_ids = list({r.document_id for r in search_response.results})

    # Find FILE nodes for these documents
    stmt = select(KnowledgeNode.id).where(
        KnowledgeNode.collection_id.in_(collection_ids),
        KnowledgeNode.kind == KnowledgeNodeKind.FILE,
        KnowledgeNode.natural_key.in_([f"file:{doc_id}" for doc_id in document_ids]),
    )

    result = await session.execute(stmt)
    node_ids.extend([row[0] for row in result.fetchall()])

    # Also try to find SYMBOL nodes by searching for symbols in these files
    # This is a simpler approach - match by file path in natural key
    uris = list({r.uri for r in search_response.results})
    for uri in uris[:5]:  # Limit to avoid too many queries
        stmt = (
            select(KnowledgeNode.id)
            .where(
                KnowledgeNode.collection_id.in_(collection_ids),
                KnowledgeNode.kind == KnowledgeNodeKind.SYMBOL,
                KnowledgeNode.natural_key.like(f"symbol:{uri}:%"),
            )
            .limit(5)
        )

        result = await session.execute(stmt)
        node_ids.extend([row[0] for row in result.fetchall()])

    return node_ids[:20]  # Limit seed nodes


async def _expand_neighborhood(
    session: AsyncSession,
    seed_node_ids: list[UUID],
    collection_ids: list[UUID],
    max_depth: int,
    max_nodes: int,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Expand the graph neighborhood from seed nodes."""
    from contextmine_core.models import KnowledgeEdge, KnowledgeNode
    from sqlalchemy import or_, select

    visited_ids: set[UUID] = set()
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    current_ids = set(seed_node_ids)

    for depth in range(max_depth + 1):
        if not current_ids or len(visited_ids) >= max_nodes:
            break

        # Fetch nodes at current level
        ids_to_fetch = current_ids - visited_ids
        if not ids_to_fetch:
            break

        stmt = select(KnowledgeNode).where(
            KnowledgeNode.id.in_(list(ids_to_fetch)),
            KnowledgeNode.collection_id.in_(collection_ids),
        )

        result = await session.execute(stmt)
        fetched_nodes = result.scalars().all()

        for node in fetched_nodes:
            if node.id not in visited_ids and len(nodes) < max_nodes:
                nodes.append(
                    GraphNode(
                        id=str(node.id),
                        kind=node.kind.value,
                        name=node.name,
                        natural_key=node.natural_key,
                        meta=node.meta or {},
                    )
                )
                visited_ids.add(node.id)

        if depth < max_depth:
            # Find edges to/from current nodes
            stmt = select(KnowledgeEdge).where(
                KnowledgeEdge.collection_id.in_(collection_ids),
                or_(
                    KnowledgeEdge.source_node_id.in_(list(visited_ids)),
                    KnowledgeEdge.target_node_id.in_(list(visited_ids)),
                ),
            )

            result = await session.execute(stmt)
            fetched_edges = result.scalars().all()

            next_ids: set[UUID] = set()
            for edge in fetched_edges:
                # Avoid duplicate edges
                edge_key = (str(edge.source_node_id), str(edge.target_node_id), edge.kind.value)
                if edge_key not in {(e.source_id, e.target_id, e.kind) for e in edges}:
                    edges.append(
                        GraphEdge(
                            source_id=str(edge.source_node_id),
                            target_id=str(edge.target_node_id),
                            kind=edge.kind.value,
                            meta=edge.meta or {},
                        )
                    )

                # Add connected nodes for next iteration
                if edge.source_node_id not in visited_ids:
                    next_ids.add(edge.source_node_id)
                if edge.target_node_id not in visited_ids:
                    next_ids.add(edge.target_node_id)

            current_ids = next_ids

    return nodes, edges


async def _gather_evidence(
    session: AsyncSession,
    node_ids: list[str],
) -> list[Evidence]:
    """Gather evidence for nodes."""
    from contextmine_core.models import KnowledgeEvidence, KnowledgeNodeEvidence
    from sqlalchemy import select

    evidence_list: list[Evidence] = []
    seen_keys: set[tuple[str, int, int]] = set()

    # Convert string IDs to UUIDs
    uuid_ids = [UUID(nid) for nid in node_ids]

    stmt = (
        select(KnowledgeEvidence)
        .join(
            KnowledgeNodeEvidence,
            KnowledgeNodeEvidence.evidence_id == KnowledgeEvidence.id,
        )
        .where(KnowledgeNodeEvidence.node_id.in_(uuid_ids))
    )

    result = await session.execute(stmt)
    for evidence in result.scalars().all():
        key = (evidence.file_path, evidence.start_line, evidence.end_line)
        if key not in seen_keys:
            seen_keys.add(key)
            evidence_list.append(
                Evidence(
                    file_path=evidence.file_path,
                    start_line=evidence.start_line,
                    end_line=evidence.end_line,
                    snippet=None,  # Could load snippet from document
                )
            )

    return evidence_list[:20]  # Limit evidence


def _build_search_only_markdown(query: str, search_response: SearchResponse) -> str:
    """Build markdown when no graph nodes are found."""
    lines = [f"## Search Results for: {query}\n"]

    for i, result in enumerate(search_response.results[:5], 1):
        lines.append(f"### {i}. {result.title}")
        lines.append(f"- **URI**: {result.uri}")
        lines.append(f"- **Score**: {result.score:.3f}")
        if result.content:
            snippet = result.content[:200] + "..." if len(result.content) > 200 else result.content
            lines.append(f"- **Preview**: {snippet}")
        lines.append("")

    return "\n".join(lines)


def _build_markdown_summary(
    query: str,
    search_response: SearchResponse,
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    evidence: list[Evidence],
) -> str:
    """Build a markdown summary of the GraphRAG result."""
    lines = [f"## GraphRAG Results for: {query}\n"]

    # Summary stats
    lines.append(
        f"Found {len(nodes)} knowledge graph nodes, {len(edges)} edges, {len(evidence)} evidence citations.\n"
    )

    # Group nodes by kind
    nodes_by_kind: dict[str, list[GraphNode]] = {}
    for node in nodes:
        nodes_by_kind.setdefault(node.kind, []).append(node)

    # Show nodes by category
    if nodes_by_kind:
        lines.append("### Knowledge Graph Nodes\n")
        for kind, kind_nodes in sorted(nodes_by_kind.items()):
            lines.append(f"**{kind.upper()}** ({len(kind_nodes)}):")
            for node in kind_nodes[:5]:  # Limit per kind
                lines.append(f"- {node.name}")
            if len(kind_nodes) > 5:
                lines.append(f"  ... and {len(kind_nodes) - 5} more")
            lines.append("")

    # Show key relationships
    if edges:
        lines.append("### Key Relationships\n")
        edge_kinds: dict[str, int] = {}
        for edge in edges:
            edge_kinds[edge.kind] = edge_kinds.get(edge.kind, 0) + 1

        for kind, count in sorted(edge_kinds.items(), key=lambda x: -x[1])[:5]:
            lines.append(f"- {kind}: {count} edges")
        lines.append("")

    # Show evidence citations
    if evidence:
        lines.append("### Evidence Citations\n")
        for ev in evidence[:5]:
            lines.append(f"- `{ev.file_path}:{ev.start_line}-{ev.end_line}`")
        if len(evidence) > 5:
            lines.append(f"  ... and {len(evidence) - 5} more citations")
        lines.append("")

    # Show search context
    if search_response.results:
        lines.append("### Search Context\n")
        for result in search_response.results[:3]:
            lines.append(f"- [{result.title}]({result.uri}) (score: {result.score:.2f})")
        lines.append("")

    return "\n".join(lines)


async def graph_neighborhood(
    session: AsyncSession,
    node_id: UUID,
    collection_id: UUID | None = None,
    depth: int = 1,
    edge_kinds: list[str] | None = None,
    max_nodes: int = 50,
) -> GraphRAGResult:
    """Get the neighborhood of a specific node.

    Args:
        session: Database session
        node_id: Starting node ID
        collection_id: Optional collection filter
        depth: Expansion depth (1-2)
        edge_kinds: Optional filter for edge kinds
        max_nodes: Maximum nodes to return

    Returns:
        GraphRAGResult with neighborhood
    """
    from contextmine_core.models import KnowledgeNode
    from sqlalchemy import select

    result = GraphRAGResult(query=f"Neighborhood of node {node_id}")

    # Get collection IDs
    if collection_id:
        collection_ids = [collection_id]
    else:
        # Get collection from the node itself
        stmt = select(KnowledgeNode.collection_id).where(KnowledgeNode.id == node_id)
        node_result = await session.execute(stmt)
        coll = node_result.scalar_one_or_none()
        if coll:
            collection_ids = [coll]
        else:
            result.summary_markdown = f"Node {node_id} not found."
            return result

    # Expand from this node
    nodes, edges = await _expand_neighborhood(session, [node_id], collection_ids, depth, max_nodes)

    # Filter edges by kind if specified
    if edge_kinds:
        edges = [e for e in edges if e.kind in edge_kinds]

    result.nodes = nodes
    result.edges = edges
    result.evidence = await _gather_evidence(session, [n.id for n in nodes])
    result.summary_markdown = _build_neighborhood_markdown(nodes, edges, result.evidence)

    return result


def _build_neighborhood_markdown(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    evidence: list[Evidence],
) -> str:
    """Build markdown for neighborhood result."""
    lines = ["## Graph Neighborhood\n"]

    lines.append(f"Found {len(nodes)} nodes and {len(edges)} edges.\n")

    if nodes:
        lines.append("### Nodes\n")
        for node in nodes:
            lines.append(f"- **{node.name}** ({node.kind})")

    if edges:
        lines.append("\n### Edges\n")
        edge_summary: dict[str, int] = {}
        for edge in edges:
            edge_summary[edge.kind] = edge_summary.get(edge.kind, 0) + 1
        for kind, count in sorted(edge_summary.items()):
            lines.append(f"- {kind}: {count}")

    return "\n".join(lines)


async def trace_path(
    session: AsyncSession,
    from_node_id: UUID,
    to_node_id: UUID,
    collection_id: UUID | None = None,
    max_hops: int = 6,
) -> GraphRAGResult:
    """Find shortest path between two nodes.

    Uses BFS to find the shortest path in the knowledge graph.

    Args:
        session: Database session
        from_node_id: Starting node ID
        to_node_id: Target node ID
        collection_id: Optional collection filter
        max_hops: Maximum path length

    Returns:
        GraphRAGResult with path nodes and edges
    """
    from collections import deque

    from contextmine_core.models import KnowledgeEdge, KnowledgeNode
    from sqlalchemy import or_, select

    result = GraphRAGResult(query=f"Path from {from_node_id} to {to_node_id}")

    # Get collection IDs
    if collection_id:
        collection_ids = [collection_id]
    else:
        stmt = select(KnowledgeNode.collection_id).where(
            KnowledgeNode.id.in_([from_node_id, to_node_id])
        )
        coll_result = await session.execute(stmt)
        collection_ids = list({row[0] for row in coll_result.fetchall()})

    if not collection_ids:
        result.summary_markdown = "Nodes not found."
        return result

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

        # Get edges from current node
        stmt = select(KnowledgeEdge).where(
            KnowledgeEdge.collection_id.in_(collection_ids),
            or_(
                KnowledgeEdge.source_node_id == current,
                KnowledgeEdge.target_node_id == current,
            ),
        )

        edge_result = await session.execute(stmt)
        for edge in edge_result.scalars().all():
            # Get the other end of the edge
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
        result.summary_markdown = f"No path found within {max_hops} hops."
        return result

    # Fetch full node data
    stmt = select(KnowledgeNode).where(KnowledgeNode.id.in_(path_nodes))
    node_result = await session.execute(stmt)

    node_map: dict[UUID, KnowledgeNode] = {n.id: n for n in node_result.scalars().all()}

    for node_id in path_nodes:
        if node_id in node_map:
            node = node_map[node_id]
            result.nodes.append(
                GraphNode(
                    id=str(node.id),
                    kind=node.kind.value,
                    name=node.name,
                    natural_key=node.natural_key,
                    meta=node.meta or {},
                )
            )

    for src, tgt, kind in path_edges:
        result.edges.append(GraphEdge(source_id=str(src), target_id=str(tgt), kind=kind, meta={}))

    result.evidence = await _gather_evidence(session, [str(n) for n in path_nodes])
    result.summary_markdown = _build_path_markdown(result.nodes, result.edges)

    return result


def _build_path_markdown(nodes: list[GraphNode], edges: list[GraphEdge]) -> str:
    """Build markdown for path result."""
    lines = ["## Path Trace\n"]

    if not nodes:
        lines.append("No path found.")
        return "\n".join(lines)

    lines.append(f"Path length: {len(nodes)} nodes, {len(edges)} edges\n")

    lines.append("### Path:")
    for i, node in enumerate(nodes):
        prefix = "→ " if i > 0 else "  "
        lines.append(f"{prefix}**{node.name}** ({node.kind})")

    return "\n".join(lines)
