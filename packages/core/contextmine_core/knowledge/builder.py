"""Knowledge Graph builder for ContextMine.

This module builds KnowledgeNodes and KnowledgeEdges from indexed Documents and Symbols.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

from contextmine_core.analyzer.extractors.graph_helpers import (
    create_node_evidence,
    upsert_edge,
    upsert_node,
)
from contextmine_core.models import (
    Document,
    KnowledgeEdgeKind,
    KnowledgeNode,
    KnowledgeNodeKind,
    Source,
    Symbol,
    SymbolEdge,
    SymbolEdgeType,
)
from sqlalchemy import delete, select

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass
class GraphBuildStats:
    """Statistics from building a knowledge graph."""

    file_nodes_created: int = 0
    symbol_nodes_created: int = 0
    edges_created: int = 0
    evidence_created: int = 0
    nodes_deleted: int = 0
    evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_nodes_created": self.file_nodes_created,
            "symbol_nodes_created": self.symbol_nodes_created,
            "edges_created": self.edges_created,
            "evidence_created": self.evidence_created,
            "nodes_deleted": self.nodes_deleted,
        }


async def _create_file_nodes(
    session: AsyncSession,
    collection_id: UUID,
    documents: list,
    stats: GraphBuildStats,
) -> dict[UUID, UUID]:
    """Create FILE nodes for all documents, returning doc_id -> node_id map."""
    doc_to_node: dict[UUID, UUID] = {}
    for doc in documents:
        file_path = doc.meta.get("file_path", doc.uri) if doc.meta else doc.uri
        node_id = await upsert_node(
            session,
            collection_id=collection_id,
            kind=KnowledgeNodeKind.FILE,
            natural_key=doc.uri,
            name=doc.title or file_path,
            meta={"document_id": str(doc.id), "file_path": file_path, "uri": doc.uri},
        )
        stats.file_nodes_created += 1
        doc_to_node[doc.id] = node_id
    return doc_to_node


async def _create_symbol_nodes_and_edges(
    session: AsyncSession,
    collection_id: UUID,
    doc,
    file_node_id: UUID,
    symbol_to_node: dict[UUID, UUID],
    stats: GraphBuildStats,
) -> None:
    """Create SYMBOL nodes, FILE_DEFINES_SYMBOL edges, and evidence for a document."""
    result = await session.execute(select(Symbol).where(Symbol.document_id == doc.id))
    symbols = result.scalars().all()

    for symbol in symbols:
        natural_key = f"{doc.uri}::{symbol.qualified_name}"
        symbol_node_id = await upsert_node(
            session,
            collection_id=collection_id,
            kind=KnowledgeNodeKind.SYMBOL,
            natural_key=natural_key,
            name=symbol.name,
            meta={
                "symbol_id": str(symbol.id),
                "document_id": str(doc.id),
                "qualified_name": symbol.qualified_name,
                "kind": symbol.kind.value,
                "start_line": symbol.start_line,
                "end_line": symbol.end_line,
                "signature": symbol.signature,
                "parent_name": symbol.parent_name,
            },
        )
        stats.symbol_nodes_created += 1
        symbol_to_node[symbol.id] = symbol_node_id

        await upsert_edge(
            session,
            collection_id=collection_id,
            source_node_id=file_node_id,
            target_node_id=symbol_node_id,
            kind=KnowledgeEdgeKind.FILE_DEFINES_SYMBOL,
            meta={},
        )
        stats.edges_created += 1

        file_path = doc.meta.get("file_path", doc.uri) if doc.meta else doc.uri
        ev_id = await create_node_evidence(
            session,
            node_id=symbol_node_id,
            file_path=file_path,
            start_line=symbol.start_line,
            end_line=symbol.end_line,
        )
        stats.evidence_created += 1
        stats.evidence_ids.append(ev_id)

    # Create SYMBOL_CONTAINS_SYMBOL edges from parent_name
    for symbol in symbols:
        if not symbol.parent_name:
            continue
        child_node_id = symbol_to_node.get(symbol.id)
        if not child_node_id:
            continue
        parent_symbol = next(
            (s for s in symbols if s.qualified_name == symbol.parent_name),
            None,
        )
        if not parent_symbol:
            continue
        parent_node_id = symbol_to_node.get(parent_symbol.id)
        if parent_node_id:
            await upsert_edge(
                session,
                collection_id=collection_id,
                source_node_id=parent_node_id,
                target_node_id=child_node_id,
                kind=KnowledgeEdgeKind.SYMBOL_CONTAINS_SYMBOL,
                meta={},
            )
            stats.edges_created += 1


_SYMBOL_EDGE_KIND_MAP = {
    SymbolEdgeType.CALLS: KnowledgeEdgeKind.SYMBOL_CALLS_SYMBOL,
    SymbolEdgeType.REFERENCES: KnowledgeEdgeKind.SYMBOL_REFERENCES_SYMBOL,
    SymbolEdgeType.INHERITS: KnowledgeEdgeKind.SYMBOL_REFERENCES_SYMBOL,
    SymbolEdgeType.IMPLEMENTS: KnowledgeEdgeKind.SYMBOL_REFERENCES_SYMBOL,
}

# Edge types that should produce file-level (FILE_IMPORTS_FILE) edges
# instead of symbol-level edges.
_FILE_LEVEL_EDGE_TYPES = {SymbolEdgeType.IMPORTS}


async def _build_symbol_to_file_node_map(
    session: AsyncSession,
    documents: list,
    doc_to_node: dict[UUID, UUID],
) -> dict[UUID, UUID]:
    """Build a symbol_id -> file_node_id lookup in batch (one query per doc)."""
    symbol_to_file: dict[UUID, UUID] = {}
    for doc in documents:
        file_node_id = doc_to_node.get(doc.id)
        if not file_node_id:
            continue
        result = await session.execute(select(Symbol.id).where(Symbol.document_id == doc.id))
        for (sym_id,) in result.all():
            symbol_to_file[sym_id] = file_node_id
    return symbol_to_file


async def _handle_file_level_edge(
    session: AsyncSession,
    collection_id: UUID,
    sym_edge: SymbolEdge,
    symbol_to_file_node: dict[UUID, UUID],
    stats: GraphBuildStats,
) -> None:
    """Create a file-to-file edge for an IMPORTS symbol edge."""
    source_file_id = symbol_to_file_node.get(sym_edge.source_symbol_id)
    target_file_id = symbol_to_file_node.get(sym_edge.target_symbol_id)
    if not source_file_id or not target_file_id:
        logger.debug(
            "Skipping import edge: could not resolve symbol %s -> %s to file nodes",
            sym_edge.source_symbol_id,
            sym_edge.target_symbol_id,
        )
        return
    if source_file_id == target_file_id:
        return  # Skip self-imports
    await upsert_edge(
        session,
        collection_id=collection_id,
        source_node_id=source_file_id,
        target_node_id=target_file_id,
        kind=KnowledgeEdgeKind.FILE_IMPORTS_FILE,
        meta={"source_line": sym_edge.source_line} if sym_edge.source_line else {},
    )
    stats.edges_created += 1


async def _handle_symbol_level_edge(
    session: AsyncSession,
    collection_id: UUID,
    sym_edge: SymbolEdge,
    symbol_to_node: dict[UUID, UUID],
    stats: GraphBuildStats,
) -> None:
    """Create a symbol-to-symbol knowledge edge."""
    source_node_id = symbol_to_node.get(sym_edge.source_symbol_id)
    target_node_id = symbol_to_node.get(sym_edge.target_symbol_id)
    if not source_node_id or not target_node_id:
        return
    kg_edge_kind = _SYMBOL_EDGE_KIND_MAP.get(sym_edge.edge_type)
    if kg_edge_kind:
        await upsert_edge(
            session,
            collection_id=collection_id,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            kind=kg_edge_kind,
            meta={"source_line": sym_edge.source_line} if sym_edge.source_line else {},
        )
        stats.edges_created += 1


async def _create_symbol_edge_graph_edges(
    session: AsyncSession,
    collection_id: UUID,
    documents: list,
    symbol_to_node: dict[UUID, UUID],
    symbol_to_file_node: dict[UUID, UUID],
    stats: GraphBuildStats,
) -> None:
    """Create knowledge graph edges from the SymbolEdge table.

    Symbol-level edges (CALLS, REFERENCES, INHERITS, IMPLEMENTS) map to
    symbol-to-symbol knowledge edges.  IMPORTS edges are lifted to
    file-to-file edges (FILE_IMPORTS_FILE) using the pre-built
    symbol_to_file_node lookup.
    """
    for doc in documents:
        result = await session.execute(
            select(SymbolEdge)
            .join(Symbol, SymbolEdge.source_symbol_id == Symbol.id)
            .where(Symbol.document_id == doc.id)
        )
        for sym_edge in result.scalars().all():
            if sym_edge.edge_type in _FILE_LEVEL_EDGE_TYPES:
                await _handle_file_level_edge(
                    session, collection_id, sym_edge, symbol_to_file_node, stats
                )
            else:
                await _handle_symbol_level_edge(
                    session, collection_id, sym_edge, symbol_to_node, stats
                )


async def build_knowledge_graph_for_source(
    session: AsyncSession,
    source_id: UUID,
) -> GraphBuildStats:
    """Build knowledge graph nodes and edges for a source.

    Creates FILE/SYMBOL nodes and associated edges. Uses upsert for idempotency.
    """
    stats = GraphBuildStats()

    result = await session.execute(select(Source.collection_id).where(Source.id == source_id))
    collection_id = result.scalar_one_or_none()
    if not collection_id:
        logger.warning("Source %s not found", source_id)
        return stats

    result = await session.execute(select(Document).where(Document.source_id == source_id))
    documents = result.scalars().all()

    doc_to_node = await _create_file_nodes(session, collection_id, documents, stats)
    symbol_to_node: dict[UUID, UUID] = {}

    for doc in documents:
        file_node_id = doc_to_node.get(doc.id)
        if not file_node_id:
            continue
        await _create_symbol_nodes_and_edges(
            session,
            collection_id,
            doc,
            file_node_id,
            symbol_to_node,
            stats,
        )

    symbol_to_file_node = await _build_symbol_to_file_node_map(
        session,
        documents,
        doc_to_node,
    )
    await _create_symbol_edge_graph_edges(
        session,
        collection_id,
        documents,
        symbol_to_node,
        symbol_to_file_node,
        stats,
    )
    return stats


async def cleanup_orphan_nodes(
    session: AsyncSession,
    collection_id: UUID,
    source_id: UUID,
) -> dict[str, int]:
    """Remove knowledge nodes for deleted documents/symbols.

    Compares existing FILE nodes against current documents and SYMBOL nodes
    against current symbols, removing any that no longer have a corresponding
    entity.

    Args:
        session: Database session
        collection_id: Collection UUID
        source_id: Source UUID

    Returns:
        Stats dict with nodes_deleted
    """
    stats: dict[str, int] = {"nodes_deleted": 0}

    # Get all document URIs for this source
    result = await session.execute(select(Document.uri).where(Document.source_id == source_id))
    current_uris = {row[0] for row in result.all()}

    # Get FILE nodes that no longer have matching documents
    result = await session.execute(
        select(KnowledgeNode.id, KnowledgeNode.natural_key).where(
            KnowledgeNode.collection_id == collection_id,
            KnowledgeNode.kind == KnowledgeNodeKind.FILE,
        )
    )

    nodes_to_delete = []
    for node_id, natural_key in result.all():
        if natural_key not in current_uris:
            nodes_to_delete.append(node_id)

    # Get current symbol qualified names for this source
    result = await session.execute(
        select(Document.uri, Symbol.qualified_name)
        .join(Symbol, Symbol.document_id == Document.id)
        .where(Document.source_id == source_id)
    )
    current_symbol_keys = {f"{uri}::{qn}" for uri, qn in result.all()}

    # Only check SYMBOL nodes scoped to this source's document URIs
    # (natural_key format: "{uri}::{qualified_name}")
    uri_prefix_set = {f"{uri}::" for uri in current_uris}
    result = await session.execute(
        select(KnowledgeNode.id, KnowledgeNode.natural_key).where(
            KnowledgeNode.collection_id == collection_id,
            KnowledgeNode.kind == KnowledgeNodeKind.SYMBOL,
        )
    )
    for node_id, natural_key in result.all():
        if (
            any(natural_key.startswith(prefix) for prefix in uri_prefix_set)
            and natural_key not in current_symbol_keys
        ):
            nodes_to_delete.append(node_id)

    if nodes_to_delete:
        # Delete nodes (cascades to edges and evidence links)
        await session.execute(delete(KnowledgeNode).where(KnowledgeNode.id.in_(nodes_to_delete)))
        stats["nodes_deleted"] = len(nodes_to_delete)

    return stats
