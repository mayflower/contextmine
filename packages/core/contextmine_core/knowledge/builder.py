"""Knowledge Graph builder from semantic snapshots.

This module builds KnowledgeNodes and KnowledgeEdges from:
- Semantic Snapshots (preferred, language-agnostic)
- Legacy: Documents/Symbols tables (backwards compatibility)

The Snapshot-based builder is the primary interface for GraphRAG.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from contextmine_core.models import (
    Document,
    KnowledgeEdge,
    KnowledgeEdgeKind,
    KnowledgeEvidence,
    KnowledgeNode,
    KnowledgeNodeEvidence,
    KnowledgeNodeKind,
    Source,
    Symbol,
    SymbolEdge,
    SymbolEdgeType,
)
from contextmine_core.semantic_snapshot import (
    Occurrence as SnapshotOccurrence,
)
from contextmine_core.semantic_snapshot import (
    OccurrenceRole,
    RelationKind,
    Snapshot,
)
from contextmine_core.semantic_snapshot import Symbol as SnapshotSymbol
from contextmine_core.semantic_snapshot import SymbolKind as SnapshotSymbolKind
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# Map snapshot SymbolKind to KnowledgeNodeKind meta field values
SNAPSHOT_KIND_TO_META: dict[SnapshotSymbolKind, str] = {
    SnapshotSymbolKind.FUNCTION: "function",
    SnapshotSymbolKind.METHOD: "method",
    SnapshotSymbolKind.CLASS: "class",
    SnapshotSymbolKind.STRUCT: "struct",
    SnapshotSymbolKind.INTERFACE: "interface",
    SnapshotSymbolKind.ENUM: "enum",
    SnapshotSymbolKind.TRAIT: "trait",
    SnapshotSymbolKind.TYPE_ALIAS: "type_alias",
    SnapshotSymbolKind.VARIABLE: "variable",
    SnapshotSymbolKind.CONSTANT: "constant",
    SnapshotSymbolKind.MODULE: "module",
    SnapshotSymbolKind.PROPERTY: "property",
    SnapshotSymbolKind.PARAMETER: "parameter",
    SnapshotSymbolKind.UNKNOWN: "unknown",
}

# Map snapshot RelationKind to KnowledgeEdgeKind
RELATION_KIND_TO_EDGE: dict[RelationKind, KnowledgeEdgeKind] = {
    RelationKind.CONTAINS: KnowledgeEdgeKind.SYMBOL_CONTAINS_SYMBOL,
    RelationKind.CALLS: KnowledgeEdgeKind.SYMBOL_CALLS_SYMBOL,
    RelationKind.REFERENCES: KnowledgeEdgeKind.SYMBOL_REFERENCES_SYMBOL,
    RelationKind.IMPORTS: KnowledgeEdgeKind.FILE_IMPORTS_FILE,
}


@dataclass
class GraphBuildStats:
    """Statistics from building a knowledge graph."""

    file_nodes_created: int = 0
    file_nodes_updated: int = 0
    symbol_nodes_created: int = 0
    symbol_nodes_updated: int = 0
    edges_created: int = 0
    edges_skipped: int = 0
    evidence_created: int = 0
    nodes_deleted: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "file_nodes_created": self.file_nodes_created,
            "file_nodes_updated": self.file_nodes_updated,
            "symbol_nodes_created": self.symbol_nodes_created,
            "symbol_nodes_updated": self.symbol_nodes_updated,
            "edges_created": self.edges_created,
            "edges_skipped": self.edges_skipped,
            "evidence_created": self.evidence_created,
            "nodes_deleted": self.nodes_deleted,
        }


async def build_graph_from_snapshot(
    session: AsyncSession,
    collection_id: UUID,
    snapshot: Snapshot,
    changed_files: set[str] | None = None,
    deleted_files: set[str] | None = None,
) -> GraphBuildStats:
    """Build knowledge graph from a semantic snapshot.

    This is the primary entry point for GraphRAG graph construction.

    Natural key conventions:
    - FILE: file:{path}
    - SYMBOL: symbol:{def_id}

    Args:
        session: Database session
        collection_id: Collection UUID
        snapshot: Semantic snapshot to build from
        changed_files: Optional set of changed file paths (for incremental)
        deleted_files: Optional set of deleted file paths

    Returns:
        GraphBuildStats with counts
    """
    stats = GraphBuildStats()

    # Handle deletions first
    if deleted_files:
        for file_path in deleted_files:
            deleted = await _delete_file_and_symbols(session, collection_id, file_path)
            stats.nodes_deleted += deleted

    # Build def_id -> node_id mapping for edge creation
    def_id_to_node: dict[str, UUID] = {}
    file_path_to_node: dict[str, UUID] = {}

    # Create FILE nodes
    for file_info in snapshot.files:
        # Skip unchanged files in incremental mode
        if changed_files is not None and file_info.path not in changed_files:
            continue

        node_id, created = await _upsert_file_node_from_snapshot(
            session, collection_id, file_info.path, file_info.language
        )
        if created:
            stats.file_nodes_created += 1
        else:
            stats.file_nodes_updated += 1

        file_path_to_node[file_info.path] = node_id

    # Create SYMBOL nodes
    for symbol in snapshot.symbols:
        # Skip symbols in unchanged files in incremental mode
        if changed_files is not None and symbol.file_path not in changed_files:
            continue

        node_id, created = await _upsert_symbol_node_from_snapshot(session, collection_id, symbol)
        if created:
            stats.symbol_nodes_created += 1
        else:
            stats.symbol_nodes_updated += 1

        def_id_to_node[symbol.def_id] = node_id

        # Create FILE_DEFINES_SYMBOL edge
        file_node_id = file_path_to_node.get(symbol.file_path)
        if file_node_id:
            edge_created = await _upsert_edge_idempotent(
                session,
                collection_id,
                source_node_id=file_node_id,
                target_node_id=node_id,
                kind=KnowledgeEdgeKind.FILE_DEFINES_SYMBOL,
            )
            if edge_created:
                stats.edges_created += 1

    # Create evidence from definition occurrences
    for occurrence in snapshot.occurrences:
        if occurrence.role != OccurrenceRole.DEFINITION:
            continue

        node_id = def_id_to_node.get(occurrence.def_id)
        if not node_id:
            continue

        ev_created = await _create_evidence_from_occurrence(session, node_id, occurrence)
        if ev_created:
            stats.evidence_created += 1

    # Create edges from relations
    for relation in snapshot.relations:
        src_node_id = def_id_to_node.get(relation.src_def_id)

        # For unresolved relations, skip (or we could create placeholder nodes)
        if not relation.resolved:
            stats.edges_skipped += 1
            continue

        dst_node_id = def_id_to_node.get(relation.dst_def_id)
        if not src_node_id or not dst_node_id:
            stats.edges_skipped += 1
            continue

        edge_kind = RELATION_KIND_TO_EDGE.get(relation.kind)
        if not edge_kind:
            stats.edges_skipped += 1
            continue

        edge_created = await _upsert_edge_idempotent(
            session,
            collection_id,
            source_node_id=src_node_id,
            target_node_id=dst_node_id,
            kind=edge_kind,
            meta={"resolved": relation.resolved, **relation.meta},
        )
        if edge_created:
            stats.edges_created += 1

    return stats


async def _upsert_file_node_from_snapshot(
    session: AsyncSession,
    collection_id: UUID,
    file_path: str,
    language: str | None,
) -> tuple[UUID, bool]:
    """Upsert a FILE node from snapshot FileInfo.

    Returns:
        Tuple of (node_id, was_created)
    """
    natural_key = f"file:{file_path}"

    stmt = pg_insert(KnowledgeNode).values(
        collection_id=collection_id,
        kind=KnowledgeNodeKind.FILE,
        natural_key=natural_key,
        name=file_path.split("/")[-1],  # Filename as display name
        meta={
            "file_path": file_path,
            "language": language,
        },
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_knowledge_node_natural",
        set_={
            "name": stmt.excluded.name,
            "meta": stmt.excluded.meta,
        },
    ).returning(KnowledgeNode.id)

    result = await session.execute(stmt)
    node_id = result.scalar_one()

    # Check if insert or update (simplified: assume created if new)
    # In practice, we'd need xmax checking for PostgreSQL
    return node_id, True


async def _upsert_symbol_node_from_snapshot(
    session: AsyncSession,
    collection_id: UUID,
    symbol: SnapshotSymbol,
) -> tuple[UUID, bool]:
    """Upsert a SYMBOL node from snapshot Symbol.

    Returns:
        Tuple of (node_id, was_created)
    """
    natural_key = f"symbol:{symbol.def_id}"

    stmt = pg_insert(KnowledgeNode).values(
        collection_id=collection_id,
        kind=KnowledgeNodeKind.SYMBOL,
        natural_key=natural_key,
        name=symbol.name or symbol.def_id.split(":")[-1],
        meta={
            "def_id": symbol.def_id,
            "file_path": symbol.file_path,
            "kind": SNAPSHOT_KIND_TO_META.get(symbol.kind, "unknown"),
            "start_line": symbol.range.start_line,
            "end_line": symbol.range.end_line,
            "start_col": symbol.range.start_col,
            "end_col": symbol.range.end_col,
            "container_def_id": symbol.container_def_id,
        },
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_knowledge_node_natural",
        set_={
            "name": stmt.excluded.name,
            "meta": stmt.excluded.meta,
        },
    ).returning(KnowledgeNode.id)

    result = await session.execute(stmt)
    node_id = result.scalar_one()

    return node_id, True


async def _upsert_edge_idempotent(
    session: AsyncSession,
    collection_id: UUID,
    source_node_id: UUID,
    target_node_id: UUID,
    kind: KnowledgeEdgeKind,
    meta: dict | None = None,
) -> bool:
    """Create an edge if it doesn't exist.

    Returns:
        True if edge was created, False if already existed
    """
    # Check if edge already exists
    result = await session.execute(
        select(KnowledgeEdge.id).where(
            KnowledgeEdge.collection_id == collection_id,
            KnowledgeEdge.source_node_id == source_node_id,
            KnowledgeEdge.target_node_id == target_node_id,
            KnowledgeEdge.kind == kind,
        )
    )
    if result.scalar_one_or_none():
        return False

    edge = KnowledgeEdge(
        collection_id=collection_id,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        kind=kind,
        meta=meta or {},
    )
    session.add(edge)
    return True


async def _create_evidence_from_occurrence(
    session: AsyncSession,
    node_id: UUID,
    occurrence: SnapshotOccurrence,
) -> bool:
    """Create evidence for a node from an occurrence.

    Returns:
        True if evidence was created, False if already existed
    """

    # Check if evidence already exists for this node
    result = await session.execute(
        select(KnowledgeNodeEvidence.evidence_id).where(KnowledgeNodeEvidence.node_id == node_id)
    )
    if result.scalar_one_or_none():
        return False

    # Create evidence
    evidence = KnowledgeEvidence(
        file_path=occurrence.file_path,
        start_line=occurrence.range.start_line,
        end_line=occurrence.range.end_line,
    )
    session.add(evidence)
    await session.flush()

    # Link to node
    link = KnowledgeNodeEvidence(
        node_id=node_id,
        evidence_id=evidence.id,
    )
    session.add(link)

    return True


async def _delete_file_and_symbols(
    session: AsyncSession,
    collection_id: UUID,
    file_path: str,
) -> int:
    """Delete FILE node and associated SYMBOL nodes for a file.

    Returns:
        Number of nodes deleted
    """
    # Delete FILE node (cascades to edges)
    file_natural_key = f"file:{file_path}"
    result = await session.execute(
        delete(KnowledgeNode)
        .where(
            KnowledgeNode.collection_id == collection_id,
            KnowledgeNode.kind == KnowledgeNodeKind.FILE,
            KnowledgeNode.natural_key == file_natural_key,
        )
        .returning(KnowledgeNode.id)
    )
    deleted = len(result.all())

    # Delete SYMBOL nodes for this file (based on meta.file_path)
    # This is tricky because we stored file_path in meta
    # We use a subquery to find matching nodes
    result = await session.execute(
        select(KnowledgeNode.id).where(
            KnowledgeNode.collection_id == collection_id,
            KnowledgeNode.kind == KnowledgeNodeKind.SYMBOL,
            KnowledgeNode.meta["file_path"].astext == file_path,
        )
    )
    symbol_node_ids = [row[0] for row in result.all()]

    if symbol_node_ids:
        await session.execute(delete(KnowledgeNode).where(KnowledgeNode.id.in_(symbol_node_ids)))
        deleted += len(symbol_node_ids)

    return deleted


# =============================================================================
# Legacy builder (backwards compatibility)
# =============================================================================


async def build_knowledge_graph_for_source(
    session: AsyncSession,
    source_id: UUID,
) -> dict:
    """Build knowledge graph nodes and edges for a source.

    Creates:
    - FILE nodes for each document
    - SYMBOL nodes for each symbol
    - FILE_DEFINES_SYMBOL edges
    - SYMBOL_CONTAINS_SYMBOL edges (from parent_name)
    - Other edges from SymbolEdge table

    Uses upsert to be idempotent on re-run.

    Args:
        session: Database session
        source_id: Source UUID to build graph for

    Returns:
        Stats dict with nodes_created, edges_created, etc.
    """
    stats = {
        "file_nodes_created": 0,
        "file_nodes_updated": 0,
        "symbol_nodes_created": 0,
        "symbol_nodes_updated": 0,
        "edges_created": 0,
        "evidence_created": 0,
    }

    # Get source and collection info
    result = await session.execute(select(Source.collection_id).where(Source.id == source_id))
    collection_id = result.scalar_one_or_none()
    if not collection_id:
        logger.warning("Source %s not found", source_id)
        return stats

    # Get all documents for this source
    result = await session.execute(select(Document).where(Document.source_id == source_id))
    documents = result.scalars().all()

    # Map document_id -> knowledge_node_id for later edge creation
    doc_to_node: dict[UUID, UUID] = {}
    symbol_to_node: dict[UUID, UUID] = {}

    # Create FILE nodes for documents
    for doc in documents:
        node_stats = await _upsert_file_node(session, collection_id, doc)
        stats["file_nodes_created"] += node_stats.get("created", 0)
        stats["file_nodes_updated"] += node_stats.get("updated", 0)
        if node_stats.get("node_id"):
            doc_to_node[doc.id] = node_stats["node_id"]

    # Create SYMBOL nodes and edges
    for doc in documents:
        file_node_id = doc_to_node.get(doc.id)
        if not file_node_id:
            continue

        # Get symbols for this document
        result = await session.execute(select(Symbol).where(Symbol.document_id == doc.id))
        symbols = result.scalars().all()

        # Build symbol -> parent mapping for containment edges
        symbol_parent_map: dict[str, str] = {}  # qualified_name -> parent_name

        for symbol in symbols:
            # Create SYMBOL node
            node_stats = await _upsert_symbol_node(session, collection_id, doc, symbol)
            stats["symbol_nodes_created"] += node_stats.get("created", 0)
            stats["symbol_nodes_updated"] += node_stats.get("updated", 0)
            if node_stats.get("node_id"):
                symbol_to_node[symbol.id] = node_stats["node_id"]

            # Track parent relationship
            if symbol.parent_name:
                symbol_parent_map[symbol.qualified_name] = symbol.parent_name

            # Create FILE_DEFINES_SYMBOL edge
            symbol_node_id = node_stats.get("node_id")
            if symbol_node_id:
                edge_stats = await _upsert_edge(
                    session,
                    collection_id,
                    source_node_id=file_node_id,
                    target_node_id=symbol_node_id,
                    kind=KnowledgeEdgeKind.FILE_DEFINES_SYMBOL,
                )
                stats["edges_created"] += edge_stats.get("created", 0)

                # Create evidence linking symbol node to source location
                ev_stats = await _create_symbol_evidence(
                    session,
                    symbol_node_id,
                    doc.id,
                    doc.meta.get("file_path", doc.uri) if doc.meta else doc.uri,
                    symbol.start_line,
                    symbol.end_line,
                )
                stats["evidence_created"] += ev_stats.get("created", 0)

        # Create SYMBOL_CONTAINS_SYMBOL edges from parent_name
        for symbol in symbols:
            if not symbol.parent_name:
                continue

            child_node_id = symbol_to_node.get(symbol.id)
            if not child_node_id:
                continue

            # Find parent symbol by qualified name
            parent_symbol = next(
                (s for s in symbols if s.qualified_name == symbol.parent_name),
                None,
            )
            if parent_symbol:
                parent_node_id = symbol_to_node.get(parent_symbol.id)
                if parent_node_id:
                    edge_stats = await _upsert_edge(
                        session,
                        collection_id,
                        source_node_id=parent_node_id,
                        target_node_id=child_node_id,
                        kind=KnowledgeEdgeKind.SYMBOL_CONTAINS_SYMBOL,
                    )
                    stats["edges_created"] += edge_stats.get("created", 0)

    # Create edges from SymbolEdge table
    edge_kind_map = {
        SymbolEdgeType.CALLS: KnowledgeEdgeKind.SYMBOL_CALLS_SYMBOL,
        SymbolEdgeType.REFERENCES: KnowledgeEdgeKind.SYMBOL_REFERENCES_SYMBOL,
        SymbolEdgeType.IMPORTS: KnowledgeEdgeKind.FILE_IMPORTS_FILE,
    }

    for doc in documents:
        result = await session.execute(
            select(SymbolEdge)
            .join(Symbol, SymbolEdge.source_symbol_id == Symbol.id)
            .where(Symbol.document_id == doc.id)
        )
        symbol_edges = result.scalars().all()

        for sym_edge in symbol_edges:
            source_node_id = symbol_to_node.get(sym_edge.source_symbol_id)
            target_node_id = symbol_to_node.get(sym_edge.target_symbol_id)

            if not source_node_id or not target_node_id:
                continue

            kg_edge_kind = edge_kind_map.get(sym_edge.edge_type)
            if kg_edge_kind:
                edge_stats = await _upsert_edge(
                    session,
                    collection_id,
                    source_node_id=source_node_id,
                    target_node_id=target_node_id,
                    kind=kg_edge_kind,
                    meta={"source_line": sym_edge.source_line} if sym_edge.source_line else {},
                )
                stats["edges_created"] += edge_stats.get("created", 0)

    return stats


async def _upsert_file_node(
    session: AsyncSession,
    collection_id: UUID,
    doc: Document,
) -> dict:
    """Upsert a FILE node for a document."""
    natural_key = doc.uri
    file_path = doc.meta.get("file_path", doc.uri) if doc.meta else doc.uri

    stmt = pg_insert(KnowledgeNode).values(
        collection_id=collection_id,
        kind=KnowledgeNodeKind.FILE,
        natural_key=natural_key,
        name=doc.title or file_path,
        meta={
            "document_id": str(doc.id),
            "file_path": file_path,
            "uri": doc.uri,
        },
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_knowledge_node_natural",
        set_={
            "name": stmt.excluded.name,
            "meta": stmt.excluded.meta,
        },
    ).returning(KnowledgeNode.id)

    result = await session.execute(stmt)
    node_id = result.scalar_one()

    # Check if this was an insert or update (simplified: count as created)
    return {"node_id": node_id, "created": 1, "updated": 0}


async def _upsert_symbol_node(
    session: AsyncSession,
    collection_id: UUID,
    doc: Document,
    symbol: Symbol,
) -> dict:
    """Upsert a SYMBOL node for a symbol."""
    # Natural key: file_uri::qualified_name
    natural_key = f"{doc.uri}::{symbol.qualified_name}"

    stmt = pg_insert(KnowledgeNode).values(
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
    stmt = stmt.on_conflict_do_update(
        constraint="uq_knowledge_node_natural",
        set_={
            "name": stmt.excluded.name,
            "meta": stmt.excluded.meta,
        },
    ).returning(KnowledgeNode.id)

    result = await session.execute(stmt)
    node_id = result.scalar_one()

    return {"node_id": node_id, "created": 1, "updated": 0}


async def _upsert_edge(
    session: AsyncSession,
    collection_id: UUID,
    source_node_id: UUID,
    target_node_id: UUID,
    kind: KnowledgeEdgeKind,
    meta: dict | None = None,
) -> dict:
    """Create an edge if it doesn't exist."""
    # Check if edge already exists
    result = await session.execute(
        select(KnowledgeEdge.id).where(
            KnowledgeEdge.collection_id == collection_id,
            KnowledgeEdge.source_node_id == source_node_id,
            KnowledgeEdge.target_node_id == target_node_id,
            KnowledgeEdge.kind == kind,
        )
    )
    existing = result.scalar_one_or_none()

    if existing:
        return {"created": 0}

    edge = KnowledgeEdge(
        collection_id=collection_id,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        kind=kind,
        meta=meta or {},
    )
    session.add(edge)
    return {"created": 1}


async def _create_symbol_evidence(
    session: AsyncSession,
    node_id: UUID,
    document_id: UUID,
    file_path: str,
    start_line: int,
    end_line: int,
) -> dict:
    """Create evidence for a symbol node."""
    # Check if evidence already exists for this node
    result = await session.execute(
        select(KnowledgeNodeEvidence.evidence_id).where(KnowledgeNodeEvidence.node_id == node_id)
    )
    existing = result.scalar_one_or_none()
    if existing:
        return {"created": 0}

    # Create evidence
    evidence = KnowledgeEvidence(
        document_id=document_id,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
    )
    session.add(evidence)
    await session.flush()

    # Link to node
    link = KnowledgeNodeEvidence(
        node_id=node_id,
        evidence_id=evidence.id,
    )
    session.add(link)

    return {"created": 1}


async def cleanup_orphan_nodes(
    session: AsyncSession,
    collection_id: UUID,
    source_id: UUID,
) -> dict:
    """Remove knowledge nodes for deleted documents/symbols.

    Compares existing nodes against current documents and removes any
    that no longer have a corresponding document.

    Args:
        session: Database session
        collection_id: Collection UUID
        source_id: Source UUID

    Returns:
        Stats dict with nodes_deleted
    """
    stats = {"nodes_deleted": 0}

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
    file_nodes = result.all()

    nodes_to_delete = []
    for node_id, natural_key in file_nodes:
        if natural_key not in current_uris:
            nodes_to_delete.append(node_id)

    if nodes_to_delete:
        # Delete nodes (cascades to edges and evidence links)
        await session.execute(delete(KnowledgeNode).where(KnowledgeNode.id.in_(nodes_to_delete)))
        stats["nodes_deleted"] = len(nodes_to_delete)

    return stats
