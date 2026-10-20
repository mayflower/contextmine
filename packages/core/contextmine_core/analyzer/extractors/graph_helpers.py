"""Shared utilities for knowledge graph materialization.

Centralizes the upsert / evidence / provenance patterns used by all
``build_*_graph`` functions so that each extractor module does not need
its own private copy.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any
from uuid import UUID

from contextmine_core.models import (
    KnowledgeEdge,
    KnowledgeEdgeKind,
    KnowledgeNode,
    KnowledgeNodeEvidence,
    KnowledgeNodeKind,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def provenance(
    *,
    mode: str,
    extractor: str,
    confidence: float,
    evidence_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Build a standardised provenance metadata dict."""
    return {
        "provenance": {
            "mode": mode,
            "extractor": extractor,
            "confidence": round(max(0.0, min(confidence, 1.0)), 4),
            "evidence_ids": list(dict.fromkeys(evidence_ids or [])),
        }
    }


def content_hash(value: str) -> str:
    """Short SHA-256 content hash for natural key construction."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]


def dedupe_strings(values: list[str]) -> list[str]:
    """Deduplicate and strip a list of strings, preserving order."""
    return list(dict.fromkeys(v.strip() for v in values if v and v.strip()))


async def upsert_node(
    session: AsyncSession,
    *,
    collection_id: UUID,
    kind: KnowledgeNodeKind,
    natural_key: str,
    name: str,
    meta: dict[str, Any],
) -> UUID:
    """Insert or update a knowledge graph node, returning its id."""
    stmt = pg_insert(KnowledgeNode).values(
        collection_id=collection_id,
        kind=kind,
        natural_key=natural_key,
        name=name,
        meta=meta,
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_knowledge_node_natural",
        set_={"name": stmt.excluded.name, "meta": stmt.excluded.meta},
    ).returning(KnowledgeNode.id)
    return (await session.execute(stmt)).scalar_one()


async def upsert_edge(
    session: AsyncSession,
    *,
    collection_id: UUID,
    source_node_id: UUID,
    target_node_id: UUID,
    kind: KnowledgeEdgeKind,
    meta: dict[str, Any],
) -> UUID:
    """Insert or update a knowledge graph edge, returning its id."""
    stmt = pg_insert(KnowledgeEdge).values(
        collection_id=collection_id,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        kind=kind,
        meta=meta,
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_knowledge_edge_unique",
        set_={"meta": stmt.excluded.meta},
    ).returning(KnowledgeEdge.id)
    return (await session.execute(stmt)).scalar_one()


# Rows per multi-row INSERT; stays well below PostgreSQL's 32767 bind parameters.
BULK_UPSERT_BATCH_SIZE = 1000


async def bulk_upsert_nodes(
    session: AsyncSession,
    *,
    collection_id: UUID,
    rows: Iterable[tuple[KnowledgeNodeKind, str, str, dict[str, Any]]],
) -> dict[tuple[KnowledgeNodeKind, str], UUID]:
    """Same result as calling upsert_node for each (kind, natural_key, name, meta) in order.

    Duplicate keys keep their last values, like consecutive upserts would.
    Returns the node id for every (kind, natural_key).
    """
    latest: dict[tuple[KnowledgeNodeKind, str], tuple[str, dict[str, Any]]] = {}
    for kind, natural_key, name, meta in rows:
        latest[(kind, natural_key)] = (name, meta)
    ids: dict[tuple[KnowledgeNodeKind, str], UUID] = {}
    items = list(latest.items())
    for start in range(0, len(items), BULK_UPSERT_BATCH_SIZE):
        batch = items[start : start + BULK_UPSERT_BATCH_SIZE]
        stmt = pg_insert(KnowledgeNode).values(
            [
                {
                    "collection_id": collection_id,
                    "kind": kind,
                    "natural_key": natural_key,
                    "name": name,
                    "meta": meta,
                }
                for (kind, natural_key), (name, meta) in batch
            ]
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_knowledge_node_natural",
            set_={"name": stmt.excluded.name, "meta": stmt.excluded.meta},
        ).returning(KnowledgeNode.id, KnowledgeNode.kind, KnowledgeNode.natural_key)
        for row in (await session.execute(stmt)).all():
            ids[(row.kind, row.natural_key)] = row.id
    return ids


async def bulk_upsert_edges(
    session: AsyncSession,
    *,
    collection_id: UUID,
    rows: Iterable[tuple[UUID, UUID, KnowledgeEdgeKind, dict[str, Any]]],
) -> None:
    """Same result as calling upsert_edge for each (source, target, kind, meta) in order."""
    latest: dict[tuple[UUID, UUID, KnowledgeEdgeKind], dict[str, Any]] = {}
    for source_node_id, target_node_id, kind, meta in rows:
        latest[(source_node_id, target_node_id, kind)] = meta
    items = list(latest.items())
    for start in range(0, len(items), BULK_UPSERT_BATCH_SIZE):
        batch = items[start : start + BULK_UPSERT_BATCH_SIZE]
        stmt = pg_insert(KnowledgeEdge).values(
            [
                {
                    "collection_id": collection_id,
                    "source_node_id": source_node_id,
                    "target_node_id": target_node_id,
                    "kind": kind,
                    "meta": meta,
                }
                for (source_node_id, target_node_id, kind), meta in batch
            ]
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_knowledge_edge_unique",
            set_={"meta": stmt.excluded.meta},
        )
        await session.execute(stmt)


async def create_node_evidence(
    session: AsyncSession,
    *,
    node_id: UUID,
    file_path: str,
    start_line: int,
    end_line: int | None = None,
    snippet: str | None = None,
) -> str:
    """Create an evidence record linked to a node, returning the evidence id."""
    from contextmine_core.models import KnowledgeEvidence

    evidence = KnowledgeEvidence(
        file_path=file_path,
        start_line=max(1, start_line),
        end_line=max(1, end_line or start_line),
        snippet=snippet,
    )
    session.add(evidence)
    await session.flush()
    session.add(KnowledgeNodeEvidence(node_id=node_id, evidence_id=evidence.id))
    return str(evidence.id)
