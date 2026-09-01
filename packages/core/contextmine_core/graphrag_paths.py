"""Bounded PostgreSQL queries for knowledge-graph paths."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from contextmine_core.models import KnowledgeEdge
from sqlalchemy import bindparam, or_, select
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.sql import Select


_UUID_ARRAY = ARRAY(PG_UUID(as_uuid=True))


def path_frontier_statement(
    collection_ids: list[UUID],
    allowed_node_ids: set[UUID] | None,
) -> Select:
    """Build one indexed edge query for a complete BFS frontier."""
    frontier = bindparam("frontier_ids", type_=_UUID_ARRAY)
    statement = select(KnowledgeEdge).where(
        KnowledgeEdge.collection_id.in_(collection_ids),
        or_(
            KnowledgeEdge.source_node_id == frontier.any_(),
            KnowledgeEdge.target_node_id == frontier.any_(),
        ),
    )
    if allowed_node_ids is not None:
        allowed = bindparam("allowed_node_ids", type_=_UUID_ARRAY)
        statement = statement.where(
            KnowledgeEdge.source_node_id == allowed.any_(),
            KnowledgeEdge.target_node_id == allowed.any_(),
        )
    return statement


async def find_shortest_path(
    session: AsyncSession,
    from_node_id: UUID,
    to_node_id: UUID,
    collection_ids: list[UUID],
    max_hops: int,
    allowed_node_ids: set[UUID] | None = None,
) -> tuple[list[UUID], list[tuple[UUID, UUID, str]]]:
    """Find a shortest undirected path with one query per BFS depth."""
    if from_node_id == to_node_id:
        return [from_node_id], []

    visited: set[UUID] = {from_node_id}
    paths: dict[UUID, tuple[list[UUID], list[tuple[UUID, UUID, str]]]] = {
        from_node_id: ([from_node_id], [])
    }
    frontier = [from_node_id]
    statement = path_frontier_statement(collection_ids, allowed_node_ids)

    for _depth in range(max_hops):
        if not frontier:
            break
        parameters: dict[str, object] = {"frontier_ids": frontier}
        if allowed_node_ids is not None:
            parameters["allowed_node_ids"] = list(allowed_node_ids)
        edges = list((await session.execute(statement, parameters)).scalars().all())

        edges_by_node: dict[UUID, list[KnowledgeEdge]] = {node_id: [] for node_id in frontier}
        for edge in edges:
            if edge.source_node_id in edges_by_node:
                edges_by_node[edge.source_node_id].append(edge)
            if edge.target_node_id in edges_by_node and edge.target_node_id != edge.source_node_id:
                edges_by_node[edge.target_node_id].append(edge)

        next_frontier: list[UUID] = []
        for current in frontier:
            node_path, edge_path = paths[current]
            for edge in edges_by_node[current]:
                next_node = (
                    edge.target_node_id if edge.source_node_id == current else edge.source_node_id
                )
                if allowed_node_ids is not None and next_node not in allowed_node_ids:
                    continue
                if next_node in visited:
                    continue
                visited.add(next_node)
                next_path = node_path + [next_node]
                next_edges = edge_path + [
                    (edge.source_node_id, edge.target_node_id, edge.kind.value)
                ]
                if next_node == to_node_id:
                    return next_path, next_edges
                paths[next_node] = (next_path, next_edges)
                next_frontier.append(next_node)
        frontier = next_frontier

    return [], []
