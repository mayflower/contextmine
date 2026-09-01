"""PostgreSQL query proof for batched knowledge-graph path traversal."""

from __future__ import annotations

import os
import uuid
from collections import deque
from typing import TYPE_CHECKING

import pytest
from contextmine_core.graphrag_paths import find_shortest_path
from contextmine_core.models import KnowledgeEdge
from sqlalchemy import event, or_, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession


async def _legacy_shortest_path(
    session: AsyncSession,
    from_node_id: UUID,
    to_node_id: UUID,
    collection_id: UUID,
    max_hops: int,
) -> tuple[list[UUID], list[tuple[UUID, UUID, str]]]:
    """Previous one-query-per-visited-node traversal, retained as the baseline."""
    queue = deque([(from_node_id, [from_node_id], [])])
    visited = {from_node_id}

    while queue:
        current, node_path, edge_path = queue.popleft()
        if len(node_path) > max_hops + 1:
            break
        if current == to_node_id:
            return node_path, edge_path
        statement = select(KnowledgeEdge).where(
            KnowledgeEdge.collection_id == collection_id,
            or_(
                KnowledgeEdge.source_node_id == current,
                KnowledgeEdge.target_node_id == current,
            ),
        )
        edges = (await session.execute(statement)).scalars().all()
        for edge in edges:
            next_node = (
                edge.target_node_id if edge.source_node_id == current else edge.source_node_id
            )
            if next_node in visited:
                continue
            visited.add(next_node)
            queue.append(
                (
                    next_node,
                    node_path + [next_node],
                    edge_path + [(edge.source_node_id, edge.target_node_id, edge.kind.value)],
                )
            )
    return [], []


def _plan_summary(plan: list[dict[str, object]]) -> tuple[float, float, int]:
    document = plan[0]
    root = document["Plan"]
    assert isinstance(root, dict)
    return (
        float(root["Total Cost"]),
        float(document["Execution Time"]),
        int(root["Actual Rows"]),
    )


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"),
    reason="TEST_DATABASE_URL not set - skipping PostgreSQL path-query test",
)
async def test_batched_frontier_preserves_path_with_fewer_queries() -> None:
    """The batched BFS returns the same path with one query per depth."""
    database_url = os.environ["TEST_DATABASE_URL"]
    if not database_url.startswith(("postgresql+asyncpg://", "postgresql://")):
        pytest.skip("TEST_DATABASE_URL must point to PostgreSQL")
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    schema = f"graph_path_test_{uuid.uuid4().hex}"
    admin_engine = create_async_engine(database_url)
    async with admin_engine.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        await connection.execute(
            text(
                f'CREATE TABLE "{schema}".knowledge_edges ('
                "id UUID PRIMARY KEY, collection_id UUID NOT NULL, "
                "source_node_id UUID NOT NULL, target_node_id UUID NOT NULL, "
                "kind TEXT NOT NULL, meta JSON NOT NULL DEFAULT '{}', "
                "created_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            )
        )
        await connection.execute(
            text(
                f'CREATE INDEX ix_graph_path_source ON "{schema}".knowledge_edges '
                "(source_node_id, kind)"
            )
        )
        await connection.execute(
            text(
                f'CREATE INDEX ix_graph_path_target ON "{schema}".knowledge_edges '
                "(target_node_id, kind)"
            )
        )
        await connection.execute(
            text(
                f'CREATE INDEX ix_graph_path_collection ON "{schema}".knowledge_edges '
                "(collection_id, kind)"
            )
        )

    engine = create_async_engine(
        database_url,
        connect_args={"server_settings": {"search_path": schema}},
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)
    query_count = 0
    count_queries = False

    @event.listens_for(engine.sync_engine, "before_cursor_execute")
    def count_edge_selects(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        nonlocal query_count
        if count_queries and statement.lstrip().upper().startswith("SELECT"):
            query_count += statement.lower().count("from knowledge_edges")

    collection_id = uuid.uuid4()
    other_collection_id = uuid.uuid4()
    start_id, middle_1_id, middle_2_id, target_id = (uuid.uuid4() for _ in range(4))
    dead_end_ids = [uuid.uuid4() for _ in range(30)]
    path_rows = [
        {
            "id": uuid.uuid4(),
            "collection_id": collection_id,
            "source_node_id": start_id,
            "target_node_id": dead_end_id,
            "kind": "symbol_calls_symbol",
        }
        for dead_end_id in dead_end_ids
    ]
    path_rows.extend(
        [
            {
                "id": uuid.uuid4(),
                "collection_id": collection_id,
                "source_node_id": start_id,
                "target_node_id": middle_1_id,
                "kind": "symbol_calls_symbol",
            },
            {
                "id": uuid.uuid4(),
                "collection_id": collection_id,
                "source_node_id": middle_1_id,
                "target_node_id": middle_2_id,
                "kind": "symbol_calls_symbol",
            },
            {
                "id": uuid.uuid4(),
                "collection_id": collection_id,
                "source_node_id": middle_2_id,
                "target_node_id": target_id,
                "kind": "symbol_calls_symbol",
            },
        ]
    )
    unrelated_rows = [
        {
            "id": uuid.uuid4(),
            "collection_id": other_collection_id,
            "source_node_id": uuid.uuid4(),
            "target_node_id": uuid.uuid4(),
            "kind": "file_imports_file",
        }
        for _ in range(2_000)
    ]
    insert_statement = text(
        "INSERT INTO knowledge_edges "
        "(id, collection_id, source_node_id, target_node_id, kind) "
        "VALUES (:id, :collection_id, :source_node_id, :target_node_id, :kind)"
    )

    try:
        async with engine.begin() as connection:
            await connection.execute(insert_statement, path_rows + unrelated_rows)
            await connection.execute(text("ANALYZE knowledge_edges"))

        async with factory() as session:
            count_queries = True
            legacy_path = await _legacy_shortest_path(
                session, start_id, target_id, collection_id, max_hops=6
            )
            legacy_queries = query_count

            query_count = 0
            batched_path = await find_shortest_path(
                session,
                start_id,
                target_id,
                [collection_id],
                max_hops=6,
            )
            batched_queries = query_count
            count_queries = False

            legacy_plan = (
                await session.execute(
                    text(
                        "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) "
                        "SELECT * FROM knowledge_edges "
                        "WHERE collection_id = :collection_id "
                        "AND (source_node_id = :start_id OR target_node_id = :start_id)"
                    ),
                    {"collection_id": collection_id, "start_id": start_id},
                )
            ).scalar_one()
            batched_plan = (
                await session.execute(
                    text(
                        "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) "
                        "SELECT * FROM knowledge_edges "
                        "WHERE collection_id = :collection_id "
                        "AND (source_node_id = ANY(:frontier_ids) "
                        "OR target_node_id = ANY(:frontier_ids))"
                    ),
                    {"collection_id": collection_id, "frontier_ids": [start_id]},
                )
            ).scalar_one()

        assert legacy_path == batched_path
        assert batched_path[0] == [start_id, middle_1_id, middle_2_id, target_id]
        assert legacy_queries >= 33
        assert batched_queries == 3
        assert batched_queries < legacy_queries

        legacy_cost, legacy_time, legacy_rows = _plan_summary(legacy_plan)
        batched_cost, batched_time, batched_rows = _plan_summary(batched_plan)
        assert legacy_rows == batched_rows == 31
        assert legacy_cost > 0 and batched_cost > 0
        assert legacy_time >= 0 and batched_time >= 0
    finally:
        await engine.dispose()
        async with admin_engine.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await admin_engine.dispose()
