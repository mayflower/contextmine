"""AGE twin graph sync against a real PostgreSQL with Apache AGE."""

from __future__ import annotations

import os
import uuid
from types import SimpleNamespace

import pytest
from contextmine_core.graph.age import (
    _age_cypher_sql,
    _edge_batch_cypher,
    _node_batch_cypher,
    ensure_age_ready,
)
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"),
    reason="TEST_DATABASE_URL not set - skipping PostgreSQL AGE sync test",
)
async def test_edge_batches_connect_the_matching_nodes() -> None:
    database_url = os.environ["TEST_DATABASE_URL"]
    if not database_url.startswith(("postgresql+asyncpg://", "postgresql://")):
        pytest.skip("TEST_DATABASE_URL must point to PostgreSQL")
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    graph_name = f"age_sync_test_{uuid.uuid4().hex}"
    nodes = [
        SimpleNamespace(id=uuid.uuid4(), natural_key=f"key:{i}", kind="symbol", name=f"n{i}")
        for i in range(50)
    ]
    edges = [
        SimpleNamespace(source_node_id=nodes[i].id, target_node_id=nodes[i + 1].id, kind="calls")
        for i in range(49)
    ]

    engine = create_async_engine(database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with factory() as session:
            await ensure_age_ready(session)
            conn = await session.connection()
            await conn.exec_driver_sql(f"SELECT create_graph('{graph_name}')")
            await conn.exec_driver_sql(_age_cypher_sql(graph_name, _node_batch_cypher(nodes)))
            await conn.exec_driver_sql(_age_cypher_sql(graph_name, _edge_batch_cypher(edges)))
            result = await conn.exec_driver_sql(
                _age_cypher_sql(
                    graph_name,
                    "MATCH (s:Node)-[r:REL]->(t:Node) RETURN [s.id, t.id, r.kind]",
                )
            )
            created = sorted(str(row[0]) for row in result.all())
            await conn.exec_driver_sql(f"SELECT drop_graph('{graph_name}', true)")
            await session.commit()
    finally:
        await engine.dispose()

    expected = sorted(
        f'["{edge.source_node_id}", "{edge.target_node_id}", "calls"]' for edge in edges
    )
    assert created == expected
