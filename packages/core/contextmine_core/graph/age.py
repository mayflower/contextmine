"""Apache AGE adapter for twin scenario graph queries.

Apache AGE requires graph names and Cypher queries as literal SQL strings,
not bind parameters. The helper ``_age_cypher_sql`` safely inlines both
using the deterministic graph name (UUID-derived, alphanumeric + underscore)
and $$-delimited Cypher text.
"""

from __future__ import annotations

import re
from uuid import UUID

from contextmine_core.models import TwinEdge, TwinNode
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

_MUTATING_PATTERNS = re.compile(
    r"\b(create|merge|set|delete|remove|drop|alter|grant|revoke|copy|call)\b",
    flags=re.IGNORECASE,
)

# Graph names are strictly alphanumeric + underscore (from UUID hex)
_SAFE_NAME = re.compile(r"^[a-z0-9_]+$")


def scenario_graph_name(scenario_id: UUID) -> str:
    """Deterministic AGE graph name for a scenario."""
    return f"twin_{str(scenario_id).replace('-', '_')}"


def _validate_graph_name(name: str) -> None:
    """Ensure graph name is safe for SQL interpolation."""
    if not _SAFE_NAME.match(name):
        raise ValueError(f"Invalid graph name: {name!r}")


def _age_cypher_sql(graph_name: str, cypher: str, result_cols: str = "v agtype") -> str:
    """Build a literal SQL string for AGE cypher() calls.

    AGE does not support bind parameters for graph names or queries.
    Graph names are UUID-derived (safe), Cypher is $$-delimited.
    """
    _validate_graph_name(graph_name)
    return f"SELECT * FROM cypher('{graph_name}', $$ {cypher} $$) AS ({result_cols})"


def ensure_read_only_cypher(query: str) -> None:
    """Raise if a Cypher query appears mutating."""
    if _MUTATING_PATTERNS.search(query):
        raise ValueError("Only read-only Cypher queries are allowed")


async def ensure_age_ready(session: AsyncSession) -> None:
    """Load AGE and set search path for cypher() calls."""
    conn = await session.connection()
    await conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS age")
    await conn.exec_driver_sql("LOAD 'age'")
    await conn.exec_driver_sql('SET search_path = ag_catalog, "$user", public')


async def sync_scenario_to_age(session: AsyncSession, scenario_id: UUID) -> None:
    """Replace AGE graph contents with the current scenario graph."""
    await ensure_age_ready(session)

    graph_name = scenario_graph_name(scenario_id)
    _validate_graph_name(graph_name)

    conn = await session.connection()

    # Create graph if it doesn't exist
    await conn.exec_driver_sql(
        f"""
        SELECT create_graph('{graph_name}')
        WHERE NOT EXISTS (
            SELECT 1 FROM ag_catalog.ag_graph WHERE name = '{graph_name}'
        )
        """
    )

    # Clear existing graph data
    await conn.exec_driver_sql(_age_cypher_sql(graph_name, "MATCH (n) DETACH DELETE n"))

    # Load nodes
    nodes = (
        (await session.execute(select(TwinNode).where(TwinNode.scenario_id == scenario_id)))
        .scalars()
        .all()
    )
    for node in nodes:
        cypher = (
            "CREATE (n:Node {"
            f"id: '{_esc(str(node.id))}', "
            f"natural_key: '{_esc(node.natural_key)}', "
            f"kind: '{_esc(node.kind)}', "
            f"name: '{_esc(node.name)}'"
            "})"
        )
        await conn.exec_driver_sql(_age_cypher_sql(graph_name, cypher))

    # Load edges
    edges = (
        (await session.execute(select(TwinEdge).where(TwinEdge.scenario_id == scenario_id)))
        .scalars()
        .all()
    )
    for edge in edges:
        cypher = (
            f"MATCH (s:Node {{id: '{_esc(str(edge.source_node_id))}'}}), "
            f"(t:Node {{id: '{_esc(str(edge.target_node_id))}'}}) "
            f"CREATE (s)-[r:REL {{kind: '{_esc(edge.kind)}'}}]->(t)"
        )
        await conn.exec_driver_sql(_age_cypher_sql(graph_name, cypher))


async def run_read_only_cypher(
    session: AsyncSession,
    scenario_id: UUID,
    query: str,
) -> list[str]:
    """Execute a read-only cypher query and return agtype rows as text."""
    ensure_read_only_cypher(query)
    await ensure_age_ready(session)

    graph_name = scenario_graph_name(scenario_id)
    _validate_graph_name(graph_name)
    sql = f"SELECT result::text FROM cypher('{graph_name}', $$ {query} $$) AS (result agtype)"
    conn = await session.connection()
    result = await conn.exec_driver_sql(sql)
    return [row[0] for row in result.all()]


def _esc(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")
