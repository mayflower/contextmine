"""Apache AGE adapter for twin scenario graph queries.

Apache AGE requires graph names and Cypher queries as literal SQL strings,
not bind parameters. The helper ``_age_cypher_sql`` safely inlines both
using the deterministic graph name (UUID-derived, alphanumeric + underscore)
and $$-delimited Cypher text.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from uuid import UUID

from contextmine_core.models import TwinAgeGraph, TwinEdge, TwinNode
from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

_MUTATING_PATTERNS = re.compile(
    r"\b(create|merge|set|delete|remove|drop|alter|grant|revoke|copy|call)\b",
    flags=re.IGNORECASE,
)

# Graph names are strictly alphanumeric + underscore (from UUID hex)
_SAFE_NAME = re.compile(r"^[a-z0-9_]+$")
_NODE_BATCH_SIZE = 800
_EDGE_BATCH_SIZE = 600


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

    # Load nodes in batches to avoid one SQL round-trip per vertex.
    nodes = (
        (await session.execute(select(TwinNode).where(TwinNode.scenario_id == scenario_id)))
        .scalars()
        .all()
    )
    for batch in _chunked(nodes, _NODE_BATCH_SIZE):
        await conn.exec_driver_sql(_age_cypher_sql(graph_name, _node_batch_cypher(batch)))

    # Load edges in batches to avoid one SQL round-trip per relationship.
    edges = (
        (await session.execute(select(TwinEdge).where(TwinEdge.scenario_id == scenario_id)))
        .scalars()
        .all()
    )
    for batch in _chunked(edges, _EDGE_BATCH_SIZE):
        await conn.exec_driver_sql(_age_cypher_sql(graph_name, _edge_batch_cypher(batch)))


def _node_batch_cypher(nodes: Iterable[TwinNode]) -> str:
    payload = ",".join(
        (
            "{"
            f"id: '{_esc(str(node.id))}', "
            f"natural_key: '{_esc(node.natural_key)}', "
            f"kind: '{_esc(node.kind)}', "
            f"name: '{_esc(node.name)}'"
            "}"
        )
        for node in nodes
    )
    return (
        f"UNWIND [{payload}] AS row "
        "CREATE (n:Node {"
        "id: row.id, "
        "natural_key: row.natural_key, "
        "kind: row.kind, "
        "name: row.name"
        "})"
    )


def _edge_batch_cypher(edges: Iterable[TwinEdge]) -> str:
    payload = ",".join(
        (
            "{"
            f"source_id: '{_esc(str(edge.source_node_id))}', "
            f"target_id: '{_esc(str(edge.target_node_id))}', "
            f"kind: '{_esc(edge.kind)}'"
            "}"
        )
        for edge in edges
    )
    # Match the endpoints in WHERE, not as a property map: AGE 1.5 plans
    # `(s:Node {id: row.source_id})` as a nested loop with a containment filter
    # over all nodes per row (51 s for 600 edges on 36k nodes), while the WHERE
    # form becomes a hash join (0.7 s), without needing an index.
    return (
        f"UNWIND [{payload}] AS row "
        "MATCH (s:Node), (t:Node) "
        "WHERE s.id = row.source_id AND t.id = row.target_id "
        "CREATE (s)-[r:REL {kind: row.kind}]->(t)"
    )


# Hash of exactly the twin fields sync_scenario_to_age copies into AGE: node id,
# natural key, kind and name, and edge endpoints and kind. Other twin changes
# (meta, layers, metrics) do not affect the AGE graph and do not count.
_AGE_FINGERPRINT_SQL = text(
    """
    SELECT md5(
        coalesce((
            SELECT string_agg(
                id::text || chr(31) || natural_key || chr(31) || kind || chr(31) || name,
                chr(30) ORDER BY id
            )
            FROM twin_nodes WHERE scenario_id = :scenario_id
        ), '')
        || chr(29) ||
        coalesce((
            SELECT string_agg(
                source_node_id::text || chr(31) || target_node_id::text || chr(31) || kind,
                chr(30) ORDER BY source_node_id, target_node_id, kind
            )
            FROM twin_edges WHERE scenario_id = :scenario_id
        ), '')
    )
    """
)


async def ensure_scenario_age_graph(session: AsyncSession, scenario_id: UUID) -> bool:
    """Make the scenario's AGE graph match its twin; rebuild only if it changed.

    Returns True when the graph was rebuilt. The caller must commit, which
    persists the graph together with the recorded fingerprint, so later
    queries can use it without rebuilding.
    """
    await ensure_age_ready(session)
    graph_name = scenario_graph_name(scenario_id)
    _validate_graph_name(graph_name)
    conn = await session.connection()
    # Serializes rebuilds of one scenario between sync workers and API requests.
    await conn.exec_driver_sql(f"SELECT pg_advisory_xact_lock(hashtext('{graph_name}'))")

    fingerprint = (
        await session.execute(_AGE_FINGERPRINT_SQL, {"scenario_id": scenario_id})
    ).scalar_one()
    stored = (
        await session.execute(
            select(TwinAgeGraph.fingerprint).where(TwinAgeGraph.scenario_id == scenario_id)
        )
    ).scalar_one_or_none()
    graph_exists = (
        await conn.exec_driver_sql(f"SELECT 1 FROM ag_catalog.ag_graph WHERE name = '{graph_name}'")
    ).first() is not None
    if graph_exists and stored == fingerprint:
        return False

    await sync_scenario_to_age(session, scenario_id)
    stmt = pg_insert(TwinAgeGraph).values(scenario_id=scenario_id, fingerprint=fingerprint)
    await session.execute(
        stmt.on_conflict_do_update(
            index_elements=[TwinAgeGraph.scenario_id],
            set_={"fingerprint": stmt.excluded.fingerprint, "built_at": func.now()},
        )
    )
    return True


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


def _chunked[T](values: list[T], size: int) -> Iterable[list[T]]:
    for i in range(0, len(values), size):
        yield values[i : i + size]
