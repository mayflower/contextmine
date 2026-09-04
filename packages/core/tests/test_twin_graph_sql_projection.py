"""Database-side graph projections must match the in-memory builders exactly.

``get_full_scenario_graph`` groups the architecture and file projections in SQL
and pushes kind filters down for the symbol projection. These tests seed a
scenario through the ORM, run every projection through the SQL path and
compare it with ``_apply_graph_projection`` over the naively loaded rows, which
is the reference the cockpit views were built against.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from contextmine_core.database import Base
from contextmine_core.models import (
    Collection,
    CollectionVisibility,
    TwinEdge,
    TwinLayer,
    TwinNode,
    TwinNodeLayer,
    TwinScenario,
    User,
)
from contextmine_core.twin.projections import GraphProjection
from contextmine_core.twin.service import (
    _apply_graph_projection,
    _load_graph_rows,
    get_full_scenario_graph,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def test_session() -> AsyncGenerator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session_maker = async_sessionmaker(engine, expire_on_commit=False)
    async with async_session_maker() as session:
        yield session

    await engine.dispose()


_FILES = [
    # (path, explicit architecture meta, layers)
    ("services/billing/api/invoice.py", None, [TwinLayer.CODE_CONTROLFLOW]),
    ("services/billing/api/tax.py", None, [TwinLayer.CODE_CONTROLFLOW]),
    (
        "services/payments/core/charge.py",
        {"domain": "payments", "container": "core", "component": "charging"},
        [TwinLayer.CODE_CONTROLFLOW, TwinLayer.PORTFOLIO_SYSTEM],
    ),
    ("apps/web/src/App.tsx", None, [TwinLayer.CODE_CONTROLFLOW]),
    ("README.md", None, []),  # too shallow for a heuristic group
    ("  ", None, []),  # blank path: never projected
]


async def _seed_graph(session: AsyncSession) -> uuid.UUID:
    """A small scenario exercising every branch the SQL path has to mirror."""
    user = User(id=uuid.uuid4(), github_user_id=1, github_login="architect")
    collection = Collection(
        id=uuid.uuid4(),
        slug="sql-projection",
        name="SQL projection",
        visibility=CollectionVisibility.PRIVATE,
        owner_user_id=user.id,
    )
    scenario = TwinScenario(
        id=uuid.uuid4(),
        collection_id=collection.id,
        name="AS-IS",
        is_as_is=True,
        version=1,
        meta={},
    )
    session.add_all([user, collection, scenario])
    await session.flush()

    nodes: list[TwinNode] = []
    layers: list[TwinNodeLayer] = []
    file_by_path: dict[str, TwinNode] = {}
    symbols_by_path: dict[str, list[TwinNode]] = {}

    def add(node: TwinNode, node_layers: list[TwinLayer]) -> TwinNode:
        nodes.append(node)
        layers.extend(TwinNodeLayer(node_id=node.id, layer=layer) for layer in node_layers)
        return node

    for index, (path, architecture, file_layers) in enumerate(_FILES):
        meta: dict[str, Any] = {"file_path": path, "loc": 10 * (index + 1)}
        if architecture:
            meta["architecture"] = architecture
        file_by_path[path] = add(
            TwinNode(
                id=uuid.uuid4(),
                scenario_id=scenario.id,
                natural_key=f"file:{path}",
                kind="File" if index % 2 else "file",  # kinds compare case-insensitively
                name=path.rsplit("/", 1)[-1],
                meta=meta,
            ),
            file_layers,
        )
        symbols_by_path[path] = []
        for symbol_index in range(3 if index < 3 else 1):
            symbol_meta: dict[str, Any] = {"file_path": path, "symbol_kind": "function"}
            if architecture:
                symbol_meta["architecture"] = architecture
            symbols_by_path[path].append(
                add(
                    TwinNode(
                        id=uuid.uuid4(),
                        scenario_id=scenario.id,
                        natural_key=f"symbol:{path}:{symbol_index}",
                        kind="symbol" if symbol_index else "class",
                        name=f"sym_{index}_{symbol_index}",
                        meta=symbol_meta,
                    ),
                    file_layers,
                )
            )

    # A node whose meta.file_path is missing and a job pointing at a file.
    detached = add(
        TwinNode(
            id=uuid.uuid4(),
            scenario_id=scenario.id,
            natural_key="symbol:detached",
            kind="symbol",
            name="detached",
            meta={},
        ),
        [TwinLayer.CODE_CONTROLFLOW],
    )
    job = add(
        TwinNode(
            id=uuid.uuid4(),
            scenario_id=scenario.id,
            natural_key="job:nightly",
            kind="job",
            name="nightly",
            meta={"file_path": "services/billing/api/invoice.py"},
        ),
        [TwinLayer.PORTFOLIO_SYSTEM],
    )
    session.add_all(nodes)
    await session.flush()
    session.add_all(layers)

    billing = symbols_by_path["services/billing/api/invoice.py"]
    tax = symbols_by_path["services/billing/api/tax.py"]
    payments = symbols_by_path["services/payments/core/charge.py"]
    web = symbols_by_path["apps/web/src/App.tsx"]
    readme = symbols_by_path["README.md"]

    edges = [
        # file_defines_symbol is ignored by the file projection but folds away
        # in the architecture projection (same group on both ends).
        *(
            TwinEdge(
                id=uuid.uuid4(),
                scenario_id=scenario.id,
                source_node_id=file_by_path[path].id,
                target_node_id=symbol.id,
                kind="file_defines_symbol",
                meta={},
            )
            for path, symbols in symbols_by_path.items()
            for symbol in symbols
        ),
        # Cross-container calls, several per pair so counts matter.
        *(
            TwinEdge(
                id=uuid.uuid4(),
                scenario_id=scenario.id,
                source_node_id=src.id,
                target_node_id=dst.id,
                kind="symbol_calls_symbol",
                meta={"weight": 2},
            )
            for src in billing
            for dst in payments
        ),
        TwinEdge(
            id=uuid.uuid4(),
            scenario_id=scenario.id,
            source_node_id=billing[1].id,
            target_node_id=payments[1].id,
            kind="symbol_references_symbol",
            meta={},
        ),
        # Same container, different files: file edge yes, arch edge no.
        TwinEdge(
            id=uuid.uuid4(),
            scenario_id=scenario.id,
            source_node_id=tax[0].id,
            target_node_id=billing[1].id,
            kind="symbol_imports_symbol",
            meta={},
        ),
        # Same file: dropped by both projections.
        TwinEdge(
            id=uuid.uuid4(),
            scenario_id=scenario.id,
            source_node_id=billing[0].id,
            target_node_id=billing[2].id,
            kind="symbol_contains_symbol",
            meta={},
        ),
        # Upper-case kind: edge kind filters compare lower-cased.
        TwinEdge(
            id=uuid.uuid4(),
            scenario_id=scenario.id,
            source_node_id=web[0].id,
            target_node_id=payments[1].id,
            kind="Symbol_Calls_Symbol",
            meta={},
        ),
        # Endpoint without a path and endpoint outside any group.
        TwinEdge(
            id=uuid.uuid4(),
            scenario_id=scenario.id,
            source_node_id=detached.id,
            target_node_id=payments[0].id,
            kind="symbol_calls_symbol",
            meta={},
        ),
        TwinEdge(
            id=uuid.uuid4(),
            scenario_id=scenario.id,
            source_node_id=readme[0].id,
            target_node_id=billing[0].id,
            kind="symbol_references_symbol",
            meta={},
        ),
        # Job → file edge crosses containers through meta.file_path.
        TwinEdge(
            id=uuid.uuid4(),
            scenario_id=scenario.id,
            source_node_id=job.id,
            target_node_id=file_by_path["services/payments/core/charge.py"].id,
            kind="job_touches_file",
            meta={},
        ),
    ]
    session.add_all(edges)
    await session.commit()
    return scenario.id


def _canonical(graph: dict[str, Any]) -> dict[str, Any]:
    """Order-insensitive view of a projection result."""
    return {
        **{key: value for key, value in graph.items() if key not in {"nodes", "edges"}},
        "nodes": sorted(graph["nodes"], key=lambda node: node["id"]),
        "edges": sorted(
            graph["edges"],
            key=lambda edge: (edge["source_node_id"], edge["target_node_id"], edge["id"]),
        ),
    }


def _norm(kinds: set[str] | None) -> set[str] | None:
    return {kind.lower() for kind in kinds} if kinds else None


_CASES: list[dict[str, Any]] = [
    {"projection": GraphProjection.ARCHITECTURE, "entity_level": "container"},
    {"projection": GraphProjection.ARCHITECTURE, "entity_level": "domain"},
    {"projection": GraphProjection.ARCHITECTURE, "entity_level": "component"},
    {"projection": GraphProjection.ARCHITECTURE, "entity_level": "bogus"},
    {"projection": GraphProjection.ARCHITECTURE, "include_kinds": {"FILE", "job"}},
    {"projection": GraphProjection.ARCHITECTURE, "exclude_kinds": {"symbol"}},
    {"projection": GraphProjection.ARCHITECTURE, "layer": TwinLayer.CODE_CONTROLFLOW},
    {"projection": GraphProjection.ARCHITECTURE, "layer": TwinLayer.PORTFOLIO_SYSTEM},
    {"projection": GraphProjection.CODE_FILE},
    {"projection": GraphProjection.CODE_FILE, "exclude_kinds": {"class"}},
    {"projection": GraphProjection.CODE_FILE, "include_edge_kinds": {"symbol_calls_symbol"}},
    {"projection": GraphProjection.CODE_FILE, "layer": TwinLayer.CODE_CONTROLFLOW},
    {"projection": GraphProjection.CODE_FILE, "layer": TwinLayer.PORTFOLIO_SYSTEM},
    {"projection": GraphProjection.CODE_SYMBOL},
    {"projection": GraphProjection.CODE_SYMBOL, "include_kinds": {"Symbol"}},
    {"projection": GraphProjection.CODE_SYMBOL, "exclude_kinds": {"class", "job"}},
    {"projection": GraphProjection.CODE_SYMBOL, "include_edge_kinds": {"symbol_calls_symbol"}},
    {"projection": GraphProjection.CODE_SYMBOL, "layer": TwinLayer.CODE_CONTROLFLOW},
    {
        "projection": GraphProjection.CODE_SYMBOL,
        "layer": TwinLayer.PORTFOLIO_SYSTEM,
        "exclude_kinds": {"job"},
        "include_edge_kinds": {"symbol_calls_symbol", "job_touches_file"},
    },
]


@pytest.mark.anyio
@pytest.mark.parametrize("case", _CASES, ids=lambda case: str(case))
async def test_sql_projection_matches_in_memory_reference(
    test_session: AsyncSession, case: dict[str, Any]
) -> None:
    scenario_id = await _seed_graph(test_session)
    layer = case.get("layer")

    nodes, edges = await _load_graph_rows(test_session, scenario_id, layer)
    reference = _apply_graph_projection(
        nodes,
        edges,
        case["projection"],
        case.get("entity_level"),
        _norm(case.get("include_kinds")),
        _norm(case.get("exclude_kinds")),
        _norm(case.get("include_edge_kinds")),
    )

    actual = await get_full_scenario_graph(
        test_session,
        scenario_id,
        layer,
        projection=case["projection"],
        entity_level=case.get("entity_level"),
        include_kinds=case.get("include_kinds"),
        exclude_kinds=case.get("exclude_kinds"),
        include_edge_kinds=case.get("include_edge_kinds"),
    )

    assert _canonical(actual) == _canonical(reference)
    # Guard against a vacuous comparison: the seed must produce a real graph.
    if not case.get("include_kinds") and not case.get("exclude_kinds"):
        assert actual["nodes"]


@pytest.mark.anyio
async def test_seed_covers_the_interesting_branches(test_session: AsyncSession) -> None:
    """The reference graph must contain folded edges, counts and mixed grouping."""
    scenario_id = await _seed_graph(test_session)

    architecture = await get_full_scenario_graph(
        test_session, scenario_id, None, projection=GraphProjection.ARCHITECTURE
    )
    assert architecture["grouping_strategy"] == "mixed"
    by_name = {node["name"]: node for node in architecture["nodes"]}
    # 2 files + 4 symbols + 1 job; the two "class" nodes are hidden by default.
    assert by_name["api"]["meta"]["member_count"] == 7
    assert by_name["core"]["meta"]["provenance"] == "explicit"
    weights = {
        (edge["source_node_id"], edge["target_node_id"]): edge["meta"]
        for edge in architecture["edges"]
    }
    api_to_core = weights[(by_name["api"]["id"], by_name["core"]["id"])]
    # 2x2 calls between visible symbols + 1 reference + 1 job edge.
    assert api_to_core["raw_edge_count"] == 6
    assert api_to_core["sample_edge_kinds"] == [
        "job_touches_file",
        "symbol_calls_symbol",
        "symbol_references_symbol",
    ]

    files = await get_full_scenario_graph(
        test_session, scenario_id, None, projection=GraphProjection.CODE_FILE
    )
    counts = {node["name"]: node["meta"]["symbol_count"] for node in files["nodes"]}
    assert counts["invoice.py"] == 4  # 3 symbols + the job that points at the file
    assert counts["README.md"] == 1
    assert "  " not in counts  # the blank path never becomes a file node
    file_edges = {
        (edge["source_node_id"], edge["target_node_id"]): edge["meta"]["weight"]
        for edge in files["edges"]
    }
    ids = {node["name"]: node["id"] for node in files["nodes"]}
    # 9 calls + 1 reference + the job edge, which resolves to invoice.py via meta.
    assert file_edges[(ids["invoice.py"], ids["charge.py"])] == 11
    assert file_edges[(ids["tax.py"], ids["invoice.py"])] == 1
