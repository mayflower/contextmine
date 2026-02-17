from __future__ import annotations

import json
import uuid
from collections.abc import AsyncGenerator

import pytest
from contextmine_core.database import Base
from contextmine_core.exports.codecharta import export_codecharta_json
from contextmine_core.models import (
    Collection,
    CollectionVisibility,
    MetricSnapshot,
    TwinEdge,
    TwinNode,
    TwinScenario,
    User,
)
from contextmine_core.twin import GraphProjection
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def test_session() -> AsyncGenerator[AsyncSession, None]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session_maker() as session:
        yield session

    await engine.dispose()


def _collect_leaf_attributes(tree: dict) -> dict[str, dict]:
    leaves: dict[str, dict] = {}

    def walk(node: dict, parents: list[str]) -> None:
        current_path = [*parents, str(node["name"])]
        if node.get("type") == "File":
            leaves["/" + "/".join(current_path)] = node.get("attributes", {})
            return
        for child in node.get("children", []):
            walk(child, current_path)

    walk(tree, [])
    return leaves


async def _seed_scenario(session: AsyncSession, scenario_name: str = "Scenario") -> TwinScenario:
    user = User(
        id=uuid.uuid4(),
        github_user_id=12345,
        github_login="architect",
    )
    collection = Collection(
        id=uuid.uuid4(),
        slug="test-collection",
        name="Test Collection",
        visibility=CollectionVisibility.PRIVATE,
        owner_user_id=user.id,
    )
    scenario = TwinScenario(
        id=uuid.uuid4(),
        collection_id=collection.id,
        name=scenario_name,
        is_as_is=True,
        version=1,
        meta={},
    )
    session.add_all([user, collection, scenario])
    await session.flush()
    return scenario


@pytest.mark.anyio
async def test_codecharta_file_projection_schema_and_edges(test_session: AsyncSession) -> None:
    scenario = await _seed_scenario(test_session, scenario_name="File Projection")

    node_a = TwinNode(
        id=uuid.uuid4(),
        scenario_id=scenario.id,
        natural_key="file:apps/api/main.py",
        kind="file",
        name="main.py",
        meta={"file_path": "apps/api/main.py"},
    )
    node_b = TwinNode(
        id=uuid.uuid4(),
        scenario_id=scenario.id,
        natural_key="file:apps/web/App.tsx",
        kind="file",
        name="App.tsx",
        meta={"file_path": "apps/web/App.tsx"},
    )
    edge = TwinEdge(
        id=uuid.uuid4(),
        scenario_id=scenario.id,
        source_node_id=node_a.id,
        target_node_id=node_b.id,
        kind="file_depends_on_file",
        meta={},
    )
    test_session.add_all(
        [
            node_a,
            node_b,
            edge,
            MetricSnapshot(
                id=uuid.uuid4(),
                scenario_id=scenario.id,
                node_natural_key=node_a.natural_key,
                loc=20,
                symbol_count=2,
                coupling=2.0,
                coverage=70.0,
                complexity=4.0,
                change_frequency=1.0,
                meta={},
            ),
            MetricSnapshot(
                id=uuid.uuid4(),
                scenario_id=scenario.id,
                node_natural_key=node_b.natural_key,
                loc=40,
                symbol_count=6,
                coupling=5.0,
                coverage=80.0,
                complexity=9.0,
                change_frequency=3.0,
                meta={},
            ),
        ]
    )
    await test_session.commit()

    payload = json.loads(
        await export_codecharta_json(
            test_session,
            scenario.id,
            projection=GraphProjection.CODE_FILE,
            entity_level="file",
        )
    )

    assert payload["projectName"] == "File Projection"
    assert payload["apiVersion"] == "1.5"
    assert isinstance(payload["fileChecksum"], str)
    assert payload["nodes"][0]["type"] == "Folder"

    leaves = _collect_leaf_attributes(payload["nodes"][0])
    assert leaves["/root/apps/api/main.py"]["loc"] == 20
    assert leaves["/root/apps/web/App.tsx"]["symbol_count"] == 6
    assert leaves["/root/apps/web/App.tsx"]["complexity"] == pytest.approx(9.0)

    assert len(payload["edges"]) == 1
    edge_payload = payload["edges"][0]
    assert edge_payload["fromNodeName"] == "/root/apps/api/main.py"
    assert edge_payload["toNodeName"] == "/root/apps/web/App.tsx"
    assert edge_payload["attributes"]["dependency_weight"] == pytest.approx(1.0)


@pytest.mark.anyio
async def test_codecharta_architecture_projection_weighted_aggregation(
    test_session: AsyncSession,
) -> None:
    scenario = await _seed_scenario(test_session, scenario_name="Architecture Projection")

    node_a = TwinNode(
        id=uuid.uuid4(),
        scenario_id=scenario.id,
        natural_key="file:services/billing/api/invoice.py",
        kind="file",
        name="invoice.py",
        meta={"file_path": "services/billing/api/invoice.py"},
    )
    node_b = TwinNode(
        id=uuid.uuid4(),
        scenario_id=scenario.id,
        natural_key="file:services/billing/api/payments.py",
        kind="file",
        name="payments.py",
        meta={"file_path": "services/billing/api/payments.py"},
    )
    node_c = TwinNode(
        id=uuid.uuid4(),
        scenario_id=scenario.id,
        natural_key="file:services/billing/worker/jobs.py",
        kind="file",
        name="jobs.py",
        meta={"file_path": "services/billing/worker/jobs.py"},
    )
    edge = TwinEdge(
        id=uuid.uuid4(),
        scenario_id=scenario.id,
        source_node_id=node_b.id,
        target_node_id=node_c.id,
        kind="file_depends_on_file",
        meta={},
    )
    test_session.add_all(
        [
            node_a,
            node_b,
            node_c,
            edge,
            MetricSnapshot(
                id=uuid.uuid4(),
                scenario_id=scenario.id,
                node_natural_key=node_a.natural_key,
                loc=10,
                symbol_count=2,
                coupling=1.0,
                coverage=50.0,
                complexity=2.0,
                change_frequency=1.0,
                meta={},
            ),
            MetricSnapshot(
                id=uuid.uuid4(),
                scenario_id=scenario.id,
                node_natural_key=node_b.natural_key,
                loc=30,
                symbol_count=4,
                coupling=5.0,
                coverage=90.0,
                complexity=6.0,
                change_frequency=3.0,
                meta={},
            ),
            MetricSnapshot(
                id=uuid.uuid4(),
                scenario_id=scenario.id,
                node_natural_key=node_c.natural_key,
                loc=20,
                symbol_count=5,
                coupling=2.0,
                coverage=60.0,
                complexity=7.0,
                change_frequency=4.0,
                meta={},
            ),
        ]
    )
    await test_session.commit()

    payload = json.loads(
        await export_codecharta_json(
            test_session,
            scenario.id,
            projection=GraphProjection.ARCHITECTURE,
            entity_level="container",
        )
    )

    leaves = _collect_leaf_attributes(payload["nodes"][0])
    api_container = leaves["/root/billing/api"]
    worker_container = leaves["/root/billing/worker"]

    assert api_container["loc"] == 40
    assert api_container["symbol_count"] == 6
    assert api_container["coverage"] == pytest.approx(80.0)
    assert api_container["coupling"] == pytest.approx(4.0)
    assert api_container["complexity"] == pytest.approx(5.0)
    assert api_container["change_frequency"] == pytest.approx(2.5)
    assert worker_container["loc"] == 20

    assert len(payload["edges"]) == 1
    edge_payload = payload["edges"][0]
    assert edge_payload["fromNodeName"] == "/root/billing/api"
    assert edge_payload["toNodeName"] == "/root/billing/worker"
    assert edge_payload["attributes"]["dependency_weight"] == pytest.approx(1.0)
