from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator

import pytest
from contextmine_core.database import Base
from contextmine_core.models import (
    Collection,
    CollectionVisibility,
    MetricSnapshot,
    TwinFinding,
    TwinNode,
    TwinScenario,
    User,
)
from contextmine_core.twin.evolution import (
    get_fitness_functions_payload,
    get_investment_utilization_payload,
    get_temporal_coupling_payload,
    replace_evolution_snapshots,
)
from sqlalchemy import select
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


async def _seed_scenario(session: AsyncSession) -> TwinScenario:
    user = User(
        id=uuid.uuid4(),
        github_user_id=221,
        github_login="evolution",
    )
    collection = Collection(
        id=uuid.uuid4(),
        slug="evolution",
        name="Evolution",
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
    return scenario


@pytest.mark.anyio
async def test_replace_snapshots_and_temporal_payload(test_session: AsyncSession) -> None:
    scenario = await _seed_scenario(test_session)

    stats = await replace_evolution_snapshots(
        test_session,
        scenario_id=scenario.id,
        ownership_rows=[
            {
                "node_natural_key": "file:src/a.py",
                "author_key": "alice",
                "author_label": "Alice",
                "additions": 10,
                "deletions": 1,
                "touches": 2,
                "ownership_share": 0.9,
                "window_days": 365,
            }
        ],
        coupling_rows=[
            {
                "entity_level": "component",
                "source_key": "component:core/a",
                "target_key": "component:core/b",
                "co_change_count": 4,
                "source_change_count": 6,
                "target_change_count": 5,
                "ratio_source_to_target": 0.66,
                "ratio_target_to_source": 0.8,
                "jaccard": 0.57,
                "cross_boundary": True,
                "window_days": 365,
            }
        ],
    )
    await test_session.flush()

    assert stats["ownership_rows"] == 1
    assert stats["coupling_rows"] == 1

    payload = await get_temporal_coupling_payload(
        test_session,
        scenario_id=scenario.id,
        entity_level="component",
        window_days=365,
        min_jaccard=0.2,
        max_edges=50,
    )

    assert payload["status"] == "ready"
    assert payload["summary"]["edges"] == 1
    assert payload["graph"]["edges"][0]["cross_boundary"] is True


@pytest.mark.anyio
async def test_investment_payload_suppresses_utilization_when_coverage_sparse(
    test_session: AsyncSession,
) -> None:
    scenario = await _seed_scenario(test_session)

    file_a = TwinNode(
        id=uuid.uuid4(),
        scenario_id=scenario.id,
        natural_key="file:services/billing/a.py",
        kind="file",
        name="a.py",
        meta={"file_path": "services/billing/a.py"},
    )
    file_b = TwinNode(
        id=uuid.uuid4(),
        scenario_id=scenario.id,
        natural_key="file:services/billing/b.py",
        kind="file",
        name="b.py",
        meta={"file_path": "services/billing/b.py"},
    )
    session_rows = [file_a, file_b]
    test_session.add_all(session_rows)

    metrics = [
        MetricSnapshot(
            id=uuid.uuid4(),
            scenario_id=scenario.id,
            node_natural_key=file_a.natural_key,
            loc=100,
            complexity=7.0,
            coupling=2.0,
            coverage=None,
            change_frequency=5.0,
            meta={"churn": 50.0},
        ),
        MetricSnapshot(
            id=uuid.uuid4(),
            scenario_id=scenario.id,
            node_natural_key=file_b.natural_key,
            loc=80,
            complexity=3.0,
            coupling=1.0,
            coverage=None,
            change_frequency=2.0,
            meta={"churn": 10.0},
        ),
    ]
    test_session.add_all(metrics)
    await test_session.flush()

    payload = await get_investment_utilization_payload(
        test_session,
        scenario_id=scenario.id,
        entity_level="component",
        window_days=365,
    )

    assert payload["status"] == "ready"
    assert payload["summary"]["utilization_available"] is False
    assert payload["items"][0]["utilization_score"] is None


@pytest.mark.anyio
async def test_fitness_payload_groups_rules(test_session: AsyncSession) -> None:
    scenario = await _seed_scenario(test_session)

    test_session.add_all(
        [
            TwinFinding(
                id=uuid.uuid4(),
                scenario_id=scenario.id,
                source_version_id=None,
                fingerprint="fp-1",
                finding_type="fitness.ff003_single_owner_hotspot",
                severity="high",
                confidence="high",
                status="open",
                filename="src/a.py",
                line_number=1,
                message="Hotspot",
                flow_data={},
                meta={"rule_id": "FF003_single_owner_hotspot", "subject": "file:src/a.py"},
            ),
            TwinFinding(
                id=uuid.uuid4(),
                scenario_id=scenario.id,
                source_version_id=None,
                fingerprint="fp-2",
                finding_type="fitness.ff004_cross_boundary_strong_coupling",
                severity="medium",
                confidence="high",
                status="resolved",
                filename="src/b.py",
                line_number=1,
                message="Coupling",
                flow_data={},
                meta={
                    "rule_id": "FF004_cross_boundary_strong_coupling",
                    "subject": "component:x <-> component:y",
                },
            ),
        ]
    )
    await test_session.flush()

    payload_open = await get_fitness_functions_payload(
        test_session,
        scenario_id=scenario.id,
        window_days=365,
        include_resolved=False,
    )
    assert payload_open["status"] == "ready"
    assert payload_open["summary"]["violations"] == 1

    payload_all = await get_fitness_functions_payload(
        test_session,
        scenario_id=scenario.id,
        window_days=365,
        include_resolved=True,
    )
    assert payload_all["summary"]["violations"] == 2
    assert len(payload_all["rules"]) == 2

    db_rows = (
        (
            await test_session.execute(
                select(TwinFinding).where(TwinFinding.scenario_id == scenario.id)
            )
        )
        .scalars()
        .all()
    )
    assert len(db_rows) == 2
