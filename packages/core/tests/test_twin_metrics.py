from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator

import pytest
from contextmine_core.database import Base
from contextmine_core.models import (
    Collection,
    CollectionVisibility,
    MetricSnapshot,
    TwinNode,
    TwinScenario,
    User,
)
from contextmine_core.twin.service import apply_file_metrics_to_scenario, refresh_metric_snapshots
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


async def _seed_scenario_with_file_node(session: AsyncSession) -> tuple[TwinScenario, TwinNode]:
    user = User(
        id=uuid.uuid4(),
        github_user_id=12345,
        github_login="architect",
    )
    collection = Collection(
        id=uuid.uuid4(),
        slug="twin-metrics",
        name="Twin Metrics",
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
    file_node = TwinNode(
        id=uuid.uuid4(),
        scenario_id=scenario.id,
        natural_key="file:src/main.py",
        kind="file",
        name="main.py",
        meta={"file_path": "src/main.py"},
    )
    session.add_all([user, collection, scenario, file_node])
    await session.flush()
    return scenario, file_node


@pytest.mark.anyio
async def test_apply_file_metrics_to_scenario_sets_change_frequency_and_churn(
    test_session: AsyncSession,
) -> None:
    scenario, file_node = await _seed_scenario_with_file_node(test_session)

    updated = await apply_file_metrics_to_scenario(
        test_session,
        scenario.id,
        [
            {
                "file_path": "src/main.py",
                "language": "python",
                "loc": 42,
                "complexity": 8.0,
                "coupling_in": 1,
                "coupling_out": 2,
                "coupling": 3.0,
                "change_frequency": 5.0,
                "churn": 18.0,
                "sources": {},
            }
        ],
    )
    await test_session.flush()

    assert updated == 1
    assert file_node.meta["change_frequency"] == pytest.approx(5.0)
    assert file_node.meta["churn"] == pytest.approx(18.0)
    assert bool(file_node.meta["metrics_structural_ready"]) is True
    assert bool(file_node.meta["coverage_ready"]) is False


@pytest.mark.anyio
async def test_refresh_metric_snapshots_keeps_churn_in_snapshot_meta(
    test_session: AsyncSession,
) -> None:
    scenario, file_node = await _seed_scenario_with_file_node(test_session)
    file_node.meta = {
        "file_path": "src/main.py",
        "metrics_structural_ready": True,
        "coverage_ready": True,
        "loc": 21,
        "symbol_count": 4,
        "coupling": 2.0,
        "coverage": 80.0,
        "complexity": 7.0,
        "change_frequency": 4.0,
        "churn": 15.0,
    }

    created = await refresh_metric_snapshots(test_session, scenario.id)
    await test_session.flush()

    assert created == 1
    snapshot = (
        (
            await test_session.execute(
                select(MetricSnapshot).where(MetricSnapshot.scenario_id == scenario.id)
            )
        )
        .scalars()
        .one()
    )
    assert snapshot.change_frequency == pytest.approx(4.0)
    assert float((snapshot.meta or {}).get("churn", 0.0)) == pytest.approx(15.0)
