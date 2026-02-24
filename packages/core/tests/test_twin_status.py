from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import pytest
from contextmine_core.database import Base
from contextmine_core.models import (
    Collection,
    CollectionVisibility,
    Source,
    SourceType,
    TwinScenario,
    TwinSourceVersion,
    User,
)
from contextmine_core.twin.ops import get_collection_twin_status
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


@pytest.mark.anyio
async def test_get_collection_twin_status_exposes_scip_and_metrics_gate_fields(
    test_session: AsyncSession,
) -> None:
    now = datetime.now(UTC)
    user = User(id=uuid.uuid4(), github_user_id=100, github_login="status-owner")
    collection = Collection(
        id=uuid.uuid4(),
        slug="status-collection",
        name="Status Collection",
        visibility=CollectionVisibility.PRIVATE,
        owner_user_id=user.id,
    )
    source = Source(
        id=uuid.uuid4(),
        collection_id=collection.id,
        type=SourceType.GITHUB,
        url="https://github.com/acme/repo",
        config={"owner": "acme", "repo": "repo"},
        enabled=True,
        schedule_interval_minutes=60,
    )
    scenario = TwinScenario(
        id=uuid.uuid4(),
        collection_id=collection.id,
        name="AS-IS",
        is_as_is=True,
        version=2,
        meta={},
    )
    source_version = TwinSourceVersion(
        id=uuid.uuid4(),
        collection_id=collection.id,
        source_id=source.id,
        revision_key="abc123",
        extractor_version="scip-kg-v1",
        status="ready",
        stats={
            "scip_projects_detected": 2,
            "scip_projects_indexed": 1,
            "scip_projects_failed": 1,
            "scip_degraded": True,
            "scip_projects_by_language": {"php": 1, "typescript": 1},
            "scip_failed_projects": [
                {"language": "php", "project_root": "/repo", "error": "index failed"}
            ],
            "metrics_requested_files": 10,
            "metrics_mapped_files": 8,
            "metrics_unmapped_sample": ["src/missing.php"],
            "metrics_gate": "fail",
        },
        started_at=now,
        finished_at=now,
    )
    test_session.add_all([user, collection, source, scenario, source_version])
    await test_session.commit()

    payload = await get_collection_twin_status(test_session, collection_id=collection.id)

    assert payload["scip_status"] == "degraded"
    assert payload["scip_projects_by_language"] == {"php": 1, "typescript": 1}
    assert payload["scip_failed_projects"][0]["language"] == "php"
    assert payload["metrics_gate"]["status"] == "fail"
    assert payload["metrics_gate"]["requested_files"] == 10
    assert payload["metrics_gate"]["mapped_files"] == 8
    assert "src/missing.php" in payload["metrics_gate"]["unmapped_sample"]
