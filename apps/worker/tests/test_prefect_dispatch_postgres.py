"""PostgreSQL concurrency proof for Prefect source dispatch."""

from __future__ import annotations

import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import contextmine_worker.flows as flows
import pytest
from contextmine_core import (
    Base,
    Collection,
    CollectionVisibility,
    Source,
    SourceType,
    SyncRun,
    SyncRunStatus,
    User,
)
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"),
    reason="TEST_DATABASE_URL not set - skipping PostgreSQL dispatcher test",
)
async def test_concurrent_dispatchers_claim_source_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two dispatchers cannot create duplicate business runs for one due source."""
    database_url = os.environ["TEST_DATABASE_URL"]
    if not database_url.startswith(("postgresql+asyncpg://", "postgresql://")):
        pytest.skip("TEST_DATABASE_URL must point to PostgreSQL")
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    schema = f"prefect_dispatch_test_{uuid.uuid4().hex}"
    admin_engine = create_async_engine(database_url)
    async with admin_engine.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        await connection.execute(
            text(f"CREATE TYPE \"{schema}\".collection_visibility AS ENUM ('global', 'private')")
        )
        await connection.execute(
            text(f"CREATE TYPE \"{schema}\".source_type AS ENUM ('github', 'web')")
        )
        await connection.execute(
            text(
                f'CREATE TYPE "{schema}".sync_run_status AS ENUM '
                "('scheduled', 'running', 'success', 'failed', 'cancelled', 'timed_out')"
            )
        )

    engine = create_async_engine(
        database_url,
        connect_args={"server_settings": {"search_path": schema}},
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)

    @asynccontextmanager
    async def test_session():
        async with factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise

    try:
        async with engine.begin() as connection:
            await connection.run_sync(
                lambda sync_connection: Base.metadata.create_all(
                    sync_connection,
                    tables=[
                        User.__table__,
                        Collection.__table__,
                        Source.__table__,
                        SyncRun.__table__,
                    ],
                )
            )

        source_id = uuid.uuid4()
        async with factory() as session:
            user = User(github_user_id=101, github_login="dispatcher-test")
            collection = Collection(
                slug=f"dispatch-{uuid.uuid4().hex}",
                name="Dispatch test",
                visibility=CollectionVisibility.PRIVATE,
                owner=user,
            )
            source = Source(
                id=source_id,
                collection=collection,
                type=SourceType.GITHUB,
                url="https://github.com/example/repository",
                enabled=True,
                schedule_interval_minutes=60,
                next_run_at=datetime.now(UTC) - timedelta(minutes=1),
            )
            session.add(source)
            await session.commit()

        monkeypatch.setattr(flows, "get_session", test_session)
        claims = await asyncio.gather(
            flows.claim_due_source_runs.fn(),
            flows.claim_due_source_runs.fn(),
        )

        assert sum(len(batch) for batch in claims) == 1
        assert {claim["source_id"] for batch in claims for claim in batch} == {str(source_id)}

        async with factory() as session:
            runs = (
                (await session.execute(select(SyncRun).where(SyncRun.source_id == source_id)))
                .scalars()
                .all()
            )
            persisted_source = await session.get(Source, source_id)
        assert len(runs) == 1
        assert runs[0].status == SyncRunStatus.SCHEDULED
        assert persisted_source is not None
        assert persisted_source.next_run_at > datetime.now(UTC)
    finally:
        await engine.dispose()
        async with admin_engine.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await admin_engine.dispose()
