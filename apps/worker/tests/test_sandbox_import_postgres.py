"""PostgreSQL proof that one sandbox result can be imported twice safely."""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import contextmine_worker.flows as flows
import pytest
from contextmine_core import (
    Base,
    Collection,
    CollectionVisibility,
    Document,
    Source,
    SourceType,
    User,
)
from contextmine_core.models import TwinEvent, TwinScenario, TwinSourceVersion
from contextmine_worker.sandbox_analysis import AnalyzedFile, SandboxAnalysisResult
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"),
    reason="TEST_DATABASE_URL not set - skipping PostgreSQL sandbox import test",
)
async def test_same_source_commit_profile_imports_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url = os.environ["TEST_DATABASE_URL"]
    if not database_url.startswith(("postgresql+asyncpg://", "postgresql://")):
        pytest.skip("TEST_DATABASE_URL must point to PostgreSQL")
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    schema = f"sandbox_import_test_{uuid.uuid4().hex}"
    admin_engine = create_async_engine(database_url)
    async with admin_engine.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        await connection.execute(
            text(f"CREATE TYPE \"{schema}\".collection_visibility AS ENUM ('global', 'private')")
        )
        await connection.execute(
            text(f"CREATE TYPE \"{schema}\".source_type AS ENUM ('github', 'web')")
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
                        Document.__table__,
                        TwinScenario.__table__,
                        TwinSourceVersion.__table__,
                        TwinEvent.__table__,
                    ],
                )
            )

        source_id = uuid.uuid4()
        collection_id = uuid.uuid4()
        async with factory() as session:
            user = User(github_user_id=202, github_login="sandbox-import-test")
            collection = Collection(
                id=collection_id,
                slug=f"sandbox-{uuid.uuid4().hex}",
                name="Sandbox import test",
                visibility=CollectionVisibility.PRIVATE,
                owner=user,
            )
            source = Source(
                id=source_id,
                collection=collection,
                type=SourceType.GITHUB,
                url="https://github.com/mayflower/contextmine",
                config={"owner": "mayflower", "repo": "contextmine", "branch": "main"},
                schedule_interval_minutes=60,
            )
            session.add(source)
            await session.commit()

        commit = "d" * 40
        result = SandboxAnalysisResult(
            source_id=source_id,
            analyzer_profile="scip-kg-v1",
            commit=commit,
            files=[AnalyzedFile(path="src/example.py", content="answer = 42\n")],
        )
        ctx = flows._SyncGitHubCtx(
            source=source,
            sync_run=type("Run", (), {"id": uuid.uuid4()})(),
            run_started_at=datetime.now(UTC),
            progress_id="progress",
            owner="mayflower",
            repo="contextmine",
            branch="main",
            new_sha=commit,
            sandbox_result=result,
        )
        monkeypatch.setattr(flows, "get_session", test_session)
        monkeypatch.setattr(flows, "update_progress_artifact", AsyncMock())
        monkeypatch.setattr(flows, "_gh_recover_unchunked_documents", AsyncMock())

        await flows._gh_phase_create_source_version(ctx)
        first_source_version_id = ctx.source_version_id
        await flows._gh_phase_create_source_version(ctx)
        assert ctx.source_version_id == first_source_version_id

        await flows._gh_phase_diff_and_index_documents(ctx)
        await flows._gh_phase_diff_and_index_documents(ctx)

        async with factory() as session:
            source_version_count = await session.scalar(
                select(func.count()).select_from(TwinSourceVersion)
            )
            event_count = await session.scalar(select(func.count()).select_from(TwinEvent))
            documents = (await session.execute(select(Document))).scalars().all()
            persisted_source = await session.get(Source, source_id)

        assert source_version_count == 1
        assert event_count == 1
        assert len(documents) == 1
        assert documents[0].content_markdown == "answer = 42\n"
        assert persisted_source is not None and persisted_source.cursor == commit
    finally:
        await engine.dispose()
        async with admin_engine.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await admin_engine.dispose()
