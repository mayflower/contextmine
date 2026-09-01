"""Tests for explicit database transaction ownership."""

import os
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from contextmine_core import database
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


def _session_factory(session: AsyncMock) -> MagicMock:
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=session)
    context.__aexit__ = AsyncMock(return_value=None)
    factory = MagicMock(return_value=context)
    return factory


@pytest.mark.anyio
async def test_read_session_does_not_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    session = AsyncMock()
    monkeypatch.setattr(database, "get_session_factory", lambda: _session_factory(session))

    async with database.get_session() as yielded:
        assert yielded is session

    session.commit.assert_not_awaited()
    session.rollback.assert_not_awaited()


@pytest.mark.anyio
async def test_session_rolls_back_when_caller_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    session = AsyncMock()
    monkeypatch.setattr(database, "get_session_factory", lambda: _session_factory(session))

    with pytest.raises(RuntimeError, match="write failed"):
        async with database.get_session():
            raise RuntimeError("write failed")

    session.commit.assert_not_awaited()
    session.rollback.assert_awaited_once()


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"),
    reason="TEST_DATABASE_URL not set - skipping PostgreSQL transaction test",
)
async def test_postgres_transaction_outcomes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prove read, commit, and rollback behavior against an isolated PostgreSQL schema."""
    database_url = os.environ["TEST_DATABASE_URL"]
    if not database_url.startswith(("postgresql+asyncpg://", "postgresql://")):
        pytest.skip("TEST_DATABASE_URL must point to PostgreSQL")

    engine = create_async_engine(database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    schema = f"transaction_test_{uuid.uuid4().hex}"
    commits = 0

    @event.listens_for(engine.sync_engine, "commit")
    def count_commit(_connection: object) -> None:
        nonlocal commits
        commits += 1

    try:
        async with engine.begin() as connection:
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            await connection.execute(text(f'CREATE TABLE "{schema}".events (value TEXT NOT NULL)'))

        monkeypatch.setattr(database, "get_session_factory", lambda: factory)

        commit_count = commits
        async with database.get_session() as session:
            result = await session.execute(text(f'SELECT count(*) FROM "{schema}".events'))
            assert result.scalar_one() == 0
        assert commits == commit_count

        async with database.get_session() as session:
            await session.execute(
                text(f'INSERT INTO "{schema}".events (value) VALUES (:value)'),
                {"value": "committed"},
            )
            await session.commit()
        assert commits == commit_count + 1

        with pytest.raises(RuntimeError, match="write failed"):
            async with database.get_session() as session:
                await session.execute(
                    text(f'INSERT INTO "{schema}".events (value) VALUES (:value)'),
                    {"value": "rolled back"},
                )
                raise RuntimeError("write failed")

        async with engine.connect() as connection:
            values = (
                (await connection.execute(text(f'SELECT value FROM "{schema}".events')))
                .scalars()
                .all()
            )
        assert values == ["committed"]
    finally:
        async with engine.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await engine.dispose()
