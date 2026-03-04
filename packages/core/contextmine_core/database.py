"""Database configuration with SQLAlchemy 2.x async support."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from contextmine_core.settings import get_settings
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


# Global engine and session factory
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Get or create the async database engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        if not settings.database_url:
            raise RuntimeError("DATABASE_URL environment variable is not set")
        url = make_url(settings.database_url)
        engine_kwargs: dict[str, object] = {
            "echo": settings.debug,
            "pool_pre_ping": True,
        }
        # SQLite uses dedicated pool classes that don't accept pool tuning kwargs.
        if not url.drivername.startswith("sqlite"):
            engine_kwargs.update(
                {
                    "pool_size": max(1, int(settings.database_pool_size)),
                    "max_overflow": max(0, int(settings.database_max_overflow)),
                    "pool_timeout": max(5, int(settings.database_pool_timeout_seconds)),
                }
            )
        _engine = create_async_engine(
            settings.database_url,
            **engine_kwargs,
        )

        # Auto-instrument SQLAlchemy if OTEL is enabled
        if settings.otel_enabled:
            from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

            # Instrument the sync engine (async engines wrap a sync engine internally)
            SQLAlchemyInstrumentor().instrument(
                engine=_engine.sync_engine,
                enable_commenter=True,
            )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the async session factory."""
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        _session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get an async database session."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def close_engine() -> None:
    """Close the database engine."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
