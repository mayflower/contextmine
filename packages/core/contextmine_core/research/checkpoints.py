"""Lifecycle for the process-wide LangGraph PostgreSQL checkpointer."""

from __future__ import annotations

import sys
from contextlib import AbstractAsyncContextManager

from contextmine_core.settings import get_settings
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy.engine import make_url

_checkpointer_context: AbstractAsyncContextManager[AsyncPostgresSaver] | None = None
_checkpointer: AsyncPostgresSaver | None = None


def _checkpoint_connection_string(database_url: str) -> str:
    url = make_url(database_url)
    if not url.drivername.startswith("postgresql"):
        raise RuntimeError("LangGraph checkpoints require PostgreSQL DATABASE_URL")
    return url.set(drivername="postgresql").render_as_string(hide_password=False)


async def init_research_checkpointer() -> AsyncPostgresSaver:
    """Open and initialize the official async PostgreSQL checkpointer once."""
    global _checkpointer, _checkpointer_context
    if _checkpointer is not None:
        return _checkpointer

    settings = get_settings()
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    context = AsyncPostgresSaver.from_conn_string(
        _checkpoint_connection_string(settings.database_url)
    )
    checkpointer = await context.__aenter__()
    try:
        await checkpointer.setup()
    except BaseException:
        await context.__aexit__(*sys.exc_info())
        raise
    _checkpointer_context = context
    _checkpointer = checkpointer
    return checkpointer


def get_research_checkpointer() -> AsyncPostgresSaver:
    """Return the checkpointer initialized by the application lifespan."""
    if _checkpointer is None:
        raise RuntimeError("Research checkpointer is not initialized")
    return _checkpointer


async def close_research_checkpointer() -> None:
    """Close the process-wide checkpointer connection."""
    global _checkpointer, _checkpointer_context
    context = _checkpointer_context
    _checkpointer = None
    _checkpointer_context = None
    if context is not None:
        await context.__aexit__(None, None, None)
