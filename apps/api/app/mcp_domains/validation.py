"""Validation status MCP tools."""

import uuid
from typing import Annotated, Any

from contextmine_core import get_session as get_db_session
from contextmine_core.validation import (
    get_latest_validation_status,
    refresh_validation_snapshots,
)
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field


class ValidationDashboardResult(BaseModel):
    """Normalized validation status returned by the existing validation core."""

    collection_id: str | None
    status: dict[str, Any]


def _optional_uuid(value: str | None) -> uuid.UUID | None:
    if value is None:
        return None
    try:
        return uuid.UUID(value)
    except ValueError as exc:
        raise ToolError(f"Invalid collection_id: {value}") from exc


async def get_validation_dashboard(
    collection_id: Annotated[str | None, Field(description="Optional collection UUID")] = None,
) -> ValidationDashboardResult:
    """Refresh and return the normalized Tekton, Argo, and Temporal status."""
    collection_uuid = _optional_uuid(collection_id)
    try:
        async with get_db_session() as db:
            await refresh_validation_snapshots(db, collection_uuid)
            payload = await get_latest_validation_status(db, collection_uuid)
            await db.commit()
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Failed to fetch validation dashboard: {exc}") from exc
    return ValidationDashboardResult(collection_id=collection_id, status=payload)


def register_validation_tools(mcp: FastMCP) -> None:
    """Register validation tools on the application MCP server."""
    mcp.tool(name="get_validation_dashboard")(get_validation_dashboard)
