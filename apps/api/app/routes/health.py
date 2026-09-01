"""Health check and configuration endpoints."""

import asyncio
import os

from contextmine_core import get_engine
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import text

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str


class FrontendConfigResponse(BaseModel):
    faroUrl: str | None = None
    version: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return process liveness without checking external dependencies."""
    return HealthResponse(status="ok")


@router.get("/health/live", response_model=HealthResponse)
async def liveness_check() -> HealthResponse:
    """Return process liveness for orchestrator probes."""
    return HealthResponse(status="ok")


@router.get("/health/ready", response_model=HealthResponse)
async def readiness_check() -> HealthResponse:
    """Return readiness after checking the required database dependency."""
    try:
        async with asyncio.timeout(2.0):
            async with get_engine().connect() as connection:
                await connection.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(status_code=503, detail="database unavailable") from exc
    return HealthResponse(status="ready")


@router.get("/config", response_model=FrontendConfigResponse)
async def frontend_config() -> FrontendConfigResponse:
    """Return frontend runtime configuration.

    This endpoint provides configuration that the frontend needs at runtime,
    such as the Faro collector URL for observability.
    """
    return FrontendConfigResponse(
        faroUrl=os.getenv("FARO_COLLECTOR_URL"),
        version=os.getenv("APP_VERSION", "0.0.0"),
    )
