"""Health check and configuration endpoints."""

import os

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Return API health status."""
    return {"status": "ok"}


@router.get("/config")
async def frontend_config() -> dict[str, str | None]:
    """Return frontend runtime configuration.

    This endpoint provides configuration that the frontend needs at runtime,
    such as the Faro collector URL for observability.
    """
    return {
        "faroUrl": os.getenv("FARO_COLLECTOR_URL"),
        "version": os.getenv("APP_VERSION", "0.0.0"),
    }
