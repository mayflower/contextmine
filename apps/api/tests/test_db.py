"""Tests for database health check endpoint."""

import os

import pytest
from httpx import AsyncClient


@pytest.mark.anyio
async def test_db_health_no_database_url(client: AsyncClient) -> None:
    """Test /api/db/health when DATABASE_URL is not configured."""
    # Clear DATABASE_URL to test unconfigured state
    original = os.environ.get("DATABASE_URL")
    if "DATABASE_URL" in os.environ:
        del os.environ["DATABASE_URL"]

    # Reset settings singleton to pick up env change
    import contextmine_core.settings

    contextmine_core.settings._settings = None

    try:
        response = await client.get("/api/db/health")
        assert response.status_code == 200
        data = response.json()
        assert data["db"] == "not_configured"
    finally:
        # Restore original DATABASE_URL if it existed
        if original is not None:
            os.environ["DATABASE_URL"] = original
        contextmine_core.settings._settings = None


@pytest.mark.anyio
@pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set - skipping integration test",
)
async def test_db_health_with_database(client: AsyncClient) -> None:
    """Test /api/db/health when database is configured and accessible."""
    response = await client.get("/api/db/health")
    assert response.status_code == 200
    data = response.json()
    assert data["db"] == "ok"
