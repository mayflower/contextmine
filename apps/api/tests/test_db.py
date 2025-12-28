"""Tests for database health check endpoint."""

import os
from unittest.mock import MagicMock, patch

import pytest
from httpx import AsyncClient


@pytest.mark.anyio
async def test_db_health_no_database_url(client: AsyncClient) -> None:
    """Test /api/db/health when DATABASE_URL is not configured."""
    # Mock settings to return None for database_url
    mock_settings = MagicMock()
    mock_settings.database_url = None

    with patch("app.routes.db.get_settings", return_value=mock_settings):
        response = await client.get("/api/db/health")
        assert response.status_code == 200
        data = response.json()
        assert data["db"] == "not_configured"


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
