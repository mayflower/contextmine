"""Tests for health check and config endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.anyio
async def test_health_check(client: AsyncClient) -> None:
    """Test that /api/health returns status ok."""
    response = await client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data == {"status": "ok"}


@pytest.mark.anyio
async def test_frontend_config(client: AsyncClient) -> None:
    """Test that /api/config returns frontend runtime config."""
    response = await client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert "faroUrl" in data
    assert "version" in data
