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
async def test_liveness_does_not_require_database(client: AsyncClient) -> None:
    response = await client.get("/api/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_readiness_reports_unavailable_database(client: AsyncClient, monkeypatch) -> None:
    def unavailable_engine():
        raise RuntimeError("database not configured")

    monkeypatch.setattr("app.routes.health.get_engine", unavailable_engine)

    response = await client.get("/api/health/ready")

    assert response.status_code == 503
    assert response.json() == {"detail": "database unavailable"}


@pytest.mark.anyio
async def test_frontend_config(client: AsyncClient) -> None:
    """Test that /api/config returns frontend runtime config."""
    response = await client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert "faroUrl" in data
    assert "version" in data
