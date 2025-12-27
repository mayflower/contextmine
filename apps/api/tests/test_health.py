"""Tests for health check endpoint."""

import pytest
from httpx import AsyncClient


@pytest.mark.anyio
async def test_health_check(client: AsyncClient) -> None:
    """Test that /api/health returns status ok."""
    response = await client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data == {"status": "ok"}
