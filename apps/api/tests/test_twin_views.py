"""Tests for extracted twin view routes."""

from typing import Any
from unittest.mock import patch

import pytest
from httpx import AsyncClient


@pytest.mark.anyio
class TestTwinViewRoutes:
    """Route-level validation for collection extracted views."""

    async def test_city_view_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get("/api/twin/collections/some-id/views/city")
        assert response.status_code == 401

    @patch("app.routes.twin.get_session")
    async def test_view_invalid_collection_id_rejected(
        self,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        import uuid

        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        response = await client.get("/api/twin/collections/not-a-uuid/views/city")
        assert response.status_code == 400
        assert "Invalid collection_id" in response.json()["detail"]

    @patch("app.routes.twin.get_session")
    async def test_topology_invalid_layer_rejected(
        self,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        import uuid

        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        collection_id = str(uuid.uuid4())
        response = await client.get(
            f"/api/twin/collections/{collection_id}/views/topology?layer=invalid-layer"
        )
        assert response.status_code == 400
        assert "Invalid layer" in response.json()["detail"]
