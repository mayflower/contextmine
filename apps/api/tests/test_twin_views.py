"""Tests for extracted twin view routes."""

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.routes.twin import _upsert_artifact
from contextmine_core.models import KnowledgeArtifactKind
from httpx import AsyncClient


@pytest.mark.anyio
class TestTwinViewRoutes:
    """Route-level validation for collection extracted views."""

    async def test_city_view_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get("/api/twin/collections/some-id/views/city")
        assert response.status_code == 401

    async def test_export_raw_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get("/api/twin/scenarios/some-scenario/exports/some-export/raw")
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

    @patch("app.routes.twin.get_session")
    async def test_topology_invalid_projection_rejected(
        self,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        import uuid

        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        collection_id = str(uuid.uuid4())
        response = await client.get(
            f"/api/twin/collections/{collection_id}/views/topology?projection=invalid-projection"
        )
        assert response.status_code == 400
        assert "Invalid projection" in response.json()["detail"]

    @patch("app.routes.twin.get_session")
    async def test_deep_dive_invalid_mode_rejected(
        self,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        import uuid

        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        collection_id = str(uuid.uuid4())
        response = await client.get(
            f"/api/twin/collections/{collection_id}/views/deep-dive?mode=not-a-mode"
        )
        assert response.status_code == 400
        assert "Invalid deep dive mode" in response.json()["detail"]

    @patch("app.routes.twin.get_session")
    async def test_graph_neighborhood_invalid_projection_rejected(
        self,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        import uuid

        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        scenario_id = str(uuid.uuid4())
        response = await client.get(
            f"/api/twin/scenarios/{scenario_id}/graph/neighborhood?node_id=abc&projection=invalid"
        )
        assert response.status_code == 400
        assert "Invalid projection" in response.json()["detail"]

    @patch("app.routes.twin.get_session")
    async def test_graph_neighborhood_invalid_hops_rejected(
        self,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        import uuid

        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        scenario_id = str(uuid.uuid4())
        response = await client.get(
            f"/api/twin/scenarios/{scenario_id}/graph/neighborhood?node_id=abc&hops=0"
        )
        assert response.status_code == 422

    @patch("app.routes.twin.get_session")
    async def test_export_raw_invalid_export_id_rejected(
        self,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        import uuid

        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        scenario_id = str(uuid.uuid4())
        response = await client.get(f"/api/twin/scenarios/{scenario_id}/exports/not-a-uuid/raw")
        assert response.status_code == 400
        assert "Invalid export_id" in response.json()["detail"]

    async def test_upsert_artifact_updates_existing(self) -> None:
        existing = MagicMock()
        existing.content = "old"
        existing.meta = {"old": "meta"}

        result = MagicMock()
        result.scalar_one_or_none.return_value = existing

        db = MagicMock()
        db.execute = AsyncMock(return_value=result)
        db.add = MagicMock()

        updated = await _upsert_artifact(
            db,
            collection_id=uuid.uuid4(),
            kind=KnowledgeArtifactKind.CC_JSON,
            name="AS-IS.cc.json",
            content='{"ok":true}',
            meta={"scenario_id": "s1"},
        )

        assert updated is existing
        assert existing.content == '{"ok":true}'
        assert existing.meta == {"scenario_id": "s1"}
        db.add.assert_not_called()

    async def test_upsert_artifact_inserts_new(self) -> None:
        result = MagicMock()
        result.scalar_one_or_none.return_value = None

        db = MagicMock()
        db.execute = AsyncMock(return_value=result)
        db.add = MagicMock()

        created = await _upsert_artifact(
            db,
            collection_id=uuid.uuid4(),
            kind=KnowledgeArtifactKind.CC_JSON,
            name="AS-IS.cc.json",
            content='{"ok":true}',
            meta={"scenario_id": "s1"},
        )

        assert created.kind == KnowledgeArtifactKind.CC_JSON
        assert created.name == "AS-IS.cc.json"
        assert created.content == '{"ok":true}'
        assert created.meta == {"scenario_id": "s1"}
        db.add.assert_called_once_with(created)
