"""Tests for ports/adapters twin view route."""

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from contextmine_core.architecture.schemas import ArchitectureFactsBundle, PortAdapterFact
from httpx import AsyncClient


@pytest.mark.anyio
class TestTwinPortsAdaptersView:
    async def test_ports_adapters_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get("/api/twin/collections/some-id/views/ports-adapters")
        assert response.status_code == 401

    @patch("app.routes.twin.get_session")
    async def test_ports_adapters_invalid_direction_rejected(
        self,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        collection_id = str(uuid.uuid4())
        response = await client.get(
            f"/api/twin/collections/{collection_id}/views/ports-adapters?direction=sideways"
        )
        assert response.status_code == 400
        assert "Invalid direction" in response.json()["detail"]

    @patch("app.routes.twin._build_arch_bundle", new_callable=AsyncMock)
    @patch("app.routes.twin._resolve_view_scenario", new_callable=AsyncMock)
    @patch("app.routes.twin._ensure_member", new_callable=AsyncMock)
    @patch("app.routes.twin.get_db_session")
    @patch("app.routes.twin.get_session")
    async def test_ports_adapters_view_filters_by_direction_and_container(
        self,
        mock_get_session: Any,
        mock_db_session_factory: Any,
        _mock_ensure_member: Any,
        mock_resolve_view_scenario: Any,
        mock_build_arch_bundle: Any,
        client: AsyncClient,
    ) -> None:
        collection_id = uuid.uuid4()
        scenario_id = uuid.uuid4()
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}

        scenario = MagicMock()
        scenario.id = scenario_id
        scenario.collection_id = collection_id
        scenario.name = "AS-IS"
        scenario.version = 1
        scenario.is_as_is = True
        scenario.base_scenario_id = None
        mock_resolve_view_scenario.return_value = scenario

        bundle = ArchitectureFactsBundle(
            collection_id=collection_id,
            scenario_id=scenario_id,
            scenario_name="AS-IS",
            ports_adapters=[
                PortAdapterFact(
                    fact_id="inbound:1",
                    direction="inbound",
                    port_name="CreateOrder",
                    adapter_name="orders_controller",
                    container="orders",
                    component="controller",
                    protocol="http",
                    source="deterministic",
                    confidence=0.9,
                ),
                PortAdapterFact(
                    fact_id="outbound:1",
                    direction="outbound",
                    port_name="orders_db",
                    adapter_name="orders_repo",
                    container="orders",
                    component="repository",
                    protocol="sql",
                    source="deterministic",
                    confidence=0.9,
                ),
            ],
            warnings=[],
        )
        mock_build_arch_bundle.return_value = bundle

        fake_db = MagicMock()

        class SessionContext:
            async def __aenter__(self):  # noqa: ANN001
                return fake_db

            async def __aexit__(self, _exc_type, _exc, _tb):  # noqa: ANN001
                return False

        mock_db_session_factory.return_value = SessionContext()

        response = await client.get(
            f"/api/twin/collections/{collection_id}/views/ports-adapters"
            "?direction=inbound&container=orders"
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["summary"]["total"] == 1
        assert payload["summary"]["inbound"] == 1
        assert payload["summary"]["outbound"] == 0
        assert payload["items"][0]["direction"] == "inbound"
