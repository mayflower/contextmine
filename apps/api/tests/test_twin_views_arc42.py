"""Tests for arc42 and drift twin view routes."""

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from contextmine_core.architecture.schemas import Arc42Document, ArchitectureFactsBundle
from httpx import AsyncClient


@pytest.mark.anyio
class TestTwinArc42Views:
    async def test_arc42_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get("/api/twin/collections/some-id/views/arc42")
        assert response.status_code == 401

    @patch("app.routes.twin.get_session")
    async def test_arc42_invalid_section_rejected(
        self,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        collection_id = str(uuid.uuid4())
        response = await client.get(
            f"/api/twin/collections/{collection_id}/views/arc42?section=does-not-exist"
        )
        assert response.status_code == 400
        assert "Invalid arc42 section" in response.json()["detail"]

    @patch("app.routes.twin._upsert_artifact", new_callable=AsyncMock)
    @patch("app.routes.twin.generate_arc42_from_facts")
    @patch("app.routes.twin._build_arch_bundle", new_callable=AsyncMock)
    @patch("app.routes.twin._resolve_view_scenario", new_callable=AsyncMock)
    @patch("app.routes.twin._ensure_member", new_callable=AsyncMock)
    @patch("app.routes.twin.get_db_session")
    @patch("app.routes.twin.get_session")
    async def test_arc42_view_returns_generated_document(
        self,
        mock_get_session: Any,
        mock_db_session_factory: Any,
        _mock_ensure_member: Any,
        mock_resolve_view_scenario: Any,
        mock_build_arch_bundle: Any,
        mock_generate_arc42: Any,
        mock_upsert_artifact: Any,
        client: AsyncClient,
    ) -> None:
        collection_id = uuid.uuid4()
        scenario_id = uuid.uuid4()
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}

        scenario = MagicMock()
        scenario.id = scenario_id
        scenario.collection_id = collection_id
        scenario.name = "AS-IS"
        scenario.version = 3
        scenario.is_as_is = True
        scenario.base_scenario_id = None
        mock_resolve_view_scenario.return_value = scenario

        bundle = ArchitectureFactsBundle(
            collection_id=collection_id,
            scenario_id=scenario_id,
            scenario_name="AS-IS",
            facts=[],
            ports_adapters=[],
            warnings=[],
        )
        mock_build_arch_bundle.return_value = bundle

        document = Arc42Document(
            collection_id=collection_id,
            scenario_id=scenario_id,
            scenario_name="AS-IS",
            title="arc42 - AS-IS",
            generated_at=datetime.now(UTC),
            sections={"1_introduction_and_goals": "Intro"},
            markdown="# arc42 - AS-IS\n",
            warnings=[],
            confidence_summary={"total": 0},
            section_coverage={"1_introduction_and_goals": True},
        )
        mock_generate_arc42.return_value = document

        artifact = MagicMock()
        artifact.id = uuid.uuid4()
        artifact.name = f"{scenario_id}.arc42.md"
        artifact.kind.value = "arc42"
        mock_upsert_artifact.return_value = artifact

        query_result = MagicMock()
        query_result.scalar_one_or_none.return_value = None

        fake_db = MagicMock()
        fake_db.execute = AsyncMock(return_value=query_result)
        fake_db.commit = AsyncMock()

        class SessionContext:
            async def __aenter__(self):  # noqa: ANN001
                return fake_db

            async def __aexit__(self, _exc_type, _exc, _tb):  # noqa: ANN001
                return False

        mock_db_session_factory.return_value = SessionContext()

        response = await client.get(f"/api/twin/collections/{collection_id}/views/arc42")
        assert response.status_code == 200
        payload = response.json()
        assert payload["artifact"]["kind"] == "arc42"
        assert payload["arc42"]["title"] == "arc42 - AS-IS"
        assert payload["facts_count"] == 0

    @patch("app.routes.twin._resolve_baseline_scenario", new_callable=AsyncMock)
    @patch("app.routes.twin._build_arch_bundle", new_callable=AsyncMock)
    @patch("app.routes.twin._resolve_view_scenario", new_callable=AsyncMock)
    @patch("app.routes.twin._ensure_member", new_callable=AsyncMock)
    @patch("app.routes.twin.get_db_session")
    @patch("app.routes.twin.get_session")
    async def test_arc42_drift_view_returns_summary(
        self,
        mock_get_session: Any,
        mock_db_session_factory: Any,
        _mock_ensure_member: Any,
        mock_resolve_view_scenario: Any,
        mock_build_arch_bundle: Any,
        mock_resolve_baseline: Any,
        client: AsyncClient,
    ) -> None:
        collection_id = uuid.uuid4()
        scenario_id = uuid.uuid4()
        baseline_id = uuid.uuid4()
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}

        scenario = MagicMock()
        scenario.id = scenario_id
        scenario.collection_id = collection_id
        scenario.name = "AS-IS"
        scenario.version = 5
        scenario.is_as_is = True
        scenario.base_scenario_id = baseline_id
        mock_resolve_view_scenario.return_value = scenario

        baseline = MagicMock()
        baseline.id = baseline_id
        baseline.collection_id = collection_id
        baseline.name = "AS-IS-OLD"
        baseline.version = 4
        baseline.is_as_is = False
        baseline.base_scenario_id = None
        mock_resolve_baseline.return_value = baseline

        current_bundle = ArchitectureFactsBundle(
            collection_id=collection_id,
            scenario_id=scenario_id,
            scenario_name="AS-IS",
            warnings=[],
        )
        baseline_bundle = ArchitectureFactsBundle(
            collection_id=collection_id,
            scenario_id=baseline_id,
            scenario_name="AS-IS-OLD",
            warnings=[],
        )
        mock_build_arch_bundle.side_effect = [current_bundle, baseline_bundle]

        fake_db = MagicMock()

        class SessionContext:
            async def __aenter__(self):  # noqa: ANN001
                return fake_db

            async def __aexit__(self, _exc_type, _exc, _tb):  # noqa: ANN001
                return False

        mock_db_session_factory.return_value = SessionContext()

        response = await client.get(f"/api/twin/collections/{collection_id}/views/arc42/drift")
        assert response.status_code == 200
        payload = response.json()
        assert payload["summary"]["severity"] in {"low", "medium"}
        assert payload["baseline_scenario"]["id"] == str(baseline_id)
