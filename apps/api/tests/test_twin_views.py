"""Tests for extracted twin view routes."""

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.routes.twin import _extract_document_lines, _upsert_artifact
from contextmine_core import SourceType
from contextmine_core.graphrag import EdgeContext, EntityContext, PathContext
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

    async def test_graphrag_view_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get("/api/twin/collections/some-id/views/graphrag")
        assert response.status_code == 401

    async def test_graphrag_evidence_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get(
            "/api/twin/collections/some-id/views/graphrag/evidence?node_id=abc"
        )
        assert response.status_code == 401

    async def test_graphrag_communities_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get("/api/twin/collections/some-id/views/graphrag/communities")
        assert response.status_code == 401

    async def test_graphrag_path_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get(
            "/api/twin/collections/some-id/views/graphrag/path?from_node_id=a&to_node_id=b"
        )
        assert response.status_code == 401

    async def test_graphrag_processes_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get("/api/twin/collections/some-id/views/graphrag/processes")
        assert response.status_code == 401

    async def test_graphrag_process_detail_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get("/api/twin/collections/some-id/views/graphrag/processes/proc_1")
        assert response.status_code == 401

    async def test_ui_map_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get("/api/twin/collections/some-id/views/ui-map")
        assert response.status_code == 401

    async def test_test_matrix_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get("/api/twin/collections/some-id/views/test-matrix")
        assert response.status_code == 401

    async def test_user_flows_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get("/api/twin/collections/some-id/views/user-flows")
        assert response.status_code == 401

    async def test_rebuild_readiness_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get("/api/twin/collections/some-id/views/rebuild-readiness")
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
    async def test_graphrag_invalid_collection_id_rejected(
        self,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        response = await client.get("/api/twin/collections/not-a-uuid/views/graphrag")
        assert response.status_code == 400
        assert "Invalid collection_id" in response.json()["detail"]

    @patch("app.routes.twin.get_session")
    async def test_graphrag_invalid_community_mode_rejected(
        self,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        collection_id = str(uuid.uuid4())
        response = await client.get(
            f"/api/twin/collections/{collection_id}/views/graphrag?community_mode=bad"
        )
        assert response.status_code == 400
        assert "Invalid community_mode" in response.json()["detail"]

    @patch("app.routes.twin._resolve_view_scenario", new_callable=AsyncMock)
    @patch("app.routes.twin._ensure_member", new_callable=AsyncMock)
    @patch("app.routes.twin.get_db_session")
    @patch("app.routes.twin.get_session")
    async def test_graphrag_view_includes_community_meta_keys(
        self,
        mock_get_session: Any,
        mock_db_session_factory: Any,
        _mock_ensure_member: Any,
        mock_resolve_view_scenario: Any,
        client: AsyncClient,
    ) -> None:
        collection_id = str(uuid.uuid4())
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}

        scenario = MagicMock()
        scenario.id = uuid.uuid4()
        scenario.collection_id = uuid.UUID(collection_id)
        scenario.name = "AS-IS"
        scenario.version = 1
        scenario.is_as_is = True
        scenario.base_scenario_id = None
        mock_resolve_view_scenario.return_value = scenario

        node = MagicMock()
        node.id = uuid.uuid4()
        node.natural_key = "symbol:test"
        node.kind.value = "symbol"
        node.name = "TestSymbol"
        node.meta = {"foo": "bar"}

        total_result = MagicMock()
        total_result.scalar_one.return_value = 1
        nodes_result = MagicMock()
        nodes_result.scalars.return_value.all.return_value = [node]
        edges_result = MagicMock()
        edges_result.scalars.return_value.all.return_value = []

        fake_db = MagicMock()
        fake_db.execute = AsyncMock(side_effect=[total_result, nodes_result, edges_result])

        class SessionContext:
            async def __aenter__(self):  # noqa: ANN001
                return fake_db

            async def __aexit__(self, _exc_type, _exc, _tb):  # noqa: ANN001
                return False

        mock_db_session_factory.return_value = SessionContext()

        response = await client.get(
            f"/api/twin/collections/{collection_id}/views/graphrag?community_mode=none"
        )
        assert response.status_code == 200
        payload = response.json()
        meta = payload["graph"]["nodes"][0]["meta"]
        assert meta["community_id"] is None
        assert meta["community_label"] is None
        assert meta["community_size"] is None
        assert meta["community_cohesion"] is None
        assert meta["community_focus"] is False

    @patch("app.routes.twin.graphrag_trace_path", new_callable=AsyncMock)
    @patch("app.routes.twin._resolve_knowledge_node", new_callable=AsyncMock)
    @patch("app.routes.twin._resolve_view_scenario", new_callable=AsyncMock)
    @patch("app.routes.twin._ensure_member", new_callable=AsyncMock)
    @patch("app.routes.twin.get_db_session")
    @patch("app.routes.twin.get_session")
    async def test_graphrag_path_view_returns_found_path(
        self,
        mock_get_session: Any,
        mock_db_session_factory: Any,
        _mock_ensure_member: Any,
        mock_resolve_view_scenario: Any,
        mock_resolve_node: Any,
        mock_trace_path: Any,
        client: AsyncClient,
    ) -> None:
        collection_id = str(uuid.uuid4())
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        mock_resolve_view_scenario.return_value = MagicMock()

        from_node = MagicMock()
        from_node.id = uuid.uuid4()
        from_node.natural_key = "symbol:from"
        from_node.kind.value = "symbol"
        from_node.name = "From"
        to_node = MagicMock()
        to_node.id = uuid.uuid4()
        to_node.natural_key = "symbol:to"
        to_node.kind.value = "symbol"
        to_node.name = "To"
        mock_resolve_node.side_effect = [from_node, to_node]

        context = MagicMock()
        context.entities = [
            EntityContext(
                node_id=from_node.id,
                kind="symbol",
                natural_key="symbol:from",
                name="From",
            ),
            EntityContext(
                node_id=to_node.id,
                kind="symbol",
                natural_key="symbol:to",
                name="To",
            ),
        ]
        context.edges = [
            EdgeContext(
                source_id=str(from_node.id),
                target_id=str(to_node.id),
                kind="symbol_calls_symbol",
            )
        ]
        context.paths = [
            PathContext(
                nodes=["symbol:from", "symbol:to"],
                edges=["symbol_calls_symbol"],
                description="From -> To",
            )
        ]
        mock_trace_path.return_value = context

        fake_db = MagicMock()

        class SessionContext:
            async def __aenter__(self):  # noqa: ANN001
                return fake_db

            async def __aexit__(self, _exc_type, _exc, _tb):  # noqa: ANN001
                return False

        mock_db_session_factory.return_value = SessionContext()

        response = await client.get(
            f"/api/twin/collections/{collection_id}/views/graphrag/path"
            f"?from_node_id={from_node.id}&to_node_id={to_node.id}&max_hops=6"
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "found"
        assert payload["path"]["hops"] == 1
        assert payload["path"]["edges"][0]["kind"] == "symbol_calls_symbol"

    @patch("app.routes.twin._detect_processes")
    @patch("app.routes.twin._compute_symbol_communities")
    @patch("app.routes.twin._resolve_view_scenario", new_callable=AsyncMock)
    @patch("app.routes.twin._ensure_member", new_callable=AsyncMock)
    @patch("app.routes.twin.get_db_session")
    @patch("app.routes.twin.get_session")
    async def test_graphrag_processes_view_returns_items(
        self,
        mock_get_session: Any,
        mock_db_session_factory: Any,
        _mock_ensure_member: Any,
        mock_resolve_view_scenario: Any,
        mock_compute_communities: Any,
        mock_detect_processes: Any,
        client: AsyncClient,
    ) -> None:
        collection_id = str(uuid.uuid4())
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        mock_resolve_view_scenario.return_value = MagicMock()
        mock_compute_communities.return_value = ({}, {})
        mock_detect_processes.return_value = [
            {
                "id": "proc_1",
                "label": "A -> B",
                "process_type": "intra_community",
                "step_count": 2,
                "community_ids": ["comm_1"],
                "entry_node_id": str(uuid.uuid4()),
                "terminal_node_id": str(uuid.uuid4()),
                "steps": [],
            }
        ]

        symbols_result = MagicMock()
        symbols_result.scalars.return_value.all.return_value = []
        edges_result = MagicMock()
        edges_result.scalars.return_value.all.return_value = []

        fake_db = MagicMock()
        fake_db.execute = AsyncMock(side_effect=[symbols_result, edges_result])

        class SessionContext:
            async def __aenter__(self):  # noqa: ANN001
                return fake_db

            async def __aexit__(self, _exc_type, _exc, _tb):  # noqa: ANN001
                return False

        mock_db_session_factory.return_value = SessionContext()

        response = await client.get(
            f"/api/twin/collections/{collection_id}/views/graphrag/processes"
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 1
        assert payload["items"][0]["id"] == "proc_1"

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
    async def test_mermaid_invalid_c4_view_rejected(
        self,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        collection_id = str(uuid.uuid4())
        response = await client.get(
            f"/api/twin/collections/{collection_id}/views/mermaid?c4_view=not-a-view"
        )
        assert response.status_code == 400
        assert "Invalid c4_view" in response.json()["detail"]

    @patch("app.routes.twin.get_session")
    async def test_mermaid_invalid_c4_scope_rejected(
        self,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        collection_id = str(uuid.uuid4())
        response = await client.get(
            f"/api/twin/collections/{collection_id}/views/mermaid?c4_scope=bad%0Avalue"
        )
        assert response.status_code == 400
        assert "Invalid c4_scope" in response.json()["detail"]

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

    @patch("app.routes.twin.export_codecharta_json", new_callable=AsyncMock)
    @patch("app.routes.twin._resolve_view_scenario", new_callable=AsyncMock)
    @patch("app.routes.twin._ensure_member", new_callable=AsyncMock)
    @patch("app.routes.twin.get_db_session")
    @patch("app.routes.twin.get_session")
    async def test_city_view_includes_change_frequency_and_churn(
        self,
        mock_get_session: Any,
        mock_db_session_factory: Any,
        _mock_ensure_member: Any,
        mock_resolve_view_scenario: Any,
        mock_export_cc_json: Any,
        client: AsyncClient,
    ) -> None:
        collection_id = str(uuid.uuid4())
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}

        scenario = MagicMock()
        scenario.id = uuid.uuid4()
        scenario.collection_id = uuid.UUID(collection_id)
        scenario.name = "AS-IS"
        scenario.version = 1
        scenario.is_as_is = True
        scenario.base_scenario_id = None
        mock_resolve_view_scenario.return_value = scenario
        mock_export_cc_json.return_value = (
            '{"projectName":"x","apiVersion":"1.5","nodes":[],"edges":[]}'
        )

        metric_a = MagicMock()
        metric_a.node_natural_key = "file:src/a.py"
        metric_a.loc = 10
        metric_a.symbol_count = 2
        metric_a.coverage = 80.0
        metric_a.complexity = 4.0
        metric_a.coupling = 3.0
        metric_a.change_frequency = 2.0
        metric_a.meta = {"churn": 5.0}

        metric_b = MagicMock()
        metric_b.node_natural_key = "file:src/b.py"
        metric_b.loc = 20
        metric_b.symbol_count = 4
        metric_b.coverage = 60.0
        metric_b.complexity = 6.0
        metric_b.coupling = 5.0
        metric_b.change_frequency = 4.0
        metric_b.meta = {"churn": 15.0}

        metrics_result = MagicMock()
        metrics_result.scalars.return_value.all.return_value = [metric_a, metric_b]

        fake_db = MagicMock()
        fake_db.execute = AsyncMock(return_value=metrics_result)

        class SessionContext:
            async def __aenter__(self):  # noqa: ANN001
                return fake_db

            async def __aexit__(self, _exc_type, _exc, _tb):  # noqa: ANN001
                return False

        mock_db_session_factory.return_value = SessionContext()

        response = await client.get(f"/api/twin/collections/{collection_id}/views/city")
        assert response.status_code == 200

        payload = response.json()
        assert payload["summary"]["change_frequency_avg"] == pytest.approx(3.0)
        assert payload["summary"]["churn_avg"] == pytest.approx(10.0)
        assert payload["hotspots"][0]["change_frequency"] == pytest.approx(4.0)
        assert payload["hotspots"][0]["churn"] == pytest.approx(15.0)
        assert payload["hotspots"][1]["change_frequency"] == pytest.approx(2.0)
        assert payload["hotspots"][1]["churn"] == pytest.approx(5.0)

    @patch("app.routes.twin.export_codecharta_json", new_callable=AsyncMock)
    @patch("app.routes.twin.get_settings")
    @patch("app.routes.twin.refresh_metric_snapshots", new_callable=AsyncMock)
    @patch("app.routes.twin._resolve_view_scenario", new_callable=AsyncMock)
    @patch("app.routes.twin._ensure_member", new_callable=AsyncMock)
    @patch("app.routes.twin.get_db_session")
    @patch("app.routes.twin.get_session")
    async def test_city_view_unavailable_has_null_change_frequency_and_churn_avgs(
        self,
        mock_get_session: Any,
        mock_db_session_factory: Any,
        _mock_ensure_member: Any,
        mock_resolve_view_scenario: Any,
        _mock_refresh_snapshots: Any,
        mock_get_settings: Any,
        mock_export_cc_json: Any,
        client: AsyncClient,
    ) -> None:
        collection_id = str(uuid.uuid4())
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        mock_get_settings.return_value = MagicMock(metrics_strict_mode=True)

        scenario = MagicMock()
        scenario.id = uuid.uuid4()
        scenario.collection_id = uuid.UUID(collection_id)
        scenario.name = "AS-IS"
        scenario.version = 1
        scenario.is_as_is = True
        scenario.base_scenario_id = None
        mock_resolve_view_scenario.return_value = scenario
        mock_export_cc_json.return_value = (
            '{"projectName":"x","apiVersion":"1.5","nodes":[],"edges":[]}'
        )

        metrics_empty_result = MagicMock()
        metrics_empty_result.scalars.return_value.all.return_value = []
        sources_result = MagicMock()
        sources_result.scalars.return_value.all.return_value = [MagicMock(type=SourceType.GITHUB)]
        jobs_result = MagicMock()
        jobs_result.scalars.return_value.all.return_value = []
        file_nodes_result = MagicMock()
        file_nodes_result.scalars.return_value.all.return_value = []

        fake_db = MagicMock()
        fake_db.execute = AsyncMock(
            side_effect=[
                metrics_empty_result,
                metrics_empty_result,
                sources_result,
                jobs_result,
                file_nodes_result,
            ]
        )
        fake_db.flush = AsyncMock()

        class SessionContext:
            async def __aenter__(self):  # noqa: ANN001
                return fake_db

            async def __aexit__(self, _exc_type, _exc, _tb):  # noqa: ANN001
                return False

        mock_db_session_factory.return_value = SessionContext()

        response = await client.get(f"/api/twin/collections/{collection_id}/views/city")
        assert response.status_code == 200

        payload = response.json()
        assert payload["summary"]["change_frequency_avg"] is None
        assert payload["summary"]["churn_avg"] is None
        assert payload["hotspots"] == []

    @patch("app.routes.twin._resolve_view_scenario", new_callable=AsyncMock)
    @patch("app.routes.twin._ensure_member", new_callable=AsyncMock)
    @patch("app.routes.twin.get_db_session")
    @patch("app.routes.twin.get_session")
    async def test_graphrag_evidence_prefers_snippet(
        self,
        mock_get_session: Any,
        mock_db_session_factory: Any,
        _mock_ensure_member: Any,
        mock_resolve_view_scenario: Any,
        client: AsyncClient,
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        mock_resolve_view_scenario.return_value = MagicMock()

        collection_id = uuid.uuid4()
        node_id = uuid.uuid4()
        evidence_id = uuid.uuid4()

        node = MagicMock()
        node.id = node_id
        node.name = "Billing Context"
        node.kind.value = "bounded_context"

        evidence = MagicMock()
        evidence.id = evidence_id
        evidence.file_path = "src/billing/rules.ts"
        evidence.start_line = 4
        evidence.end_line = 10
        evidence.snippet = "Rule: paid invoices can be finalized"
        evidence.document_id = None

        node_result = MagicMock()
        node_result.scalar_one_or_none.return_value = node
        node_result.scalars.return_value.first.return_value = node
        total_result = MagicMock()
        total_result.scalar_one.return_value = 1
        evidence_result = MagicMock()
        evidence_result.scalars.return_value.all.return_value = [evidence]

        fake_db = MagicMock()
        fake_db.execute = AsyncMock(side_effect=[node_result, total_result, evidence_result])

        class SessionContext:
            async def __aenter__(self):  # noqa: ANN001
                return fake_db

            async def __aexit__(self, _exc_type, _exc, _tb):  # noqa: ANN001
                return False

        mock_db_session_factory.return_value = SessionContext()

        response = await client.get(
            f"/api/twin/collections/{collection_id}/views/graphrag/evidence?scenario_id={uuid.uuid4()}&node_id=node-key"
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["items"][0]["text_source"] == "snippet"
        assert payload["items"][0]["text"] == "Rule: paid invoices can be finalized"

    @patch("app.routes.twin._resolve_view_scenario", new_callable=AsyncMock)
    @patch("app.routes.twin._ensure_member", new_callable=AsyncMock)
    @patch("app.routes.twin.get_db_session")
    @patch("app.routes.twin.get_session")
    async def test_graphrag_evidence_uses_document_line_fallback(
        self,
        mock_get_session: Any,
        mock_db_session_factory: Any,
        _mock_ensure_member: Any,
        mock_resolve_view_scenario: Any,
        client: AsyncClient,
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        mock_resolve_view_scenario.return_value = MagicMock()

        collection_id = uuid.uuid4()
        node_id = uuid.uuid4()
        document_id = uuid.uuid4()

        node = MagicMock()
        node.id = node_id
        node.name = "InvoiceService"
        node.kind.value = "symbol"

        evidence = MagicMock()
        evidence.id = uuid.uuid4()
        evidence.file_path = "src/billing/invoice.ts"
        evidence.start_line = 2
        evidence.end_line = 3
        evidence.snippet = None
        evidence.document_id = document_id

        document = MagicMock()
        document.id = document_id
        document.content_markdown = "line1\nline2\nline3\nline4"

        node_result = MagicMock()
        node_result.scalar_one_or_none.return_value = node
        node_result.scalars.return_value.first.return_value = node
        total_result = MagicMock()
        total_result.scalar_one.return_value = 1
        evidence_result = MagicMock()
        evidence_result.scalars.return_value.all.return_value = [evidence]
        docs_result = MagicMock()
        docs_result.scalars.return_value.all.return_value = [document]

        fake_db = MagicMock()
        fake_db.execute = AsyncMock(
            side_effect=[node_result, total_result, evidence_result, docs_result]
        )

        class SessionContext:
            async def __aenter__(self):  # noqa: ANN001
                return fake_db

            async def __aexit__(self, _exc_type, _exc, _tb):  # noqa: ANN001
                return False

        mock_db_session_factory.return_value = SessionContext()

        response = await client.get(
            f"/api/twin/collections/{collection_id}/views/graphrag/evidence?scenario_id={uuid.uuid4()}&node_id=node-key"
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["items"][0]["text_source"] == "document_lines"
        assert payload["items"][0]["text"] == "line2\nline3"

    def test_extract_document_lines_caps_size(self) -> None:
        content = "\\n".join([f"line {idx}" for idx in range(1, 20)])
        excerpt = _extract_document_lines(content, 1, 19, max_chars=30)
        assert excerpt.endswith("...")
        assert len(excerpt) <= 33

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

    @patch("app.routes.twin.export_mermaid_asis_tobe_result", new_callable=AsyncMock)
    @patch("app.routes.twin._resolve_view_scenario", new_callable=AsyncMock)
    @patch("app.routes.twin._ensure_member", new_callable=AsyncMock)
    @patch("app.routes.twin.get_db_session")
    @patch("app.routes.twin.get_session")
    async def test_mermaid_compare_includes_c4_metadata_and_warnings(
        self,
        mock_get_session: Any,
        mock_db_session_factory: Any,
        _mock_ensure_member: Any,
        mock_resolve_view_scenario: Any,
        mock_export_compare: Any,
        client: AsyncClient,
    ) -> None:
        collection_id = str(uuid.uuid4())
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}

        scenario = MagicMock()
        scenario.id = uuid.uuid4()
        scenario.collection_id = uuid.UUID(collection_id)
        scenario.name = "TO-BE"
        scenario.version = 2
        scenario.is_as_is = False
        scenario.base_scenario_id = uuid.uuid4()
        mock_resolve_view_scenario.return_value = scenario

        mock_export_compare.return_value = (
            MagicMock(content='C4Component\nComponent(a, "A")', warnings=["AS-IS warn"]),
            MagicMock(content='C4Component\nComponent(b, "B")', warnings=["TO-BE warn"]),
        )

        fake_db = MagicMock()

        class SessionContext:
            async def __aenter__(self):  # noqa: ANN001
                return fake_db

            async def __aexit__(self, _exc_type, _exc, _tb):  # noqa: ANN001
                return False

        mock_db_session_factory.return_value = SessionContext()

        response = await client.get(
            f"/api/twin/collections/{collection_id}/views/mermaid"
            "?scenario_id="
            f"{scenario.id}&compare_with_base=true&c4_view=component&c4_scope=billing&max_nodes=77"
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["mode"] == "compare"
        assert payload["c4_view"] == "component"
        assert payload["c4_scope"] == "billing"
        assert payload["max_nodes"] == 77
        assert payload["as_is_warnings"] == ["AS-IS warn"]
        assert payload["to_be_warnings"] == ["TO-BE warn"]
        assert sorted(payload["warnings"]) == ["AS-IS warn", "TO-BE warn"]
