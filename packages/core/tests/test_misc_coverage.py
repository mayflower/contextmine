"""Targeted tests for remaining core files below 80% coverage.

Covers: agent_sdk, joern, telemetry/setup, validation/connectors, validation/service,
        exports/twin_manifest, graph/age, database, embeddings, graph/builder,
        semantic_snapshot indexers (python, java, typescript, php), language_census.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import contextmine_core.database as db_module
import contextmine_core.telemetry.setup as telemetry_setup
import pytest
from contextmine_core.architecture.agent_sdk import (
    ClaudeAgentSdkUnavailableError,
    ClaudeSDKSessionManager,
    _arc42_prompt,
    _extract_json_blob,
    _render_markdown,
)
from contextmine_core.embeddings import (
    FakeEmbedder,
    GeminiEmbedder,
    OpenAIEmbedder,
    get_embedder,
    parse_embedding_model_spec,
)
from contextmine_core.graph.age import (
    _age_cypher_sql,
    _chunked,
    _esc,
    _validate_graph_name,
    ensure_read_only_cypher,
    scenario_graph_name,
)
from contextmine_core.graph.builder import GraphBuilder
from contextmine_core.graph.store import CodeGraph
from contextmine_core.joern import JoernClient, JoernResponse, parse_joern_output
from contextmine_core.models import EmbeddingProvider
from contextmine_core.semantic_snapshot.indexers.language_census import (
    LanguageCensusEntry,
    LanguageCensusReport,
    _build_not_match_dir_regex,
    _fallback_extension_census,
    _is_ignored_relative_path,
    _language_from_extension,
    _normalize_language_name,
    _parse_cloc_summary,
    build_language_census,
)
from contextmine_core.semantic_snapshot.models import (
    IndexConfig,
    Language,
    ProjectTarget,
)
from contextmine_core.validation.connectors import (
    ValidationMetric,
    _state_of,
    fetch_argo_metrics,
    fetch_tekton_metrics,
    fetch_temporal_alerts,
)
from contextmine_core.validation.service import _to_kind, refresh_validation_snapshots

pytestmark = pytest.mark.anyio

# ============================================================================
# architecture/agent_sdk.py
# ============================================================================


class TestExtractJsonBlob:
    def test_plain_json(self) -> None:
        result = _extract_json_blob('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_in_markdown_fences(self) -> None:
        raw = '```json\n{"title": "hello"}\n```'
        result = _extract_json_blob(raw)
        assert result["title"] == "hello"

    def test_json_embedded_in_text(self) -> None:
        raw = 'Some preamble text {"nested": true} trailing'
        result = _extract_json_blob(raw)
        assert result["nested"] is True

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="empty response"):
            _extract_json_blob("")

    def test_no_json_raises(self) -> None:
        with pytest.raises((ValueError, json.JSONDecodeError)):
            _extract_json_blob("just plain text with no braces")


class TestRenderMarkdown:
    def test_renders_all_sections(self) -> None:
        sections = {
            "1_introduction_and_goals": "Goals content",
            "2_constraints": "Constraints content",
        }
        result = _render_markdown("Test Title", sections)
        assert "# Test Title" in result
        assert "Goals content" in result
        assert "UNKNOWN: insufficient evidence" in result  # Missing sections


class TestArc42Prompt:
    def test_without_section_filter(self) -> None:
        prompt = _arc42_prompt(scenario_name="MyApp", section=None)
        assert "MyApp" in prompt
        assert "No section filter" in prompt

    def test_with_section_filter(self) -> None:
        prompt = _arc42_prompt(scenario_name="MyApp", section="5_building_block_view")
        assert "Focus section: 5_building_block_view" in prompt


class TestClaudeSDKSessionManager:
    async def test_unavailable_sdk_raises(self) -> None:
        manager = ClaudeSDKSessionManager()
        with (
            patch.dict("sys.modules", {"claude_code_sdk": None}),
            pytest.raises((ClaudeAgentSdkUnavailableError, ImportError)),
        ):
            await manager._get_entry(
                repo_path=Path("/tmp/nonexistent"),
                model="claude-sonnet-4-5-20250929",
                max_turns=5,
                permission_mode="bypassPermissions",
            )


# ============================================================================
# joern.py
# ============================================================================


class TestParseJoernOutput:
    def test_empty_input(self) -> None:
        assert parse_joern_output("") == []
        assert parse_joern_output("   ") == []

    def test_json_array(self) -> None:
        result = parse_joern_output("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_json_object(self) -> None:
        result = parse_joern_output('{"key": "val"}')
        assert result == {"key": "val"}

    def test_integer_output(self) -> None:
        result = parse_joern_output("42")
        assert result == 42

    def test_float_output(self) -> None:
        result = parse_joern_output("3.14")
        assert result == 3.14

    def test_plain_text(self) -> None:
        result = parse_joern_output("hello world")
        assert result == "hello world"

    def test_contextmine_result_marker(self) -> None:
        result = parse_joern_output(
            "prefix <contextmine_result> inner value </contextmine_result> suffix"
        )
        assert result == "inner value"

    def test_ansi_escape_stripped(self) -> None:
        result = parse_joern_output("\x1b[31m42\x1b[0m")
        assert result == 42

    def test_triple_quoted_json(self) -> None:
        result = parse_joern_output('"""[1,2,3]"""')
        assert result == [1, 2, 3]


class TestJoernClient:
    async def test_check_health_success(self) -> None:
        client = JoernClient("http://localhost:9090")
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("contextmine_core.joern.httpx.AsyncClient") as mock_cls:
            mock_ac = AsyncMock()
            mock_ac.get.return_value = mock_response
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_ac)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            assert await client.check_health() is True

    async def test_check_health_failure(self) -> None:
        client = JoernClient("http://localhost:9090")

        with patch("contextmine_core.joern.httpx.AsyncClient") as mock_cls:
            mock_ac = AsyncMock()
            mock_ac.get.side_effect = Exception("connection refused")
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_ac)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            assert await client.check_health() is False

    async def test_execute_query_success(self) -> None:
        client = JoernClient("http://localhost:9090")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True, "stdout": "result", "stderr": ""}

        with patch("contextmine_core.joern.httpx.AsyncClient") as mock_cls:
            mock_ac = AsyncMock()
            mock_ac.post.return_value = mock_response
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_ac)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await client.execute_query("cpg.method.l")
            assert result.success is True
            assert result.stdout == "result"

    async def test_execute_query_http_error(self) -> None:
        client = JoernClient("http://localhost:9090")
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("contextmine_core.joern.httpx.AsyncClient") as mock_cls:
            mock_ac = AsyncMock()
            mock_ac.post.return_value = mock_response
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_ac)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await client.execute_query("bad query")
            assert result.success is False
            assert "500" in result.stderr

    async def test_load_cpg(self) -> None:
        client = JoernClient("http://localhost:9090")
        with patch.object(client, "execute_query", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = JoernResponse(success=True, stdout="loaded", stderr="")
            result = await client.load_cpg("/path/to/cpg.bin")
            assert result.success is True
            mock_exec.assert_called_once()


# ============================================================================
# telemetry/setup.py
# ============================================================================


class TestTelemetrySetup:
    def test_get_tracer_returns_tracer(self) -> None:
        tracer = telemetry_setup.get_tracer("test")
        assert tracer is not None

    def test_get_meter_returns_meter(self) -> None:
        meter = telemetry_setup.get_meter("test")
        assert meter is not None

    async def test_shutdown_when_not_initialized(self) -> None:
        # Save and restore state
        orig = telemetry_setup._initialized
        telemetry_setup._initialized = False
        try:
            await telemetry_setup.shutdown_telemetry()  # Should be a no-op
        finally:
            telemetry_setup._initialized = orig

    def test_init_telemetry_disabled(self) -> None:
        """When OTEL is disabled, init_telemetry returns False."""
        orig = telemetry_setup._initialized
        telemetry_setup._initialized = False
        try:
            mock_settings = MagicMock()
            mock_settings.otel_enabled = False

            with patch("contextmine_core.settings.get_settings", return_value=mock_settings):
                result = telemetry_setup.init_telemetry()
                assert result is False
        finally:
            telemetry_setup._initialized = orig


# ============================================================================
# validation/connectors.py
# ============================================================================


class TestStateOf:
    def test_dict_with_phase(self) -> None:
        item = {"status": {"phase": "Succeeded"}}
        assert _state_of(item) == "Succeeded"

    def test_dict_with_condition(self) -> None:
        item = {"status": {"condition": "True"}}
        assert _state_of(item) == "True"

    def test_string_status(self) -> None:
        item = {"status": "running"}
        assert _state_of(item) == "running"

    def test_missing_status(self) -> None:
        assert _state_of({}) == ""

    def test_non_dict_input(self) -> None:
        assert _state_of("not a dict") == ""


class TestFetchArgoMetrics:
    async def test_no_url_returns_empty(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            # Ensure ARGO_API_URL is not set
            os.environ.pop("ARGO_API_URL", None)
            result = await fetch_argo_metrics()
            assert result == []

    async def test_returns_pass_rate(self) -> None:
        with patch.dict(os.environ, {"ARGO_API_URL": "http://argo/api/v1/workflows/default"}):
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "items": [
                    {"status": {"phase": "Succeeded"}},
                    {"status": {"phase": "Succeeded"}},
                    {"status": {"phase": "Failed"}},
                ]
            }
            mock_response.raise_for_status = MagicMock()

            with patch("contextmine_core.validation.connectors.httpx.AsyncClient") as mock_cls:
                mock_ac = AsyncMock()
                mock_ac.get.return_value = mock_response
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_ac)
                mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

                result = await fetch_argo_metrics()
                assert len(result) == 1
                assert result[0].source == "argo"
                assert result[0].key == "pass_rate"
                assert abs(result[0].value - 2 / 3) < 0.01


class TestFetchTektonMetrics:
    async def test_no_url_returns_empty(self) -> None:
        os.environ.pop("TEKTON_API_URL", None)
        result = await fetch_tekton_metrics()
        assert result == []


class TestFetchTemporalAlerts:
    async def test_no_url_returns_empty(self) -> None:
        os.environ.pop("TEMPORAL_API_URL", None)
        result = await fetch_temporal_alerts()
        assert result == []

    async def test_returns_alert_count(self) -> None:
        with patch.dict(os.environ, {"TEMPORAL_API_URL": "http://temporal/api"}):
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "alerts": [
                    {"state": "firing", "name": "high_latency"},
                    {"state": "resolved", "name": "old_alert"},
                ]
            }
            mock_response.raise_for_status = MagicMock()

            with patch("contextmine_core.validation.connectors.httpx.AsyncClient") as mock_cls:
                mock_ac = AsyncMock()
                mock_ac.get.return_value = mock_response
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_ac)
                mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

                result = await fetch_temporal_alerts()
                assert len(result) == 1
                assert result[0].value == 1.0
                assert result[0].status == "alert"


# ============================================================================
# validation/service.py
# ============================================================================


class TestValidationService:
    def test_to_kind(self) -> None:
        from contextmine_core.models import ValidationSourceKind

        assert _to_kind("argo") == ValidationSourceKind.ARGO
        assert _to_kind("tekton") == ValidationSourceKind.TEKTON
        assert _to_kind("temporal") == ValidationSourceKind.TEMPORAL

    async def test_refresh_with_no_metrics(self) -> None:
        session = AsyncMock()
        with (
            patch(
                "contextmine_core.validation.service.fetch_argo_metrics",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "contextmine_core.validation.service.fetch_tekton_metrics",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "contextmine_core.validation.service.fetch_temporal_alerts",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            count = await refresh_validation_snapshots(session, collection_id=None)
            assert count == 0

    async def test_refresh_persists_metrics(self) -> None:
        session = AsyncMock()
        metrics = [
            ValidationMetric(source="argo", key="pass_rate", value=0.9, status="ok"),
            ValidationMetric(source="tekton", key="pass_rate", value=1.0, status="ok"),
        ]
        with (
            patch(
                "contextmine_core.validation.service.fetch_argo_metrics",
                new_callable=AsyncMock,
                return_value=metrics[:1],
            ),
            patch(
                "contextmine_core.validation.service.fetch_tekton_metrics",
                new_callable=AsyncMock,
                return_value=metrics[1:],
            ),
            patch(
                "contextmine_core.validation.service.fetch_temporal_alerts",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            count = await refresh_validation_snapshots(session, collection_id=uuid.uuid4())
            assert count == 2
            assert session.add.call_count == 2


# ============================================================================
# graph/age.py
# ============================================================================


class TestAgeHelpers:
    def test_scenario_graph_name(self) -> None:
        sid = uuid.UUID("12345678-1234-1234-1234-123456789abc")
        name = scenario_graph_name(sid)
        assert name.startswith("twin_")
        assert "-" not in name

    def test_validate_graph_name_valid(self) -> None:
        _validate_graph_name("twin_abc123_def")

    def test_validate_graph_name_invalid(self) -> None:
        with pytest.raises(ValueError, match="Invalid graph name"):
            _validate_graph_name("DROP TABLE; --")

    def test_age_cypher_sql(self) -> None:
        sql = _age_cypher_sql("twin_abc", "MATCH (n) RETURN n")
        assert "cypher('twin_abc'" in sql
        assert "MATCH (n) RETURN n" in sql

    def test_ensure_read_only_cypher_match(self) -> None:
        ensure_read_only_cypher("MATCH (n) RETURN n")  # Should not raise

    def test_ensure_read_only_cypher_create_rejected(self) -> None:
        with pytest.raises(ValueError, match="read-only"):
            ensure_read_only_cypher("CREATE (n:Node)")

    def test_ensure_read_only_cypher_delete_rejected(self) -> None:
        with pytest.raises(ValueError, match="read-only"):
            ensure_read_only_cypher("MATCH (n) DELETE n")

    def test_esc(self) -> None:
        assert _esc("it's") == "it\\'s"
        assert _esc("a\\b") == "a\\\\b"

    def test_chunked(self) -> None:
        items = [1, 2, 3, 4, 5]
        batches = list(_chunked(items, 2))
        assert batches == [[1, 2], [3, 4], [5]]

    def test_chunked_empty(self) -> None:
        assert list(_chunked([], 10)) == []


# ============================================================================
# database.py
# ============================================================================


class TestDatabaseModule:
    def test_get_engine_raises_without_url(self) -> None:
        # Reset module state
        orig_engine = db_module._engine
        db_module._engine = None
        try:
            mock_settings = MagicMock()
            mock_settings.database_url = ""
            with (
                patch("contextmine_core.database.get_settings", return_value=mock_settings),
                pytest.raises(RuntimeError, match="DATABASE_URL"),
            ):
                db_module.get_engine()
        finally:
            db_module._engine = orig_engine

    async def test_close_engine_when_none(self) -> None:
        orig_engine = db_module._engine
        db_module._engine = None
        try:
            await db_module.close_engine()
        finally:
            db_module._engine = orig_engine

    async def test_close_engine_disposes(self) -> None:
        mock_engine = AsyncMock()
        orig_engine = db_module._engine
        orig_factory = db_module._session_factory
        db_module._engine = mock_engine
        db_module._session_factory = MagicMock()
        try:
            await db_module.close_engine()
            mock_engine.dispose.assert_called_once()
            assert db_module._engine is None
            assert db_module._session_factory is None
        finally:
            db_module._engine = orig_engine
            db_module._session_factory = orig_factory

    def test_get_async_session_alias(self) -> None:
        assert db_module.get_async_session is db_module.get_session


# ============================================================================
# embeddings.py
# ============================================================================


class TestFakeEmbedder:
    async def test_embed_batch(self) -> None:
        embedder = FakeEmbedder(dimension=64)
        result = await embedder.embed_batch(["hello", "world"])
        assert len(result.embeddings) == 2
        assert all(len(e) == 64 for e in result.embeddings)
        assert result.model_name == "fake-embedding-model"
        assert result.dimension == 64

    async def test_deterministic(self) -> None:
        embedder = FakeEmbedder(dimension=16)
        r1 = await embedder.embed_batch(["test text"])
        r2 = await embedder.embed_batch(["test text"])
        assert r1.embeddings[0] == r2.embeddings[0]

    async def test_embed_texts_compat(self) -> None:
        embedder = FakeEmbedder(dimension=8)
        vectors = await embedder.embed_texts(["a", "b"])
        assert len(vectors) == 2

    def test_properties(self) -> None:
        embedder = FakeEmbedder(dimension=1536)
        assert embedder.provider == EmbeddingProvider.OPENAI
        assert embedder.model_name == "fake-embedding-model"
        assert embedder.dimension == 1536


class TestOpenAIEmbedder:
    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown OpenAI model"):
            OpenAIEmbedder(model_name="gpt-4-turbo", api_key="sk-test")

    def test_missing_api_key_raises(self) -> None:
        mock_settings = MagicMock()
        mock_settings.openai_api_key = ""
        with (
            patch("contextmine_core.embeddings.get_settings", return_value=mock_settings),
            pytest.raises(ValueError, match="API key required"),
        ):
            OpenAIEmbedder(model_name="text-embedding-3-small")

    def test_properties(self) -> None:
        embedder = OpenAIEmbedder(model_name="text-embedding-3-small", api_key="sk-test")
        assert embedder.provider == EmbeddingProvider.OPENAI
        assert embedder.model_name == "text-embedding-3-small"
        assert embedder.dimension == 1536


class TestGeminiEmbedder:
    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown Gemini model"):
            GeminiEmbedder(model_name="gemini-pro", api_key="key")

    def test_missing_api_key_raises(self) -> None:
        mock_settings = MagicMock()
        mock_settings.gemini_api_key = ""
        with (
            patch("contextmine_core.embeddings.get_settings", return_value=mock_settings),
            pytest.raises(ValueError, match="API key required"),
        ):
            GeminiEmbedder(model_name="text-embedding-004")


class TestGetEmbedder:
    def test_openai_from_string(self) -> None:
        embedder = get_embedder("openai", api_key="sk-test")
        assert isinstance(embedder, OpenAIEmbedder)

    def test_gemini_from_enum(self) -> None:
        embedder = get_embedder(EmbeddingProvider.GEMINI, api_key="key")
        assert isinstance(embedder, GeminiEmbedder)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError):
            get_embedder("unknown_provider")


class TestParseEmbeddingModelSpec:
    def test_valid_spec(self) -> None:
        provider, model = parse_embedding_model_spec("openai:text-embedding-3-small")
        assert provider == EmbeddingProvider.OPENAI
        assert model == "text-embedding-3-small"

    def test_invalid_spec_no_colon(self) -> None:
        with pytest.raises(ValueError, match="Invalid model spec"):
            parse_embedding_model_spec("openai-text-embedding-3-small")


# ============================================================================
# graph/builder.py
# ============================================================================


class TestGraphBuilder:
    def test_no_treesitter_returns_empty_graph(self) -> None:
        builder = GraphBuilder(lsp_manager=None, treesitter_manager=None)
        assert not builder.has_treesitter
        assert not builder.has_lsp
        graph = builder.build_file_subgraph("file.py", content="def foo(): pass")
        assert isinstance(graph, CodeGraph)
        assert len(graph._nodes) == 0

    def test_resolve_symbol_at_without_treesitter(self) -> None:
        builder = GraphBuilder()
        result = builder.resolve_symbol_at("file.py", 10)
        assert result is None

    def test_build_multi_file_graph_empty(self) -> None:
        builder = GraphBuilder()
        graph = builder.build_multi_file_graph([])
        assert isinstance(graph, CodeGraph)

    async def test_add_definition_edges_without_lsp(self) -> None:
        builder = GraphBuilder()
        graph = CodeGraph()
        await builder.add_definition_edges(graph, "sym1", "file.py", 1, 0)
        assert len(graph._edges) == 0

    async def test_add_reference_edges_without_lsp(self) -> None:
        builder = GraphBuilder()
        graph = CodeGraph()
        await builder.add_reference_edges(graph, "sym1", "file.py", 1, 0)
        assert len(graph._edges) == 0


class TestGetGraphBuilder:
    def test_returns_builder_instance(self) -> None:
        with (
            patch("contextmine_core.lsp.manager.get_lsp_manager", side_effect=ImportError),
            patch(
                "contextmine_core.treesitter.manager.get_treesitter_manager",
                side_effect=ImportError,
            ),
        ):
            from contextmine_core.graph.builder import get_graph_builder

            builder = get_graph_builder()
            assert isinstance(builder, GraphBuilder)


# ============================================================================
# exports/twin_manifest.py
# ============================================================================


class TestTwinManifest:
    async def test_export_twin_manifest_structure(self) -> None:
        mock_session = AsyncMock()
        scenario_id = uuid.uuid4()

        mock_graph = {
            "nodes": [
                {
                    "id": str(uuid.uuid4()),
                    "kind": "api_endpoint",
                    "name": "GET /users",
                    "natural_key": "/users",
                    "meta": {},
                },
                {
                    "id": str(uuid.uuid4()),
                    "kind": "function",
                    "name": "handler",
                    "natural_key": "handler",
                    "meta": {},
                },
            ],
            "edges": [],
        }
        mock_arch_graph = {"nodes": [], "edges": []}

        with (
            patch(
                "contextmine_core.exports.twin_manifest.get_full_scenario_graph",
                new_callable=AsyncMock,
            ) as mock_gfsg,
            patch(
                "contextmine_core.exports.twin_manifest.build_ui_map_projection",
                return_value={"summary": {}, "graph": {}},
            ),
            patch(
                "contextmine_core.exports.twin_manifest.build_test_matrix_projection",
                return_value={"summary": {}, "matrix": [], "graph": {}},
            ),
            patch(
                "contextmine_core.exports.twin_manifest.build_user_flows_projection",
                return_value={"summary": {}, "flows": [], "graph": {}},
            ),
            patch(
                "contextmine_core.exports.twin_manifest.compute_rebuild_readiness",
                return_value={"known_gaps": []},
            ),
        ):
            mock_gfsg.side_effect = [mock_graph, mock_arch_graph]

            from contextmine_core.exports.twin_manifest import export_twin_manifest

            result = await export_twin_manifest(mock_session, scenario_id)

            parsed = json.loads(result)
            assert parsed["manifest_version"] == "1.0"
            assert "sections" in parsed
            assert len(parsed["sections"]["interfaces"]) == 1  # Only api_endpoint


# ============================================================================
# Semantic snapshot indexers (python, java, typescript, php)
# ============================================================================


class TestPythonIndexerBackend:
    def test_can_handle_python(self) -> None:
        from contextmine_core.semantic_snapshot.indexers.python import PythonIndexerBackend

        backend = PythonIndexerBackend()
        target = ProjectTarget(language=Language.PYTHON, root_path=Path("/tmp/proj"))
        assert backend.can_handle(target) is True

    def test_cannot_handle_java(self) -> None:
        from contextmine_core.semantic_snapshot.indexers.python import PythonIndexerBackend

        backend = PythonIndexerBackend()
        target = ProjectTarget(language=Language.JAVA, root_path=Path("/tmp/proj"))
        assert backend.can_handle(target) is False

    def test_build_command(self) -> None:
        from contextmine_core.semantic_snapshot.indexers.python import PythonIndexerBackend

        backend = PythonIndexerBackend()
        target = ProjectTarget(language=Language.PYTHON, root_path=Path("/tmp/proj"))
        cfg = IndexConfig(project_name="myproj", project_version="1.0")
        cmd = backend._build_command(target, cfg)
        assert "scip-python" in cmd
        assert "--project-name" in cmd
        assert "myproj" in cmd
        assert "--project-version" in cmd


class TestJavaIndexerBackend:
    def test_can_handle_java(self) -> None:
        from contextmine_core.semantic_snapshot.indexers.java import JavaIndexerBackend

        backend = JavaIndexerBackend()
        target = ProjectTarget(language=Language.JAVA, root_path=Path("/tmp/proj"))
        assert backend.can_handle(target) is True

    def test_cannot_handle_python(self) -> None:
        from contextmine_core.semantic_snapshot.indexers.java import JavaIndexerBackend

        backend = JavaIndexerBackend()
        target = ProjectTarget(language=Language.PYTHON, root_path=Path("/tmp/proj"))
        assert backend.can_handle(target) is False

    def test_build_command(self) -> None:
        from contextmine_core.semantic_snapshot.indexers.java import JavaIndexerBackend

        backend = JavaIndexerBackend()
        target = ProjectTarget(language=Language.JAVA, root_path=Path("/tmp/proj"))
        cfg = IndexConfig()
        cmd = backend._build_command(target, Path("/tmp/out/proj.scip"), cfg)
        assert "scip-java" in cmd
        assert "--output" in cmd

    def test_build_command_with_java_build_args(self) -> None:
        from contextmine_core.semantic_snapshot.indexers.java import JavaIndexerBackend

        backend = JavaIndexerBackend()
        target = ProjectTarget(language=Language.JAVA, root_path=Path("/tmp/proj"))
        cfg = IndexConfig(java_build_args=["clean", "build"])
        cmd = backend._build_command(target, Path("/tmp/out/proj.scip"), cfg)
        assert "--" in cmd
        assert "clean" in cmd


class TestTypescriptIndexerBackend:
    def test_can_handle_typescript(self) -> None:
        from contextmine_core.semantic_snapshot.indexers.typescript import TypescriptIndexerBackend

        backend = TypescriptIndexerBackend()
        target = ProjectTarget(language=Language.TYPESCRIPT, root_path=Path("/tmp/proj"))
        assert backend.can_handle(target) is True

    def test_can_handle_javascript(self) -> None:
        from contextmine_core.semantic_snapshot.indexers.typescript import TypescriptIndexerBackend

        backend = TypescriptIndexerBackend()
        target = ProjectTarget(language=Language.JAVASCRIPT, root_path=Path("/tmp/proj"))
        assert backend.can_handle(target) is True

    def test_cannot_handle_java(self) -> None:
        from contextmine_core.semantic_snapshot.indexers.typescript import TypescriptIndexerBackend

        backend = TypescriptIndexerBackend()
        target = ProjectTarget(language=Language.JAVA, root_path=Path("/tmp/proj"))
        assert backend.can_handle(target) is False

    def test_build_command_typescript(self) -> None:
        from contextmine_core.semantic_snapshot.indexers.typescript import TypescriptIndexerBackend

        backend = TypescriptIndexerBackend()
        target = ProjectTarget(language=Language.TYPESCRIPT, root_path=Path("/tmp/proj"))
        cmd, generated = backend._build_command(target, Path("/tmp/out/proj.scip"))
        assert "scip-typescript" in cmd
        assert generated is None  # No generated config for TS

    def test_build_command_javascript_creates_config(self, tmp_path: Path) -> None:
        from contextmine_core.semantic_snapshot.indexers.typescript import TypescriptIndexerBackend

        backend = TypescriptIndexerBackend()
        target = ProjectTarget(language=Language.JAVASCRIPT, root_path=tmp_path)
        cmd, generated = backend._build_command(target, tmp_path / "out.scip")
        assert generated is not None
        assert generated.exists()
        content = json.loads(generated.read_text())
        assert content["compilerOptions"]["allowJs"] is True
        generated.unlink()

    def test_should_install_deps_never(self) -> None:
        from contextmine_core.semantic_snapshot.indexers.typescript import TypescriptIndexerBackend
        from contextmine_core.semantic_snapshot.models import InstallDepsMode

        backend = TypescriptIndexerBackend()
        target = ProjectTarget(language=Language.TYPESCRIPT, root_path=Path("/tmp/proj"))
        cfg = IndexConfig(install_deps_mode=InstallDepsMode.NEVER)
        assert backend._should_install_deps(target, cfg) is False

    def test_should_install_deps_always(self) -> None:
        from contextmine_core.semantic_snapshot.indexers.typescript import TypescriptIndexerBackend
        from contextmine_core.semantic_snapshot.models import InstallDepsMode

        backend = TypescriptIndexerBackend()
        target = ProjectTarget(language=Language.TYPESCRIPT, root_path=Path("/tmp/proj"))
        cfg = IndexConfig(install_deps_mode=InstallDepsMode.ALWAYS)
        assert backend._should_install_deps(target, cfg) is True

    def test_should_install_deps_auto_no_node_modules(self, tmp_path: Path) -> None:
        from contextmine_core.semantic_snapshot.indexers.typescript import TypescriptIndexerBackend
        from contextmine_core.semantic_snapshot.models import InstallDepsMode

        backend = TypescriptIndexerBackend()
        target = ProjectTarget(language=Language.TYPESCRIPT, root_path=tmp_path)
        cfg = IndexConfig(install_deps_mode=InstallDepsMode.AUTO)
        assert backend._should_install_deps(target, cfg) is True

    def test_should_install_deps_auto_has_node_modules(self, tmp_path: Path) -> None:
        from contextmine_core.semantic_snapshot.indexers.typescript import TypescriptIndexerBackend
        from contextmine_core.semantic_snapshot.models import InstallDepsMode

        backend = TypescriptIndexerBackend()
        (tmp_path / "node_modules").mkdir()
        target = ProjectTarget(language=Language.TYPESCRIPT, root_path=tmp_path)
        cfg = IndexConfig(install_deps_mode=InstallDepsMode.AUTO)
        assert backend._should_install_deps(target, cfg) is False


class TestPhpIndexerBackend:
    def test_can_handle_php(self) -> None:
        from contextmine_core.semantic_snapshot.indexers.php import PhpIndexerBackend

        backend = PhpIndexerBackend()
        target = ProjectTarget(language=Language.PHP, root_path=Path("/tmp/proj"))
        assert backend.can_handle(target) is True

    def test_cannot_handle_python(self) -> None:
        from contextmine_core.semantic_snapshot.indexers.php import PhpIndexerBackend

        backend = PhpIndexerBackend()
        target = ProjectTarget(language=Language.PYTHON, root_path=Path("/tmp/proj"))
        assert backend.can_handle(target) is False

    def test_resolve_vendor_dir_default(self, tmp_path: Path) -> None:
        from contextmine_core.semantic_snapshot.indexers.php import PhpIndexerBackend

        backend = PhpIndexerBackend()
        # No composer.json
        assert backend._resolve_vendor_dir(tmp_path) == Path("vendor")

    def test_resolve_vendor_dir_custom(self, tmp_path: Path) -> None:
        from contextmine_core.semantic_snapshot.indexers.php import PhpIndexerBackend

        backend = PhpIndexerBackend()
        (tmp_path / "composer.json").write_text(json.dumps({"config": {"vendor-dir": "libs"}}))
        assert backend._resolve_vendor_dir(tmp_path) == Path("libs")

    def test_resolve_tool_path_not_found(self) -> None:
        from contextmine_core.semantic_snapshot.indexers.php import PhpIndexerBackend

        backend = PhpIndexerBackend()
        with patch("shutil.which", return_value=None):
            # Also mock fallback paths
            backend._resolve_tool_path()
            # Will return None or a fallback path if one exists on system
            # We just verify it doesn't crash

    def test_should_install_deps_never(self) -> None:
        from contextmine_core.semantic_snapshot.indexers.php import PhpIndexerBackend
        from contextmine_core.semantic_snapshot.models import InstallDepsMode

        backend = PhpIndexerBackend()
        target = ProjectTarget(language=Language.PHP, root_path=Path("/tmp/proj"))
        cfg = IndexConfig(install_deps_mode=InstallDepsMode.NEVER)
        assert backend._should_install_deps(target, cfg) is False

    def test_should_install_deps_with_force_flag(self) -> None:
        from contextmine_core.semantic_snapshot.indexers.php import PhpIndexerBackend
        from contextmine_core.semantic_snapshot.models import InstallDepsMode

        backend = PhpIndexerBackend()
        target = ProjectTarget(
            language=Language.PHP,
            root_path=Path("/tmp/proj"),
            metadata={"force_install_deps": True},
        )
        cfg = IndexConfig(install_deps_mode=InstallDepsMode.NEVER)
        assert backend._should_install_deps(target, cfg) is True


# ============================================================================
# semantic_snapshot/indexers/language_census.py
# ============================================================================


class TestLanguageCensusHelpers:
    def test_normalize_language_name(self) -> None:
        assert _normalize_language_name("Python") == Language.PYTHON
        assert _normalize_language_name("typescript") == Language.TYPESCRIPT
        assert _normalize_language_name("Rust") is None
        assert _normalize_language_name("") is None

    def test_language_from_extension(self) -> None:
        assert _language_from_extension(".py") == Language.PYTHON
        assert _language_from_extension(".ts") == Language.TYPESCRIPT
        assert _language_from_extension(".tsx") == Language.TYPESCRIPT
        assert _language_from_extension(".java") == Language.JAVA
        assert _language_from_extension(".rs") is None

    def test_is_ignored_relative_path(self) -> None:
        assert _is_ignored_relative_path(Path("node_modules/foo.js"), set())
        assert _is_ignored_relative_path(Path(".git/config"), set())
        assert not _is_ignored_relative_path(Path("src/main.py"), set())
        assert _is_ignored_relative_path(Path("src/libs/foo.py"), {Path("src/libs")})

    def test_build_not_match_dir_regex(self) -> None:
        result = _build_not_match_dir_regex({Path("src/libs"), Path("vendor")})
        assert result is not None
        assert "src/libs" in result

    def test_build_not_match_dir_regex_empty(self) -> None:
        assert _build_not_match_dir_regex(set()) is None

    def test_parse_cloc_summary(self) -> None:
        data = {
            "header": {"n_files": 10},
            "Python": {"nFiles": 5, "code": 1000, "comment": 200, "blank": 100},
            "Java": {"nFiles": 3, "code": 500, "comment": 50, "blank": 30},
            "SUM": {"nFiles": 8, "code": 1500},
        }
        entries = _parse_cloc_summary(data)
        assert Language.PYTHON in entries
        assert entries[Language.PYTHON].files == 5
        assert entries[Language.PYTHON].code == 1000
        assert Language.JAVA in entries


class TestLanguageCensusReport:
    def test_total_code(self) -> None:
        report = LanguageCensusReport()
        report.entries = {
            Language.PYTHON: LanguageCensusEntry(language=Language.PYTHON, code=100),
            Language.JAVA: LanguageCensusEntry(language=Language.JAVA, code=200),
        }
        assert report.total_code == 300


class TestFallbackExtensionCensus:
    def test_scans_directory(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("print('hello')\nprint('world')\n")
        (tmp_path / "app.ts").write_text("const x = 1;\n")
        (tmp_path / "README.md").write_text("# Docs\n")  # Not in EXTENSION_TO_LANGUAGE

        report = _fallback_extension_census(tmp_path)
        assert Language.PYTHON in report.entries
        assert Language.TYPESCRIPT in report.entries
        assert report.entries[Language.PYTHON].files == 1
        assert report.tool_name == "extension-fallback"


class TestBuildLanguageCensus:
    def test_with_cloc_unavailable(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("x = 1\n")
        with patch("shutil.which", return_value=None):
            report = build_language_census(tmp_path)
            assert (
                "cloc_not_available" in report.warnings
                or "cloc_unavailable_or_unusable_using_extension_fallback" in report.warnings
            )
