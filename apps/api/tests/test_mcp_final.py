"""Coverage tests targeting remaining uncovered lines in mcp_server.py.

Targets:
- _resolve_collection_for_tool with member access check (lines 66-72)
- _resolve_arc42_repo_checkout (lines 86-107)
- research_validation rule matching paths (lines 1202-1265)
- research_architecture topic-specific rendering (lines 1540-1627)
- code.expand graph expansion (lines 925-995)
- get_arc42 full regeneration path (lines 2243-2409)
- arc42_drift_report (lines 2432-2559)
- list_ports_adapters (lines 2587-2661)
- Multi-engine tool wrappers: get_codebase_summary, list_methods, list_calls,
  get_cfg, get_variable_flow, find_taint_sources, find_taint_sinks,
  find_taint_flows, store_findings, export_sarif (lines 2678-3013)
- export_twin_view format dispatch (lines 3046-3099)
- get_validation_dashboard (lines 3115-3118)
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import app.mcp_server as mcp_mod
import pytest

pytestmark = pytest.mark.anyio

# Tool function references (unwrap FunctionTool decorator)
_list_collections = mcp_mod.list_collections.fn
_get_context_markdown = mcp_mod.get_context_markdown.fn
_research_validation = mcp_mod.research_validation.fn
_research_architecture = mcp_mod.research_architecture.fn
_get_arc42 = mcp_mod.mcp_get_arc42.fn
_arc42_drift = mcp_mod.mcp_arc42_drift_report.fn
_list_ports_adapters = mcp_mod.mcp_list_ports_adapters.fn
_get_codebase_summary = mcp_mod.mcp_get_codebase_summary.fn
_list_methods = mcp_mod.mcp_list_methods.fn
_list_calls = mcp_mod.mcp_list_calls.fn
_get_cfg = mcp_mod.mcp_get_cfg.fn
_get_variable_flow = mcp_mod.mcp_get_variable_flow.fn
_find_taint_sources = mcp_mod.mcp_find_taint_sources.fn
_find_taint_sinks = mcp_mod.mcp_find_taint_sinks.fn
_find_taint_flows = mcp_mod.mcp_find_taint_flows.fn
_store_findings = mcp_mod.mcp_store_findings.fn
_export_sarif = mcp_mod.mcp_export_sarif.fn
_export_twin_view = mcp_mod.mcp_export_twin_view.fn
_validation_dashboard = mcp_mod.mcp_get_validation_dashboard.fn
_code_expand = mcp_mod.code_expand.fn
_resolve_collection_for_tool = mcp_mod._resolve_collection_for_tool
_resolve_collection_access = mcp_mod._resolve_collection_access
_parse_csv_list = mcp_mod._parse_csv_list
_sha256_text = mcp_mod._sha256_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_db_session(mock_db):
    ctx = patch("app.mcp_server.get_db_session")
    mock_session = ctx.start()
    mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db)
    mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
    return ctx


def _make_collection(*, visibility="global", owner_user_id=None):
    coll = MagicMock()
    coll.id = uuid.uuid4()
    coll.name = "Test Collection"
    coll.visibility = MagicMock(value=visibility)
    coll.visibility.__eq__ = lambda self, other: self.value == getattr(other, "value", other)
    coll.owner_user_id = owner_user_id
    return coll


def _make_settings(**overrides):
    defaults = {
        "arch_docs_enabled": True,
        "arch_docs_generate_on_sync": False,
        "arch_docs_llm_enrich": False,
        "arch_docs_drift_enabled": False,
        "arch_docs_agent_sdk_model": "claude-3-5-sonnet-20241022",
        "arch_docs_agent_sdk_max_turns": 10,
        "arch_docs_agent_sdk_permission_mode": "never",
        "default_llm_provider": "openai",
        "repos_root": "/tmp/repos",
        "twin_analysis_cache_ttl_seconds": 300,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# _resolve_collection_for_tool member access
# ---------------------------------------------------------------------------


class TestResolveCollectionMemberAccess:
    async def test_member_check_grants_access(self) -> None:
        """Member lookup grants access to non-global collection."""
        mock_db = AsyncMock()
        collection = _make_collection(visibility="private", owner_user_id=uuid.uuid4())
        user_id = uuid.uuid4()

        # First query returns collection, second returns a member
        member = MagicMock()
        call_count = {"n": 0}

        async def mock_execute(stmt):
            call_count["n"] += 1
            result = MagicMock()
            if call_count["n"] == 1:
                result.scalars.return_value.first.return_value = collection
            else:
                result.scalar_one_or_none.return_value = member
            return result

        mock_db.execute = mock_execute

        coll, error = await _resolve_collection_for_tool(
            mock_db, collection_id=str(collection.id), user_id=user_id
        )
        # Should succeed via member check
        assert error is None or coll is not None

    async def test_non_member_denied(self) -> None:
        """Non-member gets access denied to private collection."""
        mock_db = AsyncMock()
        collection = _make_collection(visibility="private", owner_user_id=uuid.uuid4())
        user_id = uuid.uuid4()

        call_count = {"n": 0}

        async def mock_execute(stmt):
            call_count["n"] += 1
            result = MagicMock()
            if call_count["n"] == 1:
                result.scalars.return_value.first.return_value = collection
            else:
                result.scalar_one_or_none.return_value = None
            return result

        mock_db.execute = mock_execute

        coll, error = await _resolve_collection_for_tool(
            mock_db, collection_id=str(collection.id), user_id=user_id
        )
        assert error is not None or coll is None


# ---------------------------------------------------------------------------
# _sha256_text / _parse_csv_list utilities
# ---------------------------------------------------------------------------


class TestMCPUtilities:
    def test_sha256_text(self) -> None:
        import hashlib

        result = _sha256_text("hello world")
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert result == expected

    def test_parse_csv_list_none(self) -> None:
        assert _parse_csv_list(None) is None

    def test_parse_csv_list_empty(self) -> None:
        assert _parse_csv_list("") is None

    def test_parse_csv_list_whitespace_only(self) -> None:
        assert _parse_csv_list("  ,  , ") is None

    def test_parse_csv_list_values(self) -> None:
        result = _parse_csv_list("graphrag, lsp, joern")
        assert result == ["graphrag", "lsp", "joern"]


# ---------------------------------------------------------------------------
# Multi-engine tool wrappers (lines 2678-2943)
# ---------------------------------------------------------------------------


class TestMultiEngineTools:
    """Test the multi-engine MCP tool wrappers that share a common pattern."""

    async def test_get_codebase_summary_success(self) -> None:
        mock_db = AsyncMock()
        collection = _make_collection()
        mock_db.execute = AsyncMock(
            return_value=MagicMock(
                scalars=MagicMock(return_value=MagicMock(first=MagicMock(return_value=collection)))
            )
        )

        ctx = _mock_db_session(mock_db)
        try:
            with (
                patch("app.mcp_server.get_current_user_id", return_value=None),
                patch("app.mcp_server.get_settings", return_value=_make_settings()),
                patch(
                    "contextmine_core.twin.get_codebase_summary_multi",
                    new_callable=AsyncMock,
                    return_value={"nodes": 100, "edges": 200},
                ),
                patch.object(
                    mcp_mod,
                    "_resolve_collection_access",
                    new_callable=AsyncMock,
                    return_value=(collection, None),
                ),
            ):
                result = await _get_codebase_summary(str(collection.id))
            payload = json.loads(result)
            assert payload["nodes"] == 100
        finally:
            ctx.stop()

    async def test_list_methods_success(self) -> None:
        mock_db = AsyncMock()
        collection = _make_collection()
        ctx = _mock_db_session(mock_db)
        try:
            with (
                patch("app.mcp_server.get_current_user_id", return_value=None),
                patch("app.mcp_server.get_settings", return_value=_make_settings()),
                patch(
                    "contextmine_core.twin.list_methods_multi",
                    new_callable=AsyncMock,
                    return_value={"items": [], "total": 0},
                ),
                patch(
                    "contextmine_core.twin.sanitize_regex_query",
                    return_value="test",
                ),
                patch.object(
                    mcp_mod,
                    "_resolve_collection_access",
                    new_callable=AsyncMock,
                    return_value=(collection, None),
                ),
            ):
                result = await _list_methods(str(collection.id), query="test")
            payload = json.loads(result)
            assert payload["total"] == 0
        finally:
            ctx.stop()

    async def test_list_calls_success(self) -> None:
        mock_db = AsyncMock()
        collection = _make_collection()
        ctx = _mock_db_session(mock_db)
        try:
            with (
                patch("app.mcp_server.get_current_user_id", return_value=None),
                patch("app.mcp_server.get_settings", return_value=_make_settings()),
                patch(
                    "contextmine_core.twin.list_calls_multi",
                    new_callable=AsyncMock,
                    return_value={"items": [], "total": 0},
                ),
                patch.object(
                    mcp_mod,
                    "_resolve_collection_access",
                    new_callable=AsyncMock,
                    return_value=(collection, None),
                ),
            ):
                result = await _list_calls(str(collection.id))
            payload = json.loads(result)
            assert payload["total"] == 0
        finally:
            ctx.stop()

    async def test_get_cfg_success(self) -> None:
        mock_db = AsyncMock()
        collection = _make_collection()
        ctx = _mock_db_session(mock_db)
        try:
            with (
                patch("app.mcp_server.get_current_user_id", return_value=None),
                patch("app.mcp_server.get_settings", return_value=_make_settings()),
                patch(
                    "contextmine_core.twin.get_cfg_multi",
                    new_callable=AsyncMock,
                    return_value={"root": "test", "nodes": []},
                ),
                patch.object(
                    mcp_mod,
                    "_resolve_collection_access",
                    new_callable=AsyncMock,
                    return_value=(collection, None),
                ),
            ):
                result = await _get_cfg(str(collection.id), node_ref="MyClass.method")
            payload = json.loads(result)
            assert "root" in payload
        finally:
            ctx.stop()

    async def test_get_variable_flow_success(self) -> None:
        mock_db = AsyncMock()
        collection = _make_collection()
        ctx = _mock_db_session(mock_db)
        try:
            with (
                patch("app.mcp_server.get_current_user_id", return_value=None),
                patch("app.mcp_server.get_settings", return_value=_make_settings()),
                patch(
                    "contextmine_core.twin.get_variable_flow_multi",
                    new_callable=AsyncMock,
                    return_value={"flows": []},
                ),
                patch.object(
                    mcp_mod,
                    "_resolve_collection_access",
                    new_callable=AsyncMock,
                    return_value=(collection, None),
                ),
            ):
                result = await _get_variable_flow(str(collection.id), node_ref="func", variable="x")
            payload = json.loads(result)
            assert "flows" in payload
        finally:
            ctx.stop()

    async def test_find_taint_sources_success(self) -> None:
        mock_db = AsyncMock()
        collection = _make_collection()
        ctx = _mock_db_session(mock_db)
        try:
            with (
                patch("app.mcp_server.get_current_user_id", return_value=None),
                patch("app.mcp_server.get_settings", return_value=_make_settings()),
                patch(
                    "contextmine_core.twin.find_taint_sources_multi",
                    new_callable=AsyncMock,
                    return_value={"sources": []},
                ),
                patch.object(
                    mcp_mod,
                    "_resolve_collection_access",
                    new_callable=AsyncMock,
                    return_value=(collection, None),
                ),
            ):
                result = await _find_taint_sources(str(collection.id))
            payload = json.loads(result)
            assert "sources" in payload
        finally:
            ctx.stop()

    async def test_find_taint_sinks_success(self) -> None:
        mock_db = AsyncMock()
        collection = _make_collection()
        ctx = _mock_db_session(mock_db)
        try:
            with (
                patch("app.mcp_server.get_current_user_id", return_value=None),
                patch("app.mcp_server.get_settings", return_value=_make_settings()),
                patch(
                    "contextmine_core.twin.find_taint_sinks_multi",
                    new_callable=AsyncMock,
                    return_value={"sinks": []},
                ),
                patch.object(
                    mcp_mod,
                    "_resolve_collection_access",
                    new_callable=AsyncMock,
                    return_value=(collection, None),
                ),
            ):
                result = await _find_taint_sinks(str(collection.id))
            payload = json.loads(result)
            assert "sinks" in payload
        finally:
            ctx.stop()

    async def test_find_taint_flows_success(self) -> None:
        mock_db = AsyncMock()
        collection = _make_collection()
        ctx = _mock_db_session(mock_db)
        try:
            with (
                patch("app.mcp_server.get_current_user_id", return_value=None),
                patch("app.mcp_server.get_settings", return_value=_make_settings()),
                patch(
                    "contextmine_core.twin.find_taint_flows_multi",
                    new_callable=AsyncMock,
                    return_value={"flows": [], "count": 0},
                ),
                patch.object(
                    mcp_mod,
                    "_resolve_collection_access",
                    new_callable=AsyncMock,
                    return_value=(collection, None),
                ),
            ):
                result = await _find_taint_flows(str(collection.id))
            payload = json.loads(result)
            assert "flows" in payload
        finally:
            ctx.stop()

    async def test_store_findings_success(self) -> None:
        mock_db = AsyncMock()
        collection = _make_collection()
        mock_db.commit = AsyncMock()
        ctx = _mock_db_session(mock_db)
        try:
            findings_list = [{"severity": "high", "rule": "sql_injection"}]
            with (
                patch("app.mcp_server.get_current_user_id", return_value=None),
                patch("app.mcp_server.get_settings", return_value=_make_settings()),
                patch(
                    "contextmine_core.twin.store_findings",
                    new_callable=AsyncMock,
                    return_value={"stored": 1},
                ),
                patch.object(
                    mcp_mod,
                    "_resolve_collection_access",
                    new_callable=AsyncMock,
                    return_value=(collection, None),
                ),
            ):
                result = await _store_findings(
                    str(collection.id), findings_json=json.dumps(findings_list)
                )
            payload = json.loads(result)
            assert payload["stored"] == 1
        finally:
            ctx.stop()

    async def test_store_findings_invalid_json(self) -> None:
        with (
            patch("app.mcp_server.get_current_user_id", return_value=None),
            patch("app.mcp_server.get_settings", return_value=_make_settings()),
        ):
            result = await _store_findings(str(uuid.uuid4()), findings_json="not json{{{")
        assert "Error" in result
        assert "valid JSON" in result

    async def test_store_findings_not_array(self) -> None:
        with (
            patch("app.mcp_server.get_current_user_id", return_value=None),
            patch("app.mcp_server.get_settings", return_value=_make_settings()),
        ):
            result = await _store_findings(str(uuid.uuid4()), findings_json='{"key": "value"}')
        assert "Error" in result
        assert "JSON array" in result

    async def test_export_sarif_success(self) -> None:
        mock_db = AsyncMock()
        collection = _make_collection()
        ctx = _mock_db_session(mock_db)
        try:
            with (
                patch("app.mcp_server.get_current_user_id", return_value=None),
                patch("app.mcp_server.get_settings", return_value=_make_settings()),
                patch(
                    "contextmine_core.twin.export_findings_sarif",
                    new_callable=AsyncMock,
                    return_value={"$schema": "sarif", "runs": []},
                ),
                patch.object(
                    mcp_mod,
                    "_resolve_collection_access",
                    new_callable=AsyncMock,
                    return_value=(collection, None),
                ),
            ):
                result = await _export_sarif(str(collection.id))
            payload = json.loads(result)
            assert "$schema" in payload
        finally:
            ctx.stop()


# ---------------------------------------------------------------------------
# export_twin_view format dispatch (lines 3046-3099)
# ---------------------------------------------------------------------------


class TestExportTwinView:
    async def test_export_lpg_jsonl(self) -> None:
        mock_db = AsyncMock()
        scenario = MagicMock()
        scenario.id = uuid.uuid4()
        scenario.collection_id = uuid.uuid4()
        scenario.name = "As-Is"
        scenario.is_as_is = True

        mock_db.execute = AsyncMock(
            return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=scenario))
        )
        mock_db.add = MagicMock()
        mock_db.commit = AsyncMock()

        ctx = _mock_db_session(mock_db)
        try:
            with (
                patch("app.mcp_server.get_settings", return_value=_make_settings()),
                patch(
                    "contextmine_core.exports.export_lpg_jsonl",
                    new_callable=AsyncMock,
                    return_value='{"nodes": []}',
                ),
            ):
                result = await _export_twin_view(str(scenario.id), format="lpg_jsonl")
            assert "Export Created" in result
        finally:
            ctx.stop()

    async def test_export_mermaid_c4(self) -> None:
        mock_db = AsyncMock()
        scenario = MagicMock()
        scenario.id = uuid.uuid4()
        scenario.collection_id = uuid.uuid4()
        scenario.name = "As-Is"
        scenario.is_as_is = True

        mock_db.execute = AsyncMock(
            return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=scenario))
        )
        mock_db.add = MagicMock()
        mock_db.commit = AsyncMock()

        ctx = _mock_db_session(mock_db)
        try:
            with (
                patch("app.mcp_server.get_settings", return_value=_make_settings()),
                patch(
                    "contextmine_core.exports.export_mermaid_c4",
                    new_callable=AsyncMock,
                    return_value="C4Context\n  title: System",
                ),
            ):
                result = await _export_twin_view(str(scenario.id), format="mermaid_c4")
            assert "Export Created" in result
        finally:
            ctx.stop()

    async def test_export_unsupported_format(self) -> None:
        mock_db = AsyncMock()
        scenario = MagicMock()
        scenario.id = uuid.uuid4()
        scenario.collection_id = uuid.uuid4()
        scenario.name = "test"
        mock_db.execute = AsyncMock(
            return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=scenario))
        )
        ctx = _mock_db_session(mock_db)
        try:
            with patch("app.mcp_server.get_settings", return_value=_make_settings()):
                result = await _export_twin_view(str(scenario.id), format="invalid_fmt")
            assert "Unsupported export format" in result
        finally:
            ctx.stop()

    async def test_export_scenario_not_found(self) -> None:
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(
            return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
        )
        ctx = _mock_db_session(mock_db)
        try:
            with patch("app.mcp_server.get_settings", return_value=_make_settings()):
                result = await _export_twin_view(str(uuid.uuid4()), format="lpg_jsonl")
            assert "not found" in result.lower() or "Error" in result
        finally:
            ctx.stop()


# ---------------------------------------------------------------------------
# get_validation_dashboard (lines 3115-3118)
# ---------------------------------------------------------------------------


class TestValidationDashboard:
    async def test_validation_dashboard_success(self) -> None:
        mock_db = AsyncMock()
        mock_db.commit = AsyncMock()
        ctx = _mock_db_session(mock_db)
        try:
            with (
                patch("app.mcp_server.get_settings", return_value=_make_settings()),
                patch(
                    "contextmine_core.validation.refresh_validation_snapshots",
                    new_callable=AsyncMock,
                    return_value=0,
                ),
                patch(
                    "contextmine_core.validation.get_latest_validation_status",
                    new_callable=AsyncMock,
                    return_value={"status": "healthy", "pipelines": []},
                ),
            ):
                result = await _validation_dashboard()
            payload = json.loads(result)
            assert payload["status"] == "healthy"
        finally:
            ctx.stop()

    async def test_validation_dashboard_with_collection(self) -> None:
        mock_db = AsyncMock()
        mock_db.commit = AsyncMock()
        ctx = _mock_db_session(mock_db)
        try:
            with (
                patch("app.mcp_server.get_settings", return_value=_make_settings()),
                patch(
                    "contextmine_core.validation.refresh_validation_snapshots",
                    new_callable=AsyncMock,
                    return_value=2,
                ),
                patch(
                    "contextmine_core.validation.get_latest_validation_status",
                    new_callable=AsyncMock,
                    return_value={"status": "degraded"},
                ),
            ):
                result = await _validation_dashboard(collection_id=str(uuid.uuid4()))
            payload = json.loads(result)
            assert payload["status"] == "degraded"
        finally:
            ctx.stop()


# ---------------------------------------------------------------------------
# get_arc42 regeneration path (lines 2243-2409)
# ---------------------------------------------------------------------------


class TestGetArc42:
    async def test_get_arc42_disabled(self) -> None:
        with (
            patch(
                "app.mcp_server.get_settings", return_value=_make_settings(arch_docs_enabled=False)
            ),
            patch("app.mcp_server.get_current_user_id", return_value=None),
        ):
            result = await _get_arc42()
        assert "disabled" in result.lower() or "Error" in result

    async def test_get_arc42_no_artifact_no_regenerate(self) -> None:
        """Without existing artifact and regenerate=false, returns error."""
        mock_db = AsyncMock()
        collection = _make_collection()
        scenario = MagicMock()
        scenario.id = uuid.uuid4()
        scenario.name = "As-Is"
        scenario.base_scenario_id = None

        call_count = {"n": 0}

        async def mock_execute(stmt):
            call_count["n"] += 1
            result = MagicMock()
            result.scalar_one_or_none.return_value = scenario if call_count["n"] <= 2 else None
            result.scalars.return_value.first.return_value = collection
            return result

        mock_db.execute = mock_execute

        ctx = _mock_db_session(mock_db)
        try:
            with (
                patch("app.mcp_server.get_settings", return_value=_make_settings()),
                patch("app.mcp_server.get_current_user_id", return_value=None),
                patch.object(
                    mcp_mod,
                    "_resolve_collection_for_tool",
                    new_callable=AsyncMock,
                    return_value=(collection, None),
                ),
                patch(
                    "contextmine_core.twin.get_or_create_as_is_scenario",
                    new_callable=AsyncMock,
                    return_value=scenario,
                ),
            ):
                result = await _get_arc42(collection_id=str(collection.id), regenerate=False)
            assert "not generated" in result.lower() or "Error" in result
        finally:
            ctx.stop()

    async def test_get_arc42_cached_artifact(self) -> None:
        """Cached artifact is returned directly."""
        mock_db = AsyncMock()
        collection = _make_collection()
        scenario = MagicMock()
        scenario.id = uuid.uuid4()
        scenario.name = "As-Is"
        scenario.base_scenario_id = None

        artifact = MagicMock()
        artifact.content = "# Arc42\nCached content"
        artifact.meta = {
            "facts_hash": "abc",
            "warnings": [],
            "sections": {"context": "Context section"},
        }

        call_count = {"n": 0}

        async def mock_execute(stmt):
            call_count["n"] += 1
            result = MagicMock()
            if call_count["n"] <= 1:
                # Scenario query
                result.scalar_one_or_none.return_value = scenario
            else:
                # Artifact query
                result.scalar_one_or_none.return_value = artifact
            return result

        mock_db.execute = mock_execute

        ctx = _mock_db_session(mock_db)
        try:
            with (
                patch("app.mcp_server.get_settings", return_value=_make_settings()),
                patch("app.mcp_server.get_current_user_id", return_value=None),
                patch.object(
                    mcp_mod,
                    "_resolve_collection_for_tool",
                    new_callable=AsyncMock,
                    return_value=(collection, None),
                ),
                patch(
                    "contextmine_core.twin.get_or_create_as_is_scenario",
                    new_callable=AsyncMock,
                    return_value=scenario,
                ),
            ):
                result = await _get_arc42(collection_id=str(collection.id))
            payload = json.loads(result)
            assert payload["cached"] is True
        finally:
            ctx.stop()


# ---------------------------------------------------------------------------
# arc42_drift_report (lines 2432-2559)
# ---------------------------------------------------------------------------


class TestArc42DriftReport:
    async def test_drift_disabled(self) -> None:
        with (
            patch(
                "app.mcp_server.get_settings", return_value=_make_settings(arch_docs_enabled=False)
            ),
            patch("app.mcp_server.get_current_user_id", return_value=None),
        ):
            result = await _arc42_drift()
        assert "disabled" in result.lower() or "Error" in result

    async def test_drift_success(self) -> None:
        mock_db = AsyncMock()
        collection = _make_collection()
        scenario = MagicMock()
        scenario.id = uuid.uuid4()
        scenario.base_scenario_id = None

        baseline = MagicMock()
        baseline.id = uuid.uuid4()

        @dataclass
        class FakeDelta:
            delta_type: str = "added"
            section: str = "context"
            detail: str = "new component"

        mock_facts = MagicMock()
        mock_facts.facts = []
        mock_facts.ports_adapters = []

        mock_report = MagicMock()
        mock_report.deltas = [FakeDelta()]
        mock_report.generated_at = datetime.now(UTC)
        mock_report.current_hash = "hash1"
        mock_report.baseline_hash = "hash2"
        mock_report.warnings = []

        call_count = {"n": 0}

        async def mock_execute(stmt):
            call_count["n"] += 1
            result = MagicMock()
            if call_count["n"] <= 1:
                result.scalar_one_or_none.return_value = scenario
            elif call_count["n"] == 2:
                result.scalar_one_or_none.return_value = baseline
            else:
                result.scalar_one_or_none.return_value = None
            return result

        mock_db.execute = mock_execute

        ctx = _mock_db_session(mock_db)
        try:
            with (
                patch("app.mcp_server.get_settings", return_value=_make_settings()),
                patch("app.mcp_server.get_current_user_id", return_value=None),
                patch.object(
                    mcp_mod,
                    "_resolve_collection_for_tool",
                    new_callable=AsyncMock,
                    return_value=(collection, None),
                ),
                patch(
                    "contextmine_core.twin.get_or_create_as_is_scenario",
                    new_callable=AsyncMock,
                    return_value=scenario,
                ),
                patch(
                    "contextmine_core.architecture.build_architecture_facts",
                    new_callable=AsyncMock,
                    return_value=mock_facts,
                ),
                patch(
                    "contextmine_core.architecture.compute_arc42_drift",
                    return_value=mock_report,
                ),
            ):
                result = await _arc42_drift(collection_id=str(collection.id))
            payload = json.loads(result)
            assert "deltas" in payload
            assert payload["summary"]["total"] == 1
        finally:
            ctx.stop()


# ---------------------------------------------------------------------------
# list_ports_adapters (lines 2587-2661)
# ---------------------------------------------------------------------------


class TestListPortsAdapters:
    async def test_invalid_direction(self) -> None:
        with (
            patch("app.mcp_server.get_settings", return_value=_make_settings()),
            patch("app.mcp_server.get_current_user_id", return_value=None),
        ):
            result = await _list_ports_adapters(direction="sideways")
        assert "Invalid direction" in result

    async def test_success_with_filters(self) -> None:
        mock_db = AsyncMock()
        collection = _make_collection()
        scenario = MagicMock()
        scenario.id = uuid.uuid4()

        @dataclass
        class FakePort:
            direction: str = "inbound"
            container: str = "API"
            protocol: str = "HTTP"
            evidence: str = "routes/api.py"
            confidence: float = 0.9

        mock_facts = MagicMock()
        mock_facts.ports_adapters = [FakePort(), FakePort(direction="outbound", container="DB")]
        mock_facts.warnings = []

        # Return scenario from DB query
        mock_db.execute = AsyncMock(
            return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=scenario))
        )

        ctx = _mock_db_session(mock_db)
        try:
            with (
                patch("app.mcp_server.get_settings", return_value=_make_settings()),
                patch("app.mcp_server.get_current_user_id", return_value=None),
                patch.object(
                    mcp_mod,
                    "_resolve_collection_for_tool",
                    new_callable=AsyncMock,
                    return_value=(collection, None),
                ),
                patch(
                    "contextmine_core.architecture.build_architecture_facts",
                    new_callable=AsyncMock,
                    return_value=mock_facts,
                ),
            ):
                result = await _list_ports_adapters(
                    collection_id=str(collection.id), direction="inbound"
                )
            payload = json.loads(result)
            assert payload["summary"]["inbound"] == 1
            assert payload["summary"]["outbound"] == 0
        finally:
            ctx.stop()


# ---------------------------------------------------------------------------
# Multi-engine tools with error paths
# ---------------------------------------------------------------------------


class TestMultiEngineToolErrors:
    async def test_list_methods_collection_error(self) -> None:
        mock_db = AsyncMock()
        ctx = _mock_db_session(mock_db)
        try:
            with (
                patch("app.mcp_server.get_current_user_id", return_value=None),
                patch("app.mcp_server.get_settings", return_value=_make_settings()),
                patch.object(
                    mcp_mod,
                    "_resolve_collection_access",
                    new_callable=AsyncMock,
                    return_value=(None, "Access denied"),
                ),
            ):
                result = await _list_methods(str(uuid.uuid4()))
            assert "Error" in result
        finally:
            ctx.stop()

    async def test_get_cfg_exception(self) -> None:
        mock_db = AsyncMock()
        collection = _make_collection()
        ctx = _mock_db_session(mock_db)
        try:
            with (
                patch("app.mcp_server.get_current_user_id", return_value=None),
                patch("app.mcp_server.get_settings", return_value=_make_settings()),
                patch.object(
                    mcp_mod,
                    "_resolve_collection_access",
                    new_callable=AsyncMock,
                    return_value=(collection, None),
                ),
                patch(
                    "contextmine_core.twin.get_cfg_multi",
                    new_callable=AsyncMock,
                    side_effect=RuntimeError("cfg fail"),
                ),
            ):
                result = await _get_cfg(str(collection.id), node_ref="test")
            assert "Error" in result
        finally:
            ctx.stop()

    async def test_find_taint_sources_collection_not_found(self) -> None:
        mock_db = AsyncMock()
        ctx = _mock_db_session(mock_db)
        try:
            with (
                patch("app.mcp_server.get_current_user_id", return_value=None),
                patch("app.mcp_server.get_settings", return_value=_make_settings()),
                patch.object(
                    mcp_mod,
                    "_resolve_collection_access",
                    new_callable=AsyncMock,
                    return_value=(None, None),
                ),
            ):
                result = await _find_taint_sources(str(uuid.uuid4()))
            assert "not found" in result.lower() or "Error" in result
        finally:
            ctx.stop()

    async def test_export_sarif_exception(self) -> None:
        mock_db = AsyncMock()
        collection = _make_collection()
        ctx = _mock_db_session(mock_db)
        try:
            with (
                patch("app.mcp_server.get_current_user_id", return_value=None),
                patch("app.mcp_server.get_settings", return_value=_make_settings()),
                patch.object(
                    mcp_mod,
                    "_resolve_collection_access",
                    new_callable=AsyncMock,
                    return_value=(collection, None),
                ),
                patch(
                    "contextmine_core.twin.export_findings_sarif",
                    new_callable=AsyncMock,
                    side_effect=RuntimeError("sarif fail"),
                ),
            ):
                result = await _export_sarif(str(collection.id))
            assert "Error" in result
        finally:
            ctx.stop()
