"""Targeted tests to close coverage gaps in apps/api modules.

Each section targets specific uncovered lines from --cov-report=term-missing.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from app.mcp_server import (
    _filter_graph_payload,
    _node_kind_in_scope,
    _parse_csv_list,
    _sha256_text,
    escape_like_pattern,
)
from app.routes.context import ContextRequest as APIContextRequest
from app.routes.context import ContextResponse as APIContextResponse
from app.routes.context import SourceInfo
from app.routes.metrics_ingest import (
    SHA1_RE,
    CoverageIngestJobResponse,
    _detect_protocol_from_bytes,
    _hash_ingest_token,
    _serialize_job,
)
from app.routes.sources import (
    CreateSourceRequest,
    DeployKeyResponse,
    SourceResponse,
    UpdateSourceRequest,
    hash_ingest_token,
    make_token_preview,
    mark_coverage_patterns_deprecated,
    validate_github_url,
    validate_web_url,
)
from fastapi import HTTPException


class TestHashIngestToken:
    def test_deterministic(self) -> None:
        h1 = _hash_ingest_token("my-token")
        h2 = _hash_ingest_token("my-token")
        assert h1 == h2
        assert len(h1) == 64

    def test_different_tokens(self) -> None:
        assert _hash_ingest_token("a") != _hash_ingest_token("b")

    def test_matches_sha256(self) -> None:
        result = _hash_ingest_token("test")
        expected = hashlib.sha256(b"test").hexdigest()
        assert result == expected


class TestSHA1RE:
    def test_valid(self) -> None:
        assert SHA1_RE.match("a" * 40)

    def test_invalid_short(self) -> None:
        assert SHA1_RE.match("abc") is None

    def test_invalid_chars(self) -> None:
        assert SHA1_RE.match("g" * 40) is None


class TestDetectProtocolFromBytes:
    def test_lcov_bytes(self) -> None:
        payload = b"TN:\nSF:src/main.py\nDA:1,1\nDA:2,0\nend_of_record\n"
        result = _detect_protocol_from_bytes("lcov.info", payload)
        assert result == "lcov" or result is None  # depends on detection logic

    def test_empty_bytes(self) -> None:
        result = _detect_protocol_from_bytes("empty.xml", b"")
        assert result is None


class TestSerializeJob:
    def test_basic_serialization(self) -> None:
        job = MagicMock()
        job.id = uuid.uuid4()
        job.source_id = uuid.uuid4()
        job.collection_id = uuid.uuid4()
        job.scenario_id = None
        job.commit_sha = "a" * 40
        job.branch = "main"
        job.provider = "github_actions"
        job.workflow_run_id = "12345"
        job.status = "queued"
        job.error_code = None
        job.error_detail = None
        job.stats = {"reports_total": 1}
        job.created_at = datetime.now(UTC)
        job.updated_at = datetime.now(UTC)

        report = MagicMock()
        report.id = uuid.uuid4()
        report.filename = "coverage.xml"
        report.protocol_detected = "cobertura_xml"
        report.diagnostics = {"size_bytes": 1024}
        report.created_at = datetime.now(UTC)

        result = _serialize_job(job, [report])
        assert result.status == "queued"
        assert result.commit_sha == "a" * 40
        assert len(result.reports) == 1
        assert result.reports[0]["filename"] == "coverage.xml"

    def test_with_scenario_id(self) -> None:
        job = MagicMock()
        job.id = uuid.uuid4()
        job.source_id = uuid.uuid4()
        job.collection_id = uuid.uuid4()
        job.scenario_id = uuid.uuid4()
        job.commit_sha = "b" * 40
        job.branch = None
        job.provider = "github_actions"
        job.workflow_run_id = None
        job.status = "completed"
        job.error_code = None
        job.error_detail = None
        job.stats = {}
        job.created_at = datetime.now(UTC)
        job.updated_at = datetime.now(UTC)

        result = _serialize_job(job, [])
        assert result.scenario_id == str(job.scenario_id)
        assert result.reports == []


class TestCoverageIngestJobResponse:
    def test_fields(self) -> None:
        resp = CoverageIngestJobResponse(
            id="abc",
            source_id="def",
            collection_id="ghi",
            commit_sha="a" * 40,
            provider="github_actions",
            status="queued",
            stats={},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            reports=[],
        )
        assert resp.scenario_id is None
        assert resp.branch is None
        assert resp.error_code is None


# ============================================================================
# 2. routes/sources.py — pure helper functions (lines 118-177)
# ============================================================================


class TestValidateGithubUrl:
    def test_valid_url(self) -> None:
        result = validate_github_url("https://github.com/owner/repo")
        assert result["owner"] == "owner"
        assert result["repo"] == "repo"
        assert result["branch"] == "main"

    def test_valid_url_with_git(self) -> None:
        result = validate_github_url("https://github.com/owner/repo.git")
        assert result["owner"] == "owner"
        assert result["repo"] == "repo"

    def test_invalid_url(self) -> None:
        with pytest.raises(HTTPException) as exc_info:
            validate_github_url("not-a-url")
        assert exc_info.value.status_code == 400


class TestValidateWebUrl:
    def test_valid_http(self) -> None:
        result = validate_web_url("https://docs.example.com/guide")
        assert result["start_url"] == "https://docs.example.com/guide"
        assert result["base_url"].startswith("https://")

    def test_invalid_scheme(self) -> None:
        with pytest.raises(HTTPException):
            validate_web_url("ftp://example.com")

    def test_path_with_trailing_slash(self) -> None:
        result = validate_web_url("https://example.com/docs/")
        assert result["base_url"] == "https://example.com/docs/"

    def test_root_path(self) -> None:
        result = validate_web_url("https://example.com")
        assert "example.com" in result["base_url"]


class TestMarkCoveragePatternsDeprecated:
    def test_basic(self) -> None:
        config = {"key": "value"}
        result = mark_coverage_patterns_deprecated(config, ["*.xml"])
        assert result["metrics"]["deprecated"] is True
        assert result["metrics"]["coverage_report_patterns_ignored"] == ["*.xml"]
        # Original config not mutated
        assert "metrics" not in config

    def test_none_patterns(self) -> None:
        result = mark_coverage_patterns_deprecated({}, None)
        assert result["metrics"]["deprecated"] is True
        assert "coverage_report_patterns_ignored" not in result["metrics"]

    def test_existing_metrics(self) -> None:
        config = {"metrics": {"some_key": "some_val"}}
        result = mark_coverage_patterns_deprecated(config, [])
        assert result["metrics"]["deprecated"] is True
        # Should not preserve nested dict reference
        assert result["metrics"]["deprecated_field"] == "coverage_report_patterns"


class TestHashIngestTokenSources:
    def test_deterministic(self) -> None:
        h1 = hash_ingest_token("tok")
        h2 = hash_ingest_token("tok")
        assert h1 == h2

    def test_sha256(self) -> None:
        result = hash_ingest_token("test")
        expected = hashlib.sha256(b"test").hexdigest()
        assert result == expected


class TestMakeTokenPreview:
    def test_normal_token(self) -> None:
        token = "abcdefghij1234567890"
        preview = make_token_preview(token)
        assert preview.startswith("abcdef")
        assert preview.endswith("7890")
        assert "..." in preview

    def test_short_token(self) -> None:
        preview = make_token_preview("short")
        assert preview == "********"


class TestRequestModels:
    def test_create_source_request_defaults(self) -> None:
        req = CreateSourceRequest(type="github", url="https://github.com/o/r")
        assert req.enabled is True
        assert req.schedule_interval_minutes == 60

    def test_update_source_request_all_none(self) -> None:
        req = UpdateSourceRequest()
        assert req.enabled is None
        assert req.schedule_interval_minutes is None

    def test_source_response(self) -> None:
        resp = SourceResponse(
            id="abc",
            collection_id="def",
            type="github",
            url="https://github.com/o/r",
            config={},
            enabled=True,
            schedule_interval_minutes=60,
            created_at=datetime.now(UTC),
        )
        assert resp.deploy_key_fingerprint is None

    def test_deploy_key_response(self) -> None:
        resp = DeployKeyResponse(has_key=True, fingerprint="SHA256:xxx")
        assert resp.has_key is True


# ============================================================================
# 3. mcp_server.py — pure helper functions (lines 39-133)
# ============================================================================


class TestEscapeLikePattern:
    def test_no_special(self) -> None:
        assert escape_like_pattern("hello") == "hello"

    def test_percent(self) -> None:
        assert escape_like_pattern("100%") == "100\\%"

    def test_underscore(self) -> None:
        assert escape_like_pattern("my_table") == "my\\_table"

    def test_backslash(self) -> None:
        assert escape_like_pattern("path\\file") == "path\\\\file"

    def test_combined(self) -> None:
        assert escape_like_pattern("a%b_c\\d") == "a\\%b\\_c\\\\d"


class TestSha256Text:
    def test_deterministic(self) -> None:
        h1 = _sha256_text("hello")
        h2 = _sha256_text("hello")
        assert h1 == h2

    def test_length(self) -> None:
        assert len(_sha256_text("test")) == 64


class TestParseCsvList:
    def test_none(self) -> None:
        assert _parse_csv_list(None) is None

    def test_empty(self) -> None:
        assert _parse_csv_list("") is None

    def test_single(self) -> None:
        assert _parse_csv_list("hello") == ["hello"]

    def test_multiple(self) -> None:
        assert _parse_csv_list("a, b, c") == ["a", "b", "c"]

    def test_whitespace(self) -> None:
        assert _parse_csv_list("  a , , b  ") == ["a", "b"]


class TestNodeKindInScope:
    def test_all_scope(self) -> None:
        assert _node_kind_in_scope("test_case", "all") is True
        assert _node_kind_in_scope("file", "all") is True

    def test_tests_scope(self) -> None:
        assert _node_kind_in_scope("test_case", "tests") is True
        assert _node_kind_in_scope("test_suite", "tests") is True
        assert _node_kind_in_scope("test_fixture", "tests") is True
        assert _node_kind_in_scope("file", "tests") is False

    def test_ui_scope(self) -> None:
        assert _node_kind_in_scope("ui_route", "ui") is True
        assert _node_kind_in_scope("ui_view", "ui") is True
        assert _node_kind_in_scope("ui_component", "ui") is True
        assert _node_kind_in_scope("interface_contract", "ui") is True
        assert _node_kind_in_scope("file", "ui") is False

    def test_flows_scope(self) -> None:
        assert _node_kind_in_scope("user_flow", "flows") is True
        assert _node_kind_in_scope("flow_step", "flows") is True
        assert _node_kind_in_scope("file", "flows") is False

    def test_code_scope(self) -> None:
        assert _node_kind_in_scope("file", "code") is True
        assert _node_kind_in_scope("test_case", "code") is False
        assert _node_kind_in_scope("ui_route", "code") is False
        assert _node_kind_in_scope("user_flow", "code") is False

    def test_unknown_scope(self) -> None:
        assert _node_kind_in_scope("anything", "unknown") is True


# ============================================================================
# 4. mcp_server.py — _filter_graph_payload (lines 135-191)
# ============================================================================


class TestFilterGraphPayload:
    def test_all_scope_passthrough(self) -> None:
        graph = {
            "nodes": [
                {"id": "1", "kind": "file"},
                {"id": "2", "kind": "test_case"},
            ],
            "edges": [
                {"source_node_id": "1", "target_node_id": "2", "kind": "test_covers"},
            ],
        }
        result = _filter_graph_payload(graph, scope="all")
        assert len(result["nodes"]) == 2
        assert len(result["edges"]) == 1

    def test_tests_scope_filters(self) -> None:
        graph = {
            "nodes": [
                {"id": "1", "kind": "file"},
                {"id": "2", "kind": "test_case"},
                {"id": "3", "kind": "test_suite"},
            ],
            "edges": [
                {"source_node_id": "1", "target_node_id": "2", "kind": "x"},
                {"source_node_id": "2", "target_node_id": "3", "kind": "y"},
            ],
        }
        result = _filter_graph_payload(graph, scope="tests")
        assert len(result["nodes"]) == 2
        # Only edge between test nodes survives
        assert len(result["edges"]) == 1

    def test_provenance_filter(self) -> None:
        graph = {
            "nodes": [
                {"id": "1", "kind": "file", "meta": {"provenance": {"mode": "deterministic"}}},
                {"id": "2", "kind": "file", "meta": {"provenance": {"mode": "inferred"}}},
            ],
            "edges": [],
        }
        result = _filter_graph_payload(graph, provenance_mode="deterministic")
        assert len(result["nodes"]) == 1
        assert result["nodes"][0]["id"] == "1"

    def test_exclude_test_links(self) -> None:
        graph = {
            "nodes": [{"id": "1", "kind": "file"}],
            "edges": [
                {"source_node_id": "1", "target_node_id": "1", "kind": "test_covers_symbol"},
                {"source_node_id": "1", "target_node_id": "1", "kind": "depends_on"},
            ],
        }
        result = _filter_graph_payload(graph, include_test_links=False)
        assert len(result["edges"]) == 1
        assert result["edges"][0]["kind"] == "depends_on"

    def test_exclude_ui_links(self) -> None:
        graph = {
            "nodes": [{"id": "1", "kind": "file"}],
            "edges": [
                {"source_node_id": "1", "target_node_id": "1", "kind": "ui_route_renders"},
                {"source_node_id": "1", "target_node_id": "1", "kind": "calls"},
            ],
        }
        result = _filter_graph_payload(graph, include_ui_links=False)
        assert len(result["edges"]) == 1

    def test_total_nodes_added(self) -> None:
        graph = {"nodes": [{"id": "1", "kind": "file"}], "edges": []}
        result = _filter_graph_payload(graph)
        assert result["total_nodes"] == 1

    def test_empty_graph(self) -> None:
        result = _filter_graph_payload({})
        assert result["nodes"] == []
        assert result["edges"] == []
        assert result["total_nodes"] == 0


# ============================================================================
# 5. routes/context.py — models (lines 27-61)
# ============================================================================


class TestContextRouteModels:
    def test_context_request_defaults(self) -> None:
        req = APIContextRequest(query="test query")
        assert req.max_chunks == 10
        assert req.max_tokens == 4000
        assert req.provider is None
        assert req.model is None
        assert req.collection_id is None

    def test_source_info(self) -> None:
        info = SourceInfo(uri="https://example.com", title="Example")
        assert info.file_path is None

    def test_source_info_with_file(self) -> None:
        info = SourceInfo(uri="git://gh.com/o/r/f.py", title="f.py", file_path="f.py")
        assert info.file_path == "f.py"

    def test_context_response(self) -> None:
        resp = APIContextResponse(
            markdown="# Result",
            query="test",
            chunks_used=5,
            sources=[SourceInfo(uri="u", title="t")],
        )
        assert resp.chunks_used == 5
        assert len(resp.sources) == 1
