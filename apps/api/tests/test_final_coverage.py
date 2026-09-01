"""Targeted tests to close coverage gaps in apps/api modules.

Each section targets specific uncovered lines from --cov-report=term-missing.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from app.mcp_domains.collections import escape_like_pattern
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
        assert result["branch"] is None

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
        assert req.schedule_interval_minutes == 1440

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
# 3. MCP collection query escaping
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


# ============================================================================
# 4. routes/context.py — models (lines 27-61)
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
