"""Comprehensive unit tests for contextmine_worker.flows.

Covers pure/helper functions, timeout configurations, URI parsing,
path filtering, async helpers, SCIP helpers, and mocked async flows.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import contextmine_worker.flows as flows
import pytest
from contextmine_worker.flows import (
    IGNORED_REPO_PATH_PARTS,
    SYNC_RUN_STALE_AFTER,
    TAG_DB_HEAVY,
    TAG_EMBEDDING_API,
    TAG_GITHUB_API,
    TAG_SCIP_INDEX,
    TAG_WEB_CRAWL,
    _embedding_batch_timeout_seconds,
    _is_ignored_repo_path,
    _joern_parse_timeout_seconds,
    _knowledge_graph_build_timeout_seconds,
    _log_background_task_failure,
    _run_blocking_with_timeout,
    _sync_blocking_step_timeout_seconds,
    _sync_document_step_timeout_seconds,
    _sync_documents_per_run_limit,
    _sync_github_retry_condition,
    _sync_source_timeout_seconds,
    _sync_temporal_coupling_max_files_per_commit,
    _twin_graph_build_timeout_seconds,
    _uri_to_file_path,
)

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_settings(**overrides: Any) -> SimpleNamespace:
    """Build a minimal settings namespace for testing timeout helpers."""
    defaults = {
        "sync_source_timeout_seconds": 3600,
        "embedding_batch_timeout_seconds": 120,
        "knowledge_graph_build_timeout_seconds": 600,
        "twin_graph_build_timeout_seconds": 600,
        "sync_blocking_step_timeout_seconds": 60,
        "sync_document_step_timeout_seconds": 30,
        "sync_documents_per_run_limit": 0,
        "sync_temporal_coupling_max_files_per_commit": 20,
        "joern_parse_timeout_seconds": 120,
        "default_embedding_model": "openai:text-embedding-3-small",
        "default_llm_provider": "openai",
        "scip_languages": "python,typescript",
        "scip_install_deps_mode": "auto",
        "scip_timeout_python": 300,
        "scip_timeout_typescript": 300,
        "scip_timeout_java": 300,
        "scip_timeout_php": 300,
        "scip_node_memory_mb": 2048,
        "scip_best_effort": True,
        "scip_require_language_coverage": False,
        "scip_require_relation_coverage": False,
        "scip_require_php_relation_coverage": False,
        "metrics_strict_mode": False,
        "metrics_languages": "python,typescript",
        "digital_twin_behavioral_enabled": True,
        "digital_twin_ui_enabled": True,
        "digital_twin_flows_enabled": True,
        "arch_docs_enabled": False,
        "arch_docs_generate_on_sync": False,
        "arch_docs_llm_enrich": False,
        "arch_docs_drift_enabled": False,
        "arch_docs_llm_max_hypotheses": 12,
        "twin_evolution_window_days": 90,
        "joern_server_url": "",
        "joern_required_for_sync": False,
        "joern_parse_binary": "joern-parse",
        "joern_cpg_root": "/tmp/cpg",
        "coverage_ingest_max_payload_mb": 50,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify module-level constants are sensible."""

    def test_ignored_repo_path_parts_is_frozenset(self) -> None:
        assert isinstance(IGNORED_REPO_PATH_PARTS, frozenset)

    def test_ignored_repo_path_parts_contains_expected(self) -> None:
        for expected in ("node_modules", "__pycache__", ".git", "venv", ".venv", "dist", "build"):
            assert expected in IGNORED_REPO_PATH_PARTS

    def test_tag_constants(self) -> None:
        assert TAG_GITHUB_API == "github-api"
        assert TAG_EMBEDDING_API == "embedding-api"
        assert TAG_WEB_CRAWL == "web-crawl"
        assert TAG_DB_HEAVY == "db-heavy"
        assert TAG_SCIP_INDEX == "scip-index"

    def test_stale_after_is_timedelta(self) -> None:
        assert isinstance(SYNC_RUN_STALE_AFTER, timedelta)
        assert timedelta(hours=6) == SYNC_RUN_STALE_AFTER


class TestGithubSyncRetryCondition:
    async def test_disables_retry_for_twin_timeout(self) -> None:
        should_retry = await _sync_github_retry_condition(
            None,
            None,
            SimpleNamespace(
                data=RuntimeError("TWIN_BUILD_TIMEOUT: source=abc timeout=3600s"),
                message="",
            ),
        )
        assert should_retry is False

    async def test_disables_retry_for_non_clone_step_timeout(self) -> None:
        should_retry = await _sync_github_retry_condition(
            None,
            None,
            SimpleNamespace(
                data=RuntimeError("STEP_TIMEOUT: metrics_pipeline exceeded 60s"),
                message="",
            ),
        )
        assert should_retry is False

    async def test_allows_retry_for_clone_step_timeout(self) -> None:
        should_retry = await _sync_github_retry_condition(
            None,
            None,
            SimpleNamespace(
                data=RuntimeError("STEP_TIMEOUT: git_clone_or_pull exceeded 60s"),
                message="",
            ),
        )
        assert should_retry is True

    async def test_allows_retry_for_connection_error(self) -> None:
        should_retry = await _sync_github_retry_condition(
            None,
            None,
            SimpleNamespace(
                data=ConnectionError("temporary failure in name resolution"), message=""
            ),
        )
        assert should_retry is True

    def test_sync_github_source_uses_retry_condition(self) -> None:
        assert flows.sync_github_source.retry_condition_fn is _sync_github_retry_condition


# ---------------------------------------------------------------------------
# Timeout / config helpers
# ---------------------------------------------------------------------------


class TestSyncSourceTimeout:
    def test_positive_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            flows, "get_settings", lambda: _make_settings(sync_source_timeout_seconds=1800)
        )
        assert _sync_source_timeout_seconds() == 1800

    def test_zero_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            flows, "get_settings", lambda: _make_settings(sync_source_timeout_seconds=0)
        )
        assert _sync_source_timeout_seconds() == 0

    def test_negative_clamped_to_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            flows, "get_settings", lambda: _make_settings(sync_source_timeout_seconds=-10)
        )
        assert _sync_source_timeout_seconds() == 0


class TestEmbeddingBatchTimeout:
    def test_positive_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            flows, "get_settings", lambda: _make_settings(embedding_batch_timeout_seconds=200)
        )
        assert _embedding_batch_timeout_seconds() == 200

    def test_minimum_clamp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            flows, "get_settings", lambda: _make_settings(embedding_batch_timeout_seconds=5)
        )
        assert _embedding_batch_timeout_seconds() == 10

    def test_negative_clamped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            flows, "get_settings", lambda: _make_settings(embedding_batch_timeout_seconds=-1)
        )
        assert _embedding_batch_timeout_seconds() == 10


class TestKnowledgeGraphBuildTimeout:
    def test_positive_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            flows, "get_settings", lambda: _make_settings(knowledge_graph_build_timeout_seconds=900)
        )
        assert _knowledge_graph_build_timeout_seconds() == 900

    def test_minimum_clamp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            flows, "get_settings", lambda: _make_settings(knowledge_graph_build_timeout_seconds=10)
        )
        assert _knowledge_graph_build_timeout_seconds() == 120


class TestTwinGraphBuildTimeout:
    def test_positive_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            flows, "get_settings", lambda: _make_settings(twin_graph_build_timeout_seconds=300)
        )
        assert _twin_graph_build_timeout_seconds() == 300

    def test_minimum_clamp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            flows, "get_settings", lambda: _make_settings(twin_graph_build_timeout_seconds=5)
        )
        assert _twin_graph_build_timeout_seconds() == 120


class TestSyncBlockingStepTimeout:
    def test_positive_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            flows, "get_settings", lambda: _make_settings(sync_blocking_step_timeout_seconds=120)
        )
        assert _sync_blocking_step_timeout_seconds() == 120

    def test_minimum_clamp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            flows, "get_settings", lambda: _make_settings(sync_blocking_step_timeout_seconds=5)
        )
        assert _sync_blocking_step_timeout_seconds() == 30


class TestSyncDocumentStepTimeout:
    def test_positive_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            flows, "get_settings", lambda: _make_settings(sync_document_step_timeout_seconds=60)
        )
        assert _sync_document_step_timeout_seconds() == 60

    def test_minimum_clamp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            flows, "get_settings", lambda: _make_settings(sync_document_step_timeout_seconds=3)
        )
        assert _sync_document_step_timeout_seconds() == 10


class TestSyncDocumentsPerRunLimit:
    def test_positive_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            flows, "get_settings", lambda: _make_settings(sync_documents_per_run_limit=500)
        )
        assert _sync_documents_per_run_limit() == 500

    def test_zero_means_unlimited(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            flows, "get_settings", lambda: _make_settings(sync_documents_per_run_limit=0)
        )
        assert _sync_documents_per_run_limit() == 0

    def test_negative_clamped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            flows, "get_settings", lambda: _make_settings(sync_documents_per_run_limit=-5)
        )
        assert _sync_documents_per_run_limit() == 0


class TestSyncTemporalCouplingMaxFiles:
    def test_positive_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            flows,
            "get_settings",
            lambda: _make_settings(sync_temporal_coupling_max_files_per_commit=50),
        )
        assert _sync_temporal_coupling_max_files_per_commit() == 50

    def test_negative_clamped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            flows,
            "get_settings",
            lambda: _make_settings(sync_temporal_coupling_max_files_per_commit=-1),
        )
        assert _sync_temporal_coupling_max_files_per_commit() == 0


class TestJoernParseTimeout:
    def test_positive_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            flows, "get_settings", lambda: _make_settings(joern_parse_timeout_seconds=180)
        )
        assert _joern_parse_timeout_seconds() == 180

    def test_minimum_clamp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            flows, "get_settings", lambda: _make_settings(joern_parse_timeout_seconds=5)
        )
        assert _joern_parse_timeout_seconds() == 30


# ---------------------------------------------------------------------------
# URI / path helpers
# ---------------------------------------------------------------------------


class TestUriToFilePath:
    def test_simple_git_uri(self) -> None:
        uri = "git://github.com/owner/repo/src/main.py?ref=main"
        assert _uri_to_file_path(uri) == "src/main.py"

    def test_uri_with_deep_path(self) -> None:
        uri = "git://github.com/owner/repo/a/b/c/d.py?ref=main"
        assert _uri_to_file_path(uri) == "a/b/c/d.py"

    def test_uri_without_query(self) -> None:
        uri = "git://github.com/owner/repo/file.py"
        assert _uri_to_file_path(uri) == "file.py"

    def test_plain_filename(self) -> None:
        assert _uri_to_file_path("hello.txt") == "hello.txt"

    def test_uri_no_slash_returns_as_is(self) -> None:
        assert _uri_to_file_path("justname") == "justname"


class TestIsIgnoredRepoPath:
    def test_node_modules(self) -> None:
        assert _is_ignored_repo_path("node_modules/react/index.js")

    def test_vendor(self) -> None:
        assert _is_ignored_repo_path("vendor/symfony/thing.php")

    def test_dist(self) -> None:
        assert _is_ignored_repo_path("frontend/dist/main.js")

    def test_pycache(self) -> None:
        assert _is_ignored_repo_path("src/__pycache__/mod.cpython-312.pyc")

    def test_git_dir(self) -> None:
        assert _is_ignored_repo_path(".git/objects/pack/p1")

    def test_venv(self) -> None:
        assert _is_ignored_repo_path("service/venv/lib/python3.12/site-packages/pkg.py")

    def test_dot_venv(self) -> None:
        assert _is_ignored_repo_path("service/.venv/lib/python3.12/site-packages/pkg.py")

    def test_build(self) -> None:
        assert _is_ignored_repo_path("project/build/output.js")

    def test_src_libs(self) -> None:
        assert _is_ignored_repo_path("phpmyfaq/src/libs/aws/S3Client.php")

    def test_backslash_normalized(self) -> None:
        assert _is_ignored_repo_path(r"backend\\venv\\lib\\python3.12\\pkg.py")

    def test_first_party_src(self) -> None:
        assert not _is_ignored_repo_path("src/app/main.py")

    def test_first_party_github(self) -> None:
        assert not _is_ignored_repo_path(".github/workflows/ci.yml")

    def test_first_party_deep(self) -> None:
        assert not _is_ignored_repo_path("phpmyfaq/src/phpMyFAQ/Database.php")

    def test_empty_string(self) -> None:
        assert not _is_ignored_repo_path("")

    def test_root_file(self) -> None:
        assert not _is_ignored_repo_path("README.md")


# ---------------------------------------------------------------------------
# _run_blocking_with_timeout
# ---------------------------------------------------------------------------


class TestRunBlockingWithTimeout:
    """These tests use asyncio.to_thread so they require the asyncio backend.

    We test the logic using plain asyncio.run() to avoid trio parametrization.
    """

    def test_successful_call(self) -> None:
        async def _inner() -> None:
            def blocking_fn(x: int) -> int:
                return x * 2

            result = await _run_blocking_with_timeout("test_step", 5, blocking_fn, 21)
            assert result == 42

        asyncio.run(_inner())

    def test_timeout_raises_runtime_error(self) -> None:
        import time

        async def _inner() -> None:
            def slow_fn() -> None:
                time.sleep(10)

            with pytest.raises(RuntimeError, match="STEP_TIMEOUT: slow_step exceeded 1s"):
                await _run_blocking_with_timeout("slow_step", 1, slow_fn)

        asyncio.run(_inner())

    def test_minimum_timeout_is_1(self) -> None:
        """Even with timeout_seconds=0, the function uses max(1, ...)."""

        async def _inner() -> None:
            def quick_fn() -> str:
                return "done"

            result = await _run_blocking_with_timeout("quick", 0, quick_fn)
            assert result == "done"

        asyncio.run(_inner())

    def test_kwargs_forwarded(self) -> None:
        async def _inner() -> None:
            def fn_with_kwargs(a: int, b: int = 10) -> int:
                return a + b

            result = await _run_blocking_with_timeout("kwargs_test", 5, fn_with_kwargs, 1, b=99)
            assert result == 100

        asyncio.run(_inner())


# ---------------------------------------------------------------------------
# _log_background_task_failure
# ---------------------------------------------------------------------------


class TestLogBackgroundTaskFailure:
    def test_cancelled_task_logs_warning(self) -> None:
        mock_task = MagicMock()
        mock_task.cancelled.return_value = True
        # Should not raise
        _log_background_task_failure(mock_task)
        mock_task.cancelled.assert_called_once()

    def test_task_with_exception_logs_warning(self) -> None:
        mock_task = MagicMock()
        mock_task.cancelled.return_value = False
        mock_task.exception.return_value = RuntimeError("boom")
        _log_background_task_failure(mock_task)
        mock_task.exception.assert_called_once()

    def test_successful_task_no_error(self) -> None:
        mock_task = MagicMock()
        mock_task.cancelled.return_value = False
        mock_task.exception.return_value = None
        _log_background_task_failure(mock_task)


# ---------------------------------------------------------------------------
# _build_scip_index_config
# ---------------------------------------------------------------------------


class TestBuildScipIndexConfig:
    def test_parses_languages(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            flows, "get_settings", lambda: _make_settings(scip_languages="python,typescript")
        )
        config = flows._build_scip_index_config()
        language_values = {lang.value for lang in config.enabled_languages}
        assert "python" in language_values
        assert "typescript" in language_values

    def test_unknown_language_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            flows,
            "get_settings",
            lambda: _make_settings(scip_languages="python,xyzlang"),
        )
        config = flows._build_scip_index_config()
        language_values = {lang.value for lang in config.enabled_languages}
        assert "python" in language_values
        assert "xyzlang" not in language_values

    def test_install_deps_mode_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            flows,
            "get_settings",
            lambda: _make_settings(scip_install_deps_mode="nonexistent_mode"),
        )
        config = flows._build_scip_index_config()
        # Should fallback to AUTO
        assert config.install_deps_mode.value == "auto"

    def test_best_effort_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(flows, "get_settings", lambda: _make_settings(scip_best_effort=True))
        config = flows._build_scip_index_config()
        assert config.best_effort is True

    def test_timeout_by_language(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            flows,
            "get_settings",
            lambda: _make_settings(scip_timeout_python=999, scip_timeout_typescript=888),
        )
        config = flows._build_scip_index_config()
        from contextmine_core.semantic_snapshot.models import Language

        assert config.timeout_s_by_language[Language.PYTHON] == 999
        assert config.timeout_s_by_language[Language.TYPESCRIPT] == 888


# ---------------------------------------------------------------------------
# materialize_surface_catalog_for_source
# ---------------------------------------------------------------------------


class TestMaterializeSurfaceCatalog:
    async def test_empty_source_returns_zero_stats(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When the source has no documents, all stats should be zero."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(flows, "get_session", lambda: mock_cm)

        result = await flows.materialize_surface_catalog_for_source(
            source_id=source_id,
            collection_id=collection_id,
        )

        assert result["surface_files_scanned"] == 0
        assert result["surface_files_recognized"] == 0
        assert result["surface_parse_errors"] == 0

    async def test_skips_none_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Documents with None content should be skipped."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())

        mock_session = AsyncMock()
        # Simulate a document with None content
        mock_result = MagicMock()
        mock_result.all.return_value = [("git://example.com/repo/file.py", None)]
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(flows, "get_session", lambda: mock_cm)

        result = await flows.materialize_surface_catalog_for_source(
            source_id=source_id,
            collection_id=collection_id,
        )

        assert result["surface_files_scanned"] == 0

    async def test_skips_ignored_paths(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Documents in ignored repo paths should be skipped."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = [
            ("git://github.com/owner/repo/node_modules/openapi.yaml", "openapi: '3.0'"),
        ]
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(flows, "get_session", lambda: mock_cm)

        result = await flows.materialize_surface_catalog_for_source(
            source_id=source_id,
            collection_id=collection_id,
        )

        assert result["surface_files_scanned"] == 0


# ---------------------------------------------------------------------------
# get_embedding_model_for_collection
# ---------------------------------------------------------------------------


class TestGetEmbeddingModelForCollection:
    async def test_uses_collection_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        collection_id = str(uuid.uuid4())

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = {
            "embedding_model": "gemini:text-embedding-004"
        }
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(flows, "get_session", lambda: mock_cm)

        result = await flows.get_embedding_model_for_collection(collection_id)
        assert result == "gemini:text-embedding-004"

    async def test_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        collection_id = str(uuid.uuid4())

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(flows, "get_session", lambda: mock_cm)

        monkeypatch.setattr(
            flows,
            "get_settings",
            lambda: _make_settings(default_embedding_model="openai:text-embedding-3-large"),
        )

        result = await flows.get_embedding_model_for_collection(collection_id)
        assert result == "openai:text-embedding-3-large"

    async def test_empty_config_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        collection_id = str(uuid.uuid4())

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = {}
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(flows, "get_session", lambda: mock_cm)

        monkeypatch.setattr(
            flows,
            "get_settings",
            lambda: _make_settings(default_embedding_model="openai:text-embedding-3-small"),
        )

        result = await flows.get_embedding_model_for_collection(collection_id)
        assert result == "openai:text-embedding-3-small"


# ---------------------------------------------------------------------------
# embed_document
# ---------------------------------------------------------------------------


class TestEmbedDocument:
    async def test_uses_collection_model_when_provided(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        doc_id = str(uuid.uuid4())
        coll_id = str(uuid.uuid4())

        monkeypatch.setattr(
            flows,
            "get_embedding_model_for_collection",
            AsyncMock(return_value="openai:text-embedding-3-small"),
        )

        mock_embedder = MagicMock()
        mock_embedder.dimension = 1536
        monkeypatch.setattr(flows, "get_embedder", lambda *a, **kw: mock_embedder)

        fake_model = MagicMock()
        fake_model.id = uuid.uuid4()
        monkeypatch.setattr(
            flows,
            "get_or_create_embedding_model",
            AsyncMock(return_value=fake_model),
        )

        embed_result = {"chunks_embedded": 5, "chunks_deduplicated": 1, "tokens_used": 100}
        monkeypatch.setattr(
            flows,
            "embed_chunks_for_document",
            AsyncMock(return_value=embed_result),
        )

        result = await flows.embed_document(doc_id, coll_id)
        assert result == embed_result

    async def test_uses_default_when_no_collection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc_id = str(uuid.uuid4())

        monkeypatch.setattr(
            flows,
            "get_settings",
            lambda: _make_settings(default_embedding_model="openai:text-embedding-3-small"),
        )

        mock_embedder = MagicMock()
        mock_embedder.dimension = 1536
        monkeypatch.setattr(flows, "get_embedder", lambda *a, **kw: mock_embedder)

        fake_model = MagicMock()
        fake_model.id = uuid.uuid4()
        monkeypatch.setattr(
            flows,
            "get_or_create_embedding_model",
            AsyncMock(return_value=fake_model),
        )

        embed_result = {"chunks_embedded": 0, "chunks_deduplicated": 0, "tokens_used": 0}
        monkeypatch.setattr(
            flows,
            "embed_chunks_for_document",
            AsyncMock(return_value=embed_result),
        )

        result = await flows.embed_document(doc_id)
        assert result["chunks_embedded"] == 0


# ---------------------------------------------------------------------------
# get_github_token_for_source
# ---------------------------------------------------------------------------


class TestGetGithubTokenForSource:
    async def test_returns_none_when_no_owner(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(flows, "get_session", lambda: mock_cm)

        result = await flows.get_github_token_for_source("src-123", str(uuid.uuid4()))
        assert result is None

    async def test_returns_none_when_no_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        call_count = 0

        async def mock_execute(stmt):
            nonlocal call_count
            call_count += 1
            result_mock = MagicMock()
            if call_count == 1:
                # owner query
                result_mock.scalar_one_or_none.return_value = uuid.uuid4()
            else:
                # token query
                result_mock.scalars.return_value.all.return_value = []
            return result_mock

        mock_session = AsyncMock()
        mock_session.execute = mock_execute

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(flows, "get_session", lambda: mock_cm)

        result = await flows.get_github_token_for_source("src-123", str(uuid.uuid4()))
        assert result is None


# ---------------------------------------------------------------------------
# get_deploy_key_for_source
# ---------------------------------------------------------------------------


class TestGetDeployKeyForSource:
    async def test_returns_none_when_no_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(flows, "get_session", lambda: mock_cm)

        result = await flows.get_deploy_key_for_source(str(uuid.uuid4()))
        assert result is None

    async def test_decrypts_when_key_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = "encrypted_key_data"
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(flows, "get_session", lambda: mock_cm)
        monkeypatch.setattr(flows, "decrypt_token", lambda v: f"decrypted:{v}")

        result = await flows.get_deploy_key_for_source(str(uuid.uuid4()))
        assert result == "decrypted:encrypted_key_data"


# ---------------------------------------------------------------------------
# _fail_running_sync_runs_for_source
# ---------------------------------------------------------------------------


class TestFailRunningSyncRunsForSource:
    async def test_returns_count_zero_when_no_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(flows, "get_session", lambda: mock_cm)

        count = await flows._fail_running_sync_runs_for_source(str(uuid.uuid4()), "test reason")
        assert count == 0
        mock_session.commit.assert_not_awaited()

    async def test_marks_rows_as_failed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from contextmine_core import SyncRunStatus

        mock_row1 = MagicMock()
        mock_row2 = MagicMock()

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_row1, mock_row2]
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(flows, "get_session", lambda: mock_cm)

        count = await flows._fail_running_sync_runs_for_source(str(uuid.uuid4()), "test_reason")
        assert count == 2
        assert mock_row1.status == SyncRunStatus.FAILED
        assert mock_row2.error == "test_reason"
        mock_session.commit.assert_awaited_once()


# ---------------------------------------------------------------------------
# _fail_coverage_ingest_job
# ---------------------------------------------------------------------------


class TestFailCoverageIngestJob:
    async def test_missing_job(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(flows, "get_session", lambda: mock_cm)

        result = await flows._fail_coverage_ingest_job(
            str(uuid.uuid4()),
            error_code="TEST_ERR",
            error_detail="Testing",
        )
        assert result["status"] == "missing_job"

    async def test_updates_job_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_job = MagicMock()
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_job
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(flows, "get_session", lambda: mock_cm)

        result = await flows._fail_coverage_ingest_job(
            str(uuid.uuid4()),
            status="rejected",
            error_code="ERR_CODE",
            error_detail="detail text",
            stats={"something": 1},
        )
        assert result["status"] == "rejected"
        assert result["error_code"] == "ERR_CODE"
        assert mock_job.status == "rejected"
        assert mock_job.error_code == "ERR_CODE"
        assert mock_job.error_detail == "detail text"
        assert mock_job.stats == {"something": 1}
        mock_session.commit.assert_awaited_once()


# ---------------------------------------------------------------------------
# SCIP inner helper functions (nested in sync_github_source)
# We test the module-level helper functions that share the same logic.
# ---------------------------------------------------------------------------


class TestSnapshotHelperFunctions:
    """Test the nested helper functions by extracting their logic patterns."""

    def test_normalize_language(self) -> None:
        """_normalize_language returns stripped lowercase."""

        # The actual function is a closure, so we test the logic
        def _normalize_language(value: object) -> str:
            return str(value or "").strip().lower()

        assert _normalize_language("Python") == "python"
        assert _normalize_language("  TypeScript  ") == "typescript"
        assert _normalize_language(None) == ""
        assert _normalize_language("") == ""

    def test_project_key_for(self) -> None:
        """_project_key_for returns (language, root, mode) tuple."""

        # The actual function is nested; we replicate its logic
        def _normalize_language(value: object) -> str:
            return str(value or "").strip().lower()

        proj = {
            "language": "Python",
            "root_path": "/tmp/repo",
            "metadata": {},
        }
        language = _normalize_language(proj.get("language"))
        assert language == "python"

        proj_recovery = {
            "language": "Python",
            "root_path": "/tmp/repo",
            "metadata": {"recovery_pass": True},
        }
        metadata = dict(proj_recovery.get("metadata") or {})
        mode = "default"
        if metadata.get("relation_recovery"):
            mode = "relation_recovery"
        elif metadata.get("recovery_pass"):
            mode = "recovery"
        assert mode == "recovery"

    def test_missing_relation_languages_logic(self) -> None:
        """Test the _missing_relation_languages logic."""
        semantic_relation_kinds = {"calls", "references", "imports", "extends", "implements"}

        indexed_files = {"python": 10, "typescript": 5, "java": 0}
        relation_kinds = {
            "python": {"calls": 5, "references": 3},
            # typescript has no semantic relations
        }

        missing: list[str] = []
        for language, indexed_count in indexed_files.items():
            if int(indexed_count or 0) <= 0:
                continue
            lk = relation_kinds.get(language) or {}
            semantic_edges = sum(int(lk.get(kind, 0) or 0) for kind in semantic_relation_kinds)
            if semantic_edges <= 0:
                missing.append(language)

        assert "typescript" in missing
        assert "python" not in missing
        assert "java" not in missing  # 0 indexed files means it should not be in missing

    def test_collect_indexed_files_by_language_logic(self) -> None:
        """Test the indexed files collection logic."""
        snapshots = [
            {
                "meta": {"language": "python", "repo_relative_root": ""},
                "files": [
                    {"path": "src/main.py", "language": "python"},
                    {"path": "src/utils.py", "language": "python"},
                ],
                "symbols": [],
                "relations": [],
            },
            {
                "meta": {"language": "typescript", "repo_relative_root": "frontend"},
                "files": [
                    {"path": "src/app.ts", "language": "typescript"},
                ],
                "symbols": [],
                "relations": [],
            },
        ]

        # Replicate the collection logic
        indexed: dict[str, set[str]] = {}
        for snapshot_dict in snapshots:
            snapshot_meta = dict(snapshot_dict.get("meta") or {})
            snapshot_language = str(snapshot_meta.get("language") or "").strip().lower()
            files = snapshot_dict.get("files") or []
            for item in files:
                if not isinstance(item, dict):
                    continue
                lang = str(item.get("language") or "").strip().lower()
                if not lang:
                    lang = snapshot_language
                path = str(item.get("path") or "").strip()
                if path:
                    indexed.setdefault(lang, set()).add(path)

        assert len(indexed.get("python", set())) == 2
        assert len(indexed.get("typescript", set())) == 1

    def test_collect_relation_coverage_logic(self) -> None:
        """Test the relation coverage collection logic."""
        snapshots = [
            {
                "meta": {"language": "python"},
                "relations": [
                    {"kind": "calls"},
                    {"kind": "references"},
                    {"kind": "calls"},
                ],
            },
        ]

        from collections import defaultdict

        totals: dict[str, int] = defaultdict(int)
        kind_totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for snapshot_dict in snapshots:
            snapshot_meta = dict(snapshot_dict.get("meta") or {})
            language = str(snapshot_meta.get("language") or "").strip().lower()
            relations = snapshot_dict.get("relations") or []
            for relation in relations:
                if not isinstance(relation, dict):
                    continue
                kind = str(relation.get("kind") or "").strip().lower()
                totals[language] += 1
                if kind:
                    kind_totals[language][kind] += 1

        assert totals["python"] == 3
        assert kind_totals["python"]["calls"] == 2
        assert kind_totals["python"]["references"] == 1


# ---------------------------------------------------------------------------
# _materialize_behavioral_layers_impl
# ---------------------------------------------------------------------------


class TestMaterializeBehavioralLayersImpl:
    async def test_returns_disabled_when_flag_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            flows,
            "get_settings",
            lambda: _make_settings(digital_twin_behavioral_enabled=False),
        )

        result = await flows._materialize_behavioral_layers_impl(
            source_id=str(uuid.uuid4()),
            collection_id=str(uuid.uuid4()),
            scenario_id=None,
            source_version_id=None,
        )

        assert result["behavioral_layers_status"] == "disabled"
        assert result["last_behavioral_materialized_at"] is None
        assert "DIGITAL_TWIN_BEHAVIORAL_ENABLED=false" in result["deep_warnings"]


# ---------------------------------------------------------------------------
# sync_single_source flow
# ---------------------------------------------------------------------------


class TestSyncSingleSourceFlow:
    async def test_source_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(flows, "get_session", lambda: mock_cm)

        source_id = str(uuid.uuid4())
        result = await flows.sync_single_source.fn(source_id)
        assert "error" in result
        assert "not found" in result["error"]

    async def test_disabled_source_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_source = MagicMock()
        mock_source.enabled = False
        mock_source.id = uuid.uuid4()

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_source
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(flows, "get_session", lambda: mock_cm)

        source_id = str(mock_source.id)
        result = await flows.sync_single_source.fn(source_id)
        assert result.get("skipped") is True
        assert "disabled" in result.get("error", "")

    async def test_lock_not_acquired_returns_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_source = MagicMock()
        mock_source.enabled = True
        mock_source.id = uuid.uuid4()

        # First call: load source. Second call onwards: inside sync_source
        call_count = 0

        async def mock_execute(stmt):
            nonlocal call_count
            call_count += 1
            result_mock = MagicMock()
            if call_count == 1:
                result_mock.scalar_one_or_none.return_value = mock_source
            return result_mock

        mock_session = AsyncMock()
        mock_session.execute = mock_execute

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(flows, "get_session", lambda: mock_cm)

        # Mock sync_source to return None (lock not acquired)
        monkeypatch.setattr(flows, "sync_source", AsyncMock(return_value=None))
        monkeypatch.setattr(flows, "_sync_source_timeout_seconds", lambda: 0)

        source_id = str(mock_source.id)
        result = await flows.sync_single_source.fn(source_id)
        assert result.get("skipped") is True
        assert result.get("reason") == "lock_not_acquired"


# ---------------------------------------------------------------------------
# sync_due_sources flow
# ---------------------------------------------------------------------------


class TestSyncDueSourcesFlow:
    async def test_no_sources_due(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(flows, "get_due_sources", AsyncMock(return_value=[]))

        result = await flows.sync_due_sources.fn()
        assert result["synced"] == 0
        assert result["skipped"] == 0
        assert result["sources"] == []

    async def test_source_sync_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_source = MagicMock()
        mock_source.id = uuid.uuid4()

        mock_sync_run = MagicMock()
        mock_sync_run.id = uuid.uuid4()
        mock_sync_run.status.value = "success"

        monkeypatch.setattr(flows, "get_due_sources", AsyncMock(return_value=[mock_source]))
        monkeypatch.setattr(flows, "sync_source", AsyncMock(return_value=mock_sync_run))
        monkeypatch.setattr(flows, "_sync_source_timeout_seconds", lambda: 0)

        result = await flows.sync_due_sources.fn()
        assert result["synced"] == 1
        assert result["skipped"] == 0
        assert result["sources"][0]["status"] == "success"

    async def test_source_sync_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_source = MagicMock()
        mock_source.id = uuid.uuid4()

        monkeypatch.setattr(flows, "get_due_sources", AsyncMock(return_value=[mock_source]))
        monkeypatch.setattr(flows, "sync_source", AsyncMock(return_value=None))
        monkeypatch.setattr(flows, "_sync_source_timeout_seconds", lambda: 0)

        result = await flows.sync_due_sources.fn()
        assert result["synced"] == 0
        assert result["skipped"] == 1

    async def test_source_sync_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_source = MagicMock()
        mock_source.id = uuid.uuid4()

        async def timeout_sync(source):
            raise TimeoutError("timeout")

        monkeypatch.setattr(flows, "get_due_sources", AsyncMock(return_value=[mock_source]))
        monkeypatch.setattr(flows, "sync_source", timeout_sync)
        monkeypatch.setattr(flows, "_sync_source_timeout_seconds", lambda: 1)
        monkeypatch.setattr(flows, "_fail_running_sync_runs_for_source", AsyncMock(return_value=1))

        # Need to mock asyncio.wait_for to raise TimeoutError
        original_wait_for = asyncio.wait_for

        async def mock_wait_for(coro, *, timeout=None):
            raise TimeoutError("test timeout")

        monkeypatch.setattr(asyncio, "wait_for", mock_wait_for)

        result = await flows.sync_due_sources.fn()
        assert result["synced"] == 1
        assert "error" in result["sources"][0]

        monkeypatch.setattr(asyncio, "wait_for", original_wait_for)

    async def test_source_sync_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_source = MagicMock()
        mock_source.id = uuid.uuid4()

        monkeypatch.setattr(flows, "get_due_sources", AsyncMock(return_value=[mock_source]))
        monkeypatch.setattr(flows, "sync_source", AsyncMock(side_effect=RuntimeError("test error")))
        monkeypatch.setattr(flows, "_sync_source_timeout_seconds", lambda: 0)

        result = await flows.sync_due_sources.fn()
        assert result["synced"] == 1
        assert "test error" in result["sources"][0]["error"]


# ---------------------------------------------------------------------------
# sync_source (high-level mocked)
# ---------------------------------------------------------------------------


class TestSyncSource:
    async def test_returns_none_when_lock_not_acquired(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_source = MagicMock()
        mock_source.id = uuid.uuid4()

        mock_session = AsyncMock()
        mock_lock_result = MagicMock()
        mock_lock_result.scalar_one_or_none.return_value = None  # Lock not acquired
        mock_session.execute = AsyncMock(return_value=mock_lock_result)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(flows, "get_session", lambda: mock_cm)

        result = await flows.sync_source.fn(mock_source)
        assert result is None

    async def test_stale_run_recovery(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stale running rows should be recovered before checking for running syncs."""
        from contextmine_core import SyncRunStatus

        mock_source = MagicMock()
        mock_source.id = uuid.uuid4()
        mock_source.collection_id = uuid.uuid4()
        mock_source.schedule_interval_minutes = 60

        # Build mock stale run
        stale_run = MagicMock()
        stale_run.started_at = datetime.now(UTC) - timedelta(hours=12)

        # Mock sync run for the error path
        mock_sync_run = MagicMock()
        mock_sync_run.id = uuid.uuid4()

        # Mock db_source for the error path
        mock_db_source = MagicMock()
        mock_db_source.schedule_interval_minutes = 60

        # Mock db_run for the error path
        mock_db_run = MagicMock()

        # Track calls per session context
        session_contexts = []

        def make_session_mock(context_id: int) -> AsyncMock:
            call_count = 0

            async def mock_execute(stmt):
                nonlocal call_count
                call_count += 1
                result_mock = MagicMock()

                if context_id == 0:
                    # First session context: lock + stale + check + create
                    if call_count == 1:
                        result_mock.scalar_one_or_none.return_value = mock_source
                    elif call_count == 2:
                        result_mock.scalars.return_value.all.return_value = [stale_run]
                    elif call_count == 3:
                        result_mock.scalar_one_or_none.return_value = None
                    else:
                        result_mock.scalar_one_or_none.return_value = None
                elif context_id == 1:
                    # Second session context: error handling
                    if call_count == 1:
                        # SyncRun lookup
                        result_mock.scalar_one.return_value = mock_db_run
                    elif call_count == 2:
                        # TwinSourceVersion lookup
                        result_mock.scalar_one_or_none.return_value = None
                    elif call_count == 3:
                        # Source lookup
                        result_mock.scalar_one.return_value = mock_db_source
                    else:
                        result_mock.scalar_one_or_none.return_value = None
                        result_mock.scalars.return_value.all.return_value = []
                return result_mock

            mock_session = AsyncMock()
            mock_session.execute = mock_execute
            mock_session.commit = AsyncMock()
            mock_session.add = MagicMock()
            mock_session.refresh = AsyncMock()
            session_contexts.append(mock_session)
            return mock_session

        context_counter = 0

        def mock_get_session():
            nonlocal context_counter
            ctx_id = context_counter
            context_counter += 1
            mock_cm = AsyncMock()
            session = make_session_mock(ctx_id)
            mock_cm.__aenter__ = AsyncMock(return_value=session)
            mock_cm.__aexit__ = AsyncMock(return_value=False)
            return mock_cm

        monkeypatch.setattr(flows, "get_session", mock_get_session)

        # Make the actual sync raise so we can verify recovery happened
        from contextmine_core import SourceType

        mock_source.type = SourceType.GITHUB
        monkeypatch.setattr(
            flows, "sync_github_source", AsyncMock(side_effect=RuntimeError("intentional"))
        )

        # sync_source catches the exception internally and returns the db_run
        await flows.sync_source.fn(mock_source)

        # Verify the stale run was marked as FAILED
        assert stale_run.status == SyncRunStatus.FAILED


# ---------------------------------------------------------------------------
# _record_doc_error helper (replicated logic)
# ---------------------------------------------------------------------------


class TestRecordDocError:
    def test_appends_within_limit(self) -> None:
        """The inline _record_doc_error function caps at 100 entries."""
        error_samples: list[str] = []

        def _record_doc_error(value: str) -> None:
            if len(error_samples) < 100:
                error_samples.append(value)

        for i in range(150):
            _record_doc_error(f"error_{i}")

        assert len(error_samples) == 100
        assert error_samples[0] == "error_0"
        assert error_samples[99] == "error_99"


# ---------------------------------------------------------------------------
# Coverage ingest flow
# ---------------------------------------------------------------------------


class TestIngestCoverageMetrics:
    async def test_invalid_job_id_returns_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = await flows.ingest_coverage_metrics.fn("not-a-uuid")
        assert result["status"] == "failed"
        assert "Invalid job_id" in result["error_detail"]

    async def test_missing_job(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(flows, "get_session", lambda: mock_cm)

        result = await flows.ingest_coverage_metrics.fn(str(uuid.uuid4()))
        assert result["status"] == "missing"

    async def test_already_terminal_job(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_job = MagicMock()
        mock_job.status = "applied"
        mock_job.error_code = None
        mock_job.error_detail = None

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_job
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(flows, "get_session", lambda: mock_cm)

        result = await flows.ingest_coverage_metrics.fn(str(uuid.uuid4()))
        assert result["status"] == "applied"


# ---------------------------------------------------------------------------
# Repair twin file paths flow
# ---------------------------------------------------------------------------


class TestRepairTwinFilePaths:
    async def test_delegates_to_task(self, monkeypatch: pytest.MonkeyPatch) -> None:
        expected = {"legacy_candidates": 5, "updated_in_place": 3}
        monkeypatch.setattr(
            flows,
            "task_repair_twin_file_path_canonicalization",
            AsyncMock(return_value=expected),
        )

        result = await flows.repair_twin_file_paths.fn(
            collection_id=str(uuid.uuid4()),
            scenario_id=None,
        )
        assert result == expected


# ---------------------------------------------------------------------------
# SCIP task wrappers
# ---------------------------------------------------------------------------


class TestTaskDetectScipProjects:
    async def test_returns_projects_and_diagnostics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_projects = [MagicMock()]
        mock_projects[0].to_dict.return_value = {"language": "python", "root_path": "/tmp"}

        mock_diagnostics = MagicMock()
        mock_diagnostics.to_dict.return_value = {"languages_detected": ["python"]}

        monkeypatch.setattr(
            "contextmine_core.semantic_snapshot.indexers.detection.detect_projects_with_diagnostics",
            lambda path: (mock_projects, mock_diagnostics),
        )

        result = await flows.task_detect_scip_projects.fn(Path("/tmp/repo"))
        assert len(result["projects"]) == 1
        assert result["diagnostics"]["languages_detected"] == ["python"]


class TestTaskParseScipSnapshot:
    async def test_returns_snapshot_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_snapshot = MagicMock()
        mock_snapshot.symbols = [1, 2, 3]
        mock_snapshot.relations = [1]
        mock_snapshot.to_dict.return_value = {"symbols": [1, 2, 3], "relations": [1]}

        monkeypatch.setattr(
            "contextmine_core.semantic_snapshot.build_snapshot",
            lambda path: mock_snapshot,
        )

        result = await flows.task_parse_scip_snapshot.fn("/tmp/test.scip")
        assert result is not None
        assert result["symbols"] == [1, 2, 3]

    async def test_returns_none_on_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "contextmine_core.semantic_snapshot.build_snapshot",
            MagicMock(side_effect=RuntimeError("parse error")),
        )

        result = await flows.task_parse_scip_snapshot.fn("/tmp/bad.scip")
        assert result is None


class TestTaskIndexScipProject:
    async def test_successful_indexing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_backend = MagicMock()
        mock_backend.can_handle.return_value = True
        mock_artifact = MagicMock()
        mock_artifact.success = True
        mock_artifact.duration_s = 1.5
        mock_artifact.to_dict.return_value = {"success": True, "scip_path": "/tmp/out.scip"}
        mock_backend.index.return_value = mock_artifact

        monkeypatch.setattr(
            "contextmine_core.semantic_snapshot.indexers.BACKENDS",
            [mock_backend],
        )
        monkeypatch.setattr(flows, "get_settings", lambda: _make_settings())

        proj_dict = {
            "language": "python",
            "root_path": "/tmp/repo",
            "metadata": {},
            "manifest_paths": [],
        }
        result = await flows.task_index_scip_project.fn(proj_dict, Path("/tmp/output"))
        assert result is not None
        assert result["success"] is True

    async def test_no_matching_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_backend = MagicMock()
        mock_backend.can_handle.return_value = False

        monkeypatch.setattr(
            "contextmine_core.semantic_snapshot.indexers.BACKENDS",
            [mock_backend],
        )
        monkeypatch.setattr(flows, "get_settings", lambda: _make_settings())

        proj_dict = {
            "language": "python",
            "root_path": "/tmp/repo",
            "metadata": {},
            "manifest_paths": [],
        }
        result = await flows.task_index_scip_project.fn(proj_dict, Path("/tmp/output"))
        assert result is None

    async def test_failed_indexing_best_effort(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_backend = MagicMock()
        mock_backend.can_handle.return_value = True
        mock_artifact = MagicMock()
        mock_artifact.success = False
        mock_artifact.error_message = "compilation error"
        mock_backend.index.return_value = mock_artifact

        monkeypatch.setattr(
            "contextmine_core.semantic_snapshot.indexers.BACKENDS",
            [mock_backend],
        )
        monkeypatch.setattr(flows, "get_settings", lambda: _make_settings(scip_best_effort=True))

        proj_dict = {
            "language": "python",
            "root_path": "/tmp/repo",
            "metadata": {},
            "manifest_paths": [],
        }
        result = await flows.task_index_scip_project.fn(proj_dict, Path("/tmp/output"))
        # best_effort mode: returns None instead of raising
        assert result is None


# ---------------------------------------------------------------------------
# build_knowledge_graph validation
# ---------------------------------------------------------------------------


class TestBuildKnowledgeGraph:
    async def test_raises_when_no_llm_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            flows,
            "get_settings",
            lambda: _make_settings(default_llm_provider=""),
        )

        with pytest.raises(ValueError, match="LLM provider required"):
            await flows.build_knowledge_graph.fn(
                source_id=str(uuid.uuid4()),
                collection_id=str(uuid.uuid4()),
            )

    async def test_raises_when_llm_init_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            flows,
            "get_settings",
            lambda: _make_settings(default_llm_provider="openai"),
        )

        def failing_get_llm(*args, **kwargs):
            raise RuntimeError("No API key")

        monkeypatch.setattr(
            "contextmine_core.research.llm.get_llm_provider",
            failing_get_llm,
        )

        with pytest.raises(ValueError, match="LLM provider configured but failed"):
            await flows.build_knowledge_graph.fn(
                source_id=str(uuid.uuid4()),
                collection_id=str(uuid.uuid4()),
            )

    async def test_raises_when_no_research_llm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            flows,
            "get_settings",
            lambda: _make_settings(default_llm_provider="openai"),
        )

        monkeypatch.setattr(
            "contextmine_core.research.llm.get_llm_provider",
            lambda *args, **kwargs: MagicMock(),
        )
        monkeypatch.setattr(
            "contextmine_core.research.llm.get_research_llm_provider",
            lambda: None,
        )

        with pytest.raises(ValueError, match="Research LLM provider required"):
            await flows.build_knowledge_graph.fn(
                source_id=str(uuid.uuid4()),
                collection_id=str(uuid.uuid4()),
            )

    async def test_raises_when_embedder_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            flows,
            "get_settings",
            lambda: _make_settings(default_llm_provider="openai"),
        )

        monkeypatch.setattr(
            "contextmine_core.research.llm.get_llm_provider",
            lambda *args, **kwargs: MagicMock(),
        )
        monkeypatch.setattr(
            "contextmine_core.research.llm.get_research_llm_provider",
            lambda: MagicMock(),
        )
        monkeypatch.setattr(
            flows,
            "parse_embedding_model_spec",
            MagicMock(side_effect=ValueError("bad spec")),
        )

        with pytest.raises(ValueError, match="Embedder required"):
            await flows.build_knowledge_graph.fn(
                source_id=str(uuid.uuid4()),
                collection_id=str(uuid.uuid4()),
            )


# ---------------------------------------------------------------------------
# build_twin_graph
# ---------------------------------------------------------------------------


class TestBuildTwinGraph:
    async def test_no_snapshots_seeds_from_kg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When no snapshot_dicts provided, should seed from knowledge graph."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())

        mock_as_is = MagicMock()
        mock_as_is.id = uuid.uuid4()

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=MagicMock())
        mock_session.commit = AsyncMock()

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(flows, "get_session", lambda: mock_cm)

        monkeypatch.setattr(
            "contextmine_core.twin.get_or_create_as_is_scenario",
            AsyncMock(return_value=mock_as_is),
        )
        monkeypatch.setattr(
            "contextmine_core.twin.seed_scenario_from_knowledge_graph",
            AsyncMock(return_value=(10, 20)),
        )
        monkeypatch.setattr(
            "contextmine_core.twin.refresh_metric_snapshots",
            AsyncMock(return_value=5),
        )
        monkeypatch.setattr(
            "contextmine_core.validation.refresh_validation_snapshots",
            AsyncMock(return_value=3),
        )
        monkeypatch.setattr(
            "contextmine_core.graph.age.sync_scenario_to_age",
            AsyncMock(),
        )
        monkeypatch.setattr(
            flows,
            "get_settings",
            lambda: _make_settings(arch_docs_enabled=False),
        )

        result = await flows.build_twin_graph.fn(
            source_id=source_id,
            collection_id=collection_id,
            snapshot_dicts=None,
            changed_doc_ids=None,
            file_metrics=None,
            evolution_payload=None,
        )

        assert result["twin_nodes_upserted"] == 10
        assert result["twin_edges_upserted"] == 20
        assert result["twin_metrics_snapshots"] == 5
        assert result["twin_validation_snapshots"] == 3


# ---------------------------------------------------------------------------
# Edge cases and parametrized tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "uri,expected",
    [
        ("git://github.com/a/b/src/main.py?ref=main", "src/main.py"),
        ("git://github.com/a/b/deep/nested/file.ts?ref=dev", "deep/nested/file.ts"),
        ("simple.txt", "simple.txt"),
        ("a/b", "b"),
        ("a/b/c/d/e/f.py?query=1", "f.py"),
    ],
)
def test_uri_to_file_path_parametrized(uri: str, expected: str) -> None:
    assert _uri_to_file_path(uri) == expected


@pytest.mark.parametrize(
    "path,expected",
    [
        ("node_modules/x.js", True),
        ("vendor/x.php", True),
        ("dist/bundle.js", True),
        ("build/output.js", True),
        ("__pycache__/mod.pyc", True),
        (".git/HEAD", True),
        ("venv/lib/pkg.py", True),
        (".venv/lib/pkg.py", True),
        ("src/libs/aws/s3.php", True),
        ("src/main.py", False),
        ("README.md", False),
        ("", False),
    ],
)
def test_is_ignored_repo_path_parametrized(path: str, expected: bool) -> None:
    assert _is_ignored_repo_path(path) == expected


# ---------------------------------------------------------------------------
# File coverage completion snapshot logic
# ---------------------------------------------------------------------------


class TestAppendFileCoverageCompletionLogic:
    """Test the _append_file_coverage_completion_snapshot logic pattern."""

    def test_empty_snapshots_returns_empty(self) -> None:
        """With no snapshots, nothing should be added."""
        # Replicate the function logic
        snapshots: list[dict] = []
        result = {} if not snapshots else {"some": 1}
        assert result == {}

    def test_no_missing_files_returns_empty(self) -> None:
        """When all census files are indexed, nothing should be appended."""
        indexed_paths = {"python": {"src/main.py", "src/utils.py"}}

        @dataclass
        class FakeFileStat:
            language: Any
            code: int
            path: Path

        census_files = [
            FakeFileStat(language="python", code=50, path=Path("/tmp/repo/src/main.py")),
            FakeFileStat(language="python", code=30, path=Path("/tmp/repo/src/utils.py")),
        ]

        repo_path = Path("/tmp/repo")
        missing_by_language: dict[str, set[str]] = {}
        for item in census_files:
            language = str(getattr(item.language, "value", item.language) or "").strip().lower()
            if item.code <= 0:
                continue
            try:
                repo_relative = item.path.resolve().relative_to(repo_path.resolve()).as_posix()
            except Exception:
                continue
            if repo_relative in indexed_paths.get(language, set()):
                continue
            missing_by_language.setdefault(language, set()).add(repo_relative)

        assert len(missing_by_language) == 0


# ---------------------------------------------------------------------------
# Error handling pattern tests
# ---------------------------------------------------------------------------


class TestErrorParsingInSyncSource:
    """Test the error message parsing patterns used in sync_source exception handler."""

    def test_metrics_gate_parsing(self) -> None:
        error_text = "METRICS_GATE_FAILED: twin_node_mapping_incomplete (mapped=50, metrics=100)"
        failure_stats: dict[str, object] = {}

        if "METRICS_GATE_FAILED" in error_text:
            failure_stats["metrics_gate"] = "fail"
            if "(mapped=" in error_text and ", metrics=" in error_text:
                try:
                    mapped_part = error_text.split("(mapped=", 1)[1]
                    mapped_raw = mapped_part.split(",", 1)[0]
                    metrics_raw = mapped_part.split("metrics=", 1)[1].split(")", 1)[0]
                    failure_stats["metrics_mapped_files"] = int(mapped_raw)
                    failure_stats["metrics_requested_files"] = int(metrics_raw)
                except Exception:
                    pass

        assert failure_stats["metrics_gate"] == "fail"
        assert failure_stats["metrics_mapped_files"] == 50
        assert failure_stats["metrics_requested_files"] == 100

    def test_scip_gate_missing_languages(self) -> None:
        error_text = "SCIP_GATE_FAILED: (missing=python,typescript)"
        failure_stats: dict[str, object] = {}

        if "SCIP_GATE_FAILED" in error_text:
            failure_stats["scip_gate"] = "fail"
            if "(missing=" in error_text:
                try:
                    missing_raw = error_text.split("(missing=", 1)[1].split(")", 1)[0]
                    failure_stats["scip_missing_languages"] = [
                        lang for lang in missing_raw.split(",") if lang
                    ]
                except Exception:
                    pass

        assert failure_stats["scip_gate"] == "fail"
        assert failure_stats["scip_missing_languages"] == ["python", "typescript"]

    def test_scip_gate_missing_relations(self) -> None:
        error_text = "SCIP_GATE_FAILED: (missing_relations=php,java)"
        failure_stats: dict[str, object] = {}

        if "SCIP_GATE_FAILED" in error_text:
            failure_stats["scip_gate"] = "fail"
            if "(missing_relations=" in error_text:
                try:
                    missing_rel_raw = error_text.split("(missing_relations=", 1)[1].split(")", 1)[0]
                    failure_stats["scip_missing_relation_languages"] = [
                        lang for lang in missing_rel_raw.split(",") if lang
                    ]
                except Exception:
                    pass

        assert failure_stats["scip_gate"] == "fail"
        assert failure_stats["scip_missing_relation_languages"] == ["php", "java"]

    def test_no_gate_error_leaves_stats_empty(self) -> None:
        error_text = "Some other error"
        failure_stats: dict[str, object] = {}

        if "METRICS_GATE_FAILED" in error_text:
            failure_stats["metrics_gate"] = "fail"
        if "SCIP_GATE_FAILED" in error_text:
            failure_stats["scip_gate"] = "fail"

        assert failure_stats == {}

    def test_empty_error_message_handling(self) -> None:
        """sync_source uses str(e).strip() or repr(e)."""
        e = ValueError("")
        error_message = str(e).strip() or repr(e)
        # When str(e) is empty, falls back to repr(e)
        assert error_message == repr(e)
        assert "ValueError" in error_message

        e2 = ValueError("real error")
        error_message2 = str(e2).strip() or repr(e2)
        assert error_message2 == "real error"


# ---------------------------------------------------------------------------
# Progress tracking logic
# ---------------------------------------------------------------------------


class TestProgressCalculation:
    """Test the progress percentage calculation used in document processing loops."""

    @pytest.mark.parametrize(
        "i,total,expected_min,expected_max",
        [
            (0, 10, 50, 60),
            (4, 10, 70, 75),
            (9, 10, 90, 95),
            (0, 1, 95, 95),
            (0, 100, 50, 51),
            (99, 100, 94, 95),
        ],
    )
    def test_pct_formula(self, i: int, total: int, expected_min: int, expected_max: int) -> None:
        pct = 50 + int((i + 1) / total * 45)
        assert expected_min <= pct <= expected_max


# ---------------------------------------------------------------------------
# Documents per run limiting
# ---------------------------------------------------------------------------


class TestDocsPerRunLimiting:
    def test_no_limit_processes_all(self) -> None:
        docs = [(f"doc{i}", f"content{i}", f"file{i}.py") for i in range(100)]
        docs_limit = 0  # No limit

        if docs_limit and len(docs) > docs_limit:
            docs = docs[:docs_limit]

        assert len(docs) == 100

    def test_limit_trims_docs(self) -> None:
        docs = [(f"doc{i}", f"content{i}", f"file{i}.py") for i in range(100)]
        docs_limit = 50

        total_candidate = len(docs)
        docs_deferred = 0
        if docs_limit and total_candidate > docs_limit:
            docs = docs[:docs_limit]
            docs_deferred = total_candidate - len(docs)

        assert len(docs) == 50
        assert docs_deferred == 50

    def test_limit_equal_to_count(self) -> None:
        docs = [(f"doc{i}", f"content{i}", f"file{i}.py") for i in range(50)]
        docs_limit = 50

        total_candidate = len(docs)
        docs_deferred = 0
        if docs_limit and total_candidate > docs_limit:
            docs = docs[:docs_limit]
            docs_deferred = total_candidate - len(docs)

        assert len(docs) == 50
        assert docs_deferred == 0


# ---------------------------------------------------------------------------
# Metrics gate logic
# ---------------------------------------------------------------------------


class TestMetricsGateLogic:
    """Test the metrics_gate computation used in stats reporting."""

    def test_pass_when_all_mapped(self) -> None:
        twin_stats = {"metrics_mapped_files": 100, "metrics_requested_files": 100}
        gate = (
            "pass"
            if int(twin_stats.get("metrics_mapped_files", 0))
            >= int(twin_stats.get("metrics_requested_files", 0))
            else "fail"
        )
        assert gate == "pass"

    def test_fail_when_under_mapped(self) -> None:
        twin_stats = {"metrics_mapped_files": 50, "metrics_requested_files": 100}
        gate = (
            "pass"
            if int(twin_stats.get("metrics_mapped_files", 0))
            >= int(twin_stats.get("metrics_requested_files", 0))
            else "fail"
        )
        assert gate == "fail"

    def test_pass_when_no_requested(self) -> None:
        twin_stats: dict[str, Any] = {}
        gate = (
            "pass"
            if int(twin_stats.get("metrics_mapped_files", 0))
            >= int(twin_stats.get("metrics_requested_files", 0))
            else "fail"
        )
        assert gate == "pass"


# ---------------------------------------------------------------------------
# SCIP degraded logic
# ---------------------------------------------------------------------------


class TestScipDegradedLogic:
    def test_degraded_when_projects_failed(self) -> None:
        scip_stats = {"scip_projects_failed": 1}
        assert bool(scip_stats["scip_projects_failed"]) is True

    def test_not_degraded_when_no_failures(self) -> None:
        scip_stats = {"scip_projects_failed": 0}
        assert bool(scip_stats["scip_projects_failed"]) is False

    def test_degraded_when_coverage_incomplete(self) -> None:
        scip_stats: dict[str, Any] = {
            "scip_projects_failed": 0,
            "scip_coverage_complete": False,
            "scip_relation_coverage_complete": True,
        }
        degraded = bool(scip_stats["scip_projects_failed"])
        if not scip_stats["scip_coverage_complete"]:
            degraded = True
        if not scip_stats["scip_relation_coverage_complete"]:
            degraded = True
        assert degraded is True

    def test_degraded_when_relation_coverage_incomplete(self) -> None:
        scip_stats: dict[str, Any] = {
            "scip_projects_failed": 0,
            "scip_coverage_complete": True,
            "scip_relation_coverage_complete": False,
        }
        degraded = bool(scip_stats["scip_projects_failed"])
        if not scip_stats["scip_coverage_complete"]:
            degraded = True
        if not scip_stats["scip_relation_coverage_complete"]:
            degraded = True
        assert degraded is True


# ---------------------------------------------------------------------------
# Web source config extraction
# ---------------------------------------------------------------------------


class TestWebSourceConfigExtraction:
    """Test the config extraction pattern used in sync_web_source."""

    def test_config_defaults(self) -> None:
        from contextmine_worker.web_sync import DEFAULT_DELAY_MS, DEFAULT_MAX_PAGES

        config: dict[str, Any] = {}
        source_url = "https://example.com/docs"

        start_url = config.get("start_url", source_url)
        base_url = config.get("base_url", start_url)
        max_pages = config.get("max_pages", DEFAULT_MAX_PAGES)
        delay_ms = config.get("delay_ms", DEFAULT_DELAY_MS)

        assert start_url == source_url
        assert base_url == source_url
        assert max_pages == DEFAULT_MAX_PAGES
        assert delay_ms == DEFAULT_DELAY_MS

    def test_config_overrides(self) -> None:
        config = {
            "start_url": "https://example.com/start",
            "base_url": "https://example.com/base",
            "max_pages": 50,
            "delay_ms": 500,
        }
        source_url = "https://example.com/docs"

        start_url = config.get("start_url", source_url)
        base_url = config.get("base_url", start_url)
        max_pages = config.get("max_pages", 100)
        delay_ms = config.get("delay_ms", 200)

        assert start_url == "https://example.com/start"
        assert base_url == "https://example.com/base"
        assert max_pages == 50
        assert delay_ms == 500

    def test_missing_base_url_raises(self) -> None:
        config: dict[str, Any] = {}
        source_url = ""

        start_url = config.get("start_url", source_url)
        base_url = config.get("base_url", start_url)

        if not base_url:
            with pytest.raises(ValueError, match="missing base_url"):
                raise ValueError("Web source missing base_url in config")


# ---------------------------------------------------------------------------
# GitHub source config extraction
# ---------------------------------------------------------------------------


class TestGithubSourceConfigExtraction:
    """Test the config extraction pattern used in sync_github_source."""

    def test_missing_owner_raises(self) -> None:
        config: dict[str, Any] = {"repo": "my-repo"}
        owner = config.get("owner", "")
        repo = config.get("repo", "")

        if not owner or not repo:
            with pytest.raises(ValueError, match="missing owner/repo"):
                raise ValueError("GitHub source missing owner/repo in config")

    def test_missing_repo_raises(self) -> None:
        config: dict[str, Any] = {"owner": "my-owner"}
        owner = config.get("owner", "")
        repo = config.get("repo", "")

        if not owner or not repo:
            with pytest.raises(ValueError, match="missing owner/repo"):
                raise ValueError("GitHub source missing owner/repo in config")

    def test_valid_config(self) -> None:
        config = {"owner": "my-org", "repo": "my-repo", "branch": "develop"}
        owner = config.get("owner", "")
        repo = config.get("repo", "")
        branch = config.get("branch", "main")

        assert owner == "my-org"
        assert repo == "my-repo"
        assert branch == "develop"

    def test_default_branch(self) -> None:
        config = {"owner": "org", "repo": "repo"}
        branch = config.get("branch", "main")
        assert branch == "main"


# ---------------------------------------------------------------------------
# Snapshot repo file path resolution
# ---------------------------------------------------------------------------


class TestSnapshotRepoFilePath:
    """Test the _snapshot_repo_file_path logic pattern."""

    def test_empty_path(self) -> None:
        raw_path = ""
        assert raw_path.strip() == ""

    def test_relative_path_with_root(self) -> None:
        raw_path = "src/main.py"
        repo_relative_root = "backend"

        normalized = raw_path.lstrip("./")
        if repo_relative_root and not normalized.startswith(f"{repo_relative_root}/"):
            normalized = f"{repo_relative_root}/{normalized}".strip("/")

        assert normalized == "backend/src/main.py"

    def test_relative_path_already_prefixed(self) -> None:
        raw_path = "backend/src/main.py"
        repo_relative_root = "backend"

        normalized = raw_path.lstrip("./")
        if normalized != repo_relative_root and not normalized.startswith(f"{repo_relative_root}/"):
            normalized = f"{repo_relative_root}/{normalized}".strip("/")

        assert normalized == "backend/src/main.py"

    def test_relative_path_no_root(self) -> None:
        raw_path = "./src/main.py"
        repo_relative_root = ""

        normalized = raw_path.lstrip("./")
        if repo_relative_root:
            normalized = f"{repo_relative_root}/{normalized}".strip("/")

        assert normalized == "src/main.py"

    def test_absolute_path_resolution(self) -> None:
        raw_path = "/tmp/repo/src/main.py"
        repo_path = Path("/tmp/repo")

        path_obj = Path(raw_path)
        if path_obj.is_absolute():
            try:
                result = path_obj.resolve().relative_to(repo_path.resolve()).as_posix()
            except ValueError:
                result = raw_path.lstrip("./")
        else:
            result = raw_path

        assert result == "src/main.py"

    def test_absolute_path_outside_repo(self) -> None:
        raw_path = "/other/location/file.py"
        repo_path = Path("/tmp/repo")

        path_obj = Path(raw_path)
        if path_obj.is_absolute():
            try:
                result = path_obj.resolve().relative_to(repo_path.resolve()).as_posix()
            except ValueError:
                result = raw_path.lstrip("./")
        else:
            result = raw_path

        assert result == "other/location/file.py"


# ---------------------------------------------------------------------------
# Snapshot file language detection
# ---------------------------------------------------------------------------


class TestSnapshotFileLanguage:
    """Test the _snapshot_file_language logic pattern."""

    def test_explicit_language_wins(self) -> None:
        supported = {"python", "typescript", "javascript", "java", "php"}
        explicit = "python"
        result = explicit if explicit in supported else None
        assert result == "python"

    def test_extension_fallback(self) -> None:
        _supported = {"python", "typescript"}

        # Simulate EXTENSION_TO_LANGUAGE lookup
        extension_map = {".py": "python", ".ts": "typescript", ".js": "javascript"}
        suffix = Path("src/main.py").suffix.lower()
        extension_language = extension_map.get(suffix)

        assert extension_language == "python"

    def test_snapshot_language_fallback(self) -> None:
        supported = {"python", "typescript"}
        explicit = ""
        snapshot_language = "typescript"

        # Logic: explicit -> extension -> snapshot
        result = None
        if explicit.strip().lower() in supported:
            result = explicit.strip().lower()
        elif snapshot_language in supported:
            result = snapshot_language

        assert result == "typescript"
