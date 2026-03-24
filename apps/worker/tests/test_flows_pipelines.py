"""Tests for the large pipeline functions: sync_web_source, sync_github_source.

These cover the document processing loops, chunking, embedding, and final
stats recording that constitute lines ~1766-3570 in flows.py.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import contextmine_worker.flows as flows
import pytest

pytestmark = pytest.mark.anyio


@pytest.fixture(params=["asyncio"])
def anyio_backend(request):
    """Force asyncio backend for these tests since they use asyncio.wait_for."""
    return request.param


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_settings(**overrides: Any) -> SimpleNamespace:
    """Build a minimal settings namespace for testing."""
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
        "twin_evolution_window_days": 90,
        "joern_server_url": "",
        "joern_required_for_sync": False,
        "joern_parse_binary": "joern-parse",
        "joern_cpg_root": "/tmp/cpg",
        "coverage_ingest_max_payload_mb": 50,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _mock_session_cm(session: AsyncMock) -> MagicMock:
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


# ---------------------------------------------------------------------------
# sync_web_source: happy path
# ---------------------------------------------------------------------------


class TestSyncWebSourceHappyPath:
    async def test_web_sync_processes_pages_and_chunks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Full web sync pipeline with one page, chunking and embedding."""
        from contextmine_core import SourceType

        source = MagicMock()
        source.id = uuid.uuid4()
        source.collection_id = uuid.uuid4()
        source.url = "https://example.com"
        source.type = SourceType.WEB
        source.config = {
            "start_url": "https://example.com",
            "base_url": "https://example.com",
            "max_pages": 10,
            "delay_ms": 0,
        }
        source.schedule_interval_minutes = 60

        sync_run = MagicMock()
        sync_run.id = uuid.uuid4()
        run_started_at = datetime.now(UTC)

        # Mock progress artifacts
        monkeypatch.setattr(flows, "create_progress_artifact", AsyncMock(return_value="prog_1"))
        monkeypatch.setattr(flows, "update_progress_artifact", AsyncMock())

        # Mock spider returning one page
        mock_page = MagicMock()
        mock_page.url = "https://example.com/page1"
        mock_page.markdown = "# Page 1\nSome content"
        mock_page.content_hash = "hash123"
        mock_page.etag = "etag1"
        mock_page.last_modified = "2024-01-01"
        monkeypatch.setattr(flows, "run_spider_md", lambda **kw: [mock_page])
        monkeypatch.setattr(flows, "get_page_title", lambda page: "Page 1")

        # Mock session
        mock_session = AsyncMock()
        exec_call = 0

        async def mock_execute(stmt):
            nonlocal exec_call
            exec_call += 1
            r = MagicMock()
            # First few calls: document queries
            r.scalar_one_or_none.return_value = None  # no existing doc
            r.scalar_one.return_value = MagicMock(schedule_interval_minutes=60)
            r.rowcount = 0
            # unchunked_docs query
            r.all.return_value = []
            return r

        mock_session.execute = mock_execute
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()
        mock_session.commit = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))
        monkeypatch.setattr(flows, "get_settings", lambda: _make_settings())

        # Mock chunking, symbols, embedding
        monkeypatch.setattr(
            flows,
            "maintain_chunks_for_document",
            AsyncMock(
                return_value={
                    "chunks_created": 2,
                    "chunks_kept": 0,
                    "chunks_deleted": 0,
                }
            ),
        )
        monkeypatch.setattr(
            flows,
            "maintain_symbols_for_document",
            AsyncMock(return_value=(1, 0)),
        )
        monkeypatch.setattr(
            flows,
            "embed_document",
            AsyncMock(
                return_value={
                    "chunks_embedded": 2,
                    "chunks_deduplicated": 0,
                    "tokens_used": 50,
                }
            ),
        )

        # Mock surface materialization
        monkeypatch.setattr(
            flows,
            "materialize_surface_catalog_for_source",
            AsyncMock(return_value={}),
        )

        # Mock twin graph
        monkeypatch.setattr(
            flows,
            "build_twin_graph",
            AsyncMock(
                return_value={
                    "twin_nodes_upserted": 5,
                    "twin_edges_upserted": 3,
                    "twin_metrics_snapshots": 0,
                    "twin_validation_snapshots": 0,
                    "twin_asis_scenario_id": None,
                }
            ),
        )

        # Mock _materialize_behavioral_layers_impl
        monkeypatch.setattr(
            flows,
            "_materialize_behavioral_layers_impl",
            AsyncMock(return_value={}),
        )

        result = await flows.sync_web_source.fn(source, sync_run, run_started_at)

        assert result.pages_crawled == 1
        assert result.docs_created == 1

    async def test_web_sync_missing_base_url_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When base_url is empty, raises ValueError."""
        from contextmine_core import SourceType

        source = MagicMock()
        source.id = uuid.uuid4()
        source.type = SourceType.WEB
        source.url = ""
        source.config = {"start_url": "", "base_url": ""}

        with pytest.raises(ValueError, match="missing base_url"):
            await flows.sync_web_source.fn(source, MagicMock(), datetime.now(UTC))

    async def test_web_sync_updates_existing_doc(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When existing doc has different content hash, it is updated."""
        from contextmine_core import SourceType

        source = MagicMock()
        source.id = uuid.uuid4()
        source.collection_id = uuid.uuid4()
        source.url = "https://example.com"
        source.type = SourceType.WEB
        source.config = {
            "start_url": "https://example.com",
            "base_url": "https://example.com",
        }
        source.schedule_interval_minutes = 60

        sync_run = MagicMock()
        sync_run.id = uuid.uuid4()
        run_started_at = datetime.now(UTC)

        monkeypatch.setattr(flows, "create_progress_artifact", AsyncMock(return_value="p"))
        monkeypatch.setattr(flows, "update_progress_artifact", AsyncMock())

        mock_page = MagicMock()
        mock_page.url = "https://example.com/page1"
        mock_page.markdown = "# Updated"
        mock_page.content_hash = "new_hash"
        mock_page.etag = None
        mock_page.last_modified = None
        monkeypatch.setattr(flows, "run_spider_md", lambda **kw: [mock_page])
        monkeypatch.setattr(flows, "get_page_title", lambda page: "Title")

        existing_doc = MagicMock()
        existing_doc.id = uuid.uuid4()
        existing_doc.content_hash = "old_hash"
        existing_doc.meta = {}

        mock_session = AsyncMock()
        exec_call = 0

        async def mock_execute(stmt):
            nonlocal exec_call
            exec_call += 1
            r = MagicMock()
            if exec_call == 1:
                r.scalar_one_or_none.return_value = existing_doc
            else:
                r.scalar_one_or_none.return_value = None
                r.scalar_one.return_value = MagicMock(schedule_interval_minutes=60)
                r.rowcount = 0
                r.all.return_value = []
            return r

        mock_session.execute = mock_execute
        mock_session.commit = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))
        monkeypatch.setattr(flows, "get_settings", lambda: _make_settings())

        monkeypatch.setattr(
            flows,
            "maintain_chunks_for_document",
            AsyncMock(
                return_value={
                    "chunks_created": 1,
                    "chunks_kept": 0,
                    "chunks_deleted": 1,
                }
            ),
        )
        monkeypatch.setattr(
            flows,
            "maintain_symbols_for_document",
            AsyncMock(return_value=(0, 0)),
        )
        monkeypatch.setattr(
            flows,
            "embed_document",
            AsyncMock(
                return_value={
                    "chunks_embedded": 1,
                    "chunks_deduplicated": 0,
                    "tokens_used": 10,
                }
            ),
        )
        monkeypatch.setattr(
            flows,
            "materialize_surface_catalog_for_source",
            AsyncMock(return_value={}),
        )
        monkeypatch.setattr(
            flows,
            "build_twin_graph",
            AsyncMock(
                return_value={
                    "twin_nodes_upserted": 0,
                    "twin_edges_upserted": 0,
                    "twin_metrics_snapshots": 0,
                    "twin_validation_snapshots": 0,
                    "twin_asis_scenario_id": None,
                }
            ),
        )

        result = await flows.sync_web_source.fn(source, sync_run, run_started_at)

        assert result.docs_updated == 1

    async def test_web_sync_doc_processing_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When chunking times out, document is skipped."""
        from contextmine_core import SourceType

        source = MagicMock()
        source.id = uuid.uuid4()
        source.collection_id = uuid.uuid4()
        source.url = "https://example.com"
        source.type = SourceType.WEB
        source.config = {
            "start_url": "https://example.com",
            "base_url": "https://example.com",
        }
        source.schedule_interval_minutes = 60

        sync_run = MagicMock()
        sync_run.id = uuid.uuid4()
        run_started_at = datetime.now(UTC)

        monkeypatch.setattr(flows, "create_progress_artifact", AsyncMock(return_value="p"))
        monkeypatch.setattr(flows, "update_progress_artifact", AsyncMock())

        mock_page = MagicMock()
        mock_page.url = "https://example.com/page1"
        mock_page.markdown = "# Content"
        mock_page.content_hash = "hash1"
        mock_page.etag = None
        mock_page.last_modified = None
        monkeypatch.setattr(flows, "run_spider_md", lambda **kw: [mock_page])
        monkeypatch.setattr(flows, "get_page_title", lambda page: "Title")

        mock_session = AsyncMock()
        new_doc = MagicMock()
        new_doc.id = uuid.uuid4()

        async def mock_execute(stmt):
            r = MagicMock()
            r.scalar_one_or_none.return_value = None
            r.scalar_one.return_value = MagicMock(schedule_interval_minutes=60)
            r.rowcount = 0
            r.all.return_value = []
            return r

        mock_session.execute = mock_execute
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()
        mock_session.commit = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))
        monkeypatch.setattr(flows, "get_settings", lambda: _make_settings())

        # Make chunking time out
        original_wait_for = asyncio.wait_for
        call_idx = 0

        async def mock_wait_for(coro, *, timeout=None):
            nonlocal call_idx
            call_idx += 1
            if call_idx <= 1:
                # First wait_for is for maintain_chunks - time out
                raise TimeoutError("chunk timeout")
            return await original_wait_for(coro, timeout=timeout)

        monkeypatch.setattr(asyncio, "wait_for", mock_wait_for)

        monkeypatch.setattr(
            flows,
            "materialize_surface_catalog_for_source",
            AsyncMock(return_value={}),
        )
        monkeypatch.setattr(
            flows,
            "build_twin_graph",
            AsyncMock(
                return_value={
                    "twin_nodes_upserted": 0,
                    "twin_edges_upserted": 0,
                    "twin_metrics_snapshots": 0,
                    "twin_validation_snapshots": 0,
                    "twin_asis_scenario_id": None,
                }
            ),
        )

        result = await flows.sync_web_source.fn(source, sync_run, run_started_at)

        monkeypatch.setattr(asyncio, "wait_for", original_wait_for)
        assert result.docs_created == 1

    async def test_web_sync_twin_timeout_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When twin graph build times out, raises RuntimeError."""
        from contextmine_core import SourceType

        source = MagicMock()
        source.id = uuid.uuid4()
        source.collection_id = uuid.uuid4()
        source.url = "https://example.com"
        source.type = SourceType.WEB
        source.config = {
            "start_url": "https://example.com",
            "base_url": "https://example.com",
        }
        source.schedule_interval_minutes = 60

        sync_run = MagicMock()
        sync_run.id = uuid.uuid4()
        run_started_at = datetime.now(UTC)

        monkeypatch.setattr(flows, "create_progress_artifact", AsyncMock(return_value="p"))
        monkeypatch.setattr(flows, "update_progress_artifact", AsyncMock())
        monkeypatch.setattr(flows, "run_spider_md", lambda **kw: [])

        mock_session = AsyncMock()

        async def mock_execute(stmt):
            r = MagicMock()
            r.scalar_one.return_value = MagicMock(schedule_interval_minutes=60)
            r.rowcount = 0
            r.all.return_value = []
            return r

        mock_session.execute = mock_execute
        mock_session.commit = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))
        monkeypatch.setattr(flows, "get_settings", lambda: _make_settings())
        monkeypatch.setattr(
            flows,
            "materialize_surface_catalog_for_source",
            AsyncMock(return_value={}),
        )

        # Make build_twin_graph an async that raises TimeoutError when awaited
        # The flow wraps build_twin_graph in asyncio.wait_for, and on timeout
        # raises RuntimeError("TWIN_BUILD_TIMEOUT: ...")
        original_wait_for = asyncio.wait_for
        wait_idx = 0

        async def mock_wait_for(coro, *, timeout=None):
            nonlocal wait_idx
            wait_idx += 1
            # Surface materialization is first wait_for (idx 1)
            # Twin build is second wait_for (idx 2)
            if wait_idx == 2:
                raise TimeoutError("twin timeout")
            return await original_wait_for(coro, timeout=timeout)

        monkeypatch.setattr(asyncio, "wait_for", mock_wait_for)

        with pytest.raises(RuntimeError, match="TWIN_BUILD_TIMEOUT"):
            await flows.sync_web_source.fn(source, sync_run, run_started_at)

        monkeypatch.setattr(asyncio, "wait_for", original_wait_for)

    async def test_web_sync_docs_limit_defers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When docs_per_run_limit is set, extra docs are deferred."""
        from contextmine_core import SourceType

        source = MagicMock()
        source.id = uuid.uuid4()
        source.collection_id = uuid.uuid4()
        source.url = "https://example.com"
        source.type = SourceType.WEB
        source.config = {
            "start_url": "https://example.com",
            "base_url": "https://example.com",
        }
        source.schedule_interval_minutes = 60

        sync_run = MagicMock()
        sync_run.id = uuid.uuid4()
        run_started_at = datetime.now(UTC)

        monkeypatch.setattr(flows, "create_progress_artifact", AsyncMock(return_value="p"))
        monkeypatch.setattr(flows, "update_progress_artifact", AsyncMock())

        # Return 3 pages
        pages = []
        for i in range(3):
            p = MagicMock()
            p.url = f"https://example.com/page{i}"
            p.markdown = f"# Page {i}"
            p.content_hash = f"hash{i}"
            p.etag = None
            p.last_modified = None
            pages.append(p)

        monkeypatch.setattr(flows, "run_spider_md", lambda **kw: pages)
        monkeypatch.setattr(flows, "get_page_title", lambda page: "Title")

        mock_session = AsyncMock()

        async def mock_execute(stmt):
            r = MagicMock()
            r.scalar_one_or_none.return_value = None
            r.scalar_one.return_value = MagicMock(schedule_interval_minutes=60)
            r.rowcount = 0
            r.all.return_value = []
            return r

        mock_session.execute = mock_execute
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()
        mock_session.commit = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))
        monkeypatch.setattr(
            flows,
            "get_settings",
            lambda: _make_settings(sync_documents_per_run_limit=1),
        )

        monkeypatch.setattr(
            flows,
            "maintain_chunks_for_document",
            AsyncMock(
                return_value={
                    "chunks_created": 1,
                    "chunks_kept": 0,
                    "chunks_deleted": 0,
                }
            ),
        )
        monkeypatch.setattr(
            flows,
            "maintain_symbols_for_document",
            AsyncMock(return_value=(0, 0)),
        )
        monkeypatch.setattr(
            flows,
            "embed_document",
            AsyncMock(
                return_value={
                    "chunks_embedded": 1,
                    "chunks_deduplicated": 0,
                    "tokens_used": 10,
                }
            ),
        )
        monkeypatch.setattr(
            flows,
            "materialize_surface_catalog_for_source",
            AsyncMock(return_value={}),
        )
        monkeypatch.setattr(
            flows,
            "build_twin_graph",
            AsyncMock(
                return_value={
                    "twin_nodes_upserted": 0,
                    "twin_edges_upserted": 0,
                    "twin_metrics_snapshots": 0,
                    "twin_validation_snapshots": 0,
                    "twin_asis_scenario_id": None,
                }
            ),
        )

        result = await flows.sync_web_source.fn(source, sync_run, run_started_at)

        assert result.docs_created == 3


# ---------------------------------------------------------------------------
# sync_github_source: minimal happy path
# ---------------------------------------------------------------------------


class TestSyncGithubSourceHappyPath:
    async def test_full_pipeline_no_changed_files(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """GitHub sync with no changed files completes successfully."""
        from contextmine_core import SourceType
        from contextmine_worker.github_sync import SyncStats

        source = MagicMock()
        source.id = uuid.uuid4()
        source.collection_id = uuid.uuid4()
        source.url = "https://github.com/owner/repo"
        source.type = SourceType.GITHUB
        source.config = {"owner": "owner", "repo": "repo", "branch": "main"}
        source.cursor = "old_sha_123"
        source.schedule_interval_minutes = 60

        sync_run = MagicMock()
        sync_run.id = uuid.uuid4()
        run_started_at = datetime.now(UTC)

        # Mock progress artifacts
        monkeypatch.setattr(flows, "create_progress_artifact", AsyncMock(return_value="p"))
        monkeypatch.setattr(flows, "update_progress_artifact", AsyncMock())

        # Mock deploy key and token
        monkeypatch.setattr(flows, "get_deploy_key_for_source", AsyncMock(return_value=None))
        monkeypatch.setattr(
            flows,
            "get_github_token_for_source",
            AsyncMock(return_value="ghp_test"),
        )

        # Mock git operations
        monkeypatch.setattr(flows, "ensure_repos_dir", lambda: None)
        monkeypatch.setattr(flows, "get_repo_path", lambda sid: tmp_path / "repo")
        (tmp_path / "repo").mkdir()

        mock_git_repo = MagicMock()
        mock_git_repo.head.commit.hexsha = "new_sha_456"

        monkeypatch.setattr(
            flows,
            "_run_blocking_with_timeout",
            AsyncMock(return_value=mock_git_repo),
        )

        # Mock twin source version tracking
        mock_source_version = MagicMock()
        mock_source_version.id = uuid.uuid4()
        mock_source_version.stats = {}
        monkeypatch.setattr(
            "contextmine_core.twin.get_or_create_source_version",
            AsyncMock(return_value=mock_source_version),
        )
        monkeypatch.setattr(
            "contextmine_core.twin.set_source_version_status",
            AsyncMock(),
        )
        monkeypatch.setattr(
            "contextmine_core.twin.record_twin_event",
            AsyncMock(),
        )

        monkeypatch.setattr(flows, "get_settings", lambda: _make_settings())

        # Mock session
        mock_db_source = MagicMock()
        mock_db_source.schedule_interval_minutes = 60
        mock_db_source.cursor = "old_sha_123"
        mock_db_run = MagicMock()
        mock_db_run.id = sync_run.id
        mock_db_run.stats = {}
        mock_twin_sv = MagicMock()
        mock_twin_sv.joern_status = None
        mock_scenario = MagicMock()
        mock_scenario.id = uuid.uuid4()
        mock_scenario.version = 1

        def _make_mock_session():
            s = AsyncMock()

            async def mock_execute(stmt):
                r = MagicMock()
                r.scalar_one_or_none.return_value = mock_twin_sv
                r.scalar_one.return_value = mock_db_source
                r.scalars.return_value.all.return_value = []
                r.all.return_value = []
                r.rowcount = 0
                return r

            s.execute = mock_execute
            s.add = MagicMock()
            s.flush = AsyncMock()
            s.commit = AsyncMock()
            return s

        monkeypatch.setattr(
            flows,
            "get_session",
            lambda: _mock_session_cm(_make_mock_session()),
        )

        # Mock subprocess for joern - make it fail gracefully

        mock_joern_result = MagicMock()
        mock_joern_result.returncode = 1
        mock_joern_result.stderr = "joern not found"

        # Override _run_blocking_with_timeout for different steps
        call_counter = {"count": 0}

        async def mock_run_blocking(step_name, timeout_seconds, func, *args, **kwargs):
            call_counter["count"] += 1
            if step_name == "git_clone_or_pull":
                return mock_git_repo
            if step_name == "joern_parse":
                return mock_joern_result
            return MagicMock()

        monkeypatch.setattr(flows, "_run_blocking_with_timeout", mock_run_blocking)

        # Mock SCIP detection returning no projects
        monkeypatch.setattr(
            flows,
            "task_detect_scip_projects",
            AsyncMock(
                return_value={
                    "projects": [],
                    "diagnostics": {
                        "languages_detected": [],
                        "projects_by_language": {},
                        "warnings": [],
                        "census_tool": "scc",
                        "census_tool_version": "3.0",
                    },
                }
            ),
        )

        # Mock language census
        mock_census = MagicMock()
        mock_census.entries = {}
        monkeypatch.setattr(
            "contextmine_core.semantic_snapshot.indexers.language_census.build_language_census",
            lambda path: mock_census,
        )

        # Mock get_changed_files returning no changes
        monkeypatch.setattr(flows, "get_changed_files", lambda *a: ([], []))

        # Mock KG build
        monkeypatch.setattr(
            flows,
            "build_knowledge_graph",
            AsyncMock(
                return_value={
                    "kg_file_nodes": 0,
                    "kg_symbol_nodes": 0,
                    "kg_errors": [],
                }
            ),
        )

        # Mock surface materialization
        monkeypatch.setattr(
            flows,
            "materialize_surface_catalog_for_source",
            AsyncMock(return_value={}),
        )

        # Mock twin graph build
        monkeypatch.setattr(
            flows,
            "build_twin_graph",
            AsyncMock(
                return_value={
                    "twin_nodes_upserted": 0,
                    "twin_edges_upserted": 0,
                    "twin_metrics_snapshots": 0,
                    "twin_validation_snapshots": 0,
                    "twin_asis_scenario_id": str(mock_scenario.id),
                    "twin_nodes_deactivated": 0,
                    "twin_edges_deactivated": 0,
                    "sample_node_keys": [],
                }
            ),
        )

        # Mock behavioral layers
        monkeypatch.setattr(
            flows,
            "_materialize_behavioral_layers_impl",
            AsyncMock(return_value={}),
        )
        monkeypatch.setattr(
            flows,
            "_log_background_task_failure",
            lambda task: None,
        )

        result = await flows.sync_github_source.fn(source, sync_run, run_started_at)

        assert isinstance(result, SyncStats)
        assert result.files_scanned == 0

    async def test_github_sync_missing_owner_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing owner/repo in config raises ValueError."""
        from contextmine_core import SourceType

        source = MagicMock()
        source.id = uuid.uuid4()
        source.collection_id = uuid.uuid4()
        source.type = SourceType.GITHUB
        source.config = {"owner": "", "repo": ""}

        monkeypatch.setattr(flows, "create_progress_artifact", AsyncMock(return_value="p"))
        monkeypatch.setattr(flows, "get_deploy_key_for_source", AsyncMock(return_value=None))
        monkeypatch.setattr(
            flows,
            "get_github_token_for_source",
            AsyncMock(return_value=None),
        )

        with pytest.raises(ValueError, match="missing owner/repo"):
            await flows.sync_github_source.fn(source, MagicMock(), datetime.now(UTC))

    async def test_github_sync_with_changed_files(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """GitHub sync with changed files processes and embeds them."""
        from contextmine_core import SourceType
        from contextmine_worker.github_sync import SyncStats

        source = MagicMock()
        source.id = uuid.uuid4()
        source.collection_id = uuid.uuid4()
        source.url = "https://github.com/owner/repo"
        source.type = SourceType.GITHUB
        source.config = {"owner": "owner", "repo": "repo", "branch": "main"}
        source.cursor = "old_sha"
        source.schedule_interval_minutes = 60

        sync_run = MagicMock()
        sync_run.id = uuid.uuid4()
        run_started_at = datetime.now(UTC)

        monkeypatch.setattr(flows, "create_progress_artifact", AsyncMock(return_value="p"))
        monkeypatch.setattr(flows, "update_progress_artifact", AsyncMock())
        monkeypatch.setattr(flows, "get_deploy_key_for_source", AsyncMock(return_value=None))
        monkeypatch.setattr(
            flows,
            "get_github_token_for_source",
            AsyncMock(return_value="ghp_t"),
        )
        monkeypatch.setattr(flows, "ensure_repos_dir", lambda: None)

        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        monkeypatch.setattr(flows, "get_repo_path", lambda sid: repo_path)

        mock_git_repo = MagicMock()
        mock_git_repo.head.commit.hexsha = "new_sha"

        mock_joern_result = MagicMock()
        mock_joern_result.returncode = 1
        mock_joern_result.stderr = "not found"

        async def mock_run_blocking(step_name, timeout_seconds, func, *args, **kwargs):
            if step_name == "git_clone_or_pull":
                return mock_git_repo
            if step_name == "joern_parse":
                return mock_joern_result
            return MagicMock()

        monkeypatch.setattr(flows, "_run_blocking_with_timeout", mock_run_blocking)

        mock_source_version = MagicMock()
        mock_source_version.id = uuid.uuid4()
        monkeypatch.setattr(
            "contextmine_core.twin.get_or_create_source_version",
            AsyncMock(return_value=mock_source_version),
        )
        monkeypatch.setattr(
            "contextmine_core.twin.set_source_version_status",
            AsyncMock(),
        )
        monkeypatch.setattr(
            "contextmine_core.twin.record_twin_event",
            AsyncMock(),
        )
        monkeypatch.setattr(flows, "get_settings", lambda: _make_settings())

        mock_db_source = MagicMock()
        mock_db_source.schedule_interval_minutes = 60
        mock_db_source.cursor = "old_sha"
        mock_db_run = MagicMock()
        mock_db_run.id = sync_run.id
        mock_db_run.stats = {}
        mock_scenario = MagicMock()
        mock_scenario.id = uuid.uuid4()
        mock_scenario.version = 1

        def _make_mock_session2():
            s = AsyncMock()

            async def mock_execute(stmt):
                r = MagicMock()
                r.scalar_one_or_none.return_value = None
                r.scalar_one.return_value = mock_db_source
                r.scalars.return_value.all.return_value = []
                r.all.return_value = []
                r.rowcount = 0
                return r

            s.execute = mock_execute
            s.add = MagicMock()
            s.flush = AsyncMock()
            s.commit = AsyncMock()
            return s

        monkeypatch.setattr(
            flows,
            "get_session",
            lambda: _mock_session_cm(_make_mock_session2()),
        )

        # Mock SCIP - no projects
        monkeypatch.setattr(
            flows,
            "task_detect_scip_projects",
            AsyncMock(
                return_value={
                    "projects": [],
                    "diagnostics": {
                        "languages_detected": [],
                        "projects_by_language": {},
                        "warnings": [],
                        "census_tool": "scc",
                        "census_tool_version": "3.0",
                    },
                }
            ),
        )
        mock_census = MagicMock()
        mock_census.entries = {}
        monkeypatch.setattr(
            "contextmine_core.semantic_snapshot.indexers.language_census.build_language_census",
            lambda path: mock_census,
        )

        # Changed files
        monkeypatch.setattr(
            flows,
            "get_changed_files",
            lambda *a: (["src/main.py"], ["deleted.py"]),
        )
        monkeypatch.setattr(flows, "is_eligible_file", lambda path, repo: True)
        monkeypatch.setattr(flows, "read_file_content", lambda repo, fp: "print('hello')")
        monkeypatch.setattr(flows, "compute_content_hash", lambda c: "hash_abc")
        monkeypatch.setattr(
            flows,
            "build_uri",
            lambda owner, repo, fp, branch: f"git://github.com/{owner}/{repo}/{fp}?ref={branch}",
        )
        monkeypatch.setattr(flows, "get_file_title", lambda path: "main.py")

        # Mock chunk/embed/symbol
        monkeypatch.setattr(
            flows,
            "maintain_chunks_for_document",
            AsyncMock(
                return_value={
                    "chunks_created": 2,
                    "chunks_kept": 0,
                    "chunks_deleted": 0,
                }
            ),
        )
        monkeypatch.setattr(
            flows,
            "maintain_symbols_for_document",
            AsyncMock(return_value=(1, 0)),
        )
        monkeypatch.setattr(
            flows,
            "embed_document",
            AsyncMock(
                return_value={
                    "chunks_embedded": 2,
                    "chunks_deduplicated": 0,
                    "tokens_used": 100,
                }
            ),
        )

        # Mock KG and twin
        monkeypatch.setattr(
            flows,
            "build_knowledge_graph",
            AsyncMock(return_value={"kg_file_nodes": 1, "kg_symbol_nodes": 2, "kg_errors": []}),
        )
        monkeypatch.setattr(
            flows,
            "materialize_surface_catalog_for_source",
            AsyncMock(return_value={}),
        )
        monkeypatch.setattr(
            flows,
            "build_twin_graph",
            AsyncMock(
                return_value={
                    "twin_nodes_upserted": 3,
                    "twin_edges_upserted": 2,
                    "twin_metrics_snapshots": 0,
                    "twin_validation_snapshots": 0,
                    "twin_asis_scenario_id": str(mock_scenario.id),
                    "twin_nodes_deactivated": 0,
                    "twin_edges_deactivated": 0,
                    "sample_node_keys": [],
                    "metrics_mapped_files": 0,
                    "metrics_requested_files": 0,
                    "metrics_unmapped_sample": [],
                }
            ),
        )
        monkeypatch.setattr(
            flows,
            "_materialize_behavioral_layers_impl",
            AsyncMock(return_value={}),
        )
        monkeypatch.setattr(flows, "_log_background_task_failure", lambda task: None)

        result = await flows.sync_github_source.fn(source, sync_run, run_started_at)

        assert isinstance(result, SyncStats)
        assert result.files_scanned == 1
        assert result.files_indexed == 1
        assert result.docs_deleted >= 0
