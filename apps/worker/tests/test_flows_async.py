"""Tests for async flow/task bodies in contextmine_worker.flows.

Covers the actual Prefect flow and task functions that do DB operations,
HTTP calls, and pipeline orchestration.  Every external dependency is mocked.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import contextmine_worker.flows as flows
import pytest

pytestmark = pytest.mark.anyio


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


def _mock_session_cm(session: AsyncMock) -> MagicMock:
    """Wrap a mock session in an async context manager."""
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _make_source(
    *,
    source_type: str = "github",
    config: dict | None = None,
    cursor: str | None = None,
    enabled: bool = True,
) -> MagicMock:
    """Create a mock Source object."""
    from contextmine_core import SourceType

    source = MagicMock()
    source.id = uuid.uuid4()
    source.collection_id = uuid.uuid4()
    source.url = "https://github.com/owner/repo"
    source.type = SourceType.GITHUB if source_type == "github" else SourceType.WEB
    source.enabled = enabled
    source.cursor = cursor
    source.schedule_interval_minutes = 60
    source.config = config or {"owner": "owner", "repo": "repo", "branch": "main"}
    return source


def _make_sync_run() -> MagicMock:
    """Create a mock SyncRun object."""
    run = MagicMock()
    run.id = uuid.uuid4()
    run.source_id = uuid.uuid4()
    run.status = MagicMock()
    run.status.value = "running"
    run.started_at = datetime.now(UTC)
    run.finished_at = None
    run.error = None
    run.stats = {}
    return run


# ---------------------------------------------------------------------------
# maintain_chunks_for_document
# ---------------------------------------------------------------------------


class TestMaintainChunksForDocument:
    async def test_happy_path_creates_and_deletes_chunks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When new chunks differ from existing, creates new and deletes old."""
        doc_id = str(uuid.uuid4())

        # Mock chunk_document to return 2 new chunks
        mock_chunk_1 = MagicMock(chunk_hash="hash_a", chunk_index=0, content="aaa", meta={})
        mock_chunk_2 = MagicMock(chunk_hash="hash_b", chunk_index=1, content="bbb", meta={})
        monkeypatch.setattr(
            flows, "chunk_document", lambda content, fp: [mock_chunk_1, mock_chunk_2]
        )

        # Existing chunks in DB: hash_a (kept), hash_c (deleted)
        existing_row_a = SimpleNamespace(chunk_hash="hash_a", id=uuid.uuid4())
        existing_row_c = SimpleNamespace(chunk_hash="hash_c", id=uuid.uuid4())

        mock_session = AsyncMock()
        call_count = 0

        async def mock_execute(stmt):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                # select existing chunks
                result.all.return_value = [existing_row_a, existing_row_c]
            return result

        mock_session.execute = mock_execute
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        stats = await flows.maintain_chunks_for_document.fn(doc_id, "some content", "file.py")

        assert stats["chunks_kept"] == 1  # hash_a
        assert stats["chunks_deleted"] == 1  # hash_c
        assert stats["chunks_created"] == 1  # hash_b

    async def test_no_existing_chunks_creates_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When no existing chunks, all new chunks are created."""
        doc_id = str(uuid.uuid4())

        mock_chunk = MagicMock(chunk_hash="hash_new", chunk_index=0, content="content", meta={})
        monkeypatch.setattr(flows, "chunk_document", lambda content, fp: [mock_chunk])

        mock_session = AsyncMock()
        result_mock = MagicMock()
        result_mock.all.return_value = []
        mock_session.execute = AsyncMock(return_value=result_mock)
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        stats = await flows.maintain_chunks_for_document.fn(doc_id, "content", "file.py")

        assert stats["chunks_created"] == 1
        assert stats["chunks_kept"] == 0
        assert stats["chunks_deleted"] == 0
        mock_session.add.assert_called_once()

    async def test_no_changes_keeps_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When chunk hashes are identical, nothing created or deleted."""
        doc_id = str(uuid.uuid4())

        mock_chunk = MagicMock(chunk_hash="same_hash", chunk_index=0, content="c", meta={})
        monkeypatch.setattr(flows, "chunk_document", lambda content, fp: [mock_chunk])

        existing = SimpleNamespace(chunk_hash="same_hash", id=uuid.uuid4())
        mock_session = AsyncMock()
        result_mock = MagicMock()
        result_mock.all.return_value = [existing]
        mock_session.execute = AsyncMock(return_value=result_mock)
        mock_session.commit = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        stats = await flows.maintain_chunks_for_document.fn(doc_id, "c", "file.py")

        assert stats["chunks_created"] == 0
        assert stats["chunks_kept"] == 1
        assert stats["chunks_deleted"] == 0


# ---------------------------------------------------------------------------
# get_or_create_embedding_model
# ---------------------------------------------------------------------------


class TestGetOrCreateEmbeddingModel:
    async def test_returns_existing_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from contextmine_core import EmbeddingProvider

        existing_model = MagicMock()
        existing_model.id = uuid.uuid4()

        mock_session = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = existing_model
        mock_session.execute = AsyncMock(return_value=result_mock)

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        result = await flows.get_or_create_embedding_model.fn(
            provider=EmbeddingProvider.OPENAI,
            model_name="text-embedding-3-small",
            dimension=1536,
        )

        assert result == existing_model
        mock_session.add.assert_not_called()

    async def test_creates_new_model_when_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from contextmine_core import EmbeddingProvider

        mock_session = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=result_mock)
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        await flows.get_or_create_embedding_model.fn(
            provider=EmbeddingProvider.OPENAI,
            model_name="text-embedding-3-small",
            dimension=1536,
        )

        mock_session.add.assert_called_once()
        mock_session.commit.assert_awaited_once()
        mock_session.refresh.assert_awaited_once()


# ---------------------------------------------------------------------------
# embed_chunks_for_document
# ---------------------------------------------------------------------------


class TestEmbedChunksForDocument:
    async def test_no_chunks_to_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When no chunks need embedding, returns zero stats."""
        doc_id = str(uuid.uuid4())
        model = MagicMock()
        model.id = uuid.uuid4()
        model.provider = "openai"
        model.model_name = "text-embedding-3-small"
        model.dimension = 1536

        mock_session = AsyncMock()
        result_mock = MagicMock()
        result_mock.all.return_value = []
        mock_session.execute = AsyncMock(return_value=result_mock)

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        stats = await flows.embed_chunks_for_document.fn(doc_id, model)

        assert stats["chunks_embedded"] == 0
        assert stats["chunks_deduplicated"] == 0
        assert stats["tokens_used"] == 0

    async def test_deduplication_copies_existing_embeddings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Chunks with hashes matching existing embeddings are deduplicated."""
        doc_id = str(uuid.uuid4())
        model = MagicMock()
        model.id = uuid.uuid4()
        model.provider = "openai"
        model.model_name = "text-embedding-3-small"
        model.dimension = 1536

        chunk_row = SimpleNamespace(id=uuid.uuid4(), chunk_hash="dedup_hash", content="content")

        session_call = 0

        def make_session():
            nonlocal session_call
            session_call += 1
            s = AsyncMock()
            if session_call == 1:
                # First session: find chunks to process
                r = MagicMock()
                r.all.return_value = [chunk_row]
                s.execute = AsyncMock(return_value=r)
            elif session_call == 2:
                # Second session: find existing embeddings
                r = MagicMock()
                r.all.return_value = [("dedup_hash", "[0.1,0.2,0.3]")]
                s.execute = AsyncMock(return_value=r)
            elif session_call == 3:
                # Third session: copy embeddings
                s.execute = AsyncMock(return_value=MagicMock())
                s.commit = AsyncMock()
            return _mock_session_cm(s)

        monkeypatch.setattr(flows, "get_session", make_session)
        monkeypatch.setattr(
            flows,
            "_embedding_batch_timeout_seconds",
            lambda: 120,
        )

        stats = await flows.embed_chunks_for_document.fn(doc_id, model)

        assert stats["chunks_deduplicated"] == 1
        assert stats["chunks_embedded"] == 0

    async def test_embedding_timeout_uses_fake_embedder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On embedding timeout, falls back to FakeEmbedder."""
        doc_id = str(uuid.uuid4())
        model = MagicMock()
        model.id = uuid.uuid4()
        model.provider = "openai"
        model.model_name = "text-embedding-3-small"
        model.dimension = 4

        chunk_row = SimpleNamespace(id=uuid.uuid4(), chunk_hash="new_hash", content="text")

        session_call = 0

        def make_session():
            nonlocal session_call
            session_call += 1
            s = AsyncMock()
            if session_call == 1:
                r = MagicMock()
                r.all.return_value = [chunk_row]
                s.execute = AsyncMock(return_value=r)
            elif session_call == 2:
                r = MagicMock()
                r.all.return_value = []  # no existing embeddings
                s.execute = AsyncMock(return_value=r)
            else:
                s.execute = AsyncMock(return_value=MagicMock())
                s.commit = AsyncMock()
            return _mock_session_cm(s)

        monkeypatch.setattr(flows, "get_session", make_session)
        monkeypatch.setattr(flows, "_embedding_batch_timeout_seconds", lambda: 120)

        # Mock get_embedder to return an embedder whose embed_batch times out
        mock_embedder = MagicMock()

        async def _timeout_embed(texts):
            raise TimeoutError("timeout")

        mock_embedder.embed_batch = _timeout_embed
        monkeypatch.setattr(flows, "get_embedder", lambda **kw: mock_embedder)

        # Mock FakeEmbedder
        fake_result = MagicMock()
        fake_result.embeddings = [[0.1, 0.2, 0.3, 0.4]]
        fake_result.tokens_used = 5

        mock_fake_embedder = MagicMock()
        mock_fake_embedder.embed_batch = AsyncMock(return_value=fake_result)
        monkeypatch.setattr(flows, "FakeEmbedder", lambda dimension: mock_fake_embedder)

        # We need to handle the asyncio.wait_for raising TimeoutError
        original_wait_for = asyncio.wait_for

        async def patched_wait_for(coro, *, timeout=None):
            # For embed_batch calls, raise timeout
            try:
                return await coro
            except TimeoutError:
                raise

        monkeypatch.setattr(asyncio, "wait_for", patched_wait_for)

        stats = await flows.embed_chunks_for_document.fn(doc_id, model)

        # Should have fallen back to fake embedder
        assert stats["chunks_embedded"] == 1
        assert stats["tokens_used"] == 5

        monkeypatch.setattr(asyncio, "wait_for", original_wait_for)


# ---------------------------------------------------------------------------
# get_due_sources
# ---------------------------------------------------------------------------


class TestGetDueSources:
    async def test_returns_enabled_due_sources(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_source = MagicMock()
        mock_source.id = uuid.uuid4()
        mock_source.enabled = True

        mock_session = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = [mock_source]
        mock_session.execute = AsyncMock(return_value=result_mock)

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        sources = await flows.get_due_sources.fn()
        assert len(sources) == 1
        assert sources[0] == mock_source

    async def test_returns_empty_when_none_due(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_session = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=result_mock)

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        sources = await flows.get_due_sources.fn()
        assert sources == []


# ---------------------------------------------------------------------------
# get_github_token_for_source
# ---------------------------------------------------------------------------


class TestGetGithubTokenForSourceDecrypt:
    async def test_decrypts_and_returns_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When owner and token exist, returns decrypted token."""
        call_count = 0
        owner_id = uuid.uuid4()

        async def mock_execute(stmt):
            nonlocal call_count
            call_count += 1
            result_mock = MagicMock()
            if call_count == 1:
                result_mock.scalar_one_or_none.return_value = owner_id
            else:
                token_record = MagicMock()
                token_record.access_token_encrypted = "encrypted_data"
                result_mock.scalars.return_value.all.return_value = [token_record]
            return result_mock

        mock_session = AsyncMock()
        mock_session.execute = mock_execute

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))
        monkeypatch.setattr(flows, "decrypt_token", lambda v: f"decrypted:{v}")

        result = await flows.get_github_token_for_source(str(uuid.uuid4()), str(uuid.uuid4()))
        assert result == "decrypted:encrypted_data"


# ---------------------------------------------------------------------------
# sync_source: stale recovery + dispatching
# ---------------------------------------------------------------------------


class TestSyncSourceDispatching:
    async def test_dispatches_web_source(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Web-type sources dispatch to sync_web_source."""
        from contextmine_core import SourceType, SyncRunStatus

        source = _make_source(source_type="web")
        source.type = SourceType.WEB

        # Mock session sequence: lock, stale check, existing run check, create run,
        # then final refresh
        session_idx = 0

        def make_session():
            nonlocal session_idx
            session_idx += 1
            s = AsyncMock()

            if session_idx == 1:
                # Lock + stale + existing + create
                call_count = 0

                async def exec_fn(stmt):
                    nonlocal call_count
                    call_count += 1
                    r = MagicMock()
                    if call_count == 1:
                        # Lock acquired
                        r.scalar_one_or_none.return_value = source
                    elif call_count == 2:
                        # Running runs (stale check)
                        r.scalars.return_value.all.return_value = []
                    elif call_count == 3:
                        # Existing run check
                        r.scalar_one_or_none.return_value = None
                    return r

                s.execute = exec_fn
                s.add = MagicMock()
                s.commit = AsyncMock()
                s.refresh = AsyncMock()
            elif session_idx == 2:
                # Final refresh
                mock_run = MagicMock()
                mock_run.id = uuid.uuid4()
                mock_run.status = SyncRunStatus.SUCCESS
                r = MagicMock()
                r.scalar_one.return_value = mock_run
                s.execute = AsyncMock(return_value=r)

            return _mock_session_cm(s)

        monkeypatch.setattr(flows, "get_session", make_session)

        mock_web_sync = AsyncMock()
        monkeypatch.setattr(flows, "sync_web_source", mock_web_sync)

        result = await flows.sync_source.fn(source)

        mock_web_sync.assert_awaited_once()
        assert result is not None

    async def test_dispatches_github_source(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GitHub-type sources dispatch to sync_github_source."""
        from contextmine_core import SyncRunStatus

        source = _make_source(source_type="github")

        session_idx = 0

        def make_session():
            nonlocal session_idx
            session_idx += 1
            s = AsyncMock()

            if session_idx == 1:
                call_count = 0

                async def exec_fn(stmt):
                    nonlocal call_count
                    call_count += 1
                    r = MagicMock()
                    if call_count == 1:
                        r.scalar_one_or_none.return_value = source
                    elif call_count == 2:
                        r.scalars.return_value.all.return_value = []
                    elif call_count == 3:
                        r.scalar_one_or_none.return_value = None
                    return r

                s.execute = exec_fn
                s.add = MagicMock()
                s.commit = AsyncMock()
                s.refresh = AsyncMock()
            elif session_idx == 2:
                mock_run = MagicMock()
                mock_run.id = uuid.uuid4()
                mock_run.status = SyncRunStatus.SUCCESS
                r = MagicMock()
                r.scalar_one.return_value = mock_run
                s.execute = AsyncMock(return_value=r)

            return _mock_session_cm(s)

        monkeypatch.setattr(flows, "get_session", make_session)

        mock_gh_sync = AsyncMock()
        monkeypatch.setattr(flows, "sync_github_source", mock_gh_sync)

        result = await flows.sync_source.fn(source)

        mock_gh_sync.assert_awaited_once()
        assert result is not None

    async def test_exception_marks_run_failed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When sync raises, the sync run is marked FAILED and source timestamps updated."""
        from contextmine_core import SyncRunStatus

        source = _make_source()

        session_idx = 0
        mock_db_run = MagicMock()
        mock_db_source = MagicMock()
        mock_db_source.schedule_interval_minutes = 60

        def make_session():
            nonlocal session_idx
            session_idx += 1
            s = AsyncMock()

            if session_idx == 1:
                # Lock + stale + check + create
                call_count = 0

                async def exec_fn(stmt):
                    nonlocal call_count
                    call_count += 1
                    r = MagicMock()
                    if call_count == 1:
                        r.scalar_one_or_none.return_value = source
                    elif call_count == 2:
                        r.scalars.return_value.all.return_value = []
                    elif call_count == 3:
                        r.scalar_one_or_none.return_value = None
                    return r

                s.execute = exec_fn
                s.add = MagicMock()
                s.commit = AsyncMock()
                s.refresh = AsyncMock()
            elif session_idx == 2:
                # Error handling session
                call_count2 = 0

                async def exec_fn2(stmt):
                    nonlocal call_count2
                    call_count2 += 1
                    r = MagicMock()
                    if call_count2 == 1:
                        r.scalar_one.return_value = mock_db_run
                    elif call_count2 == 2:
                        r.scalar_one_or_none.return_value = None
                    elif call_count2 == 3:
                        r.scalar_one.return_value = mock_db_source
                    else:
                        r.scalar_one_or_none.return_value = None
                        r.scalars.return_value.all.return_value = []
                    return r

                s.execute = exec_fn2
                s.commit = AsyncMock()
                s.refresh = AsyncMock()

            return _mock_session_cm(s)

        monkeypatch.setattr(flows, "get_session", make_session)
        monkeypatch.setattr(
            flows,
            "sync_github_source",
            AsyncMock(side_effect=RuntimeError("sync_broke")),
        )

        await flows.sync_source.fn(source)

        assert mock_db_run.status == SyncRunStatus.FAILED
        assert mock_db_run.error == "sync_broke"


# ---------------------------------------------------------------------------
# sync_single_source
# ---------------------------------------------------------------------------


class TestSyncSingleSourceHappyPath:
    async def test_happy_path_returns_stats(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Successful sync returns source_id, run_id, status, stats."""
        source = _make_source()

        mock_session = AsyncMock()
        r = MagicMock()
        r.scalar_one_or_none.return_value = source
        mock_session.execute = AsyncMock(return_value=r)

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        mock_run = MagicMock()
        mock_run.id = uuid.uuid4()
        mock_run.status = MagicMock()
        mock_run.status.value = "success"
        mock_run.stats = {"files_scanned": 42}

        monkeypatch.setattr(flows, "sync_source", AsyncMock(return_value=mock_run))
        monkeypatch.setattr(flows, "_sync_source_timeout_seconds", lambda: 0)

        result = await flows.sync_single_source.fn(str(source.id))

        assert result["status"] == "success"
        assert result["stats"]["files_scanned"] == 42

    async def test_timeout_recovers_and_reports(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Timeout in sync_single_source recovers running rows."""
        source = _make_source()

        mock_session = AsyncMock()
        r = MagicMock()
        r.scalar_one_or_none.return_value = source
        mock_session.execute = AsyncMock(return_value=r)

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))
        monkeypatch.setattr(flows, "_sync_source_timeout_seconds", lambda: 1)

        original_wait_for = asyncio.wait_for

        async def mock_wait_for(coro, *, timeout=None):
            raise TimeoutError("timed out")

        monkeypatch.setattr(asyncio, "wait_for", mock_wait_for)
        monkeypatch.setattr(flows, "_fail_running_sync_runs_for_source", AsyncMock(return_value=2))

        result = await flows.sync_single_source.fn(str(source.id))

        assert "error" in result
        assert "AUTO_TIMEOUT" in result["error"]
        assert result["recovered_running_rows"] == 2

        monkeypatch.setattr(asyncio, "wait_for", original_wait_for)

    async def test_generic_exception_returns_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Generic exception in sync_single_source returns error dict."""
        source = _make_source()

        mock_session = AsyncMock()
        r = MagicMock()
        r.scalar_one_or_none.return_value = source
        mock_session.execute = AsyncMock(return_value=r)

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))
        monkeypatch.setattr(flows, "_sync_source_timeout_seconds", lambda: 0)
        monkeypatch.setattr(
            flows,
            "sync_source",
            AsyncMock(side_effect=RuntimeError("unexpected")),
        )

        result = await flows.sync_single_source.fn(str(source.id))

        assert result["error"] == "unexpected"


# ---------------------------------------------------------------------------
# build_knowledge_graph: happy path with all steps
# ---------------------------------------------------------------------------


class TestBuildKnowledgeGraphHappyPath:
    async def test_all_steps_succeed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Full KG build with all steps mocked to succeed."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())

        monkeypatch.setattr(
            flows, "get_settings", lambda: _make_settings(default_llm_provider="openai")
        )

        mock_llm = MagicMock()
        monkeypatch.setattr(
            "contextmine_core.research.llm.get_llm_provider",
            lambda *a, **kw: mock_llm,
        )
        mock_research_llm = MagicMock()
        monkeypatch.setattr(
            "contextmine_core.research.llm.get_research_llm_provider",
            lambda: mock_research_llm,
        )

        mock_embedder = MagicMock()
        mock_embedder.dimension = 1536
        monkeypatch.setattr(
            flows, "parse_embedding_model_spec", lambda spec: ("openai", "text-embedding-3-small")
        )
        monkeypatch.setattr(flows, "get_embedder", lambda *a, **kw: mock_embedder)

        # Step 1: FILE/SYMBOL nodes
        mock_kg_stats = MagicMock()
        mock_kg_stats.file_nodes_created = 10
        mock_kg_stats.symbol_nodes_created = 50

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=MagicMock())
        mock_session.commit = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        monkeypatch.setattr(
            "contextmine_core.knowledge.builder.build_knowledge_graph_for_source",
            AsyncMock(return_value=mock_kg_stats),
        )
        monkeypatch.setattr(
            "contextmine_core.knowledge.builder.cleanup_orphan_nodes",
            AsyncMock(return_value={"nodes_deleted": 0}),
        )
        monkeypatch.setattr(
            "contextmine_core.knowledge.builder.cleanup_scoped_knowledge_nodes",
            AsyncMock(return_value={"nodes_deleted": 0, "evidence_deleted": 0}),
        )

        # Step 2: Business rules - mock docs query returning empty
        docs_result = MagicMock()
        docs_result.all.return_value = []
        mock_session.execute = AsyncMock(return_value=docs_result)

        # Step 5: Semantic entities
        mock_extraction_batch = MagicMock()
        monkeypatch.setattr(
            "contextmine_core.knowledge.extraction.extract_from_documents",
            AsyncMock(return_value=mock_extraction_batch),
        )
        monkeypatch.setattr(
            "contextmine_core.knowledge.extraction.persist_semantic_entities",
            AsyncMock(return_value={"entities_created": 5, "relationships_created": 3}),
        )

        # Step 6: Communities
        mock_community_result = MagicMock()
        mock_community_result.community_count = lambda level: {0: 4, 1: 2, 2: 1}.get(level, 0)
        mock_community_result.modularity = {0: 0.5, 1: 0.3, 2: 0.1}
        monkeypatch.setattr(
            "contextmine_core.knowledge.communities.detect_communities",
            AsyncMock(return_value=mock_community_result),
        )
        monkeypatch.setattr(
            "contextmine_core.knowledge.communities.persist_communities",
            AsyncMock(),
        )

        # Step 7: Summaries
        mock_summary_stats = MagicMock()
        mock_summary_stats.communities_summarized = 7
        mock_summary_stats.embeddings_created = 7
        monkeypatch.setattr(
            "contextmine_core.knowledge.summaries.generate_community_summaries",
            AsyncMock(return_value=mock_summary_stats),
        )

        result = await flows.build_knowledge_graph.fn(
            source_id=source_id,
            collection_id=collection_id,
            changed_doc_ids=None,
        )

        assert result["kg_file_nodes"] == 10
        assert result["kg_symbol_nodes"] == 50

    async def test_step_failures_are_caught_and_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Individual step failures are caught, not fatal."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())

        monkeypatch.setattr(
            flows, "get_settings", lambda: _make_settings(default_llm_provider="openai")
        )
        monkeypatch.setattr(
            "contextmine_core.research.llm.get_llm_provider",
            lambda *a, **kw: MagicMock(),
        )
        monkeypatch.setattr(
            "contextmine_core.research.llm.get_research_llm_provider",
            lambda: MagicMock(),
        )
        monkeypatch.setattr(
            flows, "parse_embedding_model_spec", lambda spec: ("openai", "text-embedding-3-small")
        )
        monkeypatch.setattr(flows, "get_embedder", lambda *a, **kw: MagicMock(dimension=1536))

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=MagicMock())
        mock_session.commit = AsyncMock()
        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        # All steps raise
        monkeypatch.setattr(
            "contextmine_core.knowledge.builder.build_knowledge_graph_for_source",
            AsyncMock(side_effect=RuntimeError("step1_fail")),
        )
        monkeypatch.setattr(
            "contextmine_core.knowledge.builder.cleanup_orphan_nodes",
            AsyncMock(return_value={"nodes_deleted": 0}),
        )
        monkeypatch.setattr(
            "contextmine_core.knowledge.builder.cleanup_scoped_knowledge_nodes",
            AsyncMock(return_value={"nodes_deleted": 0, "evidence_deleted": 0}),
        )
        monkeypatch.setattr(
            "contextmine_core.knowledge.extraction.extract_from_documents",
            AsyncMock(side_effect=RuntimeError("step5_fail")),
        )
        monkeypatch.setattr(
            "contextmine_core.knowledge.communities.detect_communities",
            AsyncMock(side_effect=RuntimeError("step6_fail")),
        )
        monkeypatch.setattr(
            "contextmine_core.knowledge.summaries.generate_community_summaries",
            AsyncMock(side_effect=RuntimeError("step7_fail")),
        )

        result = await flows.build_knowledge_graph.fn(
            source_id=source_id,
            collection_id=collection_id,
            changed_doc_ids=[],
        )

        # Errors are collected, not raised
        assert len(result["kg_errors"]) >= 1
        assert result["kg_file_nodes"] == 0
        assert result["kg_symbol_nodes"] == 0

    async def test_skips_extraction_when_no_changed_docs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When changed_doc_ids is empty list, business rules and semantic extraction skipped."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())

        monkeypatch.setattr(
            flows, "get_settings", lambda: _make_settings(default_llm_provider="openai")
        )
        monkeypatch.setattr(
            "contextmine_core.research.llm.get_llm_provider",
            lambda *a, **kw: MagicMock(),
        )
        monkeypatch.setattr(
            "contextmine_core.research.llm.get_research_llm_provider",
            lambda: MagicMock(),
        )
        monkeypatch.setattr(flows, "parse_embedding_model_spec", lambda spec: ("openai", "model"))
        monkeypatch.setattr(flows, "get_embedder", lambda *a, **kw: MagicMock(dimension=768))

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=MagicMock())
        mock_session.commit = AsyncMock()
        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        mock_kg_stats = MagicMock()
        mock_kg_stats.file_nodes_created = 0
        mock_kg_stats.symbol_nodes_created = 0
        monkeypatch.setattr(
            "contextmine_core.knowledge.builder.build_knowledge_graph_for_source",
            AsyncMock(return_value=mock_kg_stats),
        )
        monkeypatch.setattr(
            "contextmine_core.knowledge.builder.cleanup_orphan_nodes",
            AsyncMock(return_value={"nodes_deleted": 0}),
        )
        monkeypatch.setattr(
            "contextmine_core.knowledge.builder.cleanup_scoped_knowledge_nodes",
            AsyncMock(return_value={"nodes_deleted": 0, "evidence_deleted": 0}),
        )

        # These should NOT be called because changed_doc_ids=[]
        extract_mock = AsyncMock()
        monkeypatch.setattr(
            "contextmine_core.knowledge.extraction.extract_from_documents",
            extract_mock,
        )

        mock_community = MagicMock()
        mock_community.community_count = lambda level: 0
        mock_community.modularity = {0: 0, 1: 0, 2: 0}
        monkeypatch.setattr(
            "contextmine_core.knowledge.communities.detect_communities",
            AsyncMock(return_value=mock_community),
        )
        monkeypatch.setattr(
            "contextmine_core.knowledge.communities.persist_communities",
            AsyncMock(),
        )

        summary_mock = AsyncMock()
        monkeypatch.setattr(
            "contextmine_core.knowledge.summaries.generate_community_summaries",
            summary_mock,
        )

        await flows.build_knowledge_graph.fn(
            source_id=source_id,
            collection_id=collection_id,
            changed_doc_ids=[],  # Empty list
        )

        extract_mock.assert_not_awaited()
        summary_mock.assert_not_awaited()


# ---------------------------------------------------------------------------
# build_twin_graph
# ---------------------------------------------------------------------------


class TestBuildTwinGraphWithSnapshots:
    async def test_with_snapshot_dicts_ingests_and_supplements(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When snapshot_dicts provided, ingests them AND supplements from KG."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())

        mock_as_is = MagicMock()
        mock_as_is.id = uuid.uuid4()

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=MagicMock())
        mock_session.commit = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))
        monkeypatch.setattr(flows, "get_settings", lambda: _make_settings(arch_docs_enabled=False))

        monkeypatch.setattr(
            "contextmine_core.twin.get_or_create_as_is_scenario",
            AsyncMock(return_value=mock_as_is),
        )
        monkeypatch.setattr(
            "contextmine_core.twin.ingest_snapshot_into_as_is",
            AsyncMock(return_value=(mock_as_is, {"nodes_upserted": 5, "edges_upserted": 3})),
        )
        monkeypatch.setattr(
            "contextmine_core.twin.seed_scenario_from_knowledge_graph",
            AsyncMock(return_value=(2, 4)),
        )
        monkeypatch.setattr(
            "contextmine_core.twin.refresh_metric_snapshots",
            AsyncMock(return_value=1),
        )
        monkeypatch.setattr(
            "contextmine_core.validation.refresh_validation_snapshots",
            AsyncMock(return_value=0),
        )
        monkeypatch.setattr(
            "contextmine_core.graph.age.sync_scenario_to_age",
            AsyncMock(),
        )

        # Provide a snapshot dict

        monkeypatch.setattr(
            "contextmine_core.semantic_snapshot.models.Snapshot.from_dict",
            MagicMock(),
        )

        result = await flows.build_twin_graph.fn(
            source_id=source_id,
            collection_id=collection_id,
            snapshot_dicts=[{"symbols": [], "relations": [], "files": [], "meta": {}}],
            changed_doc_ids=None,
            file_metrics=None,
            evolution_payload=None,
        )

        assert result["twin_nodes_upserted"] == 7  # 5 from snapshot + 2 from KG
        assert result["twin_edges_upserted"] == 7  # 3 from snapshot + 4 from KG

    async def test_with_evolution_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Evolution payload triggers ownership/coupling persistence."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())

        mock_as_is = MagicMock()
        mock_as_is.id = uuid.uuid4()

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=MagicMock())
        mock_session.commit = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))
        monkeypatch.setattr(flows, "get_settings", lambda: _make_settings(arch_docs_enabled=False))

        monkeypatch.setattr(
            "contextmine_core.twin.get_or_create_as_is_scenario",
            AsyncMock(return_value=mock_as_is),
        )
        monkeypatch.setattr(
            "contextmine_core.twin.seed_scenario_from_knowledge_graph",
            AsyncMock(return_value=(0, 0)),
        )
        monkeypatch.setattr(
            "contextmine_core.twin.refresh_metric_snapshots",
            AsyncMock(return_value=0),
        )
        monkeypatch.setattr(
            "contextmine_core.twin.replace_evolution_snapshots",
            AsyncMock(return_value={"ownership_rows": 10, "coupling_rows": 5}),
        )
        monkeypatch.setattr(
            "contextmine_core.twin.evaluate_and_store_fitness_findings",
            AsyncMock(return_value={"created": 3, "by_type": {}, "warnings": []}),
        )
        monkeypatch.setattr(
            "contextmine_core.validation.refresh_validation_snapshots",
            AsyncMock(return_value=0),
        )
        monkeypatch.setattr(
            "contextmine_core.graph.age.sync_scenario_to_age",
            AsyncMock(),
        )

        result = await flows.build_twin_graph.fn(
            source_id=source_id,
            collection_id=collection_id,
            snapshot_dicts=None,
            changed_doc_ids=None,
            file_metrics=None,
            evolution_payload={
                "ownership_rows": [{"file": "a.py", "author": "dev1"}],
                "coupling_rows": [{"file_a": "a.py", "file_b": "b.py"}],
                "window_days": 90,
            },
        )

        assert result["evolution_ownership_rows"] == 10
        assert result["evolution_coupling_rows"] == 5
        assert result["fitness_findings_written"] == 3


# ---------------------------------------------------------------------------
# materialize_surface_catalog_for_source: documents with surfaces
# ---------------------------------------------------------------------------


class TestMaterializeSurfaceCatalogWithSurfaces:
    async def test_processes_recognized_spec_files(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When documents contain spec-like content, surfaces are extracted."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())

        mock_session = AsyncMock()
        docs_result = MagicMock()
        docs_result.all.return_value = [
            ("git://github.com/owner/repo/openapi.yaml", "openapi: '3.0.0'\ninfo:\n  title: Test"),
            ("git://github.com/owner/repo/src/main.py", "print('hello')"),
        ]
        mock_session.execute = AsyncMock(return_value=docs_result)
        mock_session.commit = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        result = await flows.materialize_surface_catalog_for_source(
            source_id=source_id,
            collection_id=collection_id,
        )

        assert result["surface_files_scanned"] == 2


# ---------------------------------------------------------------------------
# _materialize_behavioral_layers_impl: full path
# ---------------------------------------------------------------------------


class TestMaterializeBehavioralLayersImplFull:
    async def test_happy_path_extracts_tests_and_ui(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Full behavioral materialization with tests and UI extraction."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())

        monkeypatch.setattr(
            flows,
            "get_settings",
            lambda: _make_settings(
                digital_twin_behavioral_enabled=True,
                digital_twin_ui_enabled=True,
                digital_twin_flows_enabled=True,
            ),
        )

        mock_session = AsyncMock()
        docs_result = MagicMock()
        docs_result.all.return_value = [
            ("git://github.com/owner/repo/src/test_main.py", "def test_foo(): pass"),
        ]
        mock_session.execute = AsyncMock(return_value=docs_result)
        mock_session.commit = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))
        monkeypatch.setattr(
            "contextmine_core.knowledge.builder.cleanup_scoped_knowledge_nodes",
            AsyncMock(return_value={"nodes_deleted": 0, "evidence_deleted": 0}),
        )

        monkeypatch.setattr(
            "contextmine_core.analyzer.extractors.tests.extract_tests_from_files",
            lambda files: [],
        )
        monkeypatch.setattr(
            "contextmine_core.analyzer.extractors.ui.extract_ui_from_files",
            lambda files: [],
        )

        result = await flows._materialize_behavioral_layers_impl(
            source_id=source_id,
            collection_id=collection_id,
            scenario_id=None,
            source_version_id=None,
        )

        assert result["behavioral_layers_status"] == "ready"
        warnings = result["deep_warnings"]
        assert isinstance(warnings, list)
        assert "No test semantics extracted" in warnings[0]

    async def test_with_scenario_seeds_and_syncs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When scenario_id is given, seeds from KG and syncs to AGE."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())
        scenario_id = str(uuid.uuid4())

        monkeypatch.setattr(
            flows,
            "get_settings",
            lambda: _make_settings(
                digital_twin_behavioral_enabled=True,
                digital_twin_ui_enabled=False,
                digital_twin_flows_enabled=False,
            ),
        )

        mock_session = AsyncMock()
        docs_result = MagicMock()
        docs_result.all.return_value = []
        mock_session.execute = AsyncMock(return_value=docs_result)
        mock_session.commit = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        monkeypatch.setattr(
            "contextmine_core.analyzer.extractors.tests.extract_tests_from_files",
            lambda files: [],
        )

        seed_mock = AsyncMock(return_value=(0, 0))
        monkeypatch.setattr(
            "contextmine_core.twin.seed_scenario_from_knowledge_graph",
            seed_mock,
        )

        age_mock = AsyncMock()
        monkeypatch.setattr(
            "contextmine_core.graph.age.sync_scenario_to_age",
            age_mock,
        )

        result = await flows._materialize_behavioral_layers_impl(
            source_id=source_id,
            collection_id=collection_id,
            scenario_id=scenario_id,
            source_version_id=None,
        )

        seed_mock.assert_awaited_once()
        age_mock.assert_awaited_once()
        assert result["behavioral_layers_status"] == "ready"

    async def test_with_source_version_records_events(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When source_version_id is given, records twin events."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())
        source_version_id = str(uuid.uuid4())

        monkeypatch.setattr(
            flows,
            "get_settings",
            lambda: _make_settings(
                digital_twin_behavioral_enabled=True,
                digital_twin_ui_enabled=False,
                digital_twin_flows_enabled=False,
            ),
        )

        mock_source_version = MagicMock()
        mock_source_version.stats = {}

        call_count = 0

        async def mock_execute(stmt):
            nonlocal call_count
            call_count += 1
            r = MagicMock()
            if call_count == 2:
                # docs query
                r.all.return_value = []
            elif call_count == 3:
                # source version query
                r.scalar_one_or_none.return_value = mock_source_version
            else:
                r.all.return_value = []
                r.scalar_one_or_none.return_value = None
            return r

        mock_session = AsyncMock()
        mock_session.execute = mock_execute
        mock_session.commit = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        monkeypatch.setattr(
            "contextmine_core.analyzer.extractors.tests.extract_tests_from_files",
            lambda files: [],
        )

        record_mock = AsyncMock()
        monkeypatch.setattr(
            "contextmine_core.twin.record_twin_event",
            record_mock,
        )

        result = await flows._materialize_behavioral_layers_impl(
            source_id=source_id,
            collection_id=collection_id,
            scenario_id=None,
            source_version_id=source_version_id,
        )

        # Should have called record_twin_event at least twice (started + ready)
        assert record_mock.await_count >= 2
        assert result["behavioral_layers_status"] == "ready"


# ---------------------------------------------------------------------------
# materialize_behavioral_layers (task wrapper)
# ---------------------------------------------------------------------------


class TestMaterializeBehavioralLayersTask:
    async def test_success_delegates_to_impl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        expected = {"behavioral_layers_status": "ready"}
        monkeypatch.setattr(
            flows,
            "_materialize_behavioral_layers_impl",
            AsyncMock(return_value=expected),
        )

        result = await flows.materialize_behavioral_layers.fn(
            source_id=str(uuid.uuid4()),
            collection_id=str(uuid.uuid4()),
        )

        assert result == expected

    async def test_failure_records_event_and_reraises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On failure, records failed event and re-raises."""
        monkeypatch.setattr(
            flows,
            "_materialize_behavioral_layers_impl",
            AsyncMock(side_effect=RuntimeError("impl_failed")),
        )

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
        )
        mock_session.commit = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        record_mock = AsyncMock()
        monkeypatch.setattr(
            "contextmine_core.twin.record_twin_event",
            record_mock,
        )

        with pytest.raises(RuntimeError, match="impl_failed"):
            await flows.materialize_behavioral_layers.fn(
                source_id=str(uuid.uuid4()),
                collection_id=str(uuid.uuid4()),
                source_version_id=str(uuid.uuid4()),
            )

        # Should have recorded the failure event
        record_mock.assert_awaited()


# ---------------------------------------------------------------------------
# ingest_coverage_metrics: deeper paths
# ---------------------------------------------------------------------------


class TestIngestCoverageMetricsSourceNotFound:
    async def test_source_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When source is not found, returns rejected."""
        job_id = str(uuid.uuid4())
        mock_job = MagicMock()
        mock_job.id = uuid.uuid4()
        mock_job.status = "queued"
        mock_job.source_id = uuid.uuid4()
        mock_job.error_code = None
        mock_job.error_detail = None

        session_idx = 0

        def make_session():
            nonlocal session_idx
            session_idx += 1
            s = AsyncMock()

            if session_idx == 1:
                # Lock + status check
                r = MagicMock()
                r.scalar_one_or_none.return_value = mock_job
                s.execute = AsyncMock(return_value=r)
                s.commit = AsyncMock()
            elif session_idx == 2:
                # Re-read job + source lookup
                call_count = 0

                async def exec_fn(stmt):
                    nonlocal call_count
                    call_count += 1
                    r = MagicMock()
                    if call_count == 1:
                        r.scalar_one_or_none.return_value = mock_job
                    elif call_count == 2:
                        r.scalar_one_or_none.return_value = None  # Source not found
                    return r

                s.execute = exec_fn
                s.commit = AsyncMock()
            else:
                # _fail_coverage_ingest_job session
                fail_job = MagicMock()
                r = MagicMock()
                r.scalar_one_or_none.return_value = fail_job
                s.execute = AsyncMock(return_value=r)
                s.commit = AsyncMock()

            return _mock_session_cm(s)

        monkeypatch.setattr(flows, "get_session", make_session)

        result = await flows.ingest_coverage_metrics.fn(job_id)

        assert result["status"] in ("rejected", "failed")

    async def test_non_github_source_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Coverage ingest is rejected for non-GitHub sources."""
        from contextmine_core import SourceType

        job_id = str(uuid.uuid4())
        mock_job = MagicMock()
        mock_job.id = uuid.uuid4()
        mock_job.status = "queued"
        mock_job.source_id = uuid.uuid4()
        mock_job.error_code = None
        mock_job.error_detail = None

        mock_source = MagicMock()
        mock_source.type = SourceType.WEB

        session_idx = 0

        def make_session():
            nonlocal session_idx
            session_idx += 1
            s = AsyncMock()

            if session_idx == 1:
                r = MagicMock()
                r.scalar_one_or_none.return_value = mock_job
                s.execute = AsyncMock(return_value=r)
                s.commit = AsyncMock()
            elif session_idx == 2:
                call_count = 0

                async def exec_fn(stmt):
                    nonlocal call_count
                    call_count += 1
                    r = MagicMock()
                    if call_count == 1:
                        r.scalar_one_or_none.return_value = mock_job
                    elif call_count == 2:
                        r.scalar_one_or_none.return_value = mock_source
                    return r

                s.execute = exec_fn
                s.commit = AsyncMock()
            else:
                fail_job = MagicMock()
                r = MagicMock()
                r.scalar_one_or_none.return_value = fail_job
                s.execute = AsyncMock(return_value=r)
                s.commit = AsyncMock()

            return _mock_session_cm(s)

        monkeypatch.setattr(flows, "get_session", make_session)

        result = await flows.ingest_coverage_metrics.fn(job_id)

        assert result["status"] in ("rejected", "failed")


# ---------------------------------------------------------------------------
# task_repair_twin_file_path_canonicalization
# ---------------------------------------------------------------------------


class TestTaskRepairTwinFilePathCanonicalization:
    async def test_calls_repair_and_refreshes_snapshots(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        collection_id = str(uuid.uuid4())
        scenario_id = str(uuid.uuid4())

        repair_result = {
            "legacy_candidates": 10,
            "updated_in_place": 5,
            "scenarios_changed": [str(uuid.uuid4())],
            "collections_changed": [collection_id],
        }

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=MagicMock())
        mock_session.commit = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        monkeypatch.setattr(
            "contextmine_core.twin.repair_twin_file_path_canonicalization",
            AsyncMock(return_value=repair_result),
        )
        monkeypatch.setattr(
            "contextmine_core.twin.refresh_metric_snapshots",
            AsyncMock(return_value=3),
        )
        monkeypatch.setattr(
            "contextmine_core.twin.record_twin_event",
            AsyncMock(),
        )

        result = await flows.task_repair_twin_file_path_canonicalization.fn(
            collection_id=collection_id,
            scenario_id=scenario_id,
        )

        assert result["legacy_candidates"] == 10
        assert result["metric_snapshots_refreshed"] == 3


# ---------------------------------------------------------------------------
# sync_due_sources: multiple sources
# ---------------------------------------------------------------------------


class TestSyncDueSourcesMultiple:
    async def test_multiple_sources_mixed_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Sync multiple sources with success, skip, and error."""
        source1 = MagicMock()
        source1.id = uuid.uuid4()
        source2 = MagicMock()
        source2.id = uuid.uuid4()
        source3 = MagicMock()
        source3.id = uuid.uuid4()

        run1 = MagicMock()
        run1.id = uuid.uuid4()
        run1.status = MagicMock()
        run1.status.value = "success"

        call_count = 0

        async def mock_sync_source(source):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return run1
            elif call_count == 2:
                return None  # Skipped
            else:
                raise RuntimeError("source3_error")

        monkeypatch.setattr(
            flows, "get_due_sources", AsyncMock(return_value=[source1, source2, source3])
        )
        monkeypatch.setattr(flows, "sync_source", mock_sync_source)
        monkeypatch.setattr(flows, "_sync_source_timeout_seconds", lambda: 0)

        result = await flows.sync_due_sources.fn()

        assert result["synced"] == 2  # success + error
        assert result["skipped"] == 1


# ---------------------------------------------------------------------------
# _fail_running_sync_runs_for_source
# ---------------------------------------------------------------------------


class TestFailRunningSyncRunsMultiple:
    async def test_multiple_rows_all_marked_failed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from contextmine_core import SyncRunStatus

        rows = [MagicMock() for _ in range(3)]
        mock_session = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = rows
        mock_session.execute = AsyncMock(return_value=result_mock)
        mock_session.commit = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        count = await flows._fail_running_sync_runs_for_source(str(uuid.uuid4()), "timeout")
        assert count == 3
        for row in rows:
            assert row.status == SyncRunStatus.FAILED
            assert row.error == "timeout"


# ---------------------------------------------------------------------------
# _fail_coverage_ingest_job
# ---------------------------------------------------------------------------


class TestFailCoverageIngestJobWithStats:
    async def test_with_custom_stats(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_job = MagicMock()
        mock_session = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = mock_job
        mock_session.execute = AsyncMock(return_value=result_mock)
        mock_session.commit = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        result = await flows._fail_coverage_ingest_job(
            str(uuid.uuid4()),
            error_code="CUSTOM_ERR",
            error_detail="detail",
            stats={"reports_total": 5},
        )

        assert result["status"] == "failed"
        assert mock_job.stats == {"reports_total": 5}


# ---------------------------------------------------------------------------
# get_embedding_model_for_collection: edge cases
# ---------------------------------------------------------------------------


class TestGetEmbeddingModelForCollectionEdge:
    async def test_config_with_empty_embedding_model_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Config exists but embedding_model is empty string."""
        mock_session = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = {"embedding_model": ""}
        mock_session.execute = AsyncMock(return_value=result_mock)

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))
        monkeypatch.setattr(
            flows, "get_settings", lambda: _make_settings(default_embedding_model="openai:model")
        )

        result = await flows.get_embedding_model_for_collection(str(uuid.uuid4()))
        assert result == "openai:model"


# ---------------------------------------------------------------------------
# get_deploy_key_for_source
# ---------------------------------------------------------------------------


class TestGetDeployKeyForSourceEdge:
    async def test_empty_string_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty string encrypted key should be treated as no key by decrypt_token."""
        mock_session = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = ""
        mock_session.execute = AsyncMock(return_value=result_mock)

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        # Empty string is falsy, so function returns None without decrypting
        result = await flows.get_deploy_key_for_source(str(uuid.uuid4()))
        assert result is None


# ---------------------------------------------------------------------------
# embed_document
# ---------------------------------------------------------------------------


class TestEmbedDocumentIntegration:
    async def test_parses_model_spec_correctly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """embed_document parses the model spec and creates embedding model."""
        doc_id = str(uuid.uuid4())
        coll_id = str(uuid.uuid4())

        monkeypatch.setattr(
            flows,
            "get_embedding_model_for_collection",
            AsyncMock(return_value="gemini:text-embedding-004"),
        )

        parse_calls = []
        original_parse = flows.parse_embedding_model_spec

        def tracking_parse(spec):
            parse_calls.append(spec)
            return original_parse(spec)

        monkeypatch.setattr(flows, "parse_embedding_model_spec", tracking_parse)

        mock_embedder = MagicMock()
        mock_embedder.dimension = 768
        monkeypatch.setattr(flows, "get_embedder", lambda *a, **kw: mock_embedder)

        fake_model = MagicMock()
        monkeypatch.setattr(
            flows, "get_or_create_embedding_model", AsyncMock(return_value=fake_model)
        )
        monkeypatch.setattr(
            flows,
            "embed_chunks_for_document",
            AsyncMock(
                return_value={"chunks_embedded": 0, "chunks_deduplicated": 0, "tokens_used": 0}
            ),
        )

        await flows.embed_document(doc_id, coll_id)

        assert parse_calls == ["gemini:text-embedding-004"]


# ---------------------------------------------------------------------------
# sync_source: stale recovery details
# ---------------------------------------------------------------------------


class TestSyncSourceStaleRecoveryDetails:
    async def test_stale_runs_recovered_fresh_runs_block(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Stale runs are marked failed, but fresh running run blocks new sync."""
        from contextmine_core import SyncRunStatus

        source = _make_source()

        stale_run = MagicMock()
        stale_run.started_at = datetime.now(UTC) - timedelta(hours=12)

        fresh_run = MagicMock()
        fresh_run.started_at = datetime.now(UTC) - timedelta(minutes=5)

        call_count = 0

        async def mock_execute(stmt):
            nonlocal call_count
            call_count += 1
            r = MagicMock()
            if call_count == 1:
                # Lock
                r.scalar_one_or_none.return_value = source
            elif call_count == 2:
                # Running runs (includes stale + fresh)
                r.scalars.return_value.all.return_value = [stale_run, fresh_run]
            elif call_count == 3:
                # Re-check after recovery: fresh_run is still running
                r.scalar_one_or_none.return_value = fresh_run
            return r

        mock_session = AsyncMock()
        mock_session.execute = mock_execute
        mock_session.commit = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        result = await flows.sync_source.fn(source)

        # Stale run marked as failed
        assert stale_run.status == SyncRunStatus.FAILED
        # Fresh run blocked, so returned None
        assert result is None


# ---------------------------------------------------------------------------
# build_twin_graph: arch docs
# ---------------------------------------------------------------------------


class TestBuildTwinGraphArchDocs:
    async def test_arch_docs_skipped_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When arch_docs_enabled is False, no arch docs generated."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())

        mock_as_is = MagicMock()
        mock_as_is.id = uuid.uuid4()

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=MagicMock())
        mock_session.commit = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))
        monkeypatch.setattr(
            flows,
            "get_settings",
            lambda: _make_settings(arch_docs_enabled=False),
        )

        monkeypatch.setattr(
            "contextmine_core.twin.get_or_create_as_is_scenario",
            AsyncMock(return_value=mock_as_is),
        )
        monkeypatch.setattr(
            "contextmine_core.twin.seed_scenario_from_knowledge_graph",
            AsyncMock(return_value=(0, 0)),
        )
        monkeypatch.setattr(
            "contextmine_core.twin.refresh_metric_snapshots",
            AsyncMock(return_value=0),
        )
        monkeypatch.setattr(
            "contextmine_core.validation.refresh_validation_snapshots",
            AsyncMock(return_value=0),
        )
        monkeypatch.setattr(
            "contextmine_core.graph.age.sync_scenario_to_age",
            AsyncMock(),
        )

        result = await flows.build_twin_graph.fn(
            source_id=source_id,
            collection_id=collection_id,
            snapshot_dicts=None,
            changed_doc_ids=None,
            file_metrics=None,
            evolution_payload=None,
        )

        assert result["arch_facts_count"] == 0
        assert result["arch_drift_deltas"] == 0

    async def test_arch_docs_enabled_but_not_on_sync(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When arch_docs_enabled but not generate_on_sync, skipped."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())

        mock_as_is = MagicMock()
        mock_as_is.id = uuid.uuid4()

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=MagicMock())
        mock_session.commit = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))
        monkeypatch.setattr(
            flows,
            "get_settings",
            lambda: _make_settings(arch_docs_enabled=True, arch_docs_generate_on_sync=False),
        )

        monkeypatch.setattr(
            "contextmine_core.twin.get_or_create_as_is_scenario",
            AsyncMock(return_value=mock_as_is),
        )
        monkeypatch.setattr(
            "contextmine_core.twin.seed_scenario_from_knowledge_graph",
            AsyncMock(return_value=(0, 0)),
        )
        monkeypatch.setattr(
            "contextmine_core.twin.refresh_metric_snapshots",
            AsyncMock(return_value=0),
        )
        monkeypatch.setattr(
            "contextmine_core.validation.refresh_validation_snapshots",
            AsyncMock(return_value=0),
        )
        monkeypatch.setattr(
            "contextmine_core.graph.age.sync_scenario_to_age",
            AsyncMock(),
        )

        result = await flows.build_twin_graph.fn(
            source_id=source_id,
            collection_id=collection_id,
            snapshot_dicts=None,
            changed_doc_ids=None,
            file_metrics=None,
            evolution_payload=None,
        )

        assert result.get("arch_docs_sync_generation") == "skipped_requires_explicit_trigger"

    async def test_generate_arch_docs_skips_llm_when_deterministic_hash_is_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from contextmine_core.architecture.schemas import ArchitectureFactsBundle

        collection_id = uuid.uuid4()
        scenario_id = uuid.uuid4()
        deterministic_bundle = ArchitectureFactsBundle(
            collection_id=collection_id,
            scenario_id=scenario_id,
            scenario_name="AS-IS",
        )
        artifact = SimpleNamespace(meta={"facts_hash": deterministic_bundle.facts_hash()})
        as_is = SimpleNamespace(id=scenario_id, name="AS-IS")
        settings = _make_settings(arch_docs_llm_enrich=True, arch_docs_drift_enabled=True)

        build_architecture_facts = AsyncMock(return_value=deterministic_bundle)
        monkeypatch.setattr("contextmine_core.architecture.build_architecture_facts", build_architecture_facts)
        monkeypatch.setattr("contextmine_core.research.llm.get_llm_provider", lambda *_args: object())

        async def mock_execute(stmt):
            result = MagicMock()
            statement = str(stmt)
            if "knowledge_artifacts" in statement:
                result.scalar_one_or_none.return_value = artifact
                return result
            raise AssertionError(f"Unexpected statement: {statement}")

        session = AsyncMock()
        session.execute = mock_execute

        stats = await flows._generate_arch_docs_on_sync(session, settings, collection_id, as_is)

        assert build_architecture_facts.await_count == 1
        assert build_architecture_facts.await_args.kwargs["enable_llm_enrich"] is False
        assert stats["arch_skipped"] is True

    async def test_generate_arch_docs_only_uses_llm_after_hash_miss(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from contextmine_core.architecture.schemas import ArchitectureFact, ArchitectureFactsBundle

        collection_id = uuid.uuid4()
        scenario_id = uuid.uuid4()
        deterministic_bundle = ArchitectureFactsBundle(
            collection_id=collection_id,
            scenario_id=scenario_id,
            scenario_name="AS-IS",
        )
        enriched_bundle = ArchitectureFactsBundle(
            collection_id=collection_id,
            scenario_id=scenario_id,
            scenario_name="AS-IS",
        )
        enriched_bundle.facts.append(
            ArchitectureFact(
                fact_id="quality",
                fact_type="quality_summary",
                title="Quality",
                description="desc",
                source="deterministic",
                confidence=1.0,
                tags=(),
                attributes={},
                evidence=(),
            )
        )
        artifact = SimpleNamespace(meta={"facts_hash": "outdated"})
        as_is = SimpleNamespace(id=scenario_id, name="AS-IS")
        settings = _make_settings(arch_docs_llm_enrich=True, arch_docs_drift_enabled=False)

        build_architecture_facts = AsyncMock(side_effect=[deterministic_bundle, enriched_bundle])
        monkeypatch.setattr("contextmine_core.architecture.build_architecture_facts", build_architecture_facts)
        monkeypatch.setattr("contextmine_core.research.llm.get_llm_provider", lambda *_args: object())
        monkeypatch.setattr(
            "contextmine_core.architecture.generate_arc42_from_facts",
            lambda bundle, scenario, options=None: SimpleNamespace(
                generated_at=datetime.now(UTC),
                confidence_summary={},
                section_coverage={},
                warnings=[],
                sections={},
                markdown="# arc42\n",
            ),
        )

        async def mock_execute(stmt):
            result = MagicMock()
            statement = str(stmt)
            if "knowledge_artifacts" in statement:
                result.scalar_one_or_none.return_value = artifact
                return result
            raise AssertionError(f"Unexpected statement: {statement}")

        session = AsyncMock()
        session.execute = mock_execute
        session.add = MagicMock()

        await flows._generate_arch_docs_on_sync(session, settings, collection_id, as_is)

        assert build_architecture_facts.await_count == 2
        assert build_architecture_facts.await_args_list[0].kwargs["enable_llm_enrich"] is False
        assert build_architecture_facts.await_args_list[1].kwargs["enable_llm_enrich"] is True
        assert build_architecture_facts.await_args_list[1].kwargs["llm_hypothesis_limit"] == 12

    async def test_generate_arch_docs_uses_deterministic_baseline_for_drift(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from contextmine_core.architecture.schemas import ArchitectureFactsBundle

        collection_id = uuid.uuid4()
        scenario_id = uuid.uuid4()
        baseline_id = uuid.uuid4()
        deterministic_bundle = ArchitectureFactsBundle(
            collection_id=collection_id,
            scenario_id=scenario_id,
            scenario_name="AS-IS",
        )
        enriched_bundle = ArchitectureFactsBundle(
            collection_id=collection_id,
            scenario_id=scenario_id,
            scenario_name="AS-IS",
        )
        baseline_bundle = ArchitectureFactsBundle(
            collection_id=collection_id,
            scenario_id=baseline_id,
            scenario_name="BASELINE",
        )
        as_is = SimpleNamespace(id=scenario_id, name="AS-IS")
        baseline = SimpleNamespace(id=baseline_id)
        settings = _make_settings(arch_docs_llm_enrich=True, arch_docs_drift_enabled=True)

        build_architecture_facts = AsyncMock(
            side_effect=[deterministic_bundle, enriched_bundle, baseline_bundle]
        )
        monkeypatch.setattr("contextmine_core.architecture.build_architecture_facts", build_architecture_facts)
        monkeypatch.setattr("contextmine_core.research.llm.get_llm_provider", lambda *_args: object())
        monkeypatch.setattr(
            "contextmine_core.architecture.generate_arc42_from_facts",
            lambda bundle, scenario, options=None: SimpleNamespace(
                generated_at=datetime.now(UTC),
                confidence_summary={},
                section_coverage={},
                warnings=[],
                sections={},
                markdown="# arc42\n",
            ),
        )
        monkeypatch.setattr(
            "contextmine_core.architecture.compute_arc42_drift",
            lambda current, baseline_bundle, baseline_scenario_id=None: SimpleNamespace(
                generated_at=datetime.now(UTC),
                current_hash=current.facts_hash(),
                baseline_hash=baseline_bundle.facts_hash() if baseline_bundle else None,
                deltas=[],
                warnings=[],
            ),
        )

        async def mock_execute(stmt):
            result = MagicMock()
            statement = str(stmt)
            if "knowledge_artifacts" in statement:
                result.scalar_one_or_none.return_value = None
                return result
            if "FROM twin_scenarios" in statement:
                result.scalar_one_or_none.return_value = baseline
                return result
            raise AssertionError(f"Unexpected statement: {statement}")

        session = AsyncMock()
        session.execute = mock_execute
        session.add = MagicMock()

        await flows._generate_arch_docs_on_sync(session, settings, collection_id, as_is)

        assert build_architecture_facts.await_count == 3
        assert [call.kwargs["enable_llm_enrich"] for call in build_architecture_facts.await_args_list] == [
            False,
            True,
            False,
        ]


# ---------------------------------------------------------------------------
# build_twin_graph: file metrics path
# ---------------------------------------------------------------------------


class TestBuildTwinGraphFileMetrics:
    async def test_file_metrics_applied_to_scenario(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """File metrics are applied and gate computed."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())

        mock_as_is = MagicMock()
        mock_as_is.id = uuid.uuid4()

        # Build session that returns file nodes
        file_node = MagicMock()
        file_node.natural_key = "file:src/main.py"
        file_node.kind = "file"

        call_count = 0

        async def mock_execute(stmt):
            nonlocal call_count
            call_count += 1
            r = MagicMock()
            if call_count == 1:
                # TwinNode query
                r.scalars.return_value.all.return_value = [file_node]
            return r

        mock_session = AsyncMock()
        mock_session.execute = mock_execute
        mock_session.commit = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))
        monkeypatch.setattr(
            flows,
            "get_settings",
            lambda: _make_settings(arch_docs_enabled=False, metrics_strict_mode=False),
        )

        monkeypatch.setattr(
            "contextmine_core.twin.get_or_create_as_is_scenario",
            AsyncMock(return_value=mock_as_is),
        )
        monkeypatch.setattr(
            "contextmine_core.twin.seed_scenario_from_knowledge_graph",
            AsyncMock(return_value=(0, 0)),
        )
        monkeypatch.setattr(
            "contextmine_core.twin.apply_file_metrics_to_scenario",
            AsyncMock(return_value=1),
        )
        monkeypatch.setattr(
            "contextmine_core.twin.refresh_metric_snapshots",
            AsyncMock(return_value=0),
        )
        monkeypatch.setattr(
            "contextmine_core.validation.refresh_validation_snapshots",
            AsyncMock(return_value=0),
        )
        monkeypatch.setattr(
            "contextmine_core.graph.age.sync_scenario_to_age",
            AsyncMock(),
        )
        monkeypatch.setattr(
            "contextmine_core.pathing.canonicalize_repo_relative_path",
            lambda p: p,
        )

        result = await flows.build_twin_graph.fn(
            source_id=source_id,
            collection_id=collection_id,
            snapshot_dicts=None,
            changed_doc_ids=None,
            file_metrics=[{"file_path": "src/main.py", "complexity": 10}],
            evolution_payload=None,
        )

        assert result["twin_metric_nodes_enriched"] == 1
        assert result["metrics_requested_files"] == 1
