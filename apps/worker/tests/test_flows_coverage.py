"""Extended coverage tests for contextmine_worker.flows.

Targets the uncovered async functions, error paths, and large pipeline
functions that constitute the bulk of uncovered lines.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
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
# materialize_surface_catalog_for_source: edge cases
# ---------------------------------------------------------------------------


class TestMaterializeSurfaceCatalogEdgeCases:
    async def test_skips_empty_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Documents with empty content are skipped."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())

        mock_session = AsyncMock()
        docs_result = MagicMock()
        docs_result.all.return_value = [
            ("git://github.com/o/r/file.py", None),
            ("git://github.com/o/r/empty.py", ""),
        ]
        mock_session.execute = AsyncMock(return_value=docs_result)
        mock_session.commit = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        result = await flows.materialize_surface_catalog_for_source(
            source_id=source_id,
            collection_id=collection_id,
        )

        assert result["surface_files_scanned"] == 0

    async def test_skips_ignored_paths(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Documents in node_modules are skipped."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())

        mock_session = AsyncMock()
        docs_result = MagicMock()
        docs_result.all.return_value = [
            ("git://github.com/o/r/node_modules/foo/openapi.yaml", "openapi: 3.0"),
        ]
        mock_session.execute = AsyncMock(return_value=docs_result)
        mock_session.commit = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        result = await flows.materialize_surface_catalog_for_source(
            source_id=source_id,
            collection_id=collection_id,
        )

        assert result["surface_files_scanned"] == 0

    async def test_parse_error_counted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When add_file raises, it is counted as a parse error."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())

        mock_session = AsyncMock()
        docs_result = MagicMock()
        docs_result.all.return_value = [
            ("git://github.com/o/r/api.yaml", "some content"),
        ]
        mock_session.execute = AsyncMock(return_value=docs_result)
        mock_session.commit = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        mock_extractor = MagicMock()
        mock_extractor.add_file.side_effect = ValueError("bad yaml")
        mock_extractor.catalog = MagicMock()
        mock_extractor.catalog.openapi_specs = []
        mock_extractor.catalog.graphql_schemas = []
        mock_extractor.catalog.protobuf_files = []
        mock_extractor.catalog.job_definitions = []

        monkeypatch.setattr(
            "contextmine_core.analyzer.extractors.surface.SurfaceCatalogExtractor",
            lambda: mock_extractor,
        )

        result = await flows.materialize_surface_catalog_for_source(
            source_id=source_id,
            collection_id=collection_id,
        )

        assert result["surface_files_scanned"] == 1
        assert result["surface_parse_errors"] == 1

    async def test_has_surfaces_calls_build_graph(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When surfaces are found, build_surface_graph is called."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())

        mock_session = AsyncMock()
        docs_result = MagicMock()
        docs_result.all.return_value = [
            ("git://github.com/o/r/api.yaml", "openapi: '3.0.0'"),
        ]
        mock_session.execute = AsyncMock(return_value=docs_result)
        mock_session.commit = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        mock_extractor = MagicMock()
        mock_extractor.add_file.return_value = True
        mock_extractor.catalog.openapi_specs = [MagicMock()]
        mock_extractor.catalog.graphql_schemas = []
        mock_extractor.catalog.protobuf_files = []
        mock_extractor.catalog.job_definitions = []

        monkeypatch.setattr(
            "contextmine_core.analyzer.extractors.surface.SurfaceCatalogExtractor",
            lambda: mock_extractor,
        )

        mock_build = AsyncMock(
            return_value={"endpoint_nodes": 5, "job_nodes": 2, "edges_created": 3}
        )
        monkeypatch.setattr(
            "contextmine_core.analyzer.extractors.surface.build_surface_graph",
            mock_build,
        )

        result = await flows.materialize_surface_catalog_for_source(
            source_id=source_id,
            collection_id=collection_id,
        )

        assert result["endpoint_nodes"] == 5
        assert result["job_nodes"] == 2
        mock_build.assert_awaited_once()


# ---------------------------------------------------------------------------
# embed_chunks_for_document: embedding error fallback
# ---------------------------------------------------------------------------


class TestEmbedChunksEmbeddingError:
    async def test_embedding_generic_error_uses_fake_embedder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On generic embedding error, falls back to FakeEmbedder."""
        doc_id = str(uuid.uuid4())
        model = MagicMock()
        model.id = uuid.uuid4()
        model.provider = "openai"
        model.model_name = "text-embedding-3-small"
        model.dimension = 4

        chunk_row = SimpleNamespace(id=uuid.uuid4(), chunk_hash="err_hash", content="text")

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
                r.all.return_value = []
                s.execute = AsyncMock(return_value=r)
            else:
                s.execute = AsyncMock(return_value=MagicMock())
                s.commit = AsyncMock()
            return _mock_session_cm(s)

        monkeypatch.setattr(flows, "get_session", make_session)
        monkeypatch.setattr(flows, "_embedding_batch_timeout_seconds", lambda: 120)

        # Mock embedder that raises a non-timeout error
        mock_embedder = MagicMock()

        async def _error_embed(texts):
            raise ConnectionError("API down")

        mock_embedder.embed_batch = _error_embed
        monkeypatch.setattr(flows, "get_embedder", lambda **kw: mock_embedder)

        fake_result = MagicMock()
        fake_result.embeddings = [[0.1, 0.2, 0.3, 0.4]]
        fake_result.tokens_used = 3

        mock_fake_embedder = MagicMock()
        mock_fake_embedder.embed_batch = AsyncMock(return_value=fake_result)
        monkeypatch.setattr(flows, "FakeEmbedder", lambda dimension: mock_fake_embedder)

        original_wait_for = asyncio.wait_for

        async def patched_wait_for(coro, *, timeout=None):
            try:
                return await coro
            except TimeoutError:
                raise

        monkeypatch.setattr(asyncio, "wait_for", patched_wait_for)

        stats = await flows.embed_chunks_for_document.fn(doc_id, model)

        assert stats["chunks_embedded"] == 1
        assert stats["tokens_used"] == 3

        monkeypatch.setattr(asyncio, "wait_for", original_wait_for)


# ---------------------------------------------------------------------------
# get_embedding_model_for_collection
# ---------------------------------------------------------------------------


class TestGetEmbeddingModelForCollection:
    async def test_returns_collection_config_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        collection_id = str(uuid.uuid4())
        mock_session = AsyncMock()
        r = MagicMock()
        r.scalar_one_or_none.return_value = {"embedding_model": "openai:custom-model"}
        mock_session.execute = AsyncMock(return_value=r)

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        result = await flows.get_embedding_model_for_collection(collection_id)
        assert result == "openai:custom-model"

    async def test_falls_back_to_global_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        collection_id = str(uuid.uuid4())
        mock_session = AsyncMock()
        r = MagicMock()
        r.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=r)

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))
        monkeypatch.setattr(flows, "get_settings", lambda: _make_settings())

        result = await flows.get_embedding_model_for_collection(collection_id)
        assert result == "openai:text-embedding-3-small"

    async def test_empty_config_model_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        collection_id = str(uuid.uuid4())
        mock_session = AsyncMock()
        r = MagicMock()
        r.scalar_one_or_none.return_value = {"embedding_model": ""}
        mock_session.execute = AsyncMock(return_value=r)

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))
        monkeypatch.setattr(flows, "get_settings", lambda: _make_settings())

        result = await flows.get_embedding_model_for_collection(collection_id)
        assert result == "openai:text-embedding-3-small"


# ---------------------------------------------------------------------------
# embed_document
# ---------------------------------------------------------------------------


class TestEmbedDocument:
    async def test_with_collection_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())

        monkeypatch.setattr(
            flows,
            "get_embedding_model_for_collection",
            AsyncMock(return_value="openai:text-embedding-3-small"),
        )
        monkeypatch.setattr(
            flows,
            "parse_embedding_model_spec",
            lambda spec: ("openai", "text-embedding-3-small"),
        )

        mock_embedder = MagicMock()
        mock_embedder.dimension = 1536
        monkeypatch.setattr(flows, "get_embedder", lambda *a, **kw: mock_embedder)

        mock_model = MagicMock()
        monkeypatch.setattr(
            flows,
            "get_or_create_embedding_model",
            AsyncMock(return_value=mock_model),
        )
        monkeypatch.setattr(
            flows,
            "embed_chunks_for_document",
            AsyncMock(
                return_value={
                    "chunks_embedded": 5,
                    "chunks_deduplicated": 1,
                    "tokens_used": 100,
                }
            ),
        )

        result = await flows.embed_document(doc_id, collection_id)
        assert result["chunks_embedded"] == 5

    async def test_without_collection_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc_id = str(uuid.uuid4())

        monkeypatch.setattr(flows, "get_settings", lambda: _make_settings())
        monkeypatch.setattr(
            flows,
            "parse_embedding_model_spec",
            lambda spec: ("openai", "text-embedding-3-small"),
        )

        mock_embedder = MagicMock()
        mock_embedder.dimension = 1536
        monkeypatch.setattr(flows, "get_embedder", lambda *a, **kw: mock_embedder)

        mock_model = MagicMock()
        monkeypatch.setattr(
            flows,
            "get_or_create_embedding_model",
            AsyncMock(return_value=mock_model),
        )
        monkeypatch.setattr(
            flows,
            "embed_chunks_for_document",
            AsyncMock(
                return_value={
                    "chunks_embedded": 3,
                    "chunks_deduplicated": 0,
                    "tokens_used": 50,
                }
            ),
        )

        result = await flows.embed_document(doc_id)
        assert result["chunks_embedded"] == 3


# ---------------------------------------------------------------------------
# build_knowledge_graph: provider validation
# ---------------------------------------------------------------------------


class TestBuildKnowledgeGraphValidation:
    async def test_no_llm_provider_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When no LLM provider configured, raises ValueError."""
        monkeypatch.setattr(flows, "get_settings", lambda: _make_settings(default_llm_provider=""))

        with pytest.raises(ValueError, match="LLM provider required"):
            await flows.build_knowledge_graph.fn(
                source_id=str(uuid.uuid4()),
                collection_id=str(uuid.uuid4()),
            )

    async def test_llm_provider_init_fails_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When LLM provider init fails, raises ValueError."""
        monkeypatch.setattr(
            flows,
            "get_settings",
            lambda: _make_settings(default_llm_provider="openai"),
        )
        monkeypatch.setattr(
            "contextmine_core.research.llm.get_llm_provider",
            MagicMock(side_effect=RuntimeError("no key")),
        )

        with pytest.raises(ValueError, match="failed to initialize"):
            await flows.build_knowledge_graph.fn(
                source_id=str(uuid.uuid4()),
                collection_id=str(uuid.uuid4()),
            )

    async def test_no_research_llm_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When research LLM not available, raises ValueError."""
        monkeypatch.setattr(
            flows,
            "get_settings",
            lambda: _make_settings(default_llm_provider="openai"),
        )
        monkeypatch.setattr(
            "contextmine_core.research.llm.get_llm_provider",
            lambda *a, **kw: MagicMock(),
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

    async def test_embedder_init_fails_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When embedder init fails, raises ValueError."""
        monkeypatch.setattr(
            flows,
            "get_settings",
            lambda: _make_settings(default_llm_provider="openai"),
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
            flows,
            "parse_embedding_model_spec",
            MagicMock(side_effect=RuntimeError("bad spec")),
        )

        with pytest.raises(ValueError, match="Embedder required"):
            await flows.build_knowledge_graph.fn(
                source_id=str(uuid.uuid4()),
                collection_id=str(uuid.uuid4()),
            )


# ---------------------------------------------------------------------------
# build_knowledge_graph: business rules extraction step
# ---------------------------------------------------------------------------


class TestBuildKnowledgeGraphBusinessRules:
    async def test_extracts_rules_from_changed_docs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Step 2: business rules extracted from changed documents."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())
        doc_id = str(uuid.uuid4())

        monkeypatch.setattr(
            flows,
            "get_settings",
            lambda: _make_settings(default_llm_provider="openai"),
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

        async def mock_exec(stmt):
            r = MagicMock()
            r.all.return_value = [
                (
                    uuid.uuid4(),
                    "git://github.com/o/r/main.py",
                    "def foo(): raise ValueError('bad')",
                ),
            ]
            r.scalar_one_or_none.return_value = None
            return r

        mock_session.execute = mock_exec
        mock_session.commit = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        # Step 1
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

        # Step 2: rules extraction
        monkeypatch.setattr(
            "contextmine_core.treesitter.languages.detect_language",
            lambda fp: "python",
        )
        mock_rule_result = MagicMock()
        mock_rule_result.rules = [MagicMock()]
        monkeypatch.setattr(
            "contextmine_core.analyzer.extractors.rules.extract_rules_from_file",
            AsyncMock(return_value=mock_rule_result),
        )
        monkeypatch.setattr(
            "contextmine_core.analyzer.extractors.rules.build_rules_graph",
            AsyncMock(return_value={"rules_created": 2}),
        )

        # Steps 3-7: mock out remaining
        mock_erm = MagicMock()
        mock_erm.schema.tables = []
        mock_erm.add_alembic_extraction = MagicMock()
        monkeypatch.setattr(
            "contextmine_core.analyzer.extractors.erm.ERMExtractor",
            lambda: mock_erm,
        )

        mock_surf = MagicMock()
        mock_surf.add_file.return_value = False
        mock_surf.catalog.openapi_specs = []
        mock_surf.catalog.graphql_schemas = []
        mock_surf.catalog.protobuf_files = []
        mock_surf.catalog.job_definitions = []
        monkeypatch.setattr(
            "contextmine_core.analyzer.extractors.surface.SurfaceCatalogExtractor",
            lambda: mock_surf,
        )

        monkeypatch.setattr(
            "contextmine_core.knowledge.extraction.extract_from_documents",
            AsyncMock(return_value=MagicMock()),
        )
        monkeypatch.setattr(
            "contextmine_core.knowledge.extraction.persist_semantic_entities",
            AsyncMock(return_value={"entities_created": 0, "relationships_created": 0}),
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
        mock_summary = MagicMock()
        mock_summary.communities_summarized = 0
        mock_summary.embeddings_created = 0
        monkeypatch.setattr(
            "contextmine_core.knowledge.summaries.generate_community_summaries",
            AsyncMock(return_value=mock_summary),
        )

        result = await flows.build_knowledge_graph.fn(
            source_id=source_id,
            collection_id=collection_id,
            changed_doc_ids=[doc_id],
        )

        assert result["kg_business_rules"] == 2


# ---------------------------------------------------------------------------
# build_knowledge_graph: surface extraction step
# ---------------------------------------------------------------------------


class TestBuildKnowledgeGraphSurfaces:
    async def test_surfaces_extracted_and_counted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Step 4: surface extraction creates endpoint and job nodes."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())

        monkeypatch.setattr(
            flows,
            "get_settings",
            lambda: _make_settings(default_llm_provider="openai"),
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
        docs_result = MagicMock()
        docs_result.all.return_value = [
            ("git://github.com/o/r/openapi.yaml", "openapi: 3.0"),
        ]
        mock_session.execute = AsyncMock(return_value=docs_result)
        mock_session.commit = AsyncMock()
        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        # Step 1
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

        # Step 3: no tables
        mock_erm = MagicMock()
        mock_erm.schema.tables = []
        mock_erm.add_alembic_extraction = MagicMock()
        monkeypatch.setattr(
            "contextmine_core.analyzer.extractors.erm.ERMExtractor",
            lambda: mock_erm,
        )

        # Step 4: surface
        mock_surf = MagicMock()
        mock_surf.add_file.return_value = True
        mock_surf.catalog.openapi_specs = [MagicMock()]
        mock_surf.catalog.graphql_schemas = []
        mock_surf.catalog.protobuf_files = []
        mock_surf.catalog.job_definitions = []
        monkeypatch.setattr(
            "contextmine_core.analyzer.extractors.surface.SurfaceCatalogExtractor",
            lambda: mock_surf,
        )
        monkeypatch.setattr(
            "contextmine_core.analyzer.extractors.surface.build_surface_graph",
            AsyncMock(return_value={"endpoint_nodes": 10, "job_nodes": 2}),
        )

        # Steps 5-7
        monkeypatch.setattr(
            "contextmine_core.knowledge.extraction.extract_from_documents",
            AsyncMock(return_value=MagicMock()),
        )
        monkeypatch.setattr(
            "contextmine_core.knowledge.extraction.persist_semantic_entities",
            AsyncMock(return_value={"entities_created": 0, "relationships_created": 0}),
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
        mock_summary = MagicMock()
        mock_summary.communities_summarized = 0
        mock_summary.embeddings_created = 0
        monkeypatch.setattr(
            "contextmine_core.knowledge.summaries.generate_community_summaries",
            AsyncMock(return_value=mock_summary),
        )

        result = await flows.build_knowledge_graph.fn(
            source_id=source_id,
            collection_id=collection_id,
            changed_doc_ids=[],
        )

        assert result["kg_endpoints"] == 10
        assert result["kg_jobs"] == 2


# ---------------------------------------------------------------------------
# build_twin_graph: with file_metrics and metrics gate
# ---------------------------------------------------------------------------


class TestBuildTwinGraphFileMetrics:
    async def test_file_metrics_applied_and_gate_passes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """File metrics are applied; when all mapped, gate passes."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())

        mock_as_is = MagicMock()
        mock_as_is.id = uuid.uuid4()

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            return_value=MagicMock(
                scalars=MagicMock(
                    return_value=MagicMock(
                        all=MagicMock(
                            return_value=[
                                MagicMock(
                                    natural_key="file:src/main.py",
                                    kind="file",
                                    meta={},
                                ),
                            ]
                        )
                    )
                )
            )
        )
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
            "contextmine_core.twin.apply_file_metrics_to_scenario",
            AsyncMock(return_value=1),
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
        monkeypatch.setattr(
            "contextmine_core.pathing.canonicalize_repo_relative_path",
            lambda p: p,
        )

        result = await flows.build_twin_graph.fn(
            source_id=source_id,
            collection_id=collection_id,
            snapshot_dicts=None,
            changed_doc_ids=None,
            file_metrics=[{"file_path": "src/main.py", "complexity": 5}],
            evolution_payload=None,
        )

        assert result["twin_metric_nodes_enriched"] == 1

    async def test_metrics_gate_fail_strict_mode_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When metrics gate fails in strict mode, RuntimeError is raised."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())

        mock_as_is = MagicMock()
        mock_as_is.id = uuid.uuid4()

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            return_value=MagicMock(
                scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
            )
        )
        mock_session.commit = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))
        monkeypatch.setattr(
            flows,
            "get_settings",
            lambda: _make_settings(arch_docs_enabled=False, metrics_strict_mode=True),
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
            AsyncMock(return_value=0),
        )
        monkeypatch.setattr(
            "contextmine_core.pathing.canonicalize_repo_relative_path",
            lambda p: p,
        )

        with pytest.raises(RuntimeError, match="METRICS_GATE_FAILED"):
            await flows.build_twin_graph.fn(
                source_id=source_id,
                collection_id=collection_id,
                snapshot_dicts=None,
                changed_doc_ids=None,
                file_metrics=[{"file_path": "not_found.py", "complexity": 5}],
                evolution_payload=None,
            )


# ---------------------------------------------------------------------------
# _fail_running_sync_runs_for_source
# ---------------------------------------------------------------------------


class TestFailRunningSyncRunsForSource:
    async def test_marks_running_rows_failed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        source_id = str(uuid.uuid4())
        mock_run = MagicMock()

        mock_session = AsyncMock()
        r = MagicMock()
        r.scalars.return_value.all.return_value = [mock_run]
        mock_session.execute = AsyncMock(return_value=r)
        mock_session.commit = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        count = await flows._fail_running_sync_runs_for_source(source_id, "test_reason")

        assert count == 1
        assert mock_run.status == flows.SyncRunStatus.FAILED
        assert mock_run.error == "test_reason"

    async def test_no_running_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        source_id = str(uuid.uuid4())

        mock_session = AsyncMock()
        r = MagicMock()
        r.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=r)

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        count = await flows._fail_running_sync_runs_for_source(source_id, "no_rows")
        assert count == 0


# ---------------------------------------------------------------------------
# _fail_coverage_ingest_job
# ---------------------------------------------------------------------------


class TestFailCoverageIngestJob:
    async def test_updates_job_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        job_id = str(uuid.uuid4())
        mock_job = MagicMock()

        mock_session = AsyncMock()
        r = MagicMock()
        r.scalar_one_or_none.return_value = mock_job
        mock_session.execute = AsyncMock(return_value=r)
        mock_session.commit = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        result = await flows._fail_coverage_ingest_job(
            job_id,
            error_code="TEST_ERR",
            error_detail="test detail",
        )

        assert result["status"] == "failed"
        assert mock_job.error_code == "TEST_ERR"

    async def test_missing_job_returns_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        job_id = str(uuid.uuid4())

        mock_session = AsyncMock()
        r = MagicMock()
        r.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=r)

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        result = await flows._fail_coverage_ingest_job(
            job_id,
            error_code="TEST",
            error_detail="detail",
        )

        assert result["status"] == "missing_job"


# ---------------------------------------------------------------------------
# ingest_coverage_metrics
# ---------------------------------------------------------------------------


class TestIngestCoverageMetrics:
    async def test_invalid_job_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Invalid UUID job_id returns error."""
        result = await flows.ingest_coverage_metrics.fn("not-a-uuid")

        assert result["status"] == "failed"
        assert result["error_code"] == "INGEST_APPLY_FAILED"

    async def test_missing_job(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When job is not found, returns missing."""
        job_id = str(uuid.uuid4())

        mock_session = AsyncMock()
        r = MagicMock()
        r.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=r)
        mock_session.commit = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        result = await flows.ingest_coverage_metrics.fn(job_id)
        assert result["status"] == "missing"

    async def test_already_terminal_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When job already applied/failed/rejected, returns that status."""
        job_id = str(uuid.uuid4())

        mock_job = MagicMock()
        mock_job.status = "applied"
        mock_job.error_code = None
        mock_job.error_detail = None

        mock_session = AsyncMock()
        r = MagicMock()
        r.scalar_one_or_none.return_value = mock_job
        mock_session.execute = AsyncMock(return_value=r)
        mock_session.commit = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        result = await flows.ingest_coverage_metrics.fn(job_id)
        assert result["status"] == "applied"

    async def test_source_not_found_rejects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When source is not found, rejects the job."""
        job_id = str(uuid.uuid4())

        session_call = 0

        def make_session():
            nonlocal session_call
            session_call += 1
            s = AsyncMock()
            if session_call == 1:
                # Lock phase: job found, not terminal
                mock_job = MagicMock()
                mock_job.status = "queued"
                mock_job.error_code = None
                mock_job.error_detail = None
                r = MagicMock()
                r.scalar_one_or_none.return_value = mock_job
                s.execute = AsyncMock(return_value=r)
                s.commit = AsyncMock()
            elif session_call == 2:
                # Re-read phase: job found
                mock_job2 = MagicMock()
                mock_job2.id = uuid.UUID(job_id)
                mock_job2.source_id = uuid.uuid4()
                mock_job2.status = "processing"
                call_idx = 0

                async def exec_fn(stmt):
                    nonlocal call_idx
                    call_idx += 1
                    r2 = MagicMock()
                    if call_idx == 1:
                        r2.scalar_one_or_none.return_value = mock_job2
                    else:
                        r2.scalar_one_or_none.return_value = None
                    return r2

                s.execute = exec_fn
                s.commit = AsyncMock()
            else:
                # _fail_coverage_ingest_job session
                mock_fail_job = MagicMock()
                r3 = MagicMock()
                r3.scalar_one_or_none.return_value = mock_fail_job
                s.execute = AsyncMock(return_value=r3)
                s.commit = AsyncMock()
            return _mock_session_cm(s)

        monkeypatch.setattr(flows, "get_session", make_session)

        result = await flows.ingest_coverage_metrics.fn(job_id)
        assert result["status"] == "rejected"


# ---------------------------------------------------------------------------
# get_github_token_for_source: edge cases
# ---------------------------------------------------------------------------


class TestGetGithubTokenEdgeCases:
    async def test_no_owner_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_session = AsyncMock()
        r = MagicMock()
        r.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=r)

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        result = await flows.get_github_token_for_source(str(uuid.uuid4()), str(uuid.uuid4()))
        assert result is None

    async def test_no_token_record_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        call_count = 0
        owner_id = uuid.uuid4()

        async def mock_execute(stmt):
            nonlocal call_count
            call_count += 1
            r = MagicMock()
            if call_count == 1:
                r.scalar_one_or_none.return_value = owner_id
            else:
                r.scalars.return_value.all.return_value = []
            return r

        mock_session = AsyncMock()
        mock_session.execute = mock_execute

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        result = await flows.get_github_token_for_source(str(uuid.uuid4()), str(uuid.uuid4()))
        assert result is None

    async def test_multiple_tokens_warns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When multiple tokens exist, uses most recent (first)."""
        call_count = 0
        owner_id = uuid.uuid4()

        token1 = MagicMock()
        token1.access_token_encrypted = "enc1"
        token2 = MagicMock()
        token2.access_token_encrypted = "enc2"

        async def mock_execute(stmt):
            nonlocal call_count
            call_count += 1
            r = MagicMock()
            if call_count == 1:
                r.scalar_one_or_none.return_value = owner_id
            else:
                r.scalars.return_value.all.return_value = [token1, token2]
            return r

        mock_session = AsyncMock()
        mock_session.execute = mock_execute

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))
        monkeypatch.setattr(flows, "decrypt_token", lambda v: f"decrypted:{v}")

        result = await flows.get_github_token_for_source(str(uuid.uuid4()), str(uuid.uuid4()))
        assert result == "decrypted:enc1"


# ---------------------------------------------------------------------------
# get_deploy_key_for_source
# ---------------------------------------------------------------------------


class TestGetDeployKeyForSource:
    async def test_returns_decrypted_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_session = AsyncMock()
        r = MagicMock()
        r.scalar_one_or_none.return_value = "encrypted_key"
        mock_session.execute = AsyncMock(return_value=r)

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))
        monkeypatch.setattr(flows, "decrypt_token", lambda v: f"dec:{v}")

        result = await flows.get_deploy_key_for_source(str(uuid.uuid4()))
        assert result == "dec:encrypted_key"

    async def test_no_key_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_session = AsyncMock()
        r = MagicMock()
        r.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=r)

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        result = await flows.get_deploy_key_for_source(str(uuid.uuid4()))
        assert result is None


# ---------------------------------------------------------------------------
# _build_scip_index_config
# ---------------------------------------------------------------------------


class TestBuildScipIndexConfig:
    def test_builds_config_from_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(flows, "get_settings", lambda: _make_settings())

        config = flows._build_scip_index_config()

        assert config.best_effort is True
        assert config.node_memory_mb == 2048

    def test_unknown_language_warns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            flows,
            "get_settings",
            lambda: _make_settings(scip_languages="python,unknown_lang"),
        )

        config = flows._build_scip_index_config()
        # Should still have python
        assert len(config.enabled_languages) >= 1

    def test_invalid_install_mode_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            flows,
            "get_settings",
            lambda: _make_settings(scip_install_deps_mode="invalid_mode"),
        )

        config = flows._build_scip_index_config()
        from contextmine_core.semantic_snapshot.models import InstallDepsMode

        assert config.install_deps_mode == InstallDepsMode.AUTO


# ---------------------------------------------------------------------------
# task_detect_scip_projects
# ---------------------------------------------------------------------------


class TestTaskDetectScipProjects:
    async def test_detects_projects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_project = MagicMock()
        mock_project.to_dict.return_value = {
            "language": "python",
            "root_path": "/tmp",
        }

        mock_diagnostics = MagicMock()
        mock_diagnostics.to_dict.return_value = {"languages_detected": ["python"]}

        monkeypatch.setattr(
            "contextmine_core.semantic_snapshot.indexers.detection"
            ".detect_projects_with_diagnostics",
            lambda path: ([mock_project], mock_diagnostics),
        )

        result = await flows.task_detect_scip_projects.fn(Path("/tmp/repo"))

        assert len(result["projects"]) == 1
        assert result["projects"][0]["language"] == "python"


# ---------------------------------------------------------------------------
# task_index_scip_project
# ---------------------------------------------------------------------------


class TestTaskIndexScipProject:
    async def test_successful_indexing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(flows, "get_settings", lambda: _make_settings())

        mock_artifact = MagicMock()
        mock_artifact.success = True
        mock_artifact.duration_s = 1.5
        mock_artifact.to_dict.return_value = {
            "success": True,
            "scip_path": "/tmp/scip/output.scip",
        }

        mock_backend = MagicMock()
        mock_backend.can_handle.return_value = True
        mock_backend.index.return_value = mock_artifact

        monkeypatch.setattr(
            "contextmine_core.semantic_snapshot.indexers.BACKENDS",
            [mock_backend],
        )

        mock_target = MagicMock()
        mock_target.language.value = "python"
        mock_target.root_path = "/tmp/repo"
        monkeypatch.setattr(
            "contextmine_core.semantic_snapshot.models.ProjectTarget.from_dict",
            lambda d: mock_target,
        )

        result = await flows.task_index_scip_project.fn(
            {"language": "python", "root_path": "/tmp/repo"},
            Path("/tmp/output"),
        )

        assert result is not None
        assert result["success"] is True

    async def test_failed_indexing_best_effort(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(flows, "get_settings", lambda: _make_settings(scip_best_effort=True))

        mock_artifact = MagicMock()
        mock_artifact.success = False
        mock_artifact.error_message = "parser error"

        mock_backend = MagicMock()
        mock_backend.can_handle.return_value = True
        mock_backend.index.return_value = mock_artifact

        monkeypatch.setattr(
            "contextmine_core.semantic_snapshot.indexers.BACKENDS",
            [mock_backend],
        )

        mock_target = MagicMock()
        mock_target.language.value = "python"
        mock_target.root_path = "/tmp/repo"
        monkeypatch.setattr(
            "contextmine_core.semantic_snapshot.models.ProjectTarget.from_dict",
            lambda d: mock_target,
        )

        result = await flows.task_index_scip_project.fn(
            {"language": "python", "root_path": "/tmp/repo"},
            Path("/tmp/output"),
        )

        assert result is None

    async def test_backend_exception_best_effort(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(flows, "get_settings", lambda: _make_settings(scip_best_effort=True))

        mock_backend = MagicMock()
        mock_backend.can_handle.return_value = True
        mock_backend.index.side_effect = RuntimeError("crash")

        monkeypatch.setattr(
            "contextmine_core.semantic_snapshot.indexers.BACKENDS",
            [mock_backend],
        )

        mock_target = MagicMock()
        mock_target.language.value = "python"
        mock_target.root_path = "/tmp/repo"
        monkeypatch.setattr(
            "contextmine_core.semantic_snapshot.models.ProjectTarget.from_dict",
            lambda d: mock_target,
        )

        result = await flows.task_index_scip_project.fn(
            {"language": "python"},
            Path("/tmp/output"),
        )

        assert result is None

    async def test_no_matching_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(flows, "get_settings", lambda: _make_settings())

        mock_backend = MagicMock()
        mock_backend.can_handle.return_value = False

        monkeypatch.setattr(
            "contextmine_core.semantic_snapshot.indexers.BACKENDS",
            [mock_backend],
        )

        mock_target = MagicMock()
        monkeypatch.setattr(
            "contextmine_core.semantic_snapshot.models.ProjectTarget.from_dict",
            lambda d: mock_target,
        )

        result = await flows.task_index_scip_project.fn(
            {"language": "rust"},
            Path("/tmp/output"),
        )

        assert result is None

    async def test_backend_exception_non_best_effort_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            flows,
            "get_settings",
            lambda: _make_settings(scip_best_effort=False),
        )

        mock_backend = MagicMock()
        mock_backend.can_handle.return_value = True
        mock_backend.index.side_effect = RuntimeError("crash")

        monkeypatch.setattr(
            "contextmine_core.semantic_snapshot.indexers.BACKENDS",
            [mock_backend],
        )

        mock_target = MagicMock()
        mock_target.language.value = "python"
        mock_target.root_path = "/tmp/repo"
        monkeypatch.setattr(
            "contextmine_core.semantic_snapshot.models.ProjectTarget.from_dict",
            lambda d: mock_target,
        )

        with pytest.raises(RuntimeError, match="crash"):
            await flows.task_index_scip_project.fn(
                {"language": "python"},
                Path("/tmp/output"),
            )

    async def test_failed_indexing_non_best_effort(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            flows,
            "get_settings",
            lambda: _make_settings(scip_best_effort=False),
        )

        mock_artifact = MagicMock()
        mock_artifact.success = False
        mock_artifact.error_message = "parser error"

        mock_backend = MagicMock()
        mock_backend.can_handle.return_value = True
        mock_backend.index.return_value = mock_artifact

        monkeypatch.setattr(
            "contextmine_core.semantic_snapshot.indexers.BACKENDS",
            [mock_backend],
        )

        mock_target = MagicMock()
        mock_target.language.value = "python"
        mock_target.root_path = "/tmp/repo"
        monkeypatch.setattr(
            "contextmine_core.semantic_snapshot.models.ProjectTarget.from_dict",
            lambda d: mock_target,
        )

        result = await flows.task_index_scip_project.fn(
            {"language": "python"},
            Path("/tmp/output"),
        )

        assert result is None


# ---------------------------------------------------------------------------
# task_parse_scip_snapshot
# ---------------------------------------------------------------------------


class TestTaskParseScipSnapshot:
    async def test_successful_parse(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_snapshot = MagicMock()
        mock_snapshot.symbols = [1, 2, 3]
        mock_snapshot.relations = [1, 2]
        mock_snapshot.to_dict.return_value = {
            "symbols": [1, 2, 3],
            "relations": [1, 2],
        }

        monkeypatch.setattr(
            "contextmine_core.semantic_snapshot.build_snapshot",
            lambda path: mock_snapshot,
        )

        result = await flows.task_parse_scip_snapshot.fn("/tmp/test.scip")

        assert result is not None
        assert result["symbols"] == [1, 2, 3]

    async def test_parse_failure_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "contextmine_core.semantic_snapshot.build_snapshot",
            MagicMock(side_effect=RuntimeError("bad file")),
        )

        result = await flows.task_parse_scip_snapshot.fn("/tmp/bad.scip")

        assert result is None


# ---------------------------------------------------------------------------
# _materialize_behavioral_layers_impl: disabled
# ---------------------------------------------------------------------------


class TestMaterializeBehavioralLayersDisabled:
    async def test_returns_disabled_when_not_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
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


# ---------------------------------------------------------------------------
# _materialize_behavioral_layers_impl: with extractions
# ---------------------------------------------------------------------------


class TestMaterializeBehavioralLayersWithExtractions:
    async def test_with_test_and_ui_and_flows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When tests and UI are extracted, graphs are built and flows synthesized."""
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
            ("git://github.com/o/r/test_main.py", "def test_foo(): pass"),
        ]
        mock_session.execute = AsyncMock(return_value=docs_result)
        mock_session.commit = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))
        monkeypatch.setattr(
            "contextmine_core.knowledge.builder.cleanup_scoped_knowledge_nodes",
            AsyncMock(return_value={"nodes_deleted": 0, "evidence_deleted": 0}),
        )

        test_ext = MagicMock()
        ui_ext = MagicMock()
        monkeypatch.setattr(
            "contextmine_core.analyzer.extractors.tests.extract_tests_from_files",
            lambda files: [test_ext],
        )
        monkeypatch.setattr(
            "contextmine_core.analyzer.extractors.ui.extract_ui_from_files",
            lambda files: [ui_ext],
        )
        monkeypatch.setattr(
            "contextmine_core.analyzer.extractors.tests.build_tests_graph",
            AsyncMock(return_value={"test_nodes": 3}),
        )
        monkeypatch.setattr(
            "contextmine_core.analyzer.extractors.ui.build_ui_graph",
            AsyncMock(return_value={"ui_nodes": 2}),
        )
        monkeypatch.setattr(
            "contextmine_core.analyzer.extractors.flows.synthesize_user_flows",
            lambda ui, tests: MagicMock(),
        )
        monkeypatch.setattr(
            "contextmine_core.analyzer.extractors.flows.build_flows_graph",
            AsyncMock(return_value={"flow_nodes": 1}),
        )

        result = await flows._materialize_behavioral_layers_impl(
            source_id=source_id,
            collection_id=collection_id,
            scenario_id=None,
            source_version_id=None,
        )

        assert result["behavioral_layers_status"] == "ready"

    async def test_with_source_version_updates_stats(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When source_version_id is given, updates source version stats."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())
        source_version_id = str(uuid.uuid4())

        monkeypatch.setattr(
            flows,
            "get_settings",
            lambda: _make_settings(
                digital_twin_behavioral_enabled=True,
                digital_twin_ui_enabled=True,
                digital_twin_flows_enabled=False,
            ),
        )

        mock_session = AsyncMock()
        mock_source_version = MagicMock()
        mock_source_version.stats = {}

        async def mock_exec(stmt):
            r = MagicMock()
            r.all.return_value = [
                ("git://github.com/o/r/src/app.py", "print('hello')"),
            ]
            r.scalar_one_or_none.return_value = mock_source_version
            return r

        mock_session.execute = mock_exec
        mock_session.commit = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))
        monkeypatch.setattr(
            "contextmine_core.knowledge.builder.cleanup_scoped_knowledge_nodes",
            AsyncMock(return_value={"nodes_deleted": 0, "evidence_deleted": 0}),
        )

        monkeypatch.setattr(
            "contextmine_core.twin.record_twin_event",
            AsyncMock(),
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
            source_version_id=source_version_id,
        )

        assert result["behavioral_layers_status"] == "ready"


# ---------------------------------------------------------------------------
# materialize_behavioral_layers: error path
# ---------------------------------------------------------------------------


class TestMaterializeBehavioralLayersErrorPath:
    async def test_error_records_failure_event(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When _impl raises, error path records failure in source version."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())
        source_version_id = str(uuid.uuid4())

        monkeypatch.setattr(
            flows,
            "_materialize_behavioral_layers_impl",
            AsyncMock(side_effect=RuntimeError("impl failed")),
        )

        mock_session = AsyncMock()
        mock_source_version = MagicMock()
        mock_source_version.stats = {}
        r = MagicMock()
        r.scalar_one_or_none.return_value = mock_source_version
        mock_session.execute = AsyncMock(return_value=r)
        mock_session.commit = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))
        monkeypatch.setattr(
            "contextmine_core.twin.record_twin_event",
            AsyncMock(),
        )

        with pytest.raises(RuntimeError, match="impl failed"):
            await flows.materialize_behavioral_layers.fn(
                source_id=source_id,
                collection_id=collection_id,
                scenario_id=None,
                source_version_id=source_version_id,
            )

        assert mock_source_version.stats["behavioral_layers_status"] == "failed"

    async def test_error_without_source_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When no source_version_id, error propagates without DB update."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())

        monkeypatch.setattr(
            flows,
            "_materialize_behavioral_layers_impl",
            AsyncMock(side_effect=RuntimeError("fail")),
        )

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=MagicMock())
        mock_session.commit = AsyncMock()
        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))
        monkeypatch.setattr(
            "contextmine_core.twin.record_twin_event",
            AsyncMock(),
        )

        with pytest.raises(RuntimeError, match="fail"):
            await flows.materialize_behavioral_layers.fn(
                source_id=source_id,
                collection_id=collection_id,
                scenario_id=None,
                source_version_id=None,
            )


# ---------------------------------------------------------------------------
# sync_source: stale run recovery
# ---------------------------------------------------------------------------


class TestSyncSourceStaleRecovery:
    async def test_stale_run_recovered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When a stale running row exists, it is marked as failed."""
        from contextmine_core import SyncRunStatus

        source = _make_source()

        stale_run = MagicMock()
        stale_run.started_at = datetime.now(UTC) - timedelta(hours=12)

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
                        r.scalars.return_value.all.return_value = [stale_run]
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
        monkeypatch.setattr(flows, "sync_github_source", AsyncMock())

        result = await flows.sync_source.fn(source)
        assert stale_run.status == SyncRunStatus.FAILED
        assert result is not None

    async def test_lock_not_acquired(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When source lock cannot be acquired, returns None."""
        source = _make_source()

        mock_session = AsyncMock()
        r = MagicMock()
        r.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=r)

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        result = await flows.sync_source.fn(source)
        assert result is None

    async def test_existing_run_blocks_sync(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When a fresh running sync exists, returns None."""
        source = _make_source()

        mock_session = AsyncMock()
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
                existing_run = MagicMock()
                r.scalar_one_or_none.return_value = existing_run
            return r

        mock_session.execute = exec_fn
        mock_session.commit = AsyncMock()

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        result = await flows.sync_source.fn(source)
        assert result is None


# ---------------------------------------------------------------------------
# sync_source: exception error handling (error parsing paths)
# ---------------------------------------------------------------------------


class TestSyncSourceErrorParsing:
    async def test_metrics_gate_error_parsed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """METRICS_GATE_FAILED error is parsed into structured stats."""
        from contextmine_core import SyncRunStatus

        source = _make_source()

        session_idx = 0
        mock_db_run = MagicMock()
        mock_db_source = MagicMock()
        mock_db_source.schedule_interval_minutes = 60

        mock_failed_version = MagicMock()
        mock_failed_version.id = uuid.uuid4()

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
                call_count2 = 0

                async def exec_fn2(stmt):
                    nonlocal call_count2
                    call_count2 += 1
                    r = MagicMock()
                    if call_count2 == 1:
                        r.scalar_one.return_value = mock_db_run
                    elif call_count2 == 2:
                        r.scalar_one_or_none.return_value = mock_failed_version
                    elif call_count2 == 3:
                        r.scalar_one.return_value = mock_db_source
                    return r

                s.execute = exec_fn2
                s.commit = AsyncMock()
                s.refresh = AsyncMock()
            return _mock_session_cm(s)

        monkeypatch.setattr(flows, "get_session", make_session)

        error_msg = "METRICS_GATE_FAILED: twin_node_mapping_incomplete (mapped=5, metrics=10)"
        monkeypatch.setattr(
            flows,
            "sync_github_source",
            AsyncMock(side_effect=RuntimeError(error_msg)),
        )
        monkeypatch.setattr(
            "contextmine_core.twin.set_source_version_status",
            AsyncMock(),
        )
        monkeypatch.setattr(
            "contextmine_core.twin.record_twin_event",
            AsyncMock(),
        )

        await flows.sync_source.fn(source)

        assert mock_db_run.status == SyncRunStatus.FAILED
        assert "METRICS_GATE" in mock_db_run.error

    async def test_scip_gate_error_parsed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SCIP_GATE_FAILED error is parsed."""
        from contextmine_core import SyncRunStatus

        source = _make_source()

        session_idx = 0
        mock_db_run = MagicMock()
        mock_db_source = MagicMock()
        mock_db_source.schedule_interval_minutes = 60

        mock_failed_version = MagicMock()
        mock_failed_version.id = uuid.uuid4()

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
                call_count2 = 0

                async def exec_fn2(stmt):
                    nonlocal call_count2
                    call_count2 += 1
                    r = MagicMock()
                    if call_count2 == 1:
                        r.scalar_one.return_value = mock_db_run
                    elif call_count2 == 2:
                        r.scalar_one_or_none.return_value = mock_failed_version
                    elif call_count2 == 3:
                        r.scalar_one.return_value = mock_db_source
                    return r

                s.execute = exec_fn2
                s.commit = AsyncMock()
                s.refresh = AsyncMock()
            return _mock_session_cm(s)

        monkeypatch.setattr(flows, "get_session", make_session)

        error_msg = "SCIP_GATE_FAILED: (missing=python,typescript) (missing_relations=php)"
        monkeypatch.setattr(
            flows,
            "sync_github_source",
            AsyncMock(side_effect=RuntimeError(error_msg)),
        )
        monkeypatch.setattr(
            "contextmine_core.twin.set_source_version_status",
            AsyncMock(),
        )
        monkeypatch.setattr(
            "contextmine_core.twin.record_twin_event",
            AsyncMock(),
        )

        await flows.sync_source.fn(source)

        assert mock_db_run.status == SyncRunStatus.FAILED


# ---------------------------------------------------------------------------
# sync_single_source: source not found / disabled
# ---------------------------------------------------------------------------


class TestSyncSingleSourceEdgeCases:
    async def test_source_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_session = AsyncMock()
        r = MagicMock()
        r.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=r)

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        result = await flows.sync_single_source.fn(str(uuid.uuid4()))
        assert "error" in result
        assert "not found" in result["error"]

    async def test_source_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        source = _make_source(enabled=False)

        mock_session = AsyncMock()
        r = MagicMock()
        r.scalar_one_or_none.return_value = source
        mock_session.execute = AsyncMock(return_value=r)

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        result = await flows.sync_single_source.fn(str(source.id))
        assert result["skipped"] is True

    async def test_sync_run_none_returns_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        source = _make_source()

        mock_session = AsyncMock()
        r = MagicMock()
        r.scalar_one_or_none.return_value = source
        mock_session.execute = AsyncMock(return_value=r)

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))
        monkeypatch.setattr(flows, "_sync_source_timeout_seconds", lambda: 0)
        monkeypatch.setattr(flows, "sync_source", AsyncMock(return_value=None))

        result = await flows.sync_single_source.fn(str(source.id))
        assert result["skipped"] is True
        assert result["reason"] == "lock_not_acquired"

    async def test_with_timeout_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When timeout > 0, sync_source is wrapped in wait_for."""
        try:
            import sniffio

            if sniffio.current_async_library() != "asyncio":
                pytest.skip("asyncio.wait_for test requires asyncio backend")
        except Exception:
            pass

        source = _make_source()

        mock_session = AsyncMock()
        r = MagicMock()
        r.scalar_one_or_none.return_value = source
        mock_session.execute = AsyncMock(return_value=r)

        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))
        monkeypatch.setattr(flows, "_sync_source_timeout_seconds", lambda: 3600)

        mock_run = MagicMock()
        mock_run.id = uuid.uuid4()
        mock_run.status = MagicMock()
        mock_run.status.value = "success"
        mock_run.stats = {}
        monkeypatch.setattr(flows, "sync_source", AsyncMock(return_value=mock_run))

        result = await flows.sync_single_source.fn(str(source.id))
        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# sync_due_sources
# ---------------------------------------------------------------------------


class TestSyncDueSourcesFlowExtended:
    async def test_no_sources_due(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(flows, "get_due_sources", AsyncMock(return_value=[]))

        result = await flows.sync_due_sources.fn()
        assert result["synced"] == 0
        assert result["skipped"] == 0

    async def test_syncs_sources(self, monkeypatch: pytest.MonkeyPatch) -> None:
        source = _make_source()

        mock_run = MagicMock()
        mock_run.id = uuid.uuid4()
        mock_run.status = MagicMock()
        mock_run.status.value = "success"

        monkeypatch.setattr(flows, "get_due_sources", AsyncMock(return_value=[source]))
        monkeypatch.setattr(flows, "sync_source", AsyncMock(return_value=mock_run))
        monkeypatch.setattr(flows, "_sync_source_timeout_seconds", lambda: 0)

        result = await flows.sync_due_sources.fn()
        assert result["synced"] == 1

    async def test_skips_when_sync_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        source = _make_source()

        monkeypatch.setattr(flows, "get_due_sources", AsyncMock(return_value=[source]))
        monkeypatch.setattr(flows, "sync_source", AsyncMock(return_value=None))
        monkeypatch.setattr(flows, "_sync_source_timeout_seconds", lambda: 0)

        result = await flows.sync_due_sources.fn()
        assert result["skipped"] == 1

    async def test_timeout_recovers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        source = _make_source()

        monkeypatch.setattr(flows, "get_due_sources", AsyncMock(return_value=[source]))
        monkeypatch.setattr(flows, "_sync_source_timeout_seconds", lambda: 1)

        original_wait_for = asyncio.wait_for

        async def mock_wait_for(coro, *, timeout=None):
            raise TimeoutError("timed out")

        monkeypatch.setattr(asyncio, "wait_for", mock_wait_for)
        monkeypatch.setattr(
            flows,
            "_fail_running_sync_runs_for_source",
            AsyncMock(return_value=1),
        )

        result = await flows.sync_due_sources.fn()
        assert result["synced"] == 1
        assert "error" in result["sources"][0]

        monkeypatch.setattr(asyncio, "wait_for", original_wait_for)

    async def test_generic_exception_recorded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        source = _make_source()

        monkeypatch.setattr(flows, "get_due_sources", AsyncMock(return_value=[source]))
        monkeypatch.setattr(
            flows,
            "sync_source",
            AsyncMock(side_effect=RuntimeError("boom")),
        )
        monkeypatch.setattr(flows, "_sync_source_timeout_seconds", lambda: 0)

        result = await flows.sync_due_sources.fn()
        assert result["synced"] == 1
        assert result["sources"][0]["error"] == "boom"

    async def test_with_timeout_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When timeout > 0, sync_source is wrapped in wait_for."""
        source = _make_source()

        mock_run = MagicMock()
        mock_run.id = uuid.uuid4()
        mock_run.status = MagicMock()
        mock_run.status.value = "success"

        monkeypatch.setattr(flows, "get_due_sources", AsyncMock(return_value=[source]))
        monkeypatch.setattr(flows, "sync_source", AsyncMock(return_value=mock_run))
        monkeypatch.setattr(flows, "_sync_source_timeout_seconds", lambda: 3600)

        result = await flows.sync_due_sources.fn()
        assert result["synced"] == 1


# ---------------------------------------------------------------------------
# task_repair_twin_file_path_canonicalization
# ---------------------------------------------------------------------------


class TestRepairTwinFilePathCanonicalization:
    async def test_repair_refreshes_snapshots(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_session = AsyncMock()
        mock_session.commit = AsyncMock()
        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        scenario_id = str(uuid.uuid4())

        monkeypatch.setattr(
            "contextmine_core.twin.repair_twin_file_path_canonicalization",
            AsyncMock(
                return_value={
                    "scenarios_changed": [scenario_id],
                    "keys_renamed": 5,
                }
            ),
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
            collection_id=str(uuid.uuid4()),
        )

        assert result["metric_snapshots_refreshed"] == 3


# ---------------------------------------------------------------------------
# ingest_coverage_metrics: deep pipeline paths
# ---------------------------------------------------------------------------


class TestIngestCoverageMetricsDeepPipeline:
    """Tests covering the full ingest pipeline from lines 3883-4047."""

    async def test_non_github_source_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Web sources are rejected for coverage ingest."""
        job_id = str(uuid.uuid4())

        session_call = 0

        def make_session():
            nonlocal session_call
            session_call += 1
            s = AsyncMock()
            if session_call == 1:
                # Lock phase
                mock_job = MagicMock()
                mock_job.status = "queued"
                r = MagicMock()
                r.scalar_one_or_none.return_value = mock_job
                s.execute = AsyncMock(return_value=r)
                s.commit = AsyncMock()
            elif session_call == 2:
                # Re-read phase
                from contextmine_core import SourceType

                mock_job2 = MagicMock()
                mock_job2.id = uuid.UUID(job_id)
                mock_job2.source_id = uuid.uuid4()
                mock_job2.status = "processing"

                mock_source = MagicMock()
                mock_source.type = SourceType.WEB
                mock_source.id = uuid.uuid4()

                call_idx = 0

                async def exec_fn(stmt):
                    nonlocal call_idx
                    call_idx += 1
                    r2 = MagicMock()
                    if call_idx == 1:
                        r2.scalar_one_or_none.return_value = mock_job2
                    elif call_idx == 2:
                        r2.scalar_one_or_none.return_value = mock_source
                    return r2

                s.execute = exec_fn
                s.commit = AsyncMock()
            else:
                # _fail_coverage_ingest_job session
                mock_fail_job = MagicMock()
                r3 = MagicMock()
                r3.scalar_one_or_none.return_value = mock_fail_job
                s.execute = AsyncMock(return_value=r3)
                s.commit = AsyncMock()
            return _mock_session_cm(s)

        monkeypatch.setattr(flows, "get_session", make_session)

        result = await flows.ingest_coverage_metrics.fn(job_id)
        assert result["status"] == "rejected"

    async def test_sha_mismatch_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When commit SHA doesn't match source cursor, rejected."""
        job_id = str(uuid.uuid4())

        session_call = 0

        def make_session():
            nonlocal session_call
            session_call += 1
            s = AsyncMock()
            if session_call == 1:
                mock_job = MagicMock()
                mock_job.status = "queued"
                r = MagicMock()
                r.scalar_one_or_none.return_value = mock_job
                s.execute = AsyncMock(return_value=r)
                s.commit = AsyncMock()
            elif session_call == 2:
                from contextmine_core import SourceType

                mock_job2 = MagicMock()
                mock_job2.id = uuid.UUID(job_id)
                mock_job2.source_id = uuid.uuid4()
                mock_job2.commit_sha = "abc123"
                mock_job2.status = "processing"

                mock_source = MagicMock()
                mock_source.type = SourceType.GITHUB
                mock_source.id = uuid.uuid4()
                mock_source.cursor = "different_sha"

                call_idx = 0

                async def exec_fn(stmt):
                    nonlocal call_idx
                    call_idx += 1
                    r2 = MagicMock()
                    if call_idx == 1:
                        r2.scalar_one_or_none.return_value = mock_job2
                    elif call_idx == 2:
                        r2.scalar_one_or_none.return_value = mock_source
                    return r2

                s.execute = exec_fn
                s.commit = AsyncMock()
            else:
                mock_fail_job = MagicMock()
                r3 = MagicMock()
                r3.scalar_one_or_none.return_value = mock_fail_job
                s.execute = AsyncMock(return_value=r3)
                s.commit = AsyncMock()
            return _mock_session_cm(s)

        monkeypatch.setattr(flows, "get_session", make_session)

        result = await flows.ingest_coverage_metrics.fn(job_id)
        assert result["status"] == "rejected"
        assert result["error_code"] == "INGEST_SHA_MISMATCH"

    async def test_no_relevant_files_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When no file nodes with metrics found, fails."""
        job_id = str(uuid.uuid4())

        session_call = 0

        def make_session():
            nonlocal session_call
            session_call += 1
            s = AsyncMock()
            if session_call == 1:
                mock_job = MagicMock()
                mock_job.status = "queued"
                r = MagicMock()
                r.scalar_one_or_none.return_value = mock_job
                s.execute = AsyncMock(return_value=r)
                s.commit = AsyncMock()
            elif session_call == 2:
                from contextmine_core import SourceType

                mock_job2 = MagicMock()
                mock_job2.id = uuid.UUID(job_id)
                mock_job2.source_id = uuid.uuid4()
                mock_job2.commit_sha = "abc123"
                mock_job2.status = "processing"

                mock_source = MagicMock()
                mock_source.type = SourceType.GITHUB
                mock_source.id = uuid.uuid4()
                mock_source.cursor = "abc123"
                mock_source.collection_id = uuid.uuid4()

                mock_scenario = MagicMock()
                mock_scenario.id = uuid.uuid4()

                call_idx = 0

                async def exec_fn(stmt):
                    nonlocal call_idx
                    call_idx += 1
                    r2 = MagicMock()
                    if call_idx == 1:
                        r2.scalar_one_or_none.return_value = mock_job2
                    elif call_idx == 2:
                        r2.scalar_one_or_none.return_value = mock_source
                    elif call_idx == 3:
                        # file_nodes query returns empty
                        r2.scalars.return_value.all.return_value = []
                    return r2

                s.execute = exec_fn
                s.commit = AsyncMock()
                s.flush = AsyncMock()

                monkeypatch.setattr(
                    "contextmine_core.twin.get_or_create_as_is_scenario",
                    AsyncMock(return_value=mock_scenario),
                )
            else:
                mock_fail_job = MagicMock()
                r3 = MagicMock()
                r3.scalar_one_or_none.return_value = mock_fail_job
                s.execute = AsyncMock(return_value=r3)
                s.commit = AsyncMock()
            return _mock_session_cm(s)

        monkeypatch.setattr(flows, "get_session", make_session)

        result = await flows.ingest_coverage_metrics.fn(job_id)
        assert result["error_code"] == "INGEST_NO_RELEVANT_FILES"

    async def test_no_reports_fails(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """When no report files attached to job, fails."""
        job_id = str(uuid.uuid4())
        source_id = uuid.uuid4()

        # Pre-set monkeypatches that will be needed across sessions
        mock_scenario = MagicMock()
        mock_scenario.id = uuid.uuid4()
        monkeypatch.setattr(
            "contextmine_core.twin.get_or_create_as_is_scenario",
            AsyncMock(return_value=mock_scenario),
        )
        monkeypatch.setattr(
            "contextmine_core.pathing.canonicalize_repo_relative_path",
            lambda p: p,
        )
        monkeypatch.setattr(
            "contextmine_worker.github_sync.get_repo_path",
            lambda sid: tmp_path,
        )

        session_call = 0

        def make_session():
            nonlocal session_call
            session_call += 1
            s = AsyncMock()
            if session_call == 1:
                mock_job = MagicMock()
                mock_job.status = "queued"
                r = MagicMock()
                r.scalar_one_or_none.return_value = mock_job
                s.execute = AsyncMock(return_value=r)
                s.commit = AsyncMock()
            elif session_call == 2:
                from contextmine_core import SourceType

                mock_job2 = MagicMock()
                mock_job2.id = uuid.UUID(job_id)
                mock_job2.source_id = source_id
                mock_job2.commit_sha = "abc123"
                mock_job2.status = "processing"

                mock_source = MagicMock()
                mock_source.type = SourceType.GITHUB
                mock_source.id = source_id
                mock_source.cursor = "abc123"
                mock_source.collection_id = uuid.uuid4()

                mock_node = MagicMock()
                mock_node.kind = "file"
                mock_node.natural_key = "file:src/main.py"
                mock_node.meta = {
                    "source_id": str(source_id),
                    "metrics_structural_ready": True,
                }

                call_idx = 0

                async def exec_fn(stmt):
                    nonlocal call_idx
                    call_idx += 1
                    r2 = MagicMock()
                    if call_idx == 1:
                        r2.scalar_one_or_none.return_value = mock_job2
                    elif call_idx == 2:
                        r2.scalar_one_or_none.return_value = mock_source
                    elif call_idx == 3:
                        r2.scalars.return_value.all.return_value = [mock_node]
                    elif call_idx == 4:
                        r2.scalars.return_value.all.return_value = []
                    return r2

                s.execute = exec_fn
                s.commit = AsyncMock()
                s.flush = AsyncMock()
            else:
                mock_fail_job = MagicMock()
                r3 = MagicMock()
                r3.scalar_one_or_none.return_value = mock_fail_job
                s.execute = AsyncMock(return_value=r3)
                s.commit = AsyncMock()
            return _mock_session_cm(s)

        monkeypatch.setattr(flows, "get_session", make_session)

        result = await flows.ingest_coverage_metrics.fn(job_id)
        assert result["error_code"] == "INGEST_PARSE_FAILED"

    async def test_payload_too_large_rejected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When payload exceeds size limit, rejected."""
        job_id = str(uuid.uuid4())
        source_id = uuid.uuid4()

        mock_scenario = MagicMock()
        mock_scenario.id = uuid.uuid4()
        monkeypatch.setattr(
            "contextmine_core.twin.get_or_create_as_is_scenario",
            AsyncMock(return_value=mock_scenario),
        )
        monkeypatch.setattr(
            "contextmine_core.pathing.canonicalize_repo_relative_path",
            lambda p: p,
        )
        monkeypatch.setattr(
            "contextmine_worker.github_sync.get_repo_path",
            lambda sid: tmp_path,
        )
        monkeypatch.setattr(
            flows,
            "get_settings",
            lambda: _make_settings(coverage_ingest_max_payload_mb=1),
        )

        session_call = 0

        def make_session():
            nonlocal session_call
            session_call += 1
            s = AsyncMock()
            if session_call == 1:
                mock_job = MagicMock()
                mock_job.status = "queued"
                r = MagicMock()
                r.scalar_one_or_none.return_value = mock_job
                s.execute = AsyncMock(return_value=r)
                s.commit = AsyncMock()
            elif session_call == 2:
                from contextmine_core import SourceType

                mock_job2 = MagicMock()
                mock_job2.id = uuid.UUID(job_id)
                mock_job2.source_id = source_id
                mock_job2.commit_sha = "abc123"
                mock_job2.status = "processing"

                mock_source = MagicMock()
                mock_source.type = SourceType.GITHUB
                mock_source.id = source_id
                mock_source.cursor = "abc123"
                mock_source.collection_id = uuid.uuid4()

                mock_node = MagicMock()
                mock_node.kind = "file"
                mock_node.natural_key = "file:src/main.py"
                mock_node.meta = {
                    "source_id": str(source_id),
                    "metrics_structural_ready": True,
                }

                # Report with payload exceeding 1MB limit
                mock_report = MagicMock()
                mock_report.report_bytes = b"x" * (2 * 1024 * 1024)

                call_idx = 0

                async def exec_fn(stmt):
                    nonlocal call_idx
                    call_idx += 1
                    r2 = MagicMock()
                    if call_idx == 1:
                        r2.scalar_one_or_none.return_value = mock_job2
                    elif call_idx == 2:
                        r2.scalar_one_or_none.return_value = mock_source
                    elif call_idx == 3:
                        r2.scalars.return_value.all.return_value = [mock_node]
                    elif call_idx == 4:
                        r2.scalars.return_value.all.return_value = [mock_report]
                    return r2

                s.execute = exec_fn
                s.commit = AsyncMock()
                s.flush = AsyncMock()
            else:
                mock_fail_job = MagicMock()
                r3 = MagicMock()
                r3.scalar_one_or_none.return_value = mock_fail_job
                s.execute = AsyncMock(return_value=r3)
                s.commit = AsyncMock()
            return _mock_session_cm(s)

        monkeypatch.setattr(flows, "get_session", make_session)

        result = await flows.ingest_coverage_metrics.fn(job_id)
        assert result["error_code"] == "INGEST_PAYLOAD_TOO_LARGE"


# ---------------------------------------------------------------------------
# build_twin_graph: arch_docs_enabled path
# ---------------------------------------------------------------------------


class TestBuildTwinGraphArchDocs:
    async def test_arch_docs_enabled_but_no_sync_generation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When arch_docs_enabled=True but generate_on_sync=False, skips."""
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


# ---------------------------------------------------------------------------
# sync_source: empty error message path
# ---------------------------------------------------------------------------


class TestSyncSourceEmptyError:
    async def test_empty_error_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When exception has empty str(), uses repr() instead."""
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
                    return r

                s.execute = exec_fn2
                s.commit = AsyncMock()
                s.refresh = AsyncMock()
            return _mock_session_cm(s)

        monkeypatch.setattr(flows, "get_session", make_session)

        # Exception with empty string message
        monkeypatch.setattr(
            flows,
            "sync_github_source",
            AsyncMock(side_effect=RuntimeError("")),
        )
        monkeypatch.setattr(
            "contextmine_core.twin.set_source_version_status",
            AsyncMock(),
        )
        monkeypatch.setattr(
            "contextmine_core.twin.record_twin_event",
            AsyncMock(),
        )

        await flows.sync_source.fn(source)

        assert mock_db_run.status == SyncRunStatus.FAILED
        # Empty string falls back to repr()
        assert mock_db_run.error == repr(RuntimeError(""))


# ---------------------------------------------------------------------------
# repair_twin_file_paths flow
# ---------------------------------------------------------------------------


class TestRepairTwinFilePathsFlow:
    async def test_delegates_to_task(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Flow delegates to task function."""
        expected_stats = {"keys_renamed": 5, "metric_snapshots_refreshed": 3}
        monkeypatch.setattr(
            flows,
            "task_repair_twin_file_path_canonicalization",
            AsyncMock(return_value=expected_stats),
        )

        result = await flows.repair_twin_file_paths.fn()
        assert result == expected_stats
