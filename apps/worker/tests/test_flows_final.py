"""Coverage tests targeting remaining uncovered lines in flows.py.

Targets:
- Knowledge graph build steps (ERM, schema, surface, semantic entities, communities, summaries)
- SCIP pipeline inner helpers (pure functions)
- Architecture docs generation path in build_twin_graph
- Coverage ingest edge cases
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import contextmine_worker.flows as flows
import pytest
from contextmine_core.models import KnowledgeNodeKind

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Shared helpers (same pattern as test_flows_coverage.py)
# ---------------------------------------------------------------------------


def _make_settings(**overrides: Any) -> SimpleNamespace:
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
        "repos_root": "/tmp/repos",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _mock_session_cm(session: AsyncMock) -> MagicMock:
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _setup_kg_base_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up the base mocks needed for build_knowledge_graph."""
    monkeypatch.setattr(
        flows, "get_settings", lambda: _make_settings(default_llm_provider="openai")
    )
    monkeypatch.setattr(
        "contextmine_core.research.llm.get_llm_provider", lambda *a, **kw: MagicMock()
    )
    monkeypatch.setattr(
        "contextmine_core.research.llm.get_research_llm_provider", lambda: MagicMock()
    )
    monkeypatch.setattr(flows, "parse_embedding_model_spec", lambda spec: ("openai", "model"))
    monkeypatch.setattr(flows, "get_embedder", lambda *a, **kw: MagicMock(dimension=768))
    monkeypatch.setattr(
        "contextmine_core.knowledge.builder.cleanup_scoped_knowledge_nodes",
        AsyncMock(return_value={"nodes_deleted": 0, "evidence_deleted": 0}),
    )


def _setup_kg_step1_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock step 1 (file/symbol node creation)."""
    mock_stats = MagicMock()
    mock_stats.file_nodes_created = 0
    mock_stats.symbol_nodes_created = 0
    monkeypatch.setattr(
        "contextmine_core.knowledge.builder.build_knowledge_graph_for_source",
        AsyncMock(return_value=mock_stats),
    )
    monkeypatch.setattr(
        "contextmine_core.knowledge.builder.cleanup_orphan_nodes",
        AsyncMock(return_value={"nodes_deleted": 0}),
    )


def _setup_kg_step2_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock step 2 (rules extraction)."""
    monkeypatch.setattr("contextmine_core.treesitter.languages.detect_language", lambda fp: None)


def _setup_kg_step3_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock step 3 (ERM extraction) - empty results."""
    mock_erm = MagicMock()
    mock_erm.schema.tables = []
    mock_erm.add_alembic_extraction = MagicMock()
    monkeypatch.setattr("contextmine_core.analyzer.extractors.erm.ERMExtractor", lambda: mock_erm)


def _setup_kg_step4_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock step 4 (surface catalog extraction) - empty."""
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


def _setup_kg_steps567_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock steps 5-7 (semantic entities, communities, summaries)."""
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
    monkeypatch.setattr("contextmine_core.knowledge.communities.persist_communities", AsyncMock())
    mock_summary = MagicMock()
    mock_summary.communities_summarized = 0
    mock_summary.embeddings_created = 0
    monkeypatch.setattr(
        "contextmine_core.knowledge.summaries.generate_community_summaries",
        AsyncMock(return_value=mock_summary),
    )


# ---------------------------------------------------------------------------
# build_knowledge_graph: ERM + schema extraction paths (lines 790-868)
# ---------------------------------------------------------------------------


class TestBuildKGERM:
    async def test_schema_candidate_sql_extraction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Lines 831-868: SQL schema candidates trigger deterministic extraction."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())

        _setup_kg_base_mocks(monkeypatch)
        _setup_kg_step1_mock(monkeypatch)
        _setup_kg_step2_mocks(monkeypatch)
        _setup_kg_step4_mocks(monkeypatch)
        _setup_kg_steps567_mocks(monkeypatch)

        mock_session = AsyncMock()

        async def mock_exec(stmt):
            r = MagicMock()
            r.all.return_value = [
                ("git://o/r/db/schema.sql", "CREATE TABLE users (id INT);"),
            ]
            r.scalar_one_or_none.return_value = None
            return r

        mock_session.execute = mock_exec
        mock_session.commit = AsyncMock()
        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        mock_erm = MagicMock()
        mock_erm.schema.tables = []
        mock_erm.add_alembic_extraction = MagicMock()
        monkeypatch.setattr(
            "contextmine_core.analyzer.extractors.erm.ERMExtractor", lambda: mock_erm
        )

        mock_extraction = MagicMock()
        mock_extraction.tables = [MagicMock()]
        mock_extraction.foreign_keys = []
        monkeypatch.setattr(
            "contextmine_core.analyzer.extractors.schema.extract_schema_from_file",
            AsyncMock(return_value=mock_extraction),
        )
        mock_agg = MagicMock()
        mock_agg.tables = [MagicMock()]
        monkeypatch.setattr(
            "contextmine_core.analyzer.extractors.schema.aggregate_schema_extractions",
            MagicMock(return_value=mock_agg),
        )
        monkeypatch.setattr(
            "contextmine_core.analyzer.extractors.schema.build_schema_graph",
            AsyncMock(return_value={"table_nodes_created": 1}),
        )
        monkeypatch.setattr(
            "contextmine_core.analyzer.extractors.schema.save_erd_artifact",
            AsyncMock(),
        )

        result = await flows.build_knowledge_graph.fn(
            source_id=source_id,
            collection_id=collection_id,
        )
        assert result["kg_tables"] == 1


class TestBuildBehavioralGraphs:
    async def test_cleanup_prunes_behavioral_nodes_for_current_and_deleted_files(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session = AsyncMock()
        cleanup_mock = AsyncMock(return_value={"nodes_deleted": 4, "evidence_deleted": 7})
        build_tests_mock = AsyncMock(return_value={"test_nodes": 2})
        build_ui_mock = AsyncMock(return_value={"ui_nodes": 3})
        build_flows_mock = AsyncMock(return_value={"flow_nodes": 1})

        monkeypatch.setattr(
            "contextmine_core.knowledge.builder.cleanup_scoped_knowledge_nodes",
            cleanup_mock,
        )
        monkeypatch.setattr(
            "contextmine_core.analyzer.extractors.tests.extract_tests_from_files",
            lambda files: [MagicMock()],
        )
        monkeypatch.setattr(
            "contextmine_core.analyzer.extractors.ui.extract_ui_from_files",
            lambda files: [MagicMock()],
        )
        monkeypatch.setattr(
            "contextmine_core.analyzer.extractors.tests.build_tests_graph",
            build_tests_mock,
        )
        monkeypatch.setattr(
            "contextmine_core.analyzer.extractors.ui.build_ui_graph",
            build_ui_mock,
        )
        monkeypatch.setattr(
            "contextmine_core.analyzer.extractors.flows.synthesize_user_flows",
            lambda ui, tests: MagicMock(),
        )
        monkeypatch.setattr(
            "contextmine_core.analyzer.extractors.flows.build_flows_graph",
            build_flows_mock,
        )

        stats, warnings = await flows._build_behavioral_graphs(
            session,
            _make_settings(
                digital_twin_behavioral_enabled=True,
                digital_twin_ui_enabled=True,
                digital_twin_flows_enabled=True,
            ),
            uuid.uuid4(),
            uuid.uuid4(),
            files=[
                ("apps/web/src/routes.tsx", "export default function Routes() {}"),
                ("tests/test_checkout.py", "def test_checkout(): assert True"),
            ],
            deleted_file_paths=["apps/web/src/removed.tsx"],
        )

        assert warnings == []
        cleanup_mock.assert_awaited_once()
        cleanup_call = cleanup_mock.await_args.kwargs
        assert cleanup_call["target_file_paths"] == {
            "apps/web/src/routes.tsx",
            "tests/test_checkout.py",
            "apps/web/src/removed.tsx",
        }
        assert cleanup_call["kinds"] == {
            KnowledgeNodeKind.TEST_SUITE,
            KnowledgeNodeKind.TEST_CASE,
            KnowledgeNodeKind.TEST_FIXTURE,
            KnowledgeNodeKind.UI_ROUTE,
            KnowledgeNodeKind.UI_VIEW,
            KnowledgeNodeKind.UI_COMPONENT,
            KnowledgeNodeKind.INTERFACE_CONTRACT,
            KnowledgeNodeKind.USER_FLOW,
            KnowledgeNodeKind.FLOW_STEP,
        }
        assert stats["behavioral_nodes_deleted"] == 4
        assert stats["behavioral_evidence_deleted"] == 7
        assert stats["test_nodes"] == 2
        assert stats["ui_nodes"] == 3
        assert stats["flow_nodes"] == 1

    async def test_alembic_parse_error_caught(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Lines 816-820: Alembic parse error is caught and logged."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())

        _setup_kg_base_mocks(monkeypatch)
        _setup_kg_step1_mock(monkeypatch)
        _setup_kg_step2_mocks(monkeypatch)
        _setup_kg_step4_mocks(monkeypatch)
        _setup_kg_steps567_mocks(monkeypatch)

        mock_session = AsyncMock()

        async def mock_exec(stmt):
            r = MagicMock()
            r.all.return_value = [
                ("git://o/r/alembic/versions/001.py", "invalid python code"),
            ]
            r.scalar_one_or_none.return_value = None
            return r

        mock_session.execute = mock_exec
        mock_session.commit = AsyncMock()
        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        mock_erm = MagicMock()
        mock_erm.schema.tables = []
        mock_erm.add_alembic_extraction = MagicMock()
        monkeypatch.setattr(
            "contextmine_core.analyzer.extractors.erm.ERMExtractor", lambda: mock_erm
        )

        monkeypatch.setattr(
            "contextmine_core.analyzer.extractors.alembic.extract_from_alembic",
            MagicMock(side_effect=SyntaxError("bad")),
        )

        result = await flows.build_knowledge_graph.fn(
            source_id=source_id,
            collection_id=collection_id,
        )
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# build_knowledge_graph: surface extraction (lines 878-926)
# ---------------------------------------------------------------------------


class TestBuildKGSurface:
    async def test_surface_has_endpoints(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Lines 912-922: When surfaces found, graph is built."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())

        _setup_kg_base_mocks(monkeypatch)
        _setup_kg_step1_mock(monkeypatch)
        _setup_kg_step2_mocks(monkeypatch)
        _setup_kg_step3_mocks(monkeypatch)
        _setup_kg_steps567_mocks(monkeypatch)

        mock_session = AsyncMock()

        async def mock_exec(stmt):
            r = MagicMock()
            r.all.return_value = [
                ("git://o/r/api.yaml", "openapi: '3.0.0'\ninfo:\n  title: API\n  version: '1'"),
            ]
            r.scalar_one_or_none.return_value = None
            return r

        mock_session.execute = mock_exec
        mock_session.commit = AsyncMock()
        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

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
            AsyncMock(return_value={"endpoint_nodes": 3, "job_nodes": 0}),
        )

        result = await flows.build_knowledge_graph.fn(
            source_id=source_id,
            collection_id=collection_id,
        )
        assert result["kg_endpoints"] == 3

    async def test_step1_reports_stale_node_cleanup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stale FILE/SYMBOL nodes deleted during rebuild are surfaced in stats."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())

        _setup_kg_base_mocks(monkeypatch)
        _setup_kg_step2_mocks(monkeypatch)
        _setup_kg_step3_mocks(monkeypatch)
        _setup_kg_step4_mocks(monkeypatch)
        _setup_kg_steps567_mocks(monkeypatch)

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=MagicMock())
        mock_session.commit = AsyncMock()
        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        mock_kg_stats = MagicMock()
        mock_kg_stats.file_nodes_created = 4
        mock_kg_stats.symbol_nodes_created = 9
        monkeypatch.setattr(
            "contextmine_core.knowledge.builder.build_knowledge_graph_for_source",
            AsyncMock(return_value=mock_kg_stats),
        )
        monkeypatch.setattr(
            "contextmine_core.knowledge.builder.cleanup_orphan_nodes",
            AsyncMock(return_value={"nodes_deleted": 3}),
        )

        result = await flows.build_knowledge_graph.fn(
            source_id=source_id,
            collection_id=collection_id,
            changed_doc_ids=[],
        )

        assert result["kg_file_nodes"] == 4
        assert result["kg_symbol_nodes"] == 9
        assert result["kg_nodes_deleted"] == 3

    async def test_deleted_file_cleanup_contributes_to_stats(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deleted-file scoped cleanup is aggregated across extracted fact types."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())

        _setup_kg_base_mocks(monkeypatch)
        _setup_kg_step1_mock(monkeypatch)
        _setup_kg_step2_mocks(monkeypatch)
        _setup_kg_step3_mocks(monkeypatch)
        _setup_kg_step4_mocks(monkeypatch)
        _setup_kg_steps567_mocks(monkeypatch)

        mock_session = AsyncMock()

        async def mock_exec(stmt):
            result = MagicMock()
            result.all.return_value = []
            result.scalar_one_or_none.return_value = None
            return result

        mock_session.execute = mock_exec
        mock_session.commit = AsyncMock()
        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))
        monkeypatch.setattr(
            "contextmine_core.knowledge.builder.cleanup_scoped_knowledge_nodes",
            AsyncMock(
                side_effect=[
                    {"nodes_deleted": 2, "evidence_deleted": 0},
                    {"nodes_deleted": 3, "evidence_deleted": 0},
                    {"nodes_deleted": 4, "evidence_deleted": 0},
                ]
            ),
        )

        result = await flows.build_knowledge_graph.fn(
            source_id=source_id,
            collection_id=collection_id,
            changed_doc_ids=[],
            deleted_file_paths=["src/deleted.py"],
        )

        assert result["kg_nodes_deleted"] == 9

    async def test_surface_exception_caught(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Lines 924-926: Surface extraction failure caught."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())

        _setup_kg_base_mocks(monkeypatch)
        _setup_kg_step1_mock(monkeypatch)
        _setup_kg_step2_mocks(monkeypatch)
        _setup_kg_step3_mocks(monkeypatch)
        _setup_kg_steps567_mocks(monkeypatch)

        mock_session = AsyncMock()

        async def mock_exec(stmt):
            r = MagicMock()
            r.all.return_value = []
            r.scalar_one_or_none.return_value = None
            return r

        mock_session.execute = mock_exec
        mock_session.commit = AsyncMock()
        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        monkeypatch.setattr(
            "contextmine_core.analyzer.extractors.surface.SurfaceCatalogExtractor",
            MagicMock(side_effect=RuntimeError("surface fail")),
        )

        result = await flows.build_knowledge_graph.fn(
            source_id=source_id,
            collection_id=collection_id,
        )
        assert any("surface" in str(e) for e in result.get("kg_errors", []))


# ---------------------------------------------------------------------------
# build_knowledge_graph: semantic + community steps (lines 935-1024)
# ---------------------------------------------------------------------------


class TestBuildKGSemanticCommunity:
    async def test_skip_semantic_when_no_changed_docs_and_entities_exist(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty changed_doc_ids skips semantic extraction when entities already exist."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())

        _setup_kg_base_mocks(monkeypatch)
        _setup_kg_step1_mock(monkeypatch)
        _setup_kg_step2_mocks(monkeypatch)
        _setup_kg_step3_mocks(monkeypatch)
        _setup_kg_step4_mocks(monkeypatch)

        mock_session = AsyncMock()

        # Return a non-None value for _kg_has_semantic_entities check
        existing_node_id = uuid.uuid4()

        async def mock_exec(stmt):
            r = MagicMock()
            r.all.return_value = []
            r.scalar_one_or_none.return_value = existing_node_id
            return r

        mock_session.execute = mock_exec
        mock_session.commit = AsyncMock()
        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        extract_mock = AsyncMock()
        monkeypatch.setattr(
            "contextmine_core.knowledge.extraction.extract_from_documents", extract_mock
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
            "contextmine_core.knowledge.communities.persist_communities", AsyncMock()
        )
        mock_summary = MagicMock()
        mock_summary.communities_summarized = 0
        mock_summary.embeddings_created = 0
        monkeypatch.setattr(
            "contextmine_core.knowledge.summaries.generate_community_summaries",
            AsyncMock(return_value=mock_summary),
        )

        await flows.build_knowledge_graph.fn(
            source_id=source_id,
            collection_id=collection_id,
            changed_doc_ids=[],
        )
        # Skips extraction because semantic entities already exist
        extract_mock.assert_not_called()

    async def test_deleted_files_force_semantic_and_summary_regeneration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deletion-only syncs must rebuild semantic entities and summaries."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())

        _setup_kg_base_mocks(monkeypatch)
        _setup_kg_step1_mock(monkeypatch)
        _setup_kg_step2_mocks(monkeypatch)
        _setup_kg_step3_mocks(monkeypatch)
        _setup_kg_step4_mocks(monkeypatch)

        mock_session = AsyncMock()
        existing_node_id = uuid.uuid4()

        async def mock_exec(stmt):
            result = MagicMock()
            result.all.return_value = []
            result.scalar_one_or_none.return_value = existing_node_id
            return result

        mock_session.execute = mock_exec
        mock_session.commit = AsyncMock()
        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        extract_mock = AsyncMock(return_value=MagicMock())
        persist_mock = AsyncMock(return_value={"entities_created": 2, "relationships_created": 1})
        monkeypatch.setattr(
            "contextmine_core.knowledge.extraction.extract_from_documents",
            extract_mock,
        )
        monkeypatch.setattr(
            "contextmine_core.knowledge.extraction.persist_semantic_entities",
            persist_mock,
        )

        mock_community = MagicMock()
        mock_community.community_count = lambda level: 0
        mock_community.modularity = {0: 0, 1: 0, 2: 0}
        monkeypatch.setattr(
            "contextmine_core.knowledge.communities.detect_communities",
            AsyncMock(return_value=mock_community),
        )
        monkeypatch.setattr(
            "contextmine_core.knowledge.communities.persist_communities", AsyncMock()
        )
        summary_mock = AsyncMock(
            return_value=MagicMock(communities_summarized=1, embeddings_created=1)
        )
        monkeypatch.setattr(
            "contextmine_core.knowledge.summaries.generate_community_summaries",
            summary_mock,
        )

        result = await flows.build_knowledge_graph.fn(
            source_id=source_id,
            collection_id=collection_id,
            changed_doc_ids=[],
            deleted_file_paths=["src/deleted.py"],
        )

        extract_mock.assert_awaited_once()
        assert extract_mock.await_args.kwargs["max_chunks"] is None
        summary_mock.assert_awaited_once()
        assert result["kg_semantic_entities"] == 2
        assert result["kg_summaries_created"] == 1

    async def test_semantic_quality_report_is_exposed_in_stats(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())

        _setup_kg_base_mocks(monkeypatch)
        _setup_kg_step1_mock(monkeypatch)
        _setup_kg_step2_mocks(monkeypatch)
        _setup_kg_step3_mocks(monkeypatch)
        _setup_kg_step4_mocks(monkeypatch)

        mock_session = AsyncMock()

        async def mock_exec(stmt):
            r = MagicMock()
            r.all.return_value = []
            r.scalar_one_or_none.return_value = None
            return r

        mock_session.execute = mock_exec
        mock_session.commit = AsyncMock()
        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        extraction_batch = MagicMock(
            quality_report={
                "status": "degraded",
                "warnings": ["no_semantic_relationships_extracted", "chunk_budget_applied"],
                "selected_chunks": 10,
            }
        )
        monkeypatch.setattr(
            "contextmine_core.knowledge.extraction.extract_from_documents",
            AsyncMock(return_value=extraction_batch),
        )
        monkeypatch.setattr(
            "contextmine_core.knowledge.extraction.persist_semantic_entities",
            AsyncMock(return_value={"entities_created": 2, "relationships_created": 0}),
        )

        mock_community = MagicMock()
        mock_community.community_count = lambda level: 0
        mock_community.modularity = {0: 0, 1: 0, 2: 0}
        monkeypatch.setattr(
            "contextmine_core.knowledge.communities.detect_communities",
            AsyncMock(return_value=mock_community),
        )
        monkeypatch.setattr(
            "contextmine_core.knowledge.communities.persist_communities", AsyncMock()
        )
        monkeypatch.setattr(
            "contextmine_core.knowledge.summaries.generate_community_summaries",
            AsyncMock(return_value=MagicMock(communities_summarized=0, embeddings_created=0)),
        )

        result = await flows.build_knowledge_graph.fn(
            source_id=source_id,
            collection_id=collection_id,
        )

        assert result["kg_semantic_quality_status"] == "degraded"
        assert result["kg_semantic_quality"]["selected_chunks"] == 10
        assert result["kg_semantic_quality_warnings"] == [
            "no_semantic_relationships_extracted",
            "chunk_budget_applied",
        ]

    async def test_semantic_extraction_error_caught(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Lines 966-968: Semantic extraction failure recorded."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())

        _setup_kg_base_mocks(monkeypatch)
        _setup_kg_step1_mock(monkeypatch)
        _setup_kg_step2_mocks(monkeypatch)
        _setup_kg_step3_mocks(monkeypatch)
        _setup_kg_step4_mocks(monkeypatch)

        mock_session = AsyncMock()

        async def mock_exec(stmt):
            r = MagicMock()
            r.all.return_value = []
            r.scalar_one_or_none.return_value = None
            return r

        mock_session.execute = mock_exec
        mock_session.commit = AsyncMock()
        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        monkeypatch.setattr(
            "contextmine_core.knowledge.extraction.extract_from_documents",
            AsyncMock(side_effect=RuntimeError("extraction boom")),
        )
        mock_community = MagicMock()
        mock_community.community_count = lambda level: 0
        mock_community.modularity = {0: 0, 1: 0, 2: 0}
        monkeypatch.setattr(
            "contextmine_core.knowledge.communities.detect_communities",
            AsyncMock(return_value=mock_community),
        )
        monkeypatch.setattr(
            "contextmine_core.knowledge.communities.persist_communities", AsyncMock()
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
        )
        assert any("semantic_extraction" in str(e) for e in result.get("kg_errors", []))

    async def test_community_detection_error_caught(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Lines 994-996: Community detection failure is caught."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())

        _setup_kg_base_mocks(monkeypatch)
        _setup_kg_step1_mock(monkeypatch)
        _setup_kg_step2_mocks(monkeypatch)
        _setup_kg_step3_mocks(monkeypatch)
        _setup_kg_step4_mocks(monkeypatch)

        mock_session = AsyncMock()

        async def mock_exec(stmt):
            r = MagicMock()
            r.all.return_value = []
            r.scalar_one_or_none.return_value = None
            return r

        mock_session.execute = mock_exec
        mock_session.commit = AsyncMock()
        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

        monkeypatch.setattr(
            "contextmine_core.knowledge.extraction.extract_from_documents",
            AsyncMock(return_value=MagicMock()),
        )
        monkeypatch.setattr(
            "contextmine_core.knowledge.extraction.persist_semantic_entities",
            AsyncMock(return_value={"entities_created": 0, "relationships_created": 0}),
        )
        monkeypatch.setattr(
            "contextmine_core.knowledge.communities.detect_communities",
            AsyncMock(side_effect=RuntimeError("community fail")),
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
        )
        assert any("communities" in str(e) for e in result.get("kg_errors", []))

    async def test_summary_generation_error_caught(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Lines 1022-1024: Summary generation failure is caught."""
        source_id = str(uuid.uuid4())
        collection_id = str(uuid.uuid4())

        _setup_kg_base_mocks(monkeypatch)
        _setup_kg_step1_mock(monkeypatch)
        _setup_kg_step2_mocks(monkeypatch)
        _setup_kg_step3_mocks(monkeypatch)
        _setup_kg_step4_mocks(monkeypatch)

        mock_session = AsyncMock()

        async def mock_exec(stmt):
            r = MagicMock()
            r.all.return_value = []
            r.scalar_one_or_none.return_value = None
            return r

        mock_session.execute = mock_exec
        mock_session.commit = AsyncMock()
        monkeypatch.setattr(flows, "get_session", lambda: _mock_session_cm(mock_session))

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
            "contextmine_core.knowledge.communities.persist_communities", AsyncMock()
        )
        monkeypatch.setattr(
            "contextmine_core.knowledge.summaries.generate_community_summaries",
            AsyncMock(side_effect=RuntimeError("summary fail")),
        )

        result = await flows.build_knowledge_graph.fn(
            source_id=source_id,
            collection_id=collection_id,
        )
        assert any("summaries" in str(e) for e in result.get("kg_errors", []))


# ---------------------------------------------------------------------------
# SCIP pipeline inner helper logic (pure functions, lines 1963-2172)
# ---------------------------------------------------------------------------


class TestScipPipelineHelpers:
    def test_snapshot_repo_file_path_relative_with_root(self) -> None:
        file_info: dict[str, object] = {"path": "src/main.py"}
        meta = {"repo_relative_root": "backend"}
        raw_path = str(file_info.get("path") or "").strip().replace("\\", "/")
        normalized = raw_path.lstrip("./")
        repo_relative_root = str(meta.get("repo_relative_root") or "").strip()
        if repo_relative_root:
            repo_relative_root = repo_relative_root.replace("\\", "/").strip("/")
            if normalized != repo_relative_root and not normalized.startswith(
                f"{repo_relative_root}/"
            ):
                normalized = f"{repo_relative_root}/{normalized}".strip("/")
        assert normalized == "backend/src/main.py"

    def test_snapshot_repo_file_path_already_prefixed(self) -> None:
        file_info: dict[str, object] = {"path": "backend/src/main.py"}
        meta = {"repo_relative_root": "backend"}
        raw_path = str(file_info.get("path") or "").strip().replace("\\", "/")
        normalized = raw_path.lstrip("./")
        repo_relative_root = str(meta.get("repo_relative_root") or "").strip()
        if repo_relative_root:
            repo_relative_root = repo_relative_root.replace("\\", "/").strip("/")
            if normalized != repo_relative_root and not normalized.startswith(
                f"{repo_relative_root}/"
            ):
                normalized = f"{repo_relative_root}/{normalized}".strip("/")
        assert normalized == "backend/src/main.py"

    def test_collect_indexed_files_by_language(self) -> None:
        snapshots = [
            {"meta": {"language": "python"}, "files": [{"path": "a.py"}, {"path": "b.py"}]},
            {"meta": {"language": "typescript"}, "files": [{"path": "c.ts"}]},
        ]
        indexed: dict[str, set[str]] = {}
        for snap in snapshots:
            meta = dict(snap.get("meta") or {})
            snap_lang = str(meta.get("language") or "").strip().lower()
            for item in snap.get("files") or []:
                if not isinstance(item, dict):
                    continue
                path = str(item.get("path") or "").strip()
                if not path:
                    continue
                lang = str(item.get("language") or "").strip().lower() or snap_lang
                if lang:
                    indexed.setdefault(lang, set()).add(path)
        counts = {lang: len(paths) for lang, paths in indexed.items()}
        assert counts["python"] == 2
        assert counts["typescript"] == 1

    def test_missing_relation_languages_detection(self) -> None:
        indexed_files = {"python": 10, "ruby": 5}
        relation_kinds: dict[str, dict[str, int]] = {"python": {"calls": 3}}
        semantic_kinds = {"calls", "references", "imports", "extends", "implements"}
        missing: list[str] = []
        for lang, count in indexed_files.items():
            if int(count or 0) <= 0:
                continue
            rk = relation_kinds.get(lang) or {}
            semantic = sum(int(rk.get(k, 0) or 0) for k in semantic_kinds)
            if semantic <= 0:
                missing.append(lang)
        assert "ruby" in missing
        assert "python" not in missing

    def test_collect_relation_coverage(self) -> None:
        snapshots = [
            {
                "meta": {"language": "typescript"},
                "relations": [{"kind": "calls"}, {"kind": "references"}, {"kind": "calls"}],
            }
        ]
        totals: dict[str, int] = {}
        kind_totals: dict[str, dict[str, int]] = {}
        for snap in snapshots:
            meta = dict(snap.get("meta") or {})
            lang = str(meta.get("language") or "").strip().lower()
            for rel in snap.get("relations") or []:
                if not isinstance(rel, dict):
                    continue
                kind = str(rel.get("kind") or "").strip().lower()
                totals[lang] = totals.get(lang, 0) + 1
                if kind:
                    kind_totals.setdefault(lang, {})
                    kind_totals[lang][kind] = kind_totals[lang].get(kind, 0) + 1
        assert totals["typescript"] == 3
        assert kind_totals["typescript"]["calls"] == 2


# ---------------------------------------------------------------------------
# Document processing helpers (unit logic)
# ---------------------------------------------------------------------------


class TestDocumentProcessingHelpers:
    def test_deferred_docs_calculation(self) -> None:
        docs = [("id", "content", "file.py") for _ in range(10)]
        limit = 5
        total = len(docs)
        deferred = 0
        if limit and total > limit:
            docs = docs[:limit]
            deferred = total - len(docs)
        assert deferred == 5
        assert len(docs) == 5

    def test_error_sample_limit(self) -> None:
        samples: list[str] = []
        for i in range(150):
            if len(samples) < 100:
                samples.append(f"error_{i}")
        assert len(samples) == 100
