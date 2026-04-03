"""Async unit tests for knowledge/builder.py.

Tests the async database-dependent functions with mocked sessions:
- build_knowledge_graph_for_source() with mocked session
- cleanup_orphan_nodes() with mocked session
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from contextmine_core.knowledge.builder import (
    build_knowledge_graph_for_source,
    cleanup_orphan_nodes,
    cleanup_scoped_knowledge_nodes,
)
from contextmine_core.models import KnowledgeNodeKind, SymbolEdgeType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_session() -> AsyncMock:
    return AsyncMock()


def _make_mock_document(
    doc_id=None,
    source_id=None,
    uri="src/main.py",
    title="main.py",
    meta=None,
) -> MagicMock:
    doc = MagicMock()
    doc.id = doc_id or uuid4()
    doc.source_id = source_id or uuid4()
    doc.uri = uri
    doc.title = title
    doc.meta = meta or {"file_path": uri}
    return doc


def _make_mock_symbol(
    symbol_id=None,
    document_id=None,
    name="my_func",
    qualified_name="my_func",
    kind_value="function",
    start_line=1,
    end_line=10,
    signature="def my_func():",
    parent_name=None,
) -> MagicMock:
    sym = MagicMock()
    sym.id = symbol_id or uuid4()
    sym.document_id = document_id or uuid4()
    sym.name = name
    sym.qualified_name = qualified_name
    sym.kind = MagicMock()
    sym.kind.value = kind_value
    sym.start_line = start_line
    sym.end_line = end_line
    sym.signature = signature
    sym.parent_name = parent_name
    return sym


def _make_mock_symbol_edge(
    source_symbol_id,
    target_symbol_id,
    edge_type=SymbolEdgeType.CALLS,
    source_line=5,
) -> MagicMock:
    edge = MagicMock()
    edge.source_symbol_id = source_symbol_id
    edge.target_symbol_id = target_symbol_id
    edge.edge_type = edge_type
    edge.source_line = source_line
    return edge


# ===========================================================================
# build_knowledge_graph_for_source tests
# ===========================================================================


class TestBuildKnowledgeGraphForSource:
    """Tests for build_knowledge_graph_for_source with mocked session."""

    @pytest.mark.anyio
    async def test_source_not_found_returns_empty_stats(self) -> None:
        session = _mock_session()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute.return_value = mock_result

        stats = await build_knowledge_graph_for_source(session, uuid4())
        assert stats.file_nodes_created == 0
        assert stats.symbol_nodes_created == 0
        assert stats.edges_created == 0

    @pytest.mark.anyio
    async def test_creates_file_nodes(self) -> None:
        session = _mock_session()
        source_id = uuid4()
        coll_id = uuid4()
        node_id = uuid4()

        doc = _make_mock_document(source_id=source_id)

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                # Get collection_id
                result.scalar_one_or_none.return_value = coll_id
            elif call_count[0] == 1:
                # Get documents
                result.scalars.return_value = MagicMock(all=lambda: [doc])
            elif call_count[0] == 2:
                # Get symbols for doc (empty)
                result.scalars.return_value = MagicMock(all=lambda: [])
            else:
                # SymbolEdge query (empty)
                result.scalars.return_value = MagicMock(all=lambda: [])
            call_count[0] += 1
            return result

        session.execute.side_effect = mock_execute

        with (
            patch(
                "contextmine_core.knowledge.builder.upsert_node",
                new_callable=AsyncMock,
                return_value=node_id,
            ) as mock_upsert_node,
            patch(
                "contextmine_core.knowledge.builder.upsert_edge",
                new_callable=AsyncMock,
                return_value=uuid4(),
            ),
            patch(
                "contextmine_core.knowledge.builder.create_node_evidence",
                new_callable=AsyncMock,
                return_value=str(uuid4()),
            ),
        ):
            stats = await build_knowledge_graph_for_source(session, source_id)
            assert stats.file_nodes_created == 1
            mock_upsert_node.assert_called()

    @pytest.mark.anyio
    async def test_creates_symbol_nodes_and_edges(self) -> None:
        session = _mock_session()
        source_id = uuid4()
        coll_id = uuid4()
        file_node_id = uuid4()
        sym_node_id = uuid4()

        doc = _make_mock_document(source_id=source_id, uri="src/module.py")
        sym = _make_mock_symbol(
            document_id=doc.id,
            name="process",
            qualified_name="process",
        )

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                result.scalar_one_or_none.return_value = coll_id
            elif call_count[0] == 1:
                result.scalars.return_value = MagicMock(all=lambda: [doc])
            elif call_count[0] == 2:
                # Symbols for doc
                result.scalars.return_value = MagicMock(all=lambda: [sym])
            elif call_count[0] == 3:
                # SymbolEdge query (empty)
                result.scalars.return_value = MagicMock(all=lambda: [])
            else:
                result.scalars.return_value = MagicMock(all=lambda: [])
            call_count[0] += 1
            return result

        session.execute.side_effect = mock_execute

        upsert_call_count = [0]

        async def mock_upsert_node(_session_arg, **kwargs):
            upsert_call_count[0] += 1
            if upsert_call_count[0] == 1:
                return file_node_id
            return sym_node_id

        with (
            patch(
                "contextmine_core.knowledge.builder.upsert_node",
                side_effect=mock_upsert_node,
            ),
            patch(
                "contextmine_core.knowledge.builder.upsert_edge",
                new_callable=AsyncMock,
                return_value=uuid4(),
            ) as mock_upsert_edge,
            patch(
                "contextmine_core.knowledge.builder.create_node_evidence",
                new_callable=AsyncMock,
                return_value=str(uuid4()),
            ) as mock_create_evidence,
        ):
            stats = await build_knowledge_graph_for_source(session, source_id)
            assert stats.file_nodes_created == 1
            assert stats.symbol_nodes_created == 1
            assert stats.edges_created >= 1  # FILE_DEFINES_SYMBOL
            assert stats.evidence_created == 1
            mock_upsert_edge.assert_called()
            mock_create_evidence.assert_called()

    @pytest.mark.anyio
    async def test_creates_symbol_contains_symbol_edges(self) -> None:
        """Parent-child symbol relationships create SYMBOL_CONTAINS_SYMBOL edges."""
        session = _mock_session()
        source_id = uuid4()
        coll_id = uuid4()
        file_node_id = uuid4()

        doc = _make_mock_document(source_id=source_id)

        parent_sym = _make_mock_symbol(
            name="MyClass",
            qualified_name="MyClass",
            kind_value="class",
            start_line=1,
            end_line=20,
        )
        child_sym = _make_mock_symbol(
            name="my_method",
            qualified_name="MyClass.my_method",
            kind_value="method",
            start_line=5,
            end_line=15,
            parent_name="MyClass",
        )

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                result.scalar_one_or_none.return_value = coll_id
            elif call_count[0] == 1:
                result.scalars.return_value = MagicMock(all=lambda: [doc])
            elif call_count[0] == 2:
                result.scalars.return_value = MagicMock(all=lambda: [parent_sym, child_sym])
            else:
                result.scalars.return_value = MagicMock(all=lambda: [])
            call_count[0] += 1
            return result

        session.execute.side_effect = mock_execute

        parent_node_id = uuid4()
        child_node_id = uuid4()
        upsert_call_count = [0]

        async def mock_upsert_node(_session_arg, **kwargs):
            upsert_call_count[0] += 1
            if upsert_call_count[0] == 1:
                return file_node_id
            elif upsert_call_count[0] == 2:
                return parent_node_id
            return child_node_id

        with (
            patch(
                "contextmine_core.knowledge.builder.upsert_node",
                side_effect=mock_upsert_node,
            ),
            patch(
                "contextmine_core.knowledge.builder.upsert_edge",
                new_callable=AsyncMock,
                return_value=uuid4(),
            ) as _mock_upsert_edge,
            patch(
                "contextmine_core.knowledge.builder.create_node_evidence",
                new_callable=AsyncMock,
                return_value=str(uuid4()),
            ),
        ):
            stats = await build_knowledge_graph_for_source(session, source_id)
            # FILE_DEFINES_SYMBOL x2 + SYMBOL_CONTAINS_SYMBOL x1
            assert stats.edges_created >= 3

    @pytest.mark.anyio
    async def test_creates_symbol_edge_from_symbol_edge_table(self) -> None:
        """SymbolEdge CALLS/REFERENCES/IMPORTS get mapped to KG edges."""
        session = _mock_session()
        source_id = uuid4()
        coll_id = uuid4()
        file_node_id = uuid4()

        doc = _make_mock_document(source_id=source_id)
        sym1 = _make_mock_symbol(name="caller", qualified_name="caller")
        sym2 = _make_mock_symbol(name="callee", qualified_name="callee")
        sym_edge = _make_mock_symbol_edge(
            source_symbol_id=sym1.id,
            target_symbol_id=sym2.id,
            edge_type=SymbolEdgeType.CALLS,
        )

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                result.scalar_one_or_none.return_value = coll_id
            elif call_count[0] == 1:
                result.scalars.return_value = MagicMock(all=lambda: [doc])
            elif call_count[0] == 2:
                result.scalars.return_value = MagicMock(all=lambda: [sym1, sym2])
            elif call_count[0] == 3:
                # _build_symbol_to_file_node_map query (Symbol.id for doc)
                result.all.return_value = [(sym1.id,), (sym2.id,)]
            elif call_count[0] == 4:
                # SymbolEdge query
                result.scalars.return_value = MagicMock(all=lambda: [sym_edge])
            else:
                result.scalars.return_value = MagicMock(all=lambda: [])
            call_count[0] += 1
            return result

        session.execute.side_effect = mock_execute

        sym1_node_id = uuid4()
        sym2_node_id = uuid4()
        upsert_call_count = [0]

        async def mock_upsert_node(_session_arg, **kwargs):
            upsert_call_count[0] += 1
            if upsert_call_count[0] == 1:
                return file_node_id
            elif upsert_call_count[0] == 2:
                return sym1_node_id
            return sym2_node_id

        with (
            patch(
                "contextmine_core.knowledge.builder.upsert_node",
                side_effect=mock_upsert_node,
            ),
            patch(
                "contextmine_core.knowledge.builder.upsert_edge",
                new_callable=AsyncMock,
                return_value=uuid4(),
            ) as _mock_upsert_edge,
            patch(
                "contextmine_core.knowledge.builder.create_node_evidence",
                new_callable=AsyncMock,
                return_value=str(uuid4()),
            ),
        ):
            stats = await build_knowledge_graph_for_source(session, source_id)
            # FILE_DEFINES_SYMBOL x2 + SYMBOL_CALLS_SYMBOL x1
            assert stats.edges_created >= 3


# ===========================================================================
# cleanup_orphan_nodes tests
# ===========================================================================


class TestCleanupOrphanNodes:
    """Tests for cleanup_orphan_nodes with mocked session."""

    @pytest.mark.anyio
    async def test_no_orphans_nothing_deleted(self) -> None:
        session = _mock_session()
        coll_id = uuid4()
        source_id = uuid4()

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                # Get current document URIs
                result.all.return_value = [("src/main.py",)]
            elif call_count[0] == 1:
                # Get file nodes
                nid = uuid4()
                result.all.return_value = [(nid, "src/main.py")]
            call_count[0] += 1
            return result

        session.execute.side_effect = mock_execute

        stats = await cleanup_orphan_nodes(session, coll_id, source_id)
        assert stats["nodes_deleted"] == 0

    @pytest.mark.anyio
    async def test_orphan_nodes_are_deleted(self) -> None:
        session = _mock_session()
        coll_id = uuid4()
        source_id = uuid4()
        orphan_id = uuid4()

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                # Current document URIs (file was deleted)
                result.all.return_value = [("src/main.py",)]
            elif call_count[0] == 1:
                # File nodes (includes an orphan)
                result.all.return_value = [
                    (uuid4(), "src/main.py"),
                    (orphan_id, "src/deleted.py"),
                ]
            else:
                result.all.return_value = []
            call_count[0] += 1
            return result

        session.execute.side_effect = mock_execute

        stats = await cleanup_orphan_nodes(session, coll_id, source_id)
        assert stats["nodes_deleted"] == 1

    @pytest.mark.anyio
    async def test_no_documents_all_nodes_orphaned(self) -> None:
        session = _mock_session()
        coll_id = uuid4()
        source_id = uuid4()

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                # No documents
                result.all.return_value = []
            elif call_count[0] == 1:
                # Two file nodes -> both orphaned
                result.all.return_value = [
                    (uuid4(), "src/a.py"),
                    (uuid4(), "src/b.py"),
                ]
            call_count[0] += 1
            return result

        session.execute.side_effect = mock_execute

        stats = await cleanup_orphan_nodes(session, coll_id, source_id)
        assert stats["nodes_deleted"] == 2

    @pytest.mark.anyio
    async def test_no_file_nodes_nothing_deleted(self) -> None:
        session = _mock_session()
        coll_id = uuid4()
        source_id = uuid4()

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                result.all.return_value = [("src/main.py",)]
            elif call_count[0] == 1:
                # No file nodes
                result.all.return_value = []
            call_count[0] += 1
            return result

        session.execute.side_effect = mock_execute

        stats = await cleanup_orphan_nodes(session, coll_id, source_id)
        assert stats["nodes_deleted"] == 0


class TestCleanupScopedKnowledgeNodes:
    """Tests for cleanup_scoped_knowledge_nodes with mocked session."""

    @pytest.mark.anyio
    async def test_deletes_matching_nodes_and_orphan_evidence(self) -> None:
        session = _mock_session()
        node_id = uuid4()

        select_result = MagicMock()
        select_result.all.return_value = [
            (node_id, {"source_files": ["src/schema.sql"]}, None),
            (uuid4(), {"file_path": "src/other.sql"}, None),
        ]
        delete_result = MagicMock()
        session.execute.side_effect = [select_result, delete_result]

        with patch(
            "contextmine_core.knowledge.builder.cleanup_orphan_evidence",
            new_callable=AsyncMock,
            return_value=2,
        ) as cleanup_evidence_mock:
            stats = await cleanup_scoped_knowledge_nodes(
                session,
                uuid4(),
                kinds={KnowledgeNodeKind.DB_TABLE},
                target_file_paths={"src/schema.sql"},
            )

        assert stats == {"nodes_deleted": 1, "evidence_deleted": 2}
        cleanup_evidence_mock.assert_awaited_once_with(session)

    @pytest.mark.anyio
    async def test_skips_when_no_target_paths_match(self) -> None:
        session = _mock_session()

        select_result = MagicMock()
        select_result.all.return_value = [
            (uuid4(), {"file_path": "src/other.sql"}, None),
        ]
        session.execute.return_value = select_result

        stats = await cleanup_scoped_knowledge_nodes(
            session,
            uuid4(),
            kinds={KnowledgeNodeKind.DB_TABLE},
            target_file_paths={"src/schema.sql"},
        )

        assert stats == {"nodes_deleted": 0, "evidence_deleted": 0}
