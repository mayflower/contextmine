"""Async unit tests for traceability.py.

Tests the async resolver functions with mocked sessions:
- SymbolTraceResolver._ensure_knowledge_symbols() with mocked session
- SymbolTraceResolver._ensure_scip_graph() with mocked session
- SymbolTraceResolver.resolve_many() with mocked session (name_fallback path)
- resolve_symbol_refs_for_calls() with mocked session
- build_endpoint_symbol_index() with mocked session
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from contextmine_core.analyzer.extractors.traceability import (
    SymbolTraceResolver,
    build_endpoint_symbol_index,
    resolve_symbol_refs_for_calls,
)
from contextmine_core.models import KnowledgeNodeKind

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_session() -> AsyncMock:
    return AsyncMock()


def _make_kg_node_row(
    node_id=None,
    name="my_func",
    natural_key="symbol:my_func",
    meta=None,
):
    """Create a mock KnowledgeNode as returned by session queries."""
    node = MagicMock()
    node.id = node_id or uuid4()
    node.name = name
    node.natural_key = natural_key
    node.kind = KnowledgeNodeKind.SYMBOL
    node.meta = meta or {
        "file_path": "src/main.py",
        "start_line": 10,
        "end_line": 20,
        "def_id": None,
    }
    return node


def _make_endpoint_node(
    node_id=None,
    name="POST /api/users",
    natural_key="endpoint:POST:/api/users",
    meta=None,
):
    """Create a mock KnowledgeNode for API_ENDPOINT."""
    node = MagicMock()
    node.id = node_id or uuid4()
    node.name = name
    node.natural_key = natural_key
    node.kind = KnowledgeNodeKind.API_ENDPOINT
    node.meta = meta or {}
    return node


# ===========================================================================
# SymbolTraceResolver._ensure_knowledge_symbols tests
# ===========================================================================


class TestEnsureKnowledgeSymbols:
    """Tests for loading knowledge symbols into resolver caches."""

    @pytest.mark.anyio
    async def test_loads_symbols_into_caches(self) -> None:
        session = _mock_session()
        coll_id = uuid4()
        nid = uuid4()

        node = _make_kg_node_row(
            node_id=nid,
            name="process_data",
            natural_key="symbol:process_data",
            meta={"file_path": "src/main.py", "start_line": 10, "end_line": 20},
        )

        mock_result = MagicMock()
        mock_result.scalars.return_value = MagicMock(all=lambda: [node])
        session.execute.return_value = mock_result

        resolver = SymbolTraceResolver(session=session, collection_id=coll_id)
        await resolver._ensure_knowledge_symbols()

        assert resolver._kg_loaded is True
        assert nid in resolver._kg_by_node_id
        assert "process_data" in resolver._kg_by_name
        assert "src/main.py" in resolver._kg_by_file

    @pytest.mark.anyio
    async def test_only_loads_once(self) -> None:
        session = _mock_session()

        mock_result = MagicMock()
        mock_result.scalars.return_value = MagicMock(all=lambda: [])
        session.execute.return_value = mock_result

        resolver = SymbolTraceResolver(session=session, collection_id=uuid4())
        await resolver._ensure_knowledge_symbols()
        await resolver._ensure_knowledge_symbols()

        session.execute.assert_called_once()

    @pytest.mark.anyio
    async def test_def_id_indexed(self) -> None:
        session = _mock_session()
        nid = uuid4()

        node = _make_kg_node_row(
            node_id=nid,
            name="func",
            meta={
                "file_path": "src/a.py",
                "start_line": 1,
                "end_line": 5,
                "def_id": "scip:my_func_def_123",
            },
        )

        mock_result = MagicMock()
        mock_result.scalars.return_value = MagicMock(all=lambda: [node])
        session.execute.return_value = mock_result

        resolver = SymbolTraceResolver(session=session, collection_id=uuid4())
        await resolver._ensure_knowledge_symbols()

        assert "scip:my_func_def_123" in resolver._kg_by_def_id


# ===========================================================================
# SymbolTraceResolver._ensure_scip_graph tests
# ===========================================================================


class TestEnsureScipGraph:
    """Tests for loading SCIP graph into resolver caches."""

    @pytest.mark.anyio
    async def test_no_scenario_leaves_empty(self) -> None:
        session = _mock_session()

        # No scenario found
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute.return_value = mock_result

        resolver = SymbolTraceResolver(session=session, collection_id=uuid4())
        await resolver._ensure_scip_graph()

        assert resolver._scip_loaded is True
        assert resolver._twin_by_id == {}

    @pytest.mark.anyio
    async def test_loads_twin_nodes_and_edges(self) -> None:
        session = _mock_session()
        scenario_id = uuid4()
        coll_id = uuid4()

        scenario = MagicMock()
        scenario.id = scenario_id

        twin_nid1 = uuid4()
        twin_nid2 = uuid4()

        twin_node1 = MagicMock()
        twin_node1.id = twin_nid1
        twin_node1.natural_key = "symbol:func1"
        twin_node1.name = "func1"
        twin_node1.kind = "function"
        twin_node1.meta = {
            "file_path": "src/a.py",
            "range": {"start_line": 1, "end_line": 10, "start_col": 0, "end_col": 0},
            "def_id": "def1",
            "symbol_kind": "function",
        }

        twin_node2 = MagicMock()
        twin_node2.id = twin_nid2
        twin_node2.natural_key = "symbol:func2"
        twin_node2.name = "func2"
        twin_node2.kind = "function"
        twin_node2.meta = {
            "file_path": "src/b.py",
            "range": {"start_line": 5, "end_line": 15, "start_col": 0, "end_col": 0},
        }

        call_edge = MagicMock()
        call_edge.source_node_id = twin_nid1
        call_edge.target_node_id = twin_nid2
        call_edge.kind = "symbol_calls_symbol"

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                result.scalar_one_or_none.return_value = scenario
            elif call_count[0] == 1:
                result.scalars.return_value = MagicMock(all=lambda: [twin_node1, twin_node2])
            elif call_count[0] == 2:
                result.scalars.return_value = MagicMock(all=lambda: [call_edge])
            call_count[0] += 1
            return result

        session.execute.side_effect = mock_execute

        resolver = SymbolTraceResolver(session=session, collection_id=coll_id)
        await resolver._ensure_scip_graph()

        assert resolver._scip_loaded is True
        assert twin_nid1 in resolver._twin_by_id
        assert twin_nid2 in resolver._twin_by_id
        assert "src/a.py" in resolver._twin_by_file
        assert twin_nid2 in resolver._twin_calls.get(twin_nid1, [])

    @pytest.mark.anyio
    async def test_only_loads_once(self) -> None:
        session = _mock_session()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute.return_value = mock_result

        resolver = SymbolTraceResolver(session=session, collection_id=uuid4())
        await resolver._ensure_scip_graph()
        await resolver._ensure_scip_graph()

        session.execute.assert_called_once()


# ===========================================================================
# SymbolTraceResolver.resolve_many tests (name_fallback)
# ===========================================================================


class TestResolveMany:
    """Tests for resolve_many focusing on fallback path."""

    @pytest.mark.anyio
    async def test_no_call_sites_uses_fallback_hints(self) -> None:
        session = _mock_session()
        coll_id = uuid4()
        nid = uuid4()

        node = _make_kg_node_row(
            node_id=nid,
            name="process_data",
            natural_key="symbol:process_data",
            meta={"file_path": "src/main.py", "start_line": 10, "end_line": 20},
        )

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                # _ensure_knowledge_symbols
                result.scalars.return_value = MagicMock(all=lambda: [node])
            elif call_count[0] == 1:
                # _ensure_scip_graph (no scenario)
                result.scalar_one_or_none.return_value = None
            call_count[0] += 1
            return result

        session.execute.side_effect = mock_execute

        resolver = SymbolTraceResolver(session=session, collection_id=coll_id)

        refs = await resolver.resolve_many(
            call_sites=[],
            fallback_symbol_hints=["process_data"],
        )

        assert len(refs) >= 1
        assert refs[0].engine == "name_fallback"
        assert refs[0].confidence == 0.45
        assert refs[0].symbol_node_id == nid

    @pytest.mark.anyio
    async def test_empty_everything_returns_empty(self) -> None:
        session = _mock_session()

        mock_result = MagicMock()
        mock_result.scalars.return_value = MagicMock(all=lambda: [])

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                result.scalars.return_value = MagicMock(all=lambda: [])
            elif call_count[0] == 1:
                result.scalar_one_or_none.return_value = None
            call_count[0] += 1
            return result

        session.execute.side_effect = mock_execute

        resolver = SymbolTraceResolver(session=session, collection_id=uuid4())
        refs = await resolver.resolve_many(
            call_sites=[],
            fallback_symbol_hints=[],
        )
        assert refs == []

    @pytest.mark.anyio
    async def test_results_sorted_by_confidence_desc(self) -> None:
        session = _mock_session()
        coll_id = uuid4()
        nid1 = uuid4()
        nid2 = uuid4()

        node1 = _make_kg_node_row(
            node_id=nid1,
            name="alpha",
            meta={"file_path": "a.py", "start_line": 1, "end_line": 5},
        )
        node2 = _make_kg_node_row(
            node_id=nid2,
            name="beta",
            meta={"file_path": "b.py", "start_line": 1, "end_line": 5},
        )

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                result.scalars.return_value = MagicMock(all=lambda: [node1, node2])
            elif call_count[0] == 1:
                result.scalar_one_or_none.return_value = None
            call_count[0] += 1
            return result

        session.execute.side_effect = mock_execute

        resolver = SymbolTraceResolver(session=session, collection_id=coll_id)
        refs = await resolver.resolve_many(
            call_sites=[],
            fallback_symbol_hints=["alpha", "beta"],
        )

        assert len(refs) == 2
        # Same confidence => sorted by name then natural_key
        assert refs[0].symbol_name <= refs[1].symbol_name


# ===========================================================================
# resolve_symbol_refs_for_calls additional tests
# ===========================================================================


class TestResolveSymbolRefsForCallsAsync:
    """Additional async tests for resolve_symbol_refs_for_calls."""

    @pytest.mark.anyio
    async def test_empty_call_sites_list(self) -> None:
        session = _mock_session()
        with patch.object(
            SymbolTraceResolver, "resolve_many", new_callable=AsyncMock
        ) as mock_resolve:
            mock_resolve.return_value = []
            result = await resolve_symbol_refs_for_calls(
                session=session,
                collection_id=uuid4(),
                source_id=None,
                file_path="test.py",
                call_sites=[],
                fallback_symbol_hints=["hint"],
            )
            assert result == []
            # resolve_many still called with empty call_sites
            mock_resolve.assert_called_once()
            assert mock_resolve.call_args.kwargs["call_sites"] == []

    @pytest.mark.anyio
    async def test_passes_fallback_hints_through(self) -> None:
        session = _mock_session()
        with patch.object(
            SymbolTraceResolver, "resolve_many", new_callable=AsyncMock
        ) as mock_resolve:
            mock_resolve.return_value = []
            await resolve_symbol_refs_for_calls(
                session=session,
                collection_id=uuid4(),
                source_id=None,
                file_path="test.py",
                call_sites=[],
                fallback_symbol_hints=["process_data", "handle_request"],
            )
            hints = mock_resolve.call_args.kwargs["fallback_symbol_hints"]
            assert hints == ["process_data", "handle_request"]


# ===========================================================================
# build_endpoint_symbol_index tests
# ===========================================================================


class TestBuildEndpointSymbolIndex:
    """Tests for build_endpoint_symbol_index with mocked session."""

    @pytest.mark.anyio
    async def test_no_endpoints_returns_empty(self) -> None:
        session = _mock_session()

        mock_result = MagicMock()
        mock_result.scalars.return_value = MagicMock(all=lambda: [])
        session.execute.return_value = mock_result

        index = await build_endpoint_symbol_index(session=session, collection_id=uuid4())
        assert index == {}

    @pytest.mark.anyio
    async def test_endpoint_with_operation_id(self) -> None:
        session = _mock_session()
        coll_id = uuid4()
        endpoint_id = uuid4()

        endpoint = _make_endpoint_node(
            node_id=endpoint_id,
            meta={"operation_id": "createUser"},
        )

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                result.scalars.return_value = MagicMock(all=lambda: [endpoint])
            else:
                result.scalars.return_value = MagicMock(all=lambda: [])
            call_count[0] += 1
            return result

        session.execute.side_effect = mock_execute

        index = await build_endpoint_symbol_index(session=session, collection_id=coll_id)
        # "createuser" should be a token variant
        assert any(endpoint_id in ids for ids in index.values())
        assert "createuser" in index or "create_user" in index

    @pytest.mark.anyio
    async def test_endpoint_with_handler_symbol_names(self) -> None:
        session = _mock_session()
        coll_id = uuid4()
        endpoint_id = uuid4()

        endpoint = _make_endpoint_node(
            node_id=endpoint_id,
            meta={
                "handler_symbol_names": ["handleLogin", "processAuth"],
            },
        )

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                result.scalars.return_value = MagicMock(all=lambda: [endpoint])
            else:
                result.scalars.return_value = MagicMock(all=lambda: [])
            call_count[0] += 1
            return result

        session.execute.side_effect = mock_execute

        index = await build_endpoint_symbol_index(session=session, collection_id=coll_id)
        # Should have tokens from handler_symbol_names
        assert any(endpoint_id in ids for ids in index.values())

    @pytest.mark.anyio
    async def test_endpoint_with_handler_symbol_node_ids(self) -> None:
        session = _mock_session()
        coll_id = uuid4()
        endpoint_id = uuid4()
        sym_nid = uuid4()

        endpoint = _make_endpoint_node(
            node_id=endpoint_id,
            meta={
                "handler_symbol_node_ids": [str(sym_nid)],
            },
        )

        sym_node = _make_kg_node_row(
            node_id=sym_nid,
            name="create_user",
            natural_key="symbol:create_user",
        )

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                # Endpoint query
                result.scalars.return_value = MagicMock(all=lambda: [endpoint])
            elif call_count[0] == 1:
                # Symbol lookup
                result.scalars.return_value = MagicMock(all=lambda: [sym_node])
            call_count[0] += 1
            return result

        session.execute.side_effect = mock_execute

        index = await build_endpoint_symbol_index(session=session, collection_id=coll_id)
        assert any(endpoint_id in ids for ids in index.values())

    @pytest.mark.anyio
    async def test_endpoint_with_no_meta_keys(self) -> None:
        """Endpoint with empty meta still appears, but contributes no tokens."""
        session = _mock_session()
        coll_id = uuid4()
        endpoint_id = uuid4()

        endpoint = _make_endpoint_node(
            node_id=endpoint_id,
            meta={},
        )

        mock_result = MagicMock()
        mock_result.scalars.return_value = MagicMock(all=lambda: [endpoint])
        session.execute.return_value = mock_result

        index = await build_endpoint_symbol_index(session=session, collection_id=coll_id)
        # No tokens -> empty index
        assert not any(endpoint_id in ids for ids in index.values())

    @pytest.mark.anyio
    async def test_invalid_uuid_in_handler_ids_skipped(self) -> None:
        session = _mock_session()
        coll_id = uuid4()
        endpoint_id = uuid4()

        endpoint = _make_endpoint_node(
            node_id=endpoint_id,
            meta={
                "handler_symbol_node_ids": ["not-a-uuid", "also-bad"],
                "operation_id": "myOp",
            },
        )

        mock_result = MagicMock()
        mock_result.scalars.return_value = MagicMock(all=lambda: [endpoint])
        session.execute.return_value = mock_result

        index = await build_endpoint_symbol_index(session=session, collection_id=coll_id)
        # Should still have token from operation_id even though UUIDs were bad
        assert any(endpoint_id in ids for ids in index.values())
