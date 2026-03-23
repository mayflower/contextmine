"""Tests for MCP server tool implementations.

Tests the async tool functions in mcp_server.py by directly calling them
with mocked database sessions and external dependencies.

NOTE: Functions decorated with @mcp.tool() are wrapped in FunctionTool objects.
To call the underlying async function, use the `.fn` attribute (e.g. tool.fn()).
"""

import json
import uuid
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import app.mcp_server as mcp_mod
import pytest

# Convenience aliases for tool functions (unwrap FunctionTool -> coroutine)
_list_collections = mcp_mod.list_collections.fn
_list_documents = mcp_mod.list_documents.fn
_get_context_markdown = mcp_mod.get_context_markdown.fn
_code_outline = mcp_mod.code_outline.fn
_code_find_symbol = mcp_mod.code_find_symbol.fn
_code_definition = mcp_mod.code_definition.fn
_code_references = mcp_mod.code_references.fn
_code_expand = mcp_mod.code_expand.fn
_code_deep_research = mcp_mod.code_deep_research.fn
_graph_neighborhood = mcp_mod.mcp_graph_neighborhood.fn
_trace_path = mcp_mod.mcp_trace_path.fn
_graph_rag = mcp_mod.mcp_graph_rag.fn
_research_validation = mcp_mod.research_validation.fn
_research_data_model = mcp_mod.research_data_model.fn
_research_architecture = mcp_mod.research_architecture.fn
_get_arc42 = mcp_mod.mcp_get_arc42.fn
_arc42_drift = mcp_mod.mcp_arc42_drift_report.fn
_store_findings = mcp_mod.mcp_store_findings.fn
_export_twin_view = mcp_mod.mcp_export_twin_view.fn
_get_twin_graph = mcp_mod.mcp_get_twin_graph.fn
_get_twin_status = mcp_mod.mcp_get_twin_status.fn
_refresh_twin = mcp_mod.mcp_refresh_twin.fn
_validation_dashboard = mcp_mod.mcp_get_validation_dashboard.fn
_query_twin_cypher = mcp_mod.mcp_query_twin_cypher.fn
_create_intent = mcp_mod.mcp_create_architecture_intent.fn
_approve_intent = mcp_mod.mcp_approve_architecture_intent.fn
_list_ports_adapters = mcp_mod.mcp_list_ports_adapters.fn
_list_methods = mcp_mod.mcp_list_methods.fn
_list_calls = mcp_mod.mcp_list_calls.fn
_get_cfg = mcp_mod.mcp_get_cfg.fn
_get_variable_flow = mcp_mod.mcp_get_variable_flow.fn
_get_codebase_summary = mcp_mod.mcp_get_codebase_summary.fn
_find_taint_sources = mcp_mod.mcp_find_taint_sources.fn
_find_taint_sinks = mcp_mod.mcp_find_taint_sinks.fn
_find_taint_flows = mcp_mod.mcp_find_taint_flows.fn
_export_sarif = mcp_mod.mcp_export_sarif.fn
_get_twin_timeline = mcp_mod.mcp_get_twin_timeline.fn

# Direct utility function references (not decorated, callable directly)
escape_like_pattern = mcp_mod.escape_like_pattern
_node_kind_in_scope = mcp_mod._node_kind_in_scope
_filter_graph_payload = mcp_mod._filter_graph_payload
_parse_csv_list = mcp_mod._parse_csv_list
_sha256_text = mcp_mod._sha256_text
_resolve_collection_access = mcp_mod._resolve_collection_access
_resolve_collection_for_tool = mcp_mod._resolve_collection_for_tool
get_context_markdown_sync = mcp_mod.get_context_markdown_sync


# ---------------------------------------------------------------------------
# Helper: mock async context manager for get_db_session
# ---------------------------------------------------------------------------


def _mock_db_session(mock_db):
    """Create a patched get_db_session that yields mock_db."""
    ctx = patch("app.mcp_server.get_db_session")
    mock_session = ctx.start()
    mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db)
    mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
    return ctx


def _make_symbol(
    *,
    name: str = "foo",
    kind_value: str = "function",
    start_line: int = 4,
    end_line: int = 5,
    signature: str = "def foo()",
    parent_name: str | None = None,
    qualified_name: str | None = None,
    meta: dict | None = None,
) -> MagicMock:
    sym = MagicMock()
    sym.id = uuid.uuid4()
    sym.name = name
    sym.kind = MagicMock(value=kind_value)
    sym.start_line = start_line
    sym.end_line = end_line
    sym.signature = signature
    sym.parent_name = parent_name
    sym.qualified_name = qualified_name or name
    sym.meta = meta or {}
    sym.document_id = uuid.uuid4()
    return sym


def _make_document(
    *,
    uri: str = "git://repo/src/main.py",
    content_markdown: str = "line1\nline2\nline3\ndef foo():\n    pass\nline6",
) -> MagicMock:
    doc = MagicMock()
    doc.id = uuid.uuid4()
    doc.uri = uri
    doc.content_markdown = content_markdown
    return doc


# ---------------------------------------------------------------------------
# Fixtures for patching auth
# ---------------------------------------------------------------------------


@pytest.fixture
def _patch_user_id():
    """Patch get_current_user_id to return None (unauthenticated)."""
    with patch("app.mcp_server.get_current_user_id", return_value=None):
        yield


@pytest.fixture
def _patch_user_id_with_uuid():
    """Patch get_current_user_id to return a real UUID."""
    uid = uuid.uuid4()
    with patch("app.mcp_server.get_current_user_id", return_value=uid):
        yield uid


# ===========================================================================
# Utility function tests (no DB required)
# ===========================================================================


class TestEscapeLikePattern:
    def test_no_special_chars(self) -> None:
        assert escape_like_pattern("hello") == "hello"

    def test_escapes_percent(self) -> None:
        assert escape_like_pattern("50%") == "50\\%"

    def test_escapes_underscore(self) -> None:
        assert escape_like_pattern("my_table") == "my\\_table"

    def test_escapes_backslash(self) -> None:
        assert escape_like_pattern("path\\to") == "path\\\\to"

    def test_escapes_all_special_chars(self) -> None:
        assert escape_like_pattern("50%_\\x") == "50\\%\\_\\\\x"


class TestNodeKindInScope:
    def test_all_scope_accepts_everything(self) -> None:
        assert _node_kind_in_scope("file", "all") is True
        assert _node_kind_in_scope("test_suite", "all") is True
        assert _node_kind_in_scope("ui_route", "all") is True
        assert _node_kind_in_scope("user_flow", "all") is True

    def test_tests_scope(self) -> None:
        assert _node_kind_in_scope("test_suite", "tests") is True
        assert _node_kind_in_scope("test_case", "tests") is True
        assert _node_kind_in_scope("test_fixture", "tests") is True
        assert _node_kind_in_scope("file", "tests") is False
        assert _node_kind_in_scope("ui_route", "tests") is False

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

    def test_code_scope_excludes_tests_ui_flows(self) -> None:
        assert _node_kind_in_scope("file", "code") is True
        assert _node_kind_in_scope("symbol", "code") is True
        assert _node_kind_in_scope("test_suite", "code") is False
        assert _node_kind_in_scope("ui_route", "code") is False
        assert _node_kind_in_scope("user_flow", "code") is False

    def test_normalizes_whitespace_and_case(self) -> None:
        assert _node_kind_in_scope("  TEST_SUITE  ", "tests") is True

    def test_unknown_scope_accepts_everything(self) -> None:
        assert _node_kind_in_scope("file", "unknown_scope") is True


class TestFilterGraphPayload:
    def _make_graph(self) -> dict:
        return {
            "nodes": [
                {"id": "1", "kind": "file", "natural_key": "a.py", "meta": {}},
                {"id": "2", "kind": "test_suite", "natural_key": "test_a.py", "meta": {}},
                {"id": "3", "kind": "ui_route", "natural_key": "/home", "meta": {}},
            ],
            "edges": [
                {"source_node_id": "1", "target_node_id": "2", "kind": "test_covers"},
                {"source_node_id": "1", "target_node_id": "3", "kind": "ui_renders"},
            ],
            "total_nodes": 3,
        }

    def test_scope_all_returns_everything(self) -> None:
        result = _filter_graph_payload(self._make_graph(), scope="all")
        assert len(result["nodes"]) == 3
        assert len(result["edges"]) == 2

    def test_scope_code_filters(self) -> None:
        result = _filter_graph_payload(self._make_graph(), scope="code")
        assert len(result["nodes"]) == 1
        assert result["nodes"][0]["kind"] == "file"
        assert len(result["edges"]) == 0

    def test_exclude_test_links(self) -> None:
        result = _filter_graph_payload(self._make_graph(), scope="all", include_test_links=False)
        edge_kinds = [e["kind"] for e in result["edges"]]
        assert "test_covers" not in edge_kinds

    def test_exclude_ui_links(self) -> None:
        result = _filter_graph_payload(self._make_graph(), scope="all", include_ui_links=False)
        edge_kinds = [e["kind"] for e in result["edges"]]
        assert "ui_renders" not in edge_kinds

    def test_provenance_mode_filter(self) -> None:
        graph = {
            "nodes": [
                {"id": "1", "kind": "file", "meta": {"provenance": {"mode": "deterministic"}}},
                {"id": "2", "kind": "file", "meta": {"provenance": {"mode": "inferred"}}},
            ],
            "edges": [],
            "total_nodes": 2,
        }
        result = _filter_graph_payload(graph, provenance_mode="deterministic")
        assert len(result["nodes"]) == 1
        assert result["nodes"][0]["id"] == "1"


class TestParseCsvList:
    def test_none_returns_none(self) -> None:
        assert _parse_csv_list(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert _parse_csv_list("") is None

    def test_whitespace_only_returns_none(self) -> None:
        assert _parse_csv_list("  , ,  ") is None

    def test_single_value(self) -> None:
        assert _parse_csv_list("graphrag") == ["graphrag"]

    def test_multiple_values(self) -> None:
        assert _parse_csv_list("graphrag,lsp,joern") == ["graphrag", "lsp", "joern"]

    def test_strips_whitespace(self) -> None:
        assert _parse_csv_list(" graphrag , lsp ") == ["graphrag", "lsp"]


class TestSha256Text:
    def test_returns_hex_digest(self) -> None:
        result = _sha256_text("hello")
        assert isinstance(result, str)
        assert len(result) == 64

    def test_deterministic(self) -> None:
        assert _sha256_text("abc") == _sha256_text("abc")

    def test_different_inputs_produce_different_hashes(self) -> None:
        assert _sha256_text("a") != _sha256_text("b")


class TestGetContextMarkdownSync:
    def test_returns_markdown_placeholder(self) -> None:
        result = get_context_markdown_sync("test query")
        assert "# Context for: test query" in result
        assert "## Summary" in result
        assert "## Sources" in result
        assert "test mode" in result


# ===========================================================================
# _resolve_collection_access / _resolve_collection_for_tool
# ===========================================================================


class TestResolveCollectionAccess:
    @pytest.mark.anyio
    async def test_invalid_uuid(self) -> None:
        mock_db = AsyncMock()
        coll, err = await _resolve_collection_access(
            mock_db, collection_id="not-uuid", user_id=None
        )
        assert coll is None
        assert "Invalid collection_id" in err

    @pytest.mark.anyio
    async def test_collection_not_found(self) -> None:
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        coll, err = await _resolve_collection_access(
            mock_db, collection_id=str(uuid.uuid4()), user_id=None
        )
        assert coll is None
        assert "Collection not found" in err

    @pytest.mark.anyio
    async def test_global_collection_accessible(self) -> None:
        from contextmine_core import CollectionVisibility

        coll_mock = MagicMock()
        coll_mock.id = uuid.uuid4()
        coll_mock.visibility = CollectionVisibility.GLOBAL
        coll_mock.owner_user_id = None

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = coll_mock
        mock_db.execute = AsyncMock(return_value=mock_result)

        coll, err = await _resolve_collection_access(
            mock_db, collection_id=str(coll_mock.id), user_id=None
        )
        assert coll is not None
        assert err is None

    @pytest.mark.anyio
    async def test_private_collection_denied(self) -> None:
        from contextmine_core import CollectionVisibility

        coll_mock = MagicMock()
        coll_mock.id = uuid.uuid4()
        coll_mock.visibility = CollectionVisibility.PRIVATE
        coll_mock.owner_user_id = uuid.uuid4()

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = coll_mock
        mock_db.execute = AsyncMock(return_value=mock_result)

        coll, err = await _resolve_collection_access(
            mock_db, collection_id=str(coll_mock.id), user_id=None
        )
        assert coll is None
        assert "Access denied" in err

    @pytest.mark.anyio
    async def test_owner_has_access_to_private(self) -> None:
        from contextmine_core import CollectionVisibility

        owner_id = uuid.uuid4()
        coll_mock = MagicMock()
        coll_mock.id = uuid.uuid4()
        coll_mock.visibility = CollectionVisibility.PRIVATE
        coll_mock.owner_user_id = owner_id

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = coll_mock
        mock_db.execute = AsyncMock(return_value=mock_result)

        coll, err = await _resolve_collection_access(
            mock_db, collection_id=str(coll_mock.id), user_id=owner_id
        )
        assert coll is not None
        assert err is None


class TestResolveCollectionForTool:
    @pytest.mark.anyio
    async def test_with_collection_id_delegates(self) -> None:
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        coll, err = await _resolve_collection_for_tool(
            mock_db, collection_id=str(uuid.uuid4()), user_id=None
        )
        assert coll is None
        assert "Collection not found" in err

    @pytest.mark.anyio
    async def test_without_collection_id_finds_first(self) -> None:
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        coll, err = await _resolve_collection_for_tool(mock_db, collection_id=None, user_id=None)
        assert coll is None
        assert "No accessible collection found" in err


# ===========================================================================
# list_collections
# ===========================================================================


class TestListCollections:
    @pytest.mark.anyio
    async def test_returns_no_collections(self, _patch_user_id) -> None:
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_db.execute = AsyncMock(return_value=mock_result)

        ctx = _mock_db_session(mock_db)
        try:
            result = await _list_collections()
        finally:
            ctx.stop()

        assert "# No Collections Found" in result

    @pytest.mark.anyio
    async def test_returns_collections_list(self, _patch_user_id) -> None:
        coll_id = uuid.uuid4()
        mock_db = AsyncMock()
        mock_result = MagicMock()
        visibility_mock = MagicMock()
        visibility_mock.value = "global"
        mock_result.all.return_value = [
            (coll_id, "My Collection", "my-collection", visibility_mock, 3),
        ]
        mock_db.execute = AsyncMock(return_value=mock_result)

        ctx = _mock_db_session(mock_db)
        try:
            result = await _list_collections()
        finally:
            ctx.stop()

        assert "# Available Collections" in result
        assert "My Collection" in result
        assert str(coll_id) in result
        assert "Sources**: 3" in result

    @pytest.mark.anyio
    async def test_with_search_filter(self, _patch_user_id) -> None:
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_db.execute = AsyncMock(return_value=mock_result)

        ctx = _mock_db_session(mock_db)
        try:
            result = await _list_collections(search="test")
        finally:
            ctx.stop()

        assert "# No Collections Found" in result


# ===========================================================================
# list_documents
# ===========================================================================


class TestListDocuments:
    @pytest.mark.anyio
    async def test_invalid_collection_id(self, _patch_user_id) -> None:
        result = await _list_documents(collection_id="not-a-uuid")
        assert "# Error" in result
        assert "Invalid collection_id" in result

    @pytest.mark.anyio
    async def test_collection_not_found(self, _patch_user_id) -> None:
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        ctx = _mock_db_session(mock_db)
        try:
            result = await _list_documents(collection_id=str(uuid.uuid4()))
        finally:
            ctx.stop()

        assert "Collection not found" in result

    @pytest.mark.anyio
    async def test_access_denied_private_collection(self, _patch_user_id) -> None:
        from contextmine_core import CollectionVisibility

        coll = MagicMock()
        coll.id = uuid.uuid4()
        coll.visibility = CollectionVisibility.PRIVATE
        coll.owner_user_id = uuid.uuid4()

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = coll
        mock_db.execute = AsyncMock(return_value=mock_result)

        ctx = _mock_db_session(mock_db)
        try:
            result = await _list_documents(collection_id=str(coll.id))
        finally:
            ctx.stop()

        assert "Access denied" in result

    @pytest.mark.anyio
    async def test_no_documents(self, _patch_user_id) -> None:
        from contextmine_core import CollectionVisibility

        coll = MagicMock()
        coll.id = uuid.uuid4()
        coll.name = "My Collection"
        coll.visibility = CollectionVisibility.GLOBAL

        mock_db = AsyncMock()
        call_count = 0

        async def mock_execute(query):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.scalar_one_or_none.return_value = coll
            else:
                result.all.return_value = []
            return result

        mock_db.execute = mock_execute

        ctx = _mock_db_session(mock_db)
        try:
            result = await _list_documents(collection_id=str(coll.id))
        finally:
            ctx.stop()

        assert "# No Documents Found" in result

    @pytest.mark.anyio
    async def test_documents_with_topic_filter(self, _patch_user_id) -> None:
        from contextmine_core import CollectionVisibility

        coll = MagicMock()
        coll.id = uuid.uuid4()
        coll.name = "My Collection"
        coll.visibility = CollectionVisibility.GLOBAL

        doc_id = uuid.uuid4()
        mock_db = AsyncMock()
        call_count = 0

        async def mock_execute(query):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.scalar_one_or_none.return_value = coll
            else:
                result.all.return_value = [
                    (doc_id, "git://repo/src/api.py", "api.py", "https://github.com/repo"),
                ]
            return result

        mock_db.execute = mock_execute

        ctx = _mock_db_session(mock_db)
        try:
            result = await _list_documents(collection_id=str(coll.id), topic="api")
        finally:
            ctx.stop()

        assert "Documents in My Collection" in result
        assert "api.py" in result
        assert "Filtered by topic: api" in result


# ===========================================================================
# get_markdown
# ===========================================================================


class TestGetMarkdown:
    @pytest.mark.anyio
    async def test_invalid_collection_id(self, _patch_user_id) -> None:
        result = await _get_context_markdown(query="test", collection_id="bad")
        assert "# Error" in result
        assert "Invalid collection_id" in result

    @pytest.mark.anyio
    async def test_standard_mode_calls_assemble_context(self, _patch_user_id) -> None:
        @dataclass
        class FakeResponse:
            markdown: str = "# Answer\n\nSome assembled context."

        with patch("app.mcp_server.assemble_context", new_callable=AsyncMock) as mock_ac:
            mock_ac.return_value = FakeResponse()
            result = await _get_context_markdown(query="how does auth work?")

        assert "Some assembled context" in result
        mock_ac.assert_awaited_once()

    @pytest.mark.anyio
    async def test_raw_mode(self, _patch_user_id) -> None:
        with patch("app.mcp_server._get_raw_chunks", new_callable=AsyncMock) as mock_raw:
            mock_raw.return_value = "# Search Results for: test\n\n## Result 1: doc"
            result = await _get_context_markdown(query="test", raw=True)

        assert "Search Results" in result

    @pytest.mark.anyio
    async def test_topic_mode_uses_raw_chunks(self, _patch_user_id) -> None:
        with patch("app.mcp_server._get_raw_chunks", new_callable=AsyncMock) as mock_raw:
            mock_raw.return_value = "# Search Results\n\nFiltered"
            result = await _get_context_markdown(query="test", topic="api")

        assert "Search Results" in result

    @pytest.mark.anyio
    async def test_error_returns_error_message(self, _patch_user_id) -> None:
        with patch("app.mcp_server.assemble_context", new_callable=AsyncMock) as mock_ac:
            mock_ac.side_effect = RuntimeError("LLM provider not configured")
            result = await _get_context_markdown(query="test")

        assert "# Error" in result
        assert "Failed to retrieve context" in result


# ===========================================================================
# outline
# ===========================================================================


class TestOutline:
    @pytest.mark.anyio
    async def test_document_with_symbols(self) -> None:
        doc = _make_document()
        sym1 = _make_symbol(
            name="MyClass",
            kind_value="class",
            start_line=1,
            end_line=20,
            signature="class MyClass:",
            qualified_name="MyClass",
        )
        sym2 = _make_symbol(
            name="my_method",
            kind_value="method",
            start_line=5,
            end_line=10,
            signature="def my_method(self)",
            parent_name="MyClass",
            qualified_name="MyClass.my_method",
        )

        mock_db = AsyncMock()
        call_count = 0

        async def mock_execute(query):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.scalar_one_or_none.return_value = doc
            else:
                scalars_mock = MagicMock()
                scalars_mock.all.return_value = [sym1, sym2]
                result.scalars.return_value = scalars_mock
            return result

        mock_db.execute = mock_execute
        ctx = _mock_db_session(mock_db)
        try:
            result = await _code_outline(file_path="src/main.py")
        finally:
            ctx.stop()

        assert "# Outline: src/main.py" in result
        assert "MyClass" in result
        assert "my_method" in result

    @pytest.mark.anyio
    async def test_document_not_found(self) -> None:
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        ctx = _mock_db_session(mock_db)
        try:
            result = await _code_outline(file_path="nonexistent.py")
        finally:
            ctx.stop()

        assert "No symbols found" in result

    @pytest.mark.anyio
    async def test_document_with_no_symbols(self) -> None:
        doc = _make_document()
        mock_db = AsyncMock()
        call_count = 0

        async def mock_execute(query):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.scalar_one_or_none.return_value = doc
            else:
                scalars_mock = MagicMock()
                scalars_mock.all.return_value = []
                result.scalars.return_value = scalars_mock
            return result

        mock_db.execute = mock_execute
        ctx = _mock_db_session(mock_db)
        try:
            result = await _code_outline(file_path="readme.md")
        finally:
            ctx.stop()

        assert "No symbols found" in result

    @pytest.mark.anyio
    async def test_db_error(self) -> None:
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(side_effect=RuntimeError("DB connection lost"))

        ctx = _mock_db_session(mock_db)
        try:
            result = await _code_outline(file_path="src/main.py")
        finally:
            ctx.stop()

        assert "Document Not Found" in result or "Error" in result


# ===========================================================================
# find_symbol
# ===========================================================================


class TestFindSymbol:
    @pytest.mark.anyio
    async def test_symbol_found(self) -> None:
        doc = _make_document(content_markdown="line1\nline2\nline3\ndef foo():\n    pass\nline6")
        sym = _make_symbol(name="foo", start_line=4, end_line=5)

        mock_db = AsyncMock()
        call_count = 0

        async def mock_execute(query):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.scalar_one_or_none.return_value = doc
            else:
                result.scalar_one_or_none.return_value = sym
            return result

        mock_db.execute = mock_execute
        ctx = _mock_db_session(mock_db)
        try:
            result = await _code_find_symbol(file_path="src/main.py", name="foo")
        finally:
            ctx.stop()

        assert "function" in result
        assert "`foo`" in result
        assert "Lines:** 4-5" in result

    @pytest.mark.anyio
    async def test_symbol_not_found(self) -> None:
        doc = _make_document()
        mock_db = AsyncMock()
        call_count = 0

        async def mock_execute(query):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.scalar_one_or_none.return_value = doc
            else:
                result.scalar_one_or_none.return_value = None
            return result

        mock_db.execute = mock_execute
        ctx = _mock_db_session(mock_db)
        try:
            result = await _code_find_symbol(file_path="src/main.py", name="bar")
        finally:
            ctx.stop()

        assert "Symbol Not Found" in result

    @pytest.mark.anyio
    async def test_document_not_found(self) -> None:
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        ctx = _mock_db_session(mock_db)
        try:
            result = await _code_find_symbol(file_path="nonexistent.py", name="foo")
        finally:
            ctx.stop()

        assert "Symbol Not Found" in result


# ===========================================================================
# definition
# ===========================================================================


class TestDefinition:
    @pytest.mark.anyio
    async def test_symbol_at_location(self) -> None:
        doc = _make_document(
            uri="git://repo/src/main.py",
            content_markdown="line1\nline2\nline3\ndef foo():\n    pass",
        )
        sym = _make_symbol(
            name="foo",
            kind_value="function",
            start_line=4,
            end_line=5,
            signature="def foo()",
            qualified_name="main.foo",
            meta={"docstring": "A test function."},
        )

        mock_db = AsyncMock()
        call_count = 0

        async def mock_execute(query):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.scalar_one_or_none.return_value = doc
            else:
                scalars_mock = MagicMock()
                scalars_mock.first.return_value = sym
                result.scalars.return_value = scalars_mock
            return result

        mock_db.execute = mock_execute
        ctx = _mock_db_session(mock_db)
        try:
            result = await _code_definition(file_path="src/main.py", line=4, column=0)
        finally:
            ctx.stop()

        assert "# Definition Found" in result
        assert "main.foo" in result

    @pytest.mark.anyio
    async def test_document_not_found(self) -> None:
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        ctx = _mock_db_session(mock_db)
        try:
            result = await _code_definition(file_path="nonexistent.py", line=1, column=0)
        finally:
            ctx.stop()

        assert "Document Not Found" in result

    @pytest.mark.anyio
    async def test_no_symbol_at_line(self) -> None:
        doc = _make_document()
        mock_db = AsyncMock()
        call_count = 0

        async def mock_execute(query):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.scalar_one_or_none.return_value = doc
            else:
                scalars_mock = MagicMock()
                scalars_mock.first.return_value = None
                result.scalars.return_value = scalars_mock
            return result

        mock_db.execute = mock_execute
        ctx = _mock_db_session(mock_db)
        try:
            result = await _code_definition(file_path="src/main.py", line=100, column=0)
        finally:
            ctx.stop()

        assert "No Symbol Found" in result


# ===========================================================================
# references
# ===========================================================================


class TestReferences:
    @pytest.mark.anyio
    async def test_no_references(self) -> None:
        doc = _make_document()
        sym = _make_symbol(qualified_name="main.foo")

        mock_db = AsyncMock()
        call_count = 0

        async def mock_execute(query):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.scalar_one_or_none.return_value = doc
            elif call_count == 2:
                scalars_mock = MagicMock()
                scalars_mock.first.return_value = sym
                result.scalars.return_value = scalars_mock
            else:
                scalars_mock = MagicMock()
                scalars_mock.all.return_value = []
                result.scalars.return_value = scalars_mock
            return result

        mock_db.execute = mock_execute
        ctx = _mock_db_session(mock_db)
        try:
            result = await _code_references(file_path="src/main.py", line=4, column=0)
        finally:
            ctx.stop()

        assert "No References Found" in result

    @pytest.mark.anyio
    async def test_document_not_found(self) -> None:
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        ctx = _mock_db_session(mock_db)
        try:
            result = await _code_references(file_path="nonexistent.py", line=1, column=0)
        finally:
            ctx.stop()

        assert "Document Not Found" in result


# ===========================================================================
# expand
# ===========================================================================


class TestExpand:
    @pytest.mark.anyio
    async def test_no_seeds_returns_error(self) -> None:
        result = await _code_expand(seeds=[])
        assert "# Error" in result
        assert "No seed symbols" in result

    @pytest.mark.anyio
    async def test_invalid_seed_format(self) -> None:
        result = await _code_expand(seeds=["no_separator"])
        assert "# Error" in result
        assert "Invalid seed format" in result

    @pytest.mark.anyio
    async def test_import_error(self) -> None:
        with patch.dict("sys.modules", {"contextmine_core.graph": None}):
            result = await _code_expand(seeds=["src/main.py::foo"])
        assert "Error" in result


# ===========================================================================
# deep_research
# ===========================================================================


class TestDeepResearch:
    @pytest.mark.anyio
    async def test_successful_research(self) -> None:
        mock_run = MagicMock()
        mock_run.answer = "Found the answer"
        mock_run.evidence = []
        mock_run.run_id = "test-run-123456789"
        mock_run.status = MagicMock(value="done")
        mock_run.budget_used = 5
        mock_run.budget_steps = 20

        with (
            patch("contextmine_core.research.llm.get_research_llm_provider"),
            patch("contextmine_core.research.ResearchAgent") as mock_agent_cls,
            patch(
                "contextmine_core.research.format_answer_with_citations",
                return_value="Found the answer",
            ),
        ):
            mock_agent = AsyncMock()
            mock_agent.research = AsyncMock(return_value=mock_run)
            mock_agent_cls.return_value = mock_agent
            result = await _code_deep_research(question="how does X work?", budget=50)

        assert "Found the answer" in result

    @pytest.mark.anyio
    async def test_research_error(self) -> None:
        with patch(
            "contextmine_core.research.llm.get_research_llm_provider",
            side_effect=RuntimeError("No API key"),
        ):
            result = await _code_deep_research(question="test")
        assert "Research error" in result
        assert "No API key" in result

    @pytest.mark.anyio
    async def test_debug_mode(self) -> None:
        mock_run = MagicMock()
        mock_run.answer = "The answer"
        mock_run.evidence = []
        mock_run.run_id = "run-abc-123"
        mock_run.status = MagicMock(value="done")
        mock_run.budget_used = 3
        mock_run.budget_steps = 10

        with (
            patch("contextmine_core.research.llm.get_research_llm_provider"),
            patch("contextmine_core.research.ResearchAgent") as mock_agent_cls,
            patch(
                "contextmine_core.research.format_answer_with_citations", return_value="The answer"
            ),
        ):
            mock_agent = AsyncMock()
            mock_agent.research = AsyncMock(return_value=mock_run)
            mock_agent_cls.return_value = mock_agent
            result = await _code_deep_research(question="test", debug=True)

        assert "Run ID:" in result
        assert "run-abc-123" in result

    @pytest.mark.anyio
    async def test_failed_research_run(self) -> None:
        mock_run = MagicMock()
        mock_run.answer = None
        mock_run.evidence = []
        mock_run.run_id = "run-fail"
        mock_run.status = MagicMock(value="failed")
        mock_run.budget_used = 1
        mock_run.budget_steps = 10
        mock_run.error_message = "Out of budget"

        with (
            patch("contextmine_core.research.llm.get_research_llm_provider"),
            patch("contextmine_core.research.ResearchAgent") as mock_agent_cls,
        ):
            mock_agent = AsyncMock()
            mock_agent.research = AsyncMock(return_value=mock_run)
            mock_agent_cls.return_value = mock_agent
            result = await _code_deep_research(question="test")

        assert "Research failed" in result
        assert "Out of budget" in result


# ===========================================================================
# graph_neighborhood
# ===========================================================================


class TestGraphNeighborhood:
    @pytest.mark.anyio
    async def test_invalid_node_id(self) -> None:
        result = await _graph_neighborhood(node_id="not-a-uuid")
        assert "# Error" in result
        assert "Invalid node_id" in result

    @pytest.mark.anyio
    async def test_no_connections(self) -> None:
        mock_result = MagicMock()
        mock_result.entities = []

        mock_db = AsyncMock()
        with (
            patch(
                "contextmine_core.graphrag.graph_neighborhood",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
            patch("app.mcp_server.get_db_session") as mock_session,
        ):
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await _graph_neighborhood(node_id=str(uuid.uuid4()))

        assert "No Neighborhood Found" in result

    @pytest.mark.anyio
    async def test_happy_path(self) -> None:
        mock_result = MagicMock()
        mock_result.entities = [MagicMock()]
        mock_result.to_markdown.return_value = "# Neighborhood\n\n- node A\n- node B"

        mock_db = AsyncMock()
        with (
            patch(
                "contextmine_core.graphrag.graph_neighborhood",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
            patch("app.mcp_server.get_db_session") as mock_session,
        ):
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await _graph_neighborhood(node_id=str(uuid.uuid4()))

        assert "Neighborhood" in result
        assert "node A" in result

    @pytest.mark.anyio
    async def test_depth_capped_at_3(self) -> None:
        mock_result = MagicMock()
        mock_result.entities = [MagicMock()]
        mock_result.to_markdown.return_value = "# Neighborhood\n"

        mock_db = AsyncMock()
        with (
            patch(
                "contextmine_core.graphrag.graph_neighborhood",
                new_callable=AsyncMock,
                return_value=mock_result,
            ) as mock_gn,
            patch("app.mcp_server.get_db_session") as mock_session,
        ):
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
            await _graph_neighborhood(node_id=str(uuid.uuid4()), depth=10)

        assert mock_gn.call_args.kwargs["depth"] == 3


# ===========================================================================
# trace_path
# ===========================================================================


class TestTracePath:
    @pytest.mark.anyio
    async def test_invalid_from_node_id(self) -> None:
        result = await _trace_path(from_node_id="bad", to_node_id=str(uuid.uuid4()))
        assert "# Error" in result
        assert "Invalid node ID" in result

    @pytest.mark.anyio
    async def test_invalid_to_node_id(self) -> None:
        result = await _trace_path(from_node_id=str(uuid.uuid4()), to_node_id="bad")
        assert "# Error" in result
        assert "Invalid node ID" in result

    @pytest.mark.anyio
    async def test_no_path_found(self) -> None:
        mock_result = MagicMock()
        mock_result.entities = []

        mock_db = AsyncMock()
        with (
            patch(
                "contextmine_core.graphrag.trace_path",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
            patch("app.mcp_server.get_db_session") as mock_session,
        ):
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await _trace_path(from_node_id=str(uuid.uuid4()), to_node_id=str(uuid.uuid4()))

        assert "No Path Found" in result

    @pytest.mark.anyio
    async def test_path_found(self) -> None:
        mock_result = MagicMock()
        mock_result.entities = [MagicMock()]
        mock_result.to_markdown.return_value = "# Path\n\nA -> B -> C"

        mock_db = AsyncMock()
        with (
            patch(
                "contextmine_core.graphrag.trace_path",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
            patch("app.mcp_server.get_db_session") as mock_session,
        ):
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await _trace_path(from_node_id=str(uuid.uuid4()), to_node_id=str(uuid.uuid4()))

        assert "Path" in result
        assert "A -> B -> C" in result

    @pytest.mark.anyio
    async def test_max_hops_capped_at_10(self) -> None:
        mock_result = MagicMock()
        mock_result.entities = [MagicMock()]
        mock_result.to_markdown.return_value = "# Path"

        mock_db = AsyncMock()
        with (
            patch(
                "contextmine_core.graphrag.trace_path",
                new_callable=AsyncMock,
                return_value=mock_result,
            ) as mock_tp,
            patch("app.mcp_server.get_db_session") as mock_session,
        ):
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
            await _trace_path(
                from_node_id=str(uuid.uuid4()), to_node_id=str(uuid.uuid4()), max_hops=99
            )

        assert mock_tp.call_args.kwargs["max_hops"] == 10


# ===========================================================================
# graph_rag
# ===========================================================================


class TestGraphRag:
    @pytest.mark.anyio
    async def test_invalid_twin_scope(self, _patch_user_id) -> None:
        result = await _graph_rag(query="test", twin_scope="invalid")
        assert "# Error" in result
        assert "Invalid twin_scope" in result

    @pytest.mark.anyio
    async def test_context_mode_markdown(self, _patch_user_id) -> None:
        mock_pack = MagicMock()
        mock_pack.entities = []
        mock_pack.edges = []
        mock_pack.to_markdown.return_value = (
            "# GraphRAG Context: test\n\nFound 2 communities, 5 entities, 3 citations.\n"
        )

        mock_db = AsyncMock()
        with (
            patch(
                "contextmine_core.graphrag.graph_rag_context",
                new_callable=AsyncMock,
                return_value=mock_pack,
            ),
            patch("app.mcp_server.get_db_session") as mock_session,
        ):
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await _graph_rag(query="test")

        assert "GraphRAG Context" in result

    @pytest.mark.anyio
    async def test_context_mode_json(self, _patch_user_id) -> None:
        mock_pack = MagicMock()
        mock_pack.entities = []
        mock_pack.edges = []
        mock_pack.to_dict.return_value = {"query": "test", "communities": [], "entities": []}

        mock_db = AsyncMock()
        with (
            patch(
                "contextmine_core.graphrag.graph_rag_context",
                new_callable=AsyncMock,
                return_value=mock_pack,
            ),
            patch("app.mcp_server.get_db_session") as mock_session,
        ):
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await _graph_rag(query="test", format="json")

        parsed = json.loads(result)
        assert parsed["query"] == "test"

    @pytest.mark.anyio
    async def test_empty_context_returns_no_results(self, _patch_user_id) -> None:
        mock_pack = MagicMock()
        mock_pack.entities = []
        mock_pack.edges = []
        mock_pack.to_markdown.return_value = (
            "# GraphRAG Context: test\n\nFound 0 communities, 0 entities, 0 citations.\n"
        )

        mock_db = AsyncMock()
        with (
            patch(
                "contextmine_core.graphrag.graph_rag_context",
                new_callable=AsyncMock,
                return_value=mock_pack,
            ),
            patch("app.mcp_server.get_db_session") as mock_session,
        ):
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await _graph_rag(query="test")

        assert "# No Results" in result

    @pytest.mark.anyio
    async def test_error_handling(self, _patch_user_id) -> None:
        mock_db = AsyncMock()
        with (
            patch(
                "contextmine_core.graphrag.graph_rag_context",
                new_callable=AsyncMock,
                side_effect=RuntimeError("DB error"),
            ),
            patch("app.mcp_server.get_db_session") as mock_session,
        ):
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await _graph_rag(query="test")

        assert "# Error" in result
        assert "GraphRAG query failed" in result

    @pytest.mark.anyio
    async def test_rebuild_mode_json(self, _patch_user_id) -> None:
        mock_pack = MagicMock()
        mock_pack.entities = []
        mock_pack.edges = []
        mock_pack.to_dict.return_value = {
            "query": "test",
            "communities": [],
            "entities": [
                {"kind": "api_endpoint"},
                {"kind": "ui_route"},
                {"kind": "test_case"},
            ],
        }

        mock_db = AsyncMock()
        with (
            patch(
                "contextmine_core.graphrag.graph_rag_context",
                new_callable=AsyncMock,
                return_value=mock_pack,
            ),
            patch("app.mcp_server.get_db_session") as mock_session,
        ):
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await _graph_rag(query="test", format="json", rebuild_mode=True)

        parsed = json.loads(result)
        assert "rebuild_sections" in parsed
        assert "System Boundaries" in parsed["rebuild_sections"]


# ===========================================================================
# research_validation
# ===========================================================================


class TestResearchValidation:
    @pytest.mark.anyio
    async def test_no_collections(self, _patch_user_id) -> None:
        mock_db = AsyncMock()
        with (
            patch(
                "contextmine_core.search.get_accessible_collection_ids",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("app.mcp_server.get_db_session") as mock_session,
        ):
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await _research_validation(code_path="auth.py")

        assert "No Collections Available" in result

    @pytest.mark.anyio
    async def test_no_rules_found(self, _patch_user_id) -> None:
        mock_db = AsyncMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_db.execute = AsyncMock(return_value=mock_result)

        with (
            patch(
                "contextmine_core.search.get_accessible_collection_ids",
                new_callable=AsyncMock,
                return_value=[uuid.uuid4()],
            ),
            patch("app.mcp_server.get_db_session") as mock_session,
        ):
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await _research_validation(code_path="zzz_unknown.py")

        assert "No Validation Rules Found" in result

    @pytest.mark.anyio
    async def test_error_handling(self, _patch_user_id) -> None:
        with patch("app.mcp_server.get_db_session", side_effect=RuntimeError("DB down")):
            result = await _research_validation(code_path="auth.py")
        assert "# Error" in result
        assert "Failed to research validation" in result


# ===========================================================================
# research_data_model
# ===========================================================================


class TestResearchDataModel:
    @pytest.mark.anyio
    async def test_no_collections(self, _patch_user_id) -> None:
        mock_db = AsyncMock()
        with (
            patch(
                "contextmine_core.search.get_accessible_collection_ids",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("app.mcp_server.get_db_session") as mock_session,
        ):
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await _research_data_model(entity="users")

        assert "No Collections Available" in result

    @pytest.mark.anyio
    async def test_no_data_found(self, _patch_user_id) -> None:
        mock_db = AsyncMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        with (
            patch(
                "contextmine_core.search.get_accessible_collection_ids",
                new_callable=AsyncMock,
                return_value=[uuid.uuid4()],
            ),
            patch("app.mcp_server.get_db_session") as mock_session,
        ):
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await _research_data_model(entity="zzz_nonexistent")

        assert "No Data Model Found" in result


# ===========================================================================
# research_architecture
# ===========================================================================


class TestResearchArchitecture:
    @pytest.mark.anyio
    async def test_no_collections(self, _patch_user_id) -> None:
        mock_db = AsyncMock()
        with (
            patch(
                "contextmine_core.search.get_accessible_collection_ids",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("app.mcp_server.get_db_session") as mock_session,
        ):
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await _research_architecture(topic="api")

        assert "No Collections Available" in result

    @pytest.mark.anyio
    async def test_unknown_topic_returns_suggestion(self, _patch_user_id) -> None:
        mock_db = AsyncMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_db.execute = AsyncMock(return_value=mock_result)

        with (
            patch(
                "contextmine_core.search.get_accessible_collection_ids",
                new_callable=AsyncMock,
                return_value=[uuid.uuid4()],
            ),
            patch("app.mcp_server.get_db_session") as mock_session,
        ):
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await _research_architecture(topic="zzzzz")

        assert "No specific architecture information" in result
        assert "Try topics like" in result

    @pytest.mark.anyio
    async def test_error_handling(self, _patch_user_id) -> None:
        with patch("app.mcp_server.get_db_session", side_effect=RuntimeError("DB error")):
            result = await _research_architecture(topic="api")
        assert "# Error" in result
        assert "Failed to research architecture" in result


# ===========================================================================
# get_arc42
# ===========================================================================


class TestGetArc42:
    @pytest.mark.anyio
    async def test_arch_docs_disabled(self, _patch_user_id) -> None:
        with patch("app.mcp_server.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(arch_docs_enabled=False)
            result = await _get_arc42()
        assert "Architecture docs are disabled" in result

    @pytest.mark.anyio
    async def test_invalid_section(self, _patch_user_id) -> None:
        with (
            patch("app.mcp_server.get_settings") as mock_settings,
            patch("contextmine_core.architecture.normalize_arc42_section_key", return_value=None),
        ):
            mock_settings.return_value = MagicMock(arch_docs_enabled=True)
            result = await _get_arc42(section="invalid_999")
        assert "Invalid section" in result

    @pytest.mark.anyio
    async def test_error_handling(self, _patch_user_id) -> None:
        with patch("app.mcp_server.get_settings", side_effect=RuntimeError("config error")):
            result = await _get_arc42()
        assert "# Error" in result
        assert "Failed to get arc42" in result


# ===========================================================================
# arc42_drift_report
# ===========================================================================


class TestArc42DriftReport:
    @pytest.mark.anyio
    async def test_arch_docs_disabled(self, _patch_user_id) -> None:
        with patch("app.mcp_server.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(arch_docs_enabled=False)
            result = await _arc42_drift()
        assert "Architecture docs are disabled" in result

    @pytest.mark.anyio
    async def test_error_handling(self, _patch_user_id) -> None:
        with patch("app.mcp_server.get_settings", side_effect=RuntimeError("boom")):
            result = await _arc42_drift()
        assert "# Error" in result
        assert "Failed to compute arc42 drift" in result


# ===========================================================================
# store_findings
# ===========================================================================


class TestStoreFindings:
    @pytest.mark.anyio
    async def test_invalid_json(self, _patch_user_id) -> None:
        result = await _store_findings(collection_id=str(uuid.uuid4()), findings_json="not json{")
        assert "# Error" in result
        assert "valid JSON" in result

    @pytest.mark.anyio
    async def test_non_array_json(self, _patch_user_id) -> None:
        result = await _store_findings(
            collection_id=str(uuid.uuid4()), findings_json='{"not": "array"}'
        )
        assert "# Error" in result
        assert "JSON array" in result


# ===========================================================================
# export_twin_view
# ===========================================================================


class TestExportTwinView:
    @pytest.mark.anyio
    async def test_scenario_not_found(self) -> None:
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        ctx = _mock_db_session(mock_db)
        try:
            result = await _export_twin_view(scenario_id=str(uuid.uuid4()), format="lpg_jsonl")
        finally:
            ctx.stop()

        assert "Scenario not found" in result

    @pytest.mark.anyio
    async def test_unsupported_format(self) -> None:
        scenario = MagicMock()
        scenario.id = uuid.uuid4()
        scenario.collection_id = uuid.uuid4()
        scenario.name = "test"
        scenario.is_as_is = True

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = scenario
        mock_db.execute = AsyncMock(return_value=mock_result)

        ctx = _mock_db_session(mock_db)
        try:
            result = await _export_twin_view(scenario_id=str(uuid.uuid4()), format="unsupported")
        finally:
            ctx.stop()

        assert "Unsupported export format" in result


# ===========================================================================
# get_twin_graph
# ===========================================================================


class TestGetTwinGraph:
    @pytest.mark.anyio
    async def test_invalid_facet(self) -> None:
        graph = {"nodes": [], "edges": [], "total_nodes": 0}
        mock_db = AsyncMock()
        with (
            patch(
                "contextmine_core.twin.get_scenario_graph",
                new_callable=AsyncMock,
                return_value=graph,
            ),
            patch("app.mcp_server.get_db_session") as mock_session,
        ):
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await _get_twin_graph(scenario_id=str(uuid.uuid4()), facet="invalid")

        assert "Invalid facet" in result

    @pytest.mark.anyio
    async def test_json_format(self) -> None:
        graph = {
            "nodes": [{"id": "1", "kind": "file", "natural_key": "a.py"}],
            "edges": [],
            "total_nodes": 1,
        }
        mock_db = AsyncMock()
        with (
            patch(
                "contextmine_core.twin.get_scenario_graph",
                new_callable=AsyncMock,
                return_value=graph,
            ),
            patch("app.mcp_server.get_db_session") as mock_session,
        ):
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await _get_twin_graph(scenario_id=str(uuid.uuid4()), format="json")

        parsed = json.loads(result)
        assert "nodes" in parsed
        assert len(parsed["nodes"]) == 1

    @pytest.mark.anyio
    async def test_markdown_format(self) -> None:
        graph = {
            "nodes": [{"id": "1", "kind": "file", "natural_key": "a.py"}],
            "edges": [],
            "total_nodes": 1,
        }
        mock_db = AsyncMock()
        with (
            patch(
                "contextmine_core.twin.get_scenario_graph",
                new_callable=AsyncMock,
                return_value=graph,
            ),
            patch("app.mcp_server.get_db_session") as mock_session,
        ):
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await _get_twin_graph(scenario_id=str(uuid.uuid4()), format="markdown")

        assert "# Twin Graph" in result
        assert "a.py" in result


# ===========================================================================
# get_twin_status
# ===========================================================================


class TestGetTwinStatus:
    @pytest.mark.anyio
    async def test_invalid_collection_id(self, _patch_user_id) -> None:
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        ctx = _mock_db_session(mock_db)
        try:
            result = await _get_twin_status(collection_id="not-a-uuid")
        finally:
            ctx.stop()

        assert "Error" in result

    @pytest.mark.anyio
    async def test_error_handling(self, _patch_user_id) -> None:
        with patch("app.mcp_server.get_db_session", side_effect=RuntimeError("DB error")):
            result = await _get_twin_status(collection_id=str(uuid.uuid4()))
        assert "# Error" in result
        assert "Failed to fetch twin status" in result


# ===========================================================================
# create_architecture_intent / approve_architecture_intent
# ===========================================================================


class TestCreateArchitectureIntent:
    @pytest.mark.anyio
    async def test_unauthenticated(self, _patch_user_id) -> None:
        result = await _create_intent(
            scenario_id=str(uuid.uuid4()),
            action="EXTRACT_DOMAIN",
            target_type="node",
            target_id="test",
            expected_scenario_version=1,
        )
        assert "Authentication required" in result

    @pytest.mark.anyio
    async def test_scenario_not_found(self, _patch_user_id_with_uuid) -> None:
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        ctx = _mock_db_session(mock_db)
        try:
            result = await _create_intent(
                scenario_id=str(uuid.uuid4()),
                action="EXTRACT_DOMAIN",
                target_type="node",
                target_id="test",
                expected_scenario_version=1,
            )
        finally:
            ctx.stop()

        assert "Scenario not found" in result


class TestApproveArchitectureIntent:
    @pytest.mark.anyio
    async def test_unauthenticated(self, _patch_user_id) -> None:
        result = await _approve_intent(scenario_id=str(uuid.uuid4()), intent_id=str(uuid.uuid4()))
        assert "Authentication required" in result

    @pytest.mark.anyio
    async def test_scenario_not_found(self, _patch_user_id_with_uuid) -> None:
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        ctx = _mock_db_session(mock_db)
        try:
            result = await _approve_intent(
                scenario_id=str(uuid.uuid4()), intent_id=str(uuid.uuid4())
            )
        finally:
            ctx.stop()

        assert "Scenario not found" in result


# ===========================================================================
# list_ports_adapters
# ===========================================================================


class TestListPortsAdapters:
    @pytest.mark.anyio
    async def test_arch_docs_disabled(self, _patch_user_id) -> None:
        with patch("app.mcp_server.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(arch_docs_enabled=False)
            result = await _list_ports_adapters()
        assert "Architecture docs are disabled" in result

    @pytest.mark.anyio
    async def test_invalid_direction(self, _patch_user_id) -> None:
        with patch("app.mcp_server.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(arch_docs_enabled=True)
            result = await _list_ports_adapters(direction="sideways")
        assert "Invalid direction" in result

    @pytest.mark.anyio
    async def test_error_handling(self, _patch_user_id) -> None:
        with patch("app.mcp_server.get_settings", side_effect=RuntimeError("boom")):
            result = await _list_ports_adapters()
        assert "# Error" in result
        assert "Failed to list ports/adapters" in result


# ===========================================================================
# Error-path tests for multi-engine and taint tools
# ===========================================================================


class TestMultiEngineToolErrors:
    @pytest.mark.anyio
    async def test_list_methods_error(self, _patch_user_id) -> None:
        with patch("app.mcp_server.get_db_session", side_effect=RuntimeError("DB error")):
            result = await _list_methods(collection_id=str(uuid.uuid4()))
        assert "Failed to list methods" in result

    @pytest.mark.anyio
    async def test_list_calls_error(self, _patch_user_id) -> None:
        with patch("app.mcp_server.get_db_session", side_effect=RuntimeError("DB error")):
            result = await _list_calls(collection_id=str(uuid.uuid4()))
        assert "Failed to list calls" in result

    @pytest.mark.anyio
    async def test_get_cfg_error(self, _patch_user_id) -> None:
        with patch("app.mcp_server.get_db_session", side_effect=RuntimeError("DB error")):
            result = await _get_cfg(collection_id=str(uuid.uuid4()), node_ref="main.foo")
        assert "Failed to fetch CFG" in result

    @pytest.mark.anyio
    async def test_get_variable_flow_error(self, _patch_user_id) -> None:
        with patch("app.mcp_server.get_db_session", side_effect=RuntimeError("DB error")):
            result = await _get_variable_flow(collection_id=str(uuid.uuid4()), node_ref="main.foo")
        assert "Failed to fetch variable flow" in result

    @pytest.mark.anyio
    async def test_get_codebase_summary_error(self, _patch_user_id) -> None:
        with patch("app.mcp_server.get_db_session", side_effect=RuntimeError("DB error")):
            result = await _get_codebase_summary(collection_id=str(uuid.uuid4()))
        assert "Failed to fetch codebase summary" in result


class TestTaintToolErrors:
    @pytest.mark.anyio
    async def test_find_taint_sources_error(self, _patch_user_id) -> None:
        with patch("app.mcp_server.get_db_session", side_effect=RuntimeError("DB error")):
            result = await _find_taint_sources(collection_id=str(uuid.uuid4()))
        assert "Failed to find taint sources" in result

    @pytest.mark.anyio
    async def test_find_taint_sinks_error(self, _patch_user_id) -> None:
        with patch("app.mcp_server.get_db_session", side_effect=RuntimeError("DB error")):
            result = await _find_taint_sinks(collection_id=str(uuid.uuid4()))
        assert "Failed to find taint sinks" in result

    @pytest.mark.anyio
    async def test_find_taint_flows_error(self, _patch_user_id) -> None:
        with patch("app.mcp_server.get_db_session", side_effect=RuntimeError("DB error")):
            result = await _find_taint_flows(collection_id=str(uuid.uuid4()))
        assert "Failed to find taint flows" in result


class TestExportSarif:
    @pytest.mark.anyio
    async def test_error_handling(self, _patch_user_id) -> None:
        with patch("app.mcp_server.get_db_session", side_effect=RuntimeError("DB error")):
            result = await _export_sarif(collection_id=str(uuid.uuid4()))
        assert "Failed to export SARIF" in result


class TestRefreshTwin:
    @pytest.mark.anyio
    async def test_error_handling(self, _patch_user_id) -> None:
        with patch("app.mcp_server.get_db_session", side_effect=RuntimeError("DB error")):
            result = await _refresh_twin(collection_id=str(uuid.uuid4()))
        assert "Failed to refresh twin" in result


class TestValidationDashboard:
    @pytest.mark.anyio
    async def test_error_handling(self) -> None:
        with patch("app.mcp_server.get_db_session", side_effect=RuntimeError("DB error")):
            result = await _validation_dashboard()
        assert "Failed to fetch validation dashboard" in result


class TestQueryTwinCypher:
    @pytest.mark.anyio
    async def test_error_handling(self) -> None:
        with patch("app.mcp_server.get_db_session", side_effect=RuntimeError("DB error")):
            result = await _query_twin_cypher(
                scenario_id=str(uuid.uuid4()), query="MATCH (n) RETURN n"
            )
        assert "Cypher query failed" in result


class TestGetTwinTimeline:
    @pytest.mark.anyio
    async def test_error_handling(self, _patch_user_id) -> None:
        with patch("app.mcp_server.get_db_session", side_effect=RuntimeError("DB error")):
            result = await _get_twin_timeline(collection_id=str(uuid.uuid4()))
        assert "Failed to fetch twin timeline" in result
