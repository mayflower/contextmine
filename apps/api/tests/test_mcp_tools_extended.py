"""Extended tests for MCP server tool implementations.

Covers the DB-heavy MCP tools with mocked sessions returning realistic data.
Targets: get_markdown, graph_rag, get_arc42, research_*, graph_neighborhood,
trace_path, code navigation tools with symbol data, and twin-related tools.
"""

import json
import uuid
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import app.mcp_server as mcp_mod
import pytest

# Unwrap FunctionTool -> coroutine
_list_collections = mcp_mod.list_collections.fn
_list_documents = mcp_mod.list_documents.fn
_get_context_markdown = mcp_mod.get_context_markdown.fn
_code_outline = mcp_mod.code_outline.fn
_code_find_symbol = mcp_mod.code_find_symbol.fn
_code_definition = mcp_mod.code_definition.fn
_code_references = mcp_mod.code_references.fn
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
_create_intent = mcp_mod.mcp_create_architecture_intent.fn
_approve_intent = mcp_mod.mcp_approve_architecture_intent.fn
_list_methods = mcp_mod.mcp_list_methods.fn
_list_calls = mcp_mod.mcp_list_calls.fn
_get_cfg = mcp_mod.mcp_get_cfg.fn
_get_variable_flow = mcp_mod.mcp_get_variable_flow.fn
_get_codebase_summary = mcp_mod.mcp_get_codebase_summary.fn
_find_taint_sources = mcp_mod.mcp_find_taint_sources.fn
_find_taint_sinks = mcp_mod.mcp_find_taint_sinks.fn
_find_taint_flows = mcp_mod.mcp_find_taint_flows.fn
_export_sarif = mcp_mod.mcp_export_sarif.fn


def _mock_db_session(mock_db):
    """Create a patched get_db_session that yields mock_db."""
    ctx = patch("app.mcp_server.get_db_session")
    mock_session = ctx.start()
    mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db)
    mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
    return ctx


@pytest.fixture
def _patch_user_id():
    """Patch get_current_user_id to return None."""
    with patch("app.mcp_server.get_current_user_id", return_value=None):
        yield


@pytest.fixture
def _patch_user_id_with_uuid():
    """Patch get_current_user_id to return a real UUID."""
    uid = uuid.uuid4()
    with patch("app.mcp_server.get_current_user_id", return_value=uid):
        yield uid


def _make_symbol(
    *,
    name="foo",
    kind_value="function",
    start_line=4,
    end_line=5,
    signature="def foo()",
    parent_name=None,
    qualified_name=None,
    meta=None,
):
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
    uri="git://repo/src/main.py",
    content_markdown="line1\nline2\nline3\ndef foo():\n    pass\nline6",
):
    doc = MagicMock()
    doc.id = uuid.uuid4()
    doc.uri = uri
    doc.content_markdown = content_markdown
    return doc


def _make_knowledge_node(
    *,
    name="test_rule",
    kind_value="business_rule",
    natural_key="rule:test",
    meta=None,
):
    node = MagicMock()
    node.id = uuid.uuid4()
    node.name = name
    node.kind = MagicMock(value=kind_value)
    node.natural_key = natural_key
    node.meta = meta or {}
    return node


# ===========================================================================
# get_markdown (raw mode and LLM assembled)
# ===========================================================================


class TestGetContextMarkdownRaw:
    """Test _get_raw_chunks path."""

    @pytest.mark.anyio
    async def test_raw_mode_no_results(self, _patch_user_id) -> None:
        """Test raw mode returns no results message."""
        mock_search_response = MagicMock()
        mock_search_response.results = []

        with (
            patch("app.mcp_server.parse_embedding_model_spec", side_effect=Exception("no spec")),
            patch("app.mcp_server.FakeEmbedder") as mock_embedder_cls,
            patch("app.mcp_server.hybrid_search", new_callable=AsyncMock) as mock_search,
        ):
            mock_embedder = MagicMock()
            mock_embedder.embed_batch = AsyncMock(return_value=MagicMock(embeddings=[[0.1] * 10]))
            mock_embedder_cls.return_value = mock_embedder
            mock_search.return_value = mock_search_response

            result = await _get_context_markdown(query="test query", raw=True)

        assert "# No Results" in result
        assert "test query" in result

    @pytest.mark.anyio
    async def test_raw_mode_with_results(self, _patch_user_id) -> None:
        """Test raw mode returns formatted chunks."""

        @dataclass
        class FakeSearchResult:
            uri: str
            title: str
            content: str

        mock_search_response = MagicMock()
        mock_search_response.results = [
            FakeSearchResult(uri="doc://a", title="Doc A", content="Content A"),
            FakeSearchResult(uri="doc://b", title="Doc B", content="Content B"),
        ]

        with (
            patch("app.mcp_server.parse_embedding_model_spec", side_effect=Exception("no spec")),
            patch("app.mcp_server.FakeEmbedder") as mock_embedder_cls,
            patch("app.mcp_server.hybrid_search", new_callable=AsyncMock) as mock_search,
        ):
            mock_embedder = MagicMock()
            mock_embedder.embed_batch = AsyncMock(return_value=MagicMock(embeddings=[[0.1] * 10]))
            mock_embedder_cls.return_value = mock_embedder
            mock_search.return_value = mock_search_response

            result = await _get_context_markdown(query="test query", raw=True)

        assert "# Search Results" in result
        assert "Doc A" in result
        assert "Content A" in result
        assert "## Sources" in result

    @pytest.mark.anyio
    async def test_raw_mode_with_topic_filter(self, _patch_user_id) -> None:
        """Test raw mode with topic filter."""

        @dataclass
        class FakeSearchResult:
            uri: str
            title: str
            content: str

        mock_search_response = MagicMock()
        mock_search_response.results = [
            FakeSearchResult(uri="doc://a", title="Auth Flow", content="Auth content"),
            FakeSearchResult(uri="doc://b", title="Payment Guide", content="Payment content"),
        ]

        with (
            patch("app.mcp_server.parse_embedding_model_spec", side_effect=Exception("no spec")),
            patch("app.mcp_server.FakeEmbedder") as mock_embedder_cls,
            patch("app.mcp_server.hybrid_search", new_callable=AsyncMock) as mock_search,
        ):
            mock_embedder = MagicMock()
            mock_embedder.embed_batch = AsyncMock(return_value=MagicMock(embeddings=[[0.1] * 10]))
            mock_embedder_cls.return_value = mock_embedder
            mock_search.return_value = mock_search_response

            result = await _get_context_markdown(query="test query", raw=True, topic="auth")

        assert "Auth Flow" in result
        assert "Payment" not in result

    @pytest.mark.anyio
    async def test_assembled_mode_calls_assemble_context(self, _patch_user_id) -> None:
        """Test the LLM-assembled path."""
        mock_response = MagicMock()
        mock_response.markdown = "# Assembled\n\nGreat content here."

        with patch(
            "app.mcp_server.assemble_context",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await _get_context_markdown(query="test query")

        assert "# Assembled" in result
        assert "Great content" in result

    @pytest.mark.anyio
    async def test_assembled_mode_handles_exception(self, _patch_user_id) -> None:
        """Test error handling in assembled mode."""
        with patch(
            "app.mcp_server.assemble_context",
            new_callable=AsyncMock,
            side_effect=RuntimeError("LLM failed"),
        ):
            result = await _get_context_markdown(query="test query")

        assert "# Error" in result
        assert "LLM failed" in result


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
        ctx = _mock_db_session(mock_db)
        try:
            with patch(
                "contextmine_core.graphrag.graph_neighborhood",
                new=AsyncMock(return_value=mock_result),
            ):
                result = await _graph_neighborhood(node_id=str(uuid.uuid4()))
        finally:
            ctx.stop()

        assert "# No Neighborhood Found" in result

    @pytest.mark.anyio
    async def test_with_connections(self) -> None:
        mock_entity = MagicMock()
        mock_entity.kind = "file"
        mock_entity.node_id = uuid.uuid4()
        mock_result = MagicMock()
        mock_result.entities = [mock_entity]
        mock_result.to_markdown.return_value = "# Neighborhood\n\n- file: a.py"

        mock_db = AsyncMock()
        ctx = _mock_db_session(mock_db)
        try:
            with patch(
                "contextmine_core.graphrag.graph_neighborhood",
                new=AsyncMock(return_value=mock_result),
            ):
                result = await _graph_neighborhood(node_id=str(uuid.uuid4()))
        finally:
            ctx.stop()

        assert "# Neighborhood" in result

    @pytest.mark.anyio
    async def test_exception_handled(self) -> None:
        mock_db = AsyncMock()
        ctx = _mock_db_session(mock_db)
        try:
            with patch(
                "contextmine_core.graphrag.graph_neighborhood",
                new=AsyncMock(side_effect=RuntimeError("boom")),
            ):
                result = await _graph_neighborhood(node_id=str(uuid.uuid4()))
        finally:
            ctx.stop()

        assert "# Error" in result
        assert "boom" in result


# ===========================================================================
# trace_path
# ===========================================================================


class TestTracePath:
    @pytest.mark.anyio
    async def test_invalid_from_id(self) -> None:
        result = await _trace_path(from_node_id="bad", to_node_id=str(uuid.uuid4()))
        assert "# Error" in result
        assert "Invalid node ID" in result

    @pytest.mark.anyio
    async def test_no_path_found(self) -> None:
        mock_result = MagicMock()
        mock_result.entities = []

        mock_db = AsyncMock()
        ctx = _mock_db_session(mock_db)
        try:
            with patch(
                "contextmine_core.graphrag.trace_path",
                new=AsyncMock(return_value=mock_result),
            ):
                result = await _trace_path(
                    from_node_id=str(uuid.uuid4()),
                    to_node_id=str(uuid.uuid4()),
                )
        finally:
            ctx.stop()

        assert "# No Path Found" in result

    @pytest.mark.anyio
    async def test_path_found(self) -> None:
        mock_entity = MagicMock()
        mock_entity.kind = "file"
        mock_result = MagicMock()
        mock_result.entities = [mock_entity]
        mock_result.to_markdown.return_value = "# Path\n\nA -> B"

        mock_db = AsyncMock()
        ctx = _mock_db_session(mock_db)
        try:
            with patch(
                "contextmine_core.graphrag.trace_path",
                new=AsyncMock(return_value=mock_result),
            ):
                result = await _trace_path(
                    from_node_id=str(uuid.uuid4()),
                    to_node_id=str(uuid.uuid4()),
                )
        finally:
            ctx.stop()

        assert "# Path" in result


# ===========================================================================
# graph_rag
# ===========================================================================


class TestGraphRag:
    @pytest.mark.anyio
    async def test_invalid_twin_scope(self, _patch_user_id) -> None:
        result = await _graph_rag(query="test", twin_scope="invalid_scope")
        assert "# Error" in result
        assert "Invalid twin_scope" in result

    @pytest.mark.anyio
    async def test_context_mode_markdown(self, _patch_user_id) -> None:
        """Test context-only mode returning markdown."""
        mock_context = MagicMock()
        mock_context.entities = []
        mock_context.edges = []
        mock_context.to_markdown.return_value = (
            "# GraphRAG Context: test\n\nFound 2 communities, 5 entities, 3 citations.\n"
        )

        mock_db = AsyncMock()
        ctx = _mock_db_session(mock_db)
        try:
            with patch(
                "contextmine_core.graphrag.graph_rag_context",
                new=AsyncMock(return_value=mock_context),
            ):
                result = await _graph_rag(query="test")
        finally:
            ctx.stop()

        assert "GraphRAG" in result

    @pytest.mark.anyio
    async def test_context_mode_no_results(self, _patch_user_id) -> None:
        """Test context mode with empty results."""
        mock_context = MagicMock()
        mock_context.entities = []
        mock_context.edges = []
        mock_context.to_markdown.return_value = (
            "# GraphRAG Context: test\n\nFound 0 communities, 0 entities, 0 citations.\n"
        )

        mock_db = AsyncMock()
        ctx = _mock_db_session(mock_db)
        try:
            with patch(
                "contextmine_core.graphrag.graph_rag_context",
                new=AsyncMock(return_value=mock_context),
            ):
                result = await _graph_rag(query="test")
        finally:
            ctx.stop()

        assert "# No Results" in result

    @pytest.mark.anyio
    async def test_context_mode_json(self, _patch_user_id) -> None:
        """Test context mode with JSON format."""
        mock_context = MagicMock()
        mock_context.entities = []
        mock_context.edges = []
        mock_context.to_dict.return_value = {"communities": [], "entities": []}

        mock_db = AsyncMock()
        ctx = _mock_db_session(mock_db)
        try:
            with patch(
                "contextmine_core.graphrag.graph_rag_context",
                new=AsyncMock(return_value=mock_context),
            ):
                result = await _graph_rag(query="test", format="json")
        finally:
            ctx.stop()

        parsed = json.loads(result)
        assert "communities" in parsed

    @pytest.mark.anyio
    async def test_context_mode_json_rebuild_mode(self, _patch_user_id) -> None:
        """Test JSON output includes rebuild sections when rebuild_mode=True."""
        mock_entity = MagicMock()
        mock_entity.kind = "api_endpoint"
        mock_context = MagicMock()
        mock_context.entities = [mock_entity]
        mock_context.edges = []
        mock_context.to_dict.return_value = {
            "communities": [],
            "entities": [{"kind": "api_endpoint"}],
        }

        mock_db = AsyncMock()
        ctx = _mock_db_session(mock_db)
        try:
            with patch(
                "contextmine_core.graphrag.graph_rag_context",
                new=AsyncMock(return_value=mock_context),
            ):
                result = await _graph_rag(query="test", format="json", rebuild_mode=True)
        finally:
            ctx.stop()

        parsed = json.loads(result)
        assert "rebuild_sections" in parsed
        assert "System Boundaries" in parsed["rebuild_sections"]
        assert "Interfaces" in parsed["rebuild_sections"]

    @pytest.mark.anyio
    async def test_exception_handled(self, _patch_user_id) -> None:
        with patch(
            "contextmine_core.graphrag.graph_rag_context",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ):
            mock_db = AsyncMock()
            ctx = _mock_db_session(mock_db)
            try:
                result = await _graph_rag(query="test")
            finally:
                ctx.stop()

        assert "# Error" in result
        assert "boom" in result


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
                new=AsyncMock(return_value=[]),
            ),
        ):
            ctx = _mock_db_session(mock_db)
            try:
                result = await _research_validation(code_path="auth.py")
            finally:
                ctx.stop()

        assert "# No Collections Available" in result

    @pytest.mark.anyio
    async def test_no_rules_found(self, _patch_user_id) -> None:
        mock_db = AsyncMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_db.execute = AsyncMock(return_value=mock_result)

        with patch(
            "contextmine_core.search.get_accessible_collection_ids",
            new=AsyncMock(return_value=[uuid.uuid4()]),
        ):
            ctx = _mock_db_session(mock_db)
            try:
                result = await _research_validation(code_path="auth.py")
            finally:
                ctx.stop()

        assert (
            "No Validation Rules Found" in result
            or "No Business Rules" in result
            or "No business rules" in result
        )

    @pytest.mark.anyio
    async def test_with_matching_rules(self, _patch_user_id) -> None:
        rule = _make_knowledge_node(
            name="must_be_authenticated",
            meta={
                "category": "authorization",
                "severity": "error",
                "natural_language": "User must be authenticated",
                "container_name": "auth_service",
                "predicate": "user.is_authenticated",
                "failure": "raise PermissionDenied",
            },
        )

        mock_db = AsyncMock()
        call_count = 0

        async def _mock_execute(stmt):
            nonlocal call_count
            call_count += 1
            mock_scalars = MagicMock()
            mock_result = MagicMock()
            mock_result.scalars.return_value = mock_scalars
            if call_count == 1:
                mock_scalars.all.return_value = [rule]
            else:
                mock_scalars.all.return_value = []
            return mock_result

        mock_db.execute = _mock_execute

        with patch(
            "contextmine_core.search.get_accessible_collection_ids",
            new=AsyncMock(return_value=[uuid.uuid4()]),
        ):
            ctx = _mock_db_session(mock_db)
            try:
                result = await _research_validation(code_path="auth")
            finally:
                ctx.stop()

        assert "Validation Rules" in result or "Business Rules" in result

    @pytest.mark.anyio
    async def test_exception_handled(self, _patch_user_id) -> None:
        with patch(
            "contextmine_core.search.get_accessible_collection_ids",
            new=AsyncMock(side_effect=RuntimeError("DB down")),
        ):
            mock_db = AsyncMock()
            ctx = _mock_db_session(mock_db)
            try:
                result = await _research_validation(code_path="auth.py")
            finally:
                ctx.stop()

        assert "# Error" in result


# ===========================================================================
# research_data_model
# ===========================================================================


class TestResearchDataModel:
    @pytest.mark.anyio
    async def test_no_collections(self, _patch_user_id) -> None:
        mock_db = AsyncMock()

        with patch(
            "contextmine_core.search.get_accessible_collection_ids",
            new=AsyncMock(return_value=[]),
        ):
            ctx = _mock_db_session(mock_db)
            try:
                result = await _research_data_model(entity="users")
            finally:
                ctx.stop()

        assert "# No Collections Available" in result

    @pytest.mark.anyio
    async def test_no_tables_found(self, _patch_user_id) -> None:
        mock_db = AsyncMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        with patch(
            "contextmine_core.search.get_accessible_collection_ids",
            new=AsyncMock(return_value=[uuid.uuid4()]),
        ):
            ctx = _mock_db_session(mock_db)
            try:
                result = await _research_data_model(entity="users")
            finally:
                ctx.stop()

        assert "No Data Model Found" in result or "No tables" in result

    @pytest.mark.anyio
    async def test_exception_handled(self, _patch_user_id) -> None:
        with patch(
            "contextmine_core.search.get_accessible_collection_ids",
            new=AsyncMock(side_effect=RuntimeError("DB down")),
        ):
            mock_db = AsyncMock()
            ctx = _mock_db_session(mock_db)
            try:
                result = await _research_data_model(entity="users")
            finally:
                ctx.stop()

        assert "# Error" in result


# ===========================================================================
# research_architecture
# ===========================================================================


class TestResearchArchitecture:
    @pytest.mark.anyio
    async def test_no_collections(self, _patch_user_id) -> None:
        mock_db = AsyncMock()

        with patch(
            "contextmine_core.search.get_accessible_collection_ids",
            new=AsyncMock(return_value=[]),
        ):
            ctx = _mock_db_session(mock_db)
            try:
                result = await _research_architecture(topic="api")
            finally:
                ctx.stop()

        assert "# No Collections Available" in result

    @pytest.mark.anyio
    async def test_api_topic_with_endpoints(self, _patch_user_id) -> None:
        endpoint_node = _make_knowledge_node(
            name="GET /api/users",
            kind_value="api_endpoint",
            meta={"method": "GET", "path": "/api/users"},
        )

        mock_db = AsyncMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [endpoint_node]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_db.execute = AsyncMock(return_value=mock_result)

        with patch(
            "contextmine_core.search.get_accessible_collection_ids",
            new=AsyncMock(return_value=[uuid.uuid4()]),
        ):
            ctx = _mock_db_session(mock_db)
            try:
                result = await _research_architecture(topic="api")
            finally:
                ctx.stop()

        assert "Architecture: api" in result
        assert "API Endpoints" in result

    @pytest.mark.anyio
    async def test_unknown_topic_returns_fallback(self, _patch_user_id) -> None:
        mock_db = AsyncMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_db.execute = AsyncMock(return_value=mock_result)

        with patch(
            "contextmine_core.search.get_accessible_collection_ids",
            new=AsyncMock(return_value=[uuid.uuid4()]),
        ):
            ctx = _mock_db_session(mock_db)
            try:
                result = await _research_architecture(topic="quantum_computing")
            finally:
                ctx.stop()

        assert "No specific architecture information" in result

    @pytest.mark.anyio
    async def test_exception_handled(self, _patch_user_id) -> None:
        with patch(
            "contextmine_core.search.get_accessible_collection_ids",
            new=AsyncMock(side_effect=RuntimeError("DB down")),
        ):
            mock_db = AsyncMock()
            ctx = _mock_db_session(mock_db)
            try:
                result = await _research_architecture(topic="api")
            finally:
                ctx.stop()

        assert "# Error" in result


# ===========================================================================
# code_outline with symbols from DB
# ===========================================================================


class TestCodeOutlineWithSymbols:
    @pytest.mark.anyio
    async def test_outline_with_top_level_symbols(self) -> None:
        doc = _make_document()
        sym1 = _make_symbol(name="MyClass", kind_value="class", start_line=1, end_line=20)
        sym2 = _make_symbol(name="my_func", kind_value="function", start_line=22, end_line=30)

        mock_db = AsyncMock()
        call_count = 0

        async def _mock_execute(stmt):
            nonlocal call_count
            call_count += 1
            mock_result = MagicMock()
            mock_scalars = MagicMock()
            if call_count == 1:
                mock_result.scalar_one_or_none.return_value = doc
            else:
                mock_scalars.all.return_value = [sym1, sym2]
                mock_result.scalars.return_value = mock_scalars
            return mock_result

        mock_db.execute = _mock_execute

        ctx = _mock_db_session(mock_db)
        try:
            result = await _code_outline(file_path="src/main.py")
        finally:
            ctx.stop()

        assert "# Outline" in result
        assert "MyClass" in result
        assert "my_func" in result

    @pytest.mark.anyio
    async def test_outline_no_document(self) -> None:
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        ctx = _mock_db_session(mock_db)
        try:
            result = await _code_outline(file_path="nonexistent.py")
        finally:
            ctx.stop()

        assert "No symbols found" in result or "Not Found" in result


# ===========================================================================
# code_find_symbol
# ===========================================================================


class TestCodeFindSymbol:
    @pytest.mark.anyio
    async def test_find_existing_symbol(self) -> None:
        doc = _make_document(content_markdown="line1\nline2\nline3\ndef foo():\n    pass\nline6")
        sym = _make_symbol(name="foo", start_line=4, end_line=5, signature="def foo()")

        mock_db = AsyncMock()
        call_count = 0

        async def _mock_execute(stmt):
            nonlocal call_count
            call_count += 1
            mock_result = MagicMock()
            if call_count == 1:
                mock_result.scalar_one_or_none.return_value = doc
            else:
                mock_result.scalar_one_or_none.return_value = sym
            return mock_result

        mock_db.execute = _mock_execute

        ctx = _mock_db_session(mock_db)
        try:
            result = await _code_find_symbol(file_path="src/main.py", name="foo")
        finally:
            ctx.stop()

        assert "foo" in result
        assert "function" in result

    @pytest.mark.anyio
    async def test_symbol_not_found(self) -> None:
        doc = _make_document()

        mock_db = AsyncMock()
        call_count = 0

        async def _mock_execute(stmt):
            nonlocal call_count
            call_count += 1
            mock_result = MagicMock()
            if call_count == 1:
                mock_result.scalar_one_or_none.return_value = doc
            else:
                mock_result.scalar_one_or_none.return_value = None
            return mock_result

        mock_db.execute = _mock_execute

        ctx = _mock_db_session(mock_db)
        try:
            result = await _code_find_symbol(file_path="src/main.py", name="missing")
        finally:
            ctx.stop()

        assert "Symbol Not Found" in result


# ===========================================================================
# code_definition
# ===========================================================================


class TestCodeDefinition:
    @pytest.mark.anyio
    async def test_definition_found(self) -> None:
        doc = _make_document()
        sym = _make_symbol(name="foo", start_line=3, end_line=5, signature="def foo()")

        mock_db = AsyncMock()
        call_count = 0

        async def _mock_execute(stmt):
            nonlocal call_count
            call_count += 1
            mock_result = MagicMock()
            mock_scalars = MagicMock()
            if call_count == 1:
                mock_result.scalar_one_or_none.return_value = doc
            else:
                mock_scalars.first.return_value = sym
                mock_result.scalars.return_value = mock_scalars
            return mock_result

        mock_db.execute = _mock_execute

        ctx = _mock_db_session(mock_db)
        try:
            result = await _code_definition(file_path="src/main.py", line=4, column=0)
        finally:
            ctx.stop()

        assert "# Definition Found" in result
        assert "foo" in result

    @pytest.mark.anyio
    async def test_no_document(self) -> None:
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        ctx = _mock_db_session(mock_db)
        try:
            result = await _code_definition(file_path="nope.py", line=1, column=0)
        finally:
            ctx.stop()

        assert "# Document Not Found" in result


# ===========================================================================
# code_references
# ===========================================================================


class TestCodeReferences:
    @pytest.mark.anyio
    async def test_no_document(self) -> None:
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        ctx = _mock_db_session(mock_db)
        try:
            result = await _code_references(file_path="nope.py", line=1, column=0)
        finally:
            ctx.stop()

        assert "# Document Not Found" in result

    @pytest.mark.anyio
    async def test_no_symbol_at_line(self) -> None:
        doc = _make_document()
        mock_db = AsyncMock()
        call_count = 0

        async def _mock_execute(stmt):
            nonlocal call_count
            call_count += 1
            mock_result = MagicMock()
            mock_scalars = MagicMock()
            if call_count == 1:
                mock_result.scalar_one_or_none.return_value = doc
            else:
                mock_scalars.first.return_value = None
                mock_result.scalars.return_value = mock_scalars
            return mock_result

        mock_db.execute = _mock_execute

        ctx = _mock_db_session(mock_db)
        try:
            result = await _code_references(file_path="src/main.py", line=999, column=0)
        finally:
            ctx.stop()

        assert "# No Symbol Found" in result


# ===========================================================================
# create_architecture_intent
# ===========================================================================


class TestCreateArchitectureIntent:
    @pytest.mark.anyio
    async def test_unauthenticated(self) -> None:
        with patch("app.mcp_server.get_current_user_id", return_value=None):
            result = await _create_intent(
                scenario_id=str(uuid.uuid4()),
                action="EXTRACT_DOMAIN",
                target_type="node",
                target_id="some-id",
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
                target_id="some-id",
                expected_scenario_version=1,
            )
        finally:
            ctx.stop()

        assert "Scenario not found" in result


# ===========================================================================
# approve_architecture_intent
# ===========================================================================


class TestApproveArchitectureIntent:
    @pytest.mark.anyio
    async def test_unauthenticated(self) -> None:
        with patch("app.mcp_server.get_current_user_id", return_value=None):
            result = await _approve_intent(
                scenario_id=str(uuid.uuid4()),
                intent_id=str(uuid.uuid4()),
            )
        assert "Authentication required" in result


# ===========================================================================
# list_documents with access control
# ===========================================================================


class TestListDocumentsAccess:
    @pytest.mark.anyio
    async def test_access_denied_private(self, _patch_user_id) -> None:
        from contextmine_core import CollectionVisibility

        coll_mock = MagicMock()
        coll_mock.id = uuid.uuid4()
        coll_mock.visibility = CollectionVisibility.PRIVATE
        coll_mock.owner_user_id = uuid.uuid4()

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = coll_mock
        mock_db.execute = AsyncMock(return_value=mock_result)

        ctx = _mock_db_session(mock_db)
        try:
            result = await _list_documents(collection_id=str(coll_mock.id))
        finally:
            ctx.stop()

        assert "Access denied" in result

    @pytest.mark.anyio
    async def test_documents_with_topic_filter_no_matches(self, _patch_user_id) -> None:
        from contextmine_core import CollectionVisibility

        coll_mock = MagicMock()
        coll_mock.id = uuid.uuid4()
        coll_mock.name = "Test Collection"
        coll_mock.visibility = CollectionVisibility.GLOBAL
        coll_mock.owner_user_id = None

        mock_db = AsyncMock()
        call_count = 0

        async def _mock_execute(stmt):
            nonlocal call_count
            call_count += 1
            mock_result = MagicMock()
            if call_count == 1:
                mock_result.scalar_one_or_none.return_value = coll_mock
            else:
                mock_result.all.return_value = []
            return mock_result

        mock_db.execute = _mock_execute

        ctx = _mock_db_session(mock_db)
        try:
            result = await _list_documents(collection_id=str(coll_mock.id), topic="nonexistent")
        finally:
            ctx.stop()

        assert "No Documents Found" in result
        assert "nonexistent" in result
