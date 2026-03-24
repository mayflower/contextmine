"""Comprehensive tests for the research agent.

Covers:
- Helper functions (_escape_like_pattern)
- Pydantic input schemas (HybridSearchInput, OpenSpanInput, FinalizeInput)
- AgentState type
- AgentConfig dataclass
- create_tools() factory - each tool in isolation
- graph_pack scoring logic
- finalize tool
- summarize_evidence tool
- ResearchAgent._build_system_prompt
- ResearchAgent._build_graph structure
- verify_node routing logic
- run_research convenience function
- Error handling paths
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from contextmine_core.research.agent import (
    AgentConfig,
    AgentState,
    FinalizeInput,
    HybridSearchInput,
    OpenSpanInput,
    ResearchAgent,
    _escape_like_pattern,
    create_tools,
)
from contextmine_core.research.run import Evidence, ResearchRun, RunStatus


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_run(
    question: str = "How does auth work?",
    scope: str | None = None,
    budget_steps: int = 10,
) -> ResearchRun:
    """Create a ResearchRun for test use."""
    return ResearchRun.create(question=question, scope=scope, budget_steps=budget_steps)


def _make_evidence(
    id: str = "ev-abc12345-001",
    file_path: str = "src/auth.py",
    start_line: int = 10,
    end_line: int = 20,
    content: str = "def authenticate(user):\n    pass",
    reason: str = "Matched query",
    provenance: str = "hybrid",
    score: float | None = 0.85,
    symbol_kind: str | None = None,
    symbol_id: str | None = None,
) -> Evidence:
    return Evidence(
        id=id,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        content=content,
        reason=reason,
        provenance=provenance,
        score=score,
        symbol_kind=symbol_kind,
        symbol_id=symbol_id,
    )


def _make_run_holder(
    run: ResearchRun | None = None,
) -> dict[str, Any]:
    if run is None:
        run = _make_run()
    return {
        "run": run,
        "pending_answer": None,
        "confidence": 0.8,
    }


# ---------------------------------------------------------------------------
# _escape_like_pattern
# ---------------------------------------------------------------------------


class TestEscapeLikePattern:
    """Tests for the SQL LIKE pattern escaper."""

    def test_no_special_chars(self) -> None:
        assert _escape_like_pattern("hello") == "hello"

    def test_escape_percent(self) -> None:
        assert _escape_like_pattern("100%") == "100\\%"

    def test_escape_underscore(self) -> None:
        assert _escape_like_pattern("my_func") == "my\\_func"

    def test_escape_backslash(self) -> None:
        assert _escape_like_pattern("path\\to") == "path\\\\to"

    def test_escape_all_combined(self) -> None:
        result = _escape_like_pattern("a\\b%c_d")
        assert result == "a\\\\b\\%c\\_d"

    def test_empty_string(self) -> None:
        assert _escape_like_pattern("") == ""

    def test_only_special_chars(self) -> None:
        assert _escape_like_pattern("\\%_") == "\\\\\\%\\_"


# ---------------------------------------------------------------------------
# Pydantic Input Schemas
# ---------------------------------------------------------------------------


class TestInputSchemas:
    """Tests for tool input Pydantic models."""

    def test_hybrid_search_input_defaults(self) -> None:
        inp = HybridSearchInput(query="test")
        assert inp.query == "test"
        assert inp.k == 10

    def test_hybrid_search_input_custom_k(self) -> None:
        inp = HybridSearchInput(query="auth", k=5)
        assert inp.k == 5

    def test_open_span_input(self) -> None:
        inp = OpenSpanInput(file_path="src/main.py", start_line=1, end_line=50)
        assert inp.file_path == "src/main.py"
        assert inp.start_line == 1
        assert inp.end_line == 50

    def test_finalize_input_defaults(self) -> None:
        inp = FinalizeInput(answer="The answer is 42")
        assert inp.answer == "The answer is 42"
        assert inp.confidence == 0.8

    def test_finalize_input_custom_confidence(self) -> None:
        inp = FinalizeInput(answer="answer", confidence=0.95)
        assert inp.confidence == 0.95


# ---------------------------------------------------------------------------
# AgentConfig
# ---------------------------------------------------------------------------


class TestAgentConfig:
    """Tests for agent configuration."""

    def test_default_config(self) -> None:
        config = AgentConfig()
        assert config.max_steps == 10
        assert config.store_artifacts is True
        assert config.max_verification_retries == 2

    def test_custom_config(self) -> None:
        config = AgentConfig(
            max_steps=5,
            store_artifacts=False,
            max_verification_retries=3,
        )
        assert config.max_steps == 5
        assert config.store_artifacts is False
        assert config.max_verification_retries == 3


# ---------------------------------------------------------------------------
# create_tools() returns correct tool list
# ---------------------------------------------------------------------------


class TestCreateTools:
    """Tests for create_tools factory function."""

    def test_returns_list_of_tools(self) -> None:
        run_holder = _make_run_holder()
        tools = create_tools(run_holder)
        assert isinstance(tools, list)
        assert len(tools) == 18  # 18 tools total

    def test_tool_names(self) -> None:
        run_holder = _make_run_holder()
        tools = create_tools(run_holder)
        names = {t.name for t in tools}
        expected = {
            "hybrid_search",
            "open_span",
            "finalize",
            "summarize_evidence",
            "goto_definition",
            "find_references",
            "get_signature",
            "symbol_outline",
            "symbol_find",
            "symbol_enclosing",
            "symbol_callers",
            "symbol_callees",
            "graph_expand",
            "graph_pack",
            "graph_trace",
            "graphrag_search",
            "kg_neighborhood",
            "kg_path",
        }
        assert names == expected

    def test_tools_have_invoke(self) -> None:
        run_holder = _make_run_holder()
        tools = create_tools(run_holder)
        for tool in tools:
            assert hasattr(tool, "ainvoke"), f"Tool {tool.name} missing ainvoke"


# ---------------------------------------------------------------------------
# finalize tool
# ---------------------------------------------------------------------------


class TestFinalizeTool:
    """Tests for the finalize tool."""

    @pytest.mark.anyio
    async def test_finalize_sets_pending_answer(self) -> None:
        run_holder = _make_run_holder()
        tools = create_tools(run_holder)
        finalize_tool = next(t for t in tools if t.name == "finalize")

        result = await finalize_tool.ainvoke({"answer": "The answer", "confidence": 0.9})

        assert run_holder["pending_answer"] == "The answer"
        assert run_holder["confidence"] == 0.9
        assert "confidence: 0.9" in result

    @pytest.mark.anyio
    async def test_finalize_default_confidence(self) -> None:
        run_holder = _make_run_holder()
        tools = create_tools(run_holder)
        finalize_tool = next(t for t in tools if t.name == "finalize")

        result = await finalize_tool.ainvoke({"answer": "My answer"})

        assert run_holder["pending_answer"] == "My answer"
        assert "submitted for verification" in result


# ---------------------------------------------------------------------------
# graph_pack tool - pure logic, no DB
# ---------------------------------------------------------------------------


class TestGraphPackTool:
    """Tests for the graph_pack evidence ranking tool."""

    @pytest.mark.anyio
    async def test_graph_pack_no_evidence(self) -> None:
        run_holder = _make_run_holder()
        tools = create_tools(run_holder)
        pack_tool = next(t for t in tools if t.name == "graph_pack")

        result = await pack_tool.ainvoke({"target_count": 5})
        assert "No evidence collected" in result

    @pytest.mark.anyio
    async def test_graph_pack_scores_by_symbol_kind(self) -> None:
        run = _make_run()
        run.add_evidence(
            _make_evidence(
                id="ev-abc-001",
                symbol_kind="class",
                provenance="symbol_index",
            )
        )
        run.add_evidence(
            _make_evidence(
                id="ev-abc-002",
                symbol_kind="variable",
                provenance="symbol_index",
            )
        )
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        pack_tool = next(t for t in tools if t.name == "graph_pack")

        result = await pack_tool.ainvoke({"target_count": 10})

        # Class (5.0) should rank higher than variable (1.0)
        class_pos = result.index("ev-abc-001")
        var_pos = result.index("ev-abc-002")
        assert class_pos < var_pos

    @pytest.mark.anyio
    async def test_graph_pack_respects_target_count(self) -> None:
        run = _make_run()
        for i in range(5):
            run.add_evidence(
                _make_evidence(
                    id=f"ev-abc-{i:03d}",
                    provenance="hybrid",
                )
            )
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        pack_tool = next(t for t in tools if t.name == "graph_pack")

        result = await pack_tool.ainvoke({"target_count": 2})

        assert "Top 2 evidence items" in result

    @pytest.mark.anyio
    async def test_graph_pack_provenance_scoring(self) -> None:
        run = _make_run()
        run.add_evidence(
            _make_evidence(
                id="ev-abc-001",
                provenance="manual",
                score=None,
                symbol_kind=None,
            )
        )
        run.add_evidence(
            _make_evidence(
                id="ev-abc-002",
                provenance="symbol_graph",
                score=None,
                symbol_kind=None,
            )
        )
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        pack_tool = next(t for t in tools if t.name == "graph_pack")

        result = await pack_tool.ainvoke({"target_count": 10})

        # symbol_graph (1.8) > manual (1.0)
        sg_pos = result.index("ev-abc-002")
        man_pos = result.index("ev-abc-001")
        assert sg_pos < man_pos

    @pytest.mark.anyio
    async def test_graph_pack_score_bonus(self) -> None:
        run = _make_run()
        run.add_evidence(
            _make_evidence(
                id="ev-abc-001",
                score=0.0,
                provenance="hybrid",
                symbol_kind=None,
            )
        )
        run.add_evidence(
            _make_evidence(
                id="ev-abc-002",
                score=1.0,
                provenance="hybrid",
                symbol_kind=None,
            )
        )
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        pack_tool = next(t for t in tools if t.name == "graph_pack")

        result = await pack_tool.ainvoke({"target_count": 10})

        # score=1.0 gets bonus 2.0 extra
        high_pos = result.index("ev-abc-002")
        low_pos = result.index("ev-abc-001")
        assert high_pos < low_pos


# ---------------------------------------------------------------------------
# hybrid_search tool
# ---------------------------------------------------------------------------


class TestHybridSearchTool:
    """Tests for the hybrid_search tool with mocked dependencies."""

    @pytest.mark.anyio
    async def test_hybrid_search_returns_results(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        search_tool = next(t for t in tools if t.name == "hybrid_search")

        # Build mock search results
        mock_result_item = SimpleNamespace(
            uri="src/auth.py",
            content="def login():\n    pass",
            score=0.9,
        )
        mock_search_results = SimpleNamespace(results=[mock_result_item])

        mock_embedder = AsyncMock()
        mock_embedder.embed_batch.return_value = SimpleNamespace(embeddings=[[0.1, 0.2, 0.3]])

        mock_settings = MagicMock()
        mock_settings.default_embedding_model = "openai:text-embedding-3-small"

        with (
            patch(
                "contextmine_core.settings.get_settings",
                return_value=mock_settings,
            ),
            patch(
                "contextmine_core.embeddings.parse_embedding_model_spec",
                return_value=("openai", "text-embedding-3-small"),
            ),
            patch(
                "contextmine_core.embeddings.get_embedder",
                return_value=mock_embedder,
            ),
            patch(
                "contextmine_core.search.hybrid_search",
                new_callable=AsyncMock,
                return_value=mock_search_results,
            ),
        ):
            result = await search_tool.ainvoke({"query": "auth login", "k": 5})

        assert "Found 1 results" in result
        assert "src/auth.py" in result
        assert len(run.evidence) == 1
        assert run.evidence[0].provenance == "hybrid"

    @pytest.mark.anyio
    async def test_hybrid_search_no_results(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        search_tool = next(t for t in tools if t.name == "hybrid_search")

        mock_search_results = SimpleNamespace(results=[])
        mock_embedder = AsyncMock()
        mock_embedder.embed_batch.return_value = SimpleNamespace(embeddings=[[0.1]])

        mock_settings = MagicMock()
        mock_settings.default_embedding_model = "openai:text-embedding-3-small"

        with (
            patch("contextmine_core.settings.get_settings", return_value=mock_settings),
            patch(
                "contextmine_core.embeddings.parse_embedding_model_spec",
                return_value=("openai", "text-embedding-3-small"),
            ),
            patch("contextmine_core.embeddings.get_embedder", return_value=mock_embedder),
            patch(
                "contextmine_core.search.hybrid_search",
                new_callable=AsyncMock,
                return_value=mock_search_results,
            ),
        ):
            result = await search_tool.ainvoke({"query": "nonexistent"})

        assert "No results found" in result

    @pytest.mark.anyio
    async def test_hybrid_search_handles_error(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        search_tool = next(t for t in tools if t.name == "hybrid_search")

        with patch(
            "contextmine_core.settings.get_settings",
            side_effect=RuntimeError("DB down"),
        ):
            result = await search_tool.ainvoke({"query": "test"})

        assert "Search failed" in result


# ---------------------------------------------------------------------------
# open_span tool
# ---------------------------------------------------------------------------


class TestOpenSpanTool:
    """Tests for the open_span tool with mocked DB."""

    @pytest.mark.anyio
    async def test_open_span_returns_content(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        span_tool = next(t for t in tools if t.name == "open_span")

        mock_doc = MagicMock()
        mock_doc.content_markdown = "line1\nline2\nline3\nline4\nline5"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_doc

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await span_tool.ainvoke(
                {
                    "file_path": "src/main.py",
                    "start_line": 2,
                    "end_line": 4,
                }
            )

        assert "src/main.py:2-4" in result
        assert "line2" in result
        assert "line3" in result
        assert "line4" in result
        assert len(run.evidence) == 1
        assert run.evidence[0].provenance == "manual"

    @pytest.mark.anyio
    async def test_open_span_file_not_found(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        span_tool = next(t for t in tools if t.name == "open_span")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await span_tool.ainvoke(
                {
                    "file_path": "nonexistent.py",
                    "start_line": 1,
                    "end_line": 10,
                }
            )

        assert "File not found" in result
        assert len(run.evidence) == 0

    @pytest.mark.anyio
    async def test_open_span_handles_error(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        span_tool = next(t for t in tools if t.name == "open_span")

        with patch(
            "contextmine_core.database.get_async_session",
            side_effect=RuntimeError("DB error"),
        ):
            result = await span_tool.ainvoke(
                {
                    "file_path": "src/main.py",
                    "start_line": 1,
                    "end_line": 5,
                }
            )

        assert "Failed to read file" in result

    @pytest.mark.anyio
    async def test_open_span_clamps_lines(self) -> None:
        """Lines are clamped to valid range."""
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        span_tool = next(t for t in tools if t.name == "open_span")

        mock_doc = MagicMock()
        mock_doc.content_markdown = "line1\nline2\nline3"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_doc

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            # Request beyond file length
            result = await span_tool.ainvoke(
                {
                    "file_path": "src/main.py",
                    "start_line": 1,
                    "end_line": 100,
                }
            )

        assert "line1" in result
        assert "line3" in result


# ---------------------------------------------------------------------------
# goto_definition tool
# ---------------------------------------------------------------------------


class TestGotoDefinitionTool:
    """Tests for the goto_definition tool."""

    @pytest.mark.anyio
    async def test_goto_definition_found(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        defn_tool = next(t for t in tools if t.name == "goto_definition")

        mock_doc = MagicMock()
        mock_doc.uri = "src/auth.py"
        mock_doc.content_markdown = "line1\ndef authenticate(user):\n    pass\nline4"

        mock_sym = MagicMock()
        mock_sym.name = "authenticate"
        mock_sym.kind.value = "function"
        mock_sym.qualified_name = "src.auth.authenticate"
        mock_sym.start_line = 2
        mock_sym.end_line = 3
        mock_sym.document = mock_doc

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_sym]

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await defn_tool.ainvoke({"symbol_name": "authenticate"})

        assert "Found 1 definition" in result
        assert "authenticate" in result
        assert "src/auth.py" in result
        assert len(run.evidence) == 1
        assert run.evidence[0].provenance == "symbol_index"

    @pytest.mark.anyio
    async def test_goto_definition_not_found(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        defn_tool = next(t for t in tools if t.name == "goto_definition")

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await defn_tool.ainvoke({"symbol_name": "nonexistent"})

        assert "No definition found" in result

    @pytest.mark.anyio
    async def test_goto_definition_with_file_path(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        defn_tool = next(t for t in tools if t.name == "goto_definition")

        mock_doc = MagicMock()
        mock_doc.uri = "src/models.py"
        mock_doc.content_markdown = "class User:\n    pass"

        mock_sym = MagicMock()
        mock_sym.name = "User"
        mock_sym.kind.value = "class"
        mock_sym.qualified_name = "src.models.User"
        mock_sym.start_line = 1
        mock_sym.end_line = 2
        mock_sym.document = mock_doc

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_sym]

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await defn_tool.ainvoke(
                {
                    "symbol_name": "User",
                    "file_path": "src/models.py",
                }
            )

        assert "Found 1 definition" in result

    @pytest.mark.anyio
    async def test_goto_definition_handles_error(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        defn_tool = next(t for t in tools if t.name == "goto_definition")

        with patch(
            "contextmine_core.database.get_async_session",
            side_effect=RuntimeError("DB error"),
        ):
            result = await defn_tool.ainvoke({"symbol_name": "test"})

        assert "Goto definition failed" in result


# ---------------------------------------------------------------------------
# find_references tool
# ---------------------------------------------------------------------------


class TestFindReferencesTool:
    """Tests for the find_references tool."""

    @pytest.mark.anyio
    async def test_find_references_found(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        ref_tool = next(t for t in tools if t.name == "find_references")

        mock_doc = MagicMock()
        mock_doc.uri = "src/handler.py"
        mock_doc.content_markdown = "line1\nline2\nlogin()\nline4\nline5"

        mock_ref_sym = MagicMock()
        mock_ref_sym.name = "handle_request"
        mock_ref_sym.kind.value = "function"
        mock_ref_sym.qualified_name = "handler.handle_request"
        mock_ref_sym.start_line = 1
        mock_ref_sym.document = mock_doc

        mock_edge = MagicMock()
        mock_edge.edge_type.value = "calls"
        mock_edge.source_line = 3
        mock_edge.source_symbol = mock_ref_sym

        mock_target = MagicMock()
        mock_target.id = uuid.uuid4()

        # Two queries: first for target, then for edges
        mock_target_result = MagicMock()
        mock_target_result.scalars.return_value.all.return_value = [mock_target]

        mock_edges_result = MagicMock()
        mock_edges_result.scalars.return_value.all.return_value = [mock_edge]

        mock_session = AsyncMock()
        mock_session.execute.side_effect = [mock_target_result, mock_edges_result]
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await ref_tool.ainvoke({"symbol_name": "login"})

        assert "Found 1 reference" in result
        assert "calls" in result
        assert len(run.evidence) == 1

    @pytest.mark.anyio
    async def test_find_references_not_found(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        ref_tool = next(t for t in tools if t.name == "find_references")

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await ref_tool.ainvoke({"symbol_name": "nonexistent"})

        assert "not found" in result

    @pytest.mark.anyio
    async def test_find_references_handles_error(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        ref_tool = next(t for t in tools if t.name == "find_references")

        with patch(
            "contextmine_core.database.get_async_session",
            side_effect=RuntimeError("oops"),
        ):
            result = await ref_tool.ainvoke({"symbol_name": "test"})

        assert "Find references failed" in result


# ---------------------------------------------------------------------------
# get_signature tool
# ---------------------------------------------------------------------------


class TestGetSignatureTool:
    """Tests for the get_signature tool."""

    @pytest.mark.anyio
    async def test_get_signature_with_sig_and_docstring(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        sig_tool = next(t for t in tools if t.name == "get_signature")

        mock_doc = MagicMock()
        mock_doc.uri = "src/utils.py"
        mock_doc.content_markdown = "def helper(x: int) -> str:\n    '''Help.'''\n    return str(x)"

        mock_sym = MagicMock()
        mock_sym.name = "helper"
        mock_sym.kind.value = "function"
        mock_sym.qualified_name = "utils.helper"
        mock_sym.start_line = 1
        mock_sym.end_line = 3
        mock_sym.signature = "(x: int) -> str"
        mock_sym.meta = {"docstring": "Help function."}
        mock_sym.document = mock_doc

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_sym]

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await sig_tool.ainvoke({"symbol_name": "helper"})

        assert "Signature: (x: int) -> str" in result
        assert "Help function." in result
        assert len(run.evidence) == 1

    @pytest.mark.anyio
    async def test_get_signature_falls_back_to_source(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        sig_tool = next(t for t in tools if t.name == "get_signature")

        mock_doc = MagicMock()
        mock_doc.uri = "src/utils.py"
        mock_doc.content_markdown = (
            "def helper():\n    pass\n    more\n    stuff\n    end\n    extra"
        )

        mock_sym = MagicMock()
        mock_sym.name = "helper"
        mock_sym.kind.value = "function"
        mock_sym.qualified_name = "utils.helper"
        mock_sym.start_line = 1
        mock_sym.end_line = 6
        mock_sym.signature = None
        mock_sym.meta = {}
        mock_sym.document = mock_doc

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_sym]

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await sig_tool.ainvoke({"symbol_name": "helper"})

        assert "Source:" in result
        assert "def helper():" in result

    @pytest.mark.anyio
    async def test_get_signature_not_found(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        sig_tool = next(t for t in tools if t.name == "get_signature")

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await sig_tool.ainvoke({"symbol_name": "nope"})

        assert "not found" in result

    @pytest.mark.anyio
    async def test_get_signature_handles_error(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        sig_tool = next(t for t in tools if t.name == "get_signature")

        with patch(
            "contextmine_core.database.get_async_session",
            side_effect=RuntimeError("boom"),
        ):
            result = await sig_tool.ainvoke({"symbol_name": "x"})

        assert "Get signature failed" in result


# ---------------------------------------------------------------------------
# symbol_outline tool
# ---------------------------------------------------------------------------


class TestSymbolOutlineTool:
    """Tests for the symbol_outline tool."""

    @pytest.mark.anyio
    async def test_symbol_outline_found(self) -> None:
        run_holder = _make_run_holder()
        tools = create_tools(run_holder)
        outline_tool = next(t for t in tools if t.name == "symbol_outline")

        mock_doc = MagicMock()
        mock_doc.id = uuid.uuid4()

        mock_sym1 = MagicMock()
        mock_sym1.kind.value = "class"
        mock_sym1.name = "User"
        mock_sym1.start_line = 1
        mock_sym1.end_line = 20
        mock_sym1.signature = None
        mock_sym1.parent_name = None

        mock_sym2 = MagicMock()
        mock_sym2.kind.value = "method"
        mock_sym2.name = "save"
        mock_sym2.start_line = 5
        mock_sym2.end_line = 10
        mock_sym2.signature = "(self) -> bool"
        mock_sym2.parent_name = "User"

        # Two queries: first for doc, then for symbols
        mock_doc_result = MagicMock()
        mock_doc_result.scalar_one_or_none.return_value = mock_doc

        mock_sym_result = MagicMock()
        mock_sym_result.scalars.return_value.all.return_value = [mock_sym1, mock_sym2]

        mock_session = AsyncMock()
        mock_session.execute.side_effect = [mock_doc_result, mock_sym_result]
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await outline_tool.ainvoke({"file_path": "src/models.py"})

        assert "Found 2 indexed symbols" in result
        assert "class User" in result
        assert "method save" in result
        assert "(self) -> bool" in result

    @pytest.mark.anyio
    async def test_symbol_outline_file_not_found(self) -> None:
        run_holder = _make_run_holder()
        tools = create_tools(run_holder)
        outline_tool = next(t for t in tools if t.name == "symbol_outline")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await outline_tool.ainvoke({"file_path": "nope.py"})

        assert "File not found" in result

    @pytest.mark.anyio
    async def test_symbol_outline_no_symbols(self) -> None:
        run_holder = _make_run_holder()
        tools = create_tools(run_holder)
        outline_tool = next(t for t in tools if t.name == "symbol_outline")

        mock_doc = MagicMock()
        mock_doc.id = uuid.uuid4()

        mock_doc_result = MagicMock()
        mock_doc_result.scalar_one_or_none.return_value = mock_doc

        mock_sym_result = MagicMock()
        mock_sym_result.scalars.return_value.all.return_value = []

        mock_session = AsyncMock()
        mock_session.execute.side_effect = [mock_doc_result, mock_sym_result]
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await outline_tool.ainvoke({"file_path": "empty.py"})

        assert "No symbols indexed" in result

    @pytest.mark.anyio
    async def test_symbol_outline_handles_error(self) -> None:
        run_holder = _make_run_holder()
        tools = create_tools(run_holder)
        outline_tool = next(t for t in tools if t.name == "symbol_outline")

        with patch(
            "contextmine_core.database.get_async_session",
            side_effect=RuntimeError("err"),
        ):
            result = await outline_tool.ainvoke({"file_path": "x.py"})

        assert "Symbol outline failed" in result


# ---------------------------------------------------------------------------
# symbol_find tool
# ---------------------------------------------------------------------------


class TestSymbolFindTool:
    """Tests for the symbol_find tool."""

    @pytest.mark.anyio
    async def test_symbol_find_exact_match(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        find_tool = next(t for t in tools if t.name == "symbol_find")

        mock_doc = MagicMock()
        mock_doc.uri = "src/models.py"
        mock_doc.content_markdown = "class User:\n    id: int\n    name: str"

        mock_sym = MagicMock()
        mock_sym.name = "User"
        mock_sym.kind.value = "class"
        mock_sym.qualified_name = "models.User"
        mock_sym.start_line = 1
        mock_sym.end_line = 3
        mock_sym.document = mock_doc

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_sym]

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await find_tool.ainvoke({"name": "User"})

        assert "Found 1 symbol" in result
        assert "class" in result
        assert len(run.evidence) == 1

    @pytest.mark.anyio
    async def test_symbol_find_not_found(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        find_tool = next(t for t in tools if t.name == "symbol_find")

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []

        mock_session = AsyncMock()
        # First exact, then partial, both empty
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await find_tool.ainvoke({"name": "nonexistent"})

        assert "not found" in result

    @pytest.mark.anyio
    async def test_symbol_find_handles_error(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        find_tool = next(t for t in tools if t.name == "symbol_find")

        with patch(
            "contextmine_core.database.get_async_session",
            side_effect=RuntimeError("oops"),
        ):
            result = await find_tool.ainvoke({"name": "test"})

        assert "Symbol find failed" in result


# ---------------------------------------------------------------------------
# symbol_callers tool
# ---------------------------------------------------------------------------


class TestSymbolCallersTool:
    """Tests for the symbol_callers tool."""

    @pytest.mark.anyio
    async def test_symbol_callers_found(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        callers_tool = next(t for t in tools if t.name == "symbol_callers")

        mock_caller_doc = MagicMock()
        mock_caller_doc.uri = "src/handler.py"
        mock_caller_doc.content_markdown = "def handle():\n    login()\n    return"

        mock_caller_sym = MagicMock()
        mock_caller_sym.name = "handle"
        mock_caller_sym.kind.value = "function"
        mock_caller_sym.qualified_name = "handler.handle"
        mock_caller_sym.start_line = 1
        mock_caller_sym.end_line = 3
        mock_caller_sym.document = mock_caller_doc

        mock_edge = MagicMock()
        mock_edge.source_symbol = mock_caller_sym
        mock_edge.source_line = 2

        mock_target = MagicMock()
        mock_target.id = uuid.uuid4()
        mock_target.incoming_edges = [mock_edge]

        mock_target_result = MagicMock()
        mock_target_result.scalars.return_value.all.return_value = [mock_target]

        mock_edges_result = MagicMock()
        mock_edges_result.scalars.return_value.all.return_value = [mock_edge]

        mock_session = AsyncMock()
        mock_session.execute.side_effect = [mock_target_result, mock_edges_result]
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await callers_tool.ainvoke({"name": "login"})

        assert "caller" in result.lower()
        assert len(run.evidence) == 1

    @pytest.mark.anyio
    async def test_symbol_callers_not_found(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        callers_tool = next(t for t in tools if t.name == "symbol_callers")

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await callers_tool.ainvoke({"name": "xyz"})

        assert "not found" in result

    @pytest.mark.anyio
    async def test_symbol_callers_handles_error(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        callers_tool = next(t for t in tools if t.name == "symbol_callers")

        with patch(
            "contextmine_core.database.get_async_session",
            side_effect=RuntimeError("err"),
        ):
            result = await callers_tool.ainvoke({"name": "test"})

        assert "Symbol callers failed" in result


# ---------------------------------------------------------------------------
# symbol_callees tool
# ---------------------------------------------------------------------------


class TestSymbolCalleesTool:
    """Tests for the symbol_callees tool."""

    @pytest.mark.anyio
    async def test_symbol_callees_found(self) -> None:
        run_holder = _make_run_holder()
        tools = create_tools(run_holder)
        callees_tool = next(t for t in tools if t.name == "symbol_callees")

        mock_callee_doc = MagicMock()
        mock_callee_doc.uri = "src/db.py"
        mock_callee_doc.content_markdown = "def connect():\n    pass"

        mock_callee_sym = MagicMock()
        mock_callee_sym.name = "connect"
        mock_callee_sym.kind.value = "function"
        mock_callee_sym.qualified_name = "db.connect"
        mock_callee_sym.start_line = 1
        mock_callee_sym.end_line = 2
        mock_callee_sym.signature = "(host: str) -> Connection"
        mock_callee_sym.document = mock_callee_doc

        mock_edge = MagicMock()
        mock_edge.target_symbol = mock_callee_sym

        mock_source = MagicMock()
        mock_source.id = uuid.uuid4()
        mock_source.outgoing_edges = [mock_edge]

        mock_source_result = MagicMock()
        mock_source_result.scalars.return_value.all.return_value = [mock_source]

        mock_edges_result = MagicMock()
        mock_edges_result.scalars.return_value.all.return_value = [mock_edge]

        mock_session = AsyncMock()
        mock_session.execute.side_effect = [mock_source_result, mock_edges_result]
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await callees_tool.ainvoke({"name": "handle"})

        assert "calls 1 function" in result
        assert "connect" in result

    @pytest.mark.anyio
    async def test_symbol_callees_not_found(self) -> None:
        run_holder = _make_run_holder()
        tools = create_tools(run_holder)
        callees_tool = next(t for t in tools if t.name == "symbol_callees")

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await callees_tool.ainvoke({"name": "xyz"})

        assert "not found" in result

    @pytest.mark.anyio
    async def test_symbol_callees_handles_error(self) -> None:
        run_holder = _make_run_holder()
        tools = create_tools(run_holder)
        callees_tool = next(t for t in tools if t.name == "symbol_callees")

        with patch(
            "contextmine_core.database.get_async_session",
            side_effect=RuntimeError("err"),
        ):
            result = await callees_tool.ainvoke({"name": "test"})

        assert "Symbol callees failed" in result


# ---------------------------------------------------------------------------
# symbol_enclosing tool
# ---------------------------------------------------------------------------


class TestSymbolEnclosingTool:
    """Tests for the symbol_enclosing tool."""

    @pytest.mark.anyio
    async def test_symbol_enclosing_found(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        enc_tool = next(t for t in tools if t.name == "symbol_enclosing")

        mock_doc = MagicMock()
        mock_doc.id = uuid.uuid4()
        mock_doc.content_markdown = (
            "class Foo:\n    def bar(self):\n        x = 1\n        return x"
        )

        mock_sym = MagicMock()
        mock_sym.name = "bar"
        mock_sym.kind.value = "method"
        mock_sym.qualified_name = "foo.Foo.bar"
        mock_sym.start_line = 2
        mock_sym.end_line = 4
        mock_sym.document = mock_doc

        mock_doc_result = MagicMock()
        mock_doc_result.scalar_one_or_none.return_value = mock_doc

        mock_sym_result = MagicMock()
        mock_sym_result.scalars.return_value.all.return_value = [mock_sym]

        mock_session = AsyncMock()
        mock_session.execute.side_effect = [mock_doc_result, mock_sym_result]
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await enc_tool.ainvoke({"file_path": "src/foo.py", "line": 3})

        assert "method" in result
        assert "bar" in result
        assert len(run.evidence) == 1

    @pytest.mark.anyio
    async def test_symbol_enclosing_file_not_found(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        enc_tool = next(t for t in tools if t.name == "symbol_enclosing")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await enc_tool.ainvoke({"file_path": "nope.py", "line": 1})

        assert "File not found" in result

    @pytest.mark.anyio
    async def test_symbol_enclosing_line_not_in_symbol(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        enc_tool = next(t for t in tools if t.name == "symbol_enclosing")

        mock_doc = MagicMock()
        mock_doc.id = uuid.uuid4()

        mock_doc_result = MagicMock()
        mock_doc_result.scalar_one_or_none.return_value = mock_doc

        mock_sym_result = MagicMock()
        mock_sym_result.scalars.return_value.all.return_value = []

        mock_session = AsyncMock()
        mock_session.execute.side_effect = [mock_doc_result, mock_sym_result]
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await enc_tool.ainvoke({"file_path": "src/x.py", "line": 999})

        assert "not inside any indexed symbol" in result

    @pytest.mark.anyio
    async def test_symbol_enclosing_handles_error(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        enc_tool = next(t for t in tools if t.name == "symbol_enclosing")

        with patch(
            "contextmine_core.database.get_async_session",
            side_effect=RuntimeError("err"),
        ):
            result = await enc_tool.ainvoke({"file_path": "x.py", "line": 1})

        assert "Symbol enclosing failed" in result


# ---------------------------------------------------------------------------
# summarize_evidence tool
# ---------------------------------------------------------------------------


class TestSummarizeEvidenceTool:
    """Tests for the summarize_evidence tool."""

    @pytest.mark.anyio
    async def test_summarize_no_evidence(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        summ_tool = next(t for t in tools if t.name == "summarize_evidence")

        result = await summ_tool.ainvoke({"goal": "understand auth"})
        assert "No evidence collected" in result

    @pytest.mark.anyio
    async def test_summarize_with_evidence(self) -> None:
        run = _make_run()
        run.add_evidence(_make_evidence(id="ev-abc-001"))
        run.add_evidence(_make_evidence(id="ev-abc-002"))
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        summ_tool = next(t for t in tools if t.name == "summarize_evidence")

        mock_provider = AsyncMock()
        mock_provider.generate_text.return_value = "Auth uses JWT tokens. [ev-abc-001]"

        with patch(
            "contextmine_core.research.llm.get_research_llm_provider",
            return_value=mock_provider,
        ):
            result = await summ_tool.ainvoke({"goal": "understand auth"})

        assert "Evidence Summary (2 items)" in result
        assert "JWT tokens" in result

    @pytest.mark.anyio
    async def test_summarize_handles_error(self) -> None:
        run = _make_run()
        run.add_evidence(_make_evidence())
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        summ_tool = next(t for t in tools if t.name == "summarize_evidence")

        with patch(
            "contextmine_core.research.llm.get_research_llm_provider",
            side_effect=RuntimeError("LLM down"),
        ):
            result = await summ_tool.ainvoke({"goal": "test"})

        assert "Summarize failed" in result


# ---------------------------------------------------------------------------
# graphrag_search tool
# ---------------------------------------------------------------------------


class TestGraphragSearchTool:
    """Tests for the graphrag_search tool."""

    @pytest.mark.anyio
    async def test_graphrag_search_with_results(self) -> None:
        from contextmine_core.graphrag import (
            Citation,
            CommunityContext,
            ContextPack,
            EdgeContext,
            EntityContext,
        )

        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        grag_tool = next(t for t in tools if t.name == "graphrag_search")

        mock_context = ContextPack(
            query="auth",
            communities=[
                CommunityContext(
                    community_id=uuid.uuid4(),
                    level=0,
                    title="Auth Module",
                    summary="Handles authentication",
                    relevance_score=0.95,
                    member_count=5,
                ),
            ],
            entities=[
                EntityContext(
                    node_id=uuid.uuid4(),
                    kind="FILE",
                    natural_key="file:src/auth.py",
                    name="auth.py",
                    evidence=[
                        Citation(file_path="src/auth.py", start_line=1, end_line=50, snippet="code")
                    ],
                    relevance_score=0.9,
                ),
            ],
            edges=[
                EdgeContext(source_id="a", target_id="b", kind="DEFINES"),
            ],
        )

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "contextmine_core.database.get_async_session",
                return_value=mock_session,
            ),
            patch(
                "contextmine_core.graphrag.graph_rag_context",
                new_callable=AsyncMock,
                return_value=mock_context,
            ),
        ):
            result = await grag_tool.ainvoke({"query": "auth", "max_entities": 10})

        assert "Global Context" in result
        assert "Auth Module" in result
        assert "Local Context" in result
        assert "Relationships" in result
        assert len(run.evidence) == 1

    @pytest.mark.anyio
    async def test_graphrag_search_empty_results(self) -> None:
        from contextmine_core.graphrag import ContextPack

        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        grag_tool = next(t for t in tools if t.name == "graphrag_search")

        mock_context = ContextPack(query="nothing")

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "contextmine_core.database.get_async_session",
                return_value=mock_session,
            ),
            patch(
                "contextmine_core.graphrag.graph_rag_context",
                new_callable=AsyncMock,
                return_value=mock_context,
            ),
        ):
            result = await grag_tool.ainvoke({"query": "nope"})

        assert "No relevant results" in result

    @pytest.mark.anyio
    async def test_graphrag_search_handles_error(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        grag_tool = next(t for t in tools if t.name == "graphrag_search")

        with patch(
            "contextmine_core.database.get_async_session",
            side_effect=RuntimeError("boom"),
        ):
            result = await grag_tool.ainvoke({"query": "test"})

        assert "GraphRAG search failed" in result
        assert "hybrid_search as fallback" in result


# ---------------------------------------------------------------------------
# kg_neighborhood tool
# ---------------------------------------------------------------------------


class TestKgNeighborhoodTool:
    """Tests for the kg_neighborhood tool."""

    @pytest.mark.anyio
    async def test_kg_neighborhood_node_not_found(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        kgn_tool = next(t for t in tools if t.name == "kg_neighborhood")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None

        mock_session = AsyncMock()
        # First by name, then by natural_key
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await kgn_tool.ainvoke({"node_name": "nonexistent"})

        assert "not found" in result

    @pytest.mark.anyio
    async def test_kg_neighborhood_handles_error(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        kgn_tool = next(t for t in tools if t.name == "kg_neighborhood")

        with patch(
            "contextmine_core.database.get_async_session",
            side_effect=RuntimeError("err"),
        ):
            result = await kgn_tool.ainvoke({"node_name": "test"})

        assert "Knowledge Graph neighborhood failed" in result


# ---------------------------------------------------------------------------
# kg_path tool
# ---------------------------------------------------------------------------


class TestKgPathTool:
    """Tests for the kg_path tool."""

    @pytest.mark.anyio
    async def test_kg_path_source_not_found(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        path_tool = next(t for t in tools if t.name == "kg_path")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None

        mock_session = AsyncMock()
        # Returns None for both name and natural_key lookups
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await path_tool.ainvoke({"from_name": "a", "to_name": "b"})

        assert "not found" in result

    @pytest.mark.anyio
    async def test_kg_path_handles_error(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        path_tool = next(t for t in tools if t.name == "kg_path")

        with patch(
            "contextmine_core.database.get_async_session",
            side_effect=RuntimeError("err"),
        ):
            result = await path_tool.ainvoke({"from_name": "a", "to_name": "b"})

        assert "Knowledge Graph path failed" in result


# ---------------------------------------------------------------------------
# graph_expand tool
# ---------------------------------------------------------------------------


class TestGraphExpandTool:
    """Tests for the graph_expand tool."""

    @pytest.mark.anyio
    async def test_graph_expand_no_seeds(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        expand_tool = next(t for t in tools if t.name == "graph_expand")

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await expand_tool.ainvoke({"seed_names": ["nonexistent"]})

        assert "No symbols found" in result

    @pytest.mark.anyio
    async def test_graph_expand_handles_error(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        expand_tool = next(t for t in tools if t.name == "graph_expand")

        with patch(
            "contextmine_core.database.get_async_session",
            side_effect=RuntimeError("err"),
        ):
            result = await expand_tool.ainvoke({"seed_names": ["x"]})

        assert "Graph expand failed" in result

    @pytest.mark.anyio
    async def test_graph_expand_clamps_depth(self) -> None:
        """Depth is clamped to 1-5 range."""
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        expand_tool = next(t for t in tools if t.name == "graph_expand")

        mock_doc = MagicMock()
        mock_doc.uri = "src/x.py"
        mock_doc.content_markdown = "code"

        mock_sym = MagicMock()
        mock_sym.id = uuid.uuid4()
        mock_sym.name = "X"
        mock_sym.kind.value = "class"
        mock_sym.qualified_name = "x.X"
        mock_sym.start_line = 1
        mock_sym.end_line = 2
        mock_sym.document = mock_doc

        mock_seed_result = MagicMock()
        mock_seed_result.scalars.return_value.all.return_value = [mock_sym]

        # No edges found - just the seed
        mock_edges_result = MagicMock()
        mock_edges_result.scalars.return_value.all.return_value = []

        mock_session = AsyncMock()
        mock_session.execute.side_effect = [mock_seed_result, mock_edges_result]
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await expand_tool.ainvoke(
                {
                    "seed_names": ["X"],
                    "depth": 100,  # should be clamped to 5
                }
            )

        assert "Expanded to 1 symbols" in result
        assert "[SEED]" in result


# ---------------------------------------------------------------------------
# graph_trace tool
# ---------------------------------------------------------------------------


class TestGraphTraceTool:
    """Tests for the graph_trace tool."""

    @pytest.mark.anyio
    async def test_graph_trace_source_not_found(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        trace_tool = next(t for t in tools if t.name == "graph_trace")

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await trace_tool.ainvoke(
                {
                    "from_symbol": "a",
                    "to_symbol": "b",
                }
            )

        assert "not found" in result

    @pytest.mark.anyio
    async def test_graph_trace_same_symbol(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        trace_tool = next(t for t in tools if t.name == "graph_trace")

        sym_id = uuid.uuid4()

        mock_sym = MagicMock()
        mock_sym.id = sym_id
        mock_sym.name = "func"
        mock_sym.document = MagicMock()

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_sym]

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await trace_tool.ainvoke(
                {
                    "from_symbol": "func",
                    "to_symbol": "func",
                }
            )

        assert "same symbol" in result.lower()

    @pytest.mark.anyio
    async def test_graph_trace_handles_error(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        trace_tool = next(t for t in tools if t.name == "graph_trace")

        with patch(
            "contextmine_core.database.get_async_session",
            side_effect=RuntimeError("err"),
        ):
            result = await trace_tool.ainvoke(
                {
                    "from_symbol": "a",
                    "to_symbol": "b",
                }
            )

        assert "Graph trace failed" in result


# ---------------------------------------------------------------------------
# ResearchAgent._build_system_prompt
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    """Tests for the system prompt builder."""

    def test_prompt_includes_question(self) -> None:
        agent = ResearchAgent(llm_provider=MagicMock())
        prompt = agent._build_system_prompt("How does auth work?", None)
        assert "How does auth work?" in prompt

    def test_prompt_with_scope(self) -> None:
        agent = ResearchAgent(llm_provider=MagicMock())
        prompt = agent._build_system_prompt("question", "src/auth/**")
        assert "src/auth/**" in prompt
        assert "Limit your search" in prompt

    def test_prompt_without_scope(self) -> None:
        agent = ResearchAgent(llm_provider=MagicMock())
        prompt = agent._build_system_prompt("question", None)
        assert "Limit your search" not in prompt

    def test_prompt_contains_tool_descriptions(self) -> None:
        agent = ResearchAgent(llm_provider=MagicMock())
        prompt = agent._build_system_prompt("q", None)
        assert "hybrid_search" in prompt
        assert "graphrag_search" in prompt
        assert "goto_definition" in prompt
        assert "symbol_outline" in prompt
        assert "finalize" in prompt
        assert "open_span" in prompt
        assert "graph_expand" in prompt
        assert "kg_neighborhood" in prompt
        assert "kg_path" in prompt
        assert "graph_trace" in prompt

    def test_prompt_mentions_verification(self) -> None:
        agent = ResearchAgent(llm_provider=MagicMock())
        prompt = agent._build_system_prompt("q", None)
        assert "VERIFIED" in prompt
        assert "citation" in prompt.lower()


# ---------------------------------------------------------------------------
# ResearchAgent._build_graph
# ---------------------------------------------------------------------------


class TestBuildGraph:
    """Tests for graph construction."""

    def test_build_graph_creates_valid_graph(self) -> None:
        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model

        mock_provider = MagicMock()
        mock_provider._model = mock_model

        agent = ResearchAgent(llm_provider=mock_provider)
        run_holder = _make_run_holder()

        graph = agent._build_graph(run_holder)

        # Verify graph was compiled (has ainvoke method)
        assert hasattr(graph, "ainvoke")

    def test_build_graph_has_correct_nodes(self) -> None:
        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model

        mock_provider = MagicMock()
        mock_provider._model = mock_model

        agent = ResearchAgent(llm_provider=mock_provider)
        run_holder = _make_run_holder()

        graph = agent._build_graph(run_holder)

        # The compiled graph should have the expected node names
        # Access via the graph's internal structure
        node_names = set(graph.get_graph().nodes.keys())
        assert "agent" in node_names
        assert "tools" in node_names
        assert "verify" in node_names


# ---------------------------------------------------------------------------
# ResearchAgent.research - integration with mocked graph
# ---------------------------------------------------------------------------


class TestResearchMethod:
    """Tests for the high-level research() method."""

    @pytest.mark.anyio
    async def test_research_creates_run(self) -> None:
        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model

        mock_provider = MagicMock()
        mock_provider._model = mock_model

        agent = ResearchAgent(
            llm_provider=mock_provider,
            config=AgentConfig(store_artifacts=False),
        )

        # Mock _build_graph to return a mock graph that returns a final state
        mock_run = _make_run(question="test question")
        mock_run.complete("The answer")

        final_state = {
            "messages": [],
            "run": mock_run,
            "pending_answer": None,
            "verification_attempts": 0,
        }

        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = final_state

        with patch.object(agent, "_build_graph", return_value=mock_graph):
            result = await agent.research("test question")

        assert result.question == "test question"
        assert result.answer == "The answer"
        assert result.status == RunStatus.DONE

    @pytest.mark.anyio
    async def test_research_handles_exception(self) -> None:
        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model

        mock_provider = MagicMock()
        mock_provider._model = mock_model

        agent = ResearchAgent(
            llm_provider=mock_provider,
            config=AgentConfig(store_artifacts=False),
        )

        mock_graph = AsyncMock()
        mock_graph.ainvoke.side_effect = RuntimeError("Graph exploded")

        with patch.object(agent, "_build_graph", return_value=mock_graph):
            result = await agent.research("question")

        assert result.status == RunStatus.ERROR
        assert "Graph exploded" in result.error_message

    @pytest.mark.anyio
    async def test_research_stores_artifacts(self) -> None:
        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model

        mock_provider = MagicMock()
        mock_provider._model = mock_model

        agent = ResearchAgent(
            llm_provider=mock_provider,
            config=AgentConfig(store_artifacts=True),
        )

        mock_run = _make_run()
        mock_run.complete("answer")

        final_state = {
            "messages": [],
            "run": mock_run,
            "pending_answer": None,
            "verification_attempts": 0,
        }

        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = final_state

        mock_store = MagicMock()

        with (
            patch.object(agent, "_build_graph", return_value=mock_graph),
            patch(
                "contextmine_core.research.agent.get_artifact_store",
                return_value=mock_store,
            ),
        ):
            await agent.research("q")

        mock_store.save_run.assert_called_once()

    @pytest.mark.anyio
    async def test_research_artifact_save_failure_does_not_crash(self) -> None:
        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model

        mock_provider = MagicMock()
        mock_provider._model = mock_model

        agent = ResearchAgent(
            llm_provider=mock_provider,
            config=AgentConfig(store_artifacts=True),
        )

        mock_run = _make_run()
        mock_run.complete("answer")

        final_state = {
            "messages": [],
            "run": mock_run,
            "pending_answer": None,
            "verification_attempts": 0,
        }

        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = final_state

        mock_store = MagicMock()
        mock_store.save_run.side_effect = OSError("Disk full")

        with (
            patch.object(agent, "_build_graph", return_value=mock_graph),
            patch(
                "contextmine_core.research.agent.get_artifact_store",
                return_value=mock_store,
            ),
        ):
            result = await agent.research("q")

        # Should still return the run, not crash
        assert result.answer == "answer"


# ---------------------------------------------------------------------------
# run_research convenience function
# ---------------------------------------------------------------------------


class TestRunResearch:
    """Tests for the run_research convenience function."""

    @pytest.mark.anyio
    async def test_run_research_with_provider(self) -> None:
        from contextmine_core.research.agent import run_research

        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model

        mock_provider = MagicMock()
        mock_provider._model = mock_model

        mock_run = _make_run()
        mock_run.complete("result")

        with patch.object(
            ResearchAgent,
            "research",
            new_callable=AsyncMock,
            return_value=mock_run,
        ):
            result = await run_research(
                question="test",
                llm_provider=mock_provider,
                max_steps=5,
                store_artifacts=False,
            )

        assert result.answer == "result"

    @pytest.mark.anyio
    async def test_run_research_default_provider(self) -> None:
        from contextmine_core.research.agent import run_research

        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model

        mock_default_provider = MagicMock()
        mock_default_provider._model = mock_model

        mock_run = _make_run()
        mock_run.complete("answer")

        with (
            patch(
                "contextmine_core.research.llm.get_research_llm_provider",
                return_value=mock_default_provider,
            ),
            patch.object(
                ResearchAgent,
                "research",
                new_callable=AsyncMock,
                return_value=mock_run,
            ),
        ):
            result = await run_research(
                question="test",
                store_artifacts=False,
            )

        assert result.answer == "answer"


# ---------------------------------------------------------------------------
# Evidence ID generation
# ---------------------------------------------------------------------------


class TestEvidenceIdGeneration:
    """Tests that evidence IDs are generated consistently."""

    @pytest.mark.anyio
    async def test_evidence_ids_are_sequential(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        span_tool = next(t for t in tools if t.name == "open_span")

        mock_doc = MagicMock()
        mock_doc.content_markdown = "line1\nline2"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_doc

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            await span_tool.ainvoke({"file_path": "a.py", "start_line": 1, "end_line": 2})
            await span_tool.ainvoke({"file_path": "b.py", "start_line": 1, "end_line": 2})

        # Evidence IDs should be sequential
        assert len(run.evidence) == 2
        assert run.evidence[0].id.endswith("-001")
        assert run.evidence[1].id.endswith("-002")


# ---------------------------------------------------------------------------
# AgentState type
# ---------------------------------------------------------------------------


class TestAgentState:
    """Tests for AgentState TypedDict."""

    def test_agent_state_keys(self) -> None:
        """AgentState should have the expected keys."""
        annotations = AgentState.__annotations__
        assert "messages" in annotations
        assert "run" in annotations
        assert "pending_answer" in annotations
        assert "verification_attempts" in annotations


# ---------------------------------------------------------------------------
# graph_trace - path found scenario
# ---------------------------------------------------------------------------


class TestGraphTracePathFound:
    """Tests for graph_trace when a path IS found between two symbols."""

    @pytest.mark.anyio
    async def test_graph_trace_finds_path(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        trace_tool = next(t for t in tools if t.name == "graph_trace")

        src_id = uuid.uuid4()
        tgt_id = uuid.uuid4()

        mock_doc_src = MagicMock()
        mock_doc_src.uri = "src/caller.py"
        mock_doc_src.content_markdown = "def caller():\n    callee()"

        mock_doc_tgt = MagicMock()
        mock_doc_tgt.uri = "src/callee.py"
        mock_doc_tgt.content_markdown = "def callee():\n    pass"

        mock_src_sym = MagicMock()
        mock_src_sym.id = src_id
        mock_src_sym.name = "caller"
        mock_src_sym.kind.value = "function"
        mock_src_sym.qualified_name = "caller_mod.caller"
        mock_src_sym.start_line = 1
        mock_src_sym.end_line = 2
        mock_src_sym.document = mock_doc_src

        mock_tgt_sym = MagicMock()
        mock_tgt_sym.id = tgt_id
        mock_tgt_sym.name = "callee"
        mock_tgt_sym.kind.value = "function"
        mock_tgt_sym.qualified_name = "callee_mod.callee"
        mock_tgt_sym.start_line = 1
        mock_tgt_sym.end_line = 2
        mock_tgt_sym.document = mock_doc_tgt

        # Query 1: find source and target symbols
        mock_sym_result = MagicMock()
        mock_sym_result.scalars.return_value.all.return_value = [mock_src_sym, mock_tgt_sym]

        # Query 2: get outgoing edges from source (finds target)
        mock_edge = MagicMock()
        mock_edge.target_symbol = mock_tgt_sym

        mock_edges_result = MagicMock()
        mock_edges_result.scalars.return_value.all.return_value = [mock_edge]

        # Query 3: get all symbols in path
        mock_path_sym_result = MagicMock()
        mock_path_sym_result.scalars.return_value.all.return_value = [mock_src_sym, mock_tgt_sym]

        mock_session = AsyncMock()
        mock_session.execute.side_effect = [
            mock_sym_result,
            mock_edges_result,
            mock_path_sym_result,
        ]
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await trace_tool.ainvoke(
                {
                    "from_symbol": "caller",
                    "to_symbol": "callee",
                }
            )

        assert "Found 1 path" in result
        assert "caller" in result
        assert "callee" in result
        assert len(run.evidence) >= 2  # Evidence for both symbols in path

    @pytest.mark.anyio
    async def test_graph_trace_no_path(self) -> None:
        """When BFS doesn't find a path within max_depth."""
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        trace_tool = next(t for t in tools if t.name == "graph_trace")

        src_id = uuid.uuid4()
        tgt_id = uuid.uuid4()

        mock_src = MagicMock()
        mock_src.id = src_id
        mock_src.name = "a"
        mock_src.document = MagicMock()

        mock_tgt = MagicMock()
        mock_tgt.id = tgt_id
        mock_tgt.name = "b"
        mock_tgt.document = MagicMock()

        mock_sym_result = MagicMock()
        mock_sym_result.scalars.return_value.all.return_value = [mock_src, mock_tgt]

        # No edges found
        mock_edges_result = MagicMock()
        mock_edges_result.scalars.return_value.all.return_value = []

        mock_session = AsyncMock()
        mock_session.execute.side_effect = [mock_sym_result, mock_edges_result]
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await trace_tool.ainvoke(
                {
                    "from_symbol": "a",
                    "to_symbol": "b",
                    "max_depth": 2,
                }
            )

        assert "No path found" in result

    @pytest.mark.anyio
    async def test_graph_trace_target_not_found(self) -> None:
        """Only source symbol exists."""
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        trace_tool = next(t for t in tools if t.name == "graph_trace")

        mock_sym = MagicMock()
        mock_sym.id = uuid.uuid4()
        mock_sym.name = "a"
        mock_sym.document = MagicMock()

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_sym]

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await trace_tool.ainvoke(
                {
                    "from_symbol": "a",
                    "to_symbol": "b",
                }
            )

        assert "Target symbol 'b' not found" in result


# ---------------------------------------------------------------------------
# kg_neighborhood - successful result
# ---------------------------------------------------------------------------


class TestKgNeighborhoodSuccess:
    """Tests for kg_neighborhood when nodes and neighborhoods are found."""

    @pytest.mark.anyio
    async def test_kg_neighborhood_success(self) -> None:
        from contextmine_core.graphrag import (
            Citation,
            ContextPack,
            EdgeContext,
            EntityContext,
        )

        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        kgn_tool = next(t for t in tools if t.name == "kg_neighborhood")

        mock_node = MagicMock()
        mock_node.id = uuid.uuid4()
        mock_node.collection_id = uuid.uuid4()
        mock_node.kind.value = "FILE"
        mock_node.name = "auth.py"

        mock_result_found = MagicMock()
        mock_result_found.scalar_one_or_none.return_value = mock_node

        mock_context = ContextPack(
            query="auth.py",
            entities=[
                EntityContext(
                    node_id=uuid.uuid4(),
                    kind="SYMBOL",
                    natural_key="sym:login",
                    name="login",
                    evidence=[
                        Citation(
                            file_path="src/auth.py",
                            start_line=10,
                            end_line=20,
                            snippet="def login()",
                        )
                    ],
                    relevance_score=0.9,
                ),
                EntityContext(
                    node_id=uuid.uuid4(),
                    kind="SYMBOL",
                    natural_key="sym:logout",
                    name="logout",
                    evidence=[],
                    relevance_score=0.5,
                ),
            ],
            edges=[
                EdgeContext(source_id="a", target_id="b", kind="DEFINES"),
                EdgeContext(source_id="c", target_id="d", kind="CALLS"),
            ],
        )

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result_found
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "contextmine_core.database.get_async_session",
                return_value=mock_session,
            ),
            patch(
                "contextmine_core.graphrag.graph_neighborhood",
                new_callable=AsyncMock,
                return_value=mock_context,
            ),
        ):
            result = await kgn_tool.ainvoke({"node_name": "auth.py", "depth": 2})

        assert "Neighborhood of FILE: auth.py" in result
        assert "SYMBOL" in result
        assert "login" in result
        assert "Relationships" in result
        assert "DEFINES" in result
        # One entity had evidence, so one evidence item should be created
        assert len(run.evidence) >= 1

    @pytest.mark.anyio
    async def test_kg_neighborhood_fallback_to_natural_key(self) -> None:
        """When name lookup fails, tries natural_key ilike fallback."""
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        kgn_tool = next(t for t in tools if t.name == "kg_neighborhood")

        mock_node = MagicMock()
        mock_node.id = uuid.uuid4()
        mock_node.collection_id = uuid.uuid4()
        mock_node.kind.value = "FILE"
        mock_node.name = "auth.py"

        mock_not_found = MagicMock()
        mock_not_found.scalar_one_or_none.return_value = None

        mock_found = MagicMock()
        mock_found.scalar_one_or_none.return_value = mock_node

        from contextmine_core.graphrag import ContextPack

        mock_context = ContextPack(query="auth")

        mock_session = AsyncMock()
        # First lookup by name: not found. Second lookup by natural_key: found.
        mock_session.execute.side_effect = [mock_not_found, mock_found]
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "contextmine_core.database.get_async_session",
                return_value=mock_session,
            ),
            patch(
                "contextmine_core.graphrag.graph_neighborhood",
                new_callable=AsyncMock,
                return_value=mock_context,
            ),
        ):
            result = await kgn_tool.ainvoke({"node_name": "auth"})

        assert "Neighborhood of FILE: auth.py" in result


# ---------------------------------------------------------------------------
# kg_path - successful result
# ---------------------------------------------------------------------------


class TestKgPathSuccess:
    """Tests for kg_path when nodes and paths are found."""

    @pytest.mark.anyio
    async def test_kg_path_success(self) -> None:
        from contextmine_core.graphrag import (
            Citation,
            ContextPack,
            EdgeContext,
            EntityContext,
            PathContext,
        )

        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        path_tool = next(t for t in tools if t.name == "kg_path")

        from_node = MagicMock()
        from_node.id = uuid.uuid4()
        from_node.collection_id = uuid.uuid4()

        to_node = MagicMock()
        to_node.id = uuid.uuid4()
        to_node.collection_id = uuid.uuid4()

        # First two lookups find nodes by name
        mock_from_result = MagicMock()
        mock_from_result.scalar_one_or_none.return_value = from_node

        mock_to_result = MagicMock()
        mock_to_result.scalar_one_or_none.return_value = to_node

        mock_context = ContextPack(
            query="a to b",
            entities=[
                EntityContext(
                    node_id=uuid.uuid4(),
                    kind="FILE",
                    natural_key="file:a",
                    name="a",
                    evidence=[
                        Citation(file_path="src/a.py", start_line=1, end_line=10, snippet="code a")
                    ],
                ),
                EntityContext(
                    node_id=uuid.uuid4(),
                    kind="FILE",
                    natural_key="file:b",
                    name="b",
                    evidence=[
                        Citation(file_path="src/b.py", start_line=1, end_line=10, snippet="code b")
                    ],
                ),
            ],
            edges=[
                EdgeContext(source_id="a", target_id="b", kind="IMPORTS"),
            ],
            paths=[
                PathContext(
                    nodes=["file:a", "file:b"], edges=["IMPORTS"], description="a -> b via import"
                ),
            ],
        )

        mock_session = AsyncMock()
        mock_session.execute.side_effect = [mock_from_result, mock_to_result]
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "contextmine_core.database.get_async_session",
                return_value=mock_session,
            ),
            patch(
                "contextmine_core.graphrag.trace_path",
                new_callable=AsyncMock,
                return_value=mock_context,
            ),
        ):
            result = await path_tool.ainvoke({"from_name": "a", "to_name": "b"})

        assert "Path: a" in result
        assert "Route:" in result
        assert "Steps:" in result
        assert len(run.evidence) >= 2  # Evidence for both path steps

    @pytest.mark.anyio
    async def test_kg_path_no_path_found(self) -> None:
        from contextmine_core.graphrag import ContextPack

        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        path_tool = next(t for t in tools if t.name == "kg_path")

        from_node = MagicMock()
        from_node.id = uuid.uuid4()
        from_node.collection_id = uuid.uuid4()

        to_node = MagicMock()
        to_node.id = uuid.uuid4()
        to_node.collection_id = uuid.uuid4()

        mock_from_result = MagicMock()
        mock_from_result.scalar_one_or_none.return_value = from_node

        mock_to_result = MagicMock()
        mock_to_result.scalar_one_or_none.return_value = to_node

        # Empty context = no path
        mock_context = ContextPack(query="a to b")

        mock_session = AsyncMock()
        mock_session.execute.side_effect = [mock_from_result, mock_to_result]
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "contextmine_core.database.get_async_session",
                return_value=mock_session,
            ),
            patch(
                "contextmine_core.graphrag.trace_path",
                new_callable=AsyncMock,
                return_value=mock_context,
            ),
        ):
            result = await path_tool.ainvoke({"from_name": "a", "to_name": "b"})

        assert "No path found" in result

    @pytest.mark.anyio
    async def test_kg_path_target_not_found(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        path_tool = next(t for t in tools if t.name == "kg_path")

        from_node = MagicMock()
        from_node.id = uuid.uuid4()
        from_node.collection_id = uuid.uuid4()

        mock_from_result = MagicMock()
        mock_from_result.scalar_one_or_none.return_value = from_node

        # Target not found by name OR natural_key
        mock_not_found = MagicMock()
        mock_not_found.scalar_one_or_none.return_value = None

        mock_session = AsyncMock()
        mock_session.execute.side_effect = [mock_from_result, mock_not_found, mock_not_found]
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await path_tool.ainvoke({"from_name": "found_node", "to_name": "missing_node"})

        assert "Target node" in result
        assert "not found" in result


# ---------------------------------------------------------------------------
# graph_expand - successful expansion with edges
# ---------------------------------------------------------------------------


class TestGraphExpandSuccess:
    """Tests for graph_expand when BFS finds connected symbols."""

    @pytest.mark.anyio
    async def test_graph_expand_with_edges(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        expand_tool = next(t for t in tools if t.name == "graph_expand")

        seed_id = uuid.uuid4()
        neighbor_id = uuid.uuid4()

        mock_seed_doc = MagicMock()
        mock_seed_doc.uri = "src/root.py"
        mock_seed_doc.content_markdown = "class Root:\n    pass"

        mock_seed = MagicMock()
        mock_seed.id = seed_id
        mock_seed.name = "Root"
        mock_seed.kind.value = "class"
        mock_seed.qualified_name = "root.Root"
        mock_seed.start_line = 1
        mock_seed.end_line = 2
        mock_seed.document = mock_seed_doc

        mock_neighbor_doc = MagicMock()
        mock_neighbor_doc.uri = "src/child.py"
        mock_neighbor_doc.content_markdown = "class Child(Root):\n    pass"

        mock_neighbor = MagicMock()
        mock_neighbor.id = neighbor_id
        mock_neighbor.name = "Child"
        mock_neighbor.kind.value = "class"
        mock_neighbor.qualified_name = "child.Child"
        mock_neighbor.start_line = 1
        mock_neighbor.end_line = 2
        mock_neighbor.document = mock_neighbor_doc

        mock_edge = MagicMock()
        mock_edge.source_symbol = mock_neighbor
        mock_edge.target_symbol = mock_seed

        # Query 1: find seeds
        mock_seed_result = MagicMock()
        mock_seed_result.scalars.return_value.all.return_value = [mock_seed]

        # Query 2: get edges from frontier (depth 1)
        mock_edges_result = MagicMock()
        mock_edges_result.scalars.return_value.all.return_value = [mock_edge]

        # Query 3: no more edges (depth 2)
        mock_empty_edges = MagicMock()
        mock_empty_edges.scalars.return_value.all.return_value = []

        mock_session = AsyncMock()
        mock_session.execute.side_effect = [mock_seed_result, mock_edges_result, mock_empty_edges]
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await expand_tool.ainvoke(
                {
                    "seed_names": ["Root"],
                    "depth": 2,
                }
            )

        assert "Expanded to 2 symbols" in result
        assert "[SEED]" in result
        assert "Child" in result
        assert len(run.evidence) >= 2

    @pytest.mark.anyio
    async def test_graph_expand_with_edge_type_filter(self) -> None:
        """Verify edge_types parameter is handled."""
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        expand_tool = next(t for t in tools if t.name == "graph_expand")

        mock_doc = MagicMock()
        mock_doc.uri = "src/x.py"
        mock_doc.content_markdown = "class X:\n    pass"

        mock_sym = MagicMock()
        mock_sym.id = uuid.uuid4()
        mock_sym.name = "X"
        mock_sym.kind.value = "class"
        mock_sym.qualified_name = "x.X"
        mock_sym.start_line = 1
        mock_sym.end_line = 2
        mock_sym.document = mock_doc

        mock_seed_result = MagicMock()
        mock_seed_result.scalars.return_value.all.return_value = [mock_sym]

        mock_empty_edges = MagicMock()
        mock_empty_edges.scalars.return_value.all.return_value = []

        mock_session = AsyncMock()
        mock_session.execute.side_effect = [mock_seed_result, mock_empty_edges]
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await expand_tool.ainvoke(
                {
                    "seed_names": ["X"],
                    "edge_types": ["calls", "inherits"],
                    "depth": 1,
                }
            )

        assert "Expanded to 1 symbols" in result


# ---------------------------------------------------------------------------
# symbol_find - partial match fallback
# ---------------------------------------------------------------------------


class TestSymbolFindPartialMatch:
    """Tests for symbol_find partial (ilike) match fallback."""

    @pytest.mark.anyio
    async def test_symbol_find_partial_match(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        find_tool = next(t for t in tools if t.name == "symbol_find")

        mock_doc = MagicMock()
        mock_doc.uri = "src/models.py"
        mock_doc.content_markdown = "class UserProfile:\n    name: str"

        mock_sym = MagicMock()
        mock_sym.name = "UserProfile"
        mock_sym.kind.value = "class"
        mock_sym.qualified_name = "models.UserProfile"
        mock_sym.start_line = 1
        mock_sym.end_line = 2
        mock_sym.document = mock_doc

        # First query (exact match): empty
        mock_empty_result = MagicMock()
        mock_empty_result.scalars.return_value.all.return_value = []

        # Second query (partial match): found
        mock_found_result = MagicMock()
        mock_found_result.scalars.return_value.all.return_value = [mock_sym]

        mock_session = AsyncMock()
        mock_session.execute.side_effect = [mock_empty_result, mock_found_result]
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await find_tool.ainvoke({"name": "User"})

        assert "Found 1 symbol" in result
        assert "UserProfile" in result
        assert len(run.evidence) == 1


# ---------------------------------------------------------------------------
# find_references - no edges for found target
# ---------------------------------------------------------------------------


class TestFindReferencesNoEdges:
    """Test find_references when target exists but has no incoming edges."""

    @pytest.mark.anyio
    async def test_find_references_target_found_but_no_edges(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        ref_tool = next(t for t in tools if t.name == "find_references")

        mock_target = MagicMock()
        mock_target.id = uuid.uuid4()

        mock_target_result = MagicMock()
        mock_target_result.scalars.return_value.all.return_value = [mock_target]

        mock_edges_result = MagicMock()
        mock_edges_result.scalars.return_value.all.return_value = []

        mock_session = AsyncMock()
        mock_session.execute.side_effect = [mock_target_result, mock_edges_result]
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await ref_tool.ainvoke({"symbol_name": "unused_func"})

        assert "No references found" in result


# ---------------------------------------------------------------------------
# symbol_callers - no callers for existing target
# ---------------------------------------------------------------------------


class TestSymbolCallersNoCallers:
    """Test symbol_callers when target exists but has no CALLS edges."""

    @pytest.mark.anyio
    async def test_symbol_callers_no_callers(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        callers_tool = next(t for t in tools if t.name == "symbol_callers")

        mock_target = MagicMock()
        mock_target.id = uuid.uuid4()
        mock_target.incoming_edges = []

        mock_target_result = MagicMock()
        mock_target_result.scalars.return_value.all.return_value = [mock_target]

        mock_edges_result = MagicMock()
        mock_edges_result.scalars.return_value.all.return_value = []

        mock_session = AsyncMock()
        mock_session.execute.side_effect = [mock_target_result, mock_edges_result]
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await callers_tool.ainvoke({"name": "private_helper"})

        assert "No callers found" in result


# ---------------------------------------------------------------------------
# symbol_callees - no callees for existing source
# ---------------------------------------------------------------------------


class TestSymbolCalleesNoCallees:
    """Test symbol_callees when source exists but has no outgoing CALLS edges."""

    @pytest.mark.anyio
    async def test_symbol_callees_no_callees(self) -> None:
        run_holder = _make_run_holder()
        tools = create_tools(run_holder)
        callees_tool = next(t for t in tools if t.name == "symbol_callees")

        mock_source = MagicMock()
        mock_source.id = uuid.uuid4()
        mock_source.outgoing_edges = []

        mock_source_result = MagicMock()
        mock_source_result.scalars.return_value.all.return_value = [mock_source]

        mock_edges_result = MagicMock()
        mock_edges_result.scalars.return_value.all.return_value = []

        mock_session = AsyncMock()
        mock_session.execute.side_effect = [mock_source_result, mock_edges_result]
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "contextmine_core.database.get_async_session",
            return_value=mock_session,
        ):
            result = await callees_tool.ainvoke({"name": "leaf_function"})

        assert "No callees found" in result


# ---------------------------------------------------------------------------
# Multiple hybrid_search results
# ---------------------------------------------------------------------------


class TestHybridSearchMultipleResults:
    """Test hybrid_search with multiple result items."""

    @pytest.mark.anyio
    async def test_hybrid_search_multiple_results(self) -> None:
        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        search_tool = next(t for t in tools if t.name == "hybrid_search")

        mock_results = [
            SimpleNamespace(uri=f"src/file{i}.py", content=f"content {i}", score=0.9 - i * 0.1)
            for i in range(3)
        ]
        mock_search_results = SimpleNamespace(results=mock_results)

        mock_embedder = AsyncMock()
        mock_embedder.embed_batch.return_value = SimpleNamespace(embeddings=[[0.1]])

        mock_settings = MagicMock()
        mock_settings.default_embedding_model = "openai:text-embedding-3-small"

        with (
            patch("contextmine_core.settings.get_settings", return_value=mock_settings),
            patch(
                "contextmine_core.embeddings.parse_embedding_model_spec",
                return_value=("openai", "model"),
            ),
            patch("contextmine_core.embeddings.get_embedder", return_value=mock_embedder),
            patch(
                "contextmine_core.search.hybrid_search",
                new_callable=AsyncMock,
                return_value=mock_search_results,
            ),
        ):
            result = await search_tool.ainvoke({"query": "test", "k": 3})

        assert "Found 3 results" in result
        assert len(run.evidence) == 3
        for ev in run.evidence:
            assert ev.provenance == "hybrid"


# ---------------------------------------------------------------------------
# graphrag_search - entity without evidence
# ---------------------------------------------------------------------------


class TestGraphragSearchEntityWithoutEvidence:
    """Test graphrag_search handles entities without evidence citations."""

    @pytest.mark.anyio
    async def test_entity_without_evidence(self) -> None:
        from contextmine_core.graphrag import (
            ContextPack,
            EntityContext,
        )

        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        grag_tool = next(t for t in tools if t.name == "graphrag_search")

        mock_context = ContextPack(
            query="test",
            entities=[
                EntityContext(
                    node_id=uuid.uuid4(),
                    kind="DB_TABLE",
                    natural_key="table:users",
                    name="users",
                    evidence=[],  # No evidence
                    relevance_score=0.8,
                ),
            ],
        )

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "contextmine_core.database.get_async_session",
                return_value=mock_session,
            ),
            patch(
                "contextmine_core.graphrag.graph_rag_context",
                new_callable=AsyncMock,
                return_value=mock_context,
            ),
        ):
            result = await grag_tool.ainvoke({"query": "users table"})

        # Entity without evidence should still appear with a dash marker
        assert "DB_TABLE: users" in result
        # No evidence should be added to run
        assert len(run.evidence) == 0


# ---------------------------------------------------------------------------
# graphrag_search - community summary truncation
# ---------------------------------------------------------------------------


class TestGraphragCommunityTruncation:
    """Test that long community summaries get truncated."""

    @pytest.mark.anyio
    async def test_long_community_summary_truncated(self) -> None:
        from contextmine_core.graphrag import (
            CommunityContext,
            ContextPack,
        )

        run = _make_run()
        run_holder = _make_run_holder(run)
        tools = create_tools(run_holder)
        grag_tool = next(t for t in tools if t.name == "graphrag_search")

        long_summary = "A" * 500

        mock_context = ContextPack(
            query="test",
            communities=[
                CommunityContext(
                    community_id=uuid.uuid4(),
                    level=0,
                    title="Big Module",
                    summary=long_summary,
                    relevance_score=0.9,
                    member_count=10,
                ),
            ],
        )

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "contextmine_core.database.get_async_session",
                return_value=mock_session,
            ),
            patch(
                "contextmine_core.graphrag.graph_rag_context",
                new_callable=AsyncMock,
                return_value=mock_context,
            ),
        ):
            result = await grag_tool.ainvoke({"query": "test"})

        assert "Big Module" in result
        # Summary should be truncated to ~300 chars + "..."
        assert "..." in result
