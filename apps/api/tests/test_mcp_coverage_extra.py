"""Additional coverage tests for mcp_server.py uncovered lines.

Targets:
- research_validation: matched rules + candidates rendering (lines 1236-1265)
- research_architecture: UI topology + test semantics (lines 1550-1559)
- graph_rag: rebuild_mode entity counting (lines 1812-1838)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import app.mcp_server as mcp_mod
import pytest

pytestmark = pytest.mark.anyio

# Tool function references (unwrap FunctionTool decorator)
_research_validation = mcp_mod.research_validation.fn
_research_architecture = mcp_mod.research_architecture.fn
_graph_rag = mcp_mod.mcp_graph_rag.fn


def _mock_db_session(mock_db):
    ctx = patch("app.mcp_server.get_db_session")
    mock_session = ctx.start()
    mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db)
    mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
    return ctx


def _make_settings(**overrides):
    defaults = {
        "arch_docs_enabled": True,
        "arch_docs_generate_on_sync": False,
        "arch_docs_llm_enrich": False,
        "arch_docs_drift_enabled": False,
        "arch_docs_agent_sdk_model": "claude-3-5-sonnet-20241022",
        "arch_docs_agent_sdk_max_turns": 10,
        "arch_docs_agent_sdk_permission_mode": "never",
        "default_llm_provider": "openai",
        "repos_root": "/tmp/repos",
        "twin_analysis_cache_ttl_seconds": 300,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_knowledge_node(
    *,
    kind_value: str = "business_rule",
    name: str = "rule1",
    meta: dict | None = None,
    node_id: uuid.UUID | None = None,
):
    node = MagicMock()
    node.id = node_id or uuid.uuid4()
    node.name = name
    node.kind = MagicMock(value=kind_value)
    node.kind.__eq__ = lambda self, other: self.value == getattr(other, "value", other)
    node.meta = meta or {}
    return node


# ---------------------------------------------------------------------------
# research_validation: matched rules + candidates rendering
# Lines 1236-1265
# ---------------------------------------------------------------------------


class TestResearchValidationRendering:
    async def test_matched_rules_and_candidates_rendered(self) -> None:
        """When both rules and candidates match, output includes both sections."""
        mock_db = AsyncMock()
        coll_id = uuid.uuid4()

        rule = _make_knowledge_node(
            kind_value="business_rule",
            name="user_validation_rule",
            meta={
                "category": "validation",
                "severity": "error",
                "natural_language": "Users must have valid email",
                "predicate": "email_valid(user.email)",
                "failure": "InvalidEmailError",
            },
        )
        candidate = _make_knowledge_node(
            kind_value="rule_candidate",
            name="user_format_check",
            meta={
                "container_name": "validate_user_format",
                "file_path": "auth/validators.py",
                "predicate": "re.match(pattern, email)",
                "failure": "raise ValueError",
            },
        )

        # Pass collection_id to avoid get_accessible_collection_ids path.
        # With collection_id, calls are: 1=business_rules, 2=rule_candidates
        call_count = {"n": 0}

        async def mock_execute(stmt):
            call_count["n"] += 1
            result = MagicMock()
            if call_count["n"] == 1:
                # business rules query
                result.scalars.return_value.all.return_value = [rule]
                return result
            elif call_count["n"] == 2:
                # rule candidates query
                result.scalars.return_value.all.return_value = [candidate]
                return result
            else:
                # evidence queries
                result.scalars.return_value.all.return_value = []
                return result

        mock_db.execute = mock_execute
        ctx = _mock_db_session(mock_db)
        try:
            with patch("app.mcp_server.get_current_user_id", return_value=None):
                result = await _research_validation(
                    code_path="user validation",
                    collection_id=str(coll_id),
                )

            assert "Business Rules" in result
            assert "Validation Candidates" in result
            assert "user_validation_rule" in result or "Users must have valid email" in result
            assert "validate_user_format" in result
            assert "auth/validators.py" in result
        finally:
            ctx.stop()

    async def test_matched_candidates_only(self) -> None:
        mock_db = AsyncMock()
        coll_id = uuid.uuid4()

        candidate = _make_knowledge_node(
            kind_value="rule_candidate",
            name="check_rate_limit",
            meta={
                "container_name": "check_rate_limit",
                "file_path": "api/limits.py",
                "predicate": "rate > max_rate",
                "failure": "raise RateLimitExceeded",
            },
        )

        call_count = {"n": 0}

        async def mock_execute(stmt):
            call_count["n"] += 1
            result = MagicMock()
            if call_count["n"] == 1:
                # business rules query (empty)
                result.scalars.return_value.all.return_value = []
                return result
            elif call_count["n"] == 2:
                # rule candidates query
                result.scalars.return_value.all.return_value = [candidate]
                return result
            else:
                result.scalars.return_value.all.return_value = []
                return result

        mock_db.execute = mock_execute
        ctx = _mock_db_session(mock_db)
        try:
            with patch("app.mcp_server.get_current_user_id", return_value=None):
                result = await _research_validation(
                    code_path="rate limit",
                    collection_id=str(coll_id),
                )

            assert "Validation Candidates" in result
            assert "check_rate_limit" in result
        finally:
            ctx.stop()


# ---------------------------------------------------------------------------
# research_architecture: UI topology rendering (lines 1550-1559)
# ---------------------------------------------------------------------------


class TestResearchArchitectureUITopology:
    async def test_ui_topology_rendered(self) -> None:
        """When topic is 'ui', UI node kinds are counted and rendered."""
        mock_db = AsyncMock()
        coll_id = uuid.uuid4()

        ui_route = _make_knowledge_node(kind_value="ui_route", name="/dashboard")
        ui_view = _make_knowledge_node(kind_value="ui_view", name="DashboardView")
        ui_component = _make_knowledge_node(kind_value="ui_component", name="Sidebar")

        # Simulate the enum comparison
        from contextmine_core.models import KnowledgeNodeKind

        ui_route.kind = KnowledgeNodeKind.UI_ROUTE
        ui_view.kind = KnowledgeNodeKind.UI_VIEW
        ui_component.kind = KnowledgeNodeKind.UI_COMPONENT

        # With collection_id provided, first execute is the UI nodes query
        call_count = {"n": 0}

        async def mock_execute(stmt):
            call_count["n"] += 1
            result = MagicMock()
            if call_count["n"] == 1:
                # UI nodes query
                result.scalars.return_value.all.return_value = [ui_route, ui_view, ui_component]
                return result
            else:
                result.scalars.return_value.all.return_value = []
                return result

        mock_db.execute = mock_execute
        ctx = _mock_db_session(mock_db)
        try:
            with patch("app.mcp_server.get_current_user_id", return_value=None):
                result = await _research_architecture(
                    topic="ui screens",
                    collection_id=str(coll_id),
                )

            assert "UI Topology" in result
        finally:
            ctx.stop()


# ---------------------------------------------------------------------------
# graph_rag: rebuild_mode entity counting (lines 1812-1838)
# ---------------------------------------------------------------------------


@dataclass
class FakeEntity:
    kind: str
    id: str = ""


@dataclass
class FakeCitation:
    def format(self) -> str:
        return "file.py:10-20"


@dataclass
class FakeContextPack:
    entities: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    communities: list = field(default_factory=list)
    paths: list = field(default_factory=list)
    citations: list = field(default_factory=list)

    def to_dict(self):
        return {"entities": len(self.entities)}


@dataclass
class FakeGraphRAGResult:
    final_answer: str = "Test answer"
    communities_used: int = 3
    partial_answers: list = field(default_factory=list)
    context: FakeContextPack | None = None


class TestGraphRagRebuildMode:
    async def test_rebuild_mode_renders_entity_counts(self) -> None:
        """When answer=True and rebuild_mode=True, entity kind counts are rendered."""
        mock_db = AsyncMock()

        context = FakeContextPack(
            entities=[
                FakeEntity(kind="api_endpoint", id="e1"),
                FakeEntity(kind="api_endpoint", id="e2"),
                FakeEntity(kind="service_rpc", id="e3"),
                FakeEntity(kind="ui_route", id="e4"),
                FakeEntity(kind="ui_view", id="e5"),
                FakeEntity(kind="test_suite", id="e6"),
                FakeEntity(kind="user_flow", id="e7"),
            ],
            citations=[FakeCitation()],
        )
        rag_result = FakeGraphRAGResult(
            final_answer="Architecture overview here",
            communities_used=5,
            partial_answers=["p1", "p2"],
            context=context,
        )

        ctx = _mock_db_session(mock_db)
        try:
            with (
                patch("app.mcp_server.get_current_user_id", return_value=None),
                patch("app.mcp_server.get_settings", return_value=_make_settings()),
                patch(
                    "contextmine_core.graphrag.graph_rag_query",
                    new_callable=AsyncMock,
                    return_value=rag_result,
                ),
                patch(
                    "contextmine_core.research.llm.get_llm_provider",
                    return_value=MagicMock(),
                ),
            ):
                result = await _graph_rag(
                    query="architecture overview",
                    answer=True,
                    rebuild_mode=True,
                    twin_scope="all",
                )

            assert "Architecture overview here" in result
            assert "System Boundaries" in result
            assert "Interfaces" in result
            assert "UI Screens" in result
            assert "User Flows" in result
            assert "Test Obligations" in result
            assert "Key Citations" in result
        finally:
            ctx.stop()

    async def test_answer_mode_without_rebuild(self) -> None:
        """answer=True but rebuild_mode=False should not show entity breakdown."""
        mock_db = AsyncMock()

        context = FakeContextPack(
            entities=[FakeEntity(kind="api_endpoint")],
            citations=[FakeCitation()],
        )
        rag_result = FakeGraphRAGResult(
            final_answer="Simple answer",
            communities_used=2,
            partial_answers=["p1"],
            context=context,
        )

        ctx = _mock_db_session(mock_db)
        try:
            with (
                patch("app.mcp_server.get_current_user_id", return_value=None),
                patch("app.mcp_server.get_settings", return_value=_make_settings()),
                patch(
                    "contextmine_core.graphrag.graph_rag_query",
                    new_callable=AsyncMock,
                    return_value=rag_result,
                ),
                patch(
                    "contextmine_core.research.llm.get_llm_provider",
                    return_value=MagicMock(),
                ),
            ):
                result = await _graph_rag(
                    query="simple query",
                    answer=True,
                    rebuild_mode=False,
                )

            assert "Simple answer" in result
            assert "System Boundaries" not in result
        finally:
            ctx.stop()
