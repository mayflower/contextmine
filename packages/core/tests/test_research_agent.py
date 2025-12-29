"""Tests for the research agent."""

import pytest
from contextmine_core.research import (
    AgentConfig,
    ResearchRun,
)
from contextmine_core.research.actions import (
    ActionRegistry,
    ActionSelection,
    FinalizeAction,
    MockHybridSearchAction,
    OpenSpanAction,
)
from contextmine_core.research.run import RunStatus


class TestActionSchemas:
    """Tests for action schemas (still used by MCP server)."""

    def test_action_selection_with_hybrid_search(self) -> None:
        """Test creating ActionSelection for hybrid_search."""
        selection = ActionSelection(
            action="hybrid_search",
            reasoning="Need to find relevant code",
            hybrid_search={"query": "authentication", "k": 10},
        )
        assert selection.action == "hybrid_search"
        assert selection.get_action_input() is not None
        assert selection.get_action_input().query == "authentication"  # type: ignore[union-attr]

    def test_action_selection_with_finalize(self) -> None:
        """Test creating ActionSelection for finalize."""
        selection = ActionSelection(
            action="finalize",
            reasoning="Have sufficient evidence",
            finalize={"answer": "The answer is 42", "confidence": 0.95},
        )
        assert selection.action == "finalize"
        input_params = selection.get_action_input()
        assert input_params is not None
        assert input_params.answer == "The answer is 42"  # type: ignore[union-attr]


class TestActionRegistry:
    """Tests for action registry (still used by MCP server)."""

    def test_register_and_get_action(self) -> None:
        """Test registering and retrieving actions."""
        registry = ActionRegistry()
        action = FinalizeAction()
        registry.register(action)

        retrieved = registry.get("finalize")
        assert retrieved is not None
        assert retrieved.name == "finalize"

    def test_list_actions(self) -> None:
        """Test listing all actions."""
        registry = ActionRegistry()
        registry.register(FinalizeAction())
        registry.register(MockHybridSearchAction())

        actions = registry.list_actions()
        assert "finalize" in actions
        assert "hybrid_search" in actions


class TestMockHybridSearchAction:
    """Tests for the mock hybrid search action."""

    @pytest.mark.anyio
    async def test_execute_with_mock_results(self) -> None:
        """Test executing mock hybrid search."""
        run = ResearchRun.create(question="How does search work?")
        action = MockHybridSearchAction(
            mock_results=[
                {
                    "file_path": "src/search.py",
                    "start_line": 10,
                    "end_line": 20,
                    "content": "def search(): pass",
                    "score": 0.9,
                }
            ]
        )

        result = await action.execute(run, {"query": "search", "k": 5})

        assert result.success
        assert len(result.evidence) == 1
        assert result.evidence[0].file_path == "src/search.py"

    @pytest.mark.anyio
    async def test_execute_without_query(self) -> None:
        """Test that missing query returns error."""
        run = ResearchRun.create(question="Test")
        action = MockHybridSearchAction()

        result = await action.execute(run, {})

        assert not result.success
        assert "query" in result.error.lower()


class TestOpenSpanAction:
    """Tests for the open span action."""

    @pytest.mark.anyio
    async def test_execute_with_valid_file(self, tmp_path) -> None:
        """Test reading a valid file span."""
        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text("line 1\nline 2\nline 3\nline 4\nline 5\n")

        run = ResearchRun.create(question="Test")
        action = OpenSpanAction(base_path=tmp_path)

        result = await action.execute(
            run,
            {
                "file_path": "test.py",
                "start_line": 2,
                "end_line": 4,
            },
        )

        assert result.success
        assert len(result.evidence) == 1
        assert "line 2" in result.evidence[0].content
        assert "line 4" in result.evidence[0].content

    @pytest.mark.anyio
    async def test_execute_with_nonexistent_file(self, tmp_path) -> None:
        """Test that nonexistent file returns error."""
        run = ResearchRun.create(question="Test")
        action = OpenSpanAction(base_path=tmp_path)

        result = await action.execute(
            run,
            {
                "file_path": "nonexistent.py",
                "start_line": 1,
                "end_line": 10,
            },
        )

        assert not result.success
        assert "not" in result.error.lower() or "exist" in result.error.lower()


class TestFinalizeAction:
    """Tests for the finalize action."""

    @pytest.mark.anyio
    async def test_finalize_returns_success(self) -> None:
        """Test that finalize returns success for verification."""
        run = ResearchRun.create(question="Test question")
        action = FinalizeAction()

        result = await action.execute(
            run,
            {
                "answer": "The answer is here",
                "confidence": 0.9,
            },
        )

        # Finalize now returns success but doesn't complete the run
        # (verification is done by LangGraph verify node)
        assert result.success
        assert "answer" in result.data

    @pytest.mark.anyio
    async def test_finalize_requires_answer(self) -> None:
        """Test that finalize requires an answer."""
        run = ResearchRun.create(question="Test question")
        action = FinalizeAction()

        result = await action.execute(run, {})

        assert not result.success
        assert "answer" in result.error.lower()


class TestAgentConfig:
    """Tests for agent configuration."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = AgentConfig()
        assert config.max_steps == 10
        assert config.store_artifacts is True
        assert config.max_verification_retries == 2

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = AgentConfig(
            max_steps=5,
            store_artifacts=False,
            max_verification_retries=3,
        )
        assert config.max_steps == 5
        assert config.store_artifacts is False
        assert config.max_verification_retries == 3


class TestResearchRun:
    """Tests for ResearchRun creation and state."""

    def test_create_run(self) -> None:
        """Test creating a research run."""
        run = ResearchRun.create(
            question="How does X work?",
            scope="src/**",
            budget_steps=5,
        )

        assert run.question == "How does X work?"
        assert run.scope == "src/**"
        assert run.budget_steps == 5
        assert run.status == RunStatus.RUNNING
        assert run.answer is None

    def test_complete_run(self) -> None:
        """Test completing a research run."""
        run = ResearchRun.create(question="Test")
        run.complete("This is the answer")

        assert run.status == RunStatus.DONE
        assert run.answer == "This is the answer"

    def test_fail_run(self) -> None:
        """Test failing a research run."""
        run = ResearchRun.create(question="Test")
        run.fail("Something went wrong")

        assert run.status == RunStatus.ERROR
        assert run.error_message == "Something went wrong"
