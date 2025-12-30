"""Tests for the research agent."""

from contextmine_core.research import (
    AgentConfig,
    ResearchRun,
)
from contextmine_core.research.run import RunStatus


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
