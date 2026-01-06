"""Tests for community summary generation."""

from uuid import UUID

import pytest
from contextmine_core.knowledge.summaries import (
    CommunityContext,
    CommunitySummaryOutput,
    _build_summary_prompt,
    _format_llm_summary,
)


class TestCommunityContext:
    """Tests for community context dataclass."""

    def test_context_creation(self) -> None:
        """Test creating a community context with semantic entities."""
        context = CommunityContext(
            community_id=UUID("12345678-1234-5678-1234-567812345678"),
            level=1,
            member_nodes=[{"name": "User Authentication", "type": "CONCEPT"}],
            entity_names=["User Authentication"],
            entity_types={"CONCEPT": 1},
        )

        assert context.level == 1
        assert len(context.member_nodes) == 1
        assert context.entity_names == ["User Authentication"]

    def test_context_with_all_fields(self) -> None:
        """Test context with all semantic entity fields."""
        context = CommunityContext(
            community_id=UUID("12345678-1234-5678-1234-567812345678"),
            level=0,
            member_nodes=[
                {
                    "name": "Payment Processing",
                    "type": "COMPONENT",
                    "description": "Handles payments",
                },
            ],
            entity_names=["Payment Processing"],
            entity_types={"COMPONENT": 1},
            entity_descriptions=["Handles payments"],
            source_symbols=["PaymentService", "PaymentController"],
        )

        assert context.entity_descriptions == ["Handles payments"]
        assert "PaymentService" in context.source_symbols


class TestBuildSummaryPrompt:
    """Tests for prompt building."""

    def test_prompt_includes_semantic_entities(self) -> None:
        """Test that prompt includes semantic entity sections."""
        context = CommunityContext(
            community_id=UUID("12345678-1234-5678-1234-567812345678"),
            level=1,
            member_nodes=[
                {
                    "name": "User Authentication",
                    "type": "CONCEPT",
                    "description": "Handles user login",
                },
            ],
            entity_names=["User Authentication"],
            entity_types={"CONCEPT": 1},
            entity_descriptions=["Handles user login"],
            source_symbols=["AuthService"],
            evidence_snippets=["def authenticate(): pass"],
        )

        prompt = _build_summary_prompt(context)

        assert "Level: 1" in prompt
        assert "## Domain Concepts (Semantic Entities)" in prompt
        assert "User Authentication" in prompt
        assert "CONCEPT" in prompt
        assert "## Entity Types" in prompt
        assert "## Associated Code Symbols" in prompt
        assert "AuthService" in prompt
        assert "## Code Snippets" in prompt

    def test_prompt_with_minimal_context(self) -> None:
        """Test prompt building with minimal context."""
        context = CommunityContext(
            community_id=UUID("12345678-1234-5678-1234-567812345678"),
            level=0,
            member_nodes=[{"name": "Test Entity", "type": "OTHER"}],
            entity_names=["Test Entity"],
        )

        prompt = _build_summary_prompt(context)

        assert "Level: 0" in prompt
        assert "Test Entity" in prompt
        # Empty sections should not appear
        assert "## Entity Descriptions" not in prompt


class TestFormatLLMSummary:
    """Tests for LLM summary formatting."""

    def test_format_full_summary(self) -> None:
        """Test formatting a complete LLM summary."""
        summary = CommunitySummaryOutput(
            title="Data Processing Pipeline",
            responsibilities=["Process incoming data", "Validate format"],
            key_concepts=["Pipeline", "Validator"],
            key_dependencies=["pandas", "pydantic"],
            key_paths=["src/pipeline.py"],
            confidence=0.85,
        )

        text = _format_llm_summary(summary)

        assert "# Data Processing Pipeline" in text
        assert "## Responsibilities" in text
        assert "- Process incoming data" in text
        assert "## Key Concepts" in text
        assert "- Pipeline" in text
        assert "## Dependencies" in text
        assert "- pandas" in text
        assert "## Key Paths" in text
        assert "- src/pipeline.py" in text
        assert "Confidence: 85%" in text

    def test_format_minimal_summary(self) -> None:
        """Test formatting summary with only required fields."""
        summary = CommunitySummaryOutput(
            title="Simple Module",
            responsibilities=["Do one thing"],
            key_concepts=["Thing"],
            confidence=0.5,
        )

        text = _format_llm_summary(summary)

        assert "# Simple Module" in text
        assert "## Responsibilities" in text
        assert "## Dependencies" not in text  # Empty list


class TestCommunitySummaryOutput:
    """Tests for the Pydantic schema."""

    def test_valid_summary(self) -> None:
        """Test creating a valid summary."""
        summary = CommunitySummaryOutput(
            title="Test",
            responsibilities=["Do stuff"],
            key_concepts=["Concept"],
            confidence=0.9,
        )

        assert summary.title == "Test"
        assert summary.confidence == 0.9
        assert summary.key_dependencies == []

    def test_confidence_bounds(self) -> None:
        """Test that confidence must be between 0 and 1."""
        with pytest.raises(ValueError):
            CommunitySummaryOutput(
                title="Test",
                responsibilities=["Do stuff"],
                key_concepts=["Concept"],
                confidence=1.5,  # Invalid
            )
