"""Tests for LLM-based rule candidate labeling."""

import pytest
from contextmine_core.analyzer.labeling import (
    BusinessRuleOutput,
    LabelingResult,
    _compute_content_hash,
)
from contextmine_core.research.llm.mock import MockLLMProvider


class TestBusinessRuleOutput:
    """Tests for the Pydantic output schema."""

    def test_valid_output(self) -> None:
        """Test that valid output passes validation."""
        data = {
            "name": "Age Validation",
            "description": "Validates that users are at least 18 years old",
            "category": "validation",
            "severity": "error",
            "natural_language": "Users must be at least 18 years old",
            "confidence": 0.85,
            "is_valid_rule": True,
            "reasoning": "This is a clear business requirement for age verification",
        }
        output = BusinessRuleOutput.model_validate(data)
        assert output.name == "Age Validation"
        assert output.is_valid_rule is True
        assert output.confidence == 0.85

    def test_confidence_bounds(self) -> None:
        """Test that confidence is bounded 0-1."""
        data = {
            "name": "Test",
            "description": "Test",
            "category": "validation",
            "severity": "error",
            "natural_language": "Test",
            "confidence": 1.5,  # Invalid - too high
            "is_valid_rule": True,
            "reasoning": "Test",
        }
        with pytest.raises(ValueError):  # Pydantic validation error
            BusinessRuleOutput.model_validate(data)

    def test_all_required_fields(self) -> None:
        """Test that all fields are required."""
        data = {
            "name": "Test",
            # Missing other fields
        }
        with pytest.raises(ValueError):
            BusinessRuleOutput.model_validate(data)


class TestLabelingResult:
    """Tests for labeling result structure."""

    def test_labeling_result_creation(self) -> None:
        """Test creating a labeling result."""
        rule = BusinessRuleOutput(
            name="Test Rule",
            description="Test description",
            category="validation",
            severity="error",
            natural_language="Test rule in natural language",
            confidence=0.8,
            is_valid_rule=True,
            reasoning="Test reasoning",
        )
        result = LabelingResult(
            candidate_id="test-id",
            rule=rule,
            raw_response='{"test": "response"}',
        )
        assert result.candidate_id == "test-id"
        assert result.rule.name == "Test Rule"
        assert result.raw_response is not None


class TestContentHash:
    """Tests for content hash computation."""

    def test_same_content_same_hash(self) -> None:
        """Test that identical content produces same hash."""
        meta1 = {"predicate": "x < 0", "failure": "raise ValueError", "failure_kind": "raise"}
        meta2 = {"predicate": "x < 0", "failure": "raise ValueError", "failure_kind": "raise"}
        assert _compute_content_hash(meta1) == _compute_content_hash(meta2)

    def test_different_content_different_hash(self) -> None:
        """Test that different content produces different hash."""
        meta1 = {"predicate": "x < 0", "failure": "raise ValueError", "failure_kind": "raise"}
        meta2 = {"predicate": "x > 0", "failure": "raise ValueError", "failure_kind": "raise"}
        assert _compute_content_hash(meta1) != _compute_content_hash(meta2)

    def test_missing_fields_handled(self) -> None:
        """Test that missing fields don't cause errors."""
        meta = {}
        hash_result = _compute_content_hash(meta)
        assert len(hash_result) == 16  # SHA256 truncated to 16 chars


@pytest.fixture
def anyio_backend() -> str:
    """Use asyncio as the async backend."""
    return "asyncio"


class TestMockProvider:
    """Tests for mock provider integration."""

    async def test_mock_structured_output(self, _anyio_backend: str) -> None:
        """Test that mock provider returns valid structured output."""
        provider = MockLLMProvider()
        provider.set_structured_response(
            "BusinessRuleOutput",
            {
                "name": "Mock Rule",
                "description": "Mock description",
                "category": "validation",
                "severity": "error",
                "natural_language": "This is a mock rule",
                "confidence": 0.9,
                "is_valid_rule": True,
                "reasoning": "Mock reasoning",
            },
        )

        result = await provider.generate_structured(
            system="Test system",
            messages=[{"role": "user", "content": "Test prompt"}],
            output_schema=BusinessRuleOutput,
            temperature=0.0,
        )

        assert isinstance(result, BusinessRuleOutput)
        assert result.name == "Mock Rule"
        assert result.is_valid_rule is True

    async def test_mock_records_calls(self, _anyio_backend: str) -> None:
        """Test that mock provider records call history."""
        provider = MockLLMProvider()
        provider.set_structured_response(
            "BusinessRuleOutput",
            {
                "name": "Test",
                "description": "Test",
                "category": "validation",
                "severity": "error",
                "natural_language": "Test",
                "confidence": 0.5,
                "is_valid_rule": False,
                "reasoning": "Test",
            },
        )

        await provider.generate_structured(
            system="System prompt",
            messages=[{"role": "user", "content": "User prompt"}],
            output_schema=BusinessRuleOutput,
        )

        assert len(provider.call_history) == 1
        assert provider.call_history[0]["system"] == "System prompt"
        assert provider.call_history[0]["output_schema"] == "BusinessRuleOutput"


class TestLabelingValidation:
    """Tests for labeling validation rules."""

    def test_invalid_rule_not_created(self) -> None:
        """Test that invalid rules are marked but not treated as rules."""
        rule = BusinessRuleOutput(
            name="Null Check",
            description="Generic null check",
            category="other",
            severity="error",
            natural_language="Value cannot be null",
            confidence=0.3,
            is_valid_rule=False,  # Marked as not a real business rule
            reasoning="This is defensive coding, not a business rule",
        )
        assert rule.is_valid_rule is False


class TestLabelingEdgeCases:
    """Tests for edge cases in labeling."""

    def test_empty_meta_handling(self) -> None:
        """Test handling of candidates with empty metadata."""
        hash_result = _compute_content_hash({})
        assert hash_result is not None

    def test_special_characters_in_predicate(self) -> None:
        """Test handling of special characters."""
        meta = {
            "predicate": 'user.name == "O\'Connor"',
            "failure": "raise ValueError('Invalid name')",
            "failure_kind": "raise",
        }
        hash_result = _compute_content_hash(meta)
        assert len(hash_result) == 16
