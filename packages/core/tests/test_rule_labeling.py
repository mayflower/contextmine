"""Tests for LLM-based business rule extraction models."""

import pytest
from contextmine_core.analyzer.extractors.rules import (
    BusinessRuleDef,
    ExtractedRule,
    ExtractionOutput,
    RulesExtraction,
    get_rule_natural_key,
)


class TestBusinessRuleDef:
    """Tests for the BusinessRuleDef Pydantic schema."""

    def test_valid_business_rule(self) -> None:
        """Test that valid business rule passes validation."""
        rule = BusinessRuleDef(
            name="Age Validation",
            description="Validates that users are at least 18 years old",
            category="validation",
            severity="error",
            natural_language="Users must be at least 18 years old",
            evidence_snippet="if user.age < 18: raise ValueError('Too young')",
            start_line=10,
            end_line=12,
        )
        assert rule.name == "Age Validation"
        assert rule.category == "validation"
        assert rule.severity == "error"

    def test_all_required_fields(self) -> None:
        """Test that all fields are required."""
        data = {
            "name": "Test",
            # Missing other fields
        }
        with pytest.raises(ValueError):
            BusinessRuleDef.model_validate(data)


class TestExtractionOutput:
    """Tests for ExtractionOutput schema."""

    def test_extraction_with_rules(self) -> None:
        """Test extraction output with rules."""
        output = ExtractionOutput(
            rules=[
                BusinessRuleDef(
                    name="Test Rule",
                    description="Test description",
                    category="validation",
                    severity="error",
                    natural_language="Test rule in natural language",
                    evidence_snippet="if x < 0: raise",
                    start_line=1,
                    end_line=2,
                )
            ],
            reasoning="Found one validation rule",
        )
        assert len(output.rules) == 1
        assert output.rules[0].name == "Test Rule"
        assert output.reasoning == "Found one validation rule"

    def test_extraction_with_no_rules(self) -> None:
        """Test extraction output when no rules found."""
        output = ExtractionOutput(
            rules=[],
            reasoning="No business rules found in this code",
        )
        assert len(output.rules) == 0
        assert "No business rules" in output.reasoning


class TestExtractedRule:
    """Tests for ExtractedRule dataclass."""

    def test_extracted_rule_creation(self) -> None:
        """Test creating an ExtractedRule."""
        rule = ExtractedRule(
            name="Test Rule",
            description="Test description",
            category="validation",
            severity="error",
            natural_language="Test rule",
            file_path="test.py",
            start_line=10,
            end_line=15,
            evidence_snippet="if x < 0: raise",
            container_name="validate_x",
            language="python",
        )
        assert rule.name == "Test Rule"
        assert rule.file_path == "test.py"
        assert rule.container_name == "validate_x"


class TestRulesExtraction:
    """Tests for RulesExtraction dataclass."""

    def test_rules_extraction_creation(self) -> None:
        """Test creating a RulesExtraction."""
        extraction = RulesExtraction(
            file_path="test.py",
            rules=[
                ExtractedRule(
                    name="Rule 1",
                    description="Desc 1",
                    category="validation",
                    severity="error",
                    natural_language="Rule one",
                    file_path="test.py",
                    start_line=1,
                    end_line=5,
                    evidence_snippet="code",
                    container_name="func1",
                    language="python",
                )
            ],
        )
        assert extraction.file_path == "test.py"
        assert len(extraction.rules) == 1

    def test_empty_extraction(self) -> None:
        """Test creating an empty extraction."""
        extraction = RulesExtraction(file_path="empty.py")
        assert extraction.file_path == "empty.py"
        assert len(extraction.rules) == 0


class TestNaturalKey:
    """Tests for natural key generation."""

    def test_natural_key_unique_for_different_rules(self) -> None:
        """Test that different rules get different keys."""
        rule1 = ExtractedRule(
            name="Rule A",
            description="Desc A",
            category="validation",
            severity="error",
            natural_language="Must be positive",
            file_path="test.py",
            start_line=10,
            end_line=15,
            evidence_snippet="if x < 0: raise",
            container_name="validate_a",
            language="python",
        )
        rule2 = ExtractedRule(
            name="Rule B",
            description="Desc B",
            category="validation",
            severity="error",
            natural_language="Must be non-empty",
            file_path="test.py",
            start_line=20,
            end_line=25,
            evidence_snippet="if not x: raise",
            container_name="validate_b",
            language="python",
        )

        key1 = get_rule_natural_key(rule1)
        key2 = get_rule_natural_key(rule2)
        assert key1 != key2

    def test_natural_key_stable(self) -> None:
        """Test that the same rule gets the same key."""
        rule = ExtractedRule(
            name="Rule",
            description="Description",
            category="validation",
            severity="error",
            natural_language="Users must be 18+",
            file_path="auth.py",
            start_line=10,
            end_line=15,
            evidence_snippet="if age < 18: raise",
            container_name="validate_age",
            language="python",
        )

        key1 = get_rule_natural_key(rule)
        key2 = get_rule_natural_key(rule)
        assert key1 == key2


class TestCategoryValues:
    """Tests for valid category values."""

    def test_valid_categories(self) -> None:
        """Test that valid categories are accepted."""
        categories = [
            "validation",
            "authorization",
            "invariant",
            "constraint",
            "rate_limit",
            "business_logic",
        ]
        for category in categories:
            rule = BusinessRuleDef(
                name="Test",
                description="Test",
                category=category,
                severity="error",
                natural_language="Test",
                evidence_snippet="code",
                start_line=1,
                end_line=2,
            )
            assert rule.category == category


class TestSeverityValues:
    """Tests for valid severity values."""

    def test_valid_severities(self) -> None:
        """Test that valid severities are accepted."""
        severities = ["error", "warning", "info"]
        for severity in severities:
            rule = BusinessRuleDef(
                name="Test",
                description="Test",
                category="validation",
                severity=severity,
                natural_language="Test",
                evidence_snippet="code",
                start_line=1,
                end_line=2,
            )
            assert rule.severity == severity
