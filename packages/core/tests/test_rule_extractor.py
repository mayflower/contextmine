"""Tests for Business Rule extraction.

Tests the natural key generation and rule dataclass.
LLM-based extraction is tested via integration tests.
"""

import pytest
from contextmine_core.analyzer.extractors.rules import (
    ExtractedRule,
    get_rule_natural_key,
)
from contextmine_core.treesitter.languages import detect_language


class TestNaturalKey:
    """Tests for natural key generation - important for idempotency."""

    def test_natural_key_unique(self) -> None:
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

        assert get_rule_natural_key(rule1) != get_rule_natural_key(rule2)

    def test_natural_key_stable(self) -> None:
        """Test that the same rule gets the same key (idempotent)."""
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

    def test_natural_key_file_specific(self) -> None:
        """Test that same rule in different files gets different keys."""
        rule1 = ExtractedRule(
            name="Rule",
            description="Description",
            category="validation",
            severity="error",
            natural_language="Users must be 18+",
            file_path="auth.py",
            start_line=10,
            end_line=15,
            evidence_snippet="if age < 18: raise",
            container_name="<file>",
            language="python",
        )
        rule2 = ExtractedRule(
            name="Rule",
            description="Description",
            category="validation",
            severity="error",
            natural_language="Users must be 18+",
            file_path="validator.py",  # Different file
            start_line=10,
            end_line=15,
            evidence_snippet="if age < 18: raise",
            container_name="<file>",
            language="python",
        )

        assert get_rule_natural_key(rule1) != get_rule_natural_key(rule2)


class TestLanguageDetection:
    """Tests for language detection used in rule extraction."""

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("test.py", True),
            ("test.ts", True),
            ("test.tsx", True),
            ("test.js", True),
            ("Test.java", True),
            ("test.go", True),
            ("test.rs", True),
            ("test.rb", True),
            ("test.php", True),
            ("test.c", True),
            ("test.cpp", True),
            ("README.md", False),
            ("config.yaml", False),
            ("package.json", False),
        ],
    )
    def test_language_detection(self, filename: str, expected: bool) -> None:
        """Test that supported languages are detected."""
        result = detect_language(filename)
        if expected:
            assert result is not None, f"Expected {filename} to be detected"
        else:
            assert result is None, f"Expected {filename} to be ignored"
