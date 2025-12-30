"""Tests for Business Rule Candidate extraction."""

import pytest
from contextmine_core.analyzer.extractors.rules import (
    FailureKind,
    extract_rule_candidates,
    get_candidate_natural_key,
)
from contextmine_core.treesitter.manager import TreeSitterManager


def has_treesitter() -> bool:
    """Check if tree-sitter is available."""
    manager = TreeSitterManager.get_instance()
    return manager.is_available()


# Skip all tests if tree-sitter is not available
pytestmark = pytest.mark.skipif(
    not has_treesitter(), reason="tree-sitter-language-pack not installed"
)


class TestPythonRuleExtraction:
    """Tests for Python rule candidate extraction."""

    def test_extract_raise_exception(self) -> None:
        """Test extracting rule from if/raise pattern."""
        content = """
def validate_user(user):
    if user.age < 18:
        raise ValueError("User must be at least 18 years old")
    return True
"""
        result = extract_rule_candidates("test.py", content)

        assert len(result.candidates) == 1
        candidate = result.candidates[0]
        assert candidate.failure_kind == FailureKind.RAISE_EXCEPTION
        assert "age" in candidate.predicate_text.lower()
        assert "ValueError" in candidate.failure_text
        assert candidate.container_name == "validate_user"
        assert candidate.language == "python"

    def test_extract_not_condition_raise(self) -> None:
        """Test extracting rule with negated condition."""
        content = """
def check_permission(user):
    if not user.is_admin:
        raise PermissionError("Admin access required")
"""
        result = extract_rule_candidates("test.py", content)

        assert len(result.candidates) == 1
        candidate = result.candidates[0]
        assert candidate.failure_kind == FailureKind.RAISE_EXCEPTION
        assert "admin" in candidate.predicate_text.lower()

    def test_extract_assert_statement(self) -> None:
        """Test extracting rule from assert statement."""
        content = """
def process_data(data):
    assert data is not None, "Data cannot be None"
    return data.value
"""
        result = extract_rule_candidates("test.py", content)

        assert len(result.candidates) == 1
        candidate = result.candidates[0]
        assert candidate.failure_kind == FailureKind.ASSERT_FAIL
        assert "None" in candidate.predicate_text

    def test_extract_return_error(self) -> None:
        """Test extracting rule from if/return error pattern."""
        content = """
def validate_input(value):
    if value is None:
        return None  # Error case
    return value
"""
        result = extract_rule_candidates("test.py", content)

        # Should detect the None return as potential error
        assert len(result.candidates) == 1
        candidate = result.candidates[0]
        assert candidate.failure_kind == FailureKind.RETURN_ERROR

    def test_extract_multiple_rules(self) -> None:
        """Test extracting multiple rule candidates."""
        content = """
def validate_order(order):
    if order.quantity <= 0:
        raise ValueError("Quantity must be positive")

    if order.price < 0:
        raise ValueError("Price cannot be negative")

    if not order.customer:
        raise ValueError("Customer is required")

    return True
"""
        result = extract_rule_candidates("test.py", content)

        assert len(result.candidates) == 3

    def test_no_rules_in_simple_code(self) -> None:
        """Test that simple code without rules returns empty."""
        content = """
def add(a, b):
    return a + b

def multiply(x, y):
    result = x * y
    return result
"""
        result = extract_rule_candidates("test.py", content)

        assert len(result.candidates) == 0

    def test_evidence_lines_correct(self) -> None:
        """Test that evidence line numbers are correct."""
        content = """
def test():
    x = 1
    if x < 0:
        raise ValueError("Negative")
"""
        result = extract_rule_candidates("test.py", content)

        assert len(result.candidates) == 1
        candidate = result.candidates[0]
        # The if statement starts at line 4 (1-indexed)
        assert candidate.start_line == 4
        assert candidate.end_line == 5


class TestTypeScriptRuleExtraction:
    """Tests for TypeScript/JavaScript rule candidate extraction."""

    def test_extract_throw_error(self) -> None:
        """Test extracting rule from if/throw pattern."""
        content = """
function validateAge(age: number): void {
    if (age < 18) {
        throw new Error("Must be at least 18");
    }
}
"""
        result = extract_rule_candidates("test.ts", content)

        assert len(result.candidates) == 1
        candidate = result.candidates[0]
        assert candidate.failure_kind == FailureKind.THROW_ERROR
        assert "age" in candidate.predicate_text.lower()
        assert candidate.language == "typescript"

    def test_extract_negated_condition(self) -> None:
        """Test extracting rule with negated condition."""
        content = """
function checkAuth(user) {
    if (!user.isAuthenticated) {
        throw new AuthError("Not authenticated");
    }
}
"""
        result = extract_rule_candidates("test.js", content)

        assert len(result.candidates) == 1
        candidate = result.candidates[0]
        assert "Authenticated" in candidate.predicate_text

    def test_extract_return_null(self) -> None:
        """Test extracting rule from if/return null pattern."""
        content = """
function getUser(id: string): User | null {
    if (!id) {
        return null;
    }
    return db.findUser(id);
}
"""
        result = extract_rule_candidates("test.ts", content)

        assert len(result.candidates) == 1
        candidate = result.candidates[0]
        assert candidate.failure_kind == FailureKind.RETURN_ERROR


class TestConfidenceScoring:
    """Tests for confidence score calculation."""

    def test_validation_keywords_increase_confidence(self) -> None:
        """Test that validation keywords increase confidence."""
        # Code with validation keywords
        content_with_keywords = """
def validate_user(user):
    if not is_valid(user):
        raise ValidationError("Invalid user")
"""
        result1 = extract_rule_candidates("test.py", content_with_keywords)

        # Code without validation keywords
        content_without = """
def process(x):
    if x < 0:
        raise ValueError("Negative")
"""
        result2 = extract_rule_candidates("test2.py", content_without)

        assert len(result1.candidates) == 1
        assert len(result2.candidates) == 1

        # The validation keyword version should have higher confidence
        assert result1.candidates[0].confidence >= result2.candidates[0].confidence


class TestNaturalKey:
    """Tests for natural key generation."""

    def test_natural_key_unique(self) -> None:
        """Test that different rules get different keys."""
        content = """
def test():
    if x < 0:
        raise ValueError("A")
    if y < 0:
        raise ValueError("B")
"""
        result = extract_rule_candidates("test.py", content)

        assert len(result.candidates) == 2
        key1 = get_candidate_natural_key(result.candidates[0])
        key2 = get_candidate_natural_key(result.candidates[1])
        assert key1 != key2

    def test_natural_key_stable(self) -> None:
        """Test that the same rule gets the same key."""
        content = """
def test():
    if x < 0:
        raise ValueError("Error")
"""
        result1 = extract_rule_candidates("test.py", content)
        result2 = extract_rule_candidates("test.py", content)

        key1 = get_candidate_natural_key(result1.candidates[0])
        key2 = get_candidate_natural_key(result2.candidates[0])
        assert key1 == key2


class TestUnsupportedLanguages:
    """Tests for handling unsupported languages."""

    def test_unsupported_extension_returns_empty(self) -> None:
        """Test that unsupported file types return empty results."""
        content = "some content"
        result = extract_rule_candidates("test.xyz", content)
        assert len(result.candidates) == 0

    def test_markdown_returns_empty(self) -> None:
        """Test that markdown files return empty results."""
        content = "# Title\n\nSome text"
        result = extract_rule_candidates("README.md", content)
        assert len(result.candidates) == 0
