"""Tests for Business Rule extraction.

Focuses on actual parsing behavior across languages, not LLM integration.
"""

import pytest
from contextmine_core.analyzer.extractors.rules import (
    ExtractedRule,
    _get_class_node_types,
    _get_function_node_types,
    _parse_code_units,
    get_rule_natural_key,
)
from contextmine_core.treesitter.languages import TreeSitterLanguage, detect_language
from contextmine_core.treesitter.manager import TreeSitterManager


def has_treesitter() -> bool:
    """Check if tree-sitter is available."""
    manager = TreeSitterManager.get_instance()
    return manager.is_available()


pytestmark = pytest.mark.skipif(
    not has_treesitter(), reason="tree-sitter-language-pack not installed"
)


# Test fixtures for each language
PYTHON_CODE = """
def validate_user(user):
    if user.age < 18:
        raise ValueError("Too young")
    return True

def process_order(order):
    total = sum(item.price for item in order.items)
    return total
"""

TYPESCRIPT_CODE = """
function validateAge(age: number): void {
    if (age < 18) {
        throw new Error("Must be at least 18");
    }
}

const processData = (data: string): string => {
    return data.toUpperCase();
};
"""

JAVA_CODE = """
public class Validator {
    public void validate(User user) {
        if (user.getAge() < 18) {
            throw new IllegalArgumentException("Too young");
        }
    }

    public boolean isValid(String input) {
        return input != null && !input.isEmpty();
    }
}
"""

GO_CODE = """
func validateUser(user *User) error {
    if user.Age < 18 {
        return errors.New("user must be 18+")
    }
    return nil
}

func processOrder(order Order) (float64, error) {
    if order.Total < 0 {
        return 0, errors.New("invalid total")
    }
    return order.Total, nil
}
"""

RUST_CODE = """
fn validate_user(user: &User) -> Result<(), ValidationError> {
    if user.age < 18 {
        return Err(ValidationError::TooYoung);
    }
    Ok(())
}

impl UserValidator {
    fn check(&self, user: &User) -> bool {
        user.is_valid()
    }
}
"""

RUBY_CODE = """
class Validator
  def validate(user)
    raise ArgumentError, "Too young" if user.age < 18
    true
  end

  def valid?(input)
    !input.nil? && !input.empty?
  end
end
"""

PHP_CODE = """
<?php
function validateUser($user) {
    if ($user->age < 18) {
        throw new InvalidArgumentException("Too young");
    }
    return true;
}

class Validator {
    public function validate($input) {
        if (empty($input)) {
            throw new Exception("Empty input");
        }
    }
}
"""


class TestPolyglotParsing:
    """Tests for parsing code across multiple languages."""

    @pytest.mark.parametrize(
        "filename,content,language,expected_names",
        [
            ("test.py", PYTHON_CODE, TreeSitterLanguage.PYTHON, ["validate_user", "process_order"]),
            ("test.ts", TYPESCRIPT_CODE, TreeSitterLanguage.TYPESCRIPT, ["validateAge"]),
            ("Test.java", JAVA_CODE, TreeSitterLanguage.JAVA, ["Validator", "validate", "isValid"]),
            ("test.go", GO_CODE, TreeSitterLanguage.GO, ["validateUser", "processOrder"]),
            ("test.rs", RUST_CODE, TreeSitterLanguage.RUST, ["validate_user"]),
            ("test.rb", RUBY_CODE, TreeSitterLanguage.RUBY, ["Validator", "validate"]),
            ("test.php", PHP_CODE, TreeSitterLanguage.PHP, ["validateUser", "Validator"]),
        ],
    )
    def test_parse_language(
        self,
        filename: str,
        content: str,
        language: TreeSitterLanguage,
        expected_names: list[str],
    ) -> None:
        """Test parsing code units for each supported language."""
        units = _parse_code_units(filename, content, language)
        names = [u.name for u in units]

        for expected in expected_names:
            assert expected in names, f"Expected {expected} in {names}"

    def test_all_languages_have_node_types(self) -> None:
        """Test that all languages have function and class node types defined."""
        for lang in TreeSitterLanguage:
            func_types = _get_function_node_types(lang)
            class_types = _get_class_node_types(lang)
            assert len(func_types) > 0, f"No function types for {lang.value}"
            assert len(class_types) > 0, f"No class types for {lang.value}"

    def test_skip_small_functions(self) -> None:
        """Test that very small functions are skipped."""
        content = """
def tiny():
    pass
"""
        units = _parse_code_units("test.py", content, TreeSitterLanguage.PYTHON)
        assert len(units) == 0  # Function with only 2 lines should be skipped


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


class TestInternationalCode:
    """Tests for international character support - real edge case."""

    @pytest.mark.parametrize(
        "content,expected_name",
        [
            # German
            (
                """
def prüfe_benutzer(benutzer):
    if benutzer.alter < 18:
        raise ValueError("Benutzer muss mindestens 18 Jahre alt sein")
    return True
""",
                "prüfe_benutzer",
            ),
            # Spanish
            (
                """
def validar_usuario(usuario):
    if usuario.edad < 18:
        raise ValueError("El usuario debe tener al menos 18 años")
    return True
""",
                "validar_usuario",
            ),
            # Dutch
            (
                """
def controleer_gebruiker(gebruiker):
    if gebruiker.leeftijd < 18:
        raise ValueError("Gebruiker moet minimaal 18 jaar oud zijn")
    return True
""",
                "controleer_gebruiker",
            ),
        ],
    )
    def test_international_method_names(self, content: str, expected_name: str) -> None:
        """Test parsing code with international method names."""
        units = _parse_code_units("test.py", content, TreeSitterLanguage.PYTHON)
        names = [u.name for u in units]
        assert expected_name in names

    def test_japanese_comments(self) -> None:
        """Test parsing code with Japanese comments."""
        content = """
def validate_user(user):
    # ユーザーは18歳以上でなければなりません
    if user.age < 18:
        raise ValueError("ユーザーは18歳以上である必要があります")
    return True
"""
        units = _parse_code_units("test.py", content, TreeSitterLanguage.PYTHON)

        assert len(units) == 1
        assert units[0].name == "validate_user"
        assert "ユーザー" in units[0].content


class TestUnsupportedLanguages:
    """Tests for graceful handling of unsupported files."""

    def test_markdown_not_parsed(self) -> None:
        """Test that markdown detection returns None."""
        assert detect_language("README.md") is None
        assert detect_language("docs/guide.md") is None
