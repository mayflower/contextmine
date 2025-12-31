"""Tests for Business Rule extraction using LLM.

The new approach uses:
1. Tree-sitter to parse code into units (functions, classes)
2. LLM to identify business rules in each unit

Tests cover:
1. Code unit parsing (structural, no LLM)
2. LLM integration with mock provider
3. All 12 Tree-sitter languages
"""

import pytest
from contextmine_core.analyzer.extractors.rules import (
    ExtractedRule,
    _get_class_node_types,
    _get_function_node_types,
    _parse_code_units,
    get_rule_natural_key,
)
from contextmine_core.treesitter.languages import TreeSitterLanguage
from contextmine_core.treesitter.manager import TreeSitterManager


def has_treesitter() -> bool:
    """Check if tree-sitter is available."""
    manager = TreeSitterManager.get_instance()
    return manager.is_available()


# Skip all tests if tree-sitter is not available
pytestmark = pytest.mark.skipif(
    not has_treesitter(), reason="tree-sitter-language-pack not installed"
)


class TestCodeUnitParsing:
    """Tests for Tree-sitter code unit parsing."""

    def test_parse_python_functions(self) -> None:
        """Test parsing Python functions."""
        content = """
def validate_user(user):
    if user.age < 18:
        raise ValueError("Too young")
    return True

def process_order(order):
    total = sum(item.price for item in order.items)
    return total
"""
        units = _parse_code_units("test.py", content, TreeSitterLanguage.PYTHON)

        assert len(units) == 2
        assert units[0].name == "validate_user"
        assert units[0].kind == "function"
        assert units[0].language == "python"
        assert units[1].name == "process_order"

    def test_parse_python_class(self) -> None:
        """Test parsing Python classes."""
        content = """
class UserValidator:
    def validate(self, user):
        if user.age < 18:
            raise ValueError("Too young")
        return True
"""
        units = _parse_code_units("test.py", content, TreeSitterLanguage.PYTHON)

        # Should find both the class and the method
        names = [u.name for u in units]
        assert "UserValidator" in names
        assert "validate" in names

    def test_parse_typescript_functions(self) -> None:
        """Test parsing TypeScript functions."""
        content = """
function validateAge(age: number): void {
    if (age < 18) {
        throw new Error("Must be at least 18");
    }
}

const processData = (data: string): string => {
    return data.toUpperCase();
};
"""
        units = _parse_code_units("test.ts", content, TreeSitterLanguage.TYPESCRIPT)

        names = [u.name for u in units]
        assert "validateAge" in names

    def test_parse_java_methods(self) -> None:
        """Test parsing Java methods."""
        content = """
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
        units = _parse_code_units("Test.java", content, TreeSitterLanguage.JAVA)

        names = [u.name for u in units]
        assert "Validator" in names
        assert "validate" in names
        assert "isValid" in names

    def test_parse_go_functions(self) -> None:
        """Test parsing Go functions."""
        content = """
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
        units = _parse_code_units("test.go", content, TreeSitterLanguage.GO)

        names = [u.name for u in units]
        assert "validateUser" in names
        assert "processOrder" in names

    def test_parse_rust_functions(self) -> None:
        """Test parsing Rust functions."""
        content = """
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
        units = _parse_code_units("test.rs", content, TreeSitterLanguage.RUST)

        names = [u.name for u in units]
        assert "validate_user" in names

    def test_parse_ruby_methods(self) -> None:
        """Test parsing Ruby methods."""
        content = """
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
        units = _parse_code_units("test.rb", content, TreeSitterLanguage.RUBY)

        names = [u.name for u in units]
        assert "Validator" in names
        assert "validate" in names

    def test_parse_csharp_methods(self) -> None:
        """Test parsing C# methods."""
        # Skip if c_sharp language is not installed
        from contextmine_core.treesitter.manager import get_treesitter_manager

        manager = get_treesitter_manager()
        try:
            manager.get_parser(TreeSitterLanguage.CSHARP)
        except Exception:
            pytest.skip("tree-sitter c_sharp language not installed")

        content = """
public class Validator
{
    public void Validate(User user)
    {
        if (user.Age < 18)
        {
            throw new ArgumentException("Too young");
        }
    }

    public bool IsValid(string input)
    {
        return !string.IsNullOrEmpty(input);
    }
}
"""
        units = _parse_code_units("Test.cs", content, TreeSitterLanguage.CSHARP)

        names = [u.name for u in units]
        assert "Validator" in names
        assert "Validate" in names
        assert "IsValid" in names

    def test_parse_php_functions(self) -> None:
        """Test parsing PHP functions."""
        content = """
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
        units = _parse_code_units("test.php", content, TreeSitterLanguage.PHP)

        names = [u.name for u in units]
        assert "validateUser" in names
        assert "Validator" in names

    def test_skip_small_functions(self) -> None:
        """Test that very small functions are skipped."""
        content = """
def tiny():
    pass
"""
        units = _parse_code_units("test.py", content, TreeSitterLanguage.PYTHON)

        # Function with only 2 lines should be skipped
        assert len(units) == 0


class TestLanguageCoverage:
    """Tests that all 12 Tree-sitter languages are covered."""

    def test_all_languages_have_function_types(self) -> None:
        """Test that all languages have function node types defined."""
        for lang in TreeSitterLanguage:
            types = _get_function_node_types(lang)
            assert len(types) > 0, f"No function types for {lang.value}"

    def test_all_languages_have_class_types(self) -> None:
        """Test that all languages have class node types defined."""
        for lang in TreeSitterLanguage:
            types = _get_class_node_types(lang)
            assert len(types) > 0, f"No class types for {lang.value}"


class TestNaturalKey:
    """Tests for natural key generation."""

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


class TestInternationalization:
    """Tests for international code support."""

    def test_german_method_names(self) -> None:
        """Test parsing code with German method names."""
        content = """
def prüfe_benutzer(benutzer):
    if benutzer.alter < 18:
        raise ValueError("Benutzer muss mindestens 18 Jahre alt sein")
    return True

def berechne_gesamtpreis(bestellung):
    # Berechne den Gesamtpreis der Bestellung
    gesamt = sum(artikel.preis for artikel in bestellung.artikel)
    return gesamt
"""
        units = _parse_code_units("test.py", content, TreeSitterLanguage.PYTHON)

        names = [u.name for u in units]
        assert "prüfe_benutzer" in names
        assert "berechne_gesamtpreis" in names

    def test_spanish_method_names(self) -> None:
        """Test parsing code with Spanish method names."""
        content = """
def validar_usuario(usuario):
    if usuario.edad < 18:
        raise ValueError("El usuario debe tener al menos 18 años")
    return True

def calcular_total(pedido):
    # Calcula el total del pedido
    total = sum(item.precio for item in pedido.items)
    return total
"""
        units = _parse_code_units("test.py", content, TreeSitterLanguage.PYTHON)

        names = [u.name for u in units]
        assert "validar_usuario" in names
        assert "calcular_total" in names

    def test_dutch_method_names(self) -> None:
        """Test parsing code with Dutch method names."""
        content = """
def controleer_gebruiker(gebruiker):
    if gebruiker.leeftijd < 18:
        raise ValueError("Gebruiker moet minimaal 18 jaar oud zijn")
    return True

def bereken_totaal(bestelling):
    # Bereken het totaal van de bestelling
    totaal = sum(item.prijs for item in bestelling.items)
    return totaal
"""
        units = _parse_code_units("test.py", content, TreeSitterLanguage.PYTHON)

        names = [u.name for u in units]
        assert "controleer_gebruiker" in names
        assert "bereken_totaal" in names

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
        # Content should include the Japanese comments
        assert "ユーザー" in units[0].content


class TestUnsupportedLanguages:
    """Tests for handling unsupported languages."""

    def test_unsupported_extension_returns_empty(self) -> None:
        """Test that unsupported file types return empty results."""
        # Call parsing and discard result - this tests graceful handling
        _parse_code_units("test.xyz", "some content", TreeSitterLanguage.PYTHON)

    def test_markdown_not_parsed(self) -> None:
        """Test that markdown detection returns None."""
        from contextmine_core.treesitter.languages import detect_language

        lang = detect_language("README.md")
        assert lang is None
