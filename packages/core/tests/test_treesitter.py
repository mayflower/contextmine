"""Tests for the Tree-sitter module.

Focuses on actual behavior: language detection, symbol extraction, caching.
"""

from pathlib import Path

import pytest
from contextmine_core.treesitter import (
    Symbol,
    SymbolKind,
    TreeSitterLanguage,
    TreeSitterManager,
    detect_language,
    extract_outline,
    find_enclosing_symbol,
    find_symbol_by_name,
    get_symbol_content,
)


class TestLanguageDetection:
    """Tests for language detection - consolidated with parametrize."""

    @pytest.mark.parametrize(
        "path,expected",
        [
            # Python
            ("test.py", TreeSitterLanguage.PYTHON),
            ("test.pyi", TreeSitterLanguage.PYTHON),
            ("/path/to/file.py", TreeSitterLanguage.PYTHON),
            # TypeScript/JavaScript
            ("test.ts", TreeSitterLanguage.TYPESCRIPT),
            ("test.tsx", TreeSitterLanguage.TSX),
            ("test.js", TreeSitterLanguage.JAVASCRIPT),
            ("test.jsx", TreeSitterLanguage.JAVASCRIPT),
            ("test.mjs", TreeSitterLanguage.JAVASCRIPT),
            # Other languages
            ("test.rs", TreeSitterLanguage.RUST),
            ("test.go", TreeSitterLanguage.GO),
            ("test.java", TreeSitterLanguage.JAVA),
            ("test.c", TreeSitterLanguage.C),
            ("test.h", TreeSitterLanguage.C),
            ("test.cpp", TreeSitterLanguage.CPP),
            ("test.hpp", TreeSitterLanguage.CPP),
            # Unsupported
            ("test.txt", None),
            ("test.md", None),
            ("test.json", None),
            # Git URIs with query params
            ("git://github.com/owner/repo/path/file.py?ref=main", TreeSitterLanguage.PYTHON),
            ("git://github.com/owner/repo/src/app.ts?ref=develop", TreeSitterLanguage.TYPESCRIPT),
            ("git://repo/file.rs?ref=main&sha=abc123", TreeSitterLanguage.RUST),
            ("git://repo/README.md?ref=main", None),
        ],
    )
    def test_detect_language(self, path: str, expected: TreeSitterLanguage | None) -> None:
        """Test language detection from file paths."""
        assert detect_language(path) == expected


class TestTreeSitterCache:
    """Tests for caching behavior - these catch real bugs."""

    def test_cache_hit(self, tmp_path: Path) -> None:
        """Test that caching works - same content returns same tree."""
        TreeSitterManager.reset()
        manager = TreeSitterManager.get_instance()

        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    pass\n")

        tree1 = manager.parse(test_file)
        tree2 = manager.parse(test_file)

        assert tree1 is tree2
        TreeSitterManager.reset()

    def test_cache_invalidation(self, tmp_path: Path) -> None:
        """Test cache invalidation on content change."""
        TreeSitterManager.reset()
        manager = TreeSitterManager.get_instance()

        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    pass\n")
        tree1 = manager.parse(test_file)

        test_file.write_text("def world():\n    pass\n")
        tree2 = manager.parse(test_file)

        assert tree1 is not tree2
        TreeSitterManager.reset()


class TestSymbolExtraction:
    """Tests for symbol extraction - actual parsing behavior."""

    def test_extract_python_outline(self, tmp_path: Path) -> None:
        """Test extracting Python function and class definitions."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def foo():
    pass

def bar(x, y):
    return x + y

class MyClass:
    def __init__(self):
        pass

    def method(self):
        pass
""")

        symbols = extract_outline(test_file, include_children=True)
        names = [s.name for s in symbols]

        assert "foo" in names
        assert "bar" in names
        assert "MyClass" in names

    def test_extract_typescript_outline(self, tmp_path: Path) -> None:
        """Test extracting TypeScript function and class definitions."""
        test_file = tmp_path / "test.ts"
        test_file.write_text("""
function greet(name: string): string {
    return `Hello, ${name}!`;
}

class Greeter {
    greet(name: string): string {
        return `Hello, ${name}!`;
    }
}
""")

        symbols = extract_outline(test_file, include_children=True)
        names = [s.name for s in symbols]

        assert "greet" in names
        assert "Greeter" in names

    def test_find_symbol_by_name(self, tmp_path: Path) -> None:
        """Test finding a specific symbol."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def foo():
    pass

def bar():
    pass

def baz():
    pass
""")

        symbol = find_symbol_by_name(test_file, "bar")
        assert symbol is not None
        assert symbol.name == "bar"
        assert symbol.kind == SymbolKind.FUNCTION

        # Not found case
        assert find_symbol_by_name(test_file, "nonexistent") is None

    def test_find_enclosing_symbol(self, tmp_path: Path) -> None:
        """Test finding the enclosing symbol for a line."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""def foo():
    x = 1
    y = 2
    return x + y

def bar():
    pass
""")

        # Line 3 should be inside foo
        symbol = find_enclosing_symbol(test_file, 3)
        assert symbol is not None
        assert symbol.name == "foo"

        # Line 7 should be inside bar
        symbol = find_enclosing_symbol(test_file, 7)
        assert symbol is not None
        assert symbol.name == "bar"

    def test_get_symbol_content(self, tmp_path: Path) -> None:
        """Test getting symbol content."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""def foo():
    return 42
""")

        symbol = find_symbol_by_name(test_file, "foo")
        assert symbol is not None

        content = get_symbol_content(symbol)
        assert "def foo():" in content
        assert "return 42" in content

    def test_symbol_contains_line(self) -> None:
        """Test Symbol.contains_line() method."""
        symbol = Symbol(
            name="test_func",
            kind=SymbolKind.FUNCTION,
            file_path="/path/to/file.py",
            start_line=10,
            end_line=20,
            start_column=0,
            end_column=10,
            signature="def test_func():",
        )

        assert symbol.contains_line(10) is True
        assert symbol.contains_line(15) is True
        assert symbol.contains_line(20) is True
        assert symbol.contains_line(5) is False
        assert symbol.contains_line(25) is False
