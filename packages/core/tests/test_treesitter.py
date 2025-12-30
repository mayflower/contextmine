"""Tests for the Tree-sitter module."""

from pathlib import Path

from contextmine_core.treesitter import (
    EXTENSION_TO_LANGUAGE,
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
    """Tests for language detection."""

    def test_detect_python(self) -> None:
        """Test detecting Python files."""
        assert detect_language("test.py") == TreeSitterLanguage.PYTHON
        assert detect_language("test.pyi") == TreeSitterLanguage.PYTHON
        assert detect_language("/path/to/file.py") == TreeSitterLanguage.PYTHON

    def test_detect_typescript(self) -> None:
        """Test detecting TypeScript files."""
        assert detect_language("test.ts") == TreeSitterLanguage.TYPESCRIPT
        assert detect_language("test.tsx") == TreeSitterLanguage.TSX

    def test_detect_javascript(self) -> None:
        """Test detecting JavaScript files."""
        assert detect_language("test.js") == TreeSitterLanguage.JAVASCRIPT
        assert detect_language("test.jsx") == TreeSitterLanguage.JAVASCRIPT
        assert detect_language("test.mjs") == TreeSitterLanguage.JAVASCRIPT

    def test_detect_rust(self) -> None:
        """Test detecting Rust files."""
        assert detect_language("test.rs") == TreeSitterLanguage.RUST

    def test_detect_go(self) -> None:
        """Test detecting Go files."""
        assert detect_language("test.go") == TreeSitterLanguage.GO

    def test_detect_java(self) -> None:
        """Test detecting Java files."""
        assert detect_language("test.java") == TreeSitterLanguage.JAVA

    def test_detect_c_cpp(self) -> None:
        """Test detecting C/C++ files."""
        assert detect_language("test.c") == TreeSitterLanguage.C
        assert detect_language("test.h") == TreeSitterLanguage.C
        assert detect_language("test.cpp") == TreeSitterLanguage.CPP
        assert detect_language("test.hpp") == TreeSitterLanguage.CPP

    def test_detect_unsupported(self) -> None:
        """Test that unsupported file types return None."""
        assert detect_language("test.txt") is None
        assert detect_language("test.md") is None
        assert detect_language("test.json") is None

    def test_detect_with_query_params(self) -> None:
        """Test that URIs with query parameters are handled correctly.

        Git URIs from GitHub sources have the format:
        git://github.com/owner/repo/path/file.py?ref=main
        """
        # Git URIs with ref query param
        assert (
            detect_language("git://github.com/owner/repo/path/file.py?ref=main")
            == TreeSitterLanguage.PYTHON
        )
        assert (
            detect_language("git://github.com/owner/repo/src/app.ts?ref=develop")
            == TreeSitterLanguage.TYPESCRIPT
        )
        assert (
            detect_language("git://github.com/owner/repo/lib/utils.go?ref=v1.0.0")
            == TreeSitterLanguage.GO
        )

        # Multiple query params
        assert detect_language("git://repo/file.rs?ref=main&sha=abc123") == TreeSitterLanguage.RUST

        # Unsupported file with query params
        assert detect_language("git://repo/README.md?ref=main") is None

    def test_extension_to_language_completeness(self) -> None:
        """Test that extension mapping covers expected languages."""
        languages_with_extensions = set(EXTENSION_TO_LANGUAGE.values())
        assert TreeSitterLanguage.PYTHON in languages_with_extensions
        assert TreeSitterLanguage.TYPESCRIPT in languages_with_extensions
        assert TreeSitterLanguage.JAVASCRIPT in languages_with_extensions
        assert TreeSitterLanguage.RUST in languages_with_extensions
        assert TreeSitterLanguage.GO in languages_with_extensions


class TestTreeSitterManager:
    """Tests for the TreeSitterManager."""

    def test_singleton_pattern(self) -> None:
        """Test that TreeSitterManager is a singleton."""
        TreeSitterManager.reset()
        manager1 = TreeSitterManager.get_instance()
        manager2 = TreeSitterManager.get_instance()
        assert manager1 is manager2
        TreeSitterManager.reset()

    def test_is_available(self) -> None:
        """Test that tree-sitter is available."""
        TreeSitterManager.reset()
        manager = TreeSitterManager.get_instance()
        assert manager.is_available() is True
        TreeSitterManager.reset()

    def test_get_parser(self) -> None:
        """Test getting a parser for a language."""
        TreeSitterManager.reset()
        manager = TreeSitterManager.get_instance()
        parser = manager.get_parser(TreeSitterLanguage.PYTHON)
        assert parser is not None
        TreeSitterManager.reset()

    def test_parse_content(self, tmp_path: Path) -> None:
        """Test parsing file content."""
        TreeSitterManager.reset()
        manager = TreeSitterManager.get_instance()

        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    pass\n")

        tree = manager.parse(test_file)
        assert tree is not None
        assert tree.root_node.type == "module"
        TreeSitterManager.reset()

    def test_parse_with_content(self, tmp_path: Path) -> None:
        """Test parsing with provided content."""
        TreeSitterManager.reset()
        manager = TreeSitterManager.get_instance()

        test_file = tmp_path / "test.py"
        content = "class Foo:\n    def bar(self):\n        pass\n"

        tree = manager.parse(test_file, content=content)
        assert tree is not None
        TreeSitterManager.reset()

    def test_cache_hit(self, tmp_path: Path) -> None:
        """Test that caching works."""
        TreeSitterManager.reset()
        manager = TreeSitterManager.get_instance()

        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    pass\n")

        tree1 = manager.parse(test_file)
        tree2 = manager.parse(test_file)

        # Should get the same tree object from cache
        assert tree1 is tree2
        TreeSitterManager.reset()

    def test_cache_invalidation(self, tmp_path: Path) -> None:
        """Test cache invalidation on content change."""
        TreeSitterManager.reset()
        manager = TreeSitterManager.get_instance()

        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    pass\n")

        tree1 = manager.parse(test_file)

        # Change content
        test_file.write_text("def world():\n    pass\n")
        tree2 = manager.parse(test_file)

        # Should get a different tree
        assert tree1 is not tree2
        TreeSitterManager.reset()

    def test_cache_stats(self) -> None:
        """Test cache statistics."""
        TreeSitterManager.reset()
        manager = TreeSitterManager.get_instance()

        stats = manager.get_cache_stats()
        assert "cached_trees" in stats
        assert "max_size" in stats
        assert "parsers_loaded" in stats
        TreeSitterManager.reset()


class TestSymbolExtraction:
    """Tests for symbol extraction."""

    def test_extract_python_functions(self, tmp_path: Path) -> None:
        """Test extracting Python function definitions."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def foo():
    pass

def bar(x, y):
    return x + y
""")

        symbols = extract_outline(test_file)
        assert len(symbols) == 2
        assert symbols[0].name == "foo"
        assert symbols[0].kind == SymbolKind.FUNCTION
        assert symbols[1].name == "bar"

    def test_extract_python_class(self, tmp_path: Path) -> None:
        """Test extracting Python class definitions."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
class MyClass:
    def __init__(self):
        pass

    def method(self):
        pass
""")

        symbols = extract_outline(test_file, include_children=True)
        assert len(symbols) >= 1
        class_sym = symbols[0]
        assert class_sym.name == "MyClass"
        assert class_sym.kind == SymbolKind.CLASS

    def test_extract_typescript_functions(self, tmp_path: Path) -> None:
        """Test extracting TypeScript function definitions."""
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
        assert len(symbols) >= 2

        # Should find function and class
        names = [s.name for s in symbols]
        assert "greet" in names
        assert "Greeter" in names

    def test_find_symbol_by_name(self, tmp_path: Path) -> None:
        """Test finding a symbol by name."""
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

    def test_find_symbol_not_found(self, tmp_path: Path) -> None:
        """Test finding a symbol that doesn't exist."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo(): pass\n")

        symbol = find_symbol_by_name(test_file, "nonexistent")
        assert symbol is None

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
        content = """def foo():
    return 42
"""
        test_file.write_text(content)

        symbol = find_symbol_by_name(test_file, "foo")
        assert symbol is not None

        symbol_content = get_symbol_content(symbol)
        assert "def foo():" in symbol_content
        assert "return 42" in symbol_content

    def test_symbol_dataclass(self) -> None:
        """Test Symbol dataclass methods."""
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

        d = symbol.to_dict()
        assert d["name"] == "test_func"
        assert d["kind"] == "function"
        assert d["start_line"] == 10
