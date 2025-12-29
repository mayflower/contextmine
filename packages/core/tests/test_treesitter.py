"""Tests for the Tree-sitter module."""

from pathlib import Path

import pytest
from contextmine_core.research.actions.schemas import (
    ActionSelection,
    TsEnclosingSymbolInput,
    TsFindSymbolInput,
    TsOutlineInput,
)
from contextmine_core.research.actions.treesitter import (
    MockTsFindSymbolAction,
    MockTsOutlineAction,
    TsEnclosingSymbolAction,
    TsFindSymbolAction,
    TsOutlineAction,
)
from contextmine_core.research.run import ResearchRun
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


class TestTreeSitterActionSchemas:
    """Tests for Tree-sitter action schemas."""

    def test_ts_outline_input(self) -> None:
        """Test TsOutlineInput validation."""
        input_data = TsOutlineInput(file_path="test.py")
        assert input_data.file_path == "test.py"

    def test_ts_find_symbol_input(self) -> None:
        """Test TsFindSymbolInput validation."""
        input_data = TsFindSymbolInput(file_path="test.py", name="my_function")
        assert input_data.file_path == "test.py"
        assert input_data.name == "my_function"

    def test_ts_enclosing_symbol_input(self) -> None:
        """Test TsEnclosingSymbolInput validation."""
        input_data = TsEnclosingSymbolInput(file_path="test.py", line=10)
        assert input_data.file_path == "test.py"
        assert input_data.line == 10

    def test_action_selection_with_ts_outline(self) -> None:
        """Test ActionSelection with ts_outline action."""
        selection = ActionSelection(
            action="ts_outline",
            reasoning="Need to understand file structure",
            ts_outline={"file_path": "test.py"},
        )
        assert selection.action == "ts_outline"
        input_params = selection.get_action_input()
        assert input_params is not None
        assert input_params.file_path == "test.py"  # type: ignore[union-attr]

    def test_action_selection_with_ts_find_symbol(self) -> None:
        """Test ActionSelection with ts_find_symbol action."""
        selection = ActionSelection(
            action="ts_find_symbol",
            reasoning="Find specific function",
            ts_find_symbol={"file_path": "test.py", "name": "my_func"},
        )
        assert selection.action == "ts_find_symbol"

    def test_action_selection_with_ts_enclosing_symbol(self) -> None:
        """Test ActionSelection with ts_enclosing_symbol action."""
        selection = ActionSelection(
            action="ts_enclosing_symbol",
            reasoning="Find what contains this line",
            ts_enclosing_symbol={"file_path": "test.py", "line": 42},
        )
        assert selection.action == "ts_enclosing_symbol"


class TestTreeSitterActions:
    """Tests for Tree-sitter actions."""

    @pytest.mark.anyio
    async def test_ts_outline_action(self, tmp_path: Path) -> None:
        """Test TsOutlineAction."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def foo():
    pass

class Bar:
    def method(self):
        pass
""")

        action = TsOutlineAction()
        run = ResearchRun.create(question="What's in this file?")

        result = await action.execute(run, {"file_path": str(test_file)})

        assert result.success
        assert "symbols" in result.data
        assert len(result.data["symbols"]) >= 2

    @pytest.mark.anyio
    async def test_ts_outline_action_empty_file(self, tmp_path: Path) -> None:
        """Test TsOutlineAction on empty file."""
        test_file = tmp_path / "empty.py"
        test_file.write_text("")

        action = TsOutlineAction()
        run = ResearchRun.create(question="What's in this file?")

        result = await action.execute(run, {"file_path": str(test_file)})

        assert result.success
        assert result.data["symbols"] == []

    @pytest.mark.anyio
    async def test_ts_outline_action_no_file_path(self) -> None:
        """Test TsOutlineAction without file_path."""
        action = TsOutlineAction()
        run = ResearchRun.create(question="Test")

        result = await action.execute(run, {})

        assert not result.success
        assert "file_path" in result.error.lower()

    @pytest.mark.anyio
    async def test_ts_find_symbol_action(self, tmp_path: Path) -> None:
        """Test TsFindSymbolAction."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def target_function():
    return 42

def other_function():
    pass
""")

        action = TsFindSymbolAction()
        run = ResearchRun.create(question="Find target_function")

        result = await action.execute(
            run,
            {"file_path": str(test_file), "name": "target_function"},
        )

        assert result.success
        assert result.data["found"] is True
        assert result.data["symbol"]["name"] == "target_function"
        assert len(result.evidence) == 1
        assert result.evidence[0].provenance == "treesitter"
        assert result.evidence[0].symbol_id == "target_function"

    @pytest.mark.anyio
    async def test_ts_find_symbol_action_not_found(self, tmp_path: Path) -> None:
        """Test TsFindSymbolAction when symbol not found."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo(): pass\n")

        action = TsFindSymbolAction()
        run = ResearchRun.create(question="Find nonexistent")

        result = await action.execute(
            run,
            {"file_path": str(test_file), "name": "nonexistent"},
        )

        assert result.success
        assert result.data["found"] is False
        assert len(result.evidence) == 0

    @pytest.mark.anyio
    async def test_ts_enclosing_symbol_action(self, tmp_path: Path) -> None:
        """Test TsEnclosingSymbolAction."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""def my_function():
    x = 1
    y = 2
    return x + y
""")

        action = TsEnclosingSymbolAction()
        run = ResearchRun.create(question="What contains line 3?")

        result = await action.execute(
            run,
            {"file_path": str(test_file), "line": 3},
        )

        assert result.success
        assert result.data["found"] is True
        assert result.data["symbol"]["name"] == "my_function"
        assert len(result.evidence) == 1
        assert result.evidence[0].provenance == "treesitter"

    @pytest.mark.anyio
    async def test_ts_enclosing_symbol_action_not_in_symbol(self, tmp_path: Path) -> None:
        """Test TsEnclosingSymbolAction when line is not in any symbol."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""# Comment
x = 1

def foo():
    pass
""")

        action = TsEnclosingSymbolAction()
        run = ResearchRun.create(question="What contains line 2?")

        result = await action.execute(
            run,
            {"file_path": str(test_file), "line": 2},
        )

        assert result.success
        assert result.data["found"] is False
        assert len(result.evidence) == 0


class TestMockTreeSitterActions:
    """Tests for mock Tree-sitter actions."""

    @pytest.mark.anyio
    async def test_mock_ts_outline_action(self) -> None:
        """Test MockTsOutlineAction."""
        action = MockTsOutlineAction(
            mock_symbols=[
                {
                    "name": "mock_func",
                    "kind": "function",
                    "file_path": "test.py",
                    "start_line": 1,
                    "end_line": 5,
                }
            ]
        )

        run = ResearchRun.create(question="Test")
        result = await action.execute(run, {"file_path": "test.py"})

        assert result.success
        assert len(result.data["symbols"]) == 1
        assert result.data["symbols"][0]["name"] == "mock_func"

    @pytest.mark.anyio
    async def test_mock_ts_outline_action_no_file_path(self) -> None:
        """Test MockTsOutlineAction without file_path."""
        action = MockTsOutlineAction()
        run = ResearchRun.create(question="Test")

        result = await action.execute(run, {})

        assert not result.success
        assert "file_path" in result.error.lower()

    @pytest.mark.anyio
    async def test_mock_ts_find_symbol_action(self) -> None:
        """Test MockTsFindSymbolAction."""
        action = MockTsFindSymbolAction(
            mock_symbol={
                "name": "target",
                "kind": "function",
                "start_line": 10,
                "end_line": 20,
                "content": "def target(): pass",
            }
        )

        run = ResearchRun.create(question="Find target")
        result = await action.execute(
            run,
            {"file_path": "test.py", "name": "target"},
        )

        assert result.success
        assert result.data["found"] is True
        assert len(result.evidence) == 1
        assert result.evidence[0].provenance == "treesitter"

    @pytest.mark.anyio
    async def test_mock_ts_find_symbol_action_not_found(self) -> None:
        """Test MockTsFindSymbolAction with no configured result."""
        action = MockTsFindSymbolAction()
        run = ResearchRun.create(question="Find something")

        result = await action.execute(
            run,
            {"file_path": "test.py", "name": "something"},
        )

        assert result.success
        assert result.data["found"] is False
        assert len(result.evidence) == 0
