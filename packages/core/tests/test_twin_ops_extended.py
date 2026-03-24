"""Extended tests for contextmine_core.twin.ops — async analysis helpers via mocking."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

from contextmine_core.twin.ops import (
    LSP_METHOD_KINDS,
    SANITIZER_PATTERNS,
    SINK_PATTERNS,
    SOURCE_PATTERNS,
    _escape_like,
    _flatten_lsp_symbols,
    _language_patterns,
    _read_node_file_path,
    _read_node_source_id,
    _scala_escape,
    _symbol_kind_to_name,
)

# ── _scala_escape ────────────────────────────────────────────────────────


class TestScalaEscape:
    def test_no_special_chars(self) -> None:
        assert _scala_escape("hello") == "hello"

    def test_escapes_backslash(self) -> None:
        assert _scala_escape("a\\b") == "a\\\\b"

    def test_escapes_double_quote(self) -> None:
        assert _scala_escape('a"b') == 'a\\"b'

    def test_both_special_chars(self) -> None:
        assert _scala_escape('a\\b"c') == 'a\\\\b\\"c'

    def test_empty_string(self) -> None:
        assert _scala_escape("") == ""


# ── _escape_like ─────────────────────────────────────────────────────────


class TestEscapeLike:
    def test_no_wildcards(self) -> None:
        assert _escape_like("hello") == "hello"

    def test_escapes_percent(self) -> None:
        assert _escape_like("50%off") == "50\\%off"

    def test_escapes_underscore(self) -> None:
        assert _escape_like("a_b") == "a\\_b"

    def test_escapes_backslash(self) -> None:
        assert _escape_like("a\\b") == "a\\\\b"

    def test_multiple_wildcards(self) -> None:
        assert _escape_like("%_\\") == "\\%\\_\\\\"

    def test_empty_string(self) -> None:
        assert _escape_like("") == ""


# ── _symbol_kind_to_name ────────────────────────────────────────────────


class TestSymbolKindToName:
    def test_file_kind(self) -> None:
        assert _symbol_kind_to_name(1) == "file"

    def test_class_kind(self) -> None:
        assert _symbol_kind_to_name(5) == "class"

    def test_method_kind(self) -> None:
        assert _symbol_kind_to_name(6) == "method"

    def test_function_kind(self) -> None:
        assert _symbol_kind_to_name(12) == "function"

    def test_variable_kind(self) -> None:
        assert _symbol_kind_to_name(13) == "variable"

    def test_unknown_numeric_kind(self) -> None:
        assert _symbol_kind_to_name(99) == "kind_99"

    def test_string_number(self) -> None:
        assert _symbol_kind_to_name("6") == "method"

    def test_non_numeric_string(self) -> None:
        assert _symbol_kind_to_name("foo") == "foo"

    def test_none_returns_unknown(self) -> None:
        assert _symbol_kind_to_name(None) == "unknown"

    def test_module_kind(self) -> None:
        assert _symbol_kind_to_name(2) == "module"

    def test_constructor_kind(self) -> None:
        assert _symbol_kind_to_name(9) == "constructor"

    def test_enum_kind(self) -> None:
        assert _symbol_kind_to_name(10) == "enum"

    def test_interface_kind(self) -> None:
        assert _symbol_kind_to_name(11) == "interface"


# ── _flatten_lsp_symbols ────────────────────────────────────────────────


class TestFlattenLspSymbols:
    def test_empty_list(self) -> None:
        result = _flatten_lsp_symbols(file_path="/a/b.py", symbols=[])
        assert result == []

    def test_single_symbol(self) -> None:
        symbols = [
            {
                "name": "my_func",
                "kind": 12,
                "selectionRange": {"start": {"line": 5, "character": 4}},
            }
        ]
        result = _flatten_lsp_symbols(file_path="/test.py", symbols=symbols)
        assert len(result) == 1
        assert result[0]["name"] == "my_func"
        assert result[0]["kind"] == "function"
        assert result[0]["kind_id"] == 12
        assert result[0]["file_path"] == "/test.py"
        assert result[0]["line_number"] == 6  # 0-based + 1
        assert result[0]["column"] == 4

    def test_nested_children(self) -> None:
        symbols = [
            {
                "name": "MyClass",
                "kind": 5,
                "selectionRange": {"start": {"line": 0, "character": 0}},
                "children": [
                    {
                        "name": "my_method",
                        "kind": 6,
                        "selectionRange": {"start": {"line": 2, "character": 4}},
                    }
                ],
            }
        ]
        result = _flatten_lsp_symbols(file_path="/test.py", symbols=symbols)
        assert len(result) == 2
        names = {r["name"] for r in result}
        assert names == {"MyClass", "my_method"}

    def test_non_dict_items_skipped(self) -> None:
        symbols = [None, "string", 42]
        result = _flatten_lsp_symbols(file_path="/test.py", symbols=symbols)
        assert result == []

    def test_uses_range_fallback(self) -> None:
        symbols = [
            {
                "name": "foo",
                "kind": 12,
                "range": {"start": {"line": 10, "character": 2}},
            }
        ]
        result = _flatten_lsp_symbols(file_path="/test.py", symbols=symbols)
        assert result[0]["line_number"] == 11

    def test_missing_range(self) -> None:
        symbols = [{"name": "bar", "kind": 6}]
        result = _flatten_lsp_symbols(file_path="/test.py", symbols=symbols)
        assert result[0]["line_number"] == 1  # 0 + 1
        assert result[0]["column"] == 0


# ── _read_node_source_id ────────────────────────────────────────────────


class TestReadNodeSourceId:
    def test_direct_source_id(self) -> None:
        node = MagicMock()
        node.source_id = uuid4()
        node.meta = {}
        assert _read_node_source_id(node) == node.source_id

    def test_from_meta(self) -> None:
        uid = uuid4()
        node = MagicMock()
        node.source_id = None
        node.meta = {"source_id": str(uid)}
        assert _read_node_source_id(node) == uid

    def test_no_source_id(self) -> None:
        node = MagicMock()
        node.source_id = None
        node.meta = {}
        assert _read_node_source_id(node) is None

    def test_invalid_uuid_in_meta(self) -> None:
        node = MagicMock()
        node.source_id = None
        node.meta = {"source_id": "not-a-uuid"}
        assert _read_node_source_id(node) is None

    def test_none_meta(self) -> None:
        node = MagicMock()
        node.source_id = None
        node.meta = None
        assert _read_node_source_id(node) is None


# ── _read_node_file_path ────────────────────────────────────────────────


class TestReadNodeFilePath:
    def test_from_meta(self) -> None:
        node = MagicMock()
        node.meta = {"file_path": "/src/main.py"}
        node.natural_key = "file:/src/main.py"
        assert _read_node_file_path(node) == "/src/main.py"

    def test_from_natural_key(self) -> None:
        node = MagicMock()
        node.meta = {}
        node.natural_key = "file:src/app.ts"
        assert _read_node_file_path(node) == "src/app.ts"

    def test_empty_meta_no_file_prefix(self) -> None:
        node = MagicMock()
        node.meta = {}
        node.natural_key = "symbol:foo"
        assert _read_node_file_path(node) is None

    def test_none_meta(self) -> None:
        node = MagicMock()
        node.meta = None
        node.natural_key = "file:/test.py"
        assert _read_node_file_path(node) == "/test.py"

    def test_whitespace_meta_value(self) -> None:
        node = MagicMock()
        node.meta = {"file_path": "  "}
        node.natural_key = "file:fallback.py"
        assert _read_node_file_path(node) == "fallback.py"


# ── _language_patterns ──────────────────────────────────────────────────


class TestLanguagePatterns:
    def test_python(self) -> None:
        result = _language_patterns("python", SOURCE_PATTERNS)
        assert result == SOURCE_PATTERNS["python"]

    def test_none_defaults_to_python(self) -> None:
        result = _language_patterns(None, SOURCE_PATTERNS)
        assert result == SOURCE_PATTERNS["python"]

    def test_case_insensitive(self) -> None:
        result = _language_patterns("PYTHON", SOURCE_PATTERNS)
        assert result == SOURCE_PATTERNS["python"]

    def test_unknown_language_defaults_to_python(self) -> None:
        result = _language_patterns("cobol", SOURCE_PATTERNS)
        assert result == SOURCE_PATTERNS["python"]

    def test_javascript_sinks(self) -> None:
        result = _language_patterns("javascript", SINK_PATTERNS)
        assert result == SINK_PATTERNS["javascript"]

    def test_go_sanitizers(self) -> None:
        result = _language_patterns("go", SANITIZER_PATTERNS)
        assert result == SANITIZER_PATTERNS["go"]


# ── LSP_METHOD_KINDS constant ───────────────────────────────────────────


class TestLspMethodKinds:
    def test_contains_class(self) -> None:
        assert 5 in LSP_METHOD_KINDS

    def test_contains_method(self) -> None:
        assert 6 in LSP_METHOD_KINDS

    def test_contains_function(self) -> None:
        assert 12 in LSP_METHOD_KINDS

    def test_size(self) -> None:
        assert len(LSP_METHOD_KINDS) == 3


# ── Pattern dictionaries ────────────────────────────────────────────────


class TestPatternDictionaries:
    def test_source_patterns_has_required_langs(self) -> None:
        for lang in ["python", "javascript", "typescript", "java", "go", "php"]:
            assert lang in SOURCE_PATTERNS
            assert len(SOURCE_PATTERNS[lang]) > 0

    def test_sink_patterns_has_required_langs(self) -> None:
        for lang in ["python", "javascript", "typescript", "java", "go", "php"]:
            assert lang in SINK_PATTERNS
            assert len(SINK_PATTERNS[lang]) > 0

    def test_sanitizer_patterns_has_required_langs(self) -> None:
        for lang in ["python", "javascript", "typescript", "java", "go", "php"]:
            assert lang in SANITIZER_PATTERNS
            assert len(SANITIZER_PATTERNS[lang]) > 0

    def test_python_source_has_request(self) -> None:
        assert "request" in SOURCE_PATTERNS["python"]

    def test_python_sink_has_subprocess(self) -> None:
        assert "subprocess" in SINK_PATTERNS["python"]

    def test_python_sanitizer_has_escape(self) -> None:
        assert "escape" in SANITIZER_PATTERNS["python"]
