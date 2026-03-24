"""Tests for SCIP parsing helper functions and SCIPProvider methods.

Tests range parsing, symbol name extraction, kind inference, descriptor
parsing, relation kind determination, range containment/span calculations,
and enclosing symbol detection.
"""

from __future__ import annotations

import pytest
from contextmine_core.semantic_snapshot.models import (
    Range,
    RelationKind,
    SymbolKind,
)
from contextmine_core.semantic_snapshot.scip import (
    _EXTERNAL_FILE_PATH,
    _LOCAL_SYMBOL_PREFIX,
    SCIP_CALL_LIKE_SYNTAX_KINDS,
    SCIP_FALLBACK_SYMBOL_KINDS,
    SCIP_KIND_TO_SYMBOL_KIND,
    SCIP_ROLE_DEFINITION,
    SCIP_ROLE_IMPORT,
    SCIP_ROLE_READ_ACCESS,
    SCIP_ROLE_WRITE_ACCESS,
    SCIPProvider,
)


@pytest.fixture
def provider(tmp_path):
    """Create a SCIPProvider instance pointing to a temp path."""
    return SCIPProvider(tmp_path / "test.scip")


# ---------------------------------------------------------------------------
# _parse_range
# ---------------------------------------------------------------------------


class TestParseRange:
    def test_four_elements(self, provider) -> None:
        result = provider._parse_range([0, 5, 10, 15])
        assert result is not None
        assert result.start_line == 1
        assert result.start_col == 5
        assert result.end_line == 11
        assert result.end_col == 15

    def test_three_elements_same_line(self, provider) -> None:
        result = provider._parse_range([5, 10, 20])
        assert result is not None
        assert result.start_line == 6
        assert result.start_col == 10
        assert result.end_line == 6
        assert result.end_col == 20

    def test_empty_list(self, provider) -> None:
        assert provider._parse_range([]) is None

    def test_two_elements(self, provider) -> None:
        assert provider._parse_range([1, 2]) is None

    def test_five_elements(self, provider) -> None:
        assert provider._parse_range([1, 2, 3, 4, 5]) is None


# ---------------------------------------------------------------------------
# _get_language_string
# ---------------------------------------------------------------------------


class TestGetLanguageString:
    def test_empty_string(self, provider) -> None:
        assert provider._get_language_string("") is None

    def test_python(self, provider) -> None:
        assert provider._get_language_string("Python") == "python"

    def test_typescript(self, provider) -> None:
        assert provider._get_language_string("TypeScript") == "typescript"


# ---------------------------------------------------------------------------
# _extract_name_from_symbol
# ---------------------------------------------------------------------------


class TestExtractNameFromSymbol:
    def test_empty_string(self, provider) -> None:
        assert provider._extract_name_from_symbol("") is None

    def test_none(self, provider) -> None:
        assert provider._extract_name_from_symbol(None) is None

    def test_local_symbol(self, provider) -> None:
        result = provider._extract_name_from_symbol("local 42")
        assert result == "local_42"

    def test_python_method(self, provider) -> None:
        result = provider._extract_name_from_symbol(
            "scip-python python mypackage 0.1.0 mymodule/MyClass#method()."
        )
        assert result == "method"

    def test_python_class(self, provider) -> None:
        result = provider._extract_name_from_symbol(
            "scip-python python mypackage 0.1.0 mymodule/MyClass#"
        )
        assert result == "MyClass"

    def test_short_symbol(self, provider) -> None:
        result = provider._extract_name_from_symbol("scip-python python")
        assert result is None

    def test_module_descriptor(self, provider) -> None:
        result = provider._extract_name_from_symbol("scip-python python pkg 1.0 mymodule/")
        assert result == "mymodule"


# ---------------------------------------------------------------------------
# _infer_kind_and_name_from_symbol
# ---------------------------------------------------------------------------


class TestInferKindAndName:
    def test_empty(self, provider) -> None:
        kind, name = provider._infer_kind_and_name_from_symbol("")
        assert kind == SymbolKind.UNKNOWN
        assert name is None

    def test_local_symbol(self, provider) -> None:
        kind, name = provider._infer_kind_and_name_from_symbol("local 42")
        assert kind == SymbolKind.UNKNOWN

    def test_method_descriptor(self, provider) -> None:
        kind, name = provider._infer_kind_and_name_from_symbol(
            "scip-python python pkg 1.0 MyClass#method()."
        )
        assert kind == SymbolKind.METHOD
        assert name == "method"

    def test_function_descriptor(self, provider) -> None:
        kind, name = provider._infer_kind_and_name_from_symbol(
            "scip-python python pkg 1.0 my_function()."
        )
        assert kind == SymbolKind.FUNCTION
        assert name == "my_function"

    def test_class_descriptor(self, provider) -> None:
        kind, name = provider._infer_kind_and_name_from_symbol(
            "scip-python python pkg 1.0 MyClass#"
        )
        assert kind == SymbolKind.CLASS
        assert name == "MyClass"

    def test_property_descriptor(self, provider) -> None:
        kind, name = provider._infer_kind_and_name_from_symbol(
            "scip-python python pkg 1.0 MyClass#field."
        )
        assert kind == SymbolKind.PROPERTY
        assert name == "field"

    def test_module_descriptor_slash(self, provider) -> None:
        kind, name = provider._infer_kind_and_name_from_symbol(
            "scip-python python pkg 1.0 mymodule/"
        )
        assert kind == SymbolKind.MODULE

    def test_namespace_descriptor_colon(self, provider) -> None:
        kind, name = provider._infer_kind_and_name_from_symbol("scip-python python pkg 1.0 myns:")
        assert kind == SymbolKind.MODULE

    def test_type_parameter(self, provider) -> None:
        kind, name = provider._infer_kind_and_name_from_symbol(
            "scip-python python pkg 1.0 MyClass#[T]"
        )
        assert kind == SymbolKind.TYPE_ALIAS
        assert name == "T"

    def test_macro_descriptor(self, provider) -> None:
        kind, name = provider._infer_kind_and_name_from_symbol("scip-rust rust pkg 1.0 my_macro!")
        assert kind == SymbolKind.FUNCTION
        assert name == "my_macro"

    def test_parameter_descriptor(self, provider) -> None:
        kind, name = provider._infer_kind_and_name_from_symbol(
            "scip-python python pkg 1.0 func().(param)"
        )
        assert kind == SymbolKind.PARAMETER
        assert name == "param"

    def test_plain_property_dot(self, provider) -> None:
        kind, name = provider._infer_kind_and_name_from_symbol("scip-python python pkg 1.0 my_var.")
        assert kind == SymbolKind.FUNCTION
        assert name == "my_var"


# ---------------------------------------------------------------------------
# _descriptor_tail
# ---------------------------------------------------------------------------


class TestDescriptorTail:
    def test_full_symbol(self, provider) -> None:
        result = provider._descriptor_tail("scip-python python pkg 1.0 mymodule/MyClass#")
        assert result == "mymodule/MyClass#"

    def test_short_symbol(self, provider) -> None:
        result = provider._descriptor_tail("scip-python python pkg 1.0")
        assert result == ""

    def test_very_short(self, provider) -> None:
        result = provider._descriptor_tail("short")
        assert result == ""


# ---------------------------------------------------------------------------
# _last_identifier
# ---------------------------------------------------------------------------


class TestLastIdentifier:
    def test_simple(self, provider) -> None:
        assert provider._last_identifier("MyClass") == "MyClass"

    def test_with_slashes(self, provider) -> None:
        assert provider._last_identifier("module/MyClass") == "MyClass"

    def test_with_backticks(self, provider) -> None:
        assert provider._last_identifier("`my_module`") == "my_module"

    def test_empty(self, provider) -> None:
        assert provider._last_identifier("") is None

    def test_only_special_chars(self, provider) -> None:
        assert provider._last_identifier("###") is None


# ---------------------------------------------------------------------------
# _fallback_symbol_kind
# ---------------------------------------------------------------------------


class TestFallbackSymbolKind:
    def test_function_descriptor(self, provider) -> None:
        result = provider._fallback_symbol_kind("scip-python python pkg 1.0 my_func().")
        assert result == SymbolKind.FUNCTION

    def test_class_descriptor(self, provider) -> None:
        result = provider._fallback_symbol_kind("scip-python python pkg 1.0 MyClass#")
        assert result == SymbolKind.CLASS

    def test_property_descriptor(self, provider) -> None:
        result = provider._fallback_symbol_kind("scip-python python pkg 1.0 MyClass#field.")
        assert result == SymbolKind.PROPERTY

    def test_module_descriptor(self, provider) -> None:
        result = provider._fallback_symbol_kind("scip-python python pkg 1.0 mymodule/")
        assert result == SymbolKind.MODULE

    def test_unknown_defaults_to_module(self, provider) -> None:
        result = provider._fallback_symbol_kind("short")
        assert result == SymbolKind.MODULE


# ---------------------------------------------------------------------------
# _relation_kind_from_occurrence
# ---------------------------------------------------------------------------


class TestRelationKindFromOccurrence:
    def test_import_role(self, provider) -> None:
        result = provider._relation_kind_from_occurrence(
            symbol_roles=SCIP_ROLE_IMPORT,
            syntax_kind=0,
            caller_kind=SymbolKind.MODULE,
            target_kind=SymbolKind.FUNCTION,
        )
        assert result == RelationKind.IMPORTS

    def test_call_like_syntax(self, provider) -> None:
        result = provider._relation_kind_from_occurrence(
            symbol_roles=0,
            syntax_kind=15,  # IdentifierFunction
            caller_kind=SymbolKind.FUNCTION,
            target_kind=SymbolKind.FUNCTION,
        )
        assert result == RelationKind.CALLS

    def test_callable_target_and_caller(self, provider) -> None:
        result = provider._relation_kind_from_occurrence(
            symbol_roles=0,
            syntax_kind=99,  # Not call-like
            caller_kind=SymbolKind.FUNCTION,
            target_kind=SymbolKind.FUNCTION,
        )
        assert result == RelationKind.CALLS

    def test_callable_target_eligible_caller(self, provider) -> None:
        result = provider._relation_kind_from_occurrence(
            symbol_roles=0,
            syntax_kind=99,
            caller_kind=SymbolKind.CLASS,
            target_kind=SymbolKind.METHOD,
        )
        assert result == RelationKind.CALLS

    def test_write_access_prevents_call_syntax_path(self, provider) -> None:
        """Write access with eligible (non-preferred) caller => REFERENCES, not CALLS."""
        result = provider._relation_kind_from_occurrence(
            symbol_roles=SCIP_ROLE_WRITE_ACCESS,
            syntax_kind=15,
            caller_kind=SymbolKind.CLASS,  # eligible but not preferred
            target_kind=SymbolKind.VARIABLE,  # not callable
        )
        assert result == RelationKind.REFERENCES

    def test_fallback_to_references(self, provider) -> None:
        result = provider._relation_kind_from_occurrence(
            symbol_roles=0,
            syntax_kind=99,
            caller_kind=SymbolKind.CONSTANT,
            target_kind=SymbolKind.VARIABLE,
        )
        assert result == RelationKind.REFERENCES


# ---------------------------------------------------------------------------
# _range_contains and _range_span
# ---------------------------------------------------------------------------


class TestRangeContains:
    def test_exact_match(self, provider) -> None:
        r = Range(start_line=1, start_col=0, end_line=10, end_col=5)
        assert provider._range_contains(r, r) is True

    def test_inner_contained(self, provider) -> None:
        outer = Range(start_line=1, start_col=0, end_line=10, end_col=5)
        inner = Range(start_line=3, start_col=0, end_line=5, end_col=5)
        assert provider._range_contains(outer, inner) is True

    def test_inner_exceeds(self, provider) -> None:
        outer = Range(start_line=1, start_col=0, end_line=5, end_col=5)
        inner = Range(start_line=3, start_col=0, end_line=7, end_col=5)
        assert provider._range_contains(outer, inner) is False


class TestRangeSpan:
    def test_single_line(self, provider) -> None:
        r = Range(start_line=1, start_col=0, end_line=1, end_col=10)
        assert provider._range_span(r) == 10

    def test_multi_line(self, provider) -> None:
        r = Range(start_line=1, start_col=0, end_line=11, end_col=5)
        assert provider._range_span(r) == 10_000_005


# ---------------------------------------------------------------------------
# _find_enclosing_symbol_def_id
# ---------------------------------------------------------------------------


class TestFindEnclosingSymbolDefId:
    def test_empty_candidates(self, provider) -> None:
        result = provider._find_enclosing_symbol_def_id(
            file_path="test.py",
            occ_range=Range(start_line=5, start_col=0, end_line=5, end_col=10),
            enclosing_range_raw=[],
            symbols_by_file={},
            symbol_kinds={},
        )
        assert result is None

    def test_exact_enclosing_range_match(self, provider) -> None:
        r = Range(start_line=1, start_col=0, end_line=10, end_col=0)
        result = provider._find_enclosing_symbol_def_id(
            file_path="test.py",
            occ_range=Range(start_line=5, start_col=0, end_line=5, end_col=10),
            enclosing_range_raw=[0, 0, 9, 0],  # Will be parsed to 1-based
            symbols_by_file={"test.py": [("sym1", r)]},
            symbol_kinds={"sym1": SymbolKind.FUNCTION},
        )
        assert result == "sym1"

    def test_containing_range_found(self, provider) -> None:
        outer = Range(start_line=1, start_col=0, end_line=20, end_col=0)
        result = provider._find_enclosing_symbol_def_id(
            file_path="test.py",
            occ_range=Range(start_line=5, start_col=0, end_line=5, end_col=10),
            enclosing_range_raw=[],
            symbols_by_file={"test.py": [("sym1", outer)]},
            symbol_kinds={"sym1": SymbolKind.FUNCTION},
        )
        assert result == "sym1"

    def test_module_fallback(self, provider) -> None:
        module_range = Range(start_line=1, start_col=0, end_line=100, end_col=0)
        result = provider._find_enclosing_symbol_def_id(
            file_path="test.py",
            occ_range=Range(start_line=50, start_col=0, end_line=50, end_col=10),
            enclosing_range_raw=[],
            symbols_by_file={"test.py": [("mod1", module_range)]},
            symbol_kinds={"mod1": SymbolKind.MODULE},
        )
        assert result == "mod1"


# ---------------------------------------------------------------------------
# _find_contextual_symbol_def_id
# ---------------------------------------------------------------------------


class TestFindContextualSymbolDefId:
    def test_empty(self, provider) -> None:
        result = provider._find_contextual_symbol_def_id(
            file_path="test.py",
            occ_range=Range(start_line=5, start_col=0, end_line=5, end_col=10),
            definition_occurrences_by_file={},
            symbols_by_file={},
            symbol_kinds={},
        )
        assert result is None

    def test_preceding_callable(self, provider) -> None:
        r1 = Range(start_line=1, start_col=0, end_line=3, end_col=0)
        result = provider._find_contextual_symbol_def_id(
            file_path="test.py",
            occ_range=Range(start_line=5, start_col=0, end_line=5, end_col=10),
            definition_occurrences_by_file={"test.py": [("func1", r1)]},
            symbols_by_file={},
            symbol_kinds={"func1": SymbolKind.FUNCTION},
        )
        assert result == "func1"

    def test_module_fallback(self, provider) -> None:
        r1 = Range(start_line=10, start_col=0, end_line=10, end_col=0)
        result = provider._find_contextual_symbol_def_id(
            file_path="test.py",
            occ_range=Range(start_line=5, start_col=0, end_line=5, end_col=10),
            definition_occurrences_by_file={},
            symbols_by_file={"test.py": [("mod1", r1)]},
            symbol_kinds={"mod1": SymbolKind.MODULE},
        )
        assert result == "mod1"


# ---------------------------------------------------------------------------
# Constants verification
# ---------------------------------------------------------------------------


class TestConstants:
    def test_local_symbol_prefix(self) -> None:
        assert _LOCAL_SYMBOL_PREFIX == "local "

    def test_external_file_path(self) -> None:
        assert _EXTERNAL_FILE_PATH == "<external>"

    def test_kind_mapping_has_entries(self) -> None:
        assert len(SCIP_KIND_TO_SYMBOL_KIND) > 0
        assert 7 in SCIP_KIND_TO_SYMBOL_KIND  # Class

    def test_fallback_kinds(self) -> None:
        assert SymbolKind.FUNCTION in SCIP_FALLBACK_SYMBOL_KINDS
        assert SymbolKind.CLASS in SCIP_FALLBACK_SYMBOL_KINDS
        assert SymbolKind.MODULE in SCIP_FALLBACK_SYMBOL_KINDS

    def test_role_flags(self) -> None:
        assert SCIP_ROLE_DEFINITION == 0x1
        assert SCIP_ROLE_IMPORT == 0x2
        assert SCIP_ROLE_WRITE_ACCESS == 0x4
        assert SCIP_ROLE_READ_ACCESS == 0x8

    def test_call_like_syntax_kinds(self) -> None:
        assert 15 in SCIP_CALL_LIKE_SYNTAX_KINDS
        assert 16 in SCIP_CALL_LIKE_SYNTAX_KINDS

    def test_is_available_no_file(self, provider) -> None:
        assert provider.is_available() is False
