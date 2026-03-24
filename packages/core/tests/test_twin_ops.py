"""Tests for contextmine_core.twin.ops — pure/utility functions."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from contextmine_core.twin.ops import (
    _as_int,
    _flatten_lsp_symbols,
    _hash_params,
    _line_split,
    _normalize_pattern_token,
    _sarif_level,
    _scala_escape,
    _sha,
    _symbol_kind_to_name,
    _tsv_rows,
    coerce_source_ids,
    compute_analysis_context_key,
    findings_to_sarif,
    normalize_analysis_engines,
    parse_timestamp_value,
    sanitize_regex_query,
)

# ── _sha ─────────────────────────────────────────────────────────────────


class TestSha:
    def test_basic_hash(self) -> None:
        result = _sha("hello")
        expected = hashlib.sha256(b"hello").hexdigest()
        assert result == expected

    def test_empty_string(self) -> None:
        result = _sha("")
        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected

    def test_deterministic(self) -> None:
        assert _sha("test") == _sha("test")

    def test_different_inputs_differ(self) -> None:
        assert _sha("a") != _sha("b")

    def test_unicode(self) -> None:
        result = _sha("Ünïcödé")
        expected = hashlib.sha256("Ünïcödé".encode()).hexdigest()
        assert result == expected


# ── _normalize_pattern_token ─────────────────────────────────────────────


class TestNormalizePatternToken:
    def test_basic_token(self) -> None:
        assert _normalize_pattern_token("eval") == "eval"

    def test_strips_whitespace(self) -> None:
        assert _normalize_pattern_token("  eval  ") == "eval"

    def test_lowercases(self) -> None:
        assert _normalize_pattern_token("EVAL") == "eval"

    def test_dotted_returns_last_segment(self) -> None:
        assert _normalize_pattern_token("os.getenv") == "getenv"

    def test_dotted_with_whitespace(self) -> None:
        assert _normalize_pattern_token("  process.ENV  ") == "env"

    def test_multiple_dots(self) -> None:
        assert _normalize_pattern_token("a.b.c") == "c"

    def test_single_dot(self) -> None:
        # ".foo" -> rsplit gives ["", "foo"]
        assert _normalize_pattern_token(".foo") == "foo"

    def test_trailing_dot(self) -> None:
        # "foo." -> rsplit gives ["foo", ""]
        assert _normalize_pattern_token("foo.") == ""

    def test_empty_string(self) -> None:
        assert _normalize_pattern_token("") == ""


# ── compute_analysis_context_key ─────────────────────────────────────────


class TestComputeAnalysisContextKey:
    def test_deterministic(self) -> None:
        sid = uuid4()
        result1 = compute_analysis_context_key(
            source_id=sid,
            revision_key="rev1",
            extractor_version="v1",
            projection_profile="profile",
        )
        result2 = compute_analysis_context_key(
            source_id=sid,
            revision_key="rev1",
            extractor_version="v1",
            projection_profile="profile",
        )
        assert result1 == result2

    def test_different_source_id_produces_different_key(self) -> None:
        k1 = compute_analysis_context_key(
            source_id=uuid4(),
            revision_key="rev",
            extractor_version="v1",
            projection_profile="p",
        )
        k2 = compute_analysis_context_key(
            source_id=uuid4(),
            revision_key="rev",
            extractor_version="v1",
            projection_profile="p",
        )
        assert k1 != k2

    def test_different_revision_key_produces_different_key(self) -> None:
        sid = uuid4()
        k1 = compute_analysis_context_key(
            source_id=sid,
            revision_key="rev1",
            extractor_version="v1",
            projection_profile="p",
        )
        k2 = compute_analysis_context_key(
            source_id=sid,
            revision_key="rev2",
            extractor_version="v1",
            projection_profile="p",
        )
        assert k1 != k2

    def test_returns_hex_string(self) -> None:
        result = compute_analysis_context_key(
            source_id=uuid4(),
            revision_key="r",
            extractor_version="v",
            projection_profile="p",
        )
        assert len(result) == 64  # SHA-256 hex digest length
        assert all(c in "0123456789abcdef" for c in result)


# ── _hash_params ─────────────────────────────────────────────────────────


class TestHashParams:
    def test_deterministic(self) -> None:
        params = {"a": 1, "b": "two"}
        assert _hash_params(params) == _hash_params(params)

    def test_key_order_does_not_matter(self) -> None:
        # json.dumps with sort_keys=True makes order irrelevant
        assert _hash_params({"b": 2, "a": 1}) == _hash_params({"a": 1, "b": 2})

    def test_empty_dict(self) -> None:
        result = _hash_params({})
        expected = _sha(json.dumps({}, sort_keys=True, separators=(",", ":")))
        assert result == expected

    def test_nested_dict(self) -> None:
        params = {"outer": {"inner": [1, 2, 3]}}
        result = _hash_params(params)
        assert len(result) == 64

    def test_different_values_differ(self) -> None:
        assert _hash_params({"x": 1}) != _hash_params({"x": 2})


# ── parse_timestamp_value ────────────────────────────────────────────────


class TestParseTimestampValue:
    def test_none_returns_none(self) -> None:
        assert parse_timestamp_value(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert parse_timestamp_value("") is None

    def test_iso_format_with_z(self) -> None:
        result = parse_timestamp_value("2024-01-15T10:30:00Z")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30
        assert result.tzinfo is not None

    def test_iso_format_with_offset(self) -> None:
        result = parse_timestamp_value("2024-01-15T10:30:00+02:00")
        assert result is not None
        # Converted to UTC, so 10:30+02:00 = 08:30 UTC
        assert result.hour == 8
        assert result.minute == 30

    def test_naive_gets_utc(self) -> None:
        result = parse_timestamp_value("2024-06-01T12:00:00")
        assert result is not None
        assert result.tzinfo == UTC

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_timestamp_value("not-a-timestamp")


# ── coerce_source_ids ────────────────────────────────────────────────────


class TestCoerceSourceIds:
    def test_none_returns_empty_list(self) -> None:
        assert coerce_source_ids(None) == []

    def test_empty_list_returns_empty_list(self) -> None:
        assert coerce_source_ids([]) == []

    def test_valid_uuids(self) -> None:
        u1 = str(uuid4())
        u2 = str(uuid4())
        result = coerce_source_ids([u1, u2])
        assert len(result) == 2
        assert result[0] == UUID(u1)
        assert result[1] == UUID(u2)

    def test_invalid_uuid_raises(self) -> None:
        with pytest.raises(ValueError):
            coerce_source_ids(["not-a-uuid"])

    def test_single_valid_uuid(self) -> None:
        u = str(uuid4())
        result = coerce_source_ids([u])
        assert result == [UUID(u)]


# ── sanitize_regex_query ─────────────────────────────────────────────────


class TestSanitizeRegexQuery:
    def test_none_returns_none(self) -> None:
        assert sanitize_regex_query(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert sanitize_regex_query("") is None

    def test_whitespace_only_returns_none(self) -> None:
        assert sanitize_regex_query("   ") is None

    def test_simple_query(self) -> None:
        assert sanitize_regex_query("hello") == "hello"

    def test_strips_whitespace(self) -> None:
        assert sanitize_regex_query("  hello  ") == "hello"

    def test_allows_dots_dashes_stars_colons_slashes(self) -> None:
        assert sanitize_regex_query("foo.bar-baz*qux:path/file") == "foo.bar-baz*qux:path/file"

    def test_allows_spaces_in_content(self) -> None:
        assert sanitize_regex_query("hello world") == "hello world"

    def test_too_long_raises(self) -> None:
        with pytest.raises(ValueError, match="query too long"):
            sanitize_regex_query("a" * 129)

    def test_exactly_128_is_ok(self) -> None:
        result = sanitize_regex_query("a" * 128)
        assert len(result) == 128

    def test_special_chars_rejected(self) -> None:
        with pytest.raises(ValueError, match="unsupported characters"):
            sanitize_regex_query("hello;world")

    def test_parentheses_rejected(self) -> None:
        with pytest.raises(ValueError, match="unsupported characters"):
            sanitize_regex_query("foo(bar)")

    def test_square_brackets_rejected(self) -> None:
        with pytest.raises(ValueError, match="unsupported characters"):
            sanitize_regex_query("foo[0]")

    def test_pipe_rejected(self) -> None:
        with pytest.raises(ValueError, match="unsupported characters"):
            sanitize_regex_query("a|b")


# ── normalize_analysis_engines ───────────────────────────────────────────


class TestNormalizeAnalysisEngines:
    def test_none_returns_all_engines(self) -> None:
        result = normalize_analysis_engines(None)
        assert result == ["graphrag", "lsp", "joern"]

    def test_empty_list_returns_all_engines(self) -> None:
        result = normalize_analysis_engines([])
        assert result == ["graphrag", "lsp", "joern"]

    def test_single_valid_engine(self) -> None:
        assert normalize_analysis_engines(["graphrag"]) == ["graphrag"]

    def test_multiple_valid_engines(self) -> None:
        assert normalize_analysis_engines(["lsp", "joern"]) == ["lsp", "joern"]

    def test_case_insensitive(self) -> None:
        assert normalize_analysis_engines(["GRAPHRAG"]) == ["graphrag"]

    def test_strips_whitespace(self) -> None:
        assert normalize_analysis_engines(["  lsp  "]) == ["lsp"]

    def test_deduplicates(self) -> None:
        result = normalize_analysis_engines(["lsp", "lsp", "joern"])
        assert result == ["lsp", "joern"]

    def test_unsupported_engine_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported analysis engine 'badengine'"):
            normalize_analysis_engines(["badengine"])

    def test_all_empty_strings_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one analysis engine"):
            normalize_analysis_engines(["", "  "])

    def test_preserves_order(self) -> None:
        result = normalize_analysis_engines(["joern", "graphrag", "lsp"])
        assert result == ["joern", "graphrag", "lsp"]


# ── _as_int ──────────────────────────────────────────────────────────────


class TestAsInt:
    def test_int_passthrough(self) -> None:
        assert _as_int(42) == 42

    def test_float_truncates(self) -> None:
        assert _as_int(3.9) == 3

    def test_string_with_digits(self) -> None:
        assert _as_int("42") == 42

    def test_string_with_text_and_digits(self) -> None:
        assert _as_int("line 10 col 5") == 10

    def test_negative_int(self) -> None:
        assert _as_int(-7) == -7

    def test_negative_string(self) -> None:
        assert _as_int("-3") == -3

    def test_string_no_digits(self) -> None:
        assert _as_int("abc") == 0

    def test_none(self) -> None:
        assert _as_int(None) == 0

    def test_empty_string(self) -> None:
        assert _as_int("") == 0

    def test_bool_true(self) -> None:
        # bool is subclass of int
        assert _as_int(True) == 1

    def test_bool_false(self) -> None:
        assert _as_int(False) == 0

    def test_zero(self) -> None:
        assert _as_int(0) == 0


# ── _line_split ──────────────────────────────────────────────────────────


class TestLineSplit:
    def test_none_returns_empty(self) -> None:
        assert _line_split(None) == []

    def test_empty_string_returns_empty(self) -> None:
        assert _line_split("") == []

    def test_whitespace_only_returns_empty(self) -> None:
        assert _line_split("   \n  \n  ") == []

    def test_single_line(self) -> None:
        assert _line_split("hello") == ["hello"]

    def test_multiline(self) -> None:
        assert _line_split("line1\nline2\nline3") == ["line1", "line2", "line3"]

    def test_strips_individual_lines(self) -> None:
        assert _line_split("  a  \n  b  ") == ["a", "b"]

    def test_skips_blank_lines(self) -> None:
        assert _line_split("a\n\nb\n\nc") == ["a", "b", "c"]

    def test_list_input_returns_stringified(self) -> None:
        assert _line_split([1, "two", 3]) == ["1", "two", "3"]

    def test_list_is_not_split(self) -> None:
        # Lists get converted via str(item), not split
        result = _line_split(["a\nb", "c"])
        assert result == ["a\nb", "c"]

    def test_int_input(self) -> None:
        assert _line_split(42) == ["42"]


# ── _tsv_rows ────────────────────────────────────────────────────────────


class TestTsvRows:
    def test_none_returns_empty(self) -> None:
        assert _tsv_rows(None, 3) == []

    def test_empty_string_returns_empty(self) -> None:
        assert _tsv_rows("", 3) == []

    def test_basic_tsv(self) -> None:
        result = _tsv_rows("a\tb\tc", columns=3)
        assert result == [["a", "b", "c"]]

    def test_multiline_tsv(self) -> None:
        result = _tsv_rows("a\tb\nc\td", columns=2)
        assert result == [["a", "b"], ["c", "d"]]

    def test_pads_short_rows(self) -> None:
        result = _tsv_rows("a", columns=3)
        assert result == [["a", "", ""]]

    def test_truncates_long_rows(self) -> None:
        result = _tsv_rows("a\tb\tc\td\te", columns=3)
        assert result == [["a", "b", "c"]]

    def test_exact_column_count(self) -> None:
        result = _tsv_rows("x\ty", columns=2)
        assert result == [["x", "y"]]

    def test_skips_blank_lines(self) -> None:
        result = _tsv_rows("a\tb\n\nc\td", columns=2)
        assert result == [["a", "b"], ["c", "d"]]

    def test_single_column(self) -> None:
        result = _tsv_rows("hello\nworld", columns=1)
        assert result == [["hello"], ["world"]]


# ── _scala_escape ────────────────────────────────────────────────────────


class TestScalaEscape:
    def test_no_escaping_needed(self) -> None:
        assert _scala_escape("hello") == "hello"

    def test_escapes_backslash(self) -> None:
        assert _scala_escape("a\\b") == "a\\\\b"

    def test_escapes_double_quote(self) -> None:
        assert _scala_escape('a"b') == 'a\\"b'

    def test_escapes_both(self) -> None:
        assert _scala_escape('a\\b"c') == 'a\\\\b\\"c'

    def test_empty_string(self) -> None:
        assert _scala_escape("") == ""

    def test_multiple_backslashes(self) -> None:
        assert _scala_escape("\\\\") == "\\\\\\\\"

    def test_multiple_quotes(self) -> None:
        assert _scala_escape('""') == '\\"\\"'


# ── _sarif_level ─────────────────────────────────────────────────────────


class TestSarifLevel:
    def test_critical(self) -> None:
        assert _sarif_level("critical") == "error"

    def test_high(self) -> None:
        assert _sarif_level("high") == "error"

    def test_medium(self) -> None:
        assert _sarif_level("medium") == "warning"

    def test_low(self) -> None:
        assert _sarif_level("low") == "note"

    def test_none(self) -> None:
        assert _sarif_level(None) == "note"

    def test_unknown_value(self) -> None:
        assert _sarif_level("info") == "note"

    def test_empty_string(self) -> None:
        assert _sarif_level("") == "note"


# ── findings_to_sarif ───────────────────────────────────────────────────


class TestFindingsToSarif:
    def _make_finding(self, **overrides: Any) -> dict[str, Any]:
        base = {
            "finding_type": "taint_flow",
            "severity": "high",
            "message": "Potential injection",
            "filename": "app/main.py",
            "line_number": 42,
        }
        base.update(overrides)
        return base

    def test_empty_findings(self) -> None:
        cid = uuid4()
        sid = uuid4()
        result = findings_to_sarif(collection_id=cid, scenario_id=sid, findings=[])
        assert result["version"] == "2.1.0"
        assert result["$schema"] == "https://json.schemastore.org/sarif-2.1.0.json"
        assert len(result["runs"]) == 1
        run = result["runs"][0]
        assert run["results"] == []
        assert run["tool"]["driver"]["rules"] == []
        assert f"collection:{cid}:scenario:{sid}" in run["automationDetails"]["id"]

    def test_single_finding(self) -> None:
        finding = self._make_finding()
        result = findings_to_sarif(
            collection_id=uuid4(),
            scenario_id=uuid4(),
            findings=[finding],
        )
        run = result["runs"][0]
        assert len(run["results"]) == 1
        sarif_result = run["results"][0]
        assert sarif_result["ruleId"] == "taint_flow"
        assert sarif_result["level"] == "error"  # high -> error
        assert sarif_result["message"]["text"] == "Potential injection"
        location = sarif_result["locations"][0]["physicalLocation"]
        assert location["artifactLocation"]["uri"] == "app/main.py"
        assert location["region"]["startLine"] == 42

    def test_multiple_findings_same_type_single_rule(self) -> None:
        findings = [
            self._make_finding(message="finding 1"),
            self._make_finding(message="finding 2"),
        ]
        result = findings_to_sarif(
            collection_id=uuid4(),
            scenario_id=uuid4(),
            findings=findings,
        )
        run = result["runs"][0]
        assert len(run["results"]) == 2
        # Same finding_type means only one rule
        assert len(run["tool"]["driver"]["rules"]) == 1

    def test_different_finding_types_multiple_rules(self) -> None:
        findings = [
            self._make_finding(finding_type="taint_flow"),
            self._make_finding(finding_type="sql_injection"),
        ]
        result = findings_to_sarif(
            collection_id=uuid4(),
            scenario_id=uuid4(),
            findings=findings,
        )
        run = result["runs"][0]
        assert len(run["tool"]["driver"]["rules"]) == 2

    def test_severity_mapping_in_sarif(self) -> None:
        findings = [
            self._make_finding(severity="critical"),
            self._make_finding(severity="medium", finding_type="type_b"),
            self._make_finding(severity="low", finding_type="type_c"),
        ]
        result = findings_to_sarif(
            collection_id=uuid4(),
            scenario_id=uuid4(),
            findings=findings,
        )
        levels = [r["level"] for r in result["runs"][0]["results"]]
        assert levels == ["error", "warning", "note"]

    def test_missing_fields_use_defaults(self) -> None:
        finding: dict[str, Any] = {}
        result = findings_to_sarif(
            collection_id=uuid4(),
            scenario_id=uuid4(),
            findings=[finding],
        )
        sarif_result = result["runs"][0]["results"][0]
        assert sarif_result["ruleId"] == "contextmine.rule"
        assert sarif_result["level"] == "note"  # None severity -> note
        assert sarif_result["message"]["text"] == ""
        location = sarif_result["locations"][0]["physicalLocation"]
        assert location["artifactLocation"]["uri"] == ""
        assert location["region"]["startLine"] == 1

    def test_rule_short_description_formats_nicely(self) -> None:
        finding = self._make_finding(finding_type="sql_injection")
        result = findings_to_sarif(
            collection_id=uuid4(),
            scenario_id=uuid4(),
            findings=[finding],
        )
        rule = result["runs"][0]["tool"]["driver"]["rules"][0]
        assert rule["shortDescription"]["text"] == "Sql Injection"
        assert "security" in rule["properties"]["tags"]
        assert "digital-twin" in rule["properties"]["tags"]


# ── _symbol_kind_to_name ─────────────────────────────────────────────────


class TestSymbolKindToName:
    @pytest.mark.parametrize(
        "kind,expected",
        [
            (1, "file"),
            (2, "module"),
            (3, "namespace"),
            (4, "package"),
            (5, "class"),
            (6, "method"),
            (7, "property"),
            (8, "field"),
            (9, "constructor"),
            (10, "enum"),
            (11, "interface"),
            (12, "function"),
            (13, "variable"),
        ],
    )
    def test_known_kinds(self, kind: int, expected: str) -> None:
        assert _symbol_kind_to_name(kind) == expected

    def test_unknown_int(self) -> None:
        assert _symbol_kind_to_name(99) == "kind_99"

    def test_string_convertible_to_int(self) -> None:
        assert _symbol_kind_to_name("6") == "method"

    def test_non_numeric_string(self) -> None:
        assert _symbol_kind_to_name("custom") == "custom"

    def test_none(self) -> None:
        assert _symbol_kind_to_name(None) == "unknown"

    def test_zero(self) -> None:
        # 0 is not in the mapping
        assert _symbol_kind_to_name(0) == "kind_0"

    def test_float_kind(self) -> None:
        # float 5.0 -> int 5 -> "class"
        assert _symbol_kind_to_name(5.0) == "class"


# ── _flatten_lsp_symbols ────────────────────────────────────────────────


class TestFlattenLspSymbols:
    def test_empty_symbols(self) -> None:
        result = _flatten_lsp_symbols(file_path="test.py", symbols=[])
        assert result == []

    def test_single_symbol(self) -> None:
        symbols = [
            {
                "name": "MyClass",
                "kind": 5,
                "selectionRange": {
                    "start": {"line": 10, "character": 4},
                    "end": {"line": 10, "character": 11},
                },
            }
        ]
        result = _flatten_lsp_symbols(file_path="test.py", symbols=symbols)
        assert len(result) == 1
        assert result[0]["name"] == "MyClass"
        assert result[0]["kind"] == "class"
        assert result[0]["kind_id"] == 5
        assert result[0]["file_path"] == "test.py"
        assert result[0]["line_number"] == 11  # 0-indexed + 1
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
                        "selectionRange": {"start": {"line": 5, "character": 4}},
                    },
                ],
            }
        ]
        result = _flatten_lsp_symbols(file_path="test.py", symbols=symbols)
        assert len(result) == 2
        names = {r["name"] for r in result}
        assert names == {"MyClass", "my_method"}

    def test_uses_range_fallback_when_no_selection_range(self) -> None:
        symbols = [
            {
                "name": "func",
                "kind": 12,
                "range": {
                    "start": {"line": 20, "character": 0},
                    "end": {"line": 30, "character": 0},
                },
            }
        ]
        result = _flatten_lsp_symbols(file_path="test.py", symbols=symbols)
        assert result[0]["line_number"] == 21

    def test_missing_range_uses_defaults(self) -> None:
        symbols = [{"name": "func", "kind": 12}]
        result = _flatten_lsp_symbols(file_path="test.py", symbols=symbols)
        assert result[0]["line_number"] == 1  # _as_int(None) + 1 = 0 + 1
        assert result[0]["column"] == 0

    def test_non_dict_items_skipped(self) -> None:
        symbols = [None, "not_a_dict", 42, {"name": "real", "kind": 12}]
        result = _flatten_lsp_symbols(file_path="test.py", symbols=symbols)
        assert len(result) == 1
        assert result[0]["name"] == "real"

    def test_deeply_nested_children(self) -> None:
        symbols = [
            {
                "name": "outer",
                "kind": 3,
                "selectionRange": {"start": {"line": 0, "character": 0}},
                "children": [
                    {
                        "name": "middle",
                        "kind": 5,
                        "selectionRange": {"start": {"line": 1, "character": 0}},
                        "children": [
                            {
                                "name": "inner",
                                "kind": 6,
                                "selectionRange": {"start": {"line": 2, "character": 0}},
                            },
                        ],
                    },
                ],
            }
        ]
        result = _flatten_lsp_symbols(file_path="test.py", symbols=symbols)
        assert len(result) == 3
        names = {r["name"] for r in result}
        assert names == {"outer", "middle", "inner"}

    def test_missing_name_uses_empty_string(self) -> None:
        symbols = [{"kind": 12}]
        result = _flatten_lsp_symbols(file_path="test.py", symbols=symbols)
        assert result[0]["name"] == ""

    def test_file_path_propagated(self) -> None:
        symbols = [{"name": "a", "kind": 12}]
        result = _flatten_lsp_symbols(file_path="/some/path.py", symbols=symbols)
        assert result[0]["file_path"] == "/some/path.py"


# ── _escape_like (via import) ───────────────────────────────────────────


class TestEscapeLike:
    """Test _escape_like SQL LIKE wildcard escaping."""

    def test_basic(self) -> None:
        from contextmine_core.twin.ops import _escape_like

        assert _escape_like("hello") == "hello"

    def test_percent(self) -> None:
        from contextmine_core.twin.ops import _escape_like

        assert _escape_like("100%") == "100\\%"

    def test_underscore(self) -> None:
        from contextmine_core.twin.ops import _escape_like

        assert _escape_like("my_table") == "my\\_table"

    def test_backslash(self) -> None:
        from contextmine_core.twin.ops import _escape_like

        assert _escape_like("path\\file") == "path\\\\file"

    def test_all_together(self) -> None:
        from contextmine_core.twin.ops import _escape_like

        assert _escape_like("a%b_c\\d") == "a\\%b\\_c\\\\d"


# ── _language_patterns ──────────────────────────────────────────────────


class TestLanguagePatterns:
    def test_known_language(self) -> None:
        from contextmine_core.twin.ops import SOURCE_PATTERNS, _language_patterns

        result = _language_patterns("python", SOURCE_PATTERNS)
        assert result == SOURCE_PATTERNS["python"]

    def test_unknown_language_falls_back_to_python(self) -> None:
        from contextmine_core.twin.ops import SOURCE_PATTERNS, _language_patterns

        result = _language_patterns("rust", SOURCE_PATTERNS)
        assert result == SOURCE_PATTERNS["python"]

    def test_none_defaults_to_python(self) -> None:
        from contextmine_core.twin.ops import SINK_PATTERNS, _language_patterns

        result = _language_patterns(None, SINK_PATTERNS)
        assert result == SINK_PATTERNS["python"]

    def test_case_insensitive(self) -> None:
        from contextmine_core.twin.ops import SOURCE_PATTERNS, _language_patterns

        result = _language_patterns("Python", SOURCE_PATTERNS)
        assert result == SOURCE_PATTERNS["python"]


# ── _read_node_file_path / _read_node_source_id (with mock nodes) ──────


class TestReadNodeHelpers:
    """Test _read_node_file_path and _read_node_source_id with mock TwinNode objects."""

    def _mock_node(
        self,
        natural_key: str = "file:test.py",
        meta: dict | None = None,
        source_id: UUID | None = None,
    ) -> Any:
        """Create a lightweight mock that behaves like TwinNode."""

        class FakeNode:
            def __init__(
                self, *, natural_key: str, meta: dict[str, Any], source_id: UUID | None
            ) -> None:
                self.natural_key = natural_key
                self.meta = meta
                self.source_id = source_id

        return FakeNode(natural_key=natural_key, meta=meta, source_id=source_id)

    def test_read_file_path_from_meta(self) -> None:
        from contextmine_core.twin.ops import _read_node_file_path

        node = self._mock_node(meta={"file_path": "src/main.py"}, natural_key="sym:foo")
        assert _read_node_file_path(node) == "src/main.py"

    def test_read_file_path_from_natural_key(self) -> None:
        from contextmine_core.twin.ops import _read_node_file_path

        node = self._mock_node(natural_key="file:src/main.py", meta={})
        assert _read_node_file_path(node) == "src/main.py"

    def test_read_file_path_returns_none_when_missing(self) -> None:
        from contextmine_core.twin.ops import _read_node_file_path

        node = self._mock_node(natural_key="symbol:foo", meta={})
        assert _read_node_file_path(node) is None

    def test_read_file_path_strips_whitespace_in_meta(self) -> None:
        from contextmine_core.twin.ops import _read_node_file_path

        node = self._mock_node(meta={"file_path": "  src/main.py  "}, natural_key="sym:foo")
        assert _read_node_file_path(node) == "src/main.py"

    def test_read_file_path_empty_meta_file_path(self) -> None:
        from contextmine_core.twin.ops import _read_node_file_path

        node = self._mock_node(meta={"file_path": "  "}, natural_key="sym:foo")
        assert _read_node_file_path(node) is None

    def test_read_source_id_from_attribute(self) -> None:
        from contextmine_core.twin.ops import _read_node_source_id

        sid = uuid4()
        node = self._mock_node(source_id=sid)
        assert _read_node_source_id(node) == sid

    def test_read_source_id_from_meta(self) -> None:
        from contextmine_core.twin.ops import _read_node_source_id

        sid = uuid4()
        node = self._mock_node(source_id=None, meta={"source_id": str(sid)})
        assert _read_node_source_id(node) == sid

    def test_read_source_id_returns_none_when_missing(self) -> None:
        from contextmine_core.twin.ops import _read_node_source_id

        node = self._mock_node(source_id=None, meta={})
        assert _read_node_source_id(node) is None

    def test_read_source_id_invalid_uuid_in_meta_returns_none(self) -> None:
        from contextmine_core.twin.ops import _read_node_source_id

        node = self._mock_node(source_id=None, meta={"source_id": "not-a-uuid"})
        assert _read_node_source_id(node) is None


# ── SEVERITY_ORDER constant ─────────────────────────────────────────────


class TestSeverityOrder:
    def test_ordering(self) -> None:
        from contextmine_core.twin.ops import SEVERITY_ORDER

        assert SEVERITY_ORDER["critical"] > SEVERITY_ORDER["high"]
        assert SEVERITY_ORDER["high"] > SEVERITY_ORDER["medium"]
        assert SEVERITY_ORDER["medium"] > SEVERITY_ORDER["low"]


# ── SOURCE_PATTERNS / SINK_PATTERNS / SANITIZER_PATTERNS constants ──────


class TestPatternConstants:
    def test_source_patterns_cover_common_languages(self) -> None:
        from contextmine_core.twin.ops import SOURCE_PATTERNS

        expected_languages = {"python", "javascript", "typescript", "java", "go", "php"}
        assert expected_languages == set(SOURCE_PATTERNS.keys())

    def test_sink_patterns_cover_common_languages(self) -> None:
        from contextmine_core.twin.ops import SINK_PATTERNS

        expected_languages = {"python", "javascript", "typescript", "java", "go", "php"}
        assert expected_languages == set(SINK_PATTERNS.keys())

    def test_sanitizer_patterns_cover_common_languages(self) -> None:
        from contextmine_core.twin.ops import SANITIZER_PATTERNS

        expected_languages = {"python", "javascript", "typescript", "java", "go", "php"}
        assert expected_languages == set(SANITIZER_PATTERNS.keys())

    def test_all_pattern_values_are_nonempty_lists(self) -> None:
        from contextmine_core.twin.ops import SANITIZER_PATTERNS, SINK_PATTERNS, SOURCE_PATTERNS

        for patterns in (SOURCE_PATTERNS, SINK_PATTERNS, SANITIZER_PATTERNS):
            for lang, tokens in patterns.items():
                assert isinstance(tokens, list), f"Expected list for {lang}"
                assert len(tokens) > 0, f"Empty pattern list for {lang}"


# ── DEFAULT_EXTRACTOR_VERSION constant ──────────────────────────────────


class TestDefaultExtractorVersion:
    def test_value(self) -> None:
        from contextmine_core.twin.ops import DEFAULT_EXTRACTOR_VERSION

        assert DEFAULT_EXTRACTOR_VERSION == "scip-kg-v1"


# ── ANALYSIS_ENGINES / LSP_METHOD_KINDS constants ───────────────────────


class TestEngineConstants:
    def test_analysis_engines_tuple(self) -> None:
        from contextmine_core.twin.ops import ANALYSIS_ENGINES

        assert ANALYSIS_ENGINES == ("graphrag", "lsp", "joern")

    def test_lsp_method_kinds(self) -> None:
        from contextmine_core.twin.ops import LSP_METHOD_KINDS

        assert {5, 6, 12} == LSP_METHOD_KINDS


# ═══════════════════════════════════════════════════════════════════════════
# Async DB function tests with mocked AsyncSession
# ═══════════════════════════════════════════════════════════════════════════


def _make_mock_session() -> MagicMock:
    """Build a mock AsyncSession with chainable execute results."""
    session = MagicMock()
    session.execute = AsyncMock()
    session.flush = AsyncMock()
    session.add = MagicMock()
    return session


def _scalar_one_or_none(value: Any) -> MagicMock:
    """Create a mock result that returns *value* for .scalar_one_or_none()."""
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    return result


def _scalar_one(value: Any) -> MagicMock:
    """Create a mock result that returns *value* for .scalar_one()."""
    result = MagicMock()
    result.scalar_one.return_value = value
    return result


def _scalars_all(values: list[Any]) -> MagicMock:
    """Create a mock result that returns *values* for .scalars().all()."""
    result = MagicMock()
    scalars = MagicMock()
    scalars.all.return_value = values
    scalars.first.return_value = values[0] if values else None
    result.scalars.return_value = scalars
    return result


# ── get_or_create_source_version ────────────────────────────────────────


class TestGetOrCreateSourceVersion:
    @pytest.mark.anyio
    async def test_returns_existing_version(self) -> None:
        from contextmine_core.twin.ops import get_or_create_source_version

        session = _make_mock_session()
        existing = MagicMock()
        existing.id = uuid4()
        session.execute.return_value = _scalar_one_or_none(existing)

        result = await get_or_create_source_version(
            session,
            collection_id=uuid4(),
            source_id=uuid4(),
            revision_key="rev1",
        )

        assert result is existing
        session.add.assert_not_called()

    @pytest.mark.anyio
    async def test_creates_new_version_when_none_exists(self) -> None:
        from contextmine_core.twin.ops import get_or_create_source_version

        session = _make_mock_session()
        session.execute.return_value = _scalar_one_or_none(None)

        result = await get_or_create_source_version(
            session,
            collection_id=uuid4(),
            source_id=uuid4(),
            revision_key="rev1",
            language_profile="python",
            status="queued",
        )

        session.add.assert_called_once()
        await session.flush()
        assert result.revision_key == "rev1"
        assert result.status == "queued"
        assert result.language_profile == "python"
        assert result.stats == {}


# ── set_source_version_status ────────────────────────────────────────────


class TestSetSourceVersionStatus:
    @pytest.mark.anyio
    async def test_transitions_status(self) -> None:
        from contextmine_core.twin.ops import set_source_version_status

        session = _make_mock_session()
        sv = MagicMock()
        sv.id = uuid4()
        sv.status = "queued"
        sv.stats = {"old": True}
        sv.started_at = None
        sv.finished_at = None
        session.execute.return_value = _scalar_one(sv)

        result = await set_source_version_status(
            session,
            source_version_id=sv.id,
            status="materializing",
            stats={"new": True},
            started=True,
        )

        assert result.status == "materializing"
        assert result.stats == {"old": True, "new": True}
        assert result.started_at is not None
        assert result.updated_at is not None

    @pytest.mark.anyio
    async def test_finished_flag_sets_finished_at(self) -> None:
        from contextmine_core.twin.ops import set_source_version_status

        session = _make_mock_session()
        sv = MagicMock()
        sv.id = uuid4()
        sv.status = "materializing"
        sv.stats = {}
        sv.started_at = datetime(2024, 1, 1)
        sv.finished_at = None
        session.execute.return_value = _scalar_one(sv)

        result = await set_source_version_status(
            session,
            source_version_id=sv.id,
            status="ready",
            finished=True,
        )

        assert result.status == "ready"
        assert result.finished_at is not None

    @pytest.mark.anyio
    async def test_started_flag_not_overwritten_when_already_set(self) -> None:
        from contextmine_core.twin.ops import set_source_version_status

        session = _make_mock_session()
        original_start = datetime(2024, 1, 1)
        sv = MagicMock()
        sv.id = uuid4()
        sv.status = "materializing"
        sv.stats = {}
        sv.started_at = original_start
        sv.finished_at = None
        session.execute.return_value = _scalar_one(sv)

        await set_source_version_status(
            session,
            source_version_id=sv.id,
            status="ready",
            started=True,
        )

        # started_at should NOT have been overwritten
        assert sv.started_at is original_start


# ── record_twin_event ────────────────────────────────────────────────────


class TestRecordTwinEvent:
    @pytest.mark.anyio
    async def test_returns_existing_event_idempotently(self) -> None:
        from contextmine_core.twin.ops import record_twin_event

        session = _make_mock_session()
        existing = MagicMock()
        existing.id = uuid4()
        session.execute.return_value = _scalar_one_or_none(existing)

        result = await record_twin_event(
            session,
            collection_id=uuid4(),
            scenario_id=None,
            source_id=None,
            source_version_id=None,
            event_type="test",
            status="queued",
            payload=None,
            idempotency_key="key1",
        )

        assert result is existing
        session.add.assert_not_called()

    @pytest.mark.anyio
    async def test_creates_new_event_when_not_found(self) -> None:
        from contextmine_core.twin.ops import record_twin_event

        session = _make_mock_session()
        session.execute.return_value = _scalar_one_or_none(None)

        cid = uuid4()
        result = await record_twin_event(
            session,
            collection_id=cid,
            scenario_id=uuid4(),
            source_id=uuid4(),
            source_version_id=uuid4(),
            event_type="refresh_requested",
            status="queued",
            payload={"force": True},
            idempotency_key="key2",
            error="some error",
        )

        session.add.assert_called_once()
        assert result.collection_id == cid
        assert result.event_type == "refresh_requested"
        assert result.status == "queued"
        assert result.payload == {"force": True}
        assert result.error == "some error"

    @pytest.mark.anyio
    async def test_none_payload_becomes_empty_dict(self) -> None:
        from contextmine_core.twin.ops import record_twin_event

        session = _make_mock_session()
        session.execute.return_value = _scalar_one_or_none(None)

        result = await record_twin_event(
            session,
            collection_id=uuid4(),
            scenario_id=None,
            source_id=None,
            source_version_id=None,
            event_type="test",
            status="queued",
            payload=None,
            idempotency_key="key3",
        )

        assert result.payload == {}


# ── invalidate_analysis_cache_for_scenario ───────────────────────────────


class TestInvalidateAnalysisCacheForScenario:
    @pytest.mark.anyio
    async def test_returns_rowcount(self) -> None:
        from contextmine_core.twin.ops import invalidate_analysis_cache_for_scenario

        session = _make_mock_session()
        result = MagicMock()
        result.rowcount = 5
        session.execute.return_value = result

        count = await invalidate_analysis_cache_for_scenario(session, scenario_id=uuid4())

        assert count == 5

    @pytest.mark.anyio
    async def test_returns_zero_when_rowcount_is_none(self) -> None:
        from contextmine_core.twin.ops import invalidate_analysis_cache_for_scenario

        session = _make_mock_session()
        result = MagicMock(spec=[])  # no rowcount attr
        session.execute.return_value = result

        count = await invalidate_analysis_cache_for_scenario(session, scenario_id=uuid4())

        assert count == 0


# ── _resolve_status_scenario ─────────────────────────────────────────────


class TestResolveStatusScenario:
    @pytest.mark.anyio
    async def test_with_explicit_scenario_id(self) -> None:
        from contextmine_core.twin.ops import _resolve_status_scenario

        session = _make_mock_session()
        scenario = MagicMock()
        scenario.id = uuid4()
        session.execute.return_value = _scalar_one_or_none(scenario)

        result = await _resolve_status_scenario(
            session, collection_id=uuid4(), scenario_id=scenario.id
        )

        assert result is scenario

    @pytest.mark.anyio
    async def test_with_none_scenario_id_falls_back_to_as_is(self) -> None:
        from contextmine_core.twin.ops import _resolve_status_scenario

        session = _make_mock_session()
        scenario = MagicMock()
        scenario.id = uuid4()
        scenario.is_as_is = True
        session.execute.return_value = _scalar_one_or_none(scenario)

        result = await _resolve_status_scenario(session, collection_id=uuid4(), scenario_id=None)

        assert result is scenario

    @pytest.mark.anyio
    async def test_returns_none_when_not_found(self) -> None:
        from contextmine_core.twin.ops import _resolve_status_scenario

        session = _make_mock_session()
        session.execute.return_value = _scalar_one_or_none(None)

        result = await _resolve_status_scenario(session, collection_id=uuid4(), scenario_id=uuid4())

        assert result is None


# ── _current_analysis_cache_key ──────────────────────────────────────────


class TestCurrentAnalysisCacheKey:
    @pytest.mark.anyio
    async def test_empty_versions_returns_consistent_hash(self) -> None:
        from contextmine_core.twin.ops import _current_analysis_cache_key

        session = _make_mock_session()
        session.execute.return_value = _scalars_all([])

        cid = uuid4()
        result = await _current_analysis_cache_key(
            session, collection_id=cid, projection_profile="summary"
        )

        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    @pytest.mark.anyio
    async def test_with_ready_versions(self) -> None:
        from contextmine_core.twin.ops import _current_analysis_cache_key

        session = _make_mock_session()
        v1 = MagicMock()
        v1.source_id = uuid4()
        v1.revision_key = "rev1"
        v1.extractor_version = "v1"
        session.execute.return_value = _scalars_all([v1])

        result = await _current_analysis_cache_key(
            session, collection_id=uuid4(), projection_profile="test"
        )

        assert len(result) == 64

    @pytest.mark.anyio
    async def test_different_profiles_produce_different_keys(self) -> None:
        from contextmine_core.twin.ops import _current_analysis_cache_key

        session = _make_mock_session()
        session.execute.return_value = _scalars_all([])

        cid = uuid4()
        k1 = await _current_analysis_cache_key(
            session, collection_id=cid, projection_profile="profile_a"
        )
        k2 = await _current_analysis_cache_key(
            session, collection_id=cid, projection_profile="profile_b"
        )

        assert k1 != k2


# ── _read_cached_analysis ────────────────────────────────────────────────


class TestReadCachedAnalysis:
    @pytest.mark.anyio
    async def test_returns_payload_when_cache_hit(self) -> None:
        from contextmine_core.twin.ops import _read_cached_analysis

        session = _make_mock_session()
        cache_row = MagicMock()
        cache_row.payload = {"answer": 42}
        session.execute.return_value = _scalar_one_or_none(cache_row)

        result = await _read_cached_analysis(
            session,
            scenario_id=uuid4(),
            tool_name="test_tool",
            cache_key="ck",
            params_hash="ph",
        )

        assert result == {"answer": 42}

    @pytest.mark.anyio
    async def test_returns_none_on_cache_miss(self) -> None:
        from contextmine_core.twin.ops import _read_cached_analysis

        session = _make_mock_session()
        session.execute.return_value = _scalar_one_or_none(None)

        result = await _read_cached_analysis(
            session,
            scenario_id=uuid4(),
            tool_name="test_tool",
            cache_key="ck",
            params_hash="ph",
        )

        assert result is None

    @pytest.mark.anyio
    async def test_returns_empty_dict_when_payload_is_none(self) -> None:
        from contextmine_core.twin.ops import _read_cached_analysis

        session = _make_mock_session()
        cache_row = MagicMock()
        cache_row.payload = None
        session.execute.return_value = _scalar_one_or_none(cache_row)

        result = await _read_cached_analysis(
            session,
            scenario_id=uuid4(),
            tool_name="test_tool",
            cache_key="ck",
            params_hash="ph",
        )

        assert result == {}


# ── _write_cached_analysis ───────────────────────────────────────────────


class TestWriteCachedAnalysis:
    @pytest.mark.anyio
    async def test_updates_existing_row(self) -> None:
        from contextmine_core.twin.ops import _write_cached_analysis

        session = _make_mock_session()
        existing = MagicMock()
        existing.payload = {"old": True}
        session.execute.return_value = _scalar_one_or_none(existing)

        await _write_cached_analysis(
            session,
            scenario_id=uuid4(),
            tool_name="test_tool",
            cache_key="ck",
            params_hash="ph",
            payload={"new": True},
            ttl_seconds=3600,
        )

        assert existing.payload == {"new": True}
        assert existing.expires_at is not None
        session.add.assert_not_called()

    @pytest.mark.anyio
    async def test_creates_new_row_on_miss(self) -> None:
        from contextmine_core.twin.ops import _write_cached_analysis

        session = _make_mock_session()
        session.execute.return_value = _scalar_one_or_none(None)

        await _write_cached_analysis(
            session,
            scenario_id=uuid4(),
            tool_name="test_tool",
            cache_key="ck",
            params_hash="ph",
            payload={"data": 1},
            ttl_seconds=300,
        )

        session.add.assert_called_once()
        added = session.add.call_args[0][0]
        assert added.payload == {"data": 1}
        assert added.tool_name == "test_tool"

    @pytest.mark.anyio
    async def test_ttl_clamps_to_minimum_one_second(self) -> None:
        from contextmine_core.twin.ops import _write_cached_analysis

        session = _make_mock_session()
        session.execute.return_value = _scalar_one_or_none(None)

        await _write_cached_analysis(
            session,
            scenario_id=uuid4(),
            tool_name="t",
            cache_key="ck",
            params_hash="ph",
            payload={},
            ttl_seconds=0,
        )

        session.add.assert_called_once()
        added = session.add.call_args[0][0]
        # The function clamps ttl to max(0, 1) = 1
        # expires_at = now + timedelta(seconds=1) which should be in the future
        assert added.expires_at is not None


# ── _resolve_node_by_ref ─────────────────────────────────────────────────


class TestResolveNodeByRef:
    @pytest.mark.anyio
    async def test_finds_by_uuid(self) -> None:
        from contextmine_core.twin.ops import _resolve_node_by_ref

        session = _make_mock_session()
        node = MagicMock()
        node.id = uuid4()
        session.execute.return_value = _scalar_one_or_none(node)

        result = await _resolve_node_by_ref(session, scenario_id=uuid4(), node_ref=str(node.id))

        assert result is node

    @pytest.mark.anyio
    async def test_falls_back_to_natural_key_when_uuid_not_found(self) -> None:
        from contextmine_core.twin.ops import _resolve_node_by_ref

        session = _make_mock_session()
        node = MagicMock()
        node.id = uuid4()
        # First call (UUID lookup) returns None, second call (natural_key) returns the node
        session.execute.side_effect = [
            _scalar_one_or_none(None),
            _scalars_all([node]),
        ]

        result = await _resolve_node_by_ref(session, scenario_id=uuid4(), node_ref=str(uuid4()))

        assert result is node

    @pytest.mark.anyio
    async def test_finds_by_natural_key_when_non_uuid(self) -> None:
        from contextmine_core.twin.ops import _resolve_node_by_ref

        session = _make_mock_session()
        node = MagicMock()
        node.id = uuid4()
        session.execute.return_value = _scalars_all([node])

        result = await _resolve_node_by_ref(
            session, scenario_id=uuid4(), node_ref="file:src/main.py"
        )

        assert result is node

    @pytest.mark.anyio
    async def test_returns_none_when_nothing_found(self) -> None:
        from contextmine_core.twin.ops import _resolve_node_by_ref

        session = _make_mock_session()
        session.execute.return_value = _scalars_all([])

        result = await _resolve_node_by_ref(session, scenario_id=uuid4(), node_ref="nonexistent")

        assert result is None


# ── store_findings ───────────────────────────────────────────────────────


class TestStoreFindings:
    @pytest.mark.anyio
    async def test_creates_new_findings(self) -> None:
        from contextmine_core.twin.ops import store_findings

        session = _make_mock_session()
        scenario = MagicMock()
        scenario.id = uuid4()

        # _resolve_analysis_scenario -> _resolve_status_scenario -> scenario
        session.execute.side_effect = [
            # _resolve_status_scenario (called by _resolve_analysis_scenario)
            _scalar_one_or_none(scenario),
            # First finding: fingerprint lookup -> not found
            _scalar_one_or_none(None),
            # Second finding: fingerprint lookup -> not found
            _scalar_one_or_none(None),
        ]

        cid = uuid4()
        result = await store_findings(
            session,
            collection_id=cid,
            scenario_id=None,
            findings=[
                {
                    "finding_type": "taint_flow",
                    "severity": "high",
                    "message": "SQL injection",
                    "filename": "app.py",
                    "line_number": 10,
                },
                {
                    "finding_type": "xss",
                    "severity": "medium",
                    "message": "XSS risk",
                    "filename": "views.py",
                    "line_number": 20,
                },
            ],
        )

        assert result["created"] == 2
        assert len(result["ids"]) == 2
        assert result["collection_id"] == str(cid)
        assert result["scenario_id"] == str(scenario.id)
        # Two findings added
        assert session.add.call_count == 2

    @pytest.mark.anyio
    async def test_updates_existing_finding(self) -> None:
        from contextmine_core.twin.ops import store_findings

        session = _make_mock_session()
        scenario = MagicMock()
        scenario.id = uuid4()

        existing_finding = MagicMock()
        existing_finding.id = uuid4()
        existing_finding.severity = "low"
        existing_finding.confidence = "low"
        existing_finding.status = "open"
        existing_finding.message = "old message"
        existing_finding.flow_data = {}
        existing_finding.meta = {}

        session.execute.side_effect = [
            _scalar_one_or_none(scenario),  # resolve scenario
            _scalar_one_or_none(existing_finding),  # fingerprint match
        ]

        result = await store_findings(
            session,
            collection_id=uuid4(),
            scenario_id=scenario.id,
            findings=[
                {
                    "finding_type": "taint_flow",
                    "severity": "HIGH",
                    "message": "Updated message",
                    "filename": "app.py",
                    "line_number": 10,
                },
            ],
        )

        assert result["created"] == 1
        assert str(existing_finding.id) in result["ids"]
        # Existing finding should be updated
        assert existing_finding.severity == "high"
        assert existing_finding.message == "Updated message"

    @pytest.mark.anyio
    async def test_raises_when_scenario_not_found(self) -> None:
        from contextmine_core.twin.ops import store_findings

        session = _make_mock_session()
        session.execute.return_value = _scalar_one_or_none(None)

        with pytest.raises(ValueError, match="Scenario not found"):
            await store_findings(
                session,
                collection_id=uuid4(),
                scenario_id=uuid4(),
                findings=[],
            )


# ── list_findings ────────────────────────────────────────────────────────


class TestListFindings:
    @pytest.mark.anyio
    async def test_lists_findings_for_scenario(self) -> None:
        from contextmine_core.twin.ops import list_findings

        session = _make_mock_session()
        scenario = MagicMock()
        scenario.id = uuid4()

        now = datetime.now(UTC)
        finding = MagicMock()
        finding.id = uuid4()
        finding.finding_type = "taint_flow"
        finding.severity = "high"
        finding.confidence = "high"
        finding.status = "open"
        finding.filename = "app.py"
        finding.line_number = 42
        finding.message = "Injection detected"
        finding.flow_data = {"path": ["a", "b"]}
        finding.meta = {}
        finding.created_at = now
        finding.updated_at = now

        session.execute.side_effect = [
            _scalar_one_or_none(scenario),  # resolve scenario
            _scalars_all([finding]),  # findings query
        ]

        result = await list_findings(
            session,
            collection_id=uuid4(),
            scenario_id=None,
            limit=50,
            page=0,
        )

        assert result["total"] == 1
        assert len(result["items"]) == 1
        assert result["items"][0]["finding_type"] == "taint_flow"
        assert result["items"][0]["severity"] == "high"

    @pytest.mark.anyio
    async def test_filters_by_min_severity(self) -> None:
        from contextmine_core.twin.ops import list_findings

        session = _make_mock_session()
        scenario = MagicMock()
        scenario.id = uuid4()

        now = datetime.now(UTC)

        def _make_finding(sev: str) -> MagicMock:
            f = MagicMock()
            f.id = uuid4()
            f.finding_type = "test"
            f.severity = sev
            f.confidence = "high"
            f.status = "open"
            f.filename = "f.py"
            f.line_number = 1
            f.message = "msg"
            f.flow_data = {}
            f.meta = {}
            f.created_at = now
            f.updated_at = now
            return f

        findings = [_make_finding("critical"), _make_finding("low"), _make_finding("medium")]

        session.execute.side_effect = [
            _scalar_one_or_none(scenario),
            _scalars_all(findings),
        ]

        result = await list_findings(
            session,
            collection_id=uuid4(),
            scenario_id=None,
            min_severity="high",
            limit=50,
            page=0,
        )

        # Only critical is >= high (threshold = 3)
        assert result["total"] == 1
        assert result["items"][0]["severity"] == "critical"

    @pytest.mark.anyio
    async def test_pagination(self) -> None:
        from contextmine_core.twin.ops import list_findings

        session = _make_mock_session()
        scenario = MagicMock()
        scenario.id = uuid4()

        now = datetime.now(UTC)
        findings = []
        for i in range(5):
            f = MagicMock()
            f.id = uuid4()
            f.finding_type = "test"
            f.severity = "medium"
            f.confidence = "medium"
            f.status = "open"
            f.filename = f"file_{i}.py"
            f.line_number = i
            f.message = f"msg {i}"
            f.flow_data = {}
            f.meta = {}
            f.created_at = now
            f.updated_at = now
            findings.append(f)

        session.execute.side_effect = [
            _scalar_one_or_none(scenario),
            _scalars_all(findings),
        ]

        result = await list_findings(
            session,
            collection_id=uuid4(),
            scenario_id=None,
            limit=2,
            page=1,
        )

        assert result["total"] == 5
        assert len(result["items"]) == 2
        assert result["page"] == 1


# ── export_findings_sarif ────────────────────────────────────────────────


class TestExportFindingsSarif:
    @pytest.mark.anyio
    async def test_exports_sarif_format(self) -> None:
        from contextmine_core.twin.ops import export_findings_sarif

        session = _make_mock_session()
        scenario = MagicMock()
        scenario.id = uuid4()

        now = datetime.now(UTC)
        finding = MagicMock()
        finding.id = uuid4()
        finding.finding_type = "taint_flow"
        finding.severity = "critical"
        finding.confidence = "high"
        finding.status = "open"
        finding.filename = "vuln.py"
        finding.line_number = 7
        finding.message = "Dangerous"
        finding.flow_data = {}
        finding.meta = {}
        finding.created_at = now
        finding.updated_at = now

        session.execute.side_effect = [
            # list_findings -> resolve scenario
            _scalar_one_or_none(scenario),
            # list_findings -> query findings
            _scalars_all([finding]),
        ]

        cid = uuid4()
        result = await export_findings_sarif(
            session,
            collection_id=cid,
            scenario_id=None,
        )

        assert result["collection_id"] == str(cid)
        assert result["finding_count"] == 1
        sarif = result["sarif"]
        assert sarif["version"] == "2.1.0"
        assert len(sarif["runs"]) == 1
        assert len(sarif["runs"][0]["results"]) == 1
        assert sarif["runs"][0]["results"][0]["level"] == "error"

    @pytest.mark.anyio
    async def test_exports_empty_sarif_when_no_findings(self) -> None:
        from contextmine_core.twin.ops import export_findings_sarif

        session = _make_mock_session()
        scenario = MagicMock()
        scenario.id = uuid4()

        session.execute.side_effect = [
            _scalar_one_or_none(scenario),
            _scalars_all([]),
        ]

        result = await export_findings_sarif(
            session,
            collection_id=uuid4(),
            scenario_id=None,
        )

        assert result["finding_count"] == 0
        assert result["sarif"]["runs"][0]["results"] == []


# ── _with_analysis_cache ─────────────────────────────────────────────────


class TestWithAnalysisCache:
    @pytest.mark.anyio
    async def test_returns_cached_result_on_hit(self) -> None:
        from contextmine_core.twin.ops import _with_analysis_cache

        session = _make_mock_session()
        scenario = MagicMock()
        scenario.id = uuid4()

        session.execute.side_effect = [
            # _resolve_status_scenario
            _scalar_one_or_none(scenario),
            # _current_analysis_cache_key
            _scalars_all([]),
            # _read_cached_analysis
            _scalar_one_or_none(MagicMock(payload={"cached": True})),
        ]

        compute = AsyncMock(return_value={"computed": True})

        result = await _with_analysis_cache(
            session,
            collection_id=uuid4(),
            scenario_id=None,
            tool_name="test",
            params={},
            projection_profile="p",
            cache_ttl_seconds=300,
            compute=compute,
        )

        assert result == {"cached": True}
        compute.assert_not_called()

    @pytest.mark.anyio
    async def test_calls_compute_on_cache_miss(self) -> None:
        from contextmine_core.twin.ops import _with_analysis_cache

        session = _make_mock_session()
        scenario = MagicMock()
        scenario.id = uuid4()

        session.execute.side_effect = [
            # _resolve_status_scenario
            _scalar_one_or_none(scenario),
            # _current_analysis_cache_key
            _scalars_all([]),
            # _read_cached_analysis -> miss
            _scalar_one_or_none(None),
            # _write_cached_analysis -> existing row lookup
            _scalar_one_or_none(None),
        ]

        compute = AsyncMock(return_value={"computed": True})

        result = await _with_analysis_cache(
            session,
            collection_id=uuid4(),
            scenario_id=None,
            tool_name="test",
            params={},
            projection_profile="p",
            cache_ttl_seconds=300,
            compute=compute,
        )

        assert result == {"computed": True}
        compute.assert_called_once()


# ── mark_previous_source_versions_stale ──────────────────────────────────


class TestMarkPreviousSourceVersionsStale:
    @pytest.mark.anyio
    async def test_marks_old_versions_stale(self) -> None:
        from contextmine_core.twin.ops import mark_previous_source_versions_stale

        session = _make_mock_session()
        old1 = MagicMock()
        old1.status = "ready"
        old2 = MagicMock()
        old2.status = "ready"
        session.execute.return_value = _scalars_all([old1, old2])

        result = await mark_previous_source_versions_stale(
            session,
            source_id=uuid4(),
            keep_source_version_id=uuid4(),
        )

        assert result == 2
        assert old1.status == "stale"
        assert old2.status == "stale"

    @pytest.mark.anyio
    async def test_returns_zero_when_no_old_versions(self) -> None:
        from contextmine_core.twin.ops import mark_previous_source_versions_stale

        session = _make_mock_session()
        session.execute.return_value = _scalars_all([])

        result = await mark_previous_source_versions_stale(
            session,
            source_id=uuid4(),
            keep_source_version_id=uuid4(),
        )

        assert result == 0


# ── compute_revision_key_for_source ──────────────────────────────────────


class TestComputeRevisionKeyForSource:
    @pytest.mark.anyio
    async def test_github_source_uses_cursor(self) -> None:
        from contextmine_core.twin.ops import compute_revision_key_for_source

        session = _make_mock_session()
        source = MagicMock()
        source.type = MagicMock()
        source.type.__eq__ = lambda self, other: True  # match SourceType.GITHUB
        source.cursor = "abc123"
        source.id = uuid4()

        # Patch SourceType to match
        with patch("contextmine_core.twin.ops.SourceType") as MockSourceType:
            MockSourceType.GITHUB = source.type
            result = await compute_revision_key_for_source(session, source)

        assert result == "abc123"

    @pytest.mark.anyio
    async def test_non_github_source_hashes_docs(self) -> None:
        from contextmine_core.twin.ops import compute_revision_key_for_source

        session = _make_mock_session()
        source = MagicMock()
        source.type = "web"
        source.cursor = None
        source.id = uuid4()

        doc1 = MagicMock()
        doc1.uri = "https://example.com/a"
        doc1.content_hash = "hash_a"
        doc2 = MagicMock()
        doc2.uri = "https://example.com/b"
        doc2.content_hash = "hash_b"

        mock_result = MagicMock()
        mock_result.all.return_value = [doc1, doc2]
        session.execute.return_value = mock_result

        with patch("contextmine_core.twin.ops.SourceType") as MockSourceType:
            MockSourceType.GITHUB = "github"
            result = await compute_revision_key_for_source(session, source)

        assert result.startswith("web:")
        assert len(result) == 4 + 64  # "web:" + sha256 hex

    @pytest.mark.anyio
    async def test_no_docs_returns_empty_key(self) -> None:
        from contextmine_core.twin.ops import compute_revision_key_for_source

        session = _make_mock_session()
        source = MagicMock()
        source.type = "web"
        source.cursor = None
        source.id = uuid4()

        mock_result = MagicMock()
        mock_result.all.return_value = []
        session.execute.return_value = mock_result

        with patch("contextmine_core.twin.ops.SourceType") as MockSourceType:
            MockSourceType.GITHUB = "github"
            result = await compute_revision_key_for_source(session, source)

        assert result == f"source:{source.id}:empty"


# ── list_collection_twin_events ──────────────────────────────────────────


class TestListCollectionTwinEvents:
    @pytest.mark.anyio
    async def test_returns_paginated_events(self) -> None:
        from contextmine_core.twin.ops import list_collection_twin_events

        session = _make_mock_session()
        now = datetime.now(UTC)

        event = MagicMock()
        event.id = uuid4()
        event.scenario_id = uuid4()
        event.source_id = uuid4()
        event.source_version_id = uuid4()
        event.event_type = "refresh_requested"
        event.status = "queued"
        event.payload = {"force": True}
        event.event_ts = now
        event.idempotency_key = "key1"
        event.error = None

        session.execute.side_effect = [
            _scalar_one(1),  # count
            _scalars_all([event]),  # rows
        ]

        cid = uuid4()
        result = await list_collection_twin_events(
            session,
            collection_id=cid,
            page=0,
            limit=50,
        )

        assert result["collection_id"] == str(cid)
        assert result["total"] == 1
        assert len(result["items"]) == 1
        assert result["items"][0]["event_type"] == "refresh_requested"
        assert result["page"] == 0
        assert result["limit"] == 50

    @pytest.mark.anyio
    async def test_empty_events(self) -> None:
        from contextmine_core.twin.ops import list_collection_twin_events

        session = _make_mock_session()
        session.execute.side_effect = [
            _scalar_one(0),  # count
            _scalars_all([]),  # rows
        ]

        result = await list_collection_twin_events(
            session,
            collection_id=uuid4(),
        )

        assert result["total"] == 0
        assert result["items"] == []


# ── get_collection_twin_diff ─────────────────────────────────────────────


class TestGetCollectionTwinDiff:
    @pytest.mark.anyio
    async def test_raises_when_scenario_not_found(self) -> None:
        from contextmine_core.twin.ops import get_collection_twin_diff

        session = _make_mock_session()
        session.execute.return_value = _scalar_one_or_none(None)

        with pytest.raises(ValueError, match="Scenario not found"):
            await get_collection_twin_diff(
                session,
                collection_id=uuid4(),
                scenario_id=uuid4(),
                from_version=1,
                to_version=2,
            )

    @pytest.mark.anyio
    async def test_raises_when_from_exceeds_to(self) -> None:
        from contextmine_core.twin.ops import get_collection_twin_diff

        session = _make_mock_session()
        scenario = MagicMock()
        scenario.id = uuid4()
        session.execute.return_value = _scalar_one_or_none(scenario)

        with pytest.raises(ValueError, match="from_version must be <= to_version"):
            await get_collection_twin_diff(
                session,
                collection_id=uuid4(),
                scenario_id=None,
                from_version=5,
                to_version=2,
            )

    @pytest.mark.anyio
    async def test_returns_diff_with_deltas(self) -> None:
        from contextmine_core.twin.ops import get_collection_twin_diff

        session = _make_mock_session()
        scenario = MagicMock()
        scenario.id = uuid4()

        event = MagicMock()
        event.payload = {
            "nodes_upserted": 3,
            "edges_upserted": 2,
            "nodes_deactivated": 1,
            "edges_deactivated": 0,
            "sample_node_keys": ["key1", "key2"],
        }

        sample_node = MagicMock()
        sample_node.id = uuid4()
        sample_node.natural_key = "key1"
        sample_node.name = "node1"
        sample_node.kind = "file"
        sample_node.is_active = True

        session.execute.side_effect = [
            _scalar_one_or_none(scenario),  # resolve scenario
            _scalars_all([event]),  # events
            _scalars_all([sample_node]),  # sample nodes
        ]

        result = await get_collection_twin_diff(
            session,
            collection_id=uuid4(),
            scenario_id=None,
            from_version=1,
            to_version=3,
        )

        assert result["delta"]["nodes_added"] == 3
        assert result["delta"]["edges_added"] == 2
        assert result["delta"]["nodes_removed"] == 1
        assert len(result["sample_nodes"]) == 1

    @pytest.mark.anyio
    async def test_filters_events_outside_version_range(self) -> None:
        from contextmine_core.twin.ops import get_collection_twin_diff

        session = _make_mock_session()
        scenario = MagicMock()
        scenario.id = uuid4()

        # Event with version outside range should be excluded
        event_in_range = MagicMock()
        event_in_range.payload = {
            "scenario_version": 2,
            "nodes_upserted": 5,
            "edges_upserted": 3,
            "nodes_deactivated": 0,
            "edges_deactivated": 0,
        }
        event_out_of_range = MagicMock()
        event_out_of_range.payload = {
            "scenario_version": 10,
            "nodes_upserted": 100,
            "edges_upserted": 50,
            "nodes_deactivated": 99,
            "edges_deactivated": 49,
        }

        session.execute.side_effect = [
            _scalar_one_or_none(scenario),
            _scalars_all([event_in_range, event_out_of_range]),
            _scalars_all([]),  # sample nodes
        ]

        result = await get_collection_twin_diff(
            session,
            collection_id=uuid4(),
            scenario_id=None,
            from_version=1,
            to_version=5,
        )

        # Only the in-range event should contribute
        assert result["delta"]["nodes_added"] == 5
        assert result["delta"]["edges_added"] == 3


# ── get_collection_twin_status ──────────────────────────────────────────


class TestGetCollectionTwinStatus:
    @pytest.mark.anyio
    async def test_returns_status_with_no_sources(self) -> None:
        from contextmine_core.twin.ops import get_collection_twin_status

        session = _make_mock_session()
        scenario = MagicMock()
        scenario.id = uuid4()
        scenario.version = 1

        session.execute.side_effect = [
            # _resolve_status_scenario
            _scalar_one_or_none(scenario),
            # select(Source)
            _scalars_all([]),
            # select(TwinEvent)
            _scalars_all([]),
        ]

        cid = uuid4()
        result = await get_collection_twin_status(session, collection_id=cid)

        assert result["collection_id"] == str(cid)
        assert result["freshness"] == "failed"  # no sources -> failed
        assert result["sources"] == []

    @pytest.mark.anyio
    async def test_returns_ready_when_all_sources_ready(self) -> None:
        from contextmine_core.twin.ops import get_collection_twin_status

        session = _make_mock_session()
        scenario = MagicMock()
        scenario.id = uuid4()
        scenario.version = 2

        source = MagicMock()
        source.id = uuid4()

        now = datetime.now(UTC)
        sv = MagicMock()
        sv.id = uuid4()
        sv.status = "ready"
        sv.revision_key = "rev1"
        sv.finished_at = now
        sv.started_at = now
        sv.stats = {}

        session.execute.side_effect = [
            _scalar_one_or_none(scenario),
            _scalars_all([source]),
            # latest source version for source
            _scalar_one_or_none(sv),
            # last error event
            _scalar_one_or_none(None),
            # timeline events
            _scalars_all([]),
        ]

        result = await get_collection_twin_status(session, collection_id=uuid4())

        assert result["freshness"] == "ready"
        assert len(result["sources"]) == 1
        assert result["sources"][0]["status"] == "ready"

    @pytest.mark.anyio
    async def test_returns_degraded_when_failed_sources(self) -> None:
        from contextmine_core.twin.ops import get_collection_twin_status

        session = _make_mock_session()
        scenario = MagicMock()
        scenario.id = uuid4()
        scenario.version = 1

        source = MagicMock()
        source.id = uuid4()

        sv = MagicMock()
        sv.id = uuid4()
        sv.status = "failed"
        sv.revision_key = "rev1"
        sv.finished_at = None
        sv.started_at = datetime.now(UTC)
        sv.stats = {}

        err_event = MagicMock()
        err_event.error = "timeout"

        session.execute.side_effect = [
            _scalar_one_or_none(scenario),
            _scalars_all([source]),
            _scalar_one_or_none(sv),
            _scalar_one_or_none(err_event),
            _scalars_all([]),
        ]

        result = await get_collection_twin_status(session, collection_id=uuid4())

        assert result["freshness"] == "degraded"
        assert result["sources"][0]["last_error"] == "timeout"

    @pytest.mark.anyio
    async def test_aggregates_scip_stats(self) -> None:
        from contextmine_core.twin.ops import get_collection_twin_status

        session = _make_mock_session()
        scenario = MagicMock()
        scenario.id = uuid4()
        scenario.version = 1

        source = MagicMock()
        source.id = uuid4()

        now = datetime.now(UTC)
        sv = MagicMock()
        sv.id = uuid4()
        sv.status = "ready"
        sv.revision_key = "rev1"
        sv.finished_at = now
        sv.started_at = now
        sv.stats = {
            "scip_projects_detected": 3,
            "scip_projects_indexed": 3,
            "scip_projects_failed": 0,
            "scip_projects_by_language": {"python": 2, "typescript": 1},
            "metrics_requested_files": 10,
            "metrics_mapped_files": 10,
        }

        session.execute.side_effect = [
            _scalar_one_or_none(scenario),
            _scalars_all([source]),
            _scalar_one_or_none(sv),
            _scalar_one_or_none(None),
            _scalars_all([]),
        ]

        result = await get_collection_twin_status(session, collection_id=uuid4())

        assert result["scip_status"] == "ready"
        assert result["scip_projects_by_language"]["python"] == 2
        assert result["scip_projects_by_language"]["typescript"] == 1
        assert result["metrics_gate"]["status"] == "pass"

    @pytest.mark.anyio
    async def test_behavioral_layers_status(self) -> None:
        from contextmine_core.twin.ops import get_collection_twin_status

        session = _make_mock_session()
        scenario = MagicMock()
        scenario.id = uuid4()
        scenario.version = 1

        source = MagicMock()
        source.id = uuid4()

        now = datetime.now(UTC)
        sv = MagicMock()
        sv.id = uuid4()
        sv.status = "ready"
        sv.revision_key = "rev1"
        sv.finished_at = now
        sv.started_at = now
        sv.stats = {
            "behavioral_layers_status": "ready",
            "last_behavioral_materialized_at": "2024-06-01T12:00:00Z",
            "scip_projects_detected": 1,
            "scip_projects_indexed": 1,
            "scip_projects_failed": 0,
            "metrics_requested_files": 5,
            "metrics_mapped_files": 5,
        }

        session.execute.side_effect = [
            _scalar_one_or_none(scenario),
            _scalars_all([source]),
            _scalar_one_or_none(sv),
            _scalar_one_or_none(None),
            _scalars_all([]),
        ]

        result = await get_collection_twin_status(session, collection_id=uuid4())

        assert result["behavioral_layers_status"] == "ready"
        assert result["last_behavioral_materialized_at"] is not None


# ── trigger_collection_refresh ──────────────────────────────────────────


class TestTriggerCollectionRefresh:
    @pytest.mark.anyio
    async def test_creates_new_source_versions(self) -> None:
        from contextmine_core.twin.ops import trigger_collection_refresh

        session = _make_mock_session()
        source = MagicMock()
        source.id = uuid4()
        source.type = "web"
        source.cursor = None

        doc = MagicMock()
        doc.uri = "https://example.com"
        doc.content_hash = "abc123"

        # Fresh source version
        new_sv = MagicMock()
        new_sv.id = uuid4()
        new_sv.status = "queued"
        new_sv.revision_key = "web:abc"

        # Existing idempotent event
        existing_event = MagicMock()
        existing_event.id = uuid4()

        session.execute.side_effect = [
            # select(Source)
            _scalars_all([source]),
            # compute_revision_key_for_source -> select docs
            MagicMock(all=lambda: [doc]),
            # latest source version
            _scalar_one_or_none(None),
            # get_or_create_source_version -> lookup
            _scalar_one_or_none(None),
            # record_twin_event -> lookup
            _scalar_one_or_none(None),
        ]

        cid = uuid4()
        result = await trigger_collection_refresh(
            session,
            collection_id=cid,
        )

        assert result["collection_id"] == str(cid)
        assert result["created"] >= 1
        assert len(result["items"]) >= 1

    @pytest.mark.anyio
    async def test_skips_up_to_date_sources(self) -> None:
        from contextmine_core.twin.ops import trigger_collection_refresh

        session = _make_mock_session()
        source = MagicMock()
        source.id = uuid4()
        source.type = "web"
        source.cursor = None

        doc = MagicMock()
        doc.uri = "https://example.com"
        doc.content_hash = "abc123"

        # Latest version matches current revision
        latest_sv = MagicMock()
        latest_sv.id = uuid4()
        latest_sv.status = "ready"
        latest_sv.extractor_version = "scip-kg-v1"

        session.execute.side_effect = [
            _scalars_all([source]),
            MagicMock(all=lambda: [doc]),
            _scalar_one_or_none(latest_sv),
        ]

        # Make the revision key match
        import hashlib

        digest_input = f"{doc.uri}:{doc.content_hash}"
        expected_key = f"web:{hashlib.sha256(digest_input.encode()).hexdigest()}"
        latest_sv.revision_key = expected_key

        with patch("contextmine_core.twin.ops.SourceType") as MockSourceType:
            MockSourceType.GITHUB = "github"
            result = await trigger_collection_refresh(
                session,
                collection_id=uuid4(),
            )

        assert result["skipped"] >= 1

    @pytest.mark.anyio
    async def test_force_overrides_skip(self) -> None:
        from contextmine_core.twin.ops import trigger_collection_refresh

        session = _make_mock_session()
        source = MagicMock()
        source.id = uuid4()
        source.type = "web"
        source.cursor = None

        doc = MagicMock()
        doc.uri = "https://example.com"
        doc.content_hash = "abc123"

        latest_sv = MagicMock()
        latest_sv.id = uuid4()
        latest_sv.status = "ready"
        latest_sv.extractor_version = "scip-kg-v1"

        session.execute.side_effect = [
            _scalars_all([source]),
            MagicMock(all=lambda: [doc]),
            _scalar_one_or_none(latest_sv),
            # get_or_create_source_version
            _scalar_one_or_none(None),
            # record_twin_event
            _scalar_one_or_none(None),
        ]

        import hashlib

        digest_input = f"{doc.uri}:{doc.content_hash}"
        expected_key = f"web:{hashlib.sha256(digest_input.encode()).hexdigest()}"
        latest_sv.revision_key = expected_key

        with patch("contextmine_core.twin.ops.SourceType") as MockSourceType:
            MockSourceType.GITHUB = "github"
            result = await trigger_collection_refresh(
                session,
                collection_id=uuid4(),
                force=True,
            )

        assert result["created"] >= 1
