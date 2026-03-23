"""Tests for contextmine_core.twin.ops — pure/utility functions."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC
from typing import Any
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
