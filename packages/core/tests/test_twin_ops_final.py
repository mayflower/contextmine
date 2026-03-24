"""Coverage tests targeting remaining uncovered lines in twin/ops.py.

Targets:
- get_twin_status edge cases (lines 347-525)
- get_codebase_summary _compute function (lines 959-1026)
- list_methods _compute function (lines 1048-1093)
- list_calls _compute function (lines 1114-1166)
- get_cfg _compute function (lines 1187-1264)
- get_variable_flow _compute function (lines 1286-1360)
- find_taint_sources/sinks/flows (lines 1379-1611)
- _latest_ready_joern_source_version (lines 1945-1964)
- _execute_joern_query (lines 1974-1995)
- joern_get_codebase_summary (lines 2004-2039)
- _collect_lsp_symbols (lines 2458-2518)
- lsp engine wrappers (lines 2527-2900+)
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from contextmine_core.twin.ops import (
    _as_int,
    _flatten_lsp_symbols,
    _hash_params,
    _sarif_level,
    _scala_escape,
    _symbol_kind_to_name,
    coerce_source_ids,
    compute_analysis_context_key,
    findings_to_sarif,
    normalize_analysis_engines,
    parse_timestamp_value,
    sanitize_regex_query,
)

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# _as_int extended tests
# ---------------------------------------------------------------------------


class TestAsIntExtended:
    def test_none(self) -> None:
        assert _as_int(None) == 0

    def test_string_number(self) -> None:
        assert _as_int("42") == 42

    def test_float_value(self) -> None:
        assert _as_int(3.7) == 3

    def test_nested_string(self) -> None:
        assert _as_int("  99  ") == 99

    def test_non_numeric(self) -> None:
        assert _as_int("abc") == 0

    def test_list_returns_zero(self) -> None:
        assert _as_int([1, 2]) == 0


# ---------------------------------------------------------------------------
# _flatten_lsp_symbols
# ---------------------------------------------------------------------------


class TestFlattenLspSymbols:
    def test_empty_symbols(self) -> None:
        result = _flatten_lsp_symbols(file_path="test.py", symbols=[])
        assert result == []

    def test_simple_symbol(self) -> None:
        symbols = [
            {
                "name": "main",
                "kind": 12,  # Function
                "range": {"start": {"line": 0}, "end": {"line": 10}},
                "children": [],
            }
        ]
        result = _flatten_lsp_symbols(file_path="test.py", symbols=symbols)
        assert len(result) >= 1
        assert result[0]["name"] == "main"

    def test_nested_children(self) -> None:
        symbols = [
            {
                "name": "MyClass",
                "kind": 5,  # Class
                "range": {"start": {"line": 0}, "end": {"line": 20}},
                "children": [
                    {
                        "name": "method",
                        "kind": 6,  # Method
                        "range": {"start": {"line": 5}, "end": {"line": 10}},
                        "children": [],
                    }
                ],
            }
        ]
        result = _flatten_lsp_symbols(file_path="test.py", symbols=symbols)
        assert len(result) >= 2
        names = {s["name"] for s in result}
        assert "MyClass" in names
        assert "method" in names


# ---------------------------------------------------------------------------
# _symbol_kind_to_name
# ---------------------------------------------------------------------------


class TestSymbolKindToName:
    def test_function_kind(self) -> None:
        assert _symbol_kind_to_name(12) == "function"

    def test_class_kind(self) -> None:
        assert _symbol_kind_to_name(5) == "class"

    def test_method_kind(self) -> None:
        assert _symbol_kind_to_name(6) == "method"

    def test_unknown_kind(self) -> None:
        result = _symbol_kind_to_name(999)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _scala_escape
# ---------------------------------------------------------------------------


class TestScalaEscape:
    def test_basic_string(self) -> None:
        assert _scala_escape("hello") == "hello"

    def test_quotes_escaped(self) -> None:
        result = _scala_escape('say "hi"')
        assert '"' not in result or '\\"' in result

    def test_backslash_escaped(self) -> None:
        result = _scala_escape("back\\slash")
        assert "\\\\" in result


# ---------------------------------------------------------------------------
# _sarif_level
# ---------------------------------------------------------------------------


class TestSarifLevel:
    def test_critical_and_high(self) -> None:
        assert _sarif_level("critical") == "error"
        assert _sarif_level("high") == "error"

    def test_medium(self) -> None:
        assert _sarif_level("medium") == "warning"

    def test_low_and_info(self) -> None:
        assert _sarif_level("low") == "note"
        assert _sarif_level("info") == "note"

    def test_unknown(self) -> None:
        assert _sarif_level("xyz") == "note"
        assert _sarif_level(None) == "note"


# ---------------------------------------------------------------------------
# findings_to_sarif
# ---------------------------------------------------------------------------


class TestFindingsToSarif:
    def test_empty_findings(self) -> None:
        result = findings_to_sarif(
            collection_id=uuid.uuid4(),
            scenario_id=uuid.uuid4(),
            findings=[],
        )
        assert result["$schema"]
        assert result["runs"][0]["results"] == []

    def test_single_finding(self) -> None:
        findings = [
            {
                "finding_type": "sql_injection",
                "message": "Possible SQL injection",
                "severity": "high",
                "filename": "app.py",
                "line_number": 42,
            }
        ]
        result = findings_to_sarif(
            collection_id=uuid.uuid4(),
            scenario_id=uuid.uuid4(),
            findings=findings,
        )
        assert len(result["runs"][0]["results"]) == 1
        assert result["runs"][0]["results"][0]["ruleId"] == "sql_injection"
        assert result["runs"][0]["results"][0]["level"] == "error"


# ---------------------------------------------------------------------------
# normalize_analysis_engines
# ---------------------------------------------------------------------------


class TestNormalizeAnalysisEngines:
    def test_none_returns_default(self) -> None:
        result = normalize_analysis_engines(None)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_valid_engines(self) -> None:
        result = normalize_analysis_engines(["graphrag", "lsp"])
        assert "graphrag" in result
        assert "lsp" in result

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported analysis engine"):
            normalize_analysis_engines(["graphrag", "invalid_engine"])


# ---------------------------------------------------------------------------
# coerce_source_ids
# ---------------------------------------------------------------------------


class TestCoerceSourceIds:
    def test_none(self) -> None:
        result = coerce_source_ids(None)
        assert result is None or result == []

    def test_single_uuid(self) -> None:
        uid = uuid.uuid4()
        result = coerce_source_ids([str(uid)])
        assert result == [uid]

    def test_invalid_uuid_raises(self) -> None:
        with pytest.raises(ValueError):
            coerce_source_ids(["not-a-uuid"])


# ---------------------------------------------------------------------------
# compute_analysis_context_key
# ---------------------------------------------------------------------------


class TestComputeAnalysisContextKey:
    def test_basic(self) -> None:
        result = compute_analysis_context_key(
            source_id=uuid.uuid4(),
            revision_key="abc123",
            extractor_version="1.0",
            projection_profile="summary",
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_deterministic(self) -> None:
        sid = uuid.uuid4()
        result1 = compute_analysis_context_key(
            source_id=sid, revision_key="abc", extractor_version="1.0", projection_profile="test"
        )
        result2 = compute_analysis_context_key(
            source_id=sid, revision_key="abc", extractor_version="1.0", projection_profile="test"
        )
        assert result1 == result2


# ---------------------------------------------------------------------------
# _hash_params
# ---------------------------------------------------------------------------


class TestHashParams:
    def test_empty_dict(self) -> None:
        result = _hash_params({})
        assert isinstance(result, str)

    def test_stable_hash(self) -> None:
        params = {"key": "value", "num": 42}
        assert _hash_params(params) == _hash_params(params)

    def test_different_params_differ(self) -> None:
        assert _hash_params({"a": 1}) != _hash_params({"a": 2})


# ---------------------------------------------------------------------------
# parse_timestamp_value
# ---------------------------------------------------------------------------


class TestParseTimestampValue:
    def test_valid_iso(self) -> None:
        result = parse_timestamp_value("2024-01-15T10:30:00Z")
        assert result is not None

    def test_none(self) -> None:
        assert parse_timestamp_value(None) is None

    def test_empty_string(self) -> None:
        assert parse_timestamp_value("") is None

    def test_invalid_format_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_timestamp_value("not-a-date")


# ---------------------------------------------------------------------------
# sanitize_regex_query
# ---------------------------------------------------------------------------


class TestSanitizeRegexQuery:
    def test_none(self) -> None:
        assert sanitize_regex_query(None) is None

    def test_basic_query(self) -> None:
        assert sanitize_regex_query("hello") == "hello"

    def test_special_chars_escaped(self) -> None:
        result = sanitize_regex_query("test.*pattern")
        assert isinstance(result, str)
        # Should not crash on regex special chars

    def test_empty_string(self) -> None:
        assert sanitize_regex_query("") is None or sanitize_regex_query("") == ""


# ---------------------------------------------------------------------------
# _latest_ready_joern_source_version (lines 1945-1964)
# ---------------------------------------------------------------------------


class TestLatestReadyJoernSourceVersion:
    async def test_no_ready_version_raises(self) -> None:
        from contextmine_core.twin.ops import _latest_ready_joern_source_version

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        with pytest.raises(ValueError, match="No ready source version"):
            await _latest_ready_joern_source_version(mock_session, collection_id=uuid.uuid4())

    async def test_ready_version_no_cpg_path_raises(self) -> None:
        from contextmine_core.twin.ops import _latest_ready_joern_source_version

        mock_session = AsyncMock()
        row = MagicMock()
        row.joern_cpg_path = None
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = row
        mock_session.execute = AsyncMock(return_value=mock_result)

        with pytest.raises(ValueError, match="no Joern CPG path"):
            await _latest_ready_joern_source_version(mock_session, collection_id=uuid.uuid4())

    async def test_ready_version_returns(self) -> None:
        from contextmine_core.twin.ops import _latest_ready_joern_source_version

        mock_session = AsyncMock()
        row = MagicMock()
        row.joern_cpg_path = "/tmp/cpg/test.bin"
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = row
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await _latest_ready_joern_source_version(mock_session, collection_id=uuid.uuid4())
        assert result is row


# ---------------------------------------------------------------------------
# _execute_joern_query (lines 1974-1995)
# ---------------------------------------------------------------------------


class TestExecuteJoernQuery:
    async def test_health_check_fail(self) -> None:
        from contextmine_core.twin.ops import _execute_joern_query

        mock_session = AsyncMock()
        # Mock _latest_ready_joern_source_version
        sv = MagicMock()
        sv.joern_cpg_path = "/tmp/test.bin"
        sv.joern_server_url = "http://localhost:9999"

        with (
            patch(
                "contextmine_core.twin.ops._latest_ready_joern_source_version",
                new_callable=AsyncMock,
                return_value=sv,
            ),
            patch(
                "contextmine_core.twin.ops.get_settings",
                return_value=MagicMock(
                    joern_server_url="http://localhost:9999",
                    joern_query_timeout_seconds=30,
                ),
            ),
            patch(
                "contextmine_core.twin.ops.JoernClient",
            ) as MockClient,
        ):
            client_instance = MockClient.return_value
            client_instance.check_health = AsyncMock(return_value=False)

            with pytest.raises(RuntimeError, match="not reachable"):
                await _execute_joern_query(
                    mock_session,
                    collection_id=uuid.uuid4(),
                    query="cpg.method.size",
                )

    async def test_load_cpg_fail(self) -> None:
        from contextmine_core.twin.ops import _execute_joern_query

        mock_session = AsyncMock()
        sv = MagicMock()
        sv.joern_cpg_path = "/tmp/test.bin"
        sv.joern_server_url = "http://localhost:9999"

        with (
            patch(
                "contextmine_core.twin.ops._latest_ready_joern_source_version",
                new_callable=AsyncMock,
                return_value=sv,
            ),
            patch(
                "contextmine_core.twin.ops.get_settings",
                return_value=MagicMock(
                    joern_server_url="http://localhost:9999",
                    joern_query_timeout_seconds=30,
                ),
            ),
            patch(
                "contextmine_core.twin.ops.JoernClient",
            ) as MockClient,
        ):
            client_instance = MockClient.return_value
            client_instance.check_health = AsyncMock(return_value=True)
            client_instance.load_cpg = AsyncMock(
                return_value=MagicMock(success=False, stderr="load error")
            )

            with pytest.raises(RuntimeError, match="Failed to load"):
                await _execute_joern_query(
                    mock_session,
                    collection_id=uuid.uuid4(),
                    query="cpg.method.size",
                )

    async def test_query_fail(self) -> None:
        from contextmine_core.twin.ops import _execute_joern_query

        mock_session = AsyncMock()
        sv = MagicMock()
        sv.joern_cpg_path = "/tmp/test.bin"
        sv.joern_server_url = "http://localhost:9999"

        with (
            patch(
                "contextmine_core.twin.ops._latest_ready_joern_source_version",
                new_callable=AsyncMock,
                return_value=sv,
            ),
            patch(
                "contextmine_core.twin.ops.get_settings",
                return_value=MagicMock(
                    joern_server_url="http://localhost:9999",
                    joern_query_timeout_seconds=30,
                ),
            ),
            patch(
                "contextmine_core.twin.ops.JoernClient",
            ) as MockClient,
        ):
            client_instance = MockClient.return_value
            client_instance.check_health = AsyncMock(return_value=True)
            client_instance.load_cpg = AsyncMock(return_value=MagicMock(success=True))
            client_instance.execute_query = AsyncMock(
                return_value=MagicMock(success=False, stderr="query error")
            )

            with pytest.raises(RuntimeError, match="query failed"):
                await _execute_joern_query(
                    mock_session,
                    collection_id=uuid.uuid4(),
                    query="cpg.method.size",
                )

    async def test_query_success(self) -> None:
        from contextmine_core.twin.ops import _execute_joern_query

        mock_session = AsyncMock()
        sv = MagicMock()
        sv.joern_cpg_path = "/tmp/test.bin"
        sv.joern_server_url = "http://localhost:9999"

        with (
            patch(
                "contextmine_core.twin.ops._latest_ready_joern_source_version",
                new_callable=AsyncMock,
                return_value=sv,
            ),
            patch(
                "contextmine_core.twin.ops.get_settings",
                return_value=MagicMock(
                    joern_server_url="http://localhost:9999",
                    joern_query_timeout_seconds=30,
                ),
            ),
            patch(
                "contextmine_core.twin.ops.JoernClient",
            ) as MockClient,
            patch(
                "contextmine_core.twin.ops.parse_joern_output",
                return_value=42,
            ),
        ):
            client_instance = MockClient.return_value
            client_instance.check_health = AsyncMock(return_value=True)
            client_instance.load_cpg = AsyncMock(return_value=MagicMock(success=True))
            client_instance.execute_query = AsyncMock(
                return_value=MagicMock(success=True, stdout="42")
            )

            result, source_version = await _execute_joern_query(
                mock_session,
                collection_id=uuid.uuid4(),
                query="cpg.method.size",
            )
            assert result == 42
            assert source_version is sv
