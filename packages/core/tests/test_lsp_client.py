"""Tests for LSP client: dataclass helpers, parse methods, MockLspClient."""

from pathlib import Path
from typing import Any

import pytest
from contextmine_core.lsp.client import (
    SEVERITY_MAP,
    Diagnostic,
    Location,
    LspClient,
    MockLspClient,
    SymbolInfo,
)
from contextmine_core.lsp.exceptions import LspServerError

# ---------------------------------------------------------------------------
# Dataclass to_dict helpers
# ---------------------------------------------------------------------------


class TestLocationDataclass:
    def test_to_dict(self):
        loc = Location(
            file_path="/src/main.py",
            start_line=10,
            start_column=4,
            end_line=10,
            end_column=20,
        )
        d = loc.to_dict()
        assert d == {
            "file_path": "/src/main.py",
            "start_line": 10,
            "start_column": 4,
            "end_line": 10,
            "end_column": 20,
        }

    def test_equality(self):
        a = Location("a.py", 1, 0, 1, 5)
        b = Location("a.py", 1, 0, 1, 5)
        assert a == b


class TestSymbolInfoDataclass:
    def test_to_dict(self):
        info = SymbolInfo(
            name="my_func",
            kind="function",
            signature="def my_func(x: int) -> str",
            documentation="Does things.",
        )
        d = info.to_dict()
        assert d["name"] == "my_func"
        assert d["kind"] == "function"
        assert d["signature"] == "def my_func(x: int) -> str"
        assert d["documentation"] == "Does things."

    def test_to_dict_with_none_fields(self):
        info = SymbolInfo(name="x", kind="variable", signature=None, documentation=None)
        d = info.to_dict()
        assert d["signature"] is None
        assert d["documentation"] is None


class TestDiagnosticDataclass:
    def test_to_dict(self):
        diag = Diagnostic(
            file_path="/src/main.py",
            line=5,
            column=10,
            end_line=5,
            end_column=15,
            message="undefined name 'foo'",
            severity="error",
            code="E0602",
            source="pylint",
        )
        d = diag.to_dict()
        assert d["file_path"] == "/src/main.py"
        assert d["message"] == "undefined name 'foo'"
        assert d["severity"] == "error"
        assert d["code"] == "E0602"
        assert d["source"] == "pylint"

    def test_to_dict_with_none_code_source(self):
        diag = Diagnostic(
            file_path="a.py",
            line=1,
            column=0,
            end_line=1,
            end_column=5,
            message="msg",
            severity="warning",
            code=None,
            source=None,
        )
        d = diag.to_dict()
        assert d["code"] is None
        assert d["source"] is None


class TestSeverityMap:
    def test_known_severities(self):
        assert SEVERITY_MAP[1] == "error"
        assert SEVERITY_MAP[2] == "warning"
        assert SEVERITY_MAP[3] == "info"
        assert SEVERITY_MAP[4] == "hint"

    def test_four_entries(self):
        assert len(SEVERITY_MAP) == 4


# ---------------------------------------------------------------------------
# LspClient._parse_locations (tested via a client with mock server)
# ---------------------------------------------------------------------------


class TestLspClientParseLocations:
    """Test the _parse_locations helper of LspClient directly."""

    def _make_client(self) -> LspClient:
        """Create an LspClient with a dummy server."""
        from unittest.mock import MagicMock

        server = MagicMock()
        return LspClient(server=server, project_root=Path("/project"))

    def test_parse_empty_result(self):
        client = self._make_client()
        assert client._parse_locations(None) == []
        assert client._parse_locations([]) == []

    def test_parse_location_format(self):
        client = self._make_client()
        raw = [
            {
                "uri": "file:///project/src/main.py",
                "range": {
                    "start": {"line": 9, "character": 0},
                    "end": {"line": 14, "character": 1},
                },
            }
        ]
        locations = client._parse_locations(raw)
        assert len(locations) == 1
        loc = locations[0]
        assert loc.file_path == "/project/src/main.py"
        assert loc.start_line == 10  # 0-indexed -> 1-indexed
        assert loc.start_column == 0
        assert loc.end_line == 15
        assert loc.end_column == 1

    def test_parse_location_link_format(self):
        client = self._make_client()
        raw = [
            {
                "targetUri": "file:///project/lib/utils.py",
                "targetRange": {
                    "start": {"line": 4, "character": 2},
                    "end": {"line": 10, "character": 0},
                },
            }
        ]
        locations = client._parse_locations(raw)
        assert len(locations) == 1
        loc = locations[0]
        assert loc.file_path == "/project/lib/utils.py"
        assert loc.start_line == 5

    def test_parse_location_link_with_selection_range_fallback(self):
        client = self._make_client()
        raw = [
            {
                "targetUri": "file:///project/lib/utils.py",
                "targetSelectionRange": {
                    "start": {"line": 7, "character": 0},
                    "end": {"line": 7, "character": 10},
                },
            }
        ]
        locations = client._parse_locations(raw)
        assert len(locations) == 1
        assert locations[0].start_line == 8

    def test_parse_skips_items_without_uri(self):
        client = self._make_client()
        raw: list[dict[str, Any]] = [{"someOtherKey": "value"}]
        locations = client._parse_locations(raw)
        assert locations == []

    def test_parse_strips_file_prefix(self):
        client = self._make_client()
        raw = [
            {
                "uri": "file:///absolute/path.py",
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 0, "character": 5},
                },
            }
        ]
        locations = client._parse_locations(raw)
        assert locations[0].file_path == "/absolute/path.py"

    def test_parse_handles_malformed_item_gracefully(self):
        """Items that raise during parsing are skipped, not crash."""
        client = self._make_client()
        raw: list[dict[str, Any]] = [
            {
                "uri": "file:///ok.py",
                "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 5}},
            },
            # Malformed: uri present but range is bizarre
            {"uri": 123},
        ]
        # Should not raise; just skip the bad item
        locations = client._parse_locations(raw)
        assert len(locations) >= 1


# ---------------------------------------------------------------------------
# LspClient._parse_hover
# ---------------------------------------------------------------------------


class TestLspClientParseHover:
    def _make_client(self) -> LspClient:
        from unittest.mock import MagicMock

        server = MagicMock()
        return LspClient(server=server, project_root=Path("/project"))

    def test_parse_hover_none(self):
        client = self._make_client()
        assert client._parse_hover(None) is None

    def test_parse_hover_empty_contents(self):
        client = self._make_client()
        assert client._parse_hover({"contents": ""}) is None
        assert client._parse_hover({"contents": None}) is None
        assert client._parse_hover({}) is None

    def test_parse_hover_string_contents(self):
        client = self._make_client()
        result = client._parse_hover({"contents": "def my_func(x: int) -> str"})
        assert result is not None
        assert result.name == "my_func"
        assert result.kind == "function"

    def test_parse_hover_dict_contents(self):
        client = self._make_client()
        result = client._parse_hover(
            {"contents": {"value": "class MyClass:\n    '''Docstring.'''"}}
        )
        assert result is not None
        assert result.kind == "class"

    def test_parse_hover_list_contents(self):
        client = self._make_client()
        result = client._parse_hover({"contents": [{"value": "def func()"}, "Some documentation"]})
        assert result is not None
        assert result.documentation == "Some documentation"

    def test_parse_hover_detects_function(self):
        client = self._make_client()
        result = client._parse_hover({"contents": "def process_data(items: list)"})
        assert result is not None
        assert result.kind == "function"
        assert result.name == "process_data"

    def test_parse_hover_detects_class(self):
        client = self._make_client()
        result = client._parse_hover({"contents": "class MyHandler"})
        assert result is not None
        assert result.kind == "class"
        assert result.name == "MyHandler"

    def test_parse_hover_detects_variable_with_colon(self):
        client = self._make_client()
        result = client._parse_hover({"contents": "count: int"})
        assert result is not None
        assert result.kind == "variable"
        assert result.name == "count"

    def test_parse_hover_method_keyword(self):
        client = self._make_client()
        result = client._parse_hover({"contents": "method process()"})
        assert result is not None
        assert result.kind == "method"


# ---------------------------------------------------------------------------
# LspClient._resolve_path
# ---------------------------------------------------------------------------


class TestLspClientResolvePath:
    def _make_client(self) -> LspClient:
        from unittest.mock import MagicMock

        server = MagicMock()
        return LspClient(server=server, project_root=Path("/project"))

    def test_relative_path_resolved_against_project_root(self):
        client = self._make_client()
        resolved = client._resolve_path("src/main.py")
        assert resolved == str(Path("/project/src/main.py").resolve())

    def test_absolute_path_unchanged(self):
        client = self._make_client()
        resolved = client._resolve_path("/other/file.py")
        assert resolved == str(Path("/other/file.py").resolve())


# ---------------------------------------------------------------------------
# LspClient async methods raise LspServerError on failure
# ---------------------------------------------------------------------------


class TestLspClientAsyncMethods:
    @pytest.mark.anyio
    async def test_get_definition_raises_on_server_error(self):
        from unittest.mock import AsyncMock, MagicMock

        server = MagicMock()
        server.request_definition = AsyncMock(side_effect=RuntimeError("boom"))
        client = LspClient(server=server, project_root=Path("/project"))
        client._started = True

        with pytest.raises(LspServerError, match="Definition request failed"):
            await client.get_definition("test.py", 1, 0)

    @pytest.mark.anyio
    async def test_get_references_raises_on_server_error(self):
        from unittest.mock import AsyncMock, MagicMock

        server = MagicMock()
        server.request_references = AsyncMock(side_effect=RuntimeError("boom"))
        client = LspClient(server=server, project_root=Path("/project"))
        client._started = True

        with pytest.raises(LspServerError, match="References request failed"):
            await client.get_references("test.py", 1, 0)

    @pytest.mark.anyio
    async def test_get_hover_raises_on_server_error(self):
        from unittest.mock import AsyncMock, MagicMock

        server = MagicMock()
        server.request_hover = AsyncMock(side_effect=RuntimeError("boom"))
        client = LspClient(server=server, project_root=Path("/project"))
        client._started = True

        with pytest.raises(LspServerError, match="Hover request failed"):
            await client.get_hover("test.py", 1, 0)

    @pytest.mark.anyio
    async def test_get_document_symbols_raises_on_server_error(self):
        from unittest.mock import AsyncMock, MagicMock

        server = MagicMock()
        server.request_document_symbols = AsyncMock(side_effect=RuntimeError("boom"))
        client = LspClient(server=server, project_root=Path("/project"))
        client._started = True

        with pytest.raises(LspServerError, match="Document symbols request failed"):
            await client.get_document_symbols("test.py")

    @pytest.mark.anyio
    async def test_get_definition_success(self):
        from unittest.mock import AsyncMock, MagicMock

        server = MagicMock()
        server.request_definition = AsyncMock(
            return_value=[
                {
                    "uri": "file:///project/target.py",
                    "range": {
                        "start": {"line": 5, "character": 0},
                        "end": {"line": 10, "character": 1},
                    },
                }
            ]
        )
        client = LspClient(server=server, project_root=Path("/project"))
        client._started = True

        locations = await client.get_definition("test.py", 3, 4)
        assert len(locations) == 1
        assert locations[0].start_line == 6  # 0->1 indexed

    @pytest.mark.anyio
    async def test_get_document_symbols_empty(self):
        from unittest.mock import AsyncMock, MagicMock

        server = MagicMock()
        server.request_document_symbols = AsyncMock(return_value=None)
        client = LspClient(server=server, project_root=Path("/project"))
        client._started = True

        symbols = await client.get_document_symbols("test.py")
        assert symbols == []


# ---------------------------------------------------------------------------
# LspClient start / stop lifecycle
# ---------------------------------------------------------------------------


class TestLspClientLifecycle:
    @pytest.mark.anyio
    async def test_start_enters_context_manager(self):
        from unittest.mock import AsyncMock, MagicMock

        cm = AsyncMock()
        server = MagicMock()
        server.start_server.return_value = cm

        client = LspClient(server=server, project_root=Path("/project"))
        assert not client._started

        await client.start()
        assert client._started
        cm.__aenter__.assert_awaited_once()

    @pytest.mark.anyio
    async def test_stop_exits_context_manager(self):
        from unittest.mock import AsyncMock, MagicMock

        cm = AsyncMock()
        server = MagicMock()
        server.start_server.return_value = cm

        client = LspClient(server=server, project_root=Path("/project"))
        await client.start()
        await client.stop()

        assert not client._started
        cm.__aexit__.assert_awaited_once()

    @pytest.mark.anyio
    async def test_stop_when_not_started_is_noop(self):
        from unittest.mock import MagicMock

        server = MagicMock()
        client = LspClient(server=server, project_root=Path("/project"))
        # Should not raise
        await client.stop()
        assert not client._started

    @pytest.mark.anyio
    async def test_stop_handles_aexit_error(self):
        from unittest.mock import AsyncMock, MagicMock

        cm = AsyncMock()
        cm.__aexit__.side_effect = RuntimeError("stop error")
        server = MagicMock()
        server.start_server.return_value = cm

        client = LspClient(server=server, project_root=Path("/project"))
        await client.start()
        # Should not raise, just log warning
        await client.stop()
        assert not client._started


# ---------------------------------------------------------------------------
# MockLspClient
# ---------------------------------------------------------------------------


class TestMockLspClient:
    @pytest.mark.anyio
    async def test_default_project_root(self):
        client = MockLspClient()
        assert client.project_root == Path(".")

    @pytest.mark.anyio
    async def test_custom_project_root(self):
        client = MockLspClient(project_root=Path("/my/project"))
        assert client.project_root == Path("/my/project")

    @pytest.mark.anyio
    async def test_start_stop_lifecycle(self):
        client = MockLspClient()
        assert not client._started

        await client.start()
        assert client._started

        await client.stop()
        assert not client._started

    @pytest.mark.anyio
    async def test_get_definition_returns_configured(self):
        client = MockLspClient()
        loc = Location("target.py", 10, 0, 15, 1)
        client.set_definition("src.py", 5, 3, [loc])

        result = await client.get_definition("src.py", 5, 3)
        assert result == [loc]

    @pytest.mark.anyio
    async def test_get_definition_returns_empty_when_not_configured(self):
        client = MockLspClient()
        result = await client.get_definition("unknown.py", 1, 0)
        assert result == []

    @pytest.mark.anyio
    async def test_get_references_returns_configured(self):
        client = MockLspClient()
        locs = [Location("a.py", 1, 0, 1, 5), Location("b.py", 2, 0, 2, 5)]
        client.set_references("src.py", 10, 0, locs)

        result = await client.get_references("src.py", 10, 0)
        assert result == locs

    @pytest.mark.anyio
    async def test_get_hover_returns_configured(self):
        client = MockLspClient()
        info = SymbolInfo(
            name="my_var", kind="variable", signature="my_var: int", documentation=None
        )
        client.set_hover("src.py", 3, 5, info)

        result = await client.get_hover("src.py", 3, 5)
        assert result is info

    @pytest.mark.anyio
    async def test_get_hover_returns_none_when_not_configured(self):
        client = MockLspClient()
        result = await client.get_hover("unknown.py", 1, 0)
        assert result is None

    @pytest.mark.anyio
    async def test_get_document_symbols_always_empty(self):
        client = MockLspClient()
        result = await client.get_document_symbols("any_file.py")
        assert result == []
