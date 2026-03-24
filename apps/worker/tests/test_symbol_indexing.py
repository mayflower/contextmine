"""Tests for symbol_indexing module: pure helpers and mocked async functions."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from contextmine_worker.symbol_indexing import (
    SymbolStats,
    _is_code_file,
    _map_symbol_kind,
    extract_symbols_for_document,
    maintain_symbols_for_document,
)

pytestmark = pytest.mark.anyio

# ---------------------------------------------------------------------------
# _is_code_file
# ---------------------------------------------------------------------------


class TestIsCodeFile:
    @pytest.mark.parametrize(
        "uri",
        [
            "src/main.py",
            "lib/utils.ts",
            "component.jsx",
            "app.go",
            "module.rs",
            "Main.java",
            "file.c",
            "file.cpp",
        ],
    )
    def test_code_files_detected(self, uri: str) -> None:
        assert _is_code_file(uri) is True

    @pytest.mark.parametrize(
        "uri",
        [
            "README.md",
            "config.yml",
            "data.json",
            "image.png",
            "style.css",
        ],
    )
    def test_non_code_files_rejected(self, uri: str) -> None:
        assert _is_code_file(uri) is False

    def test_strips_query_params(self) -> None:
        assert _is_code_file("src/main.py?ref=HEAD") is True

    def test_case_insensitive_extension(self) -> None:
        assert _is_code_file("Module.PY") is True


# ---------------------------------------------------------------------------
# _map_symbol_kind
# ---------------------------------------------------------------------------


class TestMapSymbolKind:
    def test_known_kinds(self) -> None:
        from contextmine_core.models import SymbolKind

        assert _map_symbol_kind("function") == SymbolKind.FUNCTION
        assert _map_symbol_kind("class") == SymbolKind.CLASS
        assert _map_symbol_kind("method") == SymbolKind.METHOD
        assert _map_symbol_kind("enum") == SymbolKind.ENUM
        assert _map_symbol_kind("interface") == SymbolKind.INTERFACE
        assert _map_symbol_kind("struct") == SymbolKind.CLASS
        assert _map_symbol_kind("trait") == SymbolKind.INTERFACE
        assert _map_symbol_kind("module") == SymbolKind.MODULE

    def test_unknown_kind_falls_back_to_function(self) -> None:
        from contextmine_core.models import SymbolKind

        assert _map_symbol_kind("something_new") == SymbolKind.FUNCTION


# ---------------------------------------------------------------------------
# SymbolStats dataclass
# ---------------------------------------------------------------------------


class TestSymbolStats:
    def test_defaults(self) -> None:
        stats = SymbolStats()
        assert stats.symbols_created == 0
        assert stats.symbols_deleted == 0
        assert stats.files_processed == 0
        assert stats.files_skipped == 0


# ---------------------------------------------------------------------------
# extract_symbols_for_document (async, mocked)
# ---------------------------------------------------------------------------


class TestExtractSymbolsForDocument:
    async def test_skips_non_code_file(self) -> None:
        doc = MagicMock()
        doc.uri = "README.md"
        db = AsyncMock()

        count = await extract_symbols_for_document(db, doc)
        assert count == 0

    async def test_treesitter_import_error(self) -> None:
        doc = MagicMock()
        doc.uri = "main.py"
        doc.id = uuid.uuid4()
        db = AsyncMock()

        with (
            patch(
                "contextmine_worker.symbol_indexing.extract_outline",
                side_effect=ImportError("no treesitter"),
                create=True,
            ),
            patch.dict("sys.modules", {"contextmine_core.treesitter": None}),
        ):
            # The function catches ImportError internally
            count = await extract_symbols_for_document(db, doc)
            assert count == 0

    async def test_empty_symbols_returns_zero(self) -> None:
        doc = MagicMock()
        doc.uri = "main.py"
        doc.id = uuid.uuid4()
        doc.content_markdown = "def foo(): pass"
        db = AsyncMock()

        with patch("contextmine_core.treesitter.extract_outline", return_value=[]):
            count = await extract_symbols_for_document(db, doc)
            assert count == 0


# ---------------------------------------------------------------------------
# maintain_symbols_for_document (async, mocked)
# ---------------------------------------------------------------------------


class TestMaintainSymbols:
    async def test_document_not_found(self) -> None:
        db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        db.execute.return_value = mock_result

        created, deleted = await maintain_symbols_for_document(db, uuid.uuid4())
        assert created == 0
        assert deleted == 0
