"""Tree-sitter based actions for the research agent.

These actions use Tree-sitter to provide code structure analysis:
- ts_outline: Get outline of symbols in a file
- ts_find_symbol: Find a symbol by name
- ts_enclosing_symbol: Find the enclosing symbol at a line
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from contextmine_core.research.actions.registry import Action, ActionResult
from contextmine_core.research.run import Evidence

if TYPE_CHECKING:
    from contextmine_core.research.run import ResearchRun
    from contextmine_core.treesitter.outline import Symbol

logger = logging.getLogger(__name__)


def _symbol_to_dict(symbol: Symbol) -> dict[str, Any]:
    """Convert Symbol to dictionary for output."""
    return {
        "name": symbol.name,
        "kind": symbol.kind.value,
        "file_path": symbol.file_path,
        "start_line": symbol.start_line,
        "end_line": symbol.end_line,
        "start_column": symbol.start_column,
        "end_column": symbol.end_column,
        "signature": symbol.signature,
        "parent": symbol.parent,
    }


def _read_file_content(file_path: str, start_line: int, end_line: int) -> str:
    """Read content from a file between lines."""
    try:
        path = Path(file_path)
        if not path.exists():
            return f"[File not found: {file_path}]"
        content = path.read_text(encoding="utf-8", errors="replace")
        lines = content.split("\n")
        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), end_line)
        return "\n".join(lines[start_idx:end_idx])
    except Exception as e:
        return f"[Error reading file: {e}]"


class TsOutlineAction(Action):
    """Get the outline of symbols in a source file using Tree-sitter."""

    @property
    def name(self) -> str:
        return "ts_outline"

    @property
    def description(self) -> str:
        return (
            "Get the outline of all functions, classes, and other symbols in a file. "
            "Useful for understanding file structure quickly."
        )

    async def execute(
        self,
        run: ResearchRun,
        params: dict[str, Any],
    ) -> ActionResult:
        """Execute ts_outline action.

        Args:
            run: Current research run
            params: Must contain 'file_path'

        Returns:
            ActionResult with symbol outline
        """
        file_path = params.get("file_path", "")

        if not file_path:
            return ActionResult(
                success=False,
                output_summary="No file path provided",
                error="file_path is required",
            )

        try:
            from contextmine_core.treesitter import extract_outline

            symbols = extract_outline(file_path, include_children=True)

            if not symbols:
                return ActionResult(
                    success=True,
                    output_summary=f"No symbols found in {Path(file_path).name}",
                    data={"symbols": [], "file_path": file_path},
                )

            # Format outline for summary
            outline_lines = []
            for sym in symbols:
                indent = ""
                outline_lines.append(
                    f"{indent}{sym.kind.value} {sym.name} (L{sym.start_line}-{sym.end_line})"
                )
                for child in sym.children:
                    outline_lines.append(
                        f"  {child.kind.value} {child.name} (L{child.start_line}-{child.end_line})"
                    )

            summary = f"Found {len(symbols)} top-level symbols:\n" + "\n".join(outline_lines[:20])
            if len(outline_lines) > 20:
                summary += f"\n... and {len(outline_lines) - 20} more"

            return ActionResult(
                success=True,
                output_summary=summary,
                data={
                    "symbols": [_symbol_to_dict(s) for s in symbols],
                    "file_path": file_path,
                },
            )

        except ImportError as e:
            logger.warning("Tree-sitter not available: %s", e)
            return ActionResult(
                success=False,
                output_summary="Tree-sitter not available",
                error=str(e),
            )
        except FileNotFoundError:
            return ActionResult(
                success=False,
                output_summary=f"File not found: {file_path}",
                error=f"File not found: {file_path}",
            )
        except Exception as e:
            logger.warning("Tree-sitter outline failed: %s", e)
            return ActionResult(
                success=False,
                output_summary=f"Failed to extract outline: {e}",
                error=str(e),
            )


class TsFindSymbolAction(Action):
    """Find a symbol by name in a source file using Tree-sitter."""

    @property
    def name(self) -> str:
        return "ts_find_symbol"

    @property
    def description(self) -> str:
        return (
            "Find a specific function, class, or method by name in a file. "
            "Returns the symbol location and its source code."
        )

    async def execute(
        self,
        run: ResearchRun,
        params: dict[str, Any],
    ) -> ActionResult:
        """Execute ts_find_symbol action.

        Args:
            run: Current research run
            params: Must contain 'file_path' and 'name'

        Returns:
            ActionResult with symbol info and evidence
        """
        file_path = params.get("file_path", "")
        symbol_name = params.get("name", "")

        if not file_path:
            return ActionResult(
                success=False,
                output_summary="No file path provided",
                error="file_path is required",
            )

        if not symbol_name:
            return ActionResult(
                success=False,
                output_summary="No symbol name provided",
                error="name is required",
            )

        try:
            from contextmine_core.treesitter import find_symbol_by_name, get_symbol_content

            symbol = find_symbol_by_name(file_path, symbol_name)

            if not symbol:
                return ActionResult(
                    success=True,
                    output_summary=f"Symbol '{symbol_name}' not found in {Path(file_path).name}",
                    data={"found": False, "symbol": None},
                )

            # Get the symbol content
            content = get_symbol_content(symbol)

            # Create evidence
            evidence_id = f"ev-{run.run_id[:8]}-{len(run.evidence) + 1:03d}"
            evidence = Evidence(
                id=evidence_id,
                file_path=file_path,
                start_line=symbol.start_line,
                end_line=symbol.end_line,
                content=content[:2000],
                reason=f"Found {symbol.kind.value} '{symbol_name}' via Tree-sitter",
                provenance="treesitter",
                symbol_id=symbol.name,
                symbol_kind=symbol.kind.value,
            )

            return ActionResult(
                success=True,
                output_summary=f"Found {symbol.kind.value} '{symbol_name}' at {Path(file_path).name}:{symbol.start_line}-{symbol.end_line}",
                evidence=[evidence],
                data={
                    "found": True,
                    "symbol": _symbol_to_dict(symbol),
                },
            )

        except ImportError as e:
            logger.warning("Tree-sitter not available: %s", e)
            return ActionResult(
                success=False,
                output_summary="Tree-sitter not available",
                error=str(e),
            )
        except FileNotFoundError:
            return ActionResult(
                success=False,
                output_summary=f"File not found: {file_path}",
                error=f"File not found: {file_path}",
            )
        except Exception as e:
            logger.warning("Tree-sitter find_symbol failed: %s", e)
            return ActionResult(
                success=False,
                output_summary=f"Failed to find symbol: {e}",
                error=str(e),
            )


class TsEnclosingSymbolAction(Action):
    """Find the symbol enclosing a specific line using Tree-sitter."""

    @property
    def name(self) -> str:
        return "ts_enclosing_symbol"

    @property
    def description(self) -> str:
        return (
            "Find what function, class, or method contains a specific line. "
            "Useful for understanding context of a code location."
        )

    async def execute(
        self,
        run: ResearchRun,
        params: dict[str, Any],
    ) -> ActionResult:
        """Execute ts_enclosing_symbol action.

        Args:
            run: Current research run
            params: Must contain 'file_path' and 'line'

        Returns:
            ActionResult with enclosing symbol info
        """
        file_path = params.get("file_path", "")
        line = params.get("line", 1)

        if not file_path:
            return ActionResult(
                success=False,
                output_summary="No file path provided",
                error="file_path is required",
            )

        try:
            from contextmine_core.treesitter import find_enclosing_symbol, get_symbol_content

            symbol = find_enclosing_symbol(file_path, line)

            if not symbol:
                return ActionResult(
                    success=True,
                    output_summary=f"Line {line} is not inside any symbol in {Path(file_path).name}",
                    data={"found": False, "symbol": None, "line": line},
                )

            # Get the symbol content
            content = get_symbol_content(symbol)

            # Create evidence
            evidence_id = f"ev-{run.run_id[:8]}-{len(run.evidence) + 1:03d}"
            evidence = Evidence(
                id=evidence_id,
                file_path=file_path,
                start_line=symbol.start_line,
                end_line=symbol.end_line,
                content=content[:2000],
                reason=f"Enclosing {symbol.kind.value} '{symbol.name}' for line {line}",
                provenance="treesitter",
                symbol_id=symbol.name,
                symbol_kind=symbol.kind.value,
            )

            return ActionResult(
                success=True,
                output_summary=f"Line {line} is inside {symbol.kind.value} '{symbol.name}' (L{symbol.start_line}-{symbol.end_line})",
                evidence=[evidence],
                data={
                    "found": True,
                    "symbol": _symbol_to_dict(symbol),
                    "line": line,
                },
            )

        except ImportError as e:
            logger.warning("Tree-sitter not available: %s", e)
            return ActionResult(
                success=False,
                output_summary="Tree-sitter not available",
                error=str(e),
            )
        except FileNotFoundError:
            return ActionResult(
                success=False,
                output_summary=f"File not found: {file_path}",
                error=f"File not found: {file_path}",
            )
        except Exception as e:
            logger.warning("Tree-sitter enclosing_symbol failed: %s", e)
            return ActionResult(
                success=False,
                output_summary=f"Failed to find enclosing symbol: {e}",
                error=str(e),
            )


# =============================================================================
# MOCK ACTIONS FOR TESTING
# =============================================================================


class MockTsOutlineAction(Action):
    """Mock Tree-sitter outline action for testing."""

    def __init__(
        self,
        mock_symbols: list[dict[str, Any]] | None = None,
    ):
        """Initialize with mock data.

        Args:
            mock_symbols: List of symbol dicts to return
        """
        self._mock_symbols = mock_symbols or []

    @property
    def name(self) -> str:
        return "ts_outline"

    @property
    def description(self) -> str:
        return "Mock outline for testing."

    def set_symbols(self, symbols: list[dict[str, Any]]) -> None:
        """Set mock symbols."""
        self._mock_symbols = symbols

    async def execute(
        self,
        run: ResearchRun,
        params: dict[str, Any],
    ) -> ActionResult:
        """Return mock outline."""
        file_path = params.get("file_path", "")

        if not file_path:
            return ActionResult(
                success=False,
                output_summary="No file path provided",
                error="file_path is required",
            )

        if not self._mock_symbols:
            return ActionResult(
                success=True,
                output_summary="No symbols found (mock)",
                data={"symbols": [], "file_path": file_path},
            )

        return ActionResult(
            success=True,
            output_summary=f"Found {len(self._mock_symbols)} symbols (mock)",
            data={"symbols": self._mock_symbols, "file_path": file_path},
        )


class MockTsFindSymbolAction(Action):
    """Mock Tree-sitter find symbol action for testing."""

    def __init__(
        self,
        mock_symbol: dict[str, Any] | None = None,
    ):
        """Initialize with mock data.

        Args:
            mock_symbol: Symbol dict to return
        """
        self._mock_symbol = mock_symbol

    @property
    def name(self) -> str:
        return "ts_find_symbol"

    @property
    def description(self) -> str:
        return "Mock find symbol for testing."

    def set_symbol(self, symbol: dict[str, Any] | None) -> None:
        """Set mock symbol."""
        self._mock_symbol = symbol

    async def execute(
        self,
        run: ResearchRun,
        params: dict[str, Any],
    ) -> ActionResult:
        """Return mock symbol."""
        file_path = params.get("file_path", "")
        symbol_name = params.get("name", "")

        if not file_path:
            return ActionResult(
                success=False,
                output_summary="No file path provided",
                error="file_path is required",
            )

        if not symbol_name:
            return ActionResult(
                success=False,
                output_summary="No symbol name provided",
                error="name is required",
            )

        if not self._mock_symbol:
            return ActionResult(
                success=True,
                output_summary=f"Symbol '{symbol_name}' not found (mock)",
                data={"found": False, "symbol": None},
            )

        # Create mock evidence
        evidence_id = f"ev-{run.run_id[:8]}-{len(run.evidence) + 1:03d}"
        evidence = Evidence(
            id=evidence_id,
            file_path=file_path,
            start_line=self._mock_symbol.get("start_line", 1),
            end_line=self._mock_symbol.get("end_line", 10),
            content=self._mock_symbol.get("content", "# mock content"),
            reason=f"Found symbol '{symbol_name}' via mock Tree-sitter",
            provenance="treesitter",
            symbol_id=self._mock_symbol.get("name", symbol_name),
            symbol_kind=self._mock_symbol.get("kind", "function"),
        )

        return ActionResult(
            success=True,
            output_summary=f"Found symbol '{symbol_name}' (mock)",
            evidence=[evidence],
            data={"found": True, "symbol": self._mock_symbol},
        )
