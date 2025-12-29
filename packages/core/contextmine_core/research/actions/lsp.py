"""LSP-based actions for the research agent.

These actions use Language Server Protocol to provide code intelligence:
- lsp_definition: Go to definition
- lsp_references: Find all references
- lsp_hover: Get type/documentation
- lsp_diagnostics: Get errors/warnings
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from contextmine_core.research.actions.registry import Action, ActionResult
from contextmine_core.research.run import Evidence

if TYPE_CHECKING:
    from contextmine_core.lsp.client import LspClient, MockLspClient
    from contextmine_core.research.run import ResearchRun

logger = logging.getLogger(__name__)


async def _read_file_span(
    file_path: str,
    start_line: int,
    end_line: int,
    max_lines: int = 50,
) -> str:
    """Read a span of lines from a file.

    Args:
        file_path: Path to the file
        start_line: Starting line (1-indexed)
        end_line: Ending line (1-indexed, inclusive)
        max_lines: Maximum number of lines to read

    Returns:
        The content of the specified lines
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"[File not found: {file_path}]"

        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        # Clamp line numbers
        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), end_line)

        # Limit to max_lines
        if end_idx - start_idx > max_lines:
            end_idx = start_idx + max_lines

        return "".join(lines[start_idx:end_idx])
    except Exception as e:
        return f"[Error reading file: {e}]"


class LspDefinitionAction(Action):
    """Find the definition of a symbol using LSP."""

    def __init__(
        self,
        lsp_client: LspClient | MockLspClient | None = None,
    ):
        """Initialize the action.

        Args:
            lsp_client: Optional LSP client (uses global manager if None)
        """
        self._lsp_client = lsp_client

    @property
    def name(self) -> str:
        return "lsp_definition"

    @property
    def description(self) -> str:
        return (
            "Jump to the definition of a symbol using Language Server Protocol. "
            "Provide file path, line number (1-indexed), and column (0-indexed)."
        )

    async def _get_client(self, file_path: str) -> LspClient | MockLspClient:
        """Get or create an LSP client."""
        if self._lsp_client is not None:
            return self._lsp_client

        from contextmine_core.lsp import get_lsp_manager

        manager = get_lsp_manager()
        return await manager.get_client(file_path)

    async def execute(
        self,
        run: ResearchRun,
        params: dict[str, Any],
    ) -> ActionResult:
        """Execute go-to-definition.

        Args:
            run: Current research run
            params: Must contain 'file_path', 'line', 'column'

        Returns:
            ActionResult with definition locations as evidence
        """
        file_path = params.get("file_path", "")
        line = params.get("line", 1)
        column = params.get("column", 0)

        if not file_path:
            return ActionResult(
                success=False,
                output_summary="No file path provided",
                error="file_path is required",
            )

        try:
            client = await self._get_client(file_path)
            locations = await client.get_definition(file_path, line, column)

            if not locations:
                return ActionResult(
                    success=True,
                    output_summary=f"No definition found at {file_path}:{line}:{column}",
                    data={"locations": []},
                )

            # Create evidence for each definition location
            evidence_items: list[Evidence] = []
            for loc in locations:
                # Read the content at the location
                content = await _read_file_span(
                    loc.file_path,
                    loc.start_line,
                    loc.end_line + 10,  # Include some context
                )

                evidence_id = (
                    f"ev-{run.run_id[:8]}-{len(run.evidence) + len(evidence_items) + 1:03d}"
                )
                evidence = Evidence(
                    id=evidence_id,
                    file_path=loc.file_path,
                    start_line=loc.start_line,
                    end_line=loc.end_line + 10,
                    content=content[:2000],
                    reason="Definition found via LSP go-to-definition",
                    provenance="lsp",
                    symbol_kind="definition",
                )
                evidence_items.append(evidence)

            return ActionResult(
                success=True,
                output_summary=f"Found {len(locations)} definition(s) at {locations[0].file_path}:{locations[0].start_line}",
                evidence=evidence_items,
                data={"locations": [loc.to_dict() for loc in locations]},
            )

        except Exception as e:
            logger.warning("LSP definition failed: %s", e)
            return ActionResult(
                success=False,
                output_summary=f"LSP not available: {e}",
                error=str(e),
            )


class LspReferencesAction(Action):
    """Find all references to a symbol using LSP."""

    def __init__(
        self,
        lsp_client: LspClient | MockLspClient | None = None,
    ):
        """Initialize the action.

        Args:
            lsp_client: Optional LSP client (uses global manager if None)
        """
        self._lsp_client = lsp_client

    @property
    def name(self) -> str:
        return "lsp_references"

    @property
    def description(self) -> str:
        return (
            "Find all usages of a symbol in the codebase using Language Server Protocol. "
            "Provide file path, line number (1-indexed), and column (0-indexed)."
        )

    async def _get_client(self, file_path: str) -> LspClient | MockLspClient:
        """Get or create an LSP client."""
        if self._lsp_client is not None:
            return self._lsp_client

        from contextmine_core.lsp import get_lsp_manager

        manager = get_lsp_manager()
        return await manager.get_client(file_path)

    async def execute(
        self,
        run: ResearchRun,
        params: dict[str, Any],
    ) -> ActionResult:
        """Execute find-references.

        Args:
            run: Current research run
            params: Must contain 'file_path', 'line', 'column'

        Returns:
            ActionResult with reference locations as evidence
        """
        file_path = params.get("file_path", "")
        line = params.get("line", 1)
        column = params.get("column", 0)

        if not file_path:
            return ActionResult(
                success=False,
                output_summary="No file path provided",
                error="file_path is required",
            )

        try:
            client = await self._get_client(file_path)
            locations = await client.get_references(file_path, line, column)

            if not locations:
                return ActionResult(
                    success=True,
                    output_summary=f"No references found at {file_path}:{line}:{column}",
                    data={"locations": [], "total_references": 0},
                )

            # Create evidence for each reference (limit to first 10 to avoid flooding)
            evidence_items: list[Evidence] = []
            max_evidence = 10

            for i, loc in enumerate(locations[:max_evidence]):
                # Read a few lines around the reference
                content = await _read_file_span(
                    loc.file_path,
                    max(1, loc.start_line - 2),
                    loc.end_line + 2,
                )

                evidence_id = (
                    f"ev-{run.run_id[:8]}-{len(run.evidence) + len(evidence_items) + 1:03d}"
                )
                evidence = Evidence(
                    id=evidence_id,
                    file_path=loc.file_path,
                    start_line=max(1, loc.start_line - 2),
                    end_line=loc.end_line + 2,
                    content=content[:1500],
                    reason=f"Reference {i + 1} of {len(locations)} found via LSP",
                    provenance="lsp",
                    symbol_kind="reference",
                )
                evidence_items.append(evidence)

            return ActionResult(
                success=True,
                output_summary=f"Found {len(locations)} reference(s) across {len(set(loc.file_path for loc in locations))} files",
                evidence=evidence_items,
                data={
                    "locations": [loc.to_dict() for loc in locations],
                    "total_references": len(locations),
                },
            )

        except Exception as e:
            logger.warning("LSP references failed: %s", e)
            return ActionResult(
                success=False,
                output_summary=f"LSP not available: {e}",
                error=str(e),
            )


class LspHoverAction(Action):
    """Get hover information for a symbol using LSP."""

    def __init__(
        self,
        lsp_client: LspClient | MockLspClient | None = None,
    ):
        """Initialize the action.

        Args:
            lsp_client: Optional LSP client (uses global manager if None)
        """
        self._lsp_client = lsp_client

    @property
    def name(self) -> str:
        return "lsp_hover"

    @property
    def description(self) -> str:
        return (
            "Get type signature and documentation for a symbol using Language Server Protocol. "
            "Provide file path, line number (1-indexed), and column (0-indexed)."
        )

    async def _get_client(self, file_path: str) -> LspClient | MockLspClient:
        """Get or create an LSP client."""
        if self._lsp_client is not None:
            return self._lsp_client

        from contextmine_core.lsp import get_lsp_manager

        manager = get_lsp_manager()
        return await manager.get_client(file_path)

    async def execute(
        self,
        run: ResearchRun,
        params: dict[str, Any],
    ) -> ActionResult:
        """Execute hover.

        Args:
            run: Current research run
            params: Must contain 'file_path', 'line', 'column'

        Returns:
            ActionResult with symbol information
        """
        file_path = params.get("file_path", "")
        line = params.get("line", 1)
        column = params.get("column", 0)

        if not file_path:
            return ActionResult(
                success=False,
                output_summary="No file path provided",
                error="file_path is required",
            )

        try:
            client = await self._get_client(file_path)
            info = await client.get_hover(file_path, line, column)

            if not info:
                return ActionResult(
                    success=True,
                    output_summary=f"No hover info at {file_path}:{line}:{column}",
                    data={},
                )

            # Create evidence from hover info
            content_parts = []
            if info.signature:
                content_parts.append(f"Signature: {info.signature}")
            if info.documentation:
                content_parts.append(f"Documentation:\n{info.documentation}")

            content = "\n\n".join(content_parts) or f"{info.kind}: {info.name}"

            evidence_id = f"ev-{run.run_id[:8]}-{len(run.evidence) + 1:03d}"
            evidence = Evidence(
                id=evidence_id,
                file_path=file_path,
                start_line=line,
                end_line=line,
                content=content[:2000],
                reason=f"Hover info for {info.name} ({info.kind})",
                provenance="lsp",
                symbol_id=info.name,
                symbol_kind=info.kind,
            )

            return ActionResult(
                success=True,
                output_summary=f"Found {info.kind} '{info.name}' at {file_path}:{line}",
                evidence=[evidence],
                data=info.to_dict(),
            )

        except Exception as e:
            logger.warning("LSP hover failed: %s", e)
            return ActionResult(
                success=False,
                output_summary=f"LSP not available: {e}",
                error=str(e),
            )


class LspDiagnosticsAction(Action):
    """Get diagnostics (errors, warnings) for files using LSP."""

    def __init__(
        self,
        lsp_client: LspClient | MockLspClient | None = None,
    ):
        """Initialize the action.

        Args:
            lsp_client: Optional LSP client (uses global manager if None)
        """
        self._lsp_client = lsp_client

    @property
    def name(self) -> str:
        return "lsp_diagnostics"

    @property
    def description(self) -> str:
        return (
            "Get compiler errors and warnings for specified files using Language Server Protocol. "
            "Provide a list of file paths to check."
        )

    async def _get_client(self, file_path: str) -> LspClient | MockLspClient:
        """Get or create an LSP client."""
        if self._lsp_client is not None:
            return self._lsp_client

        from contextmine_core.lsp import get_lsp_manager

        manager = get_lsp_manager()
        return await manager.get_client(file_path)

    async def execute(
        self,
        run: ResearchRun,
        params: dict[str, Any],
    ) -> ActionResult:
        """Execute diagnostics check.

        Args:
            run: Current research run
            params: Must contain 'file_paths' list

        Returns:
            ActionResult with diagnostics information
        """
        file_paths = params.get("file_paths", [])

        if not file_paths:
            return ActionResult(
                success=False,
                output_summary="No file paths provided",
                error="file_paths is required",
            )

        # Note: Getting diagnostics from LSP is complex because it's push-based.
        # For now, we'll return a simplified implementation that indicates
        # diagnostics support is limited.
        # TODO: Implement proper diagnostics collection after file open

        try:
            # Try to get a client to verify LSP is available
            if file_paths:
                await self._get_client(file_paths[0])

            # For now, return success with a note about limitations
            return ActionResult(
                success=True,
                output_summary=f"Diagnostics check for {len(file_paths)} files (limited support)",
                data={
                    "files_checked": len(file_paths),
                    "note": "Full diagnostics require language server document sync",
                },
            )

        except Exception as e:
            logger.warning("LSP diagnostics failed: %s", e)
            return ActionResult(
                success=False,
                output_summary=f"LSP not available: {e}",
                error=str(e),
            )


# =============================================================================
# MOCK ACTIONS FOR TESTING
# =============================================================================


class MockLspDefinitionAction(Action):
    """Mock LSP definition action for testing."""

    def __init__(
        self,
        mock_locations: list[dict[str, Any]] | None = None,
    ):
        """Initialize with mock data.

        Args:
            mock_locations: List of location dicts to return
        """
        self._mock_locations = mock_locations or []

    @property
    def name(self) -> str:
        return "lsp_definition"

    @property
    def description(self) -> str:
        return "Mock go-to-definition for testing."

    def set_locations(self, locations: list[dict[str, Any]]) -> None:
        """Set mock locations."""
        self._mock_locations = locations

    async def execute(
        self,
        run: ResearchRun,
        params: dict[str, Any],
    ) -> ActionResult:
        """Return mock locations."""
        file_path = params.get("file_path", "")

        if not file_path:
            return ActionResult(
                success=False,
                output_summary="No file path provided",
                error="file_path is required",
            )

        if not self._mock_locations:
            return ActionResult(
                success=True,
                output_summary="No definition found (mock)",
                data={"locations": []},
            )

        # Create evidence from mock locations
        evidence_items: list[Evidence] = []
        for i, loc in enumerate(self._mock_locations):
            evidence_id = f"ev-{run.run_id[:8]}-{len(run.evidence) + i + 1:03d}"
            evidence = Evidence(
                id=evidence_id,
                file_path=loc.get("file_path", "mock.py"),
                start_line=loc.get("start_line", 1),
                end_line=loc.get("end_line", 10),
                content=loc.get("content", "# mock content"),
                reason="Definition found via mock LSP",
                provenance="lsp",
                symbol_kind="definition",
            )
            evidence_items.append(evidence)

        return ActionResult(
            success=True,
            output_summary=f"Found {len(self._mock_locations)} definition(s) (mock)",
            evidence=evidence_items,
            data={"locations": self._mock_locations},
        )
