"""LSP client wrapper providing a simplified API.

This module wraps the multilspy library to provide a clean, async-friendly
interface for common LSP operations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from contextmine_core.lsp.exceptions import LspServerError

if TYPE_CHECKING:
    from multilspy.language_server import LanguageServer

logger = logging.getLogger(__name__)


@dataclass
class Location:
    """A location in source code."""

    file_path: str
    start_line: int  # 1-indexed
    start_column: int  # 0-indexed
    end_line: int  # 1-indexed
    end_column: int  # 0-indexed

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "start_line": self.start_line,
            "start_column": self.start_column,
            "end_line": self.end_line,
            "end_column": self.end_column,
        }


@dataclass
class SymbolInfo:
    """Information about a symbol from hover."""

    name: str
    kind: str  # function, class, method, variable, etc.
    signature: str | None
    documentation: str | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "kind": self.kind,
            "signature": self.signature,
            "documentation": self.documentation,
        }


@dataclass
class Diagnostic:
    """A diagnostic message from the language server."""

    file_path: str
    line: int  # 1-indexed
    column: int  # 0-indexed
    end_line: int  # 1-indexed
    end_column: int  # 0-indexed
    message: str
    severity: str  # "error", "warning", "info", "hint"
    code: str | None
    source: str | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "line": self.line,
            "column": self.column,
            "end_line": self.end_line,
            "end_column": self.end_column,
            "message": self.message,
            "severity": self.severity,
            "code": self.code,
            "source": self.source,
        }


# Severity mapping from LSP codes to human-readable strings
SEVERITY_MAP = {
    1: "error",
    2: "warning",
    3: "info",
    4: "hint",
}


class LspClient:
    """Wrapper around multilspy LanguageServer with simplified API.

    Provides async methods for common LSP operations:
    - get_definition: Go to definition
    - get_references: Find all references
    - get_hover: Get type/documentation
    - get_diagnostics: Get errors/warnings
    """

    def __init__(self, server: LanguageServer, project_root: Path):
        """Initialize the client.

        Args:
            server: The multilspy LanguageServer instance
            project_root: Root directory of the project
        """
        self._server = server
        self._project_root = project_root
        self._context_manager: Any = None
        self._started = False

    @property
    def project_root(self) -> Path:
        """Get the project root."""
        return self._project_root

    async def start(self) -> None:
        """Start the language server."""
        if not self._started:
            self._context_manager = self._server.start_server()
            await self._context_manager.__aenter__()
            self._started = True
            logger.debug("LSP client started for %s", self._project_root)

    async def stop(self) -> None:
        """Stop the language server."""
        if self._started and self._context_manager:
            try:
                await self._context_manager.__aexit__(None, None, None)
            except Exception as e:
                logger.warning("Error stopping LSP server: %s", e)
            finally:
                self._started = False
                self._context_manager = None
                logger.debug("LSP client stopped for %s", self._project_root)

    def _resolve_path(self, file_path: str) -> str:
        """Resolve a file path relative to project root."""
        path = Path(file_path)
        if not path.is_absolute():
            path = self._project_root / path
        return str(path.resolve())

    async def get_definition(
        self,
        file_path: str,
        line: int,
        column: int,
    ) -> list[Location]:
        """Get definition location(s) for symbol at position.

        Args:
            file_path: Path to the source file
            line: Line number (1-indexed)
            column: Column number (0-indexed)

        Returns:
            List of definition locations
        """
        resolved_path = self._resolve_path(file_path)

        try:
            # multilspy uses 0-indexed lines
            result = await self._server.request_definition(
                resolved_path,
                line - 1,  # Convert to 0-indexed
                column,
            )

            return self._parse_locations(result)
        except Exception as e:
            logger.error("LSP definition request failed: %s", e)
            raise LspServerError(f"Definition request failed: {e}") from e

    async def get_references(
        self,
        file_path: str,
        line: int,
        column: int,
        include_declaration: bool = True,
    ) -> list[Location]:
        """Get all references to symbol at position.

        Args:
            file_path: Path to the source file
            line: Line number (1-indexed)
            column: Column number (0-indexed)
            include_declaration: Whether to include the declaration

        Returns:
            List of reference locations
        """
        resolved_path = self._resolve_path(file_path)

        try:
            # multilspy uses 0-indexed lines
            result = await self._server.request_references(
                resolved_path,
                line - 1,  # Convert to 0-indexed
                column,
            )

            return self._parse_locations(result)
        except Exception as e:
            logger.error("LSP references request failed: %s", e)
            raise LspServerError(f"References request failed: {e}") from e

    async def get_hover(
        self,
        file_path: str,
        line: int,
        column: int,
    ) -> SymbolInfo | None:
        """Get hover information for symbol at position.

        Args:
            file_path: Path to the source file
            line: Line number (1-indexed)
            column: Column number (0-indexed)

        Returns:
            SymbolInfo if available, None otherwise
        """
        resolved_path = self._resolve_path(file_path)

        try:
            # multilspy uses 0-indexed lines
            result = await self._server.request_hover(
                resolved_path,
                line - 1,  # Convert to 0-indexed
                column,
            )

            return self._parse_hover(result)
        except Exception as e:
            logger.error("LSP hover request failed: %s", e)
            raise LspServerError(f"Hover request failed: {e}") from e

    async def get_document_symbols(
        self,
        file_path: str,
    ) -> list[dict[str, Any]]:
        """Get all symbols in a document.

        Args:
            file_path: Path to the source file

        Returns:
            List of symbol information dictionaries
        """
        resolved_path = self._resolve_path(file_path)

        try:
            result = await self._server.request_document_symbols(resolved_path)
            # Cast result to expected type - LSP returns various symbol types
            if result:
                return list(result)
            return []
        except Exception as e:
            logger.error("LSP document symbols request failed: %s", e)
            raise LspServerError(f"Document symbols request failed: {e}") from e

    def _parse_locations(self, result: list[dict[str, Any]] | None) -> list[Location]:
        """Parse LSP location results into Location objects.

        Args:
            result: Raw LSP response (list of Location or LocationLink)

        Returns:
            List of Location objects
        """
        if not result:
            return []

        locations = []
        for item in result:
            try:
                # Handle both Location and LocationLink formats
                if "targetUri" in item:
                    # LocationLink format
                    uri = item["targetUri"]
                    range_obj = item.get("targetRange", item.get("targetSelectionRange", {}))
                elif "uri" in item:
                    # Location format
                    uri = item["uri"]
                    range_obj = item.get("range", {})
                else:
                    continue

                # Parse URI to file path
                file_path = uri
                if file_path.startswith("file://"):
                    file_path = file_path[7:]

                # Get range (convert from 0-indexed to 1-indexed for lines)
                start = range_obj.get("start", {})
                end = range_obj.get("end", {})

                locations.append(
                    Location(
                        file_path=file_path,
                        start_line=start.get("line", 0) + 1,  # 1-indexed
                        start_column=start.get("character", 0),
                        end_line=end.get("line", 0) + 1,  # 1-indexed
                        end_column=end.get("character", 0),
                    )
                )
            except Exception as e:
                logger.warning("Failed to parse location: %s - %s", item, e)
                continue

        return locations

    def _parse_hover(self, result: dict[str, Any] | None) -> SymbolInfo | None:
        """Parse LSP hover result into SymbolInfo.

        Args:
            result: Raw LSP hover response

        Returns:
            SymbolInfo if parseable, None otherwise
        """
        if not result:
            return None

        contents = result.get("contents")
        if not contents:
            return None

        # Extract content string
        content_str = ""
        if isinstance(contents, str):
            content_str = contents
        elif isinstance(contents, dict):
            # MarkedString or MarkupContent
            content_str = contents.get("value", str(contents))
        elif isinstance(contents, list):
            # Array of MarkedStrings
            parts = []
            for item in contents:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(item.get("value", str(item)))
            content_str = "\n".join(parts)

        if not content_str:
            return None

        # Parse the content to extract symbol info
        # This is a best-effort extraction
        lines = content_str.strip().split("\n")
        first_line = lines[0] if lines else ""

        # Try to detect signature (often the first line)
        signature = first_line if first_line else None

        # Try to detect documentation (often after the first line)
        documentation = "\n".join(lines[1:]).strip() if len(lines) > 1 else None

        # Try to detect kind from common patterns
        kind = "unknown"
        lower_content = content_str.lower()
        if "def " in lower_content or "function" in lower_content:
            kind = "function"
        elif "class " in lower_content:
            kind = "class"
        elif "method" in lower_content:
            kind = "method"
        elif "variable" in lower_content or ":" in first_line:
            kind = "variable"

        # Extract name from signature if possible
        name = "unknown"
        if signature:
            # Common patterns: "def name(", "class Name", "name: type"
            if "(" in signature:
                # Function/method
                parts = signature.split("(")[0].split()
                name = parts[-1] if parts else "unknown"
            elif ":" in signature:
                # Variable with type annotation
                name = signature.split(":")[0].strip()
            else:
                # Other (class name, etc.)
                parts = signature.split()
                name = parts[-1] if parts else "unknown"

        return SymbolInfo(
            name=name,
            kind=kind,
            signature=signature,
            documentation=documentation,
        )


class MockLspClient:
    """Mock LSP client for testing without actual language servers."""

    def __init__(self, project_root: Path | None = None):
        """Initialize mock client."""
        self._project_root = project_root or Path(".")
        self._definitions: dict[tuple[str, int, int], list[Location]] = {}
        self._references: dict[tuple[str, int, int], list[Location]] = {}
        self._hovers: dict[tuple[str, int, int], SymbolInfo] = {}
        self._started = False

    @property
    def project_root(self) -> Path:
        """Get the project root."""
        return self._project_root

    def set_definition(
        self,
        file_path: str,
        line: int,
        column: int,
        locations: list[Location],
    ) -> None:
        """Set mock definition response."""
        self._definitions[(file_path, line, column)] = locations

    def set_references(
        self,
        file_path: str,
        line: int,
        column: int,
        locations: list[Location],
    ) -> None:
        """Set mock references response."""
        self._references[(file_path, line, column)] = locations

    def set_hover(
        self,
        file_path: str,
        line: int,
        column: int,
        info: SymbolInfo,
    ) -> None:
        """Set mock hover response."""
        self._hovers[(file_path, line, column)] = info

    async def start(self) -> None:
        """Mock start."""
        self._started = True

    async def stop(self) -> None:
        """Mock stop."""
        self._started = False

    async def get_definition(
        self,
        file_path: str,
        line: int,
        column: int,
    ) -> list[Location]:
        """Get mock definition."""
        return self._definitions.get((file_path, line, column), [])

    async def get_references(
        self,
        file_path: str,
        line: int,
        column: int,
        include_declaration: bool = True,
    ) -> list[Location]:
        """Get mock references."""
        return self._references.get((file_path, line, column), [])

    async def get_hover(
        self,
        file_path: str,
        line: int,
        column: int,
    ) -> SymbolInfo | None:
        """Get mock hover."""
        return self._hovers.get((file_path, line, column))

    async def get_document_symbols(
        self,
        file_path: str,
    ) -> list[dict[str, Any]]:
        """Get mock document symbols."""
        return []
