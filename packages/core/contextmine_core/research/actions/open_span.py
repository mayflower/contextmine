"""Open span action for the research agent.

Reads a specific section of a file and registers it as evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from contextmine_core.research.actions.registry import Action, ActionResult
from contextmine_core.research.run import Evidence

if TYPE_CHECKING:
    from contextmine_core.research.run import ResearchRun


class OpenSpanAction(Action):
    """Read a specific span of lines from a file."""

    def __init__(self, base_path: Path | str | None = None):
        """Initialize with optional base path for file resolution.

        Args:
            base_path: Base directory for resolving relative file paths.
                      If None, uses current working directory.
        """
        self._base_path = Path(base_path) if base_path else Path.cwd()

    @property
    def name(self) -> str:
        return "open_span"

    @property
    def description(self) -> str:
        return "Read a specific range of lines from a file. Use to examine code found via search."

    async def execute(
        self,
        run: ResearchRun,
        params: dict[str, Any],
    ) -> ActionResult:
        """Read lines from a file.

        Args:
            run: Current research run
            params: Must contain 'file_path', 'start_line', 'end_line'

        Returns:
            ActionResult with file content as evidence
        """
        file_path = params.get("file_path", "")
        start_line = params.get("start_line", 1)
        end_line = params.get("end_line", start_line + 50)

        if not file_path:
            return ActionResult(
                success=False,
                output_summary="No file path provided",
                error="file_path parameter is required",
            )

        # Validate line numbers
        if start_line < 1:
            start_line = 1
        if end_line < start_line:
            end_line = start_line

        # Limit span size to prevent huge reads
        max_lines = 200
        if end_line - start_line > max_lines:
            end_line = start_line + max_lines

        try:
            # Resolve file path
            path = Path(file_path)
            if not path.is_absolute():
                path = self._base_path / path

            if not path.exists():
                return ActionResult(
                    success=False,
                    output_summary=f"File not found: {file_path}",
                    error=f"File does not exist: {path}",
                )

            if not path.is_file():
                return ActionResult(
                    success=False,
                    output_summary=f"Not a file: {file_path}",
                    error=f"Path is not a file: {path}",
                )

            # Check if within scope if scope is set
            if run.scope:
                # Simple glob-style matching
                try:
                    relative = path.relative_to(self._base_path)
                    if not self._matches_scope(str(relative), run.scope):
                        return ActionResult(
                            success=False,
                            output_summary=f"File outside scope: {file_path}",
                            error=f"File {file_path} is outside research scope: {run.scope}",
                        )
                except ValueError:
                    pass  # Absolute path, skip scope check

            # Read the file
            content_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            total_lines = len(content_lines)

            # Adjust end_line if beyond file length
            if end_line > total_lines:
                end_line = total_lines

            if start_line > total_lines:
                return ActionResult(
                    success=False,
                    output_summary=f"Start line {start_line} beyond file length ({total_lines})",
                    error=f"File has only {total_lines} lines",
                )

            # Extract the span (convert to 0-indexed)
            span_lines = content_lines[start_line - 1 : end_line]
            content = "\n".join(span_lines)

            # Create evidence
            evidence_id = f"ev-{run.run_id[:8]}-{len(run.evidence) + 1:03d}"
            evidence = Evidence(
                id=evidence_id,
                file_path=str(file_path),
                start_line=start_line,
                end_line=end_line,
                content=content,
                reason=f"Manually opened span from {file_path}",
                provenance="manual",
            )

            return ActionResult(
                success=True,
                output_summary=f"Read lines {start_line}-{end_line} from {file_path}",
                evidence=[evidence],
                data={
                    "file_path": str(file_path),
                    "start_line": start_line,
                    "end_line": end_line,
                    "lines_read": len(span_lines),
                    "total_file_lines": total_lines,
                    "evidence_id": evidence_id,
                },
            )

        except PermissionError:
            return ActionResult(
                success=False,
                output_summary=f"Permission denied: {file_path}",
                error=f"Cannot read file: {file_path}",
            )
        except UnicodeDecodeError as e:
            return ActionResult(
                success=False,
                output_summary=f"Encoding error in file: {file_path}",
                error=f"Cannot decode file: {e}",
            )
        except Exception as e:
            return ActionResult(
                success=False,
                output_summary=f"Error reading file: {e}",
                error=str(e),
            )

    def _matches_scope(self, file_path: str, scope: str) -> bool:
        """Check if file path matches the scope pattern.

        Supports simple glob patterns like 'src/**' or 'packages/core/**'.
        """
        import fnmatch

        # Normalize paths
        file_path = file_path.replace("\\", "/")
        scope = scope.replace("\\", "/")

        # Handle ** pattern
        if "**" in scope:
            # Check if path starts with the base pattern
            base = scope.split("**")[0].rstrip("/")
            return not (base and not file_path.startswith(base))

        return fnmatch.fnmatch(file_path, scope)
