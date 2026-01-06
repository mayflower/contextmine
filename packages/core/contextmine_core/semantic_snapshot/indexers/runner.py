"""Subprocess runner for SCIP indexers.

Provides a safe subprocess execution wrapper with timeout,
output capture, and error handling.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class CommandNotFoundError(Exception):
    """Raised when the command is not found."""

    pass


class CommandTimeoutError(Exception):
    """Raised when the command times out."""

    pass


class CommandFailedError(Exception):
    """Raised when the command fails with non-zero exit code."""

    def __init__(self, message: str, exit_code: int, stderr: str):
        super().__init__(message)
        self.exit_code = exit_code
        self.stderr = stderr


@dataclass
class CmdResult:
    """Result of running a command."""

    exit_code: int
    stdout_tail: str  # Last N chars of stdout
    stderr_tail: str  # Last N chars of stderr
    elapsed_s: float
    timed_out: bool = False


def run_cmd(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout_s: int = 300,
    logs_path: Path | None = None,
    tail_chars: int = 2000,
) -> CmdResult:
    """Run a subprocess with timeout and output capture.

    Args:
        cmd: Command and arguments
        cwd: Working directory
        env: Environment variable overrides (merged with os.environ)
        timeout_s: Timeout in seconds
        logs_path: Optional path to write full stdout/stderr
        tail_chars: Number of characters to capture in result

    Returns:
        CmdResult with exit code and output tails

    Raises:
        CommandNotFoundError: If the command is not found
        CommandTimeoutError: If the command times out
    """
    # Merge environment
    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    logger.debug("Running command: %s in %s", " ".join(cmd), cwd)
    start_time = time.monotonic()

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=run_env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )

        elapsed = time.monotonic() - start_time

        # Optionally write full logs
        if logs_path:
            logs_path.parent.mkdir(parents=True, exist_ok=True)
            with open(logs_path, "w") as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Exit code: {result.returncode}\n")
                f.write(f"Duration: {elapsed:.2f}s\n")
                f.write("\n--- STDOUT ---\n")
                f.write(result.stdout or "")
                f.write("\n--- STDERR ---\n")
                f.write(result.stderr or "")

        return CmdResult(
            exit_code=result.returncode,
            stdout_tail=result.stdout[-tail_chars:] if result.stdout else "",
            stderr_tail=result.stderr[-tail_chars:] if result.stderr else "",
            elapsed_s=elapsed,
            timed_out=False,
        )

    except FileNotFoundError as e:
        raise CommandNotFoundError(f"Command not found: {cmd[0]}") from e

    except subprocess.TimeoutExpired as e:
        elapsed = time.monotonic() - start_time

        # Capture partial output
        stdout = e.stdout.decode("utf-8", errors="replace") if e.stdout else ""
        stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""

        if logs_path:
            logs_path.parent.mkdir(parents=True, exist_ok=True)
            with open(logs_path, "w") as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"TIMEOUT after {timeout_s}s\n")
                f.write(f"Duration: {elapsed:.2f}s\n")
                f.write("\n--- STDOUT ---\n")
                f.write(stdout)
                f.write("\n--- STDERR ---\n")
                f.write(stderr)

        return CmdResult(
            exit_code=-1,
            stdout_tail=stdout[-tail_chars:] if stdout else "",
            stderr_tail=stderr[-tail_chars:] if stderr else "",
            elapsed_s=elapsed,
            timed_out=True,
        )


def check_tool_version(tool_name: str) -> tuple[bool, str]:
    """Check if a tool is available and get its version.

    Args:
        tool_name: Name of the tool (e.g., "scip-python", "scip-typescript")

    Returns:
        Tuple of (available, version_string)
    """
    try:
        # Try --version first
        result = subprocess.run(
            [tool_name, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            version = result.stdout.strip() or result.stderr.strip()
            return True, version

        # Some tools use -v
        result = subprocess.run(
            [tool_name, "-v"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            version = result.stdout.strip() or result.stderr.strip()
            return True, version

        # Tool exists but version check failed
        return True, "unknown"

    except FileNotFoundError:
        return False, ""
    except subprocess.TimeoutExpired:
        return True, "unknown"
    except Exception as e:
        logger.debug("Error checking %s version: %s", tool_name, e)
        return False, ""
