"""Shared path normalization helpers for repo-relative file paths."""

from __future__ import annotations

from pathlib import PurePosixPath


def canonicalize_repo_relative_path(path: str) -> str:
    """Canonicalize a repo-relative path.

    Rules:
    - normalize path separators to "/"
    - strip leading "./" segments
    - strip leading "/" so paths remain repo-relative
    - collapse redundant separators/segments via PurePosixPath
    """
    normalized = path.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = str(PurePosixPath(normalized))
    if normalized == ".":
        return ""
    return normalized.lstrip("/")
