"""Approximate file-level duplication ratio metrics."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

_COMMENT_PREFIXES = ("#", "//", "/*", "*", "--")


def _normalize_line(raw: str) -> str | None:
    stripped = raw.strip()
    if not stripped:
        return None
    if len(stripped) < 8:
        return None
    if stripped.startswith(_COMMENT_PREFIXES):
        return None
    return " ".join(stripped.split())


def compute_file_duplication_ratio(
    repo_root: Path,
    relevant_files: set[str],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Compute a lightweight cross-file duplication ratio for relevant files.

    Ratio is `duplicated_normalized_lines / normalized_nonblank_lines` per file.
    A normalized line is considered duplicated if it appears in at least two files.
    """
    repo_root = repo_root.resolve()

    hashes_by_file: dict[str, list[str]] = {}
    file_count_by_hash: dict[str, set[str]] = {}
    normalized_line_count: dict[str, int] = {}
    unreadable_files: list[str] = []

    for file_path in sorted(relevant_files):
        absolute = (repo_root / file_path).resolve()
        try:
            content = absolute.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            unreadable_files.append(file_path)
            hashes_by_file[file_path] = []
            normalized_line_count[file_path] = 0
            continue

        hashes: list[str] = []
        for raw_line in content.splitlines():
            normalized = _normalize_line(raw_line)
            if not normalized:
                continue
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            hashes.append(digest)
            file_count_by_hash.setdefault(digest, set()).add(file_path)

        hashes_by_file[file_path] = hashes
        normalized_line_count[file_path] = len(hashes)

    duplication_map: dict[str, float] = {}
    duplicate_line_total = 0
    normalized_line_total = 0
    for file_path in sorted(relevant_files):
        hashes = hashes_by_file.get(file_path, [])
        total = int(normalized_line_count.get(file_path, 0))
        duplicated = sum(1 for digest in hashes if len(file_count_by_hash.get(digest, set())) > 1)
        duplicate_line_total += duplicated
        normalized_line_total += total
        ratio = float(duplicated / total) if total > 0 else 0.0
        duplication_map[file_path] = ratio

    provenance = {
        "files_scanned": len(relevant_files),
        "files_unreadable": len(unreadable_files),
        "normalized_lines_total": normalized_line_total,
        "duplicated_lines_total": duplicate_line_total,
    }
    if unreadable_files:
        provenance["unreadable_files"] = unreadable_files[:50]
    return duplication_map, provenance
