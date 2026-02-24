"""Coverage report discovery and path normalization helpers."""

from __future__ import annotations

from pathlib import Path

from contextmine_core.pathing import canonicalize_repo_relative_path

DEFAULT_COVERAGE_REPORT_PATTERNS: tuple[str, ...] = (
    "**/lcov.info",
    "**/coverage.xml",
    "**/cobertura.xml",
    "**/jacoco.xml",
    "**/jacocoTestReport.xml",
    "**/clover.xml",
    "**/phpunit.xml",
)


def normalize_posix_path(path: str) -> str:
    """Normalize path separators and remove leading './' segments."""
    return canonicalize_repo_relative_path(path)


def to_repo_relative_path(
    raw_path: str,
    repo_root: Path,
    project_root: Path | None = None,
    base_dir: Path | None = None,
) -> str | None:
    """Convert a tool/report path to a repo-relative POSIX path when possible."""
    text = raw_path.strip()
    if not text:
        return None

    if text.startswith("file://"):
        text = text.removeprefix("file://")

    repo_root = repo_root.resolve()
    project_root = project_root.resolve() if project_root else None
    base_dir = base_dir.resolve() if base_dir else None
    raw = Path(text)

    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        if base_dir is not None:
            candidates.append(base_dir / raw)
        if project_root is not None:
            candidates.append(project_root / raw)
        candidates.append(repo_root / raw)

    for candidate in candidates:
        try:
            rel = candidate.resolve(strict=False).relative_to(repo_root)
        except ValueError:
            continue
        normalized = normalize_posix_path(rel.as_posix())
        return normalized or None

    normalized = normalize_posix_path(text)
    return normalized or None


def _glob_existing(base: Path, pattern: str) -> list[Path]:
    return [match for match in base.glob(pattern) if match.is_file()]


def discover_coverage_reports(
    repo_root: Path,
    project_root: Path,
    configured_patterns: list[str] | None,
    autodiscovery_enabled: bool = True,
) -> list[Path]:
    """Resolve coverage report paths with config-first discovery.

    `configured_patterns` are repository-relative glob patterns.
    """
    repo_root = repo_root.resolve()
    project_root = project_root.resolve()

    discovered: list[Path] = []

    for pattern in configured_patterns or []:
        discovered.extend(_glob_existing(repo_root, pattern))

    if not discovered and autodiscovery_enabled:
        for pattern in DEFAULT_COVERAGE_REPORT_PATTERNS:
            discovered.extend(_glob_existing(project_root, pattern))

    unique: dict[str, Path] = {}
    for report in discovered:
        unique[str(report.resolve())] = report.resolve()

    return sorted(unique.values(), key=lambda path: str(path))
