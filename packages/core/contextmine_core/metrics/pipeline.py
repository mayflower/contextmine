"""End-to-end polyglot metrics extraction pipeline with strict gating."""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from contextmine_core.metrics.complexity_loc_lizard import extract_complexity_loc_metrics
from contextmine_core.metrics.coupling_from_snapshot import compute_file_coupling_from_snapshots
from contextmine_core.metrics.coverage_reports import parse_coverage_reports
from contextmine_core.metrics.discovery import discover_coverage_reports, to_repo_relative_path
from contextmine_core.metrics.models import FileMetricRecord, MetricsGateError, ProjectMetricBundle
from contextmine_core.semantic_snapshot.models import Snapshot

EXCLUDED_DIRS = {
    "node_modules",
    "vendor",
    "dist",
    "build",
    "target",
    ".venv",
    "venv",
    "__pycache__",
}

TEST_PATTERNS = (
    "*/test/*",
    "*/tests/*",
    "*/__tests__/*",
    "*.spec.*",
    "*.test.*",
    "*_test.py",
    "*Test.java",
)

GENERATED_PATTERNS = (
    "*/generated/*",
    "*/gen/*",
    "*.generated.*",
)


def parse_metrics_languages(raw: str) -> set[str]:
    return {value.strip().lower() for value in raw.split(",") if value.strip()}


def is_relevant_production_file(file_path: str) -> bool:
    normalized = file_path.replace("\\", "/")
    lower_path = normalized.lower()
    wrapped = f"/{lower_path}/"

    for directory in EXCLUDED_DIRS:
        if f"/{directory}/" in wrapped:
            return False

    for pattern in TEST_PATTERNS:
        if fnmatch(normalized, pattern) or fnmatch(lower_path, pattern.lower()):
            return False

    for pattern in GENERATED_PATTERNS:
        if fnmatch(normalized, pattern) or fnmatch(lower_path, pattern.lower()):
            return False

    return True


def _project_key(project_root: Path, repo_root: Path) -> tuple[str, str]:
    resolved_project = project_root.resolve()
    resolved_repo = repo_root.resolve()
    try:
        relative = resolved_project.relative_to(resolved_repo).as_posix()
        if relative == ".":
            relative = ""
    except ValueError:
        relative = ""
    return (relative, str(resolved_project))


def _group_snapshots_by_project(
    snapshot_dicts: list[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for snapshot_dict in snapshot_dicts:
        meta = dict(snapshot_dict.get("meta") or {})
        repo_relative_root = str(meta.get("repo_relative_root") or "")
        project_root = str(meta.get("project_root") or "")
        key = (repo_relative_root, project_root)
        grouped.setdefault(key, []).append(snapshot_dict)
    return grouped


def _collect_relevant_files(
    repo_root: Path,
    project_root: Path,
    project_snapshots: list[dict[str, Any]],
) -> set[str]:
    relevant_files: set[str] = set()

    for snapshot_dict in project_snapshots:
        snapshot = Snapshot.from_dict(snapshot_dict)
        snapshot_project_root = Path(str(snapshot.meta.get("project_root") or project_root))

        for file_info in snapshot.files:
            repo_rel = to_repo_relative_path(
                file_info.path,
                repo_root=repo_root,
                project_root=snapshot_project_root,
            )
            if not repo_rel:
                continue
            if is_relevant_production_file(repo_rel):
                relevant_files.add(repo_rel)

    return relevant_files


def _enforce_missing_coverage(
    strict_mode: bool,
    project_root: Path,
    report_paths: list[Path],
) -> None:
    if strict_mode and not report_paths:
        raise MetricsGateError(
            "missing_coverage_report",
            details={"project_root": str(project_root)},
        )


def _enforce_coverage_mismatch(
    strict_mode: bool,
    project_root: Path,
    relevant_files: set[str],
    coverage_map: dict[str, float],
) -> None:
    if not strict_mode or not relevant_files:
        return

    matched = [path for path in relevant_files if path in coverage_map]
    if not matched:
        raise MetricsGateError(
            "coverage_path_mismatch",
            details={
                "project_root": str(project_root),
                "relevant_file_count": len(relevant_files),
                "coverage_file_count": len(coverage_map),
            },
        )


def _build_file_metrics(
    language: str,
    relevant_files: set[str],
    complexity_map: dict[str, dict[str, float | int]],
    coupling_map: dict[str, dict[str, int | float]],
    coupling_provenance: dict[str, Any],
    coverage_map: dict[str, float],
    coverage_sources: dict[str, dict[str, object]],
    strict_mode: bool,
) -> list[FileMetricRecord]:
    records: list[FileMetricRecord] = []
    missing: list[dict[str, Any]] = []

    for file_path in sorted(relevant_files):
        complexity_entry = complexity_map.get(file_path) or {}
        coupling_entry = coupling_map.get(file_path) or {}
        coverage = coverage_map.get(file_path)

        loc_value = complexity_entry.get("loc")
        complexity_value = complexity_entry.get("complexity")
        coupling_in = coupling_entry.get("coupling_in")
        coupling_out = coupling_entry.get("coupling_out")
        coupling_value = coupling_entry.get("coupling")

        required_missing: list[str] = []
        if loc_value is None:
            required_missing.append("loc")
        if complexity_value is None:
            required_missing.append("complexity")
        if coupling_in is None:
            required_missing.append("coupling_in")
        if coupling_out is None:
            required_missing.append("coupling_out")
        if coupling_value is None:
            required_missing.append("coupling")
        if coverage is None:
            required_missing.append("coverage")

        if required_missing:
            missing.append({"file_path": file_path, "missing": required_missing})
            continue

        record = FileMetricRecord(
            file_path=file_path,
            language=language,
            loc=int(loc_value),
            complexity=float(complexity_value),
            coupling_in=int(coupling_in),
            coupling_out=int(coupling_out),
            coupling=float(coupling_value),
            coverage=float(coverage),
            sources={
                "complexity": "lizard",
                "coverage": coverage_sources.get(file_path, {}),
                "coupling": coupling_provenance,
            },
        )
        records.append(record)

    if strict_mode and missing:
        raise MetricsGateError(
            "missing_required_metrics",
            details={"files": missing[:200], "missing_count": len(missing)},
        )

    return records


def run_polyglot_metrics_pipeline(
    repo_root: Path,
    project_dicts: list[dict[str, Any]],
    snapshot_dicts: list[dict[str, Any]],
    coverage_report_patterns: list[str] | None,
    strict_mode: bool,
    metrics_languages: str,
    autodiscovery_enabled: bool,
) -> list[ProjectMetricBundle]:
    """Run real metric extraction and strict gate validation."""
    repo_root = repo_root.resolve()
    enabled_languages = parse_metrics_languages(metrics_languages)
    grouped_snapshots = _group_snapshots_by_project(snapshot_dicts)

    bundles: list[ProjectMetricBundle] = []

    for project_dict in project_dicts:
        language = str(project_dict.get("language", "")).lower()
        if language not in enabled_languages:
            continue

        project_root = Path(str(project_dict["root_path"])).resolve()
        key = _project_key(project_root, repo_root)

        project_snapshots = grouped_snapshots.get(key)
        if not project_snapshots:
            continue

        relevant_files = _collect_relevant_files(repo_root, project_root, project_snapshots)
        if not relevant_files:
            continue

        report_paths = discover_coverage_reports(
            repo_root=repo_root,
            project_root=project_root,
            configured_patterns=coverage_report_patterns,
            autodiscovery_enabled=autodiscovery_enabled,
        )
        _enforce_missing_coverage(strict_mode, project_root, report_paths)

        coverage_map, coverage_sources = parse_coverage_reports(
            report_paths=report_paths,
            repo_root=repo_root,
            project_root=project_root,
        )
        _enforce_coverage_mismatch(strict_mode, project_root, relevant_files, coverage_map)

        complexity_map = extract_complexity_loc_metrics(
            repo_root=repo_root,
            project_root=project_root,
            relevant_files=relevant_files,
        )
        coupling_map, coupling_provenance = compute_file_coupling_from_snapshots(
            snapshot_dicts=project_snapshots,
            repo_root=repo_root,
            project_root=project_root,
            relevant_files=relevant_files,
        )

        files = _build_file_metrics(
            language=language,
            relevant_files=relevant_files,
            complexity_map=complexity_map,
            coupling_map=coupling_map,
            coupling_provenance=coupling_provenance,
            coverage_map=coverage_map,
            coverage_sources=coverage_sources,
            strict_mode=strict_mode,
        )

        bundles.append(
            ProjectMetricBundle(
                project_root=str(project_root),
                language=language,
                files=files,
            )
        )

    return bundles


def flatten_metric_bundles(bundles: list[ProjectMetricBundle]) -> list[FileMetricRecord]:
    """Flatten project bundles into file metric records."""
    flattened: list[FileMetricRecord] = []
    for bundle in bundles:
        flattened.extend(bundle.files)
    return flattened
