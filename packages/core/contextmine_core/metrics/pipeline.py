"""End-to-end polyglot structural metrics extraction pipeline."""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from contextmine_core.metrics.complexity_loc_lizard import extract_complexity_loc_metrics
from contextmine_core.metrics.coupling_from_snapshot import compute_file_coupling_from_snapshots
from contextmine_core.metrics.discovery import to_repo_relative_path
from contextmine_core.metrics.duplication import compute_file_duplication_ratio
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


def _normalize_language(language: str) -> str:
    return language.strip().lower()


def _project_snapshot_languages(language: str) -> tuple[str, ...]:
    normalized = _normalize_language(language)
    if normalized == "javascript":
        # Explicit compatibility: JS projects may consume scip-typescript snapshots.
        return ("javascript", "typescript")
    return (normalized,)


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


def _project_key(project_root: Path, repo_root: Path, language: str) -> tuple[str, str, str]:
    resolved_project = project_root.resolve()
    resolved_repo = repo_root.resolve()
    try:
        relative = resolved_project.relative_to(resolved_repo).as_posix()
        if relative == ".":
            relative = ""
    except ValueError:
        relative = ""
    return (relative, str(resolved_project), _normalize_language(language))


def _group_snapshots_by_project(
    snapshot_dicts: list[dict[str, Any]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for snapshot_dict in snapshot_dicts:
        meta = dict(snapshot_dict.get("meta") or {})
        repo_relative_root = str(meta.get("repo_relative_root") or "")
        if repo_relative_root == ".":
            repo_relative_root = ""
        project_root = str(meta.get("project_root") or "")
        language = _normalize_language(str(meta.get("language") or ""))
        key = (repo_relative_root, project_root, language)
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


def _build_file_metrics(
    language: str,
    relevant_files: set[str],
    complexity_map: dict[str, dict[str, float | int]],
    coupling_map: dict[str, dict[str, int | float | bool]],
    duplication_map: dict[str, float],
    coupling_provenance: dict[str, Any],
    strict_mode: bool,
) -> list[FileMetricRecord]:
    records: list[FileMetricRecord] = []
    missing: list[dict[str, Any]] = []

    for file_path in sorted(relevant_files):
        complexity_entry = complexity_map.get(file_path) or {}
        coupling_entry = coupling_map.get(file_path) or {}

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
            cohesion=float(coupling_entry.get("cohesion", 1.0) or 1.0),
            instability=float(coupling_entry.get("instability", 0.0) or 0.0),
            fan_in=int(coupling_entry.get("fan_in", 0) or 0),
            fan_out=int(coupling_entry.get("fan_out", 0) or 0),
            cycle_participation=bool(coupling_entry.get("cycle_participation", False)),
            cycle_size=int(coupling_entry.get("cycle_size", 0) or 0),
            duplication_ratio=float(duplication_map.get(file_path, 0.0) or 0.0),
            crap_score=None,
            coverage=None,
            sources={
                "complexity": "lizard",
                "coupling": coupling_provenance,
                "duplication": {
                    "provider": "line_hash",
                    "normalization": "strip_whitespace_and_comments",
                },
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
    strict_mode: bool,
    metrics_languages: str,
) -> list[ProjectMetricBundle]:
    """Run real structural metric extraction and strict validation."""
    repo_root = repo_root.resolve()
    enabled_languages = parse_metrics_languages(metrics_languages)
    grouped_snapshots = _group_snapshots_by_project(snapshot_dicts)

    bundles: list[ProjectMetricBundle] = []
    claimed_files: set[str] = set()

    for project_dict in project_dicts:
        language = _normalize_language(str(project_dict.get("language", "")))
        if language not in enabled_languages:
            continue

        project_root = Path(str(project_dict["root_path"])).resolve()
        project_snapshots: list[dict[str, Any]] = []
        for snapshot_language in _project_snapshot_languages(language):
            key = _project_key(project_root, repo_root, snapshot_language)
            project_snapshots.extend(grouped_snapshots.get(key, []))
        if not project_snapshots:
            continue

        relevant_files = _collect_relevant_files(repo_root, project_root, project_snapshots)
        relevant_files = {path for path in relevant_files if path not in claimed_files}
        if not relevant_files:
            continue

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
        duplication_map, duplication_provenance = compute_file_duplication_ratio(
            repo_root=repo_root,
            relevant_files=relevant_files,
        )
        coupling_provenance = {
            **coupling_provenance,
            "duplication": duplication_provenance,
        }

        files = _build_file_metrics(
            language=language,
            relevant_files=relevant_files,
            complexity_map=complexity_map,
            coupling_map=coupling_map,
            duplication_map=duplication_map,
            coupling_provenance=coupling_provenance,
            strict_mode=strict_mode,
        )
        claimed_files.update(record.file_path for record in files)

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
    by_file: dict[str, FileMetricRecord] = {}

    def language_priority(language: str) -> int:
        normalized = _normalize_language(language)
        if normalized == "typescript":
            return 2
        if normalized == "javascript":
            return 1
        return 0

    for bundle in bundles:
        for record in bundle.files:
            existing = by_file.get(record.file_path)
            if existing is None:
                by_file[record.file_path] = record
                continue
            if language_priority(record.language) > language_priority(existing.language):
                by_file[record.file_path] = record

    flattened.extend(by_file[path] for path in sorted(by_file.keys()))
    return flattened
