"""Complexity and LOC extraction using lizard."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from contextmine_core.metrics.discovery import to_repo_relative_path
from contextmine_core.metrics.models import MetricsGateError


def _extract_fileinfos(analysis: Any) -> list[Any]:
    for attr in ("fileinfo_list", "files"):
        value = getattr(analysis, attr, None)
        if value is not None:
            return list(value)

    if isinstance(analysis, Iterable):
        return list(analysis)

    return []


def _complexity_from_fileinfo(file_info: Any) -> float:
    function_list = list(getattr(file_info, "function_list", []) or [])
    if function_list:
        return float(
            sum(
                float(
                    getattr(function, "cyclomatic_complexity", None)
                    or getattr(function, "CCN", None)
                    or 0.0
                )
                for function in function_list
            )
        )

    average = getattr(file_info, "average_cyclomatic_complexity", None)
    count = getattr(file_info, "function_count", None)
    if average is not None and count is not None:
        return float(average) * float(count)

    return 0.0


def aggregate_lizard_metrics(
    file_infos: Iterable[Any],
    repo_root: Path,
    project_root: Path,
    relevant_files: set[str],
) -> dict[str, dict[str, float | int]]:
    """Aggregate lizard file info objects into canonical file metrics."""
    aggregated: dict[str, dict[str, float | int]] = {}

    for file_info in file_infos:
        raw_filename = str(getattr(file_info, "filename", ""))
        if not raw_filename:
            continue

        file_path = to_repo_relative_path(
            raw_filename,
            repo_root=repo_root,
            project_root=project_root,
        )
        if not file_path or (relevant_files and file_path not in relevant_files):
            continue

        nloc = int(getattr(file_info, "nloc", 0) or 0)
        complexity = _complexity_from_fileinfo(file_info)
        aggregated[file_path] = {
            "loc": nloc,
            "complexity": complexity,
        }

    return aggregated


def extract_complexity_loc_metrics(
    repo_root: Path,
    project_root: Path,
    relevant_files: set[str],
) -> dict[str, dict[str, float | int]]:
    """Run lizard and return per-file LOC/complexity for relevant files."""
    if not relevant_files:
        return {}

    try:
        import lizard
    except Exception as exc:  # pragma: no cover - exercised in integration
        raise MetricsGateError(
            "complexity_extraction_error",
            details={"reason": "lizard_not_available", "error": str(exc)},
        ) from exc

    files_to_analyze = sorted(
        str((repo_root / file_path).resolve()) for file_path in relevant_files
    )

    try:
        analysis = lizard.analyze(files_to_analyze)
    except TypeError:
        analysis = lizard.analyze(paths=files_to_analyze)
    except Exception as exc:  # pragma: no cover - exercised in integration
        raise MetricsGateError(
            "complexity_extraction_error",
            details={"reason": "lizard_runtime_error", "error": str(exc)},
        ) from exc

    file_infos = _extract_fileinfos(analysis)
    if not file_infos:
        raise MetricsGateError(
            "complexity_extraction_error",
            details={"reason": "no_fileinfos_from_lizard", "project_root": str(project_root)},
        )

    return aggregate_lizard_metrics(
        file_infos=file_infos,
        repo_root=repo_root,
        project_root=project_root,
        relevant_files=relevant_files,
    )
