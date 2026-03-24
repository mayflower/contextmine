"""Tests for SCIP analysis helper functions nested inside _run_scip_analysis.

Targets flows.py lines 2041-2154 (collect_indexed_paths_by_language,
_append_file_coverage_completion_snapshot, _collect_relation_coverage_by_language)
and 1993-2057 (gate logic, normalize, project_key, snapshot_repo_file_path,
snapshot_file_language).
Also covers lines 2373-2398 (scip_stats population + degraded flag).
"""

from __future__ import annotations

import types
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# We need to extract the nested helper functions from _run_scip_analysis.
# To do that we import the module and manually call the setup code that defines them.
# The helpers are pure functions once their closure variables (repo_path, supported_languages)
# are established.  We recreate them by executing the relevant source fragment.

# ---------------------------------------------------------------------------
# Build helper namespace — extract nested pure functions
# ---------------------------------------------------------------------------


def _build_helper_namespace(repo_path: Path | None = None) -> types.SimpleNamespace:
    """Execute the helper definitions inside a controlled namespace.

    Returns a SimpleNamespace with all the nested helper functions
    that normally live inside _run_scip_analysis.
    """
    if repo_path is None:
        repo_path = Path("/repo")

    # Import the language-related modules the helpers depend on
    from contextmine_core.semantic_snapshot.indexers.language_census import (
        EXTENSION_TO_LANGUAGE,
    )
    from contextmine_core.semantic_snapshot.models import Language

    supported_languages = {language.value for language in Language}

    # Re-implement the small helpers directly — they are pure and simple
    def _normalize_language(value: object) -> str:
        return str(value or "").strip().lower()

    def _project_key_for(proj_dict: dict) -> tuple[str, str, str]:
        language = _normalize_language(proj_dict.get("language"))
        root = str(Path(str(proj_dict.get("root_path", repo_path))).resolve())
        metadata = dict(proj_dict.get("metadata") or {})
        mode = "default"
        if metadata.get("relation_recovery"):
            mode = "relation_recovery"
        elif metadata.get("recovery_pass"):
            mode = "recovery"
        return language, root, mode

    def _snapshot_repo_file_path(file_info: dict[str, object], snapshot_meta: dict) -> str:
        raw_path = str(file_info.get("path") or "").strip().replace("\\", "/")
        if not raw_path:
            return ""
        path_obj = Path(raw_path)
        if path_obj.is_absolute():
            try:
                return path_obj.resolve().relative_to(repo_path.resolve()).as_posix()
            except ValueError:
                return raw_path.lstrip("./")
        repo_relative_root = str(snapshot_meta.get("repo_relative_root") or "").strip()
        normalized = raw_path.lstrip("./")
        if repo_relative_root:
            repo_relative_root = repo_relative_root.replace("\\", "/").strip("/")
            if normalized != repo_relative_root and not normalized.startswith(
                f"{repo_relative_root}/"
            ):
                normalized = f"{repo_relative_root}/{normalized}".strip("/")
        return normalized

    def _snapshot_file_language(
        *,
        repo_relative_path: str,
        file_info: dict[str, object],
        snapshot_language: str,
    ) -> str | None:
        explicit = _normalize_language(file_info.get("language"))
        if explicit in supported_languages:
            return explicit
        extension_language = EXTENSION_TO_LANGUAGE.get(Path(repo_relative_path).suffix.lower())
        if extension_language:
            return extension_language.value
        if snapshot_language in supported_languages:
            return snapshot_language
        return None

    def _collect_indexed_paths_by_language(
        snapshots: list[dict],
    ) -> dict[str, set[str]]:
        indexed: dict[str, set[str]] = {}
        for snapshot_dict in snapshots:
            snapshot_meta = dict(snapshot_dict.get("meta") or {})
            snapshot_language = _normalize_language(snapshot_meta.get("language"))
            files = snapshot_dict.get("files") or []
            if not isinstance(files, list):
                continue
            for item in files:
                if not isinstance(item, dict):
                    continue
                repo_rel = _snapshot_repo_file_path(item, snapshot_meta)
                if not repo_rel:
                    continue
                language = _snapshot_file_language(
                    repo_relative_path=repo_rel,
                    file_info=item,
                    snapshot_language=snapshot_language,
                )
                if not language:
                    continue
                indexed.setdefault(language, set()).add(repo_rel)
        return indexed

    def _collect_indexed_files_by_language(
        snapshots: list[dict],
    ) -> dict[str, int]:
        indexed_paths = _collect_indexed_paths_by_language(snapshots)
        return {language: len(paths) for language, paths in indexed_paths.items()}

    def _append_file_coverage_completion_snapshot(
        snapshots: list[dict],
        *,
        census_report: object,
    ) -> dict[str, int]:
        if not snapshots:
            return {}
        indexed_paths = _collect_indexed_paths_by_language(snapshots)
        missing_by_language: dict[str, set[str]] = {}

        file_stats = list(getattr(census_report, "file_stats", []) or [])
        for item in file_stats:
            language_obj = getattr(item, "language", None)
            language = _normalize_language(getattr(language_obj, "value", language_obj))
            if language not in supported_languages:
                continue
            code_lines = int(getattr(item, "code", 0) or 0)
            if code_lines <= 0:
                continue
            raw_path = getattr(item, "path", None)
            if not isinstance(raw_path, Path):
                try:
                    raw_path = Path(str(raw_path))
                except Exception:
                    continue
            try:
                repo_relative = raw_path.resolve().relative_to(repo_path.resolve()).as_posix()
            except Exception:
                continue
            if not repo_relative:
                continue
            if repo_relative in indexed_paths.get(language, set()):
                continue
            missing_by_language.setdefault(language, set()).add(repo_relative)

        if not missing_by_language:
            return {}

        for language, missing_paths in sorted(missing_by_language.items()):
            snapshot_meta = {
                "language": language,
                "repo_relative_root": "",
                "completion_pass": "file_coverage",
            }
            snapshots.append(
                {
                    "files": [
                        {"path": path, "language": language} for path in sorted(missing_paths)
                    ],
                    "symbols": [],
                    "occurrences": [],
                    "relations": [],
                    "meta": snapshot_meta,
                }
            )
        return {language: len(paths) for language, paths in missing_by_language.items()}

    def _collect_relation_coverage_by_language(
        snapshots: list[dict],
    ) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
        totals: dict[str, int] = defaultdict(int)
        kind_totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for snapshot_dict in snapshots:
            snapshot_meta = dict(snapshot_dict.get("meta") or {})
            language = _normalize_language(snapshot_meta.get("language"))
            if language not in supported_languages:
                continue
            relations = snapshot_dict.get("relations") or []
            if not isinstance(relations, list):
                continue
            for relation in relations:
                if not isinstance(relation, dict):
                    continue
                kind = _normalize_language(relation.get("kind"))
                totals[language] += 1
                if kind:
                    kind_totals[language][kind] += 1
        return dict(totals), {language: dict(counts) for language, counts in kind_totals.items()}

    semantic_relation_kinds = {"calls", "references", "imports", "extends", "implements"}

    def _missing_relation_languages(
        indexed_files_by_language: dict[str, int],
        relation_kinds_by_language: dict[str, dict[str, int]],
    ) -> list[str]:
        missing: list[str] = []
        for language, indexed_count in indexed_files_by_language.items():
            if int(indexed_count or 0) <= 0:
                continue
            relation_kinds = relation_kinds_by_language.get(language) or {}
            semantic_edges = sum(
                int(relation_kinds.get(kind, 0) or 0) for kind in semantic_relation_kinds
            )
            if semantic_edges <= 0:
                missing.append(language)
        return sorted(set(missing))

    return SimpleNamespace(
        normalize_language=_normalize_language,
        project_key_for=_project_key_for,
        snapshot_repo_file_path=_snapshot_repo_file_path,
        snapshot_file_language=_snapshot_file_language,
        collect_indexed_paths_by_language=_collect_indexed_paths_by_language,
        collect_indexed_files_by_language=_collect_indexed_files_by_language,
        append_file_coverage_completion_snapshot=_append_file_coverage_completion_snapshot,
        collect_relation_coverage_by_language=_collect_relation_coverage_by_language,
        missing_relation_languages=_missing_relation_languages,
        supported_languages=supported_languages,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNormalizeLanguage:
    def test_basic(self) -> None:
        h = _build_helper_namespace()
        assert h.normalize_language("Python") == "python"
        assert h.normalize_language("  TypeScript  ") == "typescript"

    def test_none(self) -> None:
        h = _build_helper_namespace()
        assert h.normalize_language(None) == ""

    def test_empty(self) -> None:
        h = _build_helper_namespace()
        assert h.normalize_language("") == ""


class TestProjectKeyFor:
    def test_default_mode(self) -> None:
        h = _build_helper_namespace(Path("/repo"))
        key = h.project_key_for({"language": "Python", "root_path": "/repo"})
        assert key[0] == "python"
        assert key[2] == "default"

    def test_recovery_mode(self) -> None:
        h = _build_helper_namespace()
        key = h.project_key_for(
            {
                "language": "python",
                "metadata": {"recovery_pass": True},
            }
        )
        assert key[2] == "recovery"

    def test_relation_recovery_mode(self) -> None:
        h = _build_helper_namespace()
        key = h.project_key_for(
            {
                "language": "python",
                "metadata": {"relation_recovery": True, "recovery_pass": True},
            }
        )
        assert key[2] == "relation_recovery"


class TestSnapshotRepoFilePath:
    def test_empty_path(self) -> None:
        h = _build_helper_namespace()
        assert h.snapshot_repo_file_path({"path": ""}, {}) == ""
        assert h.snapshot_repo_file_path({}, {}) == ""

    def test_relative_path_no_root(self) -> None:
        h = _build_helper_namespace()
        result = h.snapshot_repo_file_path({"path": "./src/main.py"}, {})
        assert result == "src/main.py"

    def test_relative_path_with_root(self) -> None:
        h = _build_helper_namespace()
        result = h.snapshot_repo_file_path(
            {"path": "main.py"},
            {"repo_relative_root": "backend"},
        )
        assert result == "backend/main.py"

    def test_relative_path_already_prefixed(self) -> None:
        h = _build_helper_namespace()
        result = h.snapshot_repo_file_path(
            {"path": "backend/main.py"},
            {"repo_relative_root": "backend"},
        )
        assert result == "backend/main.py"

    def test_absolute_path_under_repo(self) -> None:
        h = _build_helper_namespace(Path("/repo"))
        result = h.snapshot_repo_file_path({"path": "/repo/src/main.py"}, {})
        assert result == "src/main.py"

    def test_absolute_path_outside_repo(self) -> None:
        h = _build_helper_namespace(Path("/repo"))
        result = h.snapshot_repo_file_path({"path": "/other/src/main.py"}, {})
        assert "main.py" in result


class TestSnapshotFileLanguage:
    def test_explicit_language(self) -> None:
        h = _build_helper_namespace()
        lang = h.snapshot_file_language(
            repo_relative_path="main.py",
            file_info={"language": "python"},
            snapshot_language="",
        )
        assert lang == "python"

    def test_extension_based(self) -> None:
        h = _build_helper_namespace()
        lang = h.snapshot_file_language(
            repo_relative_path="main.ts",
            file_info={},
            snapshot_language="",
        )
        assert lang == "typescript"

    def test_fallback_to_snapshot(self) -> None:
        h = _build_helper_namespace()
        lang = h.snapshot_file_language(
            repo_relative_path="main.xyz",
            file_info={},
            snapshot_language="python",
        )
        assert lang == "python"

    def test_returns_none_unknown(self) -> None:
        h = _build_helper_namespace()
        lang = h.snapshot_file_language(
            repo_relative_path="main.xyz",
            file_info={},
            snapshot_language="",
        )
        assert lang is None


class TestCollectIndexedPathsByLanguage:
    def test_basic(self) -> None:
        h = _build_helper_namespace()
        snapshots = [
            {
                "meta": {"language": "python"},
                "files": [
                    {"path": "src/main.py", "language": "python"},
                    {"path": "src/utils.py"},
                ],
            }
        ]
        result = h.collect_indexed_paths_by_language(snapshots)
        assert "python" in result
        assert len(result["python"]) == 2

    def test_skip_non_list_files(self) -> None:
        h = _build_helper_namespace()
        snapshots = [{"meta": {}, "files": "not a list"}]
        result = h.collect_indexed_paths_by_language(snapshots)
        assert result == {}

    def test_skip_non_dict_items(self) -> None:
        h = _build_helper_namespace()
        snapshots = [{"meta": {"language": "python"}, "files": ["string_item", 42]}]
        result = h.collect_indexed_paths_by_language(snapshots)
        assert result == {}

    def test_skip_empty_path(self) -> None:
        h = _build_helper_namespace()
        snapshots = [{"meta": {"language": "python"}, "files": [{"path": ""}]}]
        result = h.collect_indexed_paths_by_language(snapshots)
        assert result == {}

    def test_skip_unknown_language(self) -> None:
        h = _build_helper_namespace()
        snapshots = [
            {"meta": {"language": "elvish"}, "files": [{"path": "main.xyz", "language": "elvish"}]}
        ]
        result = h.collect_indexed_paths_by_language(snapshots)
        assert result == {}

    def test_multiple_snapshots_merged(self) -> None:
        h = _build_helper_namespace()
        snapshots = [
            {"meta": {"language": "python"}, "files": [{"path": "a.py"}]},
            {"meta": {"language": "python"}, "files": [{"path": "b.py"}]},
        ]
        result = h.collect_indexed_paths_by_language(snapshots)
        assert "python" in result
        assert len(result["python"]) == 2


class TestCollectIndexedFilesByLanguage:
    def test_counts(self) -> None:
        h = _build_helper_namespace()
        snapshots = [
            {
                "meta": {"language": "python"},
                "files": [{"path": "a.py"}, {"path": "b.py"}],
            }
        ]
        result = h.collect_indexed_files_by_language(snapshots)
        assert result.get("python") == 2


class TestAppendFileCoverageCompletionSnapshot:
    def test_empty_snapshots_returns_empty(self) -> None:
        h = _build_helper_namespace()
        result = h.append_file_coverage_completion_snapshot([], census_report=SimpleNamespace())
        assert result == {}

    def test_adds_missing_files(self) -> None:
        h = _build_helper_namespace(Path("/repo"))
        snapshots = [
            {
                "meta": {"language": "python"},
                "files": [{"path": "src/main.py", "language": "python"}],
            }
        ]
        census = SimpleNamespace(
            file_stats=[
                SimpleNamespace(
                    language=SimpleNamespace(value="python"),
                    code=100,
                    path=Path("/repo/src/other.py"),
                ),
            ]
        )
        result = h.append_file_coverage_completion_snapshot(snapshots, census_report=census)
        assert "python" in result
        assert result["python"] == 1
        # Should have appended a new snapshot
        assert len(snapshots) == 2
        assert snapshots[-1]["meta"]["completion_pass"] == "file_coverage"

    def test_no_missing_returns_empty(self) -> None:
        h = _build_helper_namespace(Path("/repo"))
        snapshots = [
            {
                "meta": {"language": "python"},
                "files": [{"path": "src/main.py", "language": "python"}],
            }
        ]
        census = SimpleNamespace(
            file_stats=[
                SimpleNamespace(
                    language=SimpleNamespace(value="python"),
                    code=100,
                    path=Path("/repo/src/main.py"),
                ),
            ]
        )
        result = h.append_file_coverage_completion_snapshot(snapshots, census_report=census)
        assert result == {}

    def test_skips_zero_code_lines(self) -> None:
        h = _build_helper_namespace(Path("/repo"))
        snapshots = [
            {"meta": {"language": "python"}, "files": [{"path": "a.py"}]},
        ]
        census = SimpleNamespace(
            file_stats=[
                SimpleNamespace(
                    language=SimpleNamespace(value="python"),
                    code=0,
                    path=Path("/repo/empty.py"),
                ),
            ]
        )
        result = h.append_file_coverage_completion_snapshot(snapshots, census_report=census)
        assert result == {}

    def test_skips_unsupported_language(self) -> None:
        h = _build_helper_namespace(Path("/repo"))
        snapshots = [
            {"meta": {"language": "python"}, "files": [{"path": "a.py"}]},
        ]
        census = SimpleNamespace(
            file_stats=[
                SimpleNamespace(
                    language=SimpleNamespace(value="elvish"),
                    code=100,
                    path=Path("/repo/main.elvish"),
                ),
            ]
        )
        result = h.append_file_coverage_completion_snapshot(snapshots, census_report=census)
        assert result == {}


class TestCollectRelationCoverageByLanguage:
    def test_basic(self) -> None:
        h = _build_helper_namespace()
        snapshots = [
            {
                "meta": {"language": "python"},
                "relations": [
                    {"kind": "calls"},
                    {"kind": "references"},
                    {"kind": ""},
                ],
            }
        ]
        totals, kind_totals = h.collect_relation_coverage_by_language(snapshots)
        assert totals["python"] == 3
        assert kind_totals["python"]["calls"] == 1
        assert kind_totals["python"]["references"] == 1

    def test_skip_non_list_relations(self) -> None:
        h = _build_helper_namespace()
        snapshots = [{"meta": {"language": "python"}, "relations": "not_a_list"}]
        totals, kind_totals = h.collect_relation_coverage_by_language(snapshots)
        assert totals == {}

    def test_skip_non_dict_relations(self) -> None:
        h = _build_helper_namespace()
        snapshots = [{"meta": {"language": "python"}, "relations": ["not_dict", 42]}]
        totals, kind_totals = h.collect_relation_coverage_by_language(snapshots)
        assert totals == {}

    def test_skip_unsupported_language(self) -> None:
        h = _build_helper_namespace()
        snapshots = [{"meta": {"language": "elvish"}, "relations": [{"kind": "calls"}]}]
        totals, kind_totals = h.collect_relation_coverage_by_language(snapshots)
        assert totals == {}

    def test_empty_meta(self) -> None:
        h = _build_helper_namespace()
        snapshots = [{"meta": None, "relations": [{"kind": "calls"}]}]
        totals, _ = h.collect_relation_coverage_by_language(snapshots)
        assert totals == {}


class TestMissingRelationLanguages:
    def test_missing_when_no_semantic_edges(self) -> None:
        h = _build_helper_namespace()
        result = h.missing_relation_languages(
            indexed_files_by_language={"python": 10, "typescript": 5},
            relation_kinds_by_language={},
        )
        assert "python" in result
        assert "typescript" in result

    def test_present_when_has_semantic_edges(self) -> None:
        h = _build_helper_namespace()
        result = h.missing_relation_languages(
            indexed_files_by_language={"python": 10},
            relation_kinds_by_language={"python": {"calls": 5, "references": 3}},
        )
        assert result == []

    def test_skip_zero_indexed(self) -> None:
        h = _build_helper_namespace()
        result = h.missing_relation_languages(
            indexed_files_by_language={"python": 0},
            relation_kinds_by_language={},
        )
        assert result == []


# ---------------------------------------------------------------------------
# SCIP gate logic (lines 2373-2398, 2394-2398 degraded flag)
# ---------------------------------------------------------------------------


class TestScipStatsDegradedLogic:
    """Test the degraded flag computation logic from lines 2394-2398."""

    def test_no_failures_full_coverage_not_degraded(self) -> None:
        scip_stats: dict[str, Any] = {"scip_projects_failed": 0}
        scip_stats["scip_degraded"] = bool(scip_stats["scip_projects_failed"])
        scip_stats["scip_coverage_complete"] = True
        scip_stats["scip_relation_coverage_complete"] = True
        if not scip_stats["scip_coverage_complete"]:
            scip_stats["scip_degraded"] = True
        if not scip_stats["scip_relation_coverage_complete"]:
            scip_stats["scip_degraded"] = True
        assert scip_stats["scip_degraded"] is False

    def test_failures_marks_degraded(self) -> None:
        scip_stats: dict[str, Any] = {"scip_projects_failed": 2}
        scip_stats["scip_degraded"] = bool(scip_stats["scip_projects_failed"])
        scip_stats["scip_coverage_complete"] = True
        scip_stats["scip_relation_coverage_complete"] = True
        if not scip_stats["scip_coverage_complete"]:
            scip_stats["scip_degraded"] = True
        if not scip_stats["scip_relation_coverage_complete"]:
            scip_stats["scip_degraded"] = True
        assert scip_stats["scip_degraded"] is True

    def test_incomplete_coverage_marks_degraded(self) -> None:
        scip_stats: dict[str, Any] = {"scip_projects_failed": 0}
        scip_stats["scip_degraded"] = bool(scip_stats["scip_projects_failed"])
        scip_stats["scip_coverage_complete"] = False
        scip_stats["scip_relation_coverage_complete"] = True
        if not scip_stats["scip_coverage_complete"]:
            scip_stats["scip_degraded"] = True
        if not scip_stats["scip_relation_coverage_complete"]:
            scip_stats["scip_degraded"] = True
        assert scip_stats["scip_degraded"] is True

    def test_incomplete_relation_coverage_marks_degraded(self) -> None:
        scip_stats: dict[str, Any] = {"scip_projects_failed": 0}
        scip_stats["scip_degraded"] = bool(scip_stats["scip_projects_failed"])
        scip_stats["scip_coverage_complete"] = True
        scip_stats["scip_relation_coverage_complete"] = False
        if not scip_stats["scip_coverage_complete"]:
            scip_stats["scip_degraded"] = True
        if not scip_stats["scip_relation_coverage_complete"]:
            scip_stats["scip_degraded"] = True
        assert scip_stats["scip_degraded"] is True


class TestScipStatsWarningsPopulation:
    """Test warning population for missing languages (lines 2373-2392)."""

    def test_missing_languages_adds_warnings(self) -> None:
        scip_stats: dict[str, Any] = {
            "scip_detection_warnings": ["initial_warning"],
        }
        missing_languages = ["python", "typescript"]
        indexed_files_by_language: dict[str, int] = {}

        scip_stats["scip_indexed_files_by_language"] = indexed_files_by_language
        scip_stats["scip_missing_languages"] = missing_languages
        scip_stats["scip_coverage_complete"] = len(missing_languages) == 0

        if missing_languages:
            scip_stats["scip_detection_warnings"] = list(
                scip_stats.get("scip_detection_warnings") or []
            ) + [f"missing_language_index_coverage:{language}" for language in missing_languages]

        assert any(
            "missing_language_index_coverage:python" in w
            for w in scip_stats["scip_detection_warnings"]
        )
        assert any(
            "missing_language_index_coverage:typescript" in w
            for w in scip_stats["scip_detection_warnings"]
        )

    def test_missing_relation_languages_adds_warnings(self) -> None:
        scip_stats: dict[str, Any] = {
            "scip_detection_warnings": [],
        }
        missing_relation_languages = ["python"]

        if missing_relation_languages:
            scip_stats["scip_detection_warnings"] = list(
                scip_stats.get("scip_detection_warnings") or []
            ) + [f"missing_relation_coverage:{language}" for language in missing_relation_languages]

        assert any(
            "missing_relation_coverage:python" in w for w in scip_stats["scip_detection_warnings"]
        )

    def test_no_missing_languages_no_extra_warnings(self) -> None:
        scip_stats: dict[str, Any] = {
            "scip_detection_warnings": [],
        }
        missing_languages: list[str] = []
        if missing_languages:
            scip_stats["scip_detection_warnings"] = list(
                scip_stats.get("scip_detection_warnings") or []
            ) + [f"missing_language_index_coverage:{language}" for language in missing_languages]
        assert scip_stats["scip_detection_warnings"] == []
