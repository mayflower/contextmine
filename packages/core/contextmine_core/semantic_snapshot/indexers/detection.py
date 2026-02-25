"""Project detection for SCIP indexing.

Language identification is census-first (cloc), marker-assisted for root
localization, and supports multiple languages per repository root.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from contextmine_core.semantic_snapshot.indexers.language_census import (
    IGNORE_DIRS,
    LanguageCensusEntry,
    LanguageCensusReport,
    build_language_census,
)
from contextmine_core.semantic_snapshot.models import Language, ProjectTarget

logger = logging.getLogger(__name__)

# Production thresholds for medium/large repositories.
MIN_CODE_LINES = 300
MIN_FILE_COUNT = 8
MIN_CODE_SHARE = 0.10

# Relax thresholds for tiny repos/fixtures to keep local tests ergonomic.
SMALL_REPO_TOTAL_CODE = 2000


@dataclass
class DetectionDiagnostics:
    """Structured diagnostics for language/project detection."""

    census_tool: str
    census_tool_version: str | None
    total_code: int
    languages_detected: list[str] = field(default_factory=list)
    projects_by_language: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "census_tool": self.census_tool,
            "census_tool_version": self.census_tool_version,
            "total_code": self.total_code,
            "languages_detected": self.languages_detected,
            "projects_by_language": self.projects_by_language,
            "warnings": self.warnings,
        }


def detect_projects(repo_root: Path | str) -> list[ProjectTarget]:
    """Detect all indexable projects in a repository.

    - Language presence is determined from language census (cloc-first).
    - Marker files are used only to improve root localization.
    - Multiple languages in one repo are always supported.
    """
    projects, _ = detect_projects_with_diagnostics(repo_root)
    return projects


def detect_projects_with_diagnostics(
    repo_root: Path | str,
) -> tuple[list[ProjectTarget], DetectionDiagnostics]:
    """Detect projects and emit structured diagnostics."""
    root = Path(repo_root).resolve()
    report = build_language_census(root)

    qualified = _qualify_languages(report)
    if report.total_code < SMALL_REPO_TOTAL_CODE:
        # Tiny repos/fixtures may not have enough code lines for census thresholds
        # but still provide explicit build markers.
        qualified.update(_marker_languages(root))

    projects: list[ProjectTarget] = []
    warnings = list(report.warnings)

    for language in sorted(qualified, key=lambda item: item.value):
        entry = report.entries.get(language)
        roots, root_reason = _infer_roots_for_language(
            repo_root=root,
            language=language,
            report=report,
        )
        if not roots:
            roots = [root]
            root_reason = "repo_fallback"

        for candidate_root in roots:
            marker_meta = _language_marker_metadata(candidate_root, language)
            metadata = {
                **marker_meta,
                "detection_basis": "cloc" if report.tool_name == "cloc" else report.tool_name,
                "root_reason": root_reason,
                "confidence": _confidence_for_language(entry, report.total_code),
                "census_code_lines": int(entry.code if entry else 0),
                "census_file_count": int(entry.files if entry else 0),
                "census_total_code_lines": int(report.total_code),
            }
            projects.append(
                ProjectTarget(
                    language=language,
                    root_path=candidate_root,
                    metadata=metadata,
                )
            )

    projects = _dedupe_projects(projects)
    projects.sort(key=lambda p: (len(p.root_path.parts), p.language.value), reverse=True)

    dominant = _dominant_language(report)
    if dominant is not None and dominant not in {p.language for p in projects}:
        warnings.append(f"dominant_language_not_indexed:{dominant.value}")

    projects_by_language: dict[str, int] = {}
    for project in projects:
        projects_by_language[project.language.value] = (
            projects_by_language.get(project.language.value, 0) + 1
        )

    diagnostics = DetectionDiagnostics(
        census_tool=report.tool_name,
        census_tool_version=report.tool_version,
        total_code=report.total_code,
        languages_detected=sorted(item.value for item in qualified),
        projects_by_language=projects_by_language,
        warnings=sorted(set(warnings)),
    )

    return projects, diagnostics


def detect_project_at(directory: Path) -> ProjectTarget | None:
    """Backward-compatible single-project detector for one directory.

    Returns the first detected project for the directory scope.
    """
    projects, _ = detect_projects_with_diagnostics(directory)
    return projects[0] if projects else None


def _qualify_languages(report: LanguageCensusReport) -> set[Language]:
    qualified: set[Language] = set()
    total_code = max(0, int(report.total_code))

    if total_code < SMALL_REPO_TOTAL_CODE:
        min_code = 1
        min_files = 1
    else:
        min_code = MIN_CODE_LINES
        min_files = MIN_FILE_COUNT

    for language, entry in report.entries.items():
        code = max(0, int(entry.code))
        files = max(0, int(entry.files))
        share = (code / total_code) if total_code > 0 else 0.0

        if code >= min_code or files >= min_files or share >= MIN_CODE_SHARE:
            qualified.add(language)

    return qualified


def _marker_languages(repo_root: Path) -> set[Language]:
    languages: set[Language] = set()
    for language in Language:
        if _find_marker_roots(repo_root, language):
            languages.add(language)
    return languages


def _dominant_language(report: LanguageCensusReport) -> Language | None:
    total = max(0, int(report.total_code))
    if total <= 0:
        return None

    leader: tuple[Language, int] | None = None
    for language, entry in report.entries.items():
        code = max(0, int(entry.code))
        if leader is None or code > leader[1]:
            leader = (language, code)

    if leader is None:
        return None

    language, code = leader
    share = code / total
    if code >= max(MIN_CODE_LINES, 100) and share >= 0.40:
        return language
    return None


def _infer_roots_for_language(
    *,
    repo_root: Path,
    language: Language,
    report: LanguageCensusReport,
) -> tuple[list[Path], str]:
    marker_roots = _find_marker_roots(repo_root, language)
    if marker_roots:
        # Keep marker roots that actually contain code for this language where possible.
        filtered = [root for root in marker_roots if _code_under_root(report, language, root) > 0]
        if filtered:
            return filtered, "marker_assisted"
        return marker_roots, "marker_assisted"

    # Marker-free fallback: infer a clustered root from census file paths.
    clustered = _cluster_root_from_files(report, language, repo_root)
    if clustered is not None:
        return [clustered], "clustered_files"

    return [repo_root], "repo_fallback"


def _cluster_root_from_files(
    report: LanguageCensusReport,
    language: Language,
    repo_root: Path,
) -> Path | None:
    file_stats = [item for item in report.file_stats if item.language == language and item.code > 0]
    if not file_stats:
        return None

    totals_by_top_dir: dict[str, int] = {}
    total_code = 0
    for item in file_stats:
        try:
            relative = item.path.relative_to(repo_root)
        except ValueError:
            continue
        if not relative.parts:
            continue
        top = relative.parts[0]
        totals_by_top_dir[top] = totals_by_top_dir.get(top, 0) + item.code
        total_code += item.code

    if not totals_by_top_dir or total_code <= 0:
        return None

    top_dir, top_code = max(totals_by_top_dir.items(), key=lambda kv: kv[1])
    share = top_code / total_code

    # Require a meaningful concentration before selecting sub-root.
    if share < 0.80:
        return None

    candidate = (repo_root / top_dir).resolve()
    if not candidate.exists() or not candidate.is_dir():
        return None

    # Avoid shrinking to generic directories when root markers exist at repo root.
    if _language_has_root_marker(repo_root, language):
        return repo_root

    return candidate


def _code_under_root(report: LanguageCensusReport, language: Language, root: Path) -> int:
    total = 0
    root = root.resolve()
    for item in report.file_stats:
        if item.language != language:
            continue
        try:
            item.path.relative_to(root)
        except ValueError:
            continue
        total += max(0, item.code)
    return total


def _find_marker_roots(repo_root: Path, language: Language) -> list[Path]:
    roots: set[Path] = set()
    skipped_paths: set[Path] = set()

    def should_skip(path: Path) -> bool:
        if path.name in IGNORE_DIRS or path.name.startswith("."):
            return True
        resolved = path.resolve()
        return any(
            resolved == skipped or resolved.is_relative_to(skipped) for skipped in skipped_paths
        )

    stack = [repo_root]
    while stack:
        current = stack.pop()
        if should_skip(current):
            continue

        try:
            children = list(current.iterdir())
        except PermissionError:
            logger.debug("Permission denied: %s", current)
            continue

        file_names = {child.name for child in children if child.is_file()}
        if "composer.json" in file_names:
            for vendor_dir in _composer_vendor_dirs(current):
                vendor_path = (current / vendor_dir).resolve()
                if vendor_path.exists() and vendor_path.is_dir():
                    skipped_paths.add(vendor_path)

        if _directory_matches_language_markers(file_names, language):
            roots.add(current.resolve())

        for child in children:
            if child.is_dir() and not should_skip(child):
                stack.append(child)

    return sorted(roots, key=lambda p: len(p.parts), reverse=True)


def _composer_vendor_dirs(project_root: Path) -> set[Path]:
    """Resolve Composer vendor directories declared in composer.json."""
    composer_json = project_root / "composer.json"
    try:
        payload = json.loads(composer_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {Path("vendor")}

    config = payload.get("config")
    if not isinstance(config, dict):
        return {Path("vendor")}

    vendor_dir = str(config.get("vendor-dir") or "vendor").strip()
    if not vendor_dir:
        vendor_dir = "vendor"
    return {Path(vendor_dir)}


def _language_has_root_marker(repo_root: Path, language: Language) -> bool:
    try:
        file_names = {child.name for child in repo_root.iterdir() if child.is_file()}
    except OSError:
        return False
    return _directory_matches_language_markers(file_names, language)


def _directory_matches_language_markers(file_names: set[str], language: Language) -> bool:
    if language == Language.PYTHON:
        return any(
            name in file_names
            for name in {"pyproject.toml", "setup.cfg", "setup.py", "requirements.txt"}
        )
    if language == Language.TYPESCRIPT:
        return "tsconfig.json" in file_names
    if language == Language.JAVASCRIPT:
        return "package.json" in file_names and "tsconfig.json" not in file_names
    if language == Language.JAVA:
        return any(
            name in file_names
            for name in {"pom.xml", "build.gradle", "build.gradle.kts", "build.sbt"}
        )
    if language == Language.PHP:
        return "composer.json" in file_names
    return False


def _language_marker_metadata(root: Path, language: Language) -> dict[str, object]:
    if language == Language.PYTHON:
        return {
            "has_pyproject": (root / "pyproject.toml").exists(),
            "has_setup_cfg": (root / "setup.cfg").exists(),
            "has_setup_py": (root / "setup.py").exists(),
            "has_requirements": (root / "requirements.txt").exists(),
        }

    if language in {Language.TYPESCRIPT, Language.JAVASCRIPT}:
        package_manager = "npm"
        if (root / "pnpm-lock.yaml").exists():
            package_manager = "pnpm"
        elif (root / "yarn.lock").exists():
            package_manager = "yarn"
        elif (root / "bun.lockb").exists():
            package_manager = "bun"

        return {
            "has_tsconfig": (root / "tsconfig.json").exists(),
            "package_manager": package_manager,
        }

    if language == Language.JAVA:
        has_pom = (root / "pom.xml").exists()
        has_gradle = (root / "build.gradle").exists()
        has_gradle_kts = (root / "build.gradle.kts").exists()
        has_sbt = (root / "build.sbt").exists()

        if has_pom:
            build_tool = "maven"
        elif has_gradle or has_gradle_kts:
            build_tool = "gradle"
        elif has_sbt:
            build_tool = "sbt"
        else:
            build_tool = "unknown"

        return {
            "build_tool": build_tool,
            "has_pom": has_pom,
            "has_gradle": has_gradle,
            "has_gradle_kts": has_gradle_kts,
            "has_sbt": has_sbt,
        }

    if language == Language.PHP:
        return {
            "has_composer_json": (root / "composer.json").exists(),
            "has_composer_lock": (root / "composer.lock").exists(),
        }

    return {}


def _confidence_for_language(
    entry: LanguageCensusEntry | None,
    total_code: int,
) -> float:
    if entry is None:
        return 0.0

    code = max(0, int(entry.code))
    files = max(0, int(entry.files))
    share = (code / total_code) if total_code > 0 else 0.0

    confidence = 0.35
    if code >= MIN_CODE_LINES:
        confidence += 0.30
    if files >= MIN_FILE_COUNT:
        confidence += 0.15
    confidence += min(share, 1.0) * 0.20
    return round(min(confidence, 0.99), 4)


def _dedupe_projects(projects: list[ProjectTarget]) -> list[ProjectTarget]:
    seen: set[tuple[Language, Path]] = set()
    deduped: list[ProjectTarget] = []
    for project in projects:
        key = (project.language, project.root_path.resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(project)
    return deduped
