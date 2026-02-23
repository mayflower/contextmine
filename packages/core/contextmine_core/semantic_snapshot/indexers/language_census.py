"""Language census utilities for SCIP project detection.

Primary signal is `cloc` JSON output. If `cloc` is unavailable, a deterministic
extension-based fallback is used with explicit warnings.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from contextmine_core.semantic_snapshot.models import Language

logger = logging.getLogger(__name__)

# Keep consistent with project detection traversal exclusions.
IGNORE_DIRS = {
    "node_modules",
    "vendor",
    ".git",
    "dist",
    "build",
    "target",
    ".venv",
    "venv",
    "__pycache__",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".eggs",
    "egg-info",
    ".gradle",
    ".m2",
    ".idea",
    ".vscode",
}

CLOC_TO_LANGUAGE: dict[str, Language] = {
    "python": Language.PYTHON,
    "typescript": Language.TYPESCRIPT,
    "javascript": Language.JAVASCRIPT,
    "java": Language.JAVA,
    "php": Language.PHP,
}

EXTENSION_TO_LANGUAGE: dict[str, Language] = {
    ".py": Language.PYTHON,
    ".ts": Language.TYPESCRIPT,
    ".tsx": Language.TYPESCRIPT,
    ".js": Language.JAVASCRIPT,
    ".jsx": Language.JAVASCRIPT,
    ".mjs": Language.JAVASCRIPT,
    ".cjs": Language.JAVASCRIPT,
    ".java": Language.JAVA,
    ".php": Language.PHP,
    ".phtml": Language.PHP,
    ".inc": Language.PHP,
}


@dataclass(frozen=True)
class LanguageFileStat:
    """Code statistics for one file."""

    path: Path
    language: Language
    code: int


@dataclass
class LanguageCensusEntry:
    """Aggregate statistics for one language."""

    language: Language
    files: int = 0
    code: int = 0
    comments: int = 0
    blanks: int = 0


@dataclass
class LanguageCensusReport:
    """Language census report with summary + file-level breakdown."""

    entries: dict[Language, LanguageCensusEntry] = field(default_factory=dict)
    file_stats: list[LanguageFileStat] = field(default_factory=list)
    tool_name: str = "cloc"
    tool_version: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def total_code(self) -> int:
        return sum(max(0, entry.code) for entry in self.entries.values())


def build_language_census(repo_root: Path | str) -> LanguageCensusReport:
    """Build language census for a repository.

    Uses `cloc` when available; falls back to extension-based counting when not.
    """
    root = Path(repo_root)
    report = _run_cloc(root)

    # Ensure we always have file-level stats for root localization.
    if not report.file_stats:
        fallback = _fallback_extension_census(root)
        if report.entries:
            # Keep cloc aggregate counts, but use fallback file detail for localization.
            report.file_stats = fallback.file_stats
            if fallback.warnings:
                report.warnings.extend(fallback.warnings)
        else:
            report = fallback

    # Ensure entries exist even if only file-level stats were available.
    if not report.entries:
        aggregated: dict[Language, LanguageCensusEntry] = {}
        for item in report.file_stats:
            entry = aggregated.setdefault(
                item.language, LanguageCensusEntry(language=item.language)
            )
            entry.files += 1
            entry.code += max(0, int(item.code))
        report.entries = aggregated

    return report


def _run_cloc(repo_root: Path) -> LanguageCensusReport:
    report = LanguageCensusReport()
    exclude_dirs = ",".join(sorted(IGNORE_DIRS))
    cloc_bin = shutil.which("cloc")
    if not cloc_bin:
        report.warnings.append("cloc_not_available")
        return report

    try:
        version_proc = subprocess.run(
            [cloc_bin, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if version_proc.returncode == 0:
            report.tool_version = (version_proc.stdout or version_proc.stderr).strip() or None

        summary_proc = subprocess.run(
            [
                cloc_bin,
                "--json",
                "--quiet",
                f"--exclude-dir={exclude_dirs}",
                str(repo_root),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if summary_proc.returncode != 0:
            msg = (summary_proc.stderr or summary_proc.stdout or "cloc failed").strip()
            report.warnings.append(f"cloc_summary_failed: {msg}")
            return report

        parsed = json.loads(summary_proc.stdout or "{}")
        report.entries = _parse_cloc_summary(parsed)

        by_file_proc = subprocess.run(
            [
                cloc_bin,
                "--json",
                "--by-file",
                "--quiet",
                f"--exclude-dir={exclude_dirs}",
                str(repo_root),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if by_file_proc.returncode == 0:
            by_file = json.loads(by_file_proc.stdout or "{}")
            report.file_stats = _parse_cloc_by_file(by_file, repo_root)
        else:
            msg = (by_file_proc.stderr or by_file_proc.stdout or "cloc --by-file failed").strip()
            report.warnings.append(f"cloc_by_file_failed: {msg}")

        return report
    except FileNotFoundError:
        report.warnings.append("cloc_not_available")
        return report
    except subprocess.TimeoutExpired:
        report.warnings.append("cloc_timeout")
        return report
    except Exception as exc:  # noqa: BLE001
        logger.warning("Language census via cloc failed: %s", exc)
        report.warnings.append(f"cloc_error: {exc}")
        return report


def _parse_cloc_summary(parsed: dict) -> dict[Language, LanguageCensusEntry]:
    entries: dict[Language, LanguageCensusEntry] = {}
    for key, value in parsed.items():
        if key in {"header", "SUM"}:
            continue
        if not isinstance(value, dict):
            continue
        language = _normalize_language_name(str(key))
        if language is None:
            continue
        entry = entries.setdefault(language, LanguageCensusEntry(language=language))
        entry.files += int(value.get("nFiles") or 0)
        entry.code += int(value.get("code") or 0)
        entry.comments += int(value.get("comment") or 0)
        entry.blanks += int(value.get("blank") or 0)
    return entries


def _parse_cloc_by_file(parsed: dict, repo_root: Path) -> list[LanguageFileStat]:
    items: list[LanguageFileStat] = []
    for key, value in parsed.items():
        if key in {"header", "SUM"}:
            continue
        if not isinstance(value, dict):
            continue
        raw_language = str(value.get("language") or "").strip()
        language = _normalize_language_name(raw_language)
        if language is None:
            # Some cloc versions encode language in the key for --by-file.
            language = _language_from_extension(Path(key).suffix.lower())
            if language is None:
                continue
        code = int(value.get("code") or 0)

        path = Path(key)
        if not path.is_absolute():
            path = (repo_root / path).resolve()

        items.append(LanguageFileStat(path=path, language=language, code=max(0, code)))
    return items


def _fallback_extension_census(repo_root: Path) -> LanguageCensusReport:
    report = LanguageCensusReport(
        tool_name="extension-fallback",
        warnings=["cloc_unavailable_or_unusable_using_extension_fallback"],
    )
    entries: dict[Language, LanguageCensusEntry] = {}
    file_stats: list[LanguageFileStat] = []

    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(repo_root).parts
        if any(part in IGNORE_DIRS or part.startswith(".") for part in rel_parts):
            continue

        language = _language_from_extension(path.suffix.lower())
        if language is None:
            continue

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        # Approximation when cloc is unavailable: non-empty line count.
        code = sum(1 for line in content.splitlines() if line.strip())
        file_stats.append(LanguageFileStat(path=path.resolve(), language=language, code=code))

        entry = entries.setdefault(language, LanguageCensusEntry(language=language))
        entry.files += 1
        entry.code += code

    report.entries = entries
    report.file_stats = file_stats
    return report


def _normalize_language_name(raw: str) -> Language | None:
    key = raw.strip().lower()
    if not key:
        return None
    return CLOC_TO_LANGUAGE.get(key)


def _language_from_extension(ext: str) -> Language | None:
    return EXTENSION_TO_LANGUAGE.get(ext)
