"""Tests for language census parsing robustness."""

import json
import re
from pathlib import Path

from contextmine_core.semantic_snapshot.indexers.language_census import (
    LanguageCensusReport,
    _build_not_match_dir_regex,
    _fallback_extension_census,
    _load_cloc_json,
)
from contextmine_core.semantic_snapshot.models import Language


def test_load_cloc_json_handles_trailing_output() -> None:
    """Parse primary JSON payload and ignore trailing non-JSON output."""
    raw = """
{"header":{"cloc_url":"https://github.com/AlDanial/cloc"},"Python":{"nFiles":2,"blank":1,"comment":0,"code":42}}
WARNING: some trailing tool output
""".strip()
    report = LanguageCensusReport()

    parsed = _load_cloc_json(raw, report, "cloc_summary")

    assert parsed.get("Python", {}).get("code") == 42
    assert "cloc_summary_trailing_output_ignored" in report.warnings


def test_fallback_extension_census_ignores_custom_composer_vendor_dir(
    tmp_path: Path,
) -> None:
    (tmp_path / "composer.json").write_text(
        json.dumps({"name": "root/app", "config": {"vendor-dir": "src/libs"}}),
        encoding="utf-8",
    )
    (tmp_path / "src" / "libs").mkdir(parents=True)
    (tmp_path / "src" / "libs" / "thirdparty.php").write_text(
        "<?php\necho 'vendor';\n",
        encoding="utf-8",
    )
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "main.php").write_text(
        "<?php\necho 'app';\n",
        encoding="utf-8",
    )

    report = _fallback_extension_census(tmp_path)

    assert report.entries[Language.PHP].files == 1
    assert any(item.path.name == "main.php" for item in report.file_stats)
    assert all(item.path.name != "thirdparty.php" for item in report.file_stats)


def test_build_not_match_dir_regex_matches_nested_path_prefix() -> None:
    regex = _build_not_match_dir_regex({Path("src/libs"), Path("phpmyfaq/src/libs")})

    assert regex is not None
    assert re.search(regex, "/tmp/repo/src/libs") is not None
    assert re.search(regex, "/tmp/repo/phpmyfaq/src/libs/sub") is not None
    assert re.search(regex, "/tmp/repo/src/libstuff") is None
