"""Tests for language census parsing robustness."""

from contextmine_core.semantic_snapshot.indexers.language_census import (
    LanguageCensusReport,
    _load_cloc_json,
)


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
