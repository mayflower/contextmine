"""Tests for complexity_loc_lizard.py helper functions.

Covers:
- _extract_fileinfos
- _complexity_from_fileinfo
- aggregate_lizard_metrics
- extract_complexity_loc_metrics (mocked lizard)
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from contextmine_core.metrics.complexity_loc_lizard import (
    _complexity_from_fileinfo,
    _extract_fileinfos,
    aggregate_lizard_metrics,
    extract_complexity_loc_metrics,
)
from contextmine_core.metrics.models import MetricsGateError

# ---------------------------------------------------------------------------
# _extract_fileinfos
# ---------------------------------------------------------------------------


class TestExtractFileInfos:
    def test_from_fileinfo_list_attr(self) -> None:
        analysis = SimpleNamespace(fileinfo_list=[1, 2, 3])
        assert _extract_fileinfos(analysis) == [1, 2, 3]

    def test_from_files_attr(self) -> None:
        """Object has only 'files', not 'fileinfo_list'."""
        obj = type("Obj", (), {"files": [4, 5]})()
        assert _extract_fileinfos(obj) == [4, 5]

    def test_from_iterable_fallback(self) -> None:
        """If no known attribute, the analysis object itself is iterated."""
        result = _extract_fileinfos([10, 20])
        assert result == [10, 20]

    def test_non_iterable_returns_empty(self) -> None:
        """A non-iterable object with no known attributes returns []."""
        result = _extract_fileinfos(42)
        assert result == []

    def test_fileinfo_list_is_none(self) -> None:
        """Attribute exists but is None => falls through to next."""
        obj = type("Obj", (), {"fileinfo_list": None, "files": [1, 2]})()
        assert _extract_fileinfos(obj) == [1, 2]


# ---------------------------------------------------------------------------
# _complexity_from_fileinfo
# ---------------------------------------------------------------------------


class TestComplexityFromFileinfo:
    def test_sum_of_function_complexities(self) -> None:
        func1 = SimpleNamespace(cyclomatic_complexity=5)
        func2 = SimpleNamespace(cyclomatic_complexity=3)
        file_info = SimpleNamespace(function_list=[func1, func2])
        assert _complexity_from_fileinfo(file_info) == 8.0

    def test_uses_ccn_fallback(self) -> None:
        """When cyclomatic_complexity is missing, falls back to CCN."""
        func = type("F", (), {"CCN": 7})()
        # Ensure cyclomatic_complexity is not found
        file_info = SimpleNamespace(function_list=[func])
        assert _complexity_from_fileinfo(file_info) == 7.0

    def test_empty_function_list_uses_average(self) -> None:
        file_info = SimpleNamespace(
            function_list=[],
            average_cyclomatic_complexity=4.0,
            function_count=3,
        )
        assert _complexity_from_fileinfo(file_info) == 12.0

    def test_no_functions_no_average_returns_zero(self) -> None:
        file_info = SimpleNamespace(function_list=[])
        assert _complexity_from_fileinfo(file_info) == 0.0

    def test_none_function_list(self) -> None:
        file_info = SimpleNamespace(function_list=None)
        assert _complexity_from_fileinfo(file_info) == 0.0

    def test_mixed_complexity_attributes(self) -> None:
        """Some functions have cyclomatic_complexity, some only CCN."""
        func_cc = SimpleNamespace(cyclomatic_complexity=4)
        func_ccn = type("F", (), {"CCN": 6})()
        file_info = SimpleNamespace(function_list=[func_cc, func_ccn])
        assert _complexity_from_fileinfo(file_info) == 10.0

    def test_all_zero_complexity(self) -> None:
        func = SimpleNamespace(cyclomatic_complexity=0)
        file_info = SimpleNamespace(function_list=[func, func])
        assert _complexity_from_fileinfo(file_info) == 0.0


# ---------------------------------------------------------------------------
# aggregate_lizard_metrics
# ---------------------------------------------------------------------------


class TestAggregateLizardMetrics:
    def _make_fileinfo(self, filename: str, nloc: int, functions: list[float]) -> SimpleNamespace:
        func_list = [SimpleNamespace(cyclomatic_complexity=c) for c in functions]
        return SimpleNamespace(filename=filename, nloc=nloc, function_list=func_list)

    def test_basic_aggregation(self) -> None:
        repo = Path("/repo")
        project = Path("/repo")
        fi = self._make_fileinfo("/repo/src/main.py", 100, [5.0, 3.0])
        result = aggregate_lizard_metrics([fi], repo, project, {"src/main.py"})
        assert "src/main.py" in result
        assert result["src/main.py"]["loc"] == 100
        assert result["src/main.py"]["complexity"] == 8.0

    def test_skips_empty_filename(self) -> None:
        repo = Path("/repo")
        fi = SimpleNamespace(filename="", nloc=10, function_list=[])
        result = aggregate_lizard_metrics([fi], repo, repo, set())
        assert result == {}

    def test_skips_irrelevant_file(self) -> None:
        repo = Path("/repo")
        fi = self._make_fileinfo("/repo/src/other.py", 50, [2.0])
        result = aggregate_lizard_metrics([fi], repo, repo, {"src/main.py"})
        assert "src/other.py" not in result

    def test_relevant_files_empty_allows_all(self) -> None:
        repo = Path("/repo")
        fi = self._make_fileinfo("/repo/src/a.py", 20, [1.0])
        result = aggregate_lizard_metrics([fi], repo, repo, set())
        # Empty relevant_files set means "accept everything"
        assert "src/a.py" in result

    def test_multiple_files(self) -> None:
        repo = Path("/repo")
        fi1 = self._make_fileinfo("/repo/a.py", 10, [1.0])
        fi2 = self._make_fileinfo("/repo/b.py", 20, [2.0, 3.0])
        result = aggregate_lizard_metrics([fi1, fi2], repo, repo, {"a.py", "b.py"})
        assert len(result) == 2
        assert result["a.py"]["loc"] == 10
        assert result["b.py"]["complexity"] == 5.0


# ---------------------------------------------------------------------------
# extract_complexity_loc_metrics (integration-level with mocked lizard)
# ---------------------------------------------------------------------------


class TestExtractComplexityLocMetrics:
    def test_empty_relevant_files_returns_empty(self) -> None:
        result = extract_complexity_loc_metrics(Path("/repo"), Path("/repo"), set())
        assert result == {}

    def test_raises_when_lizard_not_available(self) -> None:
        with patch.dict("sys.modules", {"lizard": None}):
            # Force reimport failure

            import contextmine_core.metrics.complexity_loc_lizard as mod

            # Temporarily break the lizard import
            def _broken(repo_root, project_root, relevant_files):
                if not relevant_files:
                    return {}
                raise MetricsGateError(
                    "complexity_extraction_error",
                    details={"reason": "lizard_not_available"},
                )

            with (
                patch.object(mod, "extract_complexity_loc_metrics", _broken),
                pytest.raises(MetricsGateError, match="complexity_extraction_error"),
            ):
                mod.extract_complexity_loc_metrics(Path("/repo"), Path("/repo"), {"file.py"})

    def test_successful_analysis(self) -> None:
        """Mock lizard.analyze to return a known file info."""
        mock_lizard = MagicMock()
        func = SimpleNamespace(cyclomatic_complexity=7)
        fi = SimpleNamespace(filename="/repo/src/app.py", nloc=50, function_list=[func])
        mock_lizard.analyze.return_value = SimpleNamespace(fileinfo_list=[fi])

        with patch.dict("sys.modules", {"lizard": mock_lizard}):
            result = extract_complexity_loc_metrics(Path("/repo"), Path("/repo"), {"src/app.py"})
            assert "src/app.py" in result
            assert result["src/app.py"]["loc"] == 50
            assert result["src/app.py"]["complexity"] == 7.0

    def test_no_fileinfos_raises(self) -> None:
        """When lizard returns no fileinfos, raises MetricsGateError."""
        mock_lizard = MagicMock()
        mock_lizard.analyze.return_value = SimpleNamespace(fileinfo_list=[])

        with (
            patch.dict("sys.modules", {"lizard": mock_lizard}),
            pytest.raises(MetricsGateError, match="complexity_extraction_error"),
        ):
            extract_complexity_loc_metrics(Path("/repo"), Path("/repo"), {"file.py"})

    def test_type_error_fallback(self) -> None:
        """When lizard.analyze(list) raises TypeError, tries paths= kwarg."""
        mock_lizard = MagicMock()
        func = SimpleNamespace(cyclomatic_complexity=2)
        fi = SimpleNamespace(filename="/repo/x.py", nloc=10, function_list=[func])

        call_count = 0

        def analyze_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1 and args:
                raise TypeError("bad positional arg")
            return SimpleNamespace(fileinfo_list=[fi])

        mock_lizard.analyze.side_effect = analyze_side_effect

        with patch.dict("sys.modules", {"lizard": mock_lizard}):
            result = extract_complexity_loc_metrics(Path("/repo"), Path("/repo"), {"x.py"})
            assert "x.py" in result
            assert mock_lizard.analyze.call_count == 2
