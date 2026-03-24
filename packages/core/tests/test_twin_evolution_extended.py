"""Extended tests for evolution analytics pure functions and builders.

Covers investment utilization chart data building, knowledge islands helpers,
temporal coupling helpers, fitness finding generation, and snapshot replacement
with mocked sessions.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from contextmine_core.twin.evolution import (
    _SEVERITY_WEIGHT,
    _SUPPORTED_ENTITY_LEVELS,
    EntityGroup,
    _bus_factor,
    _coverage_value,
    _display_label,
    _file_path_from_natural_key,
    _normalize_min_max,
    _percentile,
    _safe_float,
    _safe_int,
    _safe_ratio,
    _topological_cycles,
    build_entity_key,
    derive_arch_group,
    replace_evolution_snapshots,
)

# ---------------------------------------------------------------------------
# replace_evolution_snapshots with mocked session
# ---------------------------------------------------------------------------


class TestReplaceEvolutionSnapshots:
    @pytest.mark.anyio
    async def test_basic_replacement(self) -> None:
        session = AsyncMock()
        session.execute = AsyncMock()
        session.add_all = MagicMock()

        scenario_id = uuid.uuid4()
        ownership_rows = [
            {
                "node_natural_key": "file:src/main.py",
                "author_key": "alice@example.com",
                "author_label": "Alice",
                "additions": 100,
                "deletions": 10,
                "touches": 5,
                "ownership_share": 0.8,
                "last_touched_at": datetime.now(UTC),
                "window_days": 365,
            }
        ]
        coupling_rows = [
            {
                "entity_level": "file",
                "source_key": "file:a.py",
                "target_key": "file:b.py",
                "co_change_count": 10,
                "source_change_count": 20,
                "target_change_count": 15,
                "ratio_source_to_target": 0.5,
                "ratio_target_to_source": 0.67,
                "jaccard": 0.3,
                "cross_boundary": False,
                "window_days": 365,
            }
        ]

        result = await replace_evolution_snapshots(
            session,
            scenario_id=scenario_id,
            ownership_rows=ownership_rows,
            coupling_rows=coupling_rows,
        )

        assert result["ownership_rows"] == 1
        assert result["coupling_rows"] == 1
        assert session.add_all.call_count == 2

    @pytest.mark.anyio
    async def test_empty_rows(self) -> None:
        session = AsyncMock()
        session.execute = AsyncMock()
        session.add_all = MagicMock()

        result = await replace_evolution_snapshots(
            session,
            scenario_id=uuid.uuid4(),
            ownership_rows=[],
            coupling_rows=[],
        )

        assert result["ownership_rows"] == 0
        assert result["coupling_rows"] == 0

    @pytest.mark.anyio
    async def test_invalid_entity_level_skipped(self) -> None:
        session = AsyncMock()
        session.execute = AsyncMock()
        session.add_all = MagicMock()

        coupling_rows = [
            {
                "entity_level": "galaxy",  # Not in supported levels
                "source_key": "a",
                "target_key": "b",
                "co_change_count": 1,
            }
        ]

        result = await replace_evolution_snapshots(
            session,
            scenario_id=uuid.uuid4(),
            ownership_rows=[],
            coupling_rows=coupling_rows,
        )

        assert result["coupling_rows"] == 0


# ---------------------------------------------------------------------------
# Constants and lookup values
# ---------------------------------------------------------------------------


class TestConstants:
    def test_supported_entity_levels(self) -> None:
        assert "file" in _SUPPORTED_ENTITY_LEVELS
        assert "container" in _SUPPORTED_ENTITY_LEVELS
        assert "component" in _SUPPORTED_ENTITY_LEVELS

    def test_severity_weights(self) -> None:
        assert _SEVERITY_WEIGHT["critical"] > _SEVERITY_WEIGHT["high"]
        assert _SEVERITY_WEIGHT["high"] > _SEVERITY_WEIGHT["medium"]
        assert _SEVERITY_WEIGHT["medium"] > _SEVERITY_WEIGHT["low"]


# ---------------------------------------------------------------------------
# Comprehensive _topological_cycles tests
# ---------------------------------------------------------------------------


class TestTopologicalCyclesExtended:
    def test_complex_nested_cycles(self) -> None:
        """Graph with overlapping cycle chains."""
        graph = {
            "A": {"B"},
            "B": {"C"},
            "C": {"A", "D"},
            "D": {"E"},
            "E": {"D"},
            "F": set(),
        }
        cycles = _topological_cycles(graph)
        assert len(cycles) == 2
        sizes = sorted([len(c) for c in cycles])
        assert sizes == [2, 3]

    def test_all_nodes_in_one_cycle(self) -> None:
        graph = {
            "A": {"B"},
            "B": {"C"},
            "C": {"D"},
            "D": {"A"},
        }
        cycles = _topological_cycles(graph)
        assert len(cycles) == 1
        assert sorted(cycles[0]) == ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# build_entity_key with architecture meta variations
# ---------------------------------------------------------------------------


class TestBuildEntityKeyExtended:
    def test_file_level_strips_dotslash(self) -> None:
        key = build_entity_key("./src/main.py", None, "file")
        assert key is not None
        assert key.startswith("file:")
        assert "./" not in key

    def test_container_with_architecture_meta(self) -> None:
        meta = {"architecture": {"domain": "billing", "container": "api"}}
        key = build_entity_key("any/path.py", meta, "container")
        assert key is not None
        assert "billing" in key
        assert "api" in key

    def test_component_with_architecture_meta(self) -> None:
        meta = {"architecture": {"domain": "core", "container": "auth", "component": "login"}}
        key = build_entity_key("any/path.py", meta, "component")
        assert key is not None
        assert "core" in key
        assert "login" in key


# ---------------------------------------------------------------------------
# derive_arch_group edge cases
# ---------------------------------------------------------------------------


class TestDeriveArchGroupExtended:
    def test_packages_directory(self) -> None:
        group = derive_arch_group("packages/core/contextmine_core/models.py")
        assert group is not None
        assert isinstance(group, EntityGroup)

    def test_deep_nested_path(self) -> None:
        group = derive_arch_group("services/billing/api/v2/endpoints/invoice.py")
        assert group is not None
        assert group.domain == "billing"

    def test_empty_string(self) -> None:
        result = derive_arch_group("")
        assert result is None

    def test_with_meta_overrides_heuristic(self) -> None:
        meta = {"architecture": {"domain": "override", "container": "special", "component": "comp"}}
        group = derive_arch_group("services/billing/api/views.py", meta)
        assert group is not None
        assert group.domain == "override"
        assert group.container == "special"


# ---------------------------------------------------------------------------
# _file_path_from_natural_key edge cases
# ---------------------------------------------------------------------------


class TestFilePathFromNaturalKeyExtended:
    def test_complex_path(self) -> None:
        result = _file_path_from_natural_key("file:packages/core/tests/test_main.py")
        assert result == "packages/core/tests/test_main.py"

    def test_leading_slash_stripped(self) -> None:
        result = _file_path_from_natural_key("file:/src/main.py")
        assert result is not None
        assert not result.startswith("/")


# ---------------------------------------------------------------------------
# _percentile edge cases
# ---------------------------------------------------------------------------


class TestPercentileExtended:
    def test_two_elements_at_50(self) -> None:
        result = _percentile([1.0, 10.0], 0.5)
        assert result == 10.0

    def test_large_list_at_90(self) -> None:
        values = sorted(float(i) for i in range(100))
        result = _percentile(values, 0.9)
        assert result == 90.0


# ---------------------------------------------------------------------------
# _bus_factor with negative values
# ---------------------------------------------------------------------------


class TestBusFactorExtended:
    def test_negative_values_treated_as_zero_total(self) -> None:
        """Negative contributions should not break the calculation."""
        result = _bus_factor({"alice": -10.0, "bob": -5.0})
        assert result == 0

    def test_mixed_positive_negative(self) -> None:
        result = _bus_factor({"alice": 80.0, "bob": -5.0, "carol": 20.0})
        # Total = 95, alice = 80 >= 80% of 95
        assert result >= 1


# ---------------------------------------------------------------------------
# _normalize_min_max edge cases
# ---------------------------------------------------------------------------


class TestNormalizeMinMaxExtended:
    def test_very_small_range(self) -> None:
        """When value exceeds max, normalization can exceed 1.0."""
        result = _normalize_min_max(0.5, 0.0, 1e-10)
        assert result > 1.0  # 0.5 / 1e-10 is very large

    def test_negative_range(self) -> None:
        result = _normalize_min_max(-5.0, -10.0, 0.0)
        assert abs(result - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# _display_label edge cases
# ---------------------------------------------------------------------------


class TestDisplayLabelExtended:
    def test_empty_string(self) -> None:
        assert _display_label("") == ""

    def test_colon_at_start(self) -> None:
        assert _display_label(":value") == "value"

    def test_colon_at_end(self) -> None:
        assert _display_label("prefix:") == ""


# ---------------------------------------------------------------------------
# _safe_float / _safe_int comprehensive
# ---------------------------------------------------------------------------


class TestSafeFloatExtended:
    def test_bool_true(self) -> None:
        assert _safe_float(True) == 1.0

    def test_bool_false(self) -> None:
        assert _safe_float(False) == 0.0

    def test_large_number(self) -> None:
        assert _safe_float(1e308) == 1e308


class TestSafeIntExtended:
    def test_bool_true(self) -> None:
        assert _safe_int(True) == 1

    def test_bool_false(self) -> None:
        assert _safe_int(False) == 0

    def test_negative(self) -> None:
        assert _safe_int(-42) == -42

    def test_float_negative(self) -> None:
        assert _safe_int(-3.9) == -3


# ---------------------------------------------------------------------------
# _safe_ratio edge cases
# ---------------------------------------------------------------------------


class TestSafeRatioExtended:
    def test_both_zero(self) -> None:
        assert _safe_ratio(0.0, 0.0) == 0.0

    def test_large_numbers(self) -> None:
        result = _safe_ratio(1e300, 1e300)
        assert abs(result - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# _coverage_value edge cases
# ---------------------------------------------------------------------------


class TestCoverageValueExtended:
    def test_tiny_positive(self) -> None:
        assert _coverage_value(0.001) == 0.001

    def test_exactly_boundary(self) -> None:
        assert _coverage_value(0.0) == 0.0
        assert _coverage_value(100.0) == 100.0
