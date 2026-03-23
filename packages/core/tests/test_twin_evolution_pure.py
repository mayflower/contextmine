"""Tests for pure/utility functions in contextmine_core.twin.evolution.

These tests cover EntityGroup, helper functions, fitness calculations,
and topology analysis without requiring a database.
"""

from __future__ import annotations

import pytest
from contextmine_core.twin.evolution import (
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
)

# ---------------------------------------------------------------------------
# EntityGroup dataclass
# ---------------------------------------------------------------------------


class TestEntityGroup:
    """Test the frozen EntityGroup dataclass."""

    def test_creation(self) -> None:
        group = EntityGroup(domain="billing", container="api", component="invoice")
        assert group.domain == "billing"
        assert group.container == "api"
        assert group.component == "invoice"

    def test_frozen(self) -> None:
        group = EntityGroup(domain="a", container="b", component="c")
        with pytest.raises(AttributeError):
            group.domain = "x"

    def test_equality(self) -> None:
        g1 = EntityGroup(domain="a", container="b", component="c")
        g2 = EntityGroup(domain="a", container="b", component="c")
        assert g1 == g2

    def test_inequality(self) -> None:
        g1 = EntityGroup(domain="a", container="b", component="c")
        g2 = EntityGroup(domain="x", container="b", component="c")
        assert g1 != g2

    def test_hashable(self) -> None:
        """Frozen dataclasses should be hashable (usable as dict keys)."""
        g = EntityGroup(domain="a", container="b", component="c")
        d = {g: 1}
        assert d[g] == 1


# ---------------------------------------------------------------------------
# _safe_float
# ---------------------------------------------------------------------------


class TestSafeFloat:
    def test_valid_float(self) -> None:
        assert _safe_float(3.14) == 3.14

    def test_valid_int(self) -> None:
        assert _safe_float(42) == 42.0

    def test_valid_string(self) -> None:
        assert _safe_float("2.5") == 2.5

    def test_none_returns_default(self) -> None:
        assert _safe_float(None) == 0.0

    def test_none_with_custom_default(self) -> None:
        assert _safe_float(None, default=99.9) == 99.9

    def test_invalid_string_returns_default(self) -> None:
        assert _safe_float("not_a_number") == 0.0

    def test_non_numeric_returns_default(self) -> None:
        assert _safe_float(object(), default=1.0) == 1.0


# ---------------------------------------------------------------------------
# _safe_int
# ---------------------------------------------------------------------------


class TestSafeInt:
    def test_valid_int(self) -> None:
        assert _safe_int(42) == 42

    def test_valid_float(self) -> None:
        assert _safe_int(3.9) == 3

    def test_valid_string(self) -> None:
        assert _safe_int("10") == 10

    def test_none_returns_default(self) -> None:
        assert _safe_int(None) == 0

    def test_none_with_custom_default(self) -> None:
        assert _safe_int(None, default=5) == 5

    def test_invalid_string_returns_default(self) -> None:
        assert _safe_int("not_a_number") == 0

    def test_non_numeric_returns_default(self) -> None:
        assert _safe_int(object(), default=7) == 7


# ---------------------------------------------------------------------------
# _safe_ratio
# ---------------------------------------------------------------------------


class TestSafeRatio:
    def test_normal_ratio(self) -> None:
        assert _safe_ratio(3.0, 4.0) == 0.75

    def test_zero_denominator(self) -> None:
        assert _safe_ratio(10.0, 0.0) == 0.0

    def test_negative_denominator(self) -> None:
        assert _safe_ratio(10.0, -1.0) == 0.0

    def test_zero_numerator(self) -> None:
        assert _safe_ratio(0.0, 5.0) == 0.0


# ---------------------------------------------------------------------------
# _coverage_value
# ---------------------------------------------------------------------------


class TestCoverageValue:
    def test_none_returns_none(self) -> None:
        assert _coverage_value(None) is None

    def test_normal_value(self) -> None:
        assert _coverage_value(75.0) == 75.0

    def test_clamped_above_100(self) -> None:
        assert _coverage_value(150.0) == 100.0

    def test_clamped_below_0(self) -> None:
        assert _coverage_value(-10.0) == 0.0

    def test_zero(self) -> None:
        assert _coverage_value(0.0) == 0.0

    def test_exact_100(self) -> None:
        assert _coverage_value(100.0) == 100.0


# ---------------------------------------------------------------------------
# _normalize_min_max
# ---------------------------------------------------------------------------


class TestNormalizeMinMax:
    def test_mid_range(self) -> None:
        assert _normalize_min_max(5.0, 0.0, 10.0) == 0.5

    def test_at_min(self) -> None:
        assert _normalize_min_max(0.0, 0.0, 10.0) == 0.0

    def test_at_max(self) -> None:
        assert _normalize_min_max(10.0, 0.0, 10.0) == 1.0

    def test_equal_min_max_positive_value(self) -> None:
        """When min == max, a positive value yields 1.0."""
        assert _normalize_min_max(5.0, 5.0, 5.0) == 1.0

    def test_equal_min_max_zero_value(self) -> None:
        """When min == max and value is 0, result is 0.0."""
        assert _normalize_min_max(0.0, 0.0, 0.0) == 0.0

    def test_max_less_than_min(self) -> None:
        """When max < min, the same degenerate case applies."""
        assert _normalize_min_max(5.0, 10.0, 5.0) == 1.0


# ---------------------------------------------------------------------------
# _file_path_from_natural_key
# ---------------------------------------------------------------------------


class TestFilePathFromNaturalKey:
    def test_valid_file_key(self) -> None:
        result = _file_path_from_natural_key("file:src/main.py")
        assert result == "src/main.py"

    def test_file_key_with_dotslash(self) -> None:
        """Leading ./ should be stripped by canonicalize_repo_relative_path."""
        result = _file_path_from_natural_key("file:./src/main.py")
        assert result == "src/main.py"

    def test_non_file_key_returns_none(self) -> None:
        assert _file_path_from_natural_key("symbol:SomeClass") is None

    def test_empty_after_prefix_returns_none(self) -> None:
        assert _file_path_from_natural_key("file:") is None

    def test_whitespace_only_after_prefix(self) -> None:
        assert _file_path_from_natural_key("file:   ") is None


# ---------------------------------------------------------------------------
# derive_arch_group
# ---------------------------------------------------------------------------


class TestDeriveArchGroup:
    def test_with_path_heuristic(self) -> None:
        group = derive_arch_group("services/billing/api/invoice.py")
        assert group is not None
        assert isinstance(group, EntityGroup)
        assert group.domain == "billing"
        assert group.container == "api"

    def test_with_explicit_architecture_meta(self) -> None:
        meta = {
            "architecture": {
                "domain": "core",
                "container": "auth",
                "component": "login",
            }
        }
        group = derive_arch_group("whatever/path.py", meta)
        assert group is not None
        assert group.domain == "core"
        assert group.container == "auth"
        assert group.component == "login"

    def test_none_path_no_meta(self) -> None:
        assert derive_arch_group(None) is None

    def test_none_path_with_meta_no_architecture(self) -> None:
        assert derive_arch_group(None, {"some": "thing"}) is None

    def test_apps_directory(self) -> None:
        group = derive_arch_group("apps/web/components/Button.tsx")
        assert group is not None
        assert group.domain == "web"


# ---------------------------------------------------------------------------
# build_entity_key
# ---------------------------------------------------------------------------


class TestBuildEntityKey:
    def test_file_level(self) -> None:
        key = build_entity_key("src/main.py", None, "file")
        assert key is not None
        assert key.startswith("file:")

    def test_container_level(self) -> None:
        key = build_entity_key("services/billing/api/views.py", None, "container")
        assert key is not None
        assert key.startswith("container:")
        assert "billing" in key

    def test_component_level(self) -> None:
        key = build_entity_key("services/billing/api/views.py", None, "component")
        assert key is not None
        assert key.startswith("component:")

    def test_invalid_level_returns_none(self) -> None:
        key = build_entity_key("src/main.py", None, "galaxy")
        assert key is None

    def test_empty_path_file_level(self) -> None:
        key = build_entity_key("", None, "file")
        assert key is None

    def test_none_path_container_level(self) -> None:
        key = build_entity_key(None, None, "container")
        assert key is None

    def test_with_explicit_meta(self) -> None:
        meta = {
            "architecture": {
                "domain": "core",
                "container": "billing",
            }
        }
        key = build_entity_key("any/path.py", meta, "container")
        assert key is not None
        assert "core" in key
        assert "billing" in key


# ---------------------------------------------------------------------------
# _display_label
# ---------------------------------------------------------------------------


class TestDisplayLabel:
    def test_with_colon(self) -> None:
        assert _display_label("container:core/billing") == "core/billing"

    def test_without_colon(self) -> None:
        assert _display_label("simple_key") == "simple_key"

    def test_multiple_colons(self) -> None:
        assert _display_label("type:a:b:c") == "a:b:c"


# ---------------------------------------------------------------------------
# _bus_factor
# ---------------------------------------------------------------------------


class TestBusFactor:
    def test_empty_contributions(self) -> None:
        assert _bus_factor({}) == 0

    def test_single_contributor(self) -> None:
        assert _bus_factor({"alice": 100.0}) == 1

    def test_two_equal_contributors(self) -> None:
        result = _bus_factor({"alice": 50.0, "bob": 50.0})
        assert result == 2

    def test_dominant_contributor(self) -> None:
        """When one contributor dominates, bus factor is 1."""
        result = _bus_factor({"alice": 90.0, "bob": 5.0, "carol": 5.0})
        assert result == 1

    def test_threshold_80(self) -> None:
        """Three contributors with 40%, 35%, 25% -> 40+35=75 < 80, need all 3."""
        result = _bus_factor({"alice": 40.0, "bob": 35.0, "carol": 25.0})
        assert result == 3

    def test_two_contributors_reach_threshold(self) -> None:
        """Two contributors with 50%, 40%, 10% -> 50+40=90 >= 80, need 2."""
        result = _bus_factor({"alice": 50.0, "bob": 40.0, "carol": 10.0})
        assert result == 2

    def test_many_equal_contributors(self) -> None:
        """10 equal contributors need 8 to reach 80%."""
        contributions = {f"dev_{i}": 10.0 for i in range(10)}
        result = _bus_factor(contributions)
        assert result == 8

    def test_zero_total_contributions(self) -> None:
        assert _bus_factor({"alice": 0.0, "bob": 0.0}) == 0

    def test_custom_threshold(self) -> None:
        result = _bus_factor({"alice": 50.0, "bob": 50.0}, threshold=0.5)
        assert result == 1


# ---------------------------------------------------------------------------
# _percentile
# ---------------------------------------------------------------------------


class TestPercentile:
    def test_empty_list(self) -> None:
        assert _percentile([], 0.5) == 0.0

    def test_single_element(self) -> None:
        assert _percentile([42.0], 0.5) == 42.0

    def test_median_of_sorted_list(self) -> None:
        result = _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5)
        assert result == 3.0

    def test_percentile_0(self) -> None:
        result = _percentile([10.0, 20.0, 30.0], 0.0)
        assert result == 10.0

    def test_percentile_1(self) -> None:
        result = _percentile([10.0, 20.0, 30.0], 1.0)
        assert result == 30.0

    def test_percentile_75(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = _percentile(values, 0.75)
        assert result == 4.0

    def test_percentile_clamped_above_1(self) -> None:
        result = _percentile([1.0, 2.0, 3.0], 5.0)
        assert result == 3.0

    def test_percentile_clamped_below_0(self) -> None:
        result = _percentile([1.0, 2.0, 3.0], -1.0)
        assert result == 1.0


# ---------------------------------------------------------------------------
# _topological_cycles (Tarjan SCC)
# ---------------------------------------------------------------------------


class TestTopologicalCycles:
    def test_no_cycle(self) -> None:
        """A DAG has no cycles."""
        graph = {"A": {"B"}, "B": {"C"}, "C": set()}
        assert _topological_cycles(graph) == []

    def test_simple_cycle(self) -> None:
        graph = {"A": {"B"}, "B": {"A"}}
        cycles = _topological_cycles(graph)
        assert len(cycles) == 1
        assert sorted(cycles[0]) == ["A", "B"]

    def test_self_loop_is_not_a_cycle(self) -> None:
        """Self-loops produce SCCs of size 1, which are filtered out."""
        graph = {"A": {"A"}}
        assert _topological_cycles(graph) == []

    def test_triangle_cycle(self) -> None:
        graph = {"A": {"B"}, "B": {"C"}, "C": {"A"}}
        cycles = _topological_cycles(graph)
        assert len(cycles) == 1
        assert sorted(cycles[0]) == ["A", "B", "C"]

    def test_multiple_disconnected_cycles(self) -> None:
        graph = {
            "A": {"B"},
            "B": {"A"},
            "C": {"D"},
            "D": {"C"},
            "E": set(),
        }
        cycles = _topological_cycles(graph)
        assert len(cycles) == 2

    def test_large_cycle_sorted_first(self) -> None:
        """Larger SCCs should come first in the output."""
        graph = {
            "A": {"B"},
            "B": {"C"},
            "C": {"A"},
            "X": {"Y"},
            "Y": {"X"},
        }
        cycles = _topological_cycles(graph)
        assert len(cycles) == 2
        # Sorted by (-len, component)
        assert len(cycles[0]) == 3
        assert len(cycles[1]) == 2

    def test_empty_graph(self) -> None:
        assert _topological_cycles({}) == []

    def test_single_node_no_edges(self) -> None:
        graph = {"A": set()}
        assert _topological_cycles(graph) == []
