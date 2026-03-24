"""Tests for contextmine_core.twin.evolution — pure helper functions."""

from __future__ import annotations

import pytest
from contextmine_core.twin.evolution import (
    _SEVERITY_WEIGHT,
    _SUPPORTED_ENTITY_LEVELS,
    DEFAULT_EVOLUTION_WINDOW_DAYS,
    DEFAULT_MAX_COUPLING_EDGES,
    DEFAULT_MIN_JACCARD,
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
)

# ── _safe_float ─────────────────────────────────────────────────────────


class TestSafeFloat:
    def test_normal_float(self) -> None:
        assert _safe_float(3.14) == 3.14

    def test_int_to_float(self) -> None:
        assert _safe_float(42) == 42.0

    def test_string_number(self) -> None:
        assert _safe_float("2.5") == 2.5

    def test_none_returns_default(self) -> None:
        assert _safe_float(None) == 0.0

    def test_custom_default(self) -> None:
        assert _safe_float(None, 5.0) == 5.0

    def test_invalid_string(self) -> None:
        assert _safe_float("abc") == 0.0

    def test_empty_string(self) -> None:
        assert _safe_float("") == 0.0


# ── _safe_int ───────────────────────────────────────────────────────────


class TestSafeInt:
    def test_normal_int(self) -> None:
        assert _safe_int(42) == 42

    def test_float_to_int(self) -> None:
        assert _safe_int(3.9) == 3

    def test_string_number(self) -> None:
        assert _safe_int("7") == 7

    def test_none_returns_default(self) -> None:
        assert _safe_int(None) == 0

    def test_custom_default(self) -> None:
        assert _safe_int(None, 5) == 5

    def test_invalid_string(self) -> None:
        assert _safe_int("abc") == 0


# ── _safe_ratio ─────────────────────────────────────────────────────────


class TestSafeRatio:
    def test_normal_ratio(self) -> None:
        assert _safe_ratio(3.0, 4.0) == 0.75

    def test_zero_denominator(self) -> None:
        assert _safe_ratio(1.0, 0.0) == 0.0

    def test_negative_denominator(self) -> None:
        assert _safe_ratio(1.0, -1.0) == 0.0

    def test_zero_numerator(self) -> None:
        assert _safe_ratio(0.0, 5.0) == 0.0


# ── _coverage_value ─────────────────────────────────────────────────────


class TestCoverageValue:
    def test_none(self) -> None:
        assert _coverage_value(None) is None

    def test_normal_value(self) -> None:
        assert _coverage_value(75.0) == 75.0

    def test_clamp_low(self) -> None:
        assert _coverage_value(-10.0) == 0.0

    def test_clamp_high(self) -> None:
        assert _coverage_value(150.0) == 100.0

    def test_zero(self) -> None:
        assert _coverage_value(0.0) == 0.0

    def test_hundred(self) -> None:
        assert _coverage_value(100.0) == 100.0


# ── _normalize_min_max ──────────────────────────────────────────────────


class TestNormalizeMinMax:
    def test_normal(self) -> None:
        assert _normalize_min_max(5.0, 0.0, 10.0) == 0.5

    def test_at_min(self) -> None:
        assert _normalize_min_max(0.0, 0.0, 10.0) == 0.0

    def test_at_max(self) -> None:
        assert _normalize_min_max(10.0, 0.0, 10.0) == 1.0

    def test_equal_min_max_positive(self) -> None:
        assert _normalize_min_max(5.0, 5.0, 5.0) == 1.0

    def test_equal_min_max_zero(self) -> None:
        assert _normalize_min_max(0.0, 5.0, 5.0) == 0.0


# ── _file_path_from_natural_key ─────────────────────────────────────────


class TestFilePathFromNaturalKey:
    def test_valid_key(self) -> None:
        result = _file_path_from_natural_key("file:src/main.py")
        assert result is not None
        assert "src/main.py" in result

    def test_no_file_prefix(self) -> None:
        assert _file_path_from_natural_key("symbol:foo") is None

    def test_empty_after_prefix(self) -> None:
        assert _file_path_from_natural_key("file:") is None

    def test_whitespace_after_prefix(self) -> None:
        assert _file_path_from_natural_key("file:  ") is None


# ── _display_label ──────────────────────────────────────────────────────


class TestDisplayLabel:
    def test_with_colon(self) -> None:
        assert _display_label("container:domain/api") == "domain/api"

    def test_without_colon(self) -> None:
        assert _display_label("no_colon_here") == "no_colon_here"

    def test_multiple_colons(self) -> None:
        assert _display_label("type:a:b:c") == "a:b:c"


# ── _bus_factor ─────────────────────────────────────────────────────────


class TestBusFactor:
    def test_empty(self) -> None:
        assert _bus_factor({}) == 0

    def test_single_contributor(self) -> None:
        assert _bus_factor({"alice": 100.0}) == 1

    def test_even_split(self) -> None:
        assert _bus_factor({"alice": 50.0, "bob": 50.0}) == 2

    def test_dominant_contributor(self) -> None:
        # 90% from alice, 10% from bob => bus factor 1
        assert _bus_factor({"alice": 90.0, "bob": 10.0}) == 1

    def test_three_contributors(self) -> None:
        # Need 80% threshold: alice=40, bob=30, charlie=30
        # 40/100=40% not enough, (40+30)/100=70% not enough, need all three
        result = _bus_factor({"alice": 40.0, "bob": 30.0, "charlie": 30.0})
        assert result == 3

    def test_zero_total(self) -> None:
        assert _bus_factor({"alice": 0.0, "bob": 0.0}) == 0

    def test_custom_threshold(self) -> None:
        result = _bus_factor({"alice": 50.0, "bob": 50.0}, threshold=0.5)
        assert result == 1


# ── _percentile ─────────────────────────────────────────────────────────


class TestPercentile:
    def test_empty(self) -> None:
        assert _percentile([], 0.5) == 0.0

    def test_single(self) -> None:
        assert _percentile([42.0], 0.5) == 42.0

    def test_median(self) -> None:
        result = _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5)
        assert result == 3.0

    def test_p75(self) -> None:
        result = _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.75)
        assert result == 4.0

    def test_p0(self) -> None:
        result = _percentile([10.0, 20.0, 30.0], 0.0)
        assert result == 10.0

    def test_p100(self) -> None:
        result = _percentile([10.0, 20.0, 30.0], 1.0)
        assert result == 30.0

    def test_bounds_clamping(self) -> None:
        # percentile > 1.0 should be clamped
        result = _percentile([10.0, 20.0, 30.0], 1.5)
        assert result == 30.0


# ── _topological_cycles ─────────────────────────────────────────────────


class TestTopologicalCycles:
    def test_no_cycles(self) -> None:
        graph: dict[str, set[str]] = {"a": {"b"}, "b": {"c"}, "c": set()}
        cycles = _topological_cycles(graph)
        assert cycles == []

    def test_simple_cycle(self) -> None:
        graph: dict[str, set[str]] = {"a": {"b"}, "b": {"a"}}
        cycles = _topological_cycles(graph)
        assert len(cycles) == 1
        assert set(cycles[0]) == {"a", "b"}

    def test_three_node_cycle(self) -> None:
        graph: dict[str, set[str]] = {"a": {"b"}, "b": {"c"}, "c": {"a"}}
        cycles = _topological_cycles(graph)
        assert len(cycles) == 1
        assert set(cycles[0]) == {"a", "b", "c"}

    def test_self_loop_not_detected(self) -> None:
        # Self-loops create SCC of size 1, which are filtered out
        graph: dict[str, set[str]] = {"a": {"a"}}
        cycles = _topological_cycles(graph)
        assert cycles == []

    def test_empty_graph(self) -> None:
        cycles = _topological_cycles({})
        assert cycles == []

    def test_multiple_cycles(self) -> None:
        graph: dict[str, set[str]] = {
            "a": {"b"},
            "b": {"a"},
            "c": {"d"},
            "d": {"c"},
        }
        cycles = _topological_cycles(graph)
        assert len(cycles) == 2

    def test_disconnected_with_one_cycle(self) -> None:
        graph: dict[str, set[str]] = {
            "a": {"b"},
            "b": {"a"},
            "c": {"d"},
            "d": set(),
        }
        cycles = _topological_cycles(graph)
        assert len(cycles) == 1
        assert set(cycles[0]) == {"a", "b"}


# ── EntityGroup ─────────────────────────────────────────────────────────


class TestEntityGroup:
    def test_frozen(self) -> None:
        group = EntityGroup(domain="web", container="api", component="auth")
        assert group.domain == "web"
        assert group.container == "api"
        assert group.component == "auth"
        with pytest.raises(AttributeError):
            group.domain = "other"


# ── build_entity_key ────────────────────────────────────────────────────


class TestBuildEntityKey:
    def test_file_level(self) -> None:
        result = build_entity_key("src/main.py", {}, "file")
        assert result is not None
        assert result.startswith("file:")

    def test_container_level(self) -> None:
        result = build_entity_key("src/api/routes.py", {}, "container")
        if result is not None:
            assert result.startswith("container:")

    def test_component_level(self) -> None:
        result = build_entity_key("src/api/routes.py", {}, "component")
        if result is not None:
            assert result.startswith("component:")

    def test_unknown_level(self) -> None:
        result = build_entity_key("src/main.py", {}, "unknown")
        assert result is None

    def test_empty_path(self) -> None:
        result = build_entity_key("", {}, "file")
        assert result is None


# ── Constants ───────────────────────────────────────────────────────────


class TestConstants:
    def test_default_window_days(self) -> None:
        assert DEFAULT_EVOLUTION_WINDOW_DAYS == 365

    def test_default_min_jaccard(self) -> None:
        assert DEFAULT_MIN_JACCARD == 0.2

    def test_default_max_coupling_edges(self) -> None:
        assert DEFAULT_MAX_COUPLING_EDGES == 300

    def test_supported_entity_levels(self) -> None:
        assert {"file", "container", "component"} == _SUPPORTED_ENTITY_LEVELS

    def test_severity_weight(self) -> None:
        assert _SEVERITY_WEIGHT["critical"] > _SEVERITY_WEIGHT["high"]
        assert _SEVERITY_WEIGHT["high"] > _SEVERITY_WEIGHT["medium"]
        assert _SEVERITY_WEIGHT["medium"] > _SEVERITY_WEIGHT["low"]
