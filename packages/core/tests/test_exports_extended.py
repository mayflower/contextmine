"""Extended tests for Mermaid C4 and CodeCharta export formatting helpers.

Covers pure/utility functions from both mermaid_c4.py and codecharta.py.
"""

from __future__ import annotations

from types import SimpleNamespace

from contextmine_core.exports.codecharta import (
    _arch_group_key,
    _attributes_from_metrics,
    _canonical_file_path,
    _checksum,
    _coerce_float,
    _coerce_int,
    _metric_from_meta,
    _MetricAccumulator,
    _MetricValues,
    _node_path_segments,
    _tree_from_leaf_items,
    _unique_leaf_paths,
)
from contextmine_core.exports.mermaid_c4 import (
    DEFAULT_C4_VIEW,
    SUPPORTED_C4_VIEWS,
    C4ExportResult,
    _build_relation_lines,
    _kind_value,
    _limit_nodes_by_degree,
    _normalize_c4_view,
    _normalize_scope,
    _safe_id,
    _safe_text,
)
from contextmine_core.twin import GraphProjection

# ===========================================================================
# Mermaid C4 helpers
# ===========================================================================


class TestSafeId:
    def test_basic(self) -> None:
        assert _safe_id("hello") == "hello"

    def test_dashes(self) -> None:
        assert _safe_id("a-b-c") == "a_b_c"

    def test_colons(self) -> None:
        assert _safe_id("ns:name") == "ns_name"

    def test_slashes(self) -> None:
        assert _safe_id("a/b/c") == "a_b_c"

    def test_combined(self) -> None:
        assert _safe_id("a-b:c/d") == "a_b_c_d"


class TestSafeText:
    def test_basic(self) -> None:
        assert _safe_text("hello") == "hello"

    def test_none(self) -> None:
        assert _safe_text(None, "fallback") == "fallback"

    def test_empty(self) -> None:
        assert _safe_text("", "fallback") == "fallback"

    def test_quotes_replaced(self) -> None:
        assert '"' not in _safe_text('say "hello"')

    def test_newlines_replaced(self) -> None:
        assert "\n" not in _safe_text("line1\nline2")

    def test_whitespace_stripped(self) -> None:
        assert _safe_text("  hello  ") == "hello"


class TestLimitNodesByDegree:
    def test_no_limit_needed(self) -> None:
        nodes = [{"id": "1"}, {"id": "2"}]
        edges = [{"source_node_id": "1", "target_node_id": "2"}]
        result_nodes, result_edges, was_limited = _limit_nodes_by_degree(nodes, edges, 10)
        assert len(result_nodes) == 2
        assert was_limited is False

    def test_limit_applied(self) -> None:
        nodes = [{"id": str(i), "name": f"n{i}"} for i in range(10)]
        edges = [{"source_node_id": "0", "target_node_id": str(i)} for i in range(1, 10)]
        result_nodes, result_edges, was_limited = _limit_nodes_by_degree(nodes, edges, 3)
        assert len(result_nodes) == 3
        assert was_limited is True
        # Node "0" should be included because it has highest degree
        result_ids = {n["id"] for n in result_nodes}
        assert "0" in result_ids

    def test_zero_max_nodes(self) -> None:
        nodes = [{"id": "1"}]
        edges = []
        result_nodes, result_edges, was_limited = _limit_nodes_by_degree(nodes, edges, 0)
        assert len(result_nodes) == 1
        assert was_limited is False

    def test_edges_filtered_to_kept_nodes(self) -> None:
        nodes = [{"id": "1", "name": "a"}, {"id": "2", "name": "b"}, {"id": "3", "name": "c"}]
        edges = [
            {"source_node_id": "1", "target_node_id": "2"},
            {"source_node_id": "2", "target_node_id": "3"},
            {"source_node_id": "1", "target_node_id": "3"},
        ]
        result_nodes, result_edges, was_limited = _limit_nodes_by_degree(nodes, edges, 2)
        assert was_limited is True
        kept_ids = {n["id"] for n in result_nodes}
        for edge in result_edges:
            assert edge["source_node_id"] in kept_ids
            assert edge["target_node_id"] in kept_ids


class TestBuildRelationLines:
    def test_basic_edge(self) -> None:
        edges = [{"source_node_id": "a", "target_node_id": "b", "kind": "depends_on", "meta": {}}]
        lines = _build_relation_lines(edges)
        assert len(lines) == 1
        assert "Rel(a, b," in lines[0]

    def test_edge_with_weight(self) -> None:
        edges = [
            {"source_node_id": "a", "target_node_id": "b", "kind": "calls", "meta": {"weight": 5}}
        ]
        lines = _build_relation_lines(edges)
        assert "w=5" in lines[0]

    def test_empty_source_skipped(self) -> None:
        edges = [{"source_node_id": "", "target_node_id": "b", "kind": "x", "meta": {}}]
        lines = _build_relation_lines(edges)
        assert len(lines) == 0

    def test_custom_label_key(self) -> None:
        edges = [
            {
                "source_node_id": "a",
                "target_node_id": "b",
                "kind": "default",
                "label": "custom",
                "meta": {},
            }
        ]
        lines = _build_relation_lines(edges, label_key="label")
        assert "custom" in lines[0]


class TestNormalizeC4View:
    def test_none(self) -> None:
        assert _normalize_c4_view(None) == DEFAULT_C4_VIEW

    def test_valid_views(self) -> None:
        for view in SUPPORTED_C4_VIEWS:
            assert _normalize_c4_view(view) == view

    def test_invalid_view(self) -> None:
        assert _normalize_c4_view("invalid") == DEFAULT_C4_VIEW

    def test_case_insensitive(self) -> None:
        assert _normalize_c4_view("CONTAINER") == "container"

    def test_whitespace(self) -> None:
        assert _normalize_c4_view("  component  ") == "component"


class TestNormalizeScope:
    def test_none(self) -> None:
        assert _normalize_scope(None) is None

    def test_empty(self) -> None:
        assert _normalize_scope("") is None

    def test_whitespace_only(self) -> None:
        assert _normalize_scope("   ") is None

    def test_valid_scope(self) -> None:
        assert _normalize_scope("billing") == "billing"


class TestKindValue:
    def test_enum_like(self) -> None:
        obj = SimpleNamespace(value="api_endpoint")
        assert _kind_value(obj) == "api_endpoint"

    def test_plain_string(self) -> None:
        assert _kind_value("api_endpoint") == "api_endpoint"

    def test_int(self) -> None:
        assert _kind_value(42) == "42"


class TestC4ExportResult:
    def test_creation(self) -> None:
        result = C4ExportResult(
            content='C4Container\ntitle "test"',
            c4_view="container",
            c4_scope=None,
            warnings=["test warning"],
        )
        assert result.content.startswith("C4Container")
        assert result.c4_view == "container"
        assert len(result.warnings) == 1


# ===========================================================================
# CodeCharta helpers
# ===========================================================================


class TestCoerceInt:
    def test_none(self) -> None:
        assert _coerce_int(None) == 0

    def test_bool_true(self) -> None:
        assert _coerce_int(True) == 1

    def test_bool_false(self) -> None:
        assert _coerce_int(False) == 0

    def test_int(self) -> None:
        assert _coerce_int(42) == 42

    def test_float(self) -> None:
        assert _coerce_int(3.7) == 3

    def test_string_int(self) -> None:
        assert _coerce_int("10") == 10

    def test_string_float(self) -> None:
        assert _coerce_int("3.7") == 3

    def test_empty_string(self) -> None:
        assert _coerce_int("") == 0

    def test_invalid_string(self) -> None:
        assert _coerce_int("abc") == 0

    def test_other_type(self) -> None:
        assert _coerce_int(object()) == 0


class TestCoerceFloat:
    def test_none(self) -> None:
        assert _coerce_float(None) == 0.0

    def test_bool(self) -> None:
        assert _coerce_float(True) == 1.0

    def test_int(self) -> None:
        assert _coerce_float(42) == 42.0

    def test_float(self) -> None:
        assert _coerce_float(3.14) == 3.14

    def test_string(self) -> None:
        assert _coerce_float("2.5") == 2.5

    def test_empty_string(self) -> None:
        assert _coerce_float("") == 0.0

    def test_invalid_string(self) -> None:
        assert _coerce_float("abc") == 0.0

    def test_other_type(self) -> None:
        assert _coerce_float(object()) == 0.0


class TestMetricFromMeta:
    def test_none(self) -> None:
        values = _metric_from_meta(None)
        assert values.loc == 0
        assert values.coupling == 0.0

    def test_empty_dict(self) -> None:
        values = _metric_from_meta({})
        assert values.loc == 0
        assert values.symbol_count == 0

    def test_with_data(self) -> None:
        values = _metric_from_meta(
            {
                "loc": 100,
                "symbol_count": 10,
                "coupling": 0.5,
                "coverage": 80.0,
                "complexity": 15.0,
            }
        )
        assert values.loc == 100
        assert values.symbol_count == 10
        assert values.coupling == 0.5
        assert values.coverage == 80.0


class TestMetricAccumulator:
    def test_empty_as_values(self) -> None:
        acc = _MetricAccumulator()
        values = acc.as_values()
        assert values.loc == 0
        assert values.coupling == 0.0

    def test_add_single(self) -> None:
        acc = _MetricAccumulator()
        values = _MetricValues(
            loc=100,
            symbol_count=10,
            coupling=0.5,
            coverage=80.0,
            complexity=15.0,
            cohesion=0.7,
            instability=0.3,
            fan_in=5.0,
            fan_out=3.0,
            cycle_participation=0.0,
            cycle_size=0.0,
            duplication_ratio=0.1,
            crap_score=10.0,
            change_frequency=2.0,
            churn=50.0,
        )
        acc.add(values)
        result = acc.as_values()
        assert result.loc == 100
        assert abs(result.coupling - 0.5) < 1e-6

    def test_add_multiple_weighted(self) -> None:
        acc = _MetricAccumulator()
        v1 = _MetricValues(
            loc=100,
            symbol_count=5,
            coupling=0.5,
            coverage=80.0,
            complexity=10.0,
            cohesion=0.8,
            instability=0.2,
            fan_in=3.0,
            fan_out=2.0,
            cycle_participation=0.0,
            cycle_size=0.0,
            duplication_ratio=0.0,
            crap_score=5.0,
            change_frequency=1.0,
            churn=10.0,
        )
        v2 = _MetricValues(
            loc=200,
            symbol_count=10,
            coupling=0.8,
            coverage=60.0,
            complexity=20.0,
            cohesion=0.6,
            instability=0.4,
            fan_in=6.0,
            fan_out=4.0,
            cycle_participation=1.0,
            cycle_size=3.0,
            duplication_ratio=0.2,
            crap_score=15.0,
            change_frequency=3.0,
            churn=30.0,
        )
        acc.add(v1)
        acc.add(v2)
        result = acc.as_values()
        assert result.loc == 300
        assert result.symbol_count == 15
        # Weighted average: coupling = (0.5*100 + 0.8*200) / 300 = 210/300 = 0.7
        assert abs(result.coupling - 0.7) < 1e-6


class TestAttributesFromMetrics:
    def test_basic(self) -> None:
        values = _MetricValues(
            loc=100,
            symbol_count=10,
            coupling=0.5,
            coverage=80.0,
            complexity=15.0,
            cohesion=0.7,
            instability=0.3,
            fan_in=5.0,
            fan_out=3.0,
            cycle_participation=0.0,
            cycle_size=0.0,
            duplication_ratio=0.1,
            crap_score=10.0,
            change_frequency=2.0,
            churn=50.0,
        )
        attrs = _attributes_from_metrics(values)
        assert attrs["loc"] == 100
        assert attrs["symbol_count"] == 10
        assert attrs["coupling"] == 0.5
        assert "churn" in attrs


class TestArchGroupKey:
    def test_domain_level(self) -> None:
        result = _arch_group_key("domain", ("billing", "api", "handler"))
        assert result == "domain|billing||"

    def test_container_level(self) -> None:
        result = _arch_group_key("container", ("billing", "api", "handler"))
        assert result == "container|billing|api|"

    def test_component_level(self) -> None:
        result = _arch_group_key("component", ("billing", "api", "handler"))
        assert result == "component|billing|api|handler"


class TestNodePathSegments:
    def test_code_file_with_file_path(self) -> None:
        node = {"meta": {"file_path": "src/billing/main.py"}, "name": "main"}
        result = _node_path_segments(GraphProjection.CODE_FILE, node, "file")
        assert result == ["src", "billing", "main.py"]

    def test_code_file_with_natural_key(self) -> None:
        node = {"meta": {}, "natural_key": "file:src/auth.py", "name": "auth"}
        result = _node_path_segments(GraphProjection.CODE_FILE, node, "file")
        assert result == ["src", "auth.py"]

    def test_code_file_fallback_name(self) -> None:
        node = {"meta": {}, "name": "unknown"}
        result = _node_path_segments(GraphProjection.CODE_FILE, node, "file")
        assert result == ["unknown"]

    def test_architecture_domain(self) -> None:
        node = {"meta": {"domain": "billing", "container": "api"}, "name": "billing"}
        result = _node_path_segments(GraphProjection.ARCHITECTURE, node, "domain")
        assert result == ["billing"]

    def test_architecture_container(self) -> None:
        node = {"meta": {"domain": "billing", "container": "api"}, "name": "api"}
        result = _node_path_segments(GraphProjection.ARCHITECTURE, node, "container")
        assert result == ["billing", "api"]

    def test_architecture_component(self) -> None:
        node = {
            "meta": {"domain": "billing", "container": "api", "component": "handler"},
            "name": "handler",
        }
        result = _node_path_segments(GraphProjection.ARCHITECTURE, node, "component")
        assert result == ["billing", "api", "handler"]


class TestUniqueLeafPaths:
    def test_no_duplicates(self) -> None:
        items = [
            ("id1", ["a", "b"], {"loc": 10}),
            ("id2", ["a", "c"], {"loc": 20}),
        ]
        result, paths = _unique_leaf_paths(items)
        assert len(result) == 2
        assert paths["id1"].endswith("/a/b")
        assert paths["id2"].endswith("/a/c")

    def test_duplicate_paths_resolved(self) -> None:
        items = [
            ("id1", ["a", "b"], {"loc": 10}),
            ("id2", ["a", "b"], {"loc": 20}),
        ]
        result, paths = _unique_leaf_paths(items)
        assert len(result) == 2
        assert paths["id1"] != paths["id2"]

    def test_empty_segments(self) -> None:
        items = [
            ("id1", [], {"loc": 10}),
        ]
        result, paths = _unique_leaf_paths(items)
        assert len(result) == 1
        assert "id1" in paths


class TestTreeFromLeafItems:
    def test_single_leaf(self) -> None:
        items = [("id1", ["a", "b", "c"], {"loc": 10})]
        tree, paths = _tree_from_leaf_items(items)
        assert tree["name"] == "root"
        assert tree["type"] == "Folder"
        assert len(tree["children"]) == 1

    def test_shared_prefix(self) -> None:
        items = [
            ("id1", ["a", "b"], {"loc": 10}),
            ("id2", ["a", "c"], {"loc": 20}),
        ]
        tree, paths = _tree_from_leaf_items(items)
        assert tree["name"] == "root"
        assert len(tree["children"]) == 1  # "a" folder
        a_folder = tree["children"][0]
        assert a_folder["name"] == "a"
        assert len(a_folder["children"]) == 2

    def test_leaf_nodes_have_attributes(self) -> None:
        items = [("id1", ["file.py"], {"loc": 42})]
        tree, paths = _tree_from_leaf_items(items)
        leaf = tree["children"][0]
        assert leaf["type"] == "File"
        assert leaf["attributes"]["loc"] == 42


class TestChecksum:
    def test_deterministic(self) -> None:
        payload = {"key": "value"}
        c1 = _checksum(payload)
        c2 = _checksum(payload)
        assert c1 == c2

    def test_different_payloads(self) -> None:
        assert _checksum({"a": 1}) != _checksum({"a": 2})

    def test_returns_hex_string(self) -> None:
        result = _checksum({"test": True})
        assert isinstance(result, str)
        assert len(result) == 64


class TestCanonicalFilePath:
    def test_with_file_prefix(self) -> None:
        node = SimpleNamespace(natural_key="file:src/main.py", meta={})
        result = _canonical_file_path(node)
        assert result == "src/main.py"

    def test_with_meta_file_path(self) -> None:
        node = SimpleNamespace(natural_key="other", meta={"file_path": "src/auth.py"})
        result = _canonical_file_path(node)
        assert result == "src/auth.py"

    def test_none_when_no_path(self) -> None:
        node = SimpleNamespace(natural_key="symbol:MyClass", meta={})
        result = _canonical_file_path(node)
        assert result is None

    def test_empty_prefix(self) -> None:
        node = SimpleNamespace(natural_key="file:", meta={})
        result = _canonical_file_path(node)
        assert result is None
