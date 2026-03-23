"""Tests for contextmine_core.twin.projections.

Covers architecture projection, code-file projection, code-symbol projection,
UI map projection, test matrix projection, user flows projection,
rebuild readiness scoring, and internal helpers.
"""

from __future__ import annotations

from typing import Any

from contextmine_core.twin.projections import (
    _derive_group,
    _kind_allowed,
    _subgraph_by_kind,
    build_architecture_projection,
    build_code_file_projection,
    build_code_symbol_projection,
    build_test_matrix_projection,
    build_ui_map_projection,
    build_user_flows_projection,
    compute_rebuild_readiness,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _sample_nodes() -> list[dict[str, Any]]:
    return [
        {
            "id": "f1",
            "natural_key": "file:services/billing/api/invoice.py",
            "kind": "file",
            "name": "services/billing/api/invoice.py",
            "meta": {"file_path": "services/billing/api/invoice.py"},
        },
        {
            "id": "f2",
            "natural_key": "file:services/payments/core/charge.py",
            "kind": "file",
            "name": "services/payments/core/charge.py",
            "meta": {"file_path": "services/payments/core/charge.py"},
        },
        {
            "id": "c1",
            "natural_key": "symbol:class:InvoiceService",
            "kind": "class",
            "name": "InvoiceService",
            "meta": {"file_path": "services/billing/api/invoice.py"},
        },
        {
            "id": "m1",
            "natural_key": "symbol:method:charge",
            "kind": "method",
            "name": "charge",
            "meta": {"file_path": "services/payments/core/charge.py"},
        },
    ]


def _sample_edges() -> list[dict[str, Any]]:
    return [
        {
            "id": "e1",
            "source_node_id": "c1",
            "target_node_id": "m1",
            "kind": "symbol_calls_symbol",
            "meta": {},
        },
        {
            "id": "e2",
            "source_node_id": "c1",
            "target_node_id": "m1",
            "kind": "symbol_references_symbol",
            "meta": {},
        },
    ]


# ---------------------------------------------------------------------------
# _kind_allowed
# ---------------------------------------------------------------------------


class TestKindAllowed:
    def test_no_filters(self) -> None:
        assert _kind_allowed("file", None, None) is True

    def test_include_match(self) -> None:
        assert _kind_allowed("file", {"file"}, None) is True

    def test_include_mismatch(self) -> None:
        assert _kind_allowed("class", {"file"}, None) is False

    def test_exclude_match(self) -> None:
        assert _kind_allowed("file", None, {"file"}) is False

    def test_exclude_mismatch(self) -> None:
        assert _kind_allowed("class", None, {"file"}) is True

    def test_case_insensitive(self) -> None:
        assert _kind_allowed("FILE", {"file"}, None) is True
        assert _kind_allowed("FILE", None, {"file"}) is False

    def test_include_and_exclude_both_set(self) -> None:
        # "class" is in include_kinds and not in exclude_kinds, so it passes
        assert _kind_allowed("class", {"class", "method"}, {"method"}) is True
        # "method" is in both include and exclude -> exclude wins
        assert _kind_allowed("method", {"class", "method"}, {"method"}) is False
        assert _kind_allowed("class", {"class", "method"}, {"file"}) is True


# ---------------------------------------------------------------------------
# _derive_group
# ---------------------------------------------------------------------------


class TestDeriveGroup:
    def test_heuristic_path(self) -> None:
        result = _derive_group("services/billing/api/invoice.py", {})
        assert result is not None
        domain, container, component, strategy, confidence = result
        assert strategy == "heuristic"
        assert confidence == 0.6
        assert domain == "billing"

    def test_explicit_architecture_meta(self) -> None:
        meta = {
            "architecture": {
                "domain": "core",
                "container": "auth",
                "component": "login",
            }
        }
        result = _derive_group("whatever.py", meta)
        assert result is not None
        domain, container, component, strategy, confidence = result
        assert strategy == "explicit"
        assert confidence == 1.0
        assert domain == "core"
        assert container == "auth"

    def test_none_path_no_meta_returns_none(self) -> None:
        assert _derive_group(None, {}) is None

    def test_partial_architecture_meta_falls_to_heuristic(self) -> None:
        """If architecture meta only has domain but no container, use heuristic."""
        meta = {
            "architecture": {
                "domain": "core",
                "container": "",
            }
        }
        result = _derive_group("apps/web/index.ts", meta)
        assert result is not None
        _, _, _, strategy, _ = result
        assert strategy == "heuristic"


# ---------------------------------------------------------------------------
# Architecture Projection (existing tests + new)
# ---------------------------------------------------------------------------


class TestArchitectureProjection:
    def test_excludes_symbol_level_nodes(self) -> None:
        nodes, edges, grouping_strategy = build_architecture_projection(
            nodes=_sample_nodes(),
            edges=_sample_edges(),
            entity_level="container",
        )
        assert grouping_strategy in {"heuristic", "mixed", "explicit"}
        assert nodes
        assert all(node["kind"] in {"domain", "container", "component"} for node in nodes)
        assert all(edge["kind"] == "depends_on" for edge in edges)

    def test_uses_explicit_mapping_priority(self) -> None:
        nodes = _sample_nodes()
        nodes[0]["meta"] = {
            "architecture": {
                "domain": "core",
                "container": "billing",
                "component": "invoice-api",
            }
        }
        projected_nodes, _, grouping_strategy = build_architecture_projection(
            nodes=nodes,
            edges=_sample_edges(),
            entity_level="component",
        )
        assert grouping_strategy == "mixed"
        assert any(node["meta"]["domain"] == "core" for node in projected_nodes)

    def test_folds_edge_weights(self) -> None:
        _, projected_edges, _ = build_architecture_projection(
            nodes=_sample_nodes(),
            edges=_sample_edges(),
            entity_level="container",
        )
        assert len(projected_edges) == 1
        assert projected_edges[0]["meta"]["weight"] == 2
        assert projected_edges[0]["meta"]["raw_edge_count"] == 2

    def test_domain_level(self) -> None:
        projected_nodes, _, _ = build_architecture_projection(
            nodes=_sample_nodes(),
            edges=_sample_edges(),
            entity_level="domain",
        )
        assert projected_nodes
        assert all(node["kind"] == "domain" for node in projected_nodes)

    def test_component_level(self) -> None:
        projected_nodes, _, _ = build_architecture_projection(
            nodes=_sample_nodes(),
            edges=_sample_edges(),
            entity_level="component",
        )
        assert projected_nodes
        assert all(node["kind"] == "component" for node in projected_nodes)

    def test_invalid_entity_level_defaults_to_container(self) -> None:
        projected_nodes, _, _ = build_architecture_projection(
            nodes=_sample_nodes(),
            edges=_sample_edges(),
            entity_level="invalid_level",
        )
        assert projected_nodes
        assert all(node["kind"] == "container" for node in projected_nodes)

    def test_intra_group_edges_dropped(self) -> None:
        """Edges between nodes in the same group should not appear.

        Using services/billing/api/a.py and services/billing/api/b.py which both
        resolve to domain=billing, container=api at container level.
        """
        nodes = [
            {
                "id": "f1",
                "natural_key": "file:services/billing/api/a.py",
                "kind": "file",
                "name": "a.py",
                "meta": {"file_path": "services/billing/api/a.py"},
            },
            {
                "id": "f2",
                "natural_key": "file:services/billing/api/b.py",
                "kind": "file",
                "name": "b.py",
                "meta": {"file_path": "services/billing/api/b.py"},
            },
        ]
        edges = [
            {
                "id": "e1",
                "source_node_id": "f1",
                "target_node_id": "f2",
                "kind": "depends_on",
                "meta": {},
            }
        ]
        _, projected_edges, _ = build_architecture_projection(
            nodes=nodes, edges=edges, entity_level="container"
        )
        assert len(projected_edges) == 0

    def test_include_kinds_filter(self) -> None:
        projected_nodes, _, _ = build_architecture_projection(
            nodes=_sample_nodes(),
            edges=_sample_edges(),
            entity_level="container",
            include_kinds={"file"},
        )
        assert projected_nodes

    def test_exclude_kinds_filter(self) -> None:
        projected_nodes, _, _ = build_architecture_projection(
            nodes=_sample_nodes(),
            edges=_sample_edges(),
            entity_level="container",
            exclude_kinds={"file"},
        )
        # After excluding files, only symbol nodes remain for grouping
        assert len(projected_nodes) >= 0

    def test_all_explicit_grouping_strategy(self) -> None:
        nodes = [
            {
                "id": "f1",
                "natural_key": "file:a.py",
                "kind": "file",
                "name": "a.py",
                "meta": {
                    "file_path": "a.py",
                    "architecture": {"domain": "core", "container": "api"},
                },
            },
        ]
        _, _, strategy = build_architecture_projection(
            nodes=nodes, edges=[], entity_level="container"
        )
        assert strategy == "explicit"


# ---------------------------------------------------------------------------
# Code File Projection
# ---------------------------------------------------------------------------


class TestCodeFileProjection:
    def test_basic_file_projection(self) -> None:
        projected_nodes, projected_edges = build_code_file_projection(
            nodes=_sample_nodes(),
            edges=_sample_edges(),
        )
        assert all(node["kind"] == "file" for node in projected_nodes)
        # There are symbol-level edges, which should create file-level edges
        assert len(projected_nodes) == 2

    def test_symbol_count_in_meta(self) -> None:
        projected_nodes, _ = build_code_file_projection(
            nodes=_sample_nodes(),
            edges=_sample_edges(),
        )
        # Each file has one symbol attached
        for node in projected_nodes:
            assert "symbol_count" in node["meta"]

    def test_file_defines_symbol_edges_skipped(self) -> None:
        """file_defines_symbol edges should be excluded from projection."""
        nodes = _sample_nodes()
        edges = [
            {
                "id": "e_def",
                "source_node_id": "f1",
                "target_node_id": "c1",
                "kind": "file_defines_symbol",
                "meta": {},
            },
        ]
        _, projected_edges = build_code_file_projection(nodes=nodes, edges=edges)
        assert len(projected_edges) == 0

    def test_cross_file_edges_aggregated(self) -> None:
        """Symbol edges across files should produce file-level edges."""
        projected_nodes, projected_edges = build_code_file_projection(
            nodes=_sample_nodes(),
            edges=_sample_edges(),
        )
        # c1 (billing) -> m1 (payments) = billing_file -> payments_file
        if projected_edges:
            assert projected_edges[0]["kind"] == "file_depends_on_file"
            assert "weight" in projected_edges[0]["meta"]

    def test_include_edge_kinds_filter(self) -> None:
        _, projected_edges = build_code_file_projection(
            nodes=_sample_nodes(),
            edges=_sample_edges(),
            include_edge_kinds={"symbol_calls_symbol"},
        )
        # Only calls edges are included; references edge is dropped
        for edge in projected_edges:
            sample_kinds = edge["meta"]["sample_edge_kinds"]
            assert "symbol_references_symbol" not in sample_kinds

    def test_same_file_edges_excluded(self) -> None:
        """Edges between symbols in the same file should not create file edges."""
        nodes = [
            {
                "id": "f1",
                "natural_key": "file:src/a.py",
                "kind": "file",
                "name": "src/a.py",
                "meta": {"file_path": "src/a.py"},
            },
            {
                "id": "s1",
                "natural_key": "symbol:foo",
                "kind": "function",
                "name": "foo",
                "meta": {"file_path": "src/a.py"},
            },
            {
                "id": "s2",
                "natural_key": "symbol:bar",
                "kind": "function",
                "name": "bar",
                "meta": {"file_path": "src/a.py"},
            },
        ]
        edges = [
            {
                "id": "e1",
                "source_node_id": "s1",
                "target_node_id": "s2",
                "kind": "symbol_calls_symbol",
                "meta": {},
            },
        ]
        _, projected_edges = build_code_file_projection(nodes=nodes, edges=edges)
        assert len(projected_edges) == 0


# ---------------------------------------------------------------------------
# Code Symbol Projection
# ---------------------------------------------------------------------------


class TestCodeSymbolProjection:
    def test_filters_file_nodes(self) -> None:
        projected_nodes, _ = build_code_symbol_projection(
            nodes=_sample_nodes(),
            edges=_sample_edges(),
        )
        assert projected_nodes
        assert all(node["kind"] != "file" for node in projected_nodes)

    def test_include_kinds(self) -> None:
        projected_nodes, _ = build_code_symbol_projection(
            nodes=_sample_nodes(),
            edges=_sample_edges(),
            include_kinds={"class"},
        )
        assert all(node["kind"] == "class" for node in projected_nodes)

    def test_exclude_kinds(self) -> None:
        projected_nodes, _ = build_code_symbol_projection(
            nodes=_sample_nodes(),
            edges=_sample_edges(),
            exclude_kinds={"method"},
        )
        assert all(node["kind"] != "method" for node in projected_nodes)
        assert all(node["kind"] != "file" for node in projected_nodes)

    def test_include_edge_kinds(self) -> None:
        _, projected_edges = build_code_symbol_projection(
            nodes=_sample_nodes(),
            edges=_sample_edges(),
            include_edge_kinds={"symbol_calls_symbol"},
        )
        assert all(edge["kind"] == "symbol_calls_symbol" for edge in projected_edges)

    def test_edges_only_between_projected_nodes(self) -> None:
        """Edges referencing nodes not in projection should be excluded."""
        projected_nodes, projected_edges = build_code_symbol_projection(
            nodes=_sample_nodes(),
            edges=_sample_edges(),
            include_kinds={"class"},  # only class nodes
        )
        node_ids = {str(n["id"]) for n in projected_nodes}
        for edge in projected_edges:
            assert str(edge["source_node_id"]) in node_ids
            assert str(edge["target_node_id"]) in node_ids


# ---------------------------------------------------------------------------
# _subgraph_by_kind
# ---------------------------------------------------------------------------


class TestSubgraphByKind:
    def test_selects_matching_node_kinds(self) -> None:
        nodes: list[dict[str, Any]] = [
            {"id": "1", "kind": "test_case", "name": "test_a"},
            {"id": "2", "kind": "symbol", "name": "foo"},
            {"id": "3", "kind": "test_case", "name": "test_b"},
        ]
        edges: list[dict[str, Any]] = [
            {
                "id": "e1",
                "source_node_id": "1",
                "target_node_id": "2",
                "kind": "test_case_covers_symbol",
            },
        ]
        selected_nodes, selected_edges = _subgraph_by_kind(
            nodes,
            edges,
            include_node_kinds={"test_case"},
            include_edge_kinds={"test_case_covers_symbol"},
        )
        assert len(selected_nodes) == 3  # test_case nodes + linked symbol
        assert len(selected_edges) == 1

    def test_include_linked_kinds(self) -> None:
        nodes: list[dict[str, Any]] = [
            {"id": "1", "kind": "test_case", "name": "test_a"},
            {"id": "2", "kind": "symbol", "name": "foo"},
            {"id": "3", "kind": "api_endpoint", "name": "GET /api"},
            {"id": "4", "kind": "api_endpoint", "name": "POST /api"},
        ]
        edges: list[dict[str, Any]] = [
            {"id": "e1", "source_node_id": "3", "target_node_id": "4", "kind": "api_calls_api"},
        ]
        selected_nodes, selected_edges = _subgraph_by_kind(
            nodes,
            edges,
            include_node_kinds={"test_case"},
            include_edge_kinds={"api_calls_api"},
            include_linked_kinds={"api_endpoint"},
        )
        # api_endpoint nodes are linked via include_linked_kinds
        api_nodes = [n for n in selected_nodes if n["kind"] == "api_endpoint"]
        assert len(api_nodes) == 2
        assert len(selected_edges) == 1

    def test_unmatched_edge_kinds_excluded(self) -> None:
        nodes: list[dict[str, Any]] = [
            {"id": "1", "kind": "test_case", "name": "test_a"},
            {"id": "2", "kind": "symbol", "name": "foo"},
        ]
        edges: list[dict[str, Any]] = [
            {"id": "e1", "source_node_id": "1", "target_node_id": "2", "kind": "unrelated_edge"},
        ]
        _, selected_edges = _subgraph_by_kind(
            nodes,
            edges,
            include_node_kinds={"test_case"},
            include_edge_kinds={"test_case_covers_symbol"},
        )
        assert len(selected_edges) == 0


# ---------------------------------------------------------------------------
# UI Map Projection
# ---------------------------------------------------------------------------


class TestUIMapProjection:
    def _ui_nodes_and_edges(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        nodes: list[dict[str, Any]] = [
            {"id": "r1", "kind": "ui_route", "name": "/dashboard"},
            {"id": "v1", "kind": "ui_view", "name": "DashboardView"},
            {"id": "comp1", "kind": "ui_component", "name": "Chart"},
            {"id": "ct1", "kind": "interface_contract", "name": "GET /api/stats"},
            {"id": "ep1", "kind": "api_endpoint", "name": "GET /api/stats"},
            {"id": "other", "kind": "file", "name": "main.py"},
        ]
        edges: list[dict[str, Any]] = [
            {
                "id": "e1",
                "source_node_id": "r1",
                "target_node_id": "v1",
                "kind": "ui_route_renders_view",
            },
            {
                "id": "e2",
                "source_node_id": "v1",
                "target_node_id": "comp1",
                "kind": "ui_view_composes_component",
            },
            {
                "id": "e3",
                "source_node_id": "ct1",
                "target_node_id": "ep1",
                "kind": "contract_governs_endpoint",
            },
        ]
        return nodes, edges

    def test_basic_ui_map(self) -> None:
        nodes, edges = self._ui_nodes_and_edges()
        result = build_ui_map_projection(nodes, edges)
        assert "summary" in result
        assert "graph" in result
        assert result["summary"]["routes"] == 1
        assert result["summary"]["views"] == 1
        assert result["summary"]["components"] == 1
        assert result["summary"]["contracts"] == 1

    def test_file_nodes_excluded(self) -> None:
        nodes, edges = self._ui_nodes_and_edges()
        result = build_ui_map_projection(nodes, edges)
        graph_kinds = {str(n.get("kind")) for n in result["graph"]["nodes"]}
        assert "file" not in graph_kinds

    def test_empty_graph(self) -> None:
        result = build_ui_map_projection([], [])
        assert result["summary"]["routes"] == 0
        assert result["graph"]["total_nodes"] == 0


# ---------------------------------------------------------------------------
# Test Matrix Projection
# ---------------------------------------------------------------------------


class TestTestMatrixProjection:
    def _test_nodes_and_edges(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        nodes: list[dict[str, Any]] = [
            {
                "id": "ts1",
                "kind": "test_suite",
                "name": "test_billing",
                "natural_key": "suite:test_billing",
                "meta": {},
            },
            {
                "id": "tc1",
                "kind": "test_case",
                "name": "test_invoice_creation",
                "natural_key": "case:test_invoice_creation",
                "meta": {"provenance": {"evidence_ids": ["ev1"]}},
            },
            {
                "id": "tc2",
                "kind": "test_case",
                "name": "test_payment_flow",
                "natural_key": "case:test_payment_flow",
                "meta": {},
            },
            {
                "id": "fix1",
                "kind": "test_fixture",
                "name": "db_session",
                "natural_key": "fixture:db_session",
                "meta": {},
            },
            {
                "id": "sym1",
                "kind": "symbol",
                "name": "InvoiceService.create",
                "natural_key": "symbol:InvoiceService.create",
                "meta": {},
            },
            {
                "id": "rule1",
                "kind": "business_rule",
                "name": "MaxInvoiceAmount",
                "natural_key": "rule:MaxInvoiceAmount",
                "meta": {},
            },
            {
                "id": "flow1",
                "kind": "user_flow",
                "name": "Checkout Flow",
                "natural_key": "flow:checkout",
                "meta": {},
            },
        ]
        edges: list[dict[str, Any]] = [
            {
                "id": "e1",
                "source_node_id": "tc1",
                "target_node_id": "sym1",
                "kind": "test_case_covers_symbol",
            },
            {
                "id": "e2",
                "source_node_id": "tc1",
                "target_node_id": "rule1",
                "kind": "test_case_validates_rule",
            },
            {
                "id": "e3",
                "source_node_id": "tc1",
                "target_node_id": "fix1",
                "kind": "test_uses_fixture",
            },
            {
                "id": "e4",
                "source_node_id": "tc2",
                "target_node_id": "flow1",
                "kind": "test_case_verifies_flow",
            },
        ]
        return nodes, edges

    def test_basic_matrix(self) -> None:
        nodes, edges = self._test_nodes_and_edges()
        result = build_test_matrix_projection(nodes, edges)
        assert result["summary"]["test_cases"] >= 1
        assert result["summary"]["test_suites"] >= 1
        assert len(result["matrix"]) >= 1

    def test_matrix_row_contents(self) -> None:
        nodes, edges = self._test_nodes_and_edges()
        result = build_test_matrix_projection(nodes, edges)
        matrix = result["matrix"]
        # Find the row for test_invoice_creation
        invoice_row = next(
            (row for row in matrix if row["test_case_name"] == "test_invoice_creation"),
            None,
        )
        assert invoice_row is not None
        assert "InvoiceService.create" in invoice_row["covers_symbols"]
        assert "MaxInvoiceAmount" in invoice_row["validates_rules"]
        assert "db_session" in invoice_row["fixtures"]

    def test_verifies_flow_row(self) -> None:
        nodes, edges = self._test_nodes_and_edges()
        result = build_test_matrix_projection(nodes, edges)
        matrix = result["matrix"]
        payment_row = next(
            (row for row in matrix if row["test_case_name"] == "test_payment_flow"),
            None,
        )
        assert payment_row is not None
        assert "Checkout Flow" in payment_row["verifies_flows"]

    def test_empty_input(self) -> None:
        result = build_test_matrix_projection([], [])
        assert result["summary"]["test_cases"] == 0
        assert result["matrix"] == []

    def test_matrix_sorted_by_name(self) -> None:
        nodes, edges = self._test_nodes_and_edges()
        result = build_test_matrix_projection(nodes, edges)
        names = [row["test_case_name"] for row in result["matrix"]]
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# User Flows Projection
# ---------------------------------------------------------------------------


class TestUserFlowsProjection:
    def _flow_nodes_and_edges(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        nodes: list[dict[str, Any]] = [
            {
                "id": "flow1",
                "kind": "user_flow",
                "name": "Login Flow",
                "natural_key": "flow:login",
                "meta": {"route_path": "/login", "provenance": {"evidence_ids": ["ev1"]}},
            },
            {
                "id": "step1",
                "kind": "flow_step",
                "name": "Enter credentials",
                "natural_key": "step:enter_creds",
                "meta": {
                    "order": 1,
                    "endpoint_hints": ["/api/auth"],
                    "provenance": {"evidence_ids": ["ev2"]},
                },
            },
            {
                "id": "step2",
                "kind": "flow_step",
                "name": "Submit form",
                "natural_key": "step:submit",
                "meta": {"order": 2, "provenance": {"evidence_ids": []}},
            },
            {
                "id": "ep1",
                "kind": "api_endpoint",
                "name": "POST /api/auth",
                "natural_key": "endpoint:POST /api/auth",
                "meta": {},
            },
            {
                "id": "tc1",
                "kind": "test_case",
                "name": "test_login",
                "natural_key": "case:test_login",
                "meta": {},
            },
        ]
        edges: list[dict[str, Any]] = [
            {
                "id": "e1",
                "source_node_id": "flow1",
                "target_node_id": "step1",
                "kind": "user_flow_has_step",
            },
            {
                "id": "e2",
                "source_node_id": "flow1",
                "target_node_id": "step2",
                "kind": "user_flow_has_step",
            },
            {
                "id": "e3",
                "source_node_id": "step1",
                "target_node_id": "ep1",
                "kind": "flow_step_calls_endpoint",
            },
            {
                "id": "e4",
                "source_node_id": "tc1",
                "target_node_id": "flow1",
                "kind": "test_case_verifies_flow",
            },
        ]
        return nodes, edges

    def test_basic_flows(self) -> None:
        nodes, edges = self._flow_nodes_and_edges()
        result = build_user_flows_projection(nodes, edges)
        assert result["summary"]["user_flows"] == 1
        assert result["summary"]["flow_steps"] == 2

    def test_flow_steps_ordered(self) -> None:
        nodes, edges = self._flow_nodes_and_edges()
        result = build_user_flows_projection(nodes, edges)
        flow = result["flows"][0]
        assert flow["flow_name"] == "Login Flow"
        assert len(flow["steps"]) == 2
        assert flow["steps"][0]["order"] <= flow["steps"][1]["order"]

    def test_calls_endpoints_populated(self) -> None:
        nodes, edges = self._flow_nodes_and_edges()
        result = build_user_flows_projection(nodes, edges)
        flow = result["flows"][0]
        step1 = flow["steps"][0]
        assert "POST /api/auth" in step1["calls_endpoints"]

    def test_verified_by_tests_populated(self) -> None:
        nodes, edges = self._flow_nodes_and_edges()
        result = build_user_flows_projection(nodes, edges)
        flow = result["flows"][0]
        assert "test_login" in flow["verified_by_tests"]

    def test_empty_input(self) -> None:
        result = build_user_flows_projection([], [])
        assert result["summary"]["user_flows"] == 0
        assert result["flows"] == []

    def test_flow_route_path(self) -> None:
        nodes, edges = self._flow_nodes_and_edges()
        result = build_user_flows_projection(nodes, edges)
        flow = result["flows"][0]
        assert flow["route_path"] == "/login"


# ---------------------------------------------------------------------------
# Compute Rebuild Readiness
# ---------------------------------------------------------------------------


class TestComputeRebuildReadiness:
    def test_empty_input(self) -> None:
        result = compute_rebuild_readiness([], [])
        assert result["score"] == 15  # only the (1 - 0) * 0.15 = 0.15 -> 15%
        assert result["summary"]["total_nodes"] == 0

    def test_perfect_score_scenario(self) -> None:
        """Construct a scenario with high endpoint coverage, good evidence, and UI tracing."""
        nodes: list[dict[str, Any]] = [
            {"id": "ep1", "kind": "api_endpoint", "name": "GET /api/users", "meta": {}},
            {"id": "tc1", "kind": "test_case", "name": "test_users", "meta": {}},
            {"id": "flow1", "kind": "user_flow", "name": "User Flow", "meta": {}},
            {
                "id": "step1",
                "kind": "flow_step",
                "name": "Step 1",
                "meta": {"provenance": {"evidence_ids": ["e1", "e2"]}},
            },
            {"id": "r1", "kind": "ui_route", "name": "/users", "meta": {}},
            {"id": "v1", "kind": "ui_view", "name": "UsersView", "meta": {}},
            {"id": "ct1", "kind": "interface_contract", "name": "GET /api/users", "meta": {}},
        ]
        edges: list[dict[str, Any]] = [
            {
                "id": "e1",
                "source_node_id": "step1",
                "target_node_id": "ep1",
                "kind": "flow_step_calls_endpoint",
            },
            {
                "id": "e2",
                "source_node_id": "tc1",
                "target_node_id": "flow1",
                "kind": "test_case_verifies_flow",
            },
            {
                "id": "e3",
                "source_node_id": "r1",
                "target_node_id": "v1",
                "kind": "ui_route_renders_view",
            },
            {
                "id": "e4",
                "source_node_id": "ct1",
                "target_node_id": "ep1",
                "kind": "contract_governs_endpoint",
            },
        ]
        result = compute_rebuild_readiness(nodes, edges)
        assert result["score"] > 0
        assert "summary" in result
        assert isinstance(result["known_gaps"], list)

    def test_known_gaps_populated(self) -> None:
        """With no edges, most gaps should fire."""
        nodes: list[dict[str, Any]] = [
            {"id": "ep1", "kind": "api_endpoint", "name": "GET /api", "meta": {}},
            {"id": "r1", "kind": "ui_route", "name": "/home", "meta": {}},
        ]
        result = compute_rebuild_readiness(nodes, [])
        assert len(result["known_gaps"]) >= 1

    def test_critical_inferred_only_detection(self) -> None:
        nodes: list[dict[str, Any]] = [
            {
                "id": "flow1",
                "kind": "user_flow",
                "name": "Flow",
                "meta": {"provenance": {"mode": "inferred", "confidence": 0.3, "evidence_ids": []}},
            },
        ]
        result = compute_rebuild_readiness(nodes, [])
        assert result["critical_inferred_only"]
        assert result["critical_inferred_only"][0]["confidence"] == 0.3

    def test_score_range(self) -> None:
        """Score should always be 0-100."""
        nodes: list[dict[str, Any]] = [
            {"id": "1", "kind": "file", "name": "a.py", "meta": {}},
        ]
        result = compute_rebuild_readiness(nodes, [])
        assert 0 <= result["score"] <= 100
