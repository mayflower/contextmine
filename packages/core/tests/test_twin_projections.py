from __future__ import annotations

from contextmine_core.twin.projections import (
    build_architecture_projection,
    build_code_symbol_projection,
)


def _sample_nodes() -> list[dict]:
    return [
        {
            "id": "f1",
            "natural_key": "file:services/billing/api/invoice.py",
            "kind": "file",
            "name": "services/billing/api/invoice.py",
            "meta": {},
        },
        {
            "id": "f2",
            "natural_key": "file:services/payments/core/charge.py",
            "kind": "file",
            "name": "services/payments/core/charge.py",
            "meta": {},
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


def _sample_edges() -> list[dict]:
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


def test_architecture_projection_excludes_symbol_level_nodes() -> None:
    nodes, edges, grouping_strategy = build_architecture_projection(
        nodes=_sample_nodes(),
        edges=_sample_edges(),
        entity_level="container",
    )
    assert grouping_strategy in {"heuristic", "mixed", "explicit"}
    assert nodes
    assert all(node["kind"] in {"domain", "container", "component"} for node in nodes)
    assert all(edge["kind"] == "depends_on" for edge in edges)


def test_architecture_projection_uses_explicit_mapping_priority() -> None:
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


def test_architecture_projection_folds_edge_weights() -> None:
    _, projected_edges, _ = build_architecture_projection(
        nodes=_sample_nodes(),
        edges=_sample_edges(),
        entity_level="container",
    )
    assert len(projected_edges) == 1
    assert projected_edges[0]["meta"]["weight"] == 2
    assert projected_edges[0]["meta"]["raw_edge_count"] == 2


def test_code_symbol_projection_filters_file_nodes() -> None:
    projected_nodes, _ = build_code_symbol_projection(
        nodes=_sample_nodes(),
        edges=_sample_edges(),
    )
    assert projected_nodes
    assert all(node["kind"] != "file" for node in projected_nodes)
