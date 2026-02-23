"""Graph projection helpers for architecture and code-centric cockpit views."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any


class GraphProjection(str, Enum):
    """Supported graph projection modes."""

    ARCHITECTURE = "architecture"
    CODE_FILE = "code_file"
    CODE_SYMBOL = "code_symbol"


@dataclass(frozen=True)
class ArchitectureProjectionNode:
    """Projected architecture node."""

    id: str
    natural_key: str
    kind: str
    name: str
    meta: dict[str, Any]


@dataclass(frozen=True)
class ArchitectureProjectionEdge:
    """Projected architecture edge."""

    id: str
    source_node_id: str
    target_node_id: str
    kind: str
    meta: dict[str, Any]


def _canonical_file_path(node: dict[str, Any]) -> str | None:
    kind = str(node.get("kind") or "").lower()
    natural_key = str(node.get("natural_key") or "")
    meta = node.get("meta") or {}

    if kind == "file" and natural_key.startswith("file:"):
        return natural_key.split(":", 1)[1]

    file_path = meta.get("file_path")
    if isinstance(file_path, str) and file_path.strip():
        return file_path.strip()

    return None


def _derive_group(
    path: str | None, meta: dict[str, Any]
) -> tuple[str, str, str, str, float] | None:
    architecture_meta = meta.get("architecture")
    if isinstance(architecture_meta, dict):
        explicit_domain = str(architecture_meta.get("domain") or "").strip()
        explicit_container = str(architecture_meta.get("container") or "").strip()
        explicit_component = str(architecture_meta.get("component") or "").strip()
        if explicit_domain and explicit_container:
            component = explicit_component or explicit_container
            return explicit_domain, explicit_container, component, "explicit", 1.0

    if not path:
        return None

    normalized = path.strip("/")
    parts = [p for p in normalized.split("/") if p]
    if not parts:
        return None

    if parts[0] == "services" and len(parts) >= 3:
        domain = parts[1]
        container = parts[2]
    elif parts[0] == "apps" and len(parts) >= 2:
        domain = parts[1]
        container = parts[1]
    else:
        domain = parts[0]
        container = parts[1] if len(parts) > 1 else parts[0]

    component = PurePosixPath(normalized).stem or container
    return domain, container, component, "heuristic", 0.6


def _kind_allowed(
    kind: str, include_kinds: set[str] | None, exclude_kinds: set[str] | None
) -> bool:
    norm = kind.lower()
    if include_kinds and norm not in include_kinds:
        return False
    return not (exclude_kinds and norm in exclude_kinds)


def build_architecture_projection(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    entity_level: str,
    include_kinds: set[str] | None = None,
    exclude_kinds: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Aggregate raw nodes/edges into architecture entities."""
    level = (entity_level or "container").lower()
    if level not in {"domain", "container", "component"}:
        level = "container"

    group_by_node_id: dict[str, tuple[str, str, str, str, float]] = {}

    explicit_count = 0
    heuristic_count = 0

    for node in nodes:
        kind = str(node.get("kind") or "")
        if not _kind_allowed(kind, include_kinds, exclude_kinds):
            continue
        file_path = _canonical_file_path(node)
        group = _derive_group(file_path, node.get("meta") or {})
        if not group:
            continue
        group_by_node_id[str(node["id"])] = group
        if group[3] == "explicit":
            explicit_count += 1
        else:
            heuristic_count += 1

    node_stats: dict[str, dict[str, Any]] = {}

    def make_key(group: tuple[str, str, str, str, float]) -> tuple[str, str, str, str]:
        domain, container, component, _, _ = group
        if level == "domain":
            return ("domain", domain, "", "")
        if level == "component":
            return ("component", domain, container, component)
        return ("container", domain, container, "")

    for group in group_by_node_id.values():
        key = make_key(group)
        stat = node_stats.setdefault(
            "|".join(key),
            {
                "kind": key[0],
                "domain": key[1],
                "container": key[2] or None,
                "component": key[3] or None,
                "member_count": 0,
                "confidence_sum": 0.0,
                "derived_from": set(),
            },
        )
        stat["member_count"] += 1
        stat["confidence_sum"] += float(group[4])
        stat["derived_from"].add(group[3])

    projected_nodes: list[dict[str, Any]] = []
    key_to_node_id: dict[str, str] = {}
    for idx, (key, stat) in enumerate(
        sorted(node_stats.items(), key=lambda item: item[0]), start=1
    ):
        node_id = f"arch:{level}:{idx}"
        key_to_node_id[key] = node_id
        name = stat["domain"]
        if stat["kind"] == "container" and stat["container"]:
            name = str(stat["container"])
        elif stat["kind"] == "component" and stat["component"]:
            name = str(stat["component"])

        projected_nodes.append(
            ArchitectureProjectionNode(
                id=node_id,
                natural_key=key,
                kind=str(stat["kind"]),
                name=name,
                meta={
                    "domain": stat["domain"],
                    "container": stat["container"],
                    "component": stat["component"],
                    "member_count": stat["member_count"],
                    "confidence": round(stat["confidence_sum"] / max(stat["member_count"], 1), 4),
                    "derived_from": sorted(stat["derived_from"]),
                    "level": level,
                },
            ).__dict__
        )

    edge_stats: dict[tuple[str, str], dict[str, Any]] = {}
    for edge in edges:
        src = group_by_node_id.get(str(edge.get("source_node_id")))
        dst = group_by_node_id.get(str(edge.get("target_node_id")))
        if not src or not dst:
            continue

        src_key = "|".join(make_key(src))
        dst_key = "|".join(make_key(dst))
        if src_key == dst_key:
            continue

        src_node_id = key_to_node_id.get(src_key)
        dst_node_id = key_to_node_id.get(dst_key)
        if not src_node_id or not dst_node_id:
            continue

        bucket = edge_stats.setdefault(
            (src_node_id, dst_node_id),
            {"weight": 0, "sample_edge_kinds": set(), "raw_edge_count": 0},
        )
        bucket["weight"] += 1
        bucket["raw_edge_count"] += 1
        bucket["sample_edge_kinds"].add(str(edge.get("kind") or "depends_on"))

    projected_edges: list[dict[str, Any]] = []
    for idx, ((src, dst), stat) in enumerate(sorted(edge_stats.items()), start=1):
        projected_edges.append(
            ArchitectureProjectionEdge(
                id=f"arch-edge:{idx}",
                source_node_id=src,
                target_node_id=dst,
                kind="depends_on",
                meta={
                    "weight": stat["weight"],
                    "sample_edge_kinds": sorted(stat["sample_edge_kinds"]),
                    "raw_edge_count": stat["raw_edge_count"],
                },
            ).__dict__
        )

    if explicit_count and heuristic_count:
        grouping_strategy = "mixed"
    elif explicit_count:
        grouping_strategy = "explicit"
    else:
        grouping_strategy = "heuristic"

    return projected_nodes, projected_edges, grouping_strategy


def build_code_file_projection(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    include_edge_kinds: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Project raw graph to file dependency graph."""
    node_by_id = {str(node["id"]): node for node in nodes}

    file_nodes: list[dict[str, Any]] = []
    file_by_path: dict[str, dict[str, Any]] = {}
    symbol_count_by_path: dict[str, int] = defaultdict(int)

    for node in nodes:
        kind = str(node.get("kind") or "").lower()
        path = _canonical_file_path(node)
        if kind == "file" and path:
            file_nodes.append(node)
            file_by_path[path] = node
        elif path:
            symbol_count_by_path[path] += 1

    projected_nodes: list[dict[str, Any]] = []
    for node in file_nodes:
        path = _canonical_file_path(node) or str(node.get("name") or "")
        meta = dict(node.get("meta") or {})
        meta["symbol_count"] = int(symbol_count_by_path.get(path, 0))
        projected_nodes.append(
            {
                "id": str(node["id"]),
                "natural_key": str(node["natural_key"]),
                "kind": "file",
                "name": str(node["name"]),
                "meta": meta,
            }
        )

    file_id_by_path = {path: str(node["id"]) for path, node in file_by_path.items()}
    edge_buckets: dict[tuple[str, str], dict[str, Any]] = {}

    for edge in edges:
        edge_kind = str(edge.get("kind") or "")
        if edge_kind == "file_defines_symbol":
            continue
        if include_edge_kinds and edge_kind.lower() not in include_edge_kinds:
            continue

        src_node = node_by_id.get(str(edge.get("source_node_id")))
        dst_node = node_by_id.get(str(edge.get("target_node_id")))
        if not src_node or not dst_node:
            continue

        src_path = _canonical_file_path(src_node)
        dst_path = _canonical_file_path(dst_node)
        if not src_path or not dst_path:
            continue

        src_file_id = file_id_by_path.get(src_path)
        dst_file_id = file_id_by_path.get(dst_path)
        if not src_file_id or not dst_file_id or src_file_id == dst_file_id:
            continue

        key = (src_file_id, dst_file_id)
        bucket = edge_buckets.setdefault(
            key,
            {"weight": 0, "sample_edge_kinds": set(), "raw_edge_count": 0},
        )
        bucket["weight"] += 1
        bucket["raw_edge_count"] += 1
        bucket["sample_edge_kinds"].add(edge_kind)

    projected_edges: list[dict[str, Any]] = []
    for idx, ((src, dst), bucket) in enumerate(sorted(edge_buckets.items()), start=1):
        projected_edges.append(
            {
                "id": f"file-edge:{idx}",
                "source_node_id": src,
                "target_node_id": dst,
                "kind": "file_depends_on_file",
                "meta": {
                    "weight": bucket["weight"],
                    "sample_edge_kinds": sorted(bucket["sample_edge_kinds"]),
                    "raw_edge_count": bucket["raw_edge_count"],
                },
            }
        )

    return projected_nodes, projected_edges


def build_code_symbol_projection(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    include_kinds: set[str] | None = None,
    exclude_kinds: set[str] | None = None,
    include_edge_kinds: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Project raw graph to symbol-level graph."""
    default_excluded = {"file"}
    effective_excluded = set(exclude_kinds or set()) | default_excluded

    projected_nodes = [
        node
        for node in nodes
        if _kind_allowed(str(node.get("kind") or ""), include_kinds, effective_excluded)
    ]
    node_ids = {str(node["id"]) for node in projected_nodes}

    projected_edges: list[dict[str, Any]] = []
    for edge in edges:
        edge_kind = str(edge.get("kind") or "")
        if include_edge_kinds and edge_kind.lower() not in include_edge_kinds:
            continue
        src = str(edge.get("source_node_id"))
        dst = str(edge.get("target_node_id"))
        if src in node_ids and dst in node_ids:
            projected_edges.append(edge)

    return projected_nodes, projected_edges


def _subgraph_by_kind(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    include_node_kinds: set[str],
    include_edge_kinds: set[str],
    include_linked_kinds: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    include_node_kinds = {kind.lower() for kind in include_node_kinds}
    include_edge_kinds = {kind.lower() for kind in include_edge_kinds}
    include_linked_kinds = {kind.lower() for kind in (include_linked_kinds or set())}

    node_by_id = {str(node.get("id")): node for node in nodes}
    selected_ids: set[str] = set()
    for node in nodes:
        kind = str(node.get("kind") or "").lower()
        if kind in include_node_kinds:
            selected_ids.add(str(node.get("id")))

    selected_edges: list[dict[str, Any]] = []
    for edge in edges:
        edge_kind = str(edge.get("kind") or "").lower()
        if edge_kind not in include_edge_kinds:
            continue
        src = str(edge.get("source_node_id"))
        dst = str(edge.get("target_node_id"))
        src_kind = str((node_by_id.get(src) or {}).get("kind") or "").lower()
        dst_kind = str((node_by_id.get(dst) or {}).get("kind") or "").lower()
        if src in selected_ids or dst in selected_ids:
            selected_ids.add(src)
            selected_ids.add(dst)
            selected_edges.append(edge)
            continue
        if src_kind in include_linked_kinds and dst_kind in include_linked_kinds:
            selected_ids.add(src)
            selected_ids.add(dst)
            selected_edges.append(edge)

    selected_nodes = [node for node in nodes if str(node.get("id")) in selected_ids]
    return selected_nodes, selected_edges


def build_ui_map_projection(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build UI map subgraph and summary stats."""
    ui_nodes, ui_edges = _subgraph_by_kind(
        nodes,
        edges,
        include_node_kinds={"ui_route", "ui_view", "ui_component", "interface_contract"},
        include_edge_kinds={
            "ui_route_renders_view",
            "ui_view_composes_component",
            "contract_governs_endpoint",
        },
        include_linked_kinds={"api_endpoint"},
    )
    return {
        "summary": {
            "routes": sum(1 for node in ui_nodes if str(node.get("kind")) == "ui_route"),
            "views": sum(1 for node in ui_nodes if str(node.get("kind")) == "ui_view"),
            "components": sum(1 for node in ui_nodes if str(node.get("kind")) == "ui_component"),
            "contracts": sum(
                1 for node in ui_nodes if str(node.get("kind")) == "interface_contract"
            ),
            "trace_edges": len(ui_edges),
        },
        "graph": {
            "nodes": ui_nodes,
            "edges": ui_edges,
            "total_nodes": len(ui_nodes),
        },
    }


def build_test_matrix_projection(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build test matrix subgraph and tabular rows."""
    test_nodes, test_edges = _subgraph_by_kind(
        nodes,
        edges,
        include_node_kinds={"test_suite", "test_case", "test_fixture"},
        include_edge_kinds={
            "test_case_covers_symbol",
            "test_case_validates_rule",
            "test_uses_fixture",
            "test_case_verifies_flow",
        },
        include_linked_kinds={"symbol", "business_rule", "user_flow"},
    )
    node_by_id = {str(node.get("id")): node for node in test_nodes}
    rows: dict[str, dict[str, Any]] = {}
    for edge in test_edges:
        src = str(edge.get("source_node_id"))
        dst = str(edge.get("target_node_id"))
        src_node = node_by_id.get(src) or {}
        dst_node = node_by_id.get(dst) or {}
        if str(src_node.get("kind")) != "test_case":
            continue
        case_key = str(src_node.get("natural_key") or src)
        row = rows.setdefault(
            case_key,
            {
                "test_case_id": src,
                "test_case_key": case_key,
                "test_case_name": str(src_node.get("name") or ""),
                "covers_symbols": [],
                "validates_rules": [],
                "fixtures": [],
                "verifies_flows": [],
                "evidence_ids": list(
                    ((src_node.get("meta") or {}).get("provenance") or {}).get("evidence_ids") or []
                ),
            },
        )
        target_name = str(dst_node.get("name") or "")
        kind = str(edge.get("kind") or "")
        if kind == "test_case_covers_symbol":
            row["covers_symbols"].append(target_name)
        elif kind == "test_case_validates_rule":
            row["validates_rules"].append(target_name)
        elif kind == "test_uses_fixture":
            row["fixtures"].append(target_name)
        elif kind == "test_case_verifies_flow":
            row["verifies_flows"].append(target_name)

    matrix_rows = []
    for row in rows.values():
        matrix_rows.append(
            {
                **row,
                "covers_symbols": sorted(set(row["covers_symbols"])),
                "validates_rules": sorted(set(row["validates_rules"])),
                "fixtures": sorted(set(row["fixtures"])),
                "verifies_flows": sorted(set(row["verifies_flows"])),
            }
        )
    matrix_rows.sort(key=lambda row: row["test_case_name"])

    return {
        "summary": {
            "test_cases": sum(1 for node in test_nodes if str(node.get("kind")) == "test_case"),
            "test_suites": sum(1 for node in test_nodes if str(node.get("kind")) == "test_suite"),
            "test_fixtures": sum(
                1 for node in test_nodes if str(node.get("kind")) == "test_fixture"
            ),
            "matrix_rows": len(matrix_rows),
        },
        "matrix": matrix_rows,
        "graph": {
            "nodes": test_nodes,
            "edges": test_edges,
            "total_nodes": len(test_nodes),
        },
    }


def build_user_flows_projection(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build user-flow graph and ordered flow payload."""
    flow_nodes, flow_edges = _subgraph_by_kind(
        nodes,
        edges,
        include_node_kinds={"user_flow", "flow_step"},
        include_edge_kinds={
            "user_flow_has_step",
            "flow_step_calls_endpoint",
            "test_case_verifies_flow",
        },
        include_linked_kinds={"api_endpoint", "test_case"},
    )
    node_by_id = {str(node.get("id")): node for node in flow_nodes}
    steps_by_flow: dict[str, list[dict[str, Any]]] = {}
    endpoint_by_step: dict[str, list[str]] = defaultdict(list)
    tests_by_flow: dict[str, list[str]] = defaultdict(list)

    for edge in flow_edges:
        src = str(edge.get("source_node_id"))
        dst = str(edge.get("target_node_id"))
        kind = str(edge.get("kind") or "")
        src_node = node_by_id.get(src) or {}
        dst_node = node_by_id.get(dst) or {}
        if kind == "user_flow_has_step" and str(src_node.get("kind")) == "user_flow":
            step_meta = dst_node.get("meta") or {}
            steps_by_flow.setdefault(src, []).append(
                {
                    "step_id": dst,
                    "name": str(dst_node.get("name") or ""),
                    "order": int(step_meta.get("order") or 0),
                    "endpoint_hints": list(step_meta.get("endpoint_hints") or []),
                    "evidence_ids": list(
                        (step_meta.get("provenance") or {}).get("evidence_ids") or []
                    ),
                }
            )
        elif kind == "flow_step_calls_endpoint" and str(src_node.get("kind")) == "flow_step":
            endpoint_by_step[src].append(
                str(dst_node.get("name") or dst_node.get("natural_key") or dst)
            )
        elif kind == "test_case_verifies_flow" and str(dst_node.get("kind")) == "user_flow":
            tests_by_flow[dst].append(
                str(src_node.get("name") or src_node.get("natural_key") or src)
            )

    flows_payload: list[dict[str, Any]] = []
    for node in flow_nodes:
        if str(node.get("kind")) != "user_flow":
            continue
        flow_id = str(node.get("id"))
        flow_steps = steps_by_flow.get(flow_id, [])
        flow_steps.sort(key=lambda item: (item["order"], item["name"]))
        for step in flow_steps:
            step["calls_endpoints"] = sorted(set(endpoint_by_step.get(step["step_id"], [])))
        flows_payload.append(
            {
                "flow_id": flow_id,
                "flow_key": str(node.get("natural_key") or flow_id),
                "flow_name": str(node.get("name") or ""),
                "route_path": str((node.get("meta") or {}).get("route_path") or ""),
                "steps": flow_steps,
                "verified_by_tests": sorted(set(tests_by_flow.get(flow_id, []))),
                "evidence_ids": list(
                    ((node.get("meta") or {}).get("provenance") or {}).get("evidence_ids") or []
                ),
            }
        )
    flows_payload.sort(key=lambda row: row["flow_name"])

    return {
        "summary": {
            "user_flows": len(flows_payload),
            "flow_steps": sum(len(flow["steps"]) for flow in flows_payload),
            "flow_edges": len(flow_edges),
        },
        "flows": flows_payload,
        "graph": {
            "nodes": flow_nodes,
            "edges": flow_edges,
            "total_nodes": len(flow_nodes),
        },
    }


def compute_rebuild_readiness(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute rebuild-readiness score and explainable contributors."""
    node_by_id = {str(node.get("id")): node for node in nodes}

    endpoint_ids = {
        str(node.get("id")) for node in nodes if str(node.get("kind") or "") == "api_endpoint"
    }
    test_case_ids = {
        str(node.get("id")) for node in nodes if str(node.get("kind") or "") == "test_case"
    }
    ui_route_ids = {
        str(node.get("id")) for node in nodes if str(node.get("kind") or "") == "ui_route"
    }
    flow_ids = {str(node.get("id")) for node in nodes if str(node.get("kind") or "") == "user_flow"}
    flow_step_ids = {
        str(node.get("id")) for node in nodes if str(node.get("kind") or "") == "flow_step"
    }

    tested_endpoints: set[str] = set()
    flow_evidence_total = 0
    flow_step_count = 0
    ui_routed_views: set[str] = set()
    views_with_endpoint_calls: set[str] = set()
    critical_inferred_only: list[dict[str, Any]] = []

    critical_kinds = {"interface_contract", "user_flow", "flow_step", "ui_route", "test_case"}
    for node in nodes:
        kind = str(node.get("kind") or "")
        if kind not in critical_kinds:
            continue
        provenance = (node.get("meta") or {}).get("provenance") or {}
        mode = str(provenance.get("mode") or "")
        confidence = float(provenance.get("confidence") or 0.0)
        if mode == "inferred" and confidence < 0.65:
            critical_inferred_only.append(
                {
                    "node_id": str(node.get("id")),
                    "kind": kind,
                    "name": str(node.get("name") or ""),
                    "confidence": round(confidence, 4),
                    "evidence_ids": list(provenance.get("evidence_ids") or []),
                }
            )

    for edge in edges:
        kind = str(edge.get("kind") or "")
        src = str(edge.get("source_node_id"))
        dst = str(edge.get("target_node_id"))
        if kind in {"test_case_covers_symbol", "test_case_validates_rule"} and src in test_case_ids:
            continue
        if kind == "flow_step_calls_endpoint" and src in flow_step_ids and dst in endpoint_ids:
            tested_endpoints.add(dst)
        if kind == "test_case_verifies_flow" and src in test_case_ids and dst in flow_ids:
            tested_endpoints.update(
                {
                    str(flow_edge.get("target_node_id"))
                    for flow_edge in edges
                    if str(flow_edge.get("kind") or "") == "flow_step_calls_endpoint"
                    and str(flow_edge.get("source_node_id")) in flow_step_ids
                    and str(
                        (node_by_id.get(str(flow_edge.get("source_node_id"))) or {}).get("kind")
                    )
                    == "flow_step"
                }
            )
        if kind == "ui_route_renders_view" and src in ui_route_ids:
            ui_routed_views.add(dst)
        if kind == "flow_step_calls_endpoint":
            step_meta = (node_by_id.get(src) or {}).get("meta") or {}
            if step_meta:
                flow_step_count += 1
                flow_evidence_total += len(
                    (step_meta.get("provenance") or {}).get("evidence_ids") or []
                )
        if kind == "contract_governs_endpoint":
            views_with_endpoint_calls.add(src)

    endpoint_coverage = len(tested_endpoints) / len(endpoint_ids) if endpoint_ids else 0.0
    flow_evidence_density = flow_evidence_total / flow_step_count if flow_step_count > 0 else 0.0
    ui_traceability = (
        len(views_with_endpoint_calls) / len(ui_routed_views) if ui_routed_views else 0.0
    )
    inferred_penalty = min(1.0, len(critical_inferred_only) / 10.0)

    score = (
        (endpoint_coverage * 0.35)
        + (min(flow_evidence_density / 2.0, 1.0) * 0.25)
        + (ui_traceability * 0.25)
        + ((1.0 - inferred_penalty) * 0.15)
    )
    score_100 = int(round(max(0.0, min(score, 1.0)) * 100))

    known_gaps: list[str] = []
    if endpoint_coverage < 0.5:
        known_gaps.append("Low endpoint verification coverage from tests/flows.")
    if ui_traceability < 0.5:
        known_gaps.append("Low UI to endpoint traceability.")
    if flow_evidence_density < 0.75:
        known_gaps.append("Sparse flow evidence on synthesized steps.")
    if critical_inferred_only:
        known_gaps.append("Critical inferred-only nodes require deterministic confirmation.")

    return {
        "score": score_100,
        "summary": {
            "interface_test_coverage": round(endpoint_coverage, 4),
            "flow_evidence_density": round(flow_evidence_density, 4),
            "ui_to_endpoint_traceability": round(ui_traceability, 4),
            "critical_inferred_only_count": len(critical_inferred_only),
            "total_nodes": len(nodes),
            "total_edges": len(edges),
        },
        "critical_inferred_only": critical_inferred_only[:50],
        "known_gaps": known_gaps,
    }
