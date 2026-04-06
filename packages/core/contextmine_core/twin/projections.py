"""Graph projection helpers for architecture and code-centric cockpit views."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any

from contextmine_core.architecture.recovery_model import RecoveredArchitectureModel
from contextmine_core.architecture.schemas import EvidenceRef
from contextmine_core.twin.grouping import (
    canonical_file_path_from_node,
    derive_arch_group,
)


class GraphProjection(str, Enum):
    """Supported graph projection modes."""

    ARCHITECTURE = "architecture"
    INFERRED_ARCHITECTURE = "inferred_architecture"
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


_canonical_file_path = canonical_file_path_from_node


def _derive_group(
    path: str | None, meta: dict[str, Any]
) -> tuple[str, str, str, str, float] | None:
    """Thin wrapper adding strategy/confidence to the shared grouping logic."""
    payload = meta or {}
    architecture_meta = payload.get("architecture")
    is_explicit = isinstance(architecture_meta, dict) and bool(
        str(architecture_meta.get("domain") or "").strip()
        and str(architecture_meta.get("container") or "").strip()
    )

    group = derive_arch_group(path, meta)
    if group is None:
        return None

    domain, container, component = group
    if is_explicit:
        return domain, container, component, "explicit", 1.0
    return domain, container, component, "heuristic", 0.6


def _kind_allowed(
    kind: str, include_kinds: set[str] | None, exclude_kinds: set[str] | None
) -> bool:
    norm = kind.lower()
    if include_kinds and norm not in include_kinds:
        return False
    return not (exclude_kinds and norm in exclude_kinds)


def _resolve_projected_node_name(stat: dict[str, Any]) -> str:
    """Resolve display name for a projected architecture node."""
    if stat["kind"] == "container" and stat["container"]:
        return str(stat["container"])
    if stat["kind"] == "component" and stat["component"]:
        return str(stat["component"])
    return stat["domain"]


def _make_level_key(
    level: str, group: tuple[str, str, str, str, float]
) -> tuple[str, str, str, str]:
    domain, container, component, _, _ = group
    if level == "domain":
        return ("domain", domain, "", "")
    if level == "component":
        return ("component", domain, container, component)
    return ("container", domain, container, "")


def _group_nodes_by_arch(
    nodes: list[dict[str, Any]],
    include_kinds: set[str] | None,
    exclude_kinds: set[str] | None,
) -> tuple[dict[str, tuple[str, str, str, str, float]], int, int]:
    """Assign each node to its architecture group. Return (mapping, explicit_count, heuristic_count)."""
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
    return group_by_node_id, explicit_count, heuristic_count


def _aggregate_arch_edges(
    edges: list[dict[str, Any]],
    group_by_node_id: dict[str, tuple[str, str, str, str, float]],
    key_to_node_id: dict[str, str],
    level: str,
) -> list[dict[str, Any]]:
    """Aggregate raw edges into architecture-level projected edges."""
    edge_stats: dict[tuple[str, str], dict[str, Any]] = {}
    for edge in edges:
        src = group_by_node_id.get(str(edge.get("source_node_id")))
        dst = group_by_node_id.get(str(edge.get("target_node_id")))
        if not src or not dst:
            continue
        src_key = "|".join(_make_level_key(level, src))
        dst_key = "|".join(_make_level_key(level, dst))
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
    return projected_edges


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

    group_by_node_id, explicit_count, heuristic_count = _group_nodes_by_arch(
        nodes, include_kinds, exclude_kinds
    )

    node_stats: dict[str, dict[str, Any]] = {}
    for group in group_by_node_id.values():
        key = _make_level_key(level, group)
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
        name = _resolve_projected_node_name(stat)
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

    projected_edges = _aggregate_arch_edges(edges, group_by_node_id, key_to_node_id, level)

    if explicit_count and heuristic_count:
        grouping_strategy = "mixed"
    elif explicit_count:
        grouping_strategy = "explicit"
    else:
        grouping_strategy = "heuristic"

    return projected_nodes, projected_edges, grouping_strategy


def _evidence_summary(evidence: tuple[EvidenceRef, ...]) -> list[str]:
    refs = {
        str(ref.ref).strip()
        for ref in evidence
        if isinstance(ref.ref, str) and str(ref.ref).strip()
    }
    return sorted(refs)


def _component_container_memberships(
    model: RecoveredArchitectureModel,
) -> dict[str, set[str]]:
    memberships_by_subject: dict[str, list[Any]] = defaultdict(list)
    for membership in model.memberships:
        memberships_by_subject[membership.subject_ref].append(membership)

    component_to_containers: dict[str, set[str]] = defaultdict(set)
    for memberships in memberships_by_subject.values():
        component_ids = {
            membership.entity_id
            for membership in memberships
            if membership.entity_id.startswith("component:")
        }
        container_ids = {
            membership.entity_id
            for membership in memberships
            if membership.entity_id.startswith("container:")
        }
        for component_id in component_ids:
            component_to_containers[component_id].update(container_ids)
    return component_to_containers


def _expanded_entity_node_ids(
    entity_id: str,
    component_to_containers: dict[str, set[str]],
) -> list[str]:
    if not entity_id.startswith("component:"):
        return [entity_id]
    container_ids = sorted(component_to_containers.get(entity_id) or set())
    if not container_ids:
        return [entity_id]
    return [f"{entity_id}@{container_id}" for container_id in container_ids]


def _build_entity_projection_node(
    entity: Any,
    *,
    node_id: str,
    entities_by_id: dict[str, Any],
    membership_confidence_by_entity_id: dict[str, float],
    entity_level: str,
) -> dict[str, Any]:
    meta = {
        "confidence": max(
            float(entity.confidence),
            float(membership_confidence_by_entity_id.get(entity.entity_id, 0.0)),
        ),
        "evidence_summary": _evidence_summary(entity.evidence),
        "entity_id": entity.entity_id,
        "level": entity_level,
        **dict(entity.attributes),
    }
    if "@" in node_id:
        _, container_id = node_id.split("@", 1)
        container = entities_by_id.get(container_id)
        if container is not None:
            meta["container_id"] = container_id
            meta["container_context"] = container.name
    return {
        "id": node_id,
        "natural_key": node_id,
        "kind": str(entity.kind),
        "name": str(entity.name),
        "meta": meta,
    }


def build_inferred_architecture_projection(
    model: RecoveredArchitectureModel,
    entity_level: str = "container",
) -> dict[str, Any]:
    """Project a recovered architecture model without collapsing shared memberships."""
    level = (entity_level or "container").lower()
    if level not in {"container", "component"}:
        level = "container"

    entities_by_id = {entity.entity_id: entity for entity in model.entities}
    component_to_containers = _component_container_memberships(model)
    membership_confidence_by_entity_id: dict[str, float] = {}
    for membership in model.memberships:
        membership_confidence_by_entity_id[membership.entity_id] = max(
            float(membership.confidence),
            float(membership_confidence_by_entity_id.get(membership.entity_id, 0.0)),
        )

    if level == "container":
        selected_entities = [
            entity
            for entity in model.entities
            if entity.kind in {"container", "data_store", "external_system", "message_channel"}
        ]
    else:
        selected_entities = list(model.entities)

    projected_nodes: list[dict[str, Any]] = []
    for entity in sorted(selected_entities, key=lambda row: (row.kind, row.entity_id)):
        for node_id in _expanded_entity_node_ids(entity.entity_id, component_to_containers):
            projected_nodes.append(
                _build_entity_projection_node(
                    entity,
                    node_id=node_id,
                    entities_by_id=entities_by_id,
                    membership_confidence_by_entity_id=membership_confidence_by_entity_id,
                    entity_level=level,
                )
            )

    valid_node_ids = {str(node["id"]) for node in projected_nodes}
    projected_edges: list[dict[str, Any]] = []
    seen_edge_ids: set[tuple[str, str, str]] = set()
    for relationship in sorted(
        model.relationships,
        key=lambda row: (row.source_entity_id, row.target_entity_id, row.kind),
    ):
        source_ids = _expanded_entity_node_ids(
            relationship.source_entity_id,
            component_to_containers,
        )
        target_ids = _expanded_entity_node_ids(
            relationship.target_entity_id,
            component_to_containers,
        )
        for source_id in source_ids:
            if source_id not in valid_node_ids:
                continue
            for target_id in target_ids:
                if target_id not in valid_node_ids:
                    continue
                edge_key = (source_id, target_id, relationship.kind)
                if edge_key in seen_edge_ids or source_id == target_id:
                    continue
                seen_edge_ids.add(edge_key)
                projected_edges.append(
                    {
                        "id": (
                            f"{relationship.source_entity_id}:"
                            f"{relationship.kind}:{relationship.target_entity_id}:"
                            f"{len(projected_edges) + 1}"
                        ),
                        "source_node_id": source_id,
                        "target_node_id": target_id,
                        "kind": relationship.kind,
                        "meta": {
                            "confidence": float(relationship.confidence),
                            "evidence_summary": _evidence_summary(relationship.evidence),
                            **dict(relationship.attributes),
                        },
                    }
                )

    return {
        "nodes": projected_nodes,
        "edges": projected_edges,
        "total_nodes": len(projected_nodes),
        "projection": GraphProjection.INFERRED_ARCHITECTURE.value,
        "entity_level": level,
        "grouping_strategy": "recovered",
        "summary": {
            "ambiguous_hypotheses": sum(
                1 for hypothesis in model.hypotheses if hypothesis.status == "ambiguous"
            ),
            "unresolved_hypotheses": sum(
                1 for hypothesis in model.hypotheses if hypothesis.status == "unresolved"
            ),
        },
        "warnings": list(model.warnings),
    }


def _resolve_file_edge_endpoints(
    edge: dict[str, Any],
    node_by_id: dict[str, dict[str, Any]],
    file_id_by_path: dict[str, str],
) -> tuple[str, str] | None:
    """Resolve an edge to source/target file IDs, returning None if ineligible."""
    src_node = node_by_id.get(str(edge.get("source_node_id")))
    dst_node = node_by_id.get(str(edge.get("target_node_id")))
    if not src_node or not dst_node:
        return None
    src_path = _canonical_file_path(src_node)
    dst_path = _canonical_file_path(dst_node)
    if not src_path or not dst_path:
        return None
    src_file_id = file_id_by_path.get(src_path)
    dst_file_id = file_id_by_path.get(dst_path)
    if not src_file_id or not dst_file_id or src_file_id == dst_file_id:
        return None
    return src_file_id, dst_file_id


def _classify_file_nodes(
    nodes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, int]]:
    """Separate file nodes from symbol nodes and count symbols per path."""
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
    return file_nodes, file_by_path, symbol_count_by_path


def _bucket_file_edges(
    edges: list[dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
    file_id_by_path: dict[str, str],
    include_edge_kinds: set[str] | None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Bucket edges by (source_file, target_file) for file projection."""
    edge_buckets: dict[tuple[str, str], dict[str, Any]] = {}
    for edge in edges:
        edge_kind = str(edge.get("kind") or "")
        if edge_kind == "file_defines_symbol":
            continue
        if include_edge_kinds and edge_kind.lower() not in include_edge_kinds:
            continue
        endpoints = _resolve_file_edge_endpoints(edge, node_by_id, file_id_by_path)
        if not endpoints:
            continue
        src_file_id, dst_file_id = endpoints
        bucket = edge_buckets.setdefault(
            (src_file_id, dst_file_id),
            {"weight": 0, "sample_edge_kinds": set(), "raw_edge_count": 0},
        )
        bucket["weight"] += 1
        bucket["raw_edge_count"] += 1
        bucket["sample_edge_kinds"].add(edge_kind)
    return edge_buckets


def build_code_file_projection(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    include_edge_kinds: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Project raw graph to file dependency graph."""
    node_by_id = {str(node["id"]): node for node in nodes}
    file_nodes, file_by_path, symbol_count_by_path = _classify_file_nodes(nodes)

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
    edge_buckets = _bucket_file_edges(edges, node_by_id, file_id_by_path, include_edge_kinds)

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


def _should_include_edge(
    src: str,
    dst: str,
    node_by_id: dict[str, dict[str, Any]],
    selected_ids: set[str],
    include_linked_kinds: set[str],
) -> bool:
    """Check if an edge should be included in the subgraph."""
    if src in selected_ids or dst in selected_ids:
        return True
    src_kind = str((node_by_id.get(src) or {}).get("kind") or "").lower()
    dst_kind = str((node_by_id.get(dst) or {}).get("kind") or "").lower()
    return src_kind in include_linked_kinds and dst_kind in include_linked_kinds


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
    linked_kinds = {kind.lower() for kind in (include_linked_kinds or set())}

    node_by_id = {str(node.get("id")): node for node in nodes}
    selected_ids: set[str] = {
        str(node.get("id"))
        for node in nodes
        if str(node.get("kind") or "").lower() in include_node_kinds
    }

    selected_edges: list[dict[str, Any]] = []
    for edge in edges:
        edge_kind = str(edge.get("kind") or "").lower()
        if edge_kind not in include_edge_kinds:
            continue
        src = str(edge.get("source_node_id"))
        dst = str(edge.get("target_node_id"))
        if _should_include_edge(src, dst, node_by_id, selected_ids, linked_kinds):
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


_TEST_EDGE_KIND_TO_FIELD = {
    "test_case_covers_symbol": "covers_symbols",
    "test_case_validates_rule": "validates_rules",
    "test_uses_fixture": "fixtures",
    "test_case_verifies_flow": "verifies_flows",
}


def _classify_test_edge(
    edge: dict[str, Any],
    node_by_id: dict[str, dict[str, Any]],
    rows: dict[str, dict[str, Any]],
) -> None:
    """Classify a test edge into the test matrix rows."""
    src = str(edge.get("source_node_id"))
    dst = str(edge.get("target_node_id"))
    src_node = node_by_id.get(src) or {}
    dst_node = node_by_id.get(dst) or {}
    if str(src_node.get("kind")) != "test_case":
        return
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
    field = _TEST_EDGE_KIND_TO_FIELD.get(str(edge.get("kind") or ""))
    if field:
        row[field].append(target_name)


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
        _classify_test_edge(edge, node_by_id, rows)

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


def _build_flow_step_dict(dst: str, dst_node: dict[str, Any]) -> dict[str, Any]:
    """Build a flow step payload dictionary from a destination node."""
    step_meta = dst_node.get("meta") or {}
    return {
        "step_id": dst,
        "name": str(dst_node.get("name") or ""),
        "order": int(step_meta.get("order") or 0),
        "endpoint_hints": list(step_meta.get("endpoint_hints") or []),
        "evidence_ids": list((step_meta.get("provenance") or {}).get("evidence_ids") or []),
    }


def _classify_flow_edge(
    edge: dict[str, Any],
    node_by_id: dict[str, dict[str, Any]],
    steps_by_flow: dict[str, list[dict[str, Any]]],
    endpoint_by_step: dict[str, list[str]],
    tests_by_flow: dict[str, list[str]],
) -> None:
    """Classify a single flow edge into steps, endpoints, or test links."""
    src = str(edge.get("source_node_id"))
    dst = str(edge.get("target_node_id"))
    kind = str(edge.get("kind") or "")
    src_node = node_by_id.get(src) or {}
    dst_node = node_by_id.get(dst) or {}
    src_kind = str(src_node.get("kind"))
    dst_kind = str(dst_node.get("kind"))
    if kind == "user_flow_has_step" and src_kind == "user_flow":
        steps_by_flow.setdefault(src, []).append(_build_flow_step_dict(dst, dst_node))
    elif kind == "flow_step_calls_endpoint" and src_kind == "flow_step":
        endpoint_by_step[src].append(
            str(dst_node.get("name") or dst_node.get("natural_key") or dst)
        )
    elif kind == "test_case_verifies_flow" and dst_kind == "user_flow":
        tests_by_flow[dst].append(str(src_node.get("name") or src_node.get("natural_key") or src))


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
        _classify_flow_edge(
            edge,
            node_by_id,
            steps_by_flow,
            endpoint_by_step,
            tests_by_flow,
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


def _collect_critical_inferred_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect critical nodes that are inferred-only with low confidence."""
    critical_kinds = {"interface_contract", "user_flow", "flow_step", "ui_route", "test_case"}
    result: list[dict[str, Any]] = []
    for node in nodes:
        kind = str(node.get("kind") or "")
        if kind not in critical_kinds:
            continue
        provenance = (node.get("meta") or {}).get("provenance") or {}
        mode = str(provenance.get("mode") or "")
        confidence = float(provenance.get("confidence") or 0.0)
        if mode != "inferred" or confidence >= 0.65:
            continue
        result.append(
            {
                "node_id": str(node.get("id")),
                "kind": kind,
                "name": str(node.get("name") or ""),
                "confidence": round(confidence, 4),
                "evidence_ids": list(provenance.get("evidence_ids") or []),
            }
        )
    return result


def _ids_by_kind(nodes: list[dict[str, Any]], kind: str) -> set[str]:
    """Return node IDs matching a specific kind."""
    return {str(node.get("id")) for node in nodes if str(node.get("kind") or "") == kind}


@dataclass
class _ReadinessAccumulator:
    """Mutable accumulator for rebuild-readiness edge scanning."""

    tested_endpoints: set[str]
    ui_routed_views: set[str]
    views_with_endpoint_calls: set[str]
    flow_evidence_total: int = 0
    flow_step_count: int = 0


def _process_readiness_edge(
    edge: dict[str, Any],
    acc: _ReadinessAccumulator,
    node_by_id: dict[str, dict[str, Any]],
    endpoint_ids: set[str],
    test_case_ids: set[str],
    ui_route_ids: set[str],
    flow_ids: set[str],
    flow_step_ids: set[str],
    flow_endpoint_targets: set[str],
) -> None:
    """Classify a single edge for rebuild-readiness scoring."""
    kind = str(edge.get("kind") or "")
    src = str(edge.get("source_node_id"))
    dst = str(edge.get("target_node_id"))
    if kind == "flow_step_calls_endpoint" and src in flow_step_ids and dst in endpoint_ids:
        acc.tested_endpoints.add(dst)
    if kind == "test_case_verifies_flow" and src in test_case_ids and dst in flow_ids:
        acc.tested_endpoints.update(flow_endpoint_targets)
    if kind == "ui_route_renders_view" and src in ui_route_ids:
        acc.ui_routed_views.add(dst)
    if kind == "flow_step_calls_endpoint":
        step_meta = (node_by_id.get(src) or {}).get("meta") or {}
        if step_meta:
            acc.flow_step_count += 1
            acc.flow_evidence_total += len(
                (step_meta.get("provenance") or {}).get("evidence_ids") or []
            )
    if kind == "contract_governs_endpoint":
        acc.views_with_endpoint_calls.add(src)


def _compute_readiness_gaps(
    endpoint_coverage: float,
    ui_traceability: float,
    flow_evidence_density: float,
    critical_inferred_only: list[dict[str, Any]],
) -> list[str]:
    """Compile the known-gaps list for rebuild-readiness."""
    gaps: list[str] = []
    if endpoint_coverage < 0.5:
        gaps.append("Low endpoint verification coverage from tests/flows.")
    if ui_traceability < 0.5:
        gaps.append("Low UI to endpoint traceability.")
    if flow_evidence_density < 0.75:
        gaps.append("Sparse flow evidence on synthesized steps.")
    if critical_inferred_only:
        gaps.append("Critical inferred-only nodes require deterministic confirmation.")
    return gaps


def compute_rebuild_readiness(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute rebuild-readiness score and explainable contributors."""
    node_by_id = {str(node.get("id")): node for node in nodes}
    endpoint_ids = _ids_by_kind(nodes, "api_endpoint")
    test_case_ids = _ids_by_kind(nodes, "test_case")
    ui_route_ids = _ids_by_kind(nodes, "ui_route")
    flow_ids = _ids_by_kind(nodes, "user_flow")
    flow_step_ids = _ids_by_kind(nodes, "flow_step")

    critical_inferred_only = _collect_critical_inferred_nodes(nodes)

    flow_endpoint_targets = {
        str(fe.get("target_node_id"))
        for fe in edges
        if str(fe.get("kind") or "") == "flow_step_calls_endpoint"
        and str(fe.get("source_node_id")) in flow_step_ids
    }

    acc = _ReadinessAccumulator(
        tested_endpoints=set(),
        ui_routed_views=set(),
        views_with_endpoint_calls=set(),
    )
    for edge in edges:
        _process_readiness_edge(
            edge,
            acc,
            node_by_id,
            endpoint_ids,
            test_case_ids,
            ui_route_ids,
            flow_ids,
            flow_step_ids,
            flow_endpoint_targets,
        )

    endpoint_coverage = len(acc.tested_endpoints) / len(endpoint_ids) if endpoint_ids else 0.0
    flow_evidence_density = (
        acc.flow_evidence_total / acc.flow_step_count if acc.flow_step_count > 0 else 0.0
    )
    ui_traceability = (
        len(acc.views_with_endpoint_calls) / len(acc.ui_routed_views)
        if acc.ui_routed_views
        else 0.0
    )
    inferred_penalty = min(1.0, len(critical_inferred_only) / 10.0)

    score = (
        (endpoint_coverage * 0.35)
        + (min(flow_evidence_density / 2.0, 1.0) * 0.25)
        + (ui_traceability * 0.25)
        + ((1.0 - inferred_penalty) * 0.15)
    )
    score_100 = int(round(max(0.0, min(score, 1.0)) * 100))

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
        "known_gaps": _compute_readiness_gaps(
            endpoint_coverage,
            ui_traceability,
            flow_evidence_density,
            critical_inferred_only,
        ),
    }
