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
