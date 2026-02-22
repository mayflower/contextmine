"""Coupling calculation from semantic snapshots."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from contextmine_core.metrics.discovery import to_repo_relative_path
from contextmine_core.metrics.models import MetricsGateError
from contextmine_core.semantic_snapshot.models import Snapshot


def _snapshot_project_root(snapshot: Snapshot, fallback: Path) -> Path:
    raw = str(snapshot.meta.get("project_root", "") or "")
    if not raw:
        return fallback
    return Path(raw)


def _compute_scc(
    nodes: set[str],
    edges: set[tuple[str, str]],
) -> list[set[str]]:
    adjacency: dict[str, list[str]] = {node: [] for node in nodes}
    for src, dst in edges:
        adjacency.setdefault(src, [])
        adjacency.setdefault(dst, [])
        adjacency[src].append(dst)

    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    components: list[set[str]] = []

    def strongconnect(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for neighbor in adjacency.get(node, []):
            if neighbor not in indices:
                strongconnect(neighbor)
                lowlink[node] = min(lowlink[node], lowlink[neighbor])
            elif neighbor in on_stack:
                lowlink[node] = min(lowlink[node], indices[neighbor])

        if lowlink[node] == indices[node]:
            component: set[str] = set()
            while stack:
                candidate = stack.pop()
                on_stack.discard(candidate)
                component.add(candidate)
                if candidate == node:
                    break
            components.append(component)

    for node in nodes:
        if node not in indices:
            strongconnect(node)

    return components


def compute_file_coupling_from_snapshots(
    snapshot_dicts: list[dict[str, Any]],
    repo_root: Path,
    project_root: Path,
    relevant_files: set[str],
) -> tuple[dict[str, dict[str, int | float | bool]], dict[str, Any]]:
    """Compute afferent/efferent and bidirectional coupling from snapshots."""
    coupling_in: defaultdict[str, int] = defaultdict(int)
    coupling_out: defaultdict[str, int] = defaultdict(int)
    internal_calls: defaultdict[str, int] = defaultdict(int)
    outgoing_calls: defaultdict[str, int] = defaultdict(int)
    fan_in: defaultdict[str, set[str]] = defaultdict(set)
    fan_out: defaultdict[str, set[str]] = defaultdict(set)
    inter_file_edges: set[tuple[str, str]] = set()

    symbol_to_file: dict[str, str] = {}
    snapshots = [Snapshot.from_dict(snapshot_dict) for snapshot_dict in snapshot_dicts]

    for snapshot in snapshots:
        snap_project_root = _snapshot_project_root(snapshot, project_root)
        for symbol in snapshot.symbols:
            if symbol.file_path == "<external>":
                continue
            file_path = to_repo_relative_path(
                symbol.file_path,
                repo_root=repo_root,
                project_root=snap_project_root,
            )
            if not file_path:
                continue
            symbol_to_file[symbol.def_id] = file_path

    total_relations = 0
    mapped_relations = 0

    for snapshot in snapshots:
        for relation in snapshot.relations:
            total_relations += 1
            src_file = symbol_to_file.get(relation.src_def_id)
            dst_file = symbol_to_file.get(relation.dst_def_id)

            if src_file is None and dst_file is None:
                continue

            mapped_relations += 1
            if src_file and src_file in relevant_files:
                outgoing_calls[src_file] += 1
                if relation.dst_def_id:
                    fan_out[src_file].add(str(relation.dst_def_id))
            if dst_file and dst_file in relevant_files and relation.src_def_id:
                fan_in[dst_file].add(str(relation.src_def_id))

            if src_file == dst_file:
                if src_file and src_file in relevant_files:
                    internal_calls[src_file] += 1
                continue

            if src_file and src_file in relevant_files:
                coupling_out[src_file] += 1
            if dst_file and dst_file in relevant_files:
                coupling_in[dst_file] += 1
            if (
                src_file
                and dst_file
                and src_file in relevant_files
                and dst_file in relevant_files
                and src_file != dst_file
            ):
                inter_file_edges.add((src_file, dst_file))

    if total_relations > 0 and mapped_relations == 0:
        raise MetricsGateError(
            "coupling_mapping_incomplete",
            details={
                "project_root": str(project_root),
                "relation_count": total_relations,
                "symbol_count": len(symbol_to_file),
            },
        )

    components = _compute_scc(set(relevant_files), inter_file_edges)
    component_size_by_file: dict[str, int] = {}
    for component in components:
        if len(component) <= 1:
            continue
        component_size = len(component)
        for file_path in component:
            component_size_by_file[file_path] = component_size

    coupling_map: dict[str, dict[str, int | float | bool]] = {}
    for file_path in relevant_files:
        incoming = int(coupling_in.get(file_path, 0))
        outgoing = int(coupling_out.get(file_path, 0))
        internal = int(internal_calls.get(file_path, 0))
        call_total = int(outgoing_calls.get(file_path, 0))
        cohesion = 1.0 if call_total == 0 else float(internal / call_total)
        instability = float(outgoing / (incoming + outgoing)) if (incoming + outgoing) > 0 else 0.0
        cycle_size = int(component_size_by_file.get(file_path, 0))
        coupling_map[file_path] = {
            "coupling_in": incoming,
            "coupling_out": outgoing,
            "coupling": float(incoming + outgoing),
            "cohesion": cohesion,
            "instability": instability,
            "fan_in": int(len(fan_in.get(file_path, set()))),
            "fan_out": int(len(fan_out.get(file_path, set()))),
            "cycle_participation": bool(cycle_size > 1),
            "cycle_size": cycle_size,
        }

    provenance = {
        "relations_total": total_relations,
        "relations_mapped": mapped_relations,
        "symbols_mapped": len(symbol_to_file),
        "inter_file_edges": len(inter_file_edges),
        "cyclic_components": sum(1 for c in components if len(c) > 1),
    }
    return coupling_map, provenance
