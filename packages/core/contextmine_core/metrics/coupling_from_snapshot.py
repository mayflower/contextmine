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


def compute_file_coupling_from_snapshots(
    snapshot_dicts: list[dict[str, Any]],
    repo_root: Path,
    project_root: Path,
    relevant_files: set[str],
) -> tuple[dict[str, dict[str, int | float]], dict[str, Any]]:
    """Compute afferent/efferent and bidirectional coupling from snapshots."""
    coupling_in: defaultdict[str, int] = defaultdict(int)
    coupling_out: defaultdict[str, int] = defaultdict(int)

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
            if src_file == dst_file:
                continue

            if src_file and src_file in relevant_files:
                coupling_out[src_file] += 1
            if dst_file and dst_file in relevant_files:
                coupling_in[dst_file] += 1

    if total_relations > 0 and mapped_relations == 0:
        raise MetricsGateError(
            "coupling_mapping_incomplete",
            details={
                "project_root": str(project_root),
                "relation_count": total_relations,
                "symbol_count": len(symbol_to_file),
            },
        )

    coupling_map: dict[str, dict[str, int | float]] = {}
    for file_path in relevant_files:
        incoming = int(coupling_in.get(file_path, 0))
        outgoing = int(coupling_out.get(file_path, 0))
        coupling_map[file_path] = {
            "coupling_in": incoming,
            "coupling_out": outgoing,
            "coupling": float(incoming + outgoing),
        }

    provenance = {
        "relations_total": total_relations,
        "relations_mapped": mapped_relations,
        "symbols_mapped": len(symbol_to_file),
    }
    return coupling_map, provenance
