"""CodeCharta cc.json export for twin scenarios."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, cast
from uuid import UUID

from contextmine_core.models import MetricSnapshot, TwinNode, TwinScenario
from contextmine_core.twin import GraphProjection, get_full_scenario_graph
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class _MetricValues:
    loc: int
    symbol_count: int
    coupling: float
    coverage: float
    complexity: float
    change_frequency: float


@dataclass
class _MetricAccumulator:
    loc: int = 0
    symbol_count: int = 0
    coupling_weighted: float = 0.0
    coverage_weighted: float = 0.0
    complexity_weighted: float = 0.0
    change_frequency_weighted: float = 0.0
    total_weight: float = 0.0

    def add(self, values: _MetricValues) -> None:
        self.loc += values.loc
        self.symbol_count += values.symbol_count
        weight = float(values.loc if values.loc > 0 else 1.0)
        self.total_weight += weight
        self.coupling_weighted += values.coupling * weight
        self.coverage_weighted += values.coverage * weight
        self.complexity_weighted += values.complexity * weight
        self.change_frequency_weighted += values.change_frequency * weight

    def as_values(self) -> _MetricValues:
        if self.total_weight <= 0:
            return _MetricValues(
                loc=self.loc,
                symbol_count=self.symbol_count,
                coupling=0.0,
                coverage=0.0,
                complexity=0.0,
                change_frequency=0.0,
            )
        return _MetricValues(
            loc=self.loc,
            symbol_count=self.symbol_count,
            coupling=self.coupling_weighted / self.total_weight,
            coverage=self.coverage_weighted / self.total_weight,
            complexity=self.complexity_weighted / self.total_weight,
            change_frequency=self.change_frequency_weighted / self.total_weight,
        )


def _metric_from_snapshot(snapshot: MetricSnapshot | None) -> _MetricValues:
    if snapshot is None:
        return _MetricValues(
            loc=0,
            symbol_count=0,
            coupling=0.0,
            coverage=0.0,
            complexity=0.0,
            change_frequency=0.0,
        )
    return _MetricValues(
        loc=int(snapshot.loc or 0),
        symbol_count=int(snapshot.symbol_count or 0),
        coupling=float(snapshot.coupling or 0.0),
        coverage=float(snapshot.coverage or 0.0),
        complexity=float(snapshot.complexity or 0.0),
        change_frequency=float(snapshot.change_frequency or 0.0),
    )


def _attributes_from_metrics(values: _MetricValues) -> dict[str, float | int]:
    return {
        "loc": int(values.loc),
        "symbol_count": int(values.symbol_count),
        "coupling": float(values.coupling),
        "coverage": float(values.coverage),
        "complexity": float(values.complexity),
        "change_frequency": float(values.change_frequency),
    }


def _canonical_file_path(node: TwinNode) -> str | None:
    natural_key = str(node.natural_key or "")
    if natural_key.startswith("file:"):
        return natural_key.split(":", 1)[1].strip() or None

    meta = node.meta or {}
    file_path = meta.get("file_path")
    if isinstance(file_path, str) and file_path.strip():
        return file_path.strip()
    return None


def _derive_arch_group(
    path: str | None,
    meta: dict,
) -> tuple[str, str, str] | None:
    architecture_meta = meta.get("architecture")
    if isinstance(architecture_meta, dict):
        explicit_domain = str(architecture_meta.get("domain") or "").strip()
        explicit_container = str(architecture_meta.get("container") or "").strip()
        explicit_component = str(architecture_meta.get("component") or "").strip()
        if explicit_domain and explicit_container:
            component = explicit_component or explicit_container
            return explicit_domain, explicit_container, component

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
    return domain, container, component


def _arch_group_key(
    entity_level: str,
    group: tuple[str, str, str],
) -> str:
    domain, container, component = group
    if entity_level == "domain":
        return f"domain|{domain}||"
    if entity_level == "component":
        return f"component|{domain}|{container}|{component}"
    return f"container|{domain}|{container}|"


def _node_path_segments(
    projection: GraphProjection,
    node: dict,
    entity_level: str,
) -> list[str]:
    if projection == GraphProjection.CODE_FILE:
        meta = node.get("meta") or {}
        file_path = meta.get("file_path")
        if isinstance(file_path, str) and file_path.strip():
            return [part for part in file_path.strip("/").split("/") if part]
        natural_key = str(node.get("natural_key") or "")
        if natural_key.startswith("file:"):
            return [part for part in natural_key.split(":", 1)[1].strip("/").split("/") if part]
        name = str(node.get("name") or "unknown")
        return [name]

    meta = node.get("meta") or {}
    domain = str(meta.get("domain") or "").strip()
    container = str(meta.get("container") or "").strip()
    component = str(meta.get("component") or "").strip()

    if entity_level == "domain":
        return [domain or str(node.get("name") or "domain")]
    if entity_level == "component":
        return [
            domain or "domain",
            container or "container",
            component or str(node.get("name") or "component"),
        ]
    return [
        domain or "domain",
        container or str(node.get("name") or "container"),
    ]


def _unique_leaf_paths(
    items: list[tuple[str, list[str], dict[str, float | int]]],
) -> tuple[
    list[tuple[str, list[str], dict[str, float | int]]],
    dict[str, str],
]:
    seen_paths: set[tuple[str, ...]] = set()
    with_unique_paths: list[tuple[str, list[str], dict[str, float | int]]] = []
    absolute_path_by_id: dict[str, str] = {}

    for node_id, path_segments, attributes in items:
        segments = [seg for seg in path_segments if seg]
        if not segments:
            segments = [str(node_id)]
        candidate = segments[:]
        suffix = 2
        key = tuple(candidate)
        while key in seen_paths:
            candidate = [*segments[:-1], f"{segments[-1]}_{suffix}"]
            key = tuple(candidate)
            suffix += 1
        seen_paths.add(key)
        with_unique_paths.append((node_id, candidate, attributes))
        absolute_path_by_id[node_id] = "/" + "/".join(["root", *candidate])

    return with_unique_paths, absolute_path_by_id


def _tree_from_leaf_items(
    items: list[tuple[str, list[str], dict[str, float | int]]],
) -> tuple[dict, dict[str, str]]:
    unique_items, absolute_path_by_id = _unique_leaf_paths(items)
    root: dict[str, Any] = {
        "name": "root",
        "type": "Folder",
        "attributes": {},
        "children": [],
    }

    for _, path_segments, attributes in unique_items:
        cursor: dict[str, Any] = root
        for depth, segment in enumerate(path_segments):
            is_leaf = depth == len(path_segments) - 1
            children = cast(list[dict[str, Any]], cursor.setdefault("children", []))
            next_node = next((child for child in children if child["name"] == segment), None)
            if next_node is None:
                next_node = {
                    "name": segment,
                    "type": "File" if is_leaf else "Folder",
                    "attributes": attributes if is_leaf else {},
                }
                if not is_leaf:
                    next_node["children"] = []
                children.append(next_node)
            cursor = next_node

    return root, absolute_path_by_id


def _checksum(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


async def export_codecharta_json(
    session: AsyncSession,
    scenario_id: UUID,
    projection: GraphProjection = GraphProjection.ARCHITECTURE,
    entity_level: str | None = None,
) -> str:
    """Export scenario metrics in an official CodeCharta cc.json schema."""
    if projection == GraphProjection.CODE_SYMBOL:
        projection = GraphProjection.CODE_FILE

    effective_entity_level = "container" if projection == GraphProjection.ARCHITECTURE else "file"
    if entity_level:
        effective_entity_level = entity_level

    scenario = (
        await session.execute(select(TwinScenario).where(TwinScenario.id == scenario_id))
    ).scalar_one_or_none()

    metrics = (
        (
            await session.execute(
                select(MetricSnapshot).where(MetricSnapshot.scenario_id == scenario_id)
            )
        )
        .scalars()
        .all()
    )

    metric_by_natural_key = {metric.node_natural_key: metric for metric in metrics}

    graph = await get_full_scenario_graph(
        session=session,
        scenario_id=scenario_id,
        layer=None,
        projection=projection,
        entity_level=effective_entity_level,
        include_kinds={"file"} if projection == GraphProjection.ARCHITECTURE else None,
    )

    leaf_items: list[tuple[str, list[str], dict[str, float | int]]] = []
    if projection == GraphProjection.CODE_FILE:
        for node in graph["nodes"]:
            node_id = str(node["id"])
            values = _metric_from_snapshot(
                metric_by_natural_key.get(str(node.get("natural_key") or ""))
            )
            leaf_items.append(
                (
                    node_id,
                    _node_path_segments(projection, node, effective_entity_level),
                    _attributes_from_metrics(values),
                )
            )
    else:
        file_nodes = (
            (
                await session.execute(
                    select(TwinNode).where(
                        TwinNode.scenario_id == scenario_id,
                        TwinNode.kind == "file",
                    )
                )
            )
            .scalars()
            .all()
        )
        aggregated: dict[str, _MetricAccumulator] = {}
        for file_node in file_nodes:
            group = _derive_arch_group(_canonical_file_path(file_node), file_node.meta or {})
            if not group:
                continue
            key = _arch_group_key(effective_entity_level, group)
            bucket = aggregated.setdefault(key, _MetricAccumulator())
            bucket.add(_metric_from_snapshot(metric_by_natural_key.get(file_node.natural_key)))

        for node in graph["nodes"]:
            node_id = str(node["id"])
            key = str(node.get("natural_key") or "")
            values = aggregated.get(key, _MetricAccumulator()).as_values()
            leaf_items.append(
                (
                    node_id,
                    _node_path_segments(projection, node, effective_entity_level),
                    _attributes_from_metrics(values),
                )
            )

    root, path_by_id = _tree_from_leaf_items(leaf_items)

    cc_edges = []
    for edge in graph["edges"]:
        source = path_by_id.get(str(edge.get("source_node_id")))
        target = path_by_id.get(str(edge.get("target_node_id")))
        if not source or not target or source == target:
            continue
        weight = float((edge.get("meta") or {}).get("weight") or 1.0)
        cc_edges.append(
            {
                "fromNodeName": source,
                "toNodeName": target,
                "attributes": {
                    "dependency_weight": weight,
                },
            }
        )

    payload = {
        "projectName": scenario.name if scenario else f"scenario-{scenario_id}",
        "apiVersion": "1.5",
        "nodes": [root],
        "edges": cc_edges,
        "attributeTypes": {
            "nodes": {
                "loc": "absolute",
                "symbol_count": "absolute",
                "coupling": "absolute",
                "coverage": "relative",
                "complexity": "absolute",
                "change_frequency": "absolute",
            },
            "edges": {
                "dependency_weight": "absolute",
            },
        },
    }
    payload["fileChecksum"] = _checksum(payload)
    return json.dumps(payload, indent=2)
