"""Evolution analytics for Architecture Cockpit (investment, ownership, coupling, fitness)."""

from __future__ import annotations

import math
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import PurePosixPath
from typing import Any
from uuid import UUID

from contextmine_core.models import (
    MetricSnapshot,
    TwinEdge,
    TwinFinding,
    TwinNode,
    TwinOwnershipSnapshot,
    TwinTemporalCouplingSnapshot,
)
from contextmine_core.pathing import canonicalize_repo_relative_path
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

DEFAULT_EVOLUTION_WINDOW_DAYS = 365
DEFAULT_MIN_JACCARD = 0.2
DEFAULT_MAX_COUPLING_EDGES = 300

_SUPPORTED_ENTITY_LEVELS = {"file", "container", "component"}
_SEVERITY_WEIGHT = {"critical": 4, "high": 3, "medium": 2, "low": 1}


@dataclass(frozen=True)
class EntityGroup:
    domain: str
    container: str
    component: str


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _coverage_value(value: float | None) -> float | None:
    if value is None:
        return None
    return float(max(0.0, min(100.0, value)))


def _normalize_min_max(value: float, min_value: float, max_value: float) -> float:
    if max_value <= min_value:
        return 1.0 if value > 0 else 0.0
    return (value - min_value) / (max_value - min_value)


def _file_path_from_natural_key(node_natural_key: str) -> str | None:
    if not node_natural_key.startswith("file:"):
        return None
    raw = node_natural_key.removeprefix("file:").strip()
    if not raw:
        return None
    return canonicalize_repo_relative_path(raw)


def derive_arch_group(path: str | None, meta: dict[str, Any] | None = None) -> EntityGroup | None:
    """Resolve domain/container/component from explicit architecture meta or heuristics."""
    payload = meta or {}
    architecture_meta = payload.get("architecture")
    if isinstance(architecture_meta, dict):
        explicit_domain = str(architecture_meta.get("domain") or "").strip()
        explicit_container = str(architecture_meta.get("container") or "").strip()
        explicit_component = str(architecture_meta.get("component") or "").strip()
        if explicit_domain and explicit_container:
            return EntityGroup(
                domain=explicit_domain,
                container=explicit_container,
                component=explicit_component or explicit_container,
            )

    if not path:
        return None

    normalized = canonicalize_repo_relative_path(path)
    if not normalized:
        return None

    parts = [part for part in normalized.strip("/").split("/") if part]
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
    return EntityGroup(domain=domain, container=container, component=component)


def build_entity_key(
    path: str | None, meta: dict[str, Any] | None, entity_level: str
) -> str | None:
    if entity_level == "file":
        canonical = canonicalize_repo_relative_path(path or "")
        return f"file:{canonical}" if canonical else None

    group = derive_arch_group(path, meta)
    if not group:
        return None

    if entity_level == "container":
        return f"container:{group.domain}/{group.container}"
    if entity_level == "component":
        return f"component:{group.domain}/{group.container}/{group.component}"
    return None


def _display_label(entity_key: str) -> str:
    if ":" not in entity_key:
        return entity_key
    return entity_key.split(":", 1)[1]


def _bus_factor(contributions: dict[str, float], threshold: float = 0.8) -> int:
    if not contributions:
        return 0
    total = sum(float(value) for value in contributions.values())
    if total <= 0:
        return 0

    running = 0.0
    for index, value in enumerate(sorted(contributions.values(), reverse=True), start=1):
        running += float(value)
        if running / total >= threshold:
            return index
    return len(contributions)


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]

    bounded = max(0.0, min(1.0, percentile))
    index = int(math.ceil((len(sorted_values) - 1) * bounded))
    return sorted_values[index]


def _topological_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """Return SCCs larger than 1 as cycle candidates."""
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    cycles: list[list[str]] = []

    def strongconnect(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for neighbor in graph.get(node, set()):
            if neighbor not in indices:
                strongconnect(neighbor)
                lowlink[node] = min(lowlink[node], lowlink[neighbor])
            elif neighbor in on_stack:
                lowlink[node] = min(lowlink[node], indices[neighbor])

        if lowlink[node] == indices[node]:
            component: list[str] = []
            while stack:
                current = stack.pop()
                on_stack.discard(current)
                component.append(current)
                if current == node:
                    break
            if len(component) > 1:
                cycles.append(sorted(component))

    for node in sorted(graph):
        if node not in indices:
            strongconnect(node)

    cycles.sort(key=lambda component: (-len(component), component))
    return cycles


async def replace_evolution_snapshots(
    session: AsyncSession,
    *,
    scenario_id: UUID,
    ownership_rows: list[dict[str, Any]],
    coupling_rows: list[dict[str, Any]],
) -> dict[str, int]:
    """Replace snapshot rows for one scenario atomically within current transaction."""
    await session.execute(
        delete(TwinOwnershipSnapshot).where(TwinOwnershipSnapshot.scenario_id == scenario_id)
    )
    await session.execute(
        delete(TwinTemporalCouplingSnapshot).where(
            TwinTemporalCouplingSnapshot.scenario_id == scenario_id
        )
    )

    ownership_models: list[TwinOwnershipSnapshot] = []
    for row in ownership_rows:
        ownership_models.append(
            TwinOwnershipSnapshot(
                id=uuid.uuid4(),
                scenario_id=scenario_id,
                node_natural_key=str(row.get("node_natural_key", "")),
                author_key=str(row.get("author_key", "unknown")),
                author_label=str(row.get("author_label", "unknown")),
                additions=_safe_int(row.get("additions")),
                deletions=_safe_int(row.get("deletions")),
                touches=_safe_int(row.get("touches")),
                ownership_share=_safe_float(row.get("ownership_share")),
                last_touched_at=row.get("last_touched_at"),
                window_days=_safe_int(row.get("window_days"), DEFAULT_EVOLUTION_WINDOW_DAYS),
            )
        )

    coupling_models: list[TwinTemporalCouplingSnapshot] = []
    for row in coupling_rows:
        entity_level = str(row.get("entity_level", "file")).strip().lower()
        if entity_level not in _SUPPORTED_ENTITY_LEVELS:
            continue
        coupling_models.append(
            TwinTemporalCouplingSnapshot(
                id=uuid.uuid4(),
                scenario_id=scenario_id,
                entity_level=entity_level,
                source_key=str(row.get("source_key", "")),
                target_key=str(row.get("target_key", "")),
                co_change_count=_safe_int(row.get("co_change_count")),
                source_change_count=_safe_int(row.get("source_change_count")),
                target_change_count=_safe_int(row.get("target_change_count")),
                ratio_source_to_target=_safe_float(row.get("ratio_source_to_target")),
                ratio_target_to_source=_safe_float(row.get("ratio_target_to_source")),
                jaccard=_safe_float(row.get("jaccard")),
                cross_boundary=bool(row.get("cross_boundary", False)),
                window_days=_safe_int(row.get("window_days"), DEFAULT_EVOLUTION_WINDOW_DAYS),
            )
        )

    if ownership_models:
        session.add_all(ownership_models)
    if coupling_models:
        session.add_all(coupling_models)

    return {
        "ownership_rows": len(ownership_models),
        "coupling_rows": len(coupling_models),
    }


async def _metric_context(
    session: AsyncSession,
    scenario_id: UUID,
) -> tuple[list[MetricSnapshot], dict[str, dict[str, Any]], dict[str, str]]:
    metrics: list[MetricSnapshot] = list(
        (
            await session.execute(
                select(MetricSnapshot).where(MetricSnapshot.scenario_id == scenario_id)
            )
        )
        .scalars()
        .all()
    )
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

    node_meta = {node.natural_key: dict(node.meta or {}) for node in file_nodes}
    path_to_natural_key: dict[str, str] = {}
    for node in file_nodes:
        path = _file_path_from_natural_key(node.natural_key)
        if path:
            path_to_natural_key[path] = node.natural_key
    return metrics, node_meta, path_to_natural_key


async def get_investment_utilization_payload(
    session: AsyncSession,
    *,
    scenario_id: UUID,
    entity_level: str,
    window_days: int,
) -> dict[str, Any]:
    if entity_level not in {"container", "component"}:
        raise ValueError("entity_level must be 'container' or 'component'")

    metrics, node_meta, _ = await _metric_context(session, scenario_id)
    if not metrics:
        return {
            "status": "unavailable",
            "reason": "no_real_metrics",
            "window_days": window_days,
            "summary": {
                "total_entities": 0,
                "coverage_entity_ratio": 0.0,
                "utilization_available": False,
                "quadrants": {
                    "strength": 0,
                    "overinvestment": 0,
                    "efficient_core": 0,
                    "opportunity_or_retire": 0,
                    "unknown": 0,
                },
            },
            "items": [],
            "warnings": ["No metric snapshots available for selected scenario."],
        }

    groups: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "loc": 0,
            "change_frequency_sum": 0.0,
            "churn_sum": 0.0,
            "coverage_sum": 0.0,
            "coverage_count": 0,
            "files": 0,
        }
    )

    for metric in metrics:
        path = _file_path_from_natural_key(metric.node_natural_key)
        meta = node_meta.get(metric.node_natural_key, {})
        entity_key = build_entity_key(path, meta, entity_level)
        if not entity_key:
            continue

        row = groups[entity_key]
        row["loc"] += _safe_int(metric.loc)
        row["files"] += 1
        row["change_frequency_sum"] += _safe_float(metric.change_frequency)
        row["churn_sum"] += _safe_float((metric.meta or {}).get("churn"))
        coverage = _coverage_value(metric.coverage)
        if coverage is not None:
            row["coverage_sum"] += coverage
            row["coverage_count"] += 1

    if not groups:
        return {
            "status": "unavailable",
            "reason": "no_mappable_entities",
            "window_days": window_days,
            "summary": {
                "total_entities": 0,
                "coverage_entity_ratio": 0.0,
                "utilization_available": False,
                "quadrants": {
                    "strength": 0,
                    "overinvestment": 0,
                    "efficient_core": 0,
                    "opportunity_or_retire": 0,
                    "unknown": 0,
                },
            },
            "items": [],
            "warnings": ["No metrics could be mapped to architecture entities."],
        }

    churn_values: list[float] = []
    change_values: list[float] = []
    enriched: dict[str, dict[str, Any]] = {}

    for entity_key, row in groups.items():
        files = max(1, int(row["files"]))
        churn_avg = float(row["churn_sum"]) / files
        change_avg = float(row["change_frequency_sum"]) / files
        churn_log = math.log1p(max(churn_avg, 0.0))
        churn_values.append(churn_log)
        change_values.append(change_avg)

        coverage_avg = None
        if row["coverage_count"] > 0:
            coverage_avg = float(row["coverage_sum"]) / float(row["coverage_count"])

        enriched[entity_key] = {
            "entity_key": entity_key,
            "label": _display_label(entity_key),
            "loc": int(row["loc"]),
            "files": files,
            "churn_avg": round(churn_avg, 4),
            "change_frequency_avg": round(change_avg, 4),
            "coverage_avg": round(coverage_avg, 4) if coverage_avg is not None else None,
            "_churn_log": churn_log,
            "_change_raw": change_avg,
        }

    churn_min, churn_max = min(churn_values), max(churn_values)
    change_min, change_max = min(change_values), max(change_values)

    entities_with_coverage = sum(1 for row in enriched.values() if row["coverage_avg"] is not None)
    coverage_entity_ratio = _safe_ratio(float(entities_with_coverage), float(len(enriched)))
    utilization_available = coverage_entity_ratio >= 0.3

    quadrants = {
        "strength": 0,
        "overinvestment": 0,
        "efficient_core": 0,
        "opportunity_or_retire": 0,
        "unknown": 0,
    }

    items: list[dict[str, Any]] = []
    for entity_key, row in sorted(enriched.items(), key=lambda item: item[0]):
        churn_norm = _normalize_min_max(float(row["_churn_log"]), churn_min, churn_max)
        change_norm = _normalize_min_max(float(row["_change_raw"]), change_min, change_max)
        investment_score = round((0.6 * churn_norm) + (0.4 * change_norm), 4)

        utilization_score = None
        if utilization_available and row["coverage_avg"] is not None:
            utilization_score = round(float(row["coverage_avg"]) / 100.0, 4)

        if utilization_score is None:
            quadrant = "unknown"
        elif investment_score >= 0.5 and utilization_score >= 0.5:
            quadrant = "strength"
        elif investment_score >= 0.5 and utilization_score < 0.5:
            quadrant = "overinvestment"
        elif investment_score < 0.5 and utilization_score >= 0.5:
            quadrant = "efficient_core"
        else:
            quadrant = "opportunity_or_retire"

        quadrants[quadrant] += 1

        items.append(
            {
                "entity_key": entity_key,
                "label": row["label"],
                "size": int(row["loc"]),
                "investment_score": investment_score,
                "utilization_score": utilization_score,
                "coverage_avg": row["coverage_avg"],
                "change_frequency_avg": row["change_frequency_avg"],
                "churn_avg": row["churn_avg"],
                "quadrant": quadrant,
            }
        )

    warnings: list[str] = []
    if not utilization_available:
        warnings.append(
            "Coverage coverage across entities is below 30%; utilization axis suppressed."
        )

    items.sort(
        key=lambda item: (
            float(item["investment_score"]),
            float(item["utilization_score"] or 0.0),
            float(item["size"]),
        ),
        reverse=True,
    )

    return {
        "status": "ready",
        "reason": "ok",
        "window_days": window_days,
        "summary": {
            "total_entities": len(items),
            "coverage_entity_ratio": round(coverage_entity_ratio, 4),
            "utilization_available": utilization_available,
            "quadrants": quadrants,
        },
        "items": items,
        "warnings": warnings,
    }


async def get_knowledge_islands_payload(
    session: AsyncSession,
    *,
    scenario_id: UUID,
    entity_level: str,
    window_days: int,
    ownership_threshold: float,
) -> dict[str, Any]:
    if entity_level not in {"container", "component"}:
        raise ValueError("entity_level must be 'container' or 'component'")

    ownership_rows = (
        (
            await session.execute(
                select(TwinOwnershipSnapshot).where(
                    TwinOwnershipSnapshot.scenario_id == scenario_id,
                    TwinOwnershipSnapshot.window_days == window_days,
                )
            )
        )
        .scalars()
        .all()
    )

    if not ownership_rows:
        return {
            "status": "unavailable",
            "reason": "no_git_history",
            "window_days": window_days,
            "summary": {
                "files": 0,
                "entities": 0,
                "bus_factor_global": 0,
                "single_owner_files": 0,
            },
            "entities": [],
            "at_risk_files": [],
            "warnings": ["No ownership snapshots available for selected scenario/window."],
        }

    metrics_rows = (
        (
            await session.execute(
                select(MetricSnapshot).where(MetricSnapshot.scenario_id == scenario_id)
            )
        )
        .scalars()
        .all()
    )
    churn_by_file = {
        row.node_natural_key: _safe_float((row.meta or {}).get("churn")) for row in metrics_rows
    }
    coverage_by_file = {row.node_natural_key: _coverage_value(row.coverage) for row in metrics_rows}

    by_file: dict[str, list[TwinOwnershipSnapshot]] = defaultdict(list)
    for row in ownership_rows:
        by_file[row.node_natural_key].append(row)

    entity_contrib: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    entity_files: dict[str, list[dict[str, Any]]] = defaultdict(list)
    global_contrib: dict[str, float] = defaultdict(float)
    dominant_files: list[dict[str, Any]] = []

    for node_natural_key, rows in by_file.items():
        path = _file_path_from_natural_key(node_natural_key)
        entity_key = build_entity_key(path, {}, entity_level)
        if not entity_key:
            continue

        dominant = max(rows, key=lambda row: (float(row.ownership_share), int(row.additions)))
        dominant_share = float(dominant.ownership_share)
        additions_total = float(sum(int(row.additions) for row in rows))
        last_touched = max(
            (row.last_touched_at for row in rows if row.last_touched_at), default=None
        )

        for row in rows:
            entity_contrib[entity_key][row.author_label] += float(row.additions)
            global_contrib[row.author_label] += float(row.additions)

        single_owner = dominant_share >= ownership_threshold
        file_row = {
            "node_natural_key": node_natural_key,
            "path": path,
            "entity_key": entity_key,
            "dominant_owner": dominant.author_label,
            "dominant_share": round(dominant_share, 4),
            "additions_total": int(additions_total),
            "touches": int(sum(int(row.touches) for row in rows)),
            "single_owner": single_owner,
            "churn": round(churn_by_file.get(node_natural_key, 0.0), 4),
            "coverage": coverage_by_file.get(node_natural_key),
            "last_touched_at": last_touched.isoformat() if last_touched else None,
        }

        entity_files[entity_key].append(file_row)
        dominant_files.append(file_row)

    entities: list[dict[str, Any]] = []
    for entity_key, files in sorted(entity_files.items(), key=lambda item: item[0]):
        contributions = entity_contrib[entity_key]
        bus_factor = _bus_factor(contributions, threshold=0.8)
        dominant_owner = (
            max(contributions.items(), key=lambda item: item[1])[0] if contributions else None
        )
        total_additions = float(sum(contributions.values()))
        dominant_share = (
            round(
                _safe_ratio(float(contributions.get(dominant_owner or "", 0.0)), total_additions), 4
            )
            if dominant_owner
            else 0.0
        )
        single_owner_files = sum(1 for file_row in files if file_row["single_owner"])

        entities.append(
            {
                "entity_key": entity_key,
                "label": _display_label(entity_key),
                "files": len(files),
                "bus_factor": bus_factor,
                "dominant_owner": dominant_owner,
                "dominant_share": dominant_share,
                "single_owner_ratio": round(
                    _safe_ratio(float(single_owner_files), float(len(files))), 4
                ),
            }
        )

    churn_values = sorted(
        [float(row["churn"]) for row in dominant_files if float(row["churn"]) > 0],
    )
    churn_p75 = _percentile(churn_values, 0.75)

    at_risk_files = [
        row
        for row in dominant_files
        if float(row["dominant_share"]) >= ownership_threshold and float(row["churn"]) >= churn_p75
    ]
    at_risk_files.sort(
        key=lambda row: (
            float(row["churn"]),
            float(row["dominant_share"]),
            int(row["touches"]),
        ),
        reverse=True,
    )

    bus_factor_global = _bus_factor(global_contrib, threshold=0.8)

    entities.sort(
        key=lambda row: (
            int(row["bus_factor"]),
            -float(row["single_owner_ratio"]),
            row["label"],
        )
    )

    return {
        "status": "ready",
        "reason": "ok",
        "window_days": window_days,
        "summary": {
            "files": len(dominant_files),
            "entities": len(entities),
            "bus_factor_global": bus_factor_global,
            "single_owner_files": sum(1 for row in dominant_files if row["single_owner"]),
            "churn_p75": round(churn_p75, 4),
        },
        "entities": entities,
        "at_risk_files": at_risk_files[:50],
        "warnings": [],
    }


async def get_temporal_coupling_payload(
    session: AsyncSession,
    *,
    scenario_id: UUID,
    entity_level: str,
    window_days: int,
    min_jaccard: float,
    max_edges: int,
) -> dict[str, Any]:
    if entity_level not in _SUPPORTED_ENTITY_LEVELS:
        raise ValueError("entity_level must be one of: file, container, component")

    rows = (
        (
            await session.execute(
                select(TwinTemporalCouplingSnapshot)
                .where(
                    TwinTemporalCouplingSnapshot.scenario_id == scenario_id,
                    TwinTemporalCouplingSnapshot.entity_level == entity_level,
                    TwinTemporalCouplingSnapshot.window_days == window_days,
                    TwinTemporalCouplingSnapshot.jaccard >= min_jaccard,
                )
                .order_by(
                    TwinTemporalCouplingSnapshot.jaccard.desc(),
                    TwinTemporalCouplingSnapshot.co_change_count.desc(),
                )
                .limit(max_edges)
            )
        )
        .scalars()
        .all()
    )

    if not rows:
        return {
            "status": "unavailable",
            "reason": "no_temporal_coupling",
            "window_days": window_days,
            "entity_level": entity_level,
            "summary": {
                "nodes": 0,
                "edges": 0,
                "cross_boundary_edges": 0,
                "avg_jaccard": 0.0,
            },
            "graph": {"nodes": [], "edges": []},
            "warnings": ["No coupling edges matched current filters."],
        }

    node_keys: set[str] = set()
    edges: list[dict[str, Any]] = []
    for row in rows:
        node_keys.add(row.source_key)
        node_keys.add(row.target_key)
        edges.append(
            {
                "id": str(row.id),
                "source": row.source_key,
                "target": row.target_key,
                "co_change_count": int(row.co_change_count),
                "source_change_count": int(row.source_change_count),
                "target_change_count": int(row.target_change_count),
                "ratio_source_to_target": round(float(row.ratio_source_to_target), 4),
                "ratio_target_to_source": round(float(row.ratio_target_to_source), 4),
                "jaccard": round(float(row.jaccard), 4),
                "cross_boundary": bool(row.cross_boundary),
            }
        )

    nodes = [
        {
            "id": key,
            "key": key,
            "label": _display_label(key),
            "entity_level": entity_level,
        }
        for key in sorted(node_keys)
    ]

    avg_jaccard = sum(float(edge["jaccard"]) for edge in edges) / len(edges)

    return {
        "status": "ready",
        "reason": "ok",
        "window_days": window_days,
        "entity_level": entity_level,
        "summary": {
            "nodes": len(nodes),
            "edges": len(edges),
            "cross_boundary_edges": sum(1 for edge in edges if edge["cross_boundary"]),
            "avg_jaccard": round(avg_jaccard, 4),
        },
        "graph": {
            "nodes": nodes,
            "edges": edges,
        },
        "warnings": [],
    }


async def evaluate_and_store_fitness_findings(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID,
    window_days: int,
) -> dict[str, Any]:
    """Evaluate advisory fitness rules and persist findings for the scenario."""

    findings: list[dict[str, Any]] = []
    warnings: list[str] = []
    now = datetime.now(UTC)

    # FF001: Cyclic dependencies on derived component graph.
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
    node_to_component: dict[UUID, str] = {}
    file_path_by_component: dict[str, str] = {}
    for node in file_nodes:
        path = _file_path_from_natural_key(node.natural_key)
        component_key = build_entity_key(path, node.meta or {}, "component")
        if not component_key:
            continue
        node_to_component[node.id] = component_key
        if path and component_key not in file_path_by_component:
            file_path_by_component[component_key] = path

    edge_rows = (
        (await session.execute(select(TwinEdge).where(TwinEdge.scenario_id == scenario_id)))
        .scalars()
        .all()
    )
    dep_tokens = ("depends", "calls", "imports", "references", "uses", "invokes")
    component_graph: dict[str, set[str]] = defaultdict(set)
    for edge in edge_rows:
        kind = str(edge.kind or "").lower()
        if not any(token in kind for token in dep_tokens):
            continue

        source_component = node_to_component.get(edge.source_node_id)
        target_component = node_to_component.get(edge.target_node_id)
        if not source_component or not target_component:
            continue
        if source_component == target_component:
            continue
        component_graph[source_component].add(target_component)

    for cycle in _topological_cycles(component_graph)[:200]:
        subject = " -> ".join(cycle)
        anchor = cycle[0]
        findings.append(
            {
                "finding_type": "fitness.ff001_cyclic_dependencies",
                "severity": "high",
                "confidence": "high",
                "status": "open",
                "filename": file_path_by_component.get(anchor, "__architecture__/component_cycle"),
                "line_number": 1,
                "message": f"Cyclic component dependency detected: {subject}",
                "meta": {
                    "rule_id": "FF001_cyclic_dependencies",
                    "subject": subject,
                    "cycle_components": cycle,
                },
            }
        )

    # FF002: Overinvestment low utilization (component).
    investment_payload = await get_investment_utilization_payload(
        session,
        scenario_id=scenario_id,
        entity_level="component",
        window_days=window_days,
    )
    if investment_payload.get("status") == "ready":
        for item in investment_payload.get("items", [])[:400]:
            if item.get("quadrant") != "overinvestment":
                continue
            entity_key = str(item.get("entity_key", ""))
            findings.append(
                {
                    "finding_type": "fitness.ff002_overinvestment_low_utilization",
                    "severity": "medium",
                    "confidence": "medium",
                    "status": "open",
                    "filename": file_path_by_component.get(
                        entity_key, "__architecture__/investment"
                    ),
                    "line_number": 1,
                    "message": (
                        f"Overinvestment detected for {item.get('label')}: "
                        f"investment={item.get('investment_score')}, utilization={item.get('utilization_score')}"
                    ),
                    "meta": {
                        "rule_id": "FF002_overinvestment_low_utilization",
                        "subject": entity_key,
                        "investment_score": item.get("investment_score"),
                        "utilization_score": item.get("utilization_score"),
                    },
                }
            )
    else:
        warnings.append("FF002 skipped: investment/utilization payload unavailable")

    # FF003 and FF005 use ownership + metric overlays.
    ownership_rows = (
        (
            await session.execute(
                select(TwinOwnershipSnapshot).where(
                    TwinOwnershipSnapshot.scenario_id == scenario_id,
                    TwinOwnershipSnapshot.window_days == window_days,
                )
            )
        )
        .scalars()
        .all()
    )
    metrics_rows = (
        (
            await session.execute(
                select(MetricSnapshot).where(MetricSnapshot.scenario_id == scenario_id)
            )
        )
        .scalars()
        .all()
    )
    churn_by_file = {
        row.node_natural_key: _safe_float((row.meta or {}).get("churn")) for row in metrics_rows
    }
    coverage_by_file = {row.node_natural_key: _coverage_value(row.coverage) for row in metrics_rows}

    if ownership_rows:
        by_file: dict[str, list[TwinOwnershipSnapshot]] = defaultdict(list)
        for row in ownership_rows:
            by_file[row.node_natural_key].append(row)

        churn_values = sorted(
            [
                float(churn_by_file.get(file_key, 0.0))
                for file_key in by_file
                if churn_by_file.get(file_key, 0.0) > 0
            ]
        )
        churn_p75 = _percentile(churn_values, 0.75)

        for node_natural_key, rows in by_file.items():
            dominant = max(rows, key=lambda row: (float(row.ownership_share), int(row.additions)))
            dominant_share = float(dominant.ownership_share)
            churn = float(churn_by_file.get(node_natural_key, 0.0))
            path = _file_path_from_natural_key(node_natural_key) or node_natural_key

            if dominant_share >= 0.8 and churn >= churn_p75:
                findings.append(
                    {
                        "finding_type": "fitness.ff003_single_owner_hotspot",
                        "severity": "high" if churn >= max(churn_p75, 50.0) else "medium",
                        "confidence": "high",
                        "status": "open",
                        "filename": path,
                        "line_number": 1,
                        "message": (
                            f"Single-owner hotspot: {dominant.author_label} owns {round(dominant_share * 100, 1)}% "
                            f"with churn {round(churn, 2)}"
                        ),
                        "meta": {
                            "rule_id": "FF003_single_owner_hotspot",
                            "subject": node_natural_key,
                            "dominant_owner": dominant.author_label,
                            "dominant_share": round(dominant_share, 4),
                            "churn": round(churn, 4),
                            "churn_p75": round(churn_p75, 4),
                        },
                    }
                )

            coverage = coverage_by_file.get(node_natural_key)
            last_touched = max(
                (row.last_touched_at for row in rows if row.last_touched_at),
                default=None,
            )
            if coverage is None or last_touched is None:
                continue

            if last_touched <= now - timedelta(days=180) and coverage < 20.0:
                findings.append(
                    {
                        "finding_type": "fitness.ff005_stale_low_utilization",
                        "severity": "medium",
                        "confidence": "medium",
                        "status": "open",
                        "filename": path,
                        "line_number": 1,
                        "message": (
                            f"Stale low-utilization file: last touched {last_touched.date().isoformat()} "
                            f"with coverage {round(coverage, 2)}"
                        ),
                        "meta": {
                            "rule_id": "FF005_stale_low_utilization",
                            "subject": node_natural_key,
                            "last_touched_at": last_touched.isoformat(),
                            "coverage": round(float(coverage), 4),
                        },
                    }
                )
    else:
        warnings.append("FF003/FF005 skipped: ownership snapshots unavailable")

    # FF004: Cross-boundary coupling with high jaccard.
    coupling_rows = (
        (
            await session.execute(
                select(TwinTemporalCouplingSnapshot).where(
                    TwinTemporalCouplingSnapshot.scenario_id == scenario_id,
                    TwinTemporalCouplingSnapshot.entity_level == "component",
                    TwinTemporalCouplingSnapshot.window_days == window_days,
                    TwinTemporalCouplingSnapshot.cross_boundary.is_(True),
                    TwinTemporalCouplingSnapshot.jaccard >= 0.35,
                )
            )
        )
        .scalars()
        .all()
    )

    for row in sorted(coupling_rows, key=lambda item: item.jaccard, reverse=True)[:200]:
        subject = f"{row.source_key} <-> {row.target_key}"
        findings.append(
            {
                "finding_type": "fitness.ff004_cross_boundary_strong_coupling",
                "severity": "high" if float(row.jaccard) >= 0.6 else "medium",
                "confidence": "high",
                "status": "open",
                "filename": "__architecture__/temporal_coupling",
                "line_number": 1,
                "message": (
                    f"Strong cross-boundary coupling detected: {subject} (jaccard={round(float(row.jaccard), 4)})"
                ),
                "meta": {
                    "rule_id": "FF004_cross_boundary_strong_coupling",
                    "subject": subject,
                    "source_key": row.source_key,
                    "target_key": row.target_key,
                    "jaccard": round(float(row.jaccard), 4),
                    "co_change_count": int(row.co_change_count),
                },
            }
        )

    if findings:
        from contextmine_core.twin.ops import store_findings

        await store_findings(
            session,
            collection_id=collection_id,
            scenario_id=scenario_id,
            findings=findings,
        )

    by_type: dict[str, int] = defaultdict(int)
    severity: dict[str, int] = defaultdict(int)
    for row in findings:
        by_type[str(row["finding_type"])] += 1
        severity[str(row["severity"]).lower()] += 1

    return {
        "created": len(findings),
        "by_type": dict(by_type),
        "severity": dict(severity),
        "warnings": warnings,
    }


async def get_fitness_functions_payload(
    session: AsyncSession,
    *,
    scenario_id: UUID,
    window_days: int,
    include_resolved: bool,
) -> dict[str, Any]:
    stmt = select(TwinFinding).where(
        TwinFinding.scenario_id == scenario_id,
        TwinFinding.finding_type.ilike("fitness.%"),
    )
    if not include_resolved:
        stmt = stmt.where(TwinFinding.status != "resolved")

    rows = (await session.execute(stmt.order_by(TwinFinding.created_at.desc()))).scalars().all()

    rules_index: dict[str, dict[str, Any]] = {}
    violations: list[dict[str, Any]] = []

    for row in rows:
        meta = dict(row.meta or {})
        rule_id = str(meta.get("rule_id") or row.finding_type)
        summary = rules_index.setdefault(
            rule_id,
            {
                "rule_id": rule_id,
                "finding_type": row.finding_type,
                "count": 0,
                "open": 0,
                "resolved": 0,
                "highest_severity": "low",
            },
        )
        summary["count"] += 1
        if str(row.status).lower() == "resolved":
            summary["resolved"] += 1
        else:
            summary["open"] += 1

        severity = str(row.severity).lower()
        if _SEVERITY_WEIGHT.get(severity, 0) > _SEVERITY_WEIGHT.get(
            str(summary["highest_severity"]), 0
        ):
            summary["highest_severity"] = severity

        violations.append(
            {
                "id": str(row.id),
                "rule_id": rule_id,
                "finding_type": row.finding_type,
                "severity": row.severity,
                "confidence": row.confidence,
                "status": row.status,
                "subject": meta.get("subject"),
                "message": row.message,
                "filename": row.filename,
                "line_number": row.line_number,
                "created_at": row.created_at.isoformat(),
                "updated_at": row.updated_at.isoformat(),
                "meta": meta,
            }
        )

    rules = sorted(
        rules_index.values(),
        key=lambda row: (
            -_SEVERITY_WEIGHT.get(str(row["highest_severity"]).lower(), 0),
            -int(row["open"]),
            row["rule_id"],
        ),
    )

    violations.sort(
        key=lambda row: (
            -_SEVERITY_WEIGHT.get(str(row["severity"]).lower(), 0),
            row["rule_id"],
            row["filename"],
        )
    )

    return {
        "status": "ready" if rules else "unavailable",
        "reason": "ok" if rules else "no_fitness_findings",
        "window_days": window_days,
        "summary": {
            "rules": len(rules),
            "violations": len(violations),
            "open": sum(int(rule["open"]) for rule in rules),
            "resolved": sum(int(rule["resolved"]) for rule in rules),
            "highest_severity": (
                max(
                    (str(rule["highest_severity"]) for rule in rules),
                    key=lambda item: _SEVERITY_WEIGHT.get(item, 0),
                )
                if rules
                else "low"
            ),
        },
        "rules": rules,
        "violations": violations[:500],
        "warnings": [] if rules else ["No persisted fitness findings available for scenario."],
    }
