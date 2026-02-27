"""Digital twin and architecture intent routes."""

from __future__ import annotations

import json
import math
import uuid
from collections import defaultdict, deque
from dataclasses import asdict
from typing import Any, Literal

from contextmine_core import Collection, CollectionMember, get_settings
from contextmine_core import get_session as get_db_session
from contextmine_core.architecture import (
    SECTION_TITLES,
    build_architecture_facts,
    compute_arc42_drift,
    generate_arc42_from_facts,
    normalize_arc42_section_key,
)
from contextmine_core.architecture_intents import ArchitectureIntentV1
from contextmine_core.exports import (
    export_codecharta_json,
    export_cx2,
    export_cx2_from_graph,
    export_jgf,
    export_jgf_from_graph,
    export_lpg_jsonl,
    export_lpg_jsonl_from_graph,
    export_mermaid_asis_tobe,
    export_mermaid_asis_tobe_result,
    export_mermaid_c4,
    export_mermaid_c4_result,
    export_twin_manifest,
)
from contextmine_core.graph.age import run_read_only_cypher, sync_scenario_to_age
from contextmine_core.graphrag import trace_path as graphrag_trace_path
from contextmine_core.models import (
    CommunityMember,
    CoverageIngestJob,
    Document,
    EmbeddingTargetType,
    KnowledgeArtifact,
    KnowledgeArtifactKind,
    KnowledgeCommunity,
    KnowledgeEdge,
    KnowledgeEdgeKind,
    KnowledgeEmbedding,
    KnowledgeEvidence,
    KnowledgeNode,
    KnowledgeNodeEvidence,
    KnowledgeNodeKind,
    MetricSnapshot,
    Source,
    SourceType,
    TwinLayer,
    TwinNode,
    TwinScenario,
)
from contextmine_core.twin import (
    DEFAULT_EVOLUTION_WINDOW_DAYS,
    DEFAULT_MAX_COUPLING_EDGES,
    DEFAULT_MIN_JACCARD,
    GraphProjection,
    approve_and_execute_intent,
    coerce_source_ids,
    create_to_be_scenario,
    export_findings_sarif,
    find_taint_flows_multi,
    find_taint_sinks_multi,
    find_taint_sources_multi,
    get_cfg_multi,
    get_codebase_summary_multi,
    get_collection_twin_diff,
    get_collection_twin_status,
    get_fitness_functions_payload,
    get_full_scenario_graph,
    get_investment_utilization_payload,
    get_knowledge_islands_payload,
    get_or_create_as_is_scenario,
    get_scenario_graph,
    get_temporal_coupling_payload,
    get_variable_flow_multi,
    list_calls_multi,
    list_collection_twin_events,
    list_findings,
    list_methods_multi,
    list_scenario_patches,
    parse_timestamp_value,
    refresh_metric_snapshots,
    sanitize_regex_query,
    store_findings,
    submit_intent,
    trigger_collection_refresh,
)
from contextmine_core.twin.projections import (
    build_test_matrix_projection,
    build_ui_map_projection,
    build_user_flows_projection,
    compute_rebuild_readiness,
)
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import func, literal_column, select

from app.middleware import get_session

router = APIRouter(prefix="/twin", tags=["twin"])


class CreateScenarioRequest(BaseModel):
    collection_id: str
    name: str = Field(min_length=1, max_length=255)


class CypherRequest(BaseModel):
    query: str = Field(min_length=1)


class ExportRequest(BaseModel):
    format: Literal["lpg_jsonl", "cc_json", "cx2", "jgf", "mermaid_c4", "twin_manifest"]
    projection: Literal["architecture", "code_file", "code_symbol"] | None = None
    entity_level: Literal["domain", "container", "component", "file", "symbol"] | None = None
    c4_view: Literal["context", "container", "component", "code", "deployment"] | None = None
    c4_scope: str | None = None
    max_nodes: int | None = Field(default=None, ge=10, le=5000)


class TwinRefreshRequest(BaseModel):
    source_ids: list[str] | None = None
    force: bool = False


class StoreFindingsRequest(BaseModel):
    scenario_id: str | None = None
    findings: list[dict[str, Any]] = Field(default_factory=list)


class TaintFlowsRequest(BaseModel):
    scenario_id: str | None = None
    language: str | None = None
    max_hops: int = Field(default=6, ge=1, le=20)
    max_results: int = Field(default=50, ge=1, le=200)
    engines: list[str] | None = None


def _user_id_or_401(request: Request) -> uuid.UUID:
    session = get_session(request)
    user_id = session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return uuid.UUID(user_id)


async def _load_scenario(db, scenario_id: str) -> TwinScenario:
    try:
        scenario_uuid = uuid.UUID(scenario_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid scenario_id") from e

    scenario = (
        await db.execute(select(TwinScenario).where(TwinScenario.id == scenario_uuid))
    ).scalar_one_or_none()
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return scenario


def _parse_collection_id(collection_id: str) -> uuid.UUID:
    try:
        return uuid.UUID(collection_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid collection_id") from e


def _parse_optional_scenario_id(scenario_id: str | None) -> uuid.UUID | None:
    if not scenario_id:
        return None
    try:
        return uuid.UUID(scenario_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid scenario_id") from e


def _parse_engines_query(engines: str | None) -> list[str] | None:
    if not engines:
        return None
    values = [item.strip() for item in engines.split(",") if item.strip()]
    return values or None


def _ensure_evolution_enabled() -> None:
    settings = get_settings()
    if not settings.twin_evolution_view_enabled:
        raise HTTPException(status_code=404, detail="Evolution view is disabled")


def _serialize_scenario(scenario: TwinScenario) -> dict:
    return {
        "id": str(scenario.id),
        "collection_id": str(scenario.collection_id),
        "name": scenario.name,
        "version": scenario.version,
        "is_as_is": scenario.is_as_is,
        "base_scenario_id": str(scenario.base_scenario_id) if scenario.base_scenario_id else None,
    }


async def _ensure_owner(db, collection_id: uuid.UUID, user_id: uuid.UUID) -> None:
    collection = (
        await db.execute(select(Collection).where(Collection.id == collection_id))
    ).scalar_one_or_none()
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    if collection.owner_user_id != user_id:
        raise HTTPException(status_code=403, detail="Only collection owner can execute intents")


async def _ensure_member(db, collection_id: uuid.UUID, user_id: uuid.UUID) -> None:
    collection = (
        await db.execute(select(Collection).where(Collection.id == collection_id))
    ).scalar_one_or_none()
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    if collection.owner_user_id == user_id:
        return
    membership = (
        await db.execute(
            select(CollectionMember).where(
                CollectionMember.collection_id == collection_id,
                CollectionMember.user_id == user_id,
            )
        )
    ).scalar_one_or_none()
    if not membership:
        raise HTTPException(status_code=403, detail="Access denied")


async def _upsert_artifact(
    db,
    *,
    collection_id: uuid.UUID,
    kind: KnowledgeArtifactKind,
    name: str,
    content: str,
    meta: dict,
) -> KnowledgeArtifact:
    existing = (
        await db.execute(
            select(KnowledgeArtifact).where(
                KnowledgeArtifact.collection_id == collection_id,
                KnowledgeArtifact.kind == kind,
                KnowledgeArtifact.name == name,
            )
        )
    ).scalar_one_or_none()
    if existing:
        existing.content = content
        existing.meta = meta
        return existing

    artifact = KnowledgeArtifact(
        id=uuid.uuid4(),
        collection_id=collection_id,
        kind=kind,
        name=name,
        content=content,
        meta=meta,
    )
    db.add(artifact)
    return artifact


async def _can_access_collection(db, collection_id: uuid.UUID, user_id: uuid.UUID) -> bool:
    collection = (
        await db.execute(select(Collection).where(Collection.id == collection_id))
    ).scalar_one_or_none()
    if not collection:
        return False
    if collection.owner_user_id == user_id:
        return True
    membership = (
        await db.execute(
            select(CollectionMember).where(
                CollectionMember.collection_id == collection_id,
                CollectionMember.user_id == user_id,
            )
        )
    ).scalar_one_or_none()
    return membership is not None


async def _resolve_view_scenario(
    db,
    collection_id: uuid.UUID,
    scenario_id: str | None,
) -> TwinScenario:
    if scenario_id:
        try:
            scenario_uuid = uuid.UUID(scenario_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail="Invalid scenario_id") from e
        scenario = (
            await db.execute(
                select(TwinScenario).where(
                    TwinScenario.id == scenario_uuid,
                    TwinScenario.collection_id == collection_id,
                )
            )
        ).scalar_one_or_none()
        if not scenario:
            raise HTTPException(status_code=404, detail="Scenario not found in collection")
        return scenario

    scenario = (
        await db.execute(
            select(TwinScenario).where(
                TwinScenario.collection_id == collection_id,
                TwinScenario.is_as_is.is_(True),
            )
        )
    ).scalar_one_or_none()
    if scenario:
        return scenario

    return await get_or_create_as_is_scenario(db, collection_id, user_id=None)


def _parse_layer(layer: str | None) -> TwinLayer | None:
    if not layer:
        return None
    try:
        return TwinLayer(layer)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid layer") from e


def _parse_projection(projection: str | None) -> GraphProjection:
    if not projection:
        return GraphProjection.CODE_SYMBOL
    try:
        return GraphProjection(projection)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid projection") from e


def _parse_c4_view(c4_view: str | None) -> str:
    if not c4_view:
        return "container"
    normalized = c4_view.strip().lower()
    if normalized not in {"context", "container", "component", "code", "deployment"}:
        raise HTTPException(status_code=400, detail="Invalid c4_view")
    return normalized


def _parse_c4_scope(c4_scope: str | None) -> str | None:
    if c4_scope is None:
        return None
    normalized = c4_scope.strip()
    if not normalized:
        return None
    if any(ord(char) < 32 for char in normalized):
        raise HTTPException(status_code=400, detail="Invalid c4_scope")
    if len(normalized) > 255:
        raise HTTPException(status_code=400, detail="Invalid c4_scope")
    return normalized


def _parse_kind_filter(raw_value: str | None) -> set[str] | None:
    if not raw_value:
        return None
    values = {value.strip().lower() for value in raw_value.split(",") if value.strip()}
    return values or None


def _arc42_artifact_name(scenario_id: uuid.UUID) -> str:
    return f"{scenario_id}.arc42.md"


def _serialize_arc42_document(document) -> dict[str, Any]:
    return {
        "title": document.title,
        "generated_at": document.generated_at.isoformat(),
        "sections": document.sections,
        "markdown": document.markdown,
        "warnings": document.warnings,
        "confidence_summary": document.confidence_summary,
        "section_coverage": document.section_coverage,
    }


def _serialize_port_adapter_fact(fact) -> dict[str, Any]:
    return asdict(fact)


def _parse_db_table_from_natural_key(natural_key: str | None) -> str | None:
    if not natural_key:
        return None
    if not natural_key.startswith("db:"):
        return None
    body = natural_key[3:]
    if not body:
        return None
    if "." in body:
        return body.split(".", 1)[0] or None
    return body


def _serialize_erm_column(node: KnowledgeNode) -> dict[str, Any]:
    meta = node.meta if isinstance(node.meta, dict) else {}
    return {
        "id": str(node.id),
        "natural_key": node.natural_key,
        "name": node.name,
        "table": meta.get("table"),
        "type": meta.get("type"),
        "nullable": bool(meta.get("nullable", True)),
        "primary_key": bool(meta.get("primary_key", False)),
        "foreign_key": meta.get("foreign_key"),
    }


def _serialize_erm_table(node: KnowledgeNode, columns: list[dict[str, Any]]) -> dict[str, Any]:
    meta = node.meta if isinstance(node.meta, dict) else {}
    description = getattr(node, "description", None)
    return {
        "id": str(node.id),
        "natural_key": node.natural_key,
        "name": node.name,
        "description": description,
        "column_count": int(meta.get("column_count", len(columns) or 0)),
        "primary_keys": meta.get("primary_keys", []),
        "columns": columns,
    }


async def _resolve_baseline_scenario(
    db,
    *,
    collection_id: uuid.UUID,
    scenario: TwinScenario,
    baseline_scenario_id: str | None,
) -> TwinScenario | None:
    if baseline_scenario_id:
        try:
            baseline_uuid = uuid.UUID(baseline_scenario_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail="Invalid baseline_scenario_id") from e
        baseline = (
            await db.execute(
                select(TwinScenario).where(
                    TwinScenario.id == baseline_uuid,
                    TwinScenario.collection_id == collection_id,
                )
            )
        ).scalar_one_or_none()
        if not baseline:
            raise HTTPException(status_code=404, detail="Baseline scenario not found in collection")
        return baseline

    if scenario.base_scenario_id:
        baseline = (
            await db.execute(
                select(TwinScenario).where(
                    TwinScenario.id == scenario.base_scenario_id,
                    TwinScenario.collection_id == collection_id,
                )
            )
        ).scalar_one_or_none()
        if baseline:
            return baseline

    return (
        await db.execute(
            select(TwinScenario)
            .where(
                TwinScenario.collection_id == collection_id,
                TwinScenario.id != scenario.id,
            )
            .order_by(TwinScenario.version.desc(), TwinScenario.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()


def _resolve_arch_llm_provider(settings) -> Any | None:
    if not settings.arch_docs_llm_enrich:
        return None
    if not settings.default_llm_provider:
        return None
    try:
        from contextmine_core.research.llm import get_llm_provider

        return get_llm_provider(settings.default_llm_provider)
    except Exception:
        return None


async def _build_arch_bundle(db, *, collection_id: uuid.UUID, scenario: TwinScenario):
    settings = get_settings()
    llm_provider = _resolve_arch_llm_provider(settings)
    return await build_architecture_facts(
        db,
        collection_id=collection_id,
        scenario_id=scenario.id,
        enable_llm_enrich=settings.arch_docs_llm_enrich,
        llm_provider=llm_provider,
    )


def _parse_knowledge_node_kinds(raw_value: str | None) -> set[KnowledgeNodeKind] | None:
    raw_kinds = _parse_kind_filter(raw_value)
    if not raw_kinds:
        return None
    parsed: set[KnowledgeNodeKind] = set()
    for raw_kind in raw_kinds:
        try:
            parsed.add(KnowledgeNodeKind(raw_kind))
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid knowledge node kind: {raw_kind}",
            ) from e
    return parsed


def _parse_knowledge_edge_kinds(raw_value: str | None) -> set[KnowledgeEdgeKind] | None:
    raw_kinds = _parse_kind_filter(raw_value)
    if not raw_kinds:
        return None
    parsed: set[KnowledgeEdgeKind] = set()
    for raw_kind in raw_kinds:
        try:
            parsed.add(KnowledgeEdgeKind(raw_kind))
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid knowledge edge kind: {raw_kind}",
            ) from e
    return parsed


def _parse_graphrag_community_mode(raw_value: str | None) -> str:
    if not raw_value:
        return "color"
    normalized = raw_value.strip().lower()
    if normalized not in {"none", "color", "focus"}:
        raise HTTPException(status_code=400, detail="Invalid community_mode")
    return normalized


def _parse_semantic_map_mode(raw_value: str | None) -> str:
    if not raw_value:
        return "code_structure"
    normalized = raw_value.strip().lower()
    if normalized not in {"code_structure", "semantic"}:
        raise HTTPException(status_code=400, detail="Invalid map_mode")
    return normalized


def _parse_pgvector_text(raw_value: str | None) -> list[float]:
    if not raw_value:
        return []
    stripped = raw_value.strip()
    if not stripped or stripped == "[]":
        return []
    if stripped.startswith("[") and stripped.endswith("]"):
        stripped = stripped[1:-1]
    values: list[float] = []
    for token in stripped.split(","):
        item = token.strip()
        if not item:
            continue
        try:
            values.append(float(item))
        except ValueError:
            continue
    return values


def _extract_domain_token(path_like: str | None) -> str | None:
    if not path_like:
        return None
    normalized = str(path_like).strip().lower()
    if not normalized:
        return None
    for prefix in ("file:", "symbol:", "entity:"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
    if "://" in normalized:
        remainder = normalized.split("://", 1)[1]
        normalized = remainder.split("/", 1)[1] if "/" in remainder else remainder
    normalized = normalized.split("?", 1)[0].strip("/")
    parts = [part for part in normalized.split("/") if part]
    if not parts:
        return None
    if "." in parts[-1]:
        parts = parts[:-1]
    if not parts:
        return None

    ignored = {
        "src",
        "main",
        "lib",
        "app",
        "apps",
        "package",
        "packages",
        "core",
        "contextmine_core",
        "services",
        "service",
        "api",
        "backend",
        "frontend",
        "server",
        "client",
        "code",
        "python",
        "java",
        "javascript",
        "typescript",
        "ts",
        "js",
    }
    for part in reversed(parts):
        token = part.strip().replace("_", "-")
        if len(token) < 2:
            continue
        if token in ignored:
            continue
        return token
    return parts[-1]


def _node_domain_info(
    node: KnowledgeNode,
) -> tuple[str | None, set[str]]:
    meta = _safe_meta(node.meta)
    source_refs: set[str] = set()

    file_path = meta.get("file_path")
    if isinstance(file_path, str) and file_path.strip():
        source_refs.add(file_path)

    for key in ("source_files", "source_symbols"):
        raw_refs = meta.get(key)
        if isinstance(raw_refs, list):
            for ref in raw_refs:
                if isinstance(ref, str) and ref.strip():
                    source_refs.add(ref)
        elif isinstance(raw_refs, str) and raw_refs.strip():
            source_refs.add(raw_refs)

    if node.natural_key:
        source_refs.add(node.natural_key)

    domains = [token for token in (_extract_domain_token(ref) for ref in source_refs) if token]
    if not domains:
        return None, source_refs
    counts: dict[str, int] = defaultdict(int)
    for token in domains:
        counts[token] += 1
    dominant_domain = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return dominant_domain, source_refs


def _community_profile(
    member_nodes: list[KnowledgeNode],
) -> dict[str, Any]:
    kind_counts: dict[str, int] = defaultdict(int)
    domain_counts: dict[str, int] = defaultdict(int)
    source_refs: set[str] = set()
    node_domain_by_id: dict[uuid.UUID, str] = {}
    name_tokens: set[str] = set()

    for member in member_nodes:
        kind_counts[member.kind.value] += 1
        dominant_domain, node_sources = _node_domain_info(member)
        source_refs.update(node_sources)
        if dominant_domain:
            domain_counts[dominant_domain] += 1
            node_domain_by_id[member.id] = dominant_domain

        for token in member.name.lower().replace("_", " ").replace("-", " ").split():
            if len(token) >= 4:
                name_tokens.add(token)

    total_domains = sum(domain_counts.values())
    dominant_domain = None
    dominant_ratio = 0.0
    if domain_counts:
        dominant_domain, count = sorted(
            domain_counts.items(), key=lambda item: (-item[1], item[0])
        )[0]
        dominant_ratio = count / total_domains if total_domains > 0 else 0.0

    top_kinds = [
        {"kind": kind, "count": count}
        for kind, count in sorted(kind_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    domain_rows = [
        {"domain": domain, "count": count}
        for domain, count in sorted(domain_counts.items(), key=lambda item: (-item[1], item[0]))
    ]

    return {
        "top_kinds": top_kinds,
        "domain_counts": domain_rows,
        "dominant_domain": dominant_domain,
        "dominant_ratio": round(dominant_ratio, 4),
        "node_domain_by_id": node_domain_by_id,
        "source_refs": sorted(source_refs),
        "name_tokens": sorted(name_tokens),
    }


def _build_projection_coeffs(dim: int, seed: int) -> list[float]:
    state = seed & 0xFFFFFFFF
    coeffs: list[float] = []
    for _ in range(dim):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        coeff = ((state & 0xFFFF) / 32767.5) - 1.0
        coeffs.append(coeff)
    return coeffs


def _project_vectors(points: list[dict[str, Any]], vector_key: str) -> None:
    vectors = [point[vector_key] for point in points if point.get(vector_key)]
    if not vectors:
        for point in points:
            point["x"] = point.get("x", 0.0)
            point["y"] = point.get("y", 0.0)
        return

    dim = max(len(vector) for vector in vectors)
    coeff_x = _build_projection_coeffs(dim, seed=17)
    coeff_y = _build_projection_coeffs(dim, seed=89)

    for index, point in enumerate(points):
        vector = point.get(vector_key) or []
        if not vector:
            point["x"] = math.sin(index * 0.73)
            point["y"] = math.cos(index * 1.13)
            continue
        padded = vector + [0.0] * (dim - len(vector))
        point["x"] = float(
            sum(value * coeff for value, coeff in zip(padded, coeff_x, strict=False))
        )
        point["y"] = float(
            sum(value * coeff for value, coeff in zip(padded, coeff_y, strict=False))
        )


def _normalize_xy(points: list[dict[str, Any]]) -> None:
    if not points:
        return
    min_x = min(float(point.get("x", 0.0)) for point in points)
    max_x = max(float(point.get("x", 0.0)) for point in points)
    min_y = min(float(point.get("y", 0.0)) for point in points)
    max_y = max(float(point.get("y", 0.0)) for point in points)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)
    for point in points:
        x = float(point.get("x", 0.0))
        y = float(point.get("y", 0.0))
        point["x"] = round(((x - min_x) / span_x) * 2 - 1, 6)
        point["y"] = round(((y - min_y) / span_y) * 2 - 1, 6)


def _layout_code_structure_points(
    point_ids: list[str],
    pair_weights: dict[tuple[str, str], float],
) -> dict[str, tuple[float, float]]:
    if not point_ids:
        return {}
    if len(point_ids) == 1:
        return {point_ids[0]: (0.0, 0.0)}
    if len(point_ids) > 220:
        return {
            point_id: (
                round(math.sin(index * 0.79), 6),
                round(math.cos(index * 1.07), 6),
            )
            for index, point_id in enumerate(point_ids)
        }

    positions: dict[str, list[float]] = {}
    for index, point_id in enumerate(point_ids):
        angle = (2 * math.pi * index) / max(len(point_ids), 1)
        positions[point_id] = [math.cos(angle), math.sin(angle)]

    repulsion = 0.055
    spring = 0.04
    damping = 0.82
    step = 0.06
    for _ in range(120):
        delta: dict[str, list[float]] = {point_id: [0.0, 0.0] for point_id in point_ids}

        for i, source_id in enumerate(point_ids):
            sx, sy = positions[source_id]
            for target_id in point_ids[i + 1 :]:
                tx, ty = positions[target_id]
                dx = sx - tx
                dy = sy - ty
                dist_sq = max(dx * dx + dy * dy, 1e-4)
                force = repulsion / dist_sq
                fx = force * dx
                fy = force * dy
                delta[source_id][0] += fx
                delta[source_id][1] += fy
                delta[target_id][0] -= fx
                delta[target_id][1] -= fy

        for (source_id, target_id), weight in pair_weights.items():
            sx, sy = positions[source_id]
            tx, ty = positions[target_id]
            dx = tx - sx
            dy = ty - sy
            dist = max(math.sqrt(dx * dx + dy * dy), 1e-6)
            desired = 0.55
            force = spring * weight * (dist - desired)
            fx = force * (dx / dist)
            fy = force * (dy / dist)
            delta[source_id][0] += fx
            delta[source_id][1] += fy
            delta[target_id][0] -= fx
            delta[target_id][1] -= fy

        for point_id in point_ids:
            positions[point_id][0] += max(min(delta[point_id][0] * step, 0.16), -0.16)
            positions[point_id][1] += max(min(delta[point_id][1] * step, 0.16), -0.16)

        step *= damping

    return {
        point_id: (round(positions[point_id][0], 6), round(positions[point_id][1], 6))
        for point_id in point_ids
    }


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    dim = min(len(left), len(right))
    if dim == 0:
        return 0.0
    dot = sum(left[index] * right[index] for index in range(dim))
    left_norm = math.sqrt(sum(value * value for value in left[:dim]))
    right_norm = math.sqrt(sum(value * value for value in right[:dim]))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _detect_isolated_point_ids(
    points: list[dict[str, Any]],
    *,
    distance_multiplier: float = 1.2,
) -> tuple[set[str], dict[str, float]]:
    if len(points) <= 1:
        return {point["id"] for point in points}, {point["id"]: 0.0 for point in points}
    nearest_distances: dict[str, float] = {}
    for point in points:
        px = float(point.get("x", 0.0))
        py = float(point.get("y", 0.0))
        nearest = float("inf")
        for other in points:
            if other["id"] == point["id"]:
                continue
            ox = float(other.get("x", 0.0))
            oy = float(other.get("y", 0.0))
            distance = math.sqrt((px - ox) ** 2 + (py - oy) ** 2)
            nearest = min(nearest, distance)
        nearest_distances[point["id"]] = nearest if nearest != float("inf") else 0.0

    values = list(nearest_distances.values())
    mean_distance = sum(values) / len(values)
    variance = sum((value - mean_distance) ** 2 for value in values) / len(values)
    threshold = mean_distance + math.sqrt(variance) * max(distance_multiplier, 0.0)
    isolated_ids = {
        point_id for point_id, distance in nearest_distances.items() if distance >= threshold
    }
    return isolated_ids, nearest_distances


def _build_semantic_map_signals(
    points: list[dict[str, Any]],
    *,
    mode: str,
    mixed_cluster_max_dominant_ratio: float = 0.55,
    isolated_distance_multiplier: float = 1.2,
    semantic_duplication_min_similarity: float | None = None,
    semantic_duplication_max_source_overlap: float | None = None,
    misplaced_min_dominant_ratio: float = 0.6,
) -> dict[str, list[dict[str, Any]]]:
    isolated_ids, nearest_distances = _detect_isolated_point_ids(
        points, distance_multiplier=isolated_distance_multiplier
    )
    if semantic_duplication_min_similarity is None:
        semantic_duplication_min_similarity = 0.86 if mode == "semantic" else 0.35
    if semantic_duplication_max_source_overlap is None:
        semantic_duplication_max_source_overlap = 0.30 if mode == "semantic" else 0.35

    mixed_clusters: list[dict[str, Any]] = []
    isolated_points: list[dict[str, Any]] = []
    semantic_duplication: list[dict[str, Any]] = []
    misplaced_code: list[dict[str, Any]] = []

    for point in points:
        point_id = str(point["id"])
        member_count = int(point.get("member_count", 0))
        dominant_ratio = float(point.get("dominant_ratio", 0.0))
        domain_counts = point.get("domain_counts", [])
        if (
            member_count >= 4
            and len(domain_counts) >= 2
            and dominant_ratio < mixed_cluster_max_dominant_ratio
        ):
            mixed_clusters.append(
                {
                    "community_id": point_id,
                    "label": point.get("label", point_id),
                    "score": round(1.0 - dominant_ratio, 4),
                    "anchor_node_id": point.get("anchor_node_id", ""),
                    "reason": "Domain signals are strongly mixed inside this cluster.",
                }
            )
        if point_id in isolated_ids:
            isolated_points.append(
                {
                    "community_id": point_id,
                    "label": point.get("label", point_id),
                    "score": round(float(nearest_distances.get(point_id, 0.0)), 4),
                    "anchor_node_id": point.get("anchor_node_id", ""),
                    "reason": "This cluster is far from other clusters in the map projection.",
                }
            )

        dominant_domain = point.get("dominant_domain")
        misplaced_nodes = point.get("misplaced_nodes", [])
        if dominant_domain and dominant_ratio >= misplaced_min_dominant_ratio and misplaced_nodes:
            misplaced_code.append(
                {
                    "community_id": point_id,
                    "label": point.get("label", point_id),
                    "score": round(len(misplaced_nodes) / max(member_count, 1), 4),
                    "anchor_node_id": misplaced_nodes[0]["id"],
                    "reason": f"Some members diverge from dominant domain '{dominant_domain}'.",
                    "sample_nodes": misplaced_nodes[:6],
                }
            )

    for index, left in enumerate(points):
        left_tokens = set(left.get("name_tokens", []))
        left_sources = set(left.get("source_refs", []))
        left_vector = left.get("vector") or []
        for right in points[index + 1 :]:
            right_tokens = set(right.get("name_tokens", []))
            right_sources = set(right.get("source_refs", []))
            right_vector = right.get("vector") or []
            source_overlap = _jaccard(left_sources, right_sources)

            if mode == "semantic":
                similarity = _cosine_similarity(left_vector, right_vector)
                if (
                    similarity < semantic_duplication_min_similarity
                    or source_overlap >= semantic_duplication_max_source_overlap
                ):
                    continue
            else:
                similarity = _jaccard(left_tokens, right_tokens)
                if (
                    similarity < semantic_duplication_min_similarity
                    or source_overlap >= semantic_duplication_max_source_overlap
                ):
                    continue

            semantic_duplication.append(
                {
                    "left_community_id": left["id"],
                    "right_community_id": right["id"],
                    "left_label": left.get("label", left["id"]),
                    "right_label": right.get("label", right["id"]),
                    "score": round(similarity, 4),
                    "anchor_node_id": left.get("anchor_node_id", ""),
                    "reason": "Two clusters are semantically close but have little source overlap.",
                }
            )

    semantic_duplication.sort(key=lambda item: float(item["score"]), reverse=True)
    mixed_clusters.sort(key=lambda item: float(item["score"]), reverse=True)
    misplaced_code.sort(key=lambda item: float(item["score"]), reverse=True)
    isolated_points.sort(key=lambda item: float(item["score"]), reverse=True)

    return {
        "mixed_clusters": mixed_clusters[:20],
        "isolated_points": isolated_points[:20],
        "semantic_duplication": semantic_duplication[:20],
        "misplaced_code": misplaced_code[:20],
    }


def _escape_like_pattern(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _truncate_text(value: str, max_chars: int = 2000) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "..."


def _extract_document_lines(
    content: str | None,
    start_line: int,
    end_line: int,
    *,
    max_chars: int = 2000,
) -> str:
    if not content:
        return ""
    lines = content.splitlines()
    if not lines:
        return ""
    start = max(1, start_line)
    end = max(start, end_line)
    if start > len(lines):
        return ""
    excerpt = "\n".join(lines[start - 1 : min(end, len(lines))]).strip()
    if not excerpt:
        return ""
    return _truncate_text(excerpt, max_chars=max_chars)


def _topology_entity_level(layer: TwinLayer | None, explicit: str | None) -> str:
    if explicit:
        return explicit
    if layer == TwinLayer.PORTFOLIO_SYSTEM:
        return "domain"
    if layer == TwinLayer.COMPONENT_INTERFACE:
        return "component"
    return "container"


def _extract_neighborhood(
    nodes: list[dict],
    edges: list[dict],
    root_node_id: str,
    hops: int,
    limit: int,
) -> tuple[list[dict], list[dict]]:
    node_by_id = {str(node.get("id")): node for node in nodes}
    if root_node_id not in node_by_id:
        return [], []

    seen = {root_node_id}
    frontier = {root_node_id}
    adjacency: dict[str, set[str]] = {}
    for edge in edges:
        src = str(edge.get("source_node_id"))
        dst = str(edge.get("target_node_id"))
        adjacency.setdefault(src, set()).add(dst)
        adjacency.setdefault(dst, set()).add(src)

    for _ in range(max(hops, 0)):
        if not frontier:
            break
        next_frontier: set[str] = set()
        for node_id in frontier:
            for neighbor in adjacency.get(node_id, set()):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                next_frontier.add(neighbor)
                if len(seen) >= limit:
                    break
            if len(seen) >= limit:
                break
        frontier = next_frontier

    neighborhood_nodes = [node for node_id, node in node_by_id.items() if node_id in seen]
    neighborhood_ids = {str(node.get("id")) for node in neighborhood_nodes}
    neighborhood_edges = [
        edge
        for edge in edges
        if str(edge.get("source_node_id")) in neighborhood_ids
        and str(edge.get("target_node_id")) in neighborhood_ids
    ]
    return neighborhood_nodes, neighborhood_edges


def _paginate_graph(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    page: int,
    limit: int,
) -> dict[str, Any]:
    sorted_nodes = sorted(
        nodes,
        key=lambda node: (
            str(node.get("kind") or ""),
            str(node.get("name") or ""),
            str(node.get("natural_key") or ""),
            str(node.get("id") or ""),
        ),
    )
    total_nodes = len(sorted_nodes)
    start = max(page, 0) * max(limit, 1)
    end = start + max(limit, 1)
    page_nodes = sorted_nodes[start:end]
    page_ids = {str(node.get("id")) for node in page_nodes}
    page_edges = [
        edge
        for edge in edges
        if str(edge.get("source_node_id")) in page_ids
        and str(edge.get("target_node_id")) in page_ids
    ]
    return {
        "nodes": page_nodes,
        "edges": page_edges,
        "page": max(page, 0),
        "limit": max(limit, 1),
        "total_nodes": total_nodes,
    }


def _safe_meta(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _symbol_sort_key(node: KnowledgeNode) -> tuple[str, str, str]:
    return (node.natural_key or "", node.name or "", str(node.id))


def _community_label(node_ids: list[uuid.UUID], node_by_id: dict[uuid.UUID, KnowledgeNode]) -> str:
    token_counts: dict[str, int] = defaultdict(int)
    for node_id in node_ids:
        node = node_by_id[node_id]
        meta = _safe_meta(node.meta)
        file_path = str(meta.get("file_path") or "").strip("/")
        token = ""
        if file_path:
            parts = [part for part in file_path.split("/") if part]
            if len(parts) >= 2:
                token = parts[-2]
            elif parts:
                token = parts[0]
        if not token:
            suffix = (node.natural_key or "").split(":")[-1]
            token = suffix.split(".")[0].split("/")[0]
        token = token.strip().replace("_", " ").replace("-", " ")
        if len(token) >= 3:
            token_counts[token.lower()] += 1

    if token_counts:
        best = sorted(token_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        return best[:1].upper() + best[1:]
    return "Community"


def _compute_symbol_communities(
    symbol_nodes: list[KnowledgeNode],
    symbol_edges: list[KnowledgeEdge],
) -> tuple[dict[uuid.UUID, str], dict[str, dict[str, Any]]]:
    if not symbol_nodes:
        return {}, {}

    node_by_id = {node.id: node for node in symbol_nodes}
    adjacency: dict[uuid.UUID, set[uuid.UUID]] = {node.id: set() for node in symbol_nodes}
    for edge in symbol_edges:
        src = edge.source_node_id
        dst = edge.target_node_id
        if src == dst:
            continue
        if src in adjacency and dst in adjacency:
            adjacency[src].add(dst)
            adjacency[dst].add(src)

    visited: set[uuid.UUID] = set()
    components: list[dict[str, Any]] = []

    for node in sorted(symbol_nodes, key=_symbol_sort_key):
        if node.id in visited:
            continue
        queue: deque[uuid.UUID] = deque([node.id])
        visited.add(node.id)
        members: list[uuid.UUID] = []
        while queue:
            current = queue.popleft()
            members.append(current)
            for neighbor in sorted(
                adjacency.get(current, set()),
                key=lambda node_id: _symbol_sort_key(node_by_id[node_id]),
            ):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)

        members = sorted(members, key=lambda node_id: _symbol_sort_key(node_by_id[node_id]))
        member_set = set(members)
        internal_edges = 0
        for src in members:
            for dst in adjacency.get(src, set()):
                if dst in member_set and str(src) < str(dst):
                    internal_edges += 1
        possible_edges = (len(members) * (len(members) - 1)) // 2
        cohesion = 1.0 if possible_edges == 0 else round(internal_edges / possible_edges, 4)
        label = _community_label(members, node_by_id)
        components.append(
            {
                "member_ids": members,
                "label": label,
                "cohesion": cohesion,
                "first_key": node_by_id[members[0]].natural_key or "",
            }
        )

    components.sort(
        key=lambda comp: (-len(comp["member_ids"]), comp["label"].lower(), comp["first_key"])
    )

    node_to_community: dict[uuid.UUID, str] = {}
    communities: dict[str, dict[str, Any]] = {}
    for index, component in enumerate(components, start=1):
        community_id = f"comm_{index}"
        member_nodes = [node_by_id[node_id] for node_id in component["member_ids"]]
        for member in member_nodes:
            node_to_community[member.id] = community_id
        kind_counts: dict[str, int] = defaultdict(int)
        for member in member_nodes:
            kind_counts[member.kind.value] += 1
        top_kind_rows = [
            {
                "kind": kind,
                "count": count,
            }
            for kind, count in sorted(
                kind_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]
        communities[community_id] = {
            "id": community_id,
            "label": f"{component['label']} ({len(member_nodes)})",
            "size": len(member_nodes),
            "cohesion": component["cohesion"],
            "top_kinds": top_kind_rows,
            "sample_nodes": [
                {
                    "id": str(member.id),
                    "name": member.name,
                    "kind": member.kind.value,
                    "natural_key": member.natural_key,
                }
                for member in member_nodes[:6]
            ],
            "member_node_ids": [member.id for member in member_nodes],
        }

    return node_to_community, communities


GRAPHRAG_CODE_NOISE_NODE_KINDS: set[KnowledgeNodeKind] = {
    KnowledgeNodeKind.FILE,
    KnowledgeNodeKind.SYMBOL,
}
GRAPHRAG_CODE_NOISE_EDGE_KINDS: set[KnowledgeEdgeKind] = {
    KnowledgeEdgeKind.FILE_DEFINES_SYMBOL,
    KnowledgeEdgeKind.SYMBOL_CONTAINS_SYMBOL,
    KnowledgeEdgeKind.FILE_IMPORTS_FILE,
    KnowledgeEdgeKind.SYMBOL_CALLS_SYMBOL,
    KnowledgeEdgeKind.SYMBOL_REFERENCES_SYMBOL,
}
GRAPHRAG_SEMANTIC_NODE_KINDS: set[KnowledgeNodeKind] = {
    kind for kind in KnowledgeNodeKind if kind not in GRAPHRAG_CODE_NOISE_NODE_KINDS
}
GRAPHRAG_SEMANTIC_EDGE_KINDS: set[KnowledgeEdgeKind] = {
    kind for kind in KnowledgeEdgeKind if kind not in GRAPHRAG_CODE_NOISE_EDGE_KINDS
}


async def _load_community_graph(
    db: Any,
    collection_id: uuid.UUID,
) -> tuple[list[KnowledgeNode], list[KnowledgeEdge], str]:
    """Load preferred graph for community/process views.

    Preference order:
    1. SYMBOL nodes with semantic dependency edges (calls/references)
    2. Semantic/architecture node kinds with non-code edges
    3. Fallback to all knowledge nodes/edges
    """
    symbol_nodes = (
        (
            await db.execute(
                select(KnowledgeNode).where(
                    KnowledgeNode.collection_id == collection_id,
                    KnowledgeNode.kind == KnowledgeNodeKind.SYMBOL,
                )
            )
        )
        .scalars()
        .all()
    )
    if symbol_nodes:
        symbol_node_ids = {node.id for node in symbol_nodes}
        symbol_edges: list[KnowledgeEdge] = []
        if symbol_node_ids:
            symbol_edges = (
                (
                    await db.execute(
                        select(KnowledgeEdge).where(
                            KnowledgeEdge.collection_id == collection_id,
                            KnowledgeEdge.kind.in_(
                                [
                                    KnowledgeEdgeKind.SYMBOL_CALLS_SYMBOL,
                                    KnowledgeEdgeKind.SYMBOL_REFERENCES_SYMBOL,
                                ]
                            ),
                            KnowledgeEdge.source_node_id.in_(symbol_node_ids),
                            KnowledgeEdge.target_node_id.in_(symbol_node_ids),
                        )
                    )
                )
                .scalars()
                .all()
            )
        if symbol_edges:
            return symbol_nodes, symbol_edges, "symbol"

    semantic_nodes = (
        (
            await db.execute(
                select(KnowledgeNode).where(
                    KnowledgeNode.collection_id == collection_id,
                    KnowledgeNode.kind.in_(GRAPHRAG_SEMANTIC_NODE_KINDS),
                )
            )
        )
        .scalars()
        .all()
    )
    if semantic_nodes:
        semantic_node_ids = {node.id for node in semantic_nodes}
        semantic_edges: list[KnowledgeEdge] = []
        if semantic_node_ids:
            semantic_edges = (
                (
                    await db.execute(
                        select(KnowledgeEdge).where(
                            KnowledgeEdge.collection_id == collection_id,
                            KnowledgeEdge.kind.in_(GRAPHRAG_SEMANTIC_EDGE_KINDS),
                            KnowledgeEdge.source_node_id.in_(semantic_node_ids),
                            KnowledgeEdge.target_node_id.in_(semantic_node_ids),
                        )
                    )
                )
                .scalars()
                .all()
            )
        if semantic_edges:
            return semantic_nodes, semantic_edges, "semantic"

    nodes = (
        (
            await db.execute(
                select(KnowledgeNode).where(KnowledgeNode.collection_id == collection_id)
            )
        )
        .scalars()
        .all()
    )
    edges = (
        (
            await db.execute(
                select(KnowledgeEdge).where(KnowledgeEdge.collection_id == collection_id)
            )
        )
        .scalars()
        .all()
    )
    return nodes, edges, "knowledge_fallback"


async def _build_structural_community_points(
    db: Any,
    *,
    collection_id: uuid.UUID,
    include_node_kinds: set[KnowledgeNodeKind] | None,
    exclude_node_kinds: set[KnowledgeNodeKind] | None,
    include_edge_kinds: set[KnowledgeEdgeKind] | None,
    page: int,
    limit: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    graph_nodes, graph_edges, graph_source = await _load_community_graph(db, collection_id)
    if graph_source == "knowledge_fallback":
        warnings.append(
            "No symbol dependency graph found. Using knowledge graph communities as fallback."
        )

    if include_node_kinds:
        graph_nodes = [node for node in graph_nodes if node.kind in include_node_kinds]
    if exclude_node_kinds:
        graph_nodes = [node for node in graph_nodes if node.kind not in exclude_node_kinds]

    node_ids = {node.id for node in graph_nodes}
    if not node_ids:
        return [], warnings

    graph_edges = [
        edge
        for edge in graph_edges
        if edge.source_node_id in node_ids and edge.target_node_id in node_ids
    ]
    if include_edge_kinds:
        graph_edges = [edge for edge in graph_edges if edge.kind in include_edge_kinds]

    node_to_community, communities = _compute_symbol_communities(graph_nodes, graph_edges)
    node_by_id = {node.id: node for node in graph_nodes}

    pair_weights: dict[tuple[str, str], float] = defaultdict(float)
    for edge in graph_edges:
        source_community = node_to_community.get(edge.source_node_id)
        target_community = node_to_community.get(edge.target_node_id)
        if not source_community or not target_community or source_community == target_community:
            continue
        pair = (
            (source_community, target_community)
            if source_community < target_community
            else (target_community, source_community)
        )
        pair_weights[pair] += 1.0

    ordered_communities = sorted(
        communities.values(),
        key=lambda item: (-int(item["size"]), str(item["label"])),
    )
    if page > 0 or limit > 0:
        start = page * limit
        end = start + limit
        ordered_communities = ordered_communities[start:end]

    point_ids = [str(item["id"]) for item in ordered_communities]
    point_id_set = set(point_ids)
    scoped_weights = {
        pair: weight
        for pair, weight in pair_weights.items()
        if pair[0] in point_id_set and pair[1] in point_id_set
    }
    positions = _layout_code_structure_points(point_ids, scoped_weights)

    points: list[dict[str, Any]] = []
    for community in ordered_communities:
        member_ids = list(community.get("member_node_ids", []))
        member_nodes = [node_by_id[node_id] for node_id in member_ids if node_id in node_by_id]
        profile = _community_profile(member_nodes)
        dominant_domain = profile["dominant_domain"]
        dominant_ratio = float(profile["dominant_ratio"])
        misplaced_nodes: list[dict[str, Any]] = []
        if dominant_domain and dominant_ratio >= 0.6:
            for member in member_nodes:
                member_domain = profile["node_domain_by_id"].get(member.id)
                if member_domain and member_domain != dominant_domain:
                    misplaced_nodes.append(
                        {
                            "id": str(member.id),
                            "name": member.name,
                            "kind": member.kind.value,
                            "domain": member_domain,
                        }
                    )

        x, y = positions.get(str(community["id"]), (0.0, 0.0))
        points.append(
            {
                "id": str(community["id"]),
                "label": str(community["label"]),
                "x": x,
                "y": y,
                "member_count": int(community["size"]),
                "cohesion": float(community["cohesion"]),
                "top_kinds": profile["top_kinds"],
                "domain_counts": profile["domain_counts"],
                "dominant_domain": dominant_domain,
                "dominant_ratio": dominant_ratio,
                "summary": None,
                "anchor_node_id": str(community["sample_nodes"][0]["id"])
                if community.get("sample_nodes")
                else "",
                "sample_nodes": list(community.get("sample_nodes", [])),
                "member_node_ids": [
                    str(member_node_id) for member_node_id in community.get("member_node_ids", [])
                ],
                "vector": [],
                "source_refs": profile["source_refs"],
                "name_tokens": profile["name_tokens"],
                "misplaced_nodes": misplaced_nodes,
            }
        )

    _normalize_xy(points)
    return points, warnings


def _dedupe_traces(traces: list[list[uuid.UUID]]) -> list[list[uuid.UUID]]:
    ordered = sorted(
        traces,
        key=lambda trace: (-len(trace), "->".join(str(node_id) for node_id in trace)),
    )
    kept: list[list[uuid.UUID]] = []
    kept_keys: list[str] = []
    for trace in ordered:
        trace_key = "->".join(str(node_id) for node_id in trace)
        if any(trace_key in existing_key for existing_key in kept_keys):
            continue
        kept.append(trace)
        kept_keys.append(trace_key)
    return kept


def _trace_process_paths(
    *,
    entry_id: uuid.UUID,
    outgoing: dict[uuid.UUID, list[uuid.UUID]],
    max_depth: int,
    max_branching: int,
    min_steps: int,
) -> list[list[uuid.UUID]]:
    traces: list[list[uuid.UUID]] = []
    queue: deque[tuple[uuid.UUID, list[uuid.UUID]]] = deque([(entry_id, [entry_id])])

    while queue and len(traces) < max_branching * 3:
        current, path = queue.popleft()
        next_nodes = outgoing.get(current, [])
        if not next_nodes or len(path) >= max_depth:
            if len(path) >= min_steps:
                traces.append(path)
            continue

        added = False
        for next_node in next_nodes[:max_branching]:
            if next_node in path:
                continue
            queue.append((next_node, [*path, next_node]))
            added = True
        if not added and len(path) >= min_steps:
            traces.append(path)

    return traces


def _detect_processes(
    symbol_nodes: list[KnowledgeNode],
    symbol_edges: list[KnowledgeEdge],
    community_by_node_id: dict[uuid.UUID, str],
) -> list[dict[str, Any]]:
    if not symbol_nodes:
        return []

    node_by_id = {node.id: node for node in symbol_nodes}
    outgoing_raw: dict[uuid.UUID, set[uuid.UUID]] = defaultdict(set)
    incoming_degree: dict[uuid.UUID, int] = defaultdict(int)

    for edge in symbol_edges:
        src = edge.source_node_id
        dst = edge.target_node_id
        if src == dst or src not in node_by_id or dst not in node_by_id:
            continue
        if dst not in outgoing_raw[src]:
            outgoing_raw[src].add(dst)
            incoming_degree[dst] += 1

    outgoing: dict[uuid.UUID, list[uuid.UUID]] = {}
    for node_id, targets in outgoing_raw.items():
        outgoing[node_id] = sorted(
            targets, key=lambda target_id: _symbol_sort_key(node_by_id[target_id])
        )

    candidates = [node.id for node in symbol_nodes if outgoing.get(node.id)]
    candidates.sort(
        key=lambda node_id: (
            incoming_degree.get(node_id, 0),
            -len(outgoing.get(node_id, [])),
            _symbol_sort_key(node_by_id[node_id]),
        )
    )

    entry_points = [node_id for node_id in candidates if incoming_degree.get(node_id, 0) == 0][:200]
    if not entry_points:
        entry_points = candidates[:200]

    all_traces: list[list[uuid.UUID]] = []
    for entry_id in entry_points:
        all_traces.extend(
            _trace_process_paths(
                entry_id=entry_id,
                outgoing=outgoing,
                max_depth=10,
                max_branching=4,
                min_steps=2,
            )
        )
        if len(all_traces) >= 200:
            break

    traces = _dedupe_traces(all_traces)
    traces = sorted(
        traces, key=lambda trace: (-len(trace), "->".join(str(node_id) for node_id in trace))
    )[:75]

    processes: list[dict[str, Any]] = []
    for index, trace in enumerate(traces, start=1):
        entry = node_by_id[trace[0]]
        terminal = node_by_id[trace[-1]]
        community_ids = sorted(
            {community_by_node_id[node_id] for node_id in trace if node_id in community_by_node_id}
        )
        process_type = "cross_community" if len(community_ids) > 1 else "intra_community"
        process_id = f"proc_{index}"
        processes.append(
            {
                "id": process_id,
                "label": f"{entry.name} -> {terminal.name}",
                "process_type": process_type,
                "step_count": len(trace),
                "community_ids": community_ids,
                "entry_node_id": str(entry.id),
                "terminal_node_id": str(terminal.id),
                "steps": [
                    {
                        "step": step_index + 1,
                        "node_id": str(node_by_id[node_id].id),
                        "node_name": node_by_id[node_id].name,
                        "node_kind": node_by_id[node_id].kind.value,
                        "node_natural_key": node_by_id[node_id].natural_key,
                    }
                    for step_index, node_id in enumerate(trace)
                ],
            }
        )

    return processes


def _serialize_graphrag_node(
    node: KnowledgeNode,
    *,
    community_mode: str,
    focused_community_id: str | None,
    community_by_node_id: dict[uuid.UUID, str],
    communities: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    meta = _safe_meta(node.meta)
    community_id = community_by_node_id.get(node.id) if community_mode != "none" else None
    community_summary = communities.get(community_id) if community_id else None
    meta["community_id"] = community_id
    meta["community_label"] = community_summary["label"] if community_summary else None
    meta["community_size"] = community_summary["size"] if community_summary else None
    meta["community_cohesion"] = community_summary["cohesion"] if community_summary else None
    meta["community_focus"] = bool(focused_community_id and community_id == focused_community_id)
    return {
        "id": str(node.id),
        "natural_key": node.natural_key,
        "kind": node.kind.value,
        "name": node.name,
        "meta": meta,
    }


async def _resolve_knowledge_node(
    db: Any,
    collection_id: uuid.UUID,
    node_ref: str,
) -> KnowledgeNode | None:
    node: KnowledgeNode | None = None
    try:
        node_uuid = uuid.UUID(node_ref)
    except ValueError:
        node_uuid = None

    if node_uuid:
        node = (
            await db.execute(
                select(KnowledgeNode).where(
                    KnowledgeNode.collection_id == collection_id,
                    KnowledgeNode.id == node_uuid,
                )
            )
        ).scalar_one_or_none()
    if node:
        return node

    return (
        (
            await db.execute(
                select(KnowledgeNode)
                .where(
                    KnowledgeNode.collection_id == collection_id,
                    KnowledgeNode.natural_key == node_ref,
                )
                .limit(1)
            )
        )
        .scalars()
        .first()
    )


@router.post("/scenarios")
async def create_scenario(request: Request, body: CreateScenarioRequest) -> dict:
    user_id = _user_id_or_401(request)
    try:
        collection_id = uuid.UUID(body.collection_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid collection_id") from e

    async with get_db_session() as db:
        await _ensure_owner(db, collection_id, user_id)
        scenario = await create_to_be_scenario(
            session=db,
            collection_id=collection_id,
            name=body.name,
            user_id=user_id,
        )
        await db.commit()
        return {
            "id": str(scenario.id),
            "collection_id": str(scenario.collection_id),
            "name": scenario.name,
            "base_scenario_id": str(scenario.base_scenario_id)
            if scenario.base_scenario_id
            else None,
            "is_as_is": scenario.is_as_is,
            "version": scenario.version,
        }


@router.get("/scenarios")
async def list_scenarios(request: Request, collection_id: str | None = None) -> dict:
    """List scenarios, optionally filtered by collection."""
    user_id = _user_id_or_401(request)
    async with get_db_session() as db:
        stmt = select(TwinScenario)
        if collection_id:
            try:
                collection_uuid = uuid.UUID(collection_id)
            except ValueError as e:
                raise HTTPException(status_code=400, detail="Invalid collection_id") from e
            await _ensure_member(db, collection_uuid, user_id)
            stmt = stmt.where(TwinScenario.collection_id == collection_uuid)

        scenarios = (
            (await db.execute(stmt.order_by(TwinScenario.created_at.desc()))).scalars().all()
        )
        if not collection_id:
            allowed: list[TwinScenario] = []
            for scenario in scenarios:
                if await _can_access_collection(db, scenario.collection_id, user_id):
                    allowed.append(scenario)
            scenarios = allowed

        return {
            "scenarios": [
                {
                    "id": str(s.id),
                    "collection_id": str(s.collection_id),
                    "name": s.name,
                    "is_as_is": s.is_as_is,
                    "version": s.version,
                    "base_scenario_id": str(s.base_scenario_id) if s.base_scenario_id else None,
                }
                for s in scenarios
            ]
        }


@router.get("/scenarios/{scenario_id}")
async def get_scenario(request: Request, scenario_id: str) -> dict:
    user_id = _user_id_or_401(request)
    async with get_db_session() as db:
        scenario = await _load_scenario(db, scenario_id)
        await _ensure_member(db, scenario.collection_id, user_id)
        return {
            "id": str(scenario.id),
            "collection_id": str(scenario.collection_id),
            "name": scenario.name,
            "base_scenario_id": str(scenario.base_scenario_id)
            if scenario.base_scenario_id
            else None,
            "is_as_is": scenario.is_as_is,
            "version": scenario.version,
            "meta": scenario.meta,
            "created_at": scenario.created_at,
            "updated_at": scenario.updated_at,
        }


@router.post("/scenarios/{scenario_id}/intents")
async def create_intent(request: Request, scenario_id: str, body: ArchitectureIntentV1) -> dict:
    user_id = _user_id_or_401(request)

    async with get_db_session() as db:
        scenario = await _load_scenario(db, scenario_id)
        await _ensure_owner(db, scenario.collection_id, user_id)

        if body.scenario_id != scenario.id:
            raise HTTPException(status_code=400, detail="scenario_id mismatch")

        try:
            intent = await submit_intent(
                session=db,
                scenario=scenario,
                intent=body,
                requested_by=user_id,
                auto_execute=True,
            )
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e

        await db.commit()
        return {
            "id": str(intent.id),
            "status": intent.status.value,
            "risk_level": intent.risk_level.value,
            "requires_approval": intent.requires_approval,
            "scenario_version": scenario.version,
        }


@router.post("/scenarios/{scenario_id}/intents/{intent_id}/approve")
async def approve_intent(request: Request, scenario_id: str, intent_id: str) -> dict:
    user_id = _user_id_or_401(request)

    try:
        intent_uuid = uuid.UUID(intent_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid intent_id") from e

    async with get_db_session() as db:
        scenario = await _load_scenario(db, scenario_id)
        await _ensure_owner(db, scenario.collection_id, user_id)

        try:
            intent = await approve_and_execute_intent(db, scenario, intent_uuid)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e

        await db.commit()
        return {
            "id": str(intent.id),
            "status": intent.status.value,
            "scenario_version": scenario.version,
        }


@router.get("/scenarios/{scenario_id}/patches")
async def get_patches(request: Request, scenario_id: str) -> dict:
    user_id = _user_id_or_401(request)

    async with get_db_session() as db:
        scenario = await _load_scenario(db, scenario_id)
        await _ensure_member(db, scenario.collection_id, user_id)
        patches = await list_scenario_patches(db, scenario.id)
        return {
            "scenario_id": str(scenario.id),
            "patches": [
                {
                    "id": str(p.id),
                    "scenario_version": p.scenario_version,
                    "intent_id": str(p.intent_id) if p.intent_id else None,
                    "ops": p.patch_ops,
                    "created_at": p.created_at,
                }
                for p in patches
            ],
        }


@router.get("/scenarios/{scenario_id}/graph")
async def graph_view(
    request: Request,
    scenario_id: str,
    layer: str | None = Query(default=None),
    projection: str | None = Query(default=GraphProjection.CODE_SYMBOL.value),
    entity_level: str | None = Query(default=None),
    include_kinds: str | None = Query(default=None),
    exclude_kinds: str | None = Query(default=None),
    page: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=5000),
) -> dict:
    user_id = _user_id_or_401(request)
    layer_enum = _parse_layer(layer)
    projection_enum = _parse_projection(projection)

    async with get_db_session() as db:
        scenario = await _load_scenario(db, scenario_id)
        await _ensure_member(db, scenario.collection_id, user_id)
        return await get_scenario_graph(
            db,
            scenario.id,
            layer_enum,
            page,
            limit,
            projection=projection_enum,
            entity_level=entity_level,
            include_kinds=_parse_kind_filter(include_kinds),
            exclude_kinds=_parse_kind_filter(exclude_kinds),
        )


@router.get("/scenarios/{scenario_id}/graph/neighborhood")
async def graph_neighborhood_view(
    request: Request,
    scenario_id: str,
    node_id: str = Query(min_length=1),
    projection: str | None = Query(default=GraphProjection.CODE_SYMBOL.value),
    entity_level: str | None = Query(default=None),
    include_kinds: str | None = Query(default=None),
    exclude_kinds: str | None = Query(default=None),
    hops: int = Query(default=1, ge=1, le=4),
    limit: int = Query(default=200, ge=1, le=2000),
) -> dict:
    user_id = _user_id_or_401(request)
    projection_enum = _parse_projection(projection)

    async with get_db_session() as db:
        scenario = await _load_scenario(db, scenario_id)
        await _ensure_member(db, scenario.collection_id, user_id)
        graph = await get_full_scenario_graph(
            db,
            scenario.id,
            layer=None,
            projection=projection_enum,
            entity_level=entity_level,
            include_kinds=_parse_kind_filter(include_kinds),
            exclude_kinds=_parse_kind_filter(exclude_kinds),
        )

        all_nodes = graph["nodes"]
        resolved_node_id = node_id
        if not any(str(node.get("id")) == node_id for node in all_nodes):
            by_key = next(
                (
                    str(node.get("id"))
                    for node in all_nodes
                    if str(node.get("natural_key")) == node_id
                ),
                None,
            )
            if by_key:
                resolved_node_id = by_key
            else:
                raise HTTPException(status_code=404, detail="Node not found in projected graph")

        nodes, edges = _extract_neighborhood(
            nodes=all_nodes,
            edges=graph["edges"],
            root_node_id=resolved_node_id,
            hops=hops,
            limit=limit,
        )
        return {
            "scenario_id": str(scenario.id),
            "node_id": resolved_node_id,
            "hops": hops,
            "projection": graph["projection"],
            "graph": {
                "nodes": nodes,
                "edges": edges,
                "page": 0,
                "limit": limit,
                "total_nodes": len(nodes),
                "projection": graph["projection"],
                "entity_level": graph["entity_level"],
                "grouping_strategy": graph["grouping_strategy"],
                "excluded_kinds": graph["excluded_kinds"],
            },
        }


@router.get("/collections/{collection_id}/status")
async def twin_status_view(
    request: Request,
    collection_id: str,
    scenario_id: str | None = Query(default=None),
) -> dict:
    user_id = _user_id_or_401(request)
    collection_uuid = _parse_collection_id(collection_id)
    scenario_uuid = _parse_optional_scenario_id(scenario_id)
    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        return await get_collection_twin_status(
            db,
            collection_id=collection_uuid,
            scenario_id=scenario_uuid,
        )


@router.get("/collections/{collection_id}/timeline")
async def twin_timeline_view(
    request: Request,
    collection_id: str,
    scenario_id: str | None = Query(default=None),
    source_id: str | None = Query(default=None),
    event_type: str | None = Query(default=None),
    status: str | None = Query(default=None),
    from_ts: str | None = Query(default=None),
    to_ts: str | None = Query(default=None),
    page: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict:
    del scenario_id
    user_id = _user_id_or_401(request)
    collection_uuid = _parse_collection_id(collection_id)
    try:
        source_uuid = uuid.UUID(source_id) if source_id else None
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid source_id") from e
    try:
        parsed_from = parse_timestamp_value(from_ts)
        parsed_to = parse_timestamp_value(to_ts)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid timestamp format") from e

    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        return await list_collection_twin_events(
            db,
            collection_id=collection_uuid,
            page=page,
            limit=limit,
            source_id=source_uuid,
            event_type=event_type,
            status=status,
            from_ts=parsed_from,
            to_ts=parsed_to,
        )


@router.post("/collections/{collection_id}/refresh")
async def twin_refresh(
    request: Request,
    collection_id: str,
    body: TwinRefreshRequest,
) -> dict:
    user_id = _user_id_or_401(request)
    collection_uuid = _parse_collection_id(collection_id)
    try:
        source_ids = coerce_source_ids(body.source_ids)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid source_ids") from e
    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        payload = await trigger_collection_refresh(
            db,
            collection_id=collection_uuid,
            source_ids=source_ids or None,
            force=body.force,
        )
        await db.commit()
        return payload


@router.get("/collections/{collection_id}/views/diff")
async def twin_diff_view(
    request: Request,
    collection_id: str,
    from_version: int = Query(ge=1),
    to_version: int = Query(ge=1),
    scenario_id: str | None = Query(default=None),
) -> dict:
    user_id = _user_id_or_401(request)
    collection_uuid = _parse_collection_id(collection_id)
    scenario_uuid = _parse_optional_scenario_id(scenario_id)
    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        try:
            return await get_collection_twin_diff(
                db,
                collection_id=collection_uuid,
                scenario_id=scenario_uuid,
                from_version=from_version,
                to_version=to_version,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e


@router.get("/collections/{collection_id}/analysis/summary")
async def twin_analysis_summary(
    request: Request,
    collection_id: str,
    scenario_id: str | None = Query(default=None),
    engines: str | None = Query(default=None),
) -> dict:
    user_id = _user_id_or_401(request)
    settings = get_settings()
    collection_uuid = _parse_collection_id(collection_id)
    scenario_uuid = _parse_optional_scenario_id(scenario_id)
    selected_engines = _parse_engines_query(engines)
    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        try:
            return await get_codebase_summary_multi(
                db,
                collection_id=collection_uuid,
                scenario_id=scenario_uuid,
                cache_ttl_seconds=settings.twin_analysis_cache_ttl_seconds,
                engines=selected_engines,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e


@router.get("/collections/{collection_id}/analysis/methods")
async def twin_analysis_methods(
    request: Request,
    collection_id: str,
    scenario_id: str | None = Query(default=None),
    query: str | None = Query(default=None),
    page: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    engines: str | None = Query(default=None),
) -> dict:
    user_id = _user_id_or_401(request)
    settings = get_settings()
    collection_uuid = _parse_collection_id(collection_id)
    scenario_uuid = _parse_optional_scenario_id(scenario_id)
    selected_engines = _parse_engines_query(engines)
    try:
        safe_query = sanitize_regex_query(query)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        try:
            return await list_methods_multi(
                db,
                collection_id=collection_uuid,
                scenario_id=scenario_uuid,
                query=safe_query,
                page=page,
                limit=limit,
                cache_ttl_seconds=settings.twin_analysis_cache_ttl_seconds,
                engines=selected_engines,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e


@router.get("/collections/{collection_id}/analysis/calls")
async def twin_analysis_calls(
    request: Request,
    collection_id: str,
    scenario_id: str | None = Query(default=None),
    page: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    engines: str | None = Query(default=None),
) -> dict:
    user_id = _user_id_or_401(request)
    settings = get_settings()
    collection_uuid = _parse_collection_id(collection_id)
    scenario_uuid = _parse_optional_scenario_id(scenario_id)
    selected_engines = _parse_engines_query(engines)
    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        try:
            return await list_calls_multi(
                db,
                collection_id=collection_uuid,
                scenario_id=scenario_uuid,
                page=page,
                limit=limit,
                cache_ttl_seconds=settings.twin_analysis_cache_ttl_seconds,
                engines=selected_engines,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e


@router.get("/collections/{collection_id}/analysis/cfg")
async def twin_analysis_cfg(
    request: Request,
    collection_id: str,
    node_ref: str = Query(min_length=1),
    scenario_id: str | None = Query(default=None),
    depth: int = Query(default=2, ge=1, le=8),
    engines: str | None = Query(default=None),
) -> dict:
    user_id = _user_id_or_401(request)
    settings = get_settings()
    collection_uuid = _parse_collection_id(collection_id)
    scenario_uuid = _parse_optional_scenario_id(scenario_id)
    selected_engines = _parse_engines_query(engines)
    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        try:
            return await get_cfg_multi(
                db,
                collection_id=collection_uuid,
                scenario_id=scenario_uuid,
                node_ref=node_ref,
                depth=depth,
                cache_ttl_seconds=settings.twin_analysis_cache_ttl_seconds,
                engines=selected_engines,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e


@router.get("/collections/{collection_id}/analysis/variable-flow")
async def twin_analysis_variable_flow(
    request: Request,
    collection_id: str,
    node_ref: str = Query(min_length=1),
    variable: str | None = Query(default=None),
    scenario_id: str | None = Query(default=None),
    max_hops: int = Query(default=6, ge=1, le=20),
    engines: str | None = Query(default=None),
) -> dict:
    user_id = _user_id_or_401(request)
    settings = get_settings()
    collection_uuid = _parse_collection_id(collection_id)
    scenario_uuid = _parse_optional_scenario_id(scenario_id)
    selected_engines = _parse_engines_query(engines)
    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        try:
            return await get_variable_flow_multi(
                db,
                collection_id=collection_uuid,
                scenario_id=scenario_uuid,
                node_ref=node_ref,
                variable=variable,
                max_hops=max_hops,
                cache_ttl_seconds=settings.twin_analysis_cache_ttl_seconds,
                engines=selected_engines,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e


@router.get("/collections/{collection_id}/analysis/taint/sources")
async def twin_analysis_taint_sources(
    request: Request,
    collection_id: str,
    scenario_id: str | None = Query(default=None),
    language: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=300),
    engines: str | None = Query(default=None),
) -> dict:
    user_id = _user_id_or_401(request)
    settings = get_settings()
    collection_uuid = _parse_collection_id(collection_id)
    scenario_uuid = _parse_optional_scenario_id(scenario_id)
    selected_engines = _parse_engines_query(engines)
    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        try:
            return await find_taint_sources_multi(
                db,
                collection_id=collection_uuid,
                scenario_id=scenario_uuid,
                language=language,
                limit=limit,
                cache_ttl_seconds=settings.twin_analysis_cache_ttl_seconds,
                engines=selected_engines,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e


@router.get("/collections/{collection_id}/analysis/taint/sinks")
async def twin_analysis_taint_sinks(
    request: Request,
    collection_id: str,
    scenario_id: str | None = Query(default=None),
    language: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=300),
    engines: str | None = Query(default=None),
) -> dict:
    user_id = _user_id_or_401(request)
    settings = get_settings()
    collection_uuid = _parse_collection_id(collection_id)
    scenario_uuid = _parse_optional_scenario_id(scenario_id)
    selected_engines = _parse_engines_query(engines)
    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        try:
            return await find_taint_sinks_multi(
                db,
                collection_id=collection_uuid,
                scenario_id=scenario_uuid,
                language=language,
                limit=limit,
                cache_ttl_seconds=settings.twin_analysis_cache_ttl_seconds,
                engines=selected_engines,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e


@router.post("/collections/{collection_id}/analysis/taint/flows")
async def twin_analysis_taint_flows(
    request: Request,
    collection_id: str,
    body: TaintFlowsRequest,
) -> dict:
    user_id = _user_id_or_401(request)
    settings = get_settings()
    collection_uuid = _parse_collection_id(collection_id)
    scenario_uuid = _parse_optional_scenario_id(body.scenario_id)
    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        try:
            return await find_taint_flows_multi(
                db,
                collection_id=collection_uuid,
                scenario_id=scenario_uuid,
                language=body.language,
                max_hops=body.max_hops,
                max_results=body.max_results,
                cache_ttl_seconds=settings.twin_analysis_cache_ttl_seconds,
                engines=body.engines,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e


@router.post("/collections/{collection_id}/analysis/findings/store")
async def twin_store_findings(
    request: Request,
    collection_id: str,
    body: StoreFindingsRequest,
) -> dict:
    user_id = _user_id_or_401(request)
    collection_uuid = _parse_collection_id(collection_id)
    scenario_uuid = _parse_optional_scenario_id(body.scenario_id)
    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        payload = await store_findings(
            db,
            collection_id=collection_uuid,
            scenario_id=scenario_uuid,
            findings=body.findings,
        )
        await db.commit()
        return payload


@router.get("/collections/{collection_id}/analysis/findings")
async def twin_list_findings(
    request: Request,
    collection_id: str,
    scenario_id: str | None = Query(default=None),
    status: str | None = Query(default=None),
    min_severity: str | None = Query(default=None),
    page: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
) -> dict:
    user_id = _user_id_or_401(request)
    collection_uuid = _parse_collection_id(collection_id)
    scenario_uuid = _parse_optional_scenario_id(scenario_id)
    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        return await list_findings(
            db,
            collection_id=collection_uuid,
            scenario_id=scenario_uuid,
            status=status,
            min_severity=min_severity,
            page=page,
            limit=limit,
        )


@router.get("/collections/{collection_id}/analysis/findings/sarif")
async def twin_export_findings_sarif(
    request: Request,
    collection_id: str,
    scenario_id: str | None = Query(default=None),
    status: str | None = Query(default=None),
    min_severity: str | None = Query(default=None),
) -> dict:
    user_id = _user_id_or_401(request)
    collection_uuid = _parse_collection_id(collection_id)
    scenario_uuid = _parse_optional_scenario_id(scenario_id)
    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        return await export_findings_sarif(
            db,
            collection_id=collection_uuid,
            scenario_id=scenario_uuid,
            status=status,
            min_severity=min_severity,
        )


@router.get("/collections/{collection_id}/views/arc42")
async def arc42_view(
    request: Request,
    collection_id: str,
    scenario_id: str | None = Query(default=None),
    section: str | None = Query(default=None),
    regenerate: bool = Query(default=False),
) -> dict[str, Any]:
    user_id = _user_id_or_401(request)
    collection_uuid = _parse_collection_id(collection_id)
    section_key = normalize_arc42_section_key(section)
    if section and not section_key:
        raise HTTPException(status_code=400, detail="Invalid arc42 section")

    settings = get_settings()
    if not settings.arch_docs_enabled:
        raise HTTPException(status_code=503, detail="Architecture docs are disabled")

    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        scenario = await _resolve_view_scenario(db, collection_uuid, scenario_id)
        artifact_name = _arc42_artifact_name(scenario.id)

        existing = (
            await db.execute(
                select(KnowledgeArtifact).where(
                    KnowledgeArtifact.collection_id == collection_uuid,
                    KnowledgeArtifact.kind == KnowledgeArtifactKind.ARC42,
                    KnowledgeArtifact.name == artifact_name,
                )
            )
        ).scalar_one_or_none()

        if existing and not regenerate:
            stored_meta = existing.meta or {}
            stored_sections = stored_meta.get("sections")
            if isinstance(stored_sections, dict):
                if section_key:
                    filtered_sections = (
                        {section_key: stored_sections.get(section_key, "")}
                        if section_key in SECTION_TITLES
                        else {}
                    )
                    markdown = ""
                    if section_key in filtered_sections:
                        markdown = (
                            f"# arc42 - {scenario.name}\n\n"
                            f"## {SECTION_TITLES[section_key]}\n"
                            f"{filtered_sections[section_key]}\n"
                        )
                    return {
                        "collection_id": str(collection_uuid),
                        "scenario": _serialize_scenario(scenario),
                        "artifact": {
                            "id": str(existing.id),
                            "name": existing.name,
                            "kind": existing.kind.value,
                            "cached": True,
                        },
                        "section": section_key,
                        "arc42": {
                            "title": f"arc42 - {scenario.name}",
                            "generated_at": stored_meta.get("generated_at"),
                            "sections": filtered_sections,
                            "markdown": markdown,
                            "warnings": stored_meta.get("warnings", []),
                            "confidence_summary": stored_meta.get("confidence_summary", {}),
                            "section_coverage": {
                                section_key: bool(filtered_sections.get(section_key))
                            },
                        },
                        "facts_hash": stored_meta.get("facts_hash"),
                        "warnings": stored_meta.get("warnings", []),
                    }

                return {
                    "collection_id": str(collection_uuid),
                    "scenario": _serialize_scenario(scenario),
                    "artifact": {
                        "id": str(existing.id),
                        "name": existing.name,
                        "kind": existing.kind.value,
                        "cached": True,
                    },
                    "section": section_key,
                    "arc42": {
                        "title": f"arc42 - {scenario.name}",
                        "generated_at": stored_meta.get("generated_at"),
                        "sections": stored_sections,
                        "markdown": existing.content,
                        "warnings": stored_meta.get("warnings", []),
                        "confidence_summary": stored_meta.get("confidence_summary", {}),
                        "section_coverage": stored_meta.get("section_coverage", {}),
                    },
                    "facts_hash": stored_meta.get("facts_hash"),
                    "warnings": stored_meta.get("warnings", []),
                }

        bundle = await _build_arch_bundle(db, collection_id=collection_uuid, scenario=scenario)
        full_document = generate_arc42_from_facts(bundle, scenario, options={})
        selected_document = (
            generate_arc42_from_facts(bundle, scenario, options={"section": section_key})
            if section_key
            else full_document
        )

        artifact = await _upsert_artifact(
            db,
            collection_id=collection_uuid,
            kind=KnowledgeArtifactKind.ARC42,
            name=artifact_name,
            content=full_document.markdown,
            meta={
                "scenario_id": str(scenario.id),
                "generated_at": full_document.generated_at.isoformat(),
                "facts_hash": bundle.facts_hash(),
                "confidence_summary": full_document.confidence_summary,
                "section_coverage": full_document.section_coverage,
                "warnings": full_document.warnings,
                "sections": full_document.sections,
            },
        )
        await db.commit()

        return {
            "collection_id": str(collection_uuid),
            "scenario": _serialize_scenario(scenario),
            "artifact": {
                "id": str(artifact.id),
                "name": artifact.name,
                "kind": artifact.kind.value,
                "cached": False,
            },
            "section": section_key,
            "arc42": _serialize_arc42_document(selected_document),
            "facts_hash": bundle.facts_hash(),
            "facts_count": len(bundle.facts),
            "ports_adapters_count": len(bundle.ports_adapters),
            "warnings": sorted(set([*bundle.warnings, *full_document.warnings])),
        }


@router.get("/collections/{collection_id}/views/arc42/drift")
async def arc42_drift_view(
    request: Request,
    collection_id: str,
    scenario_id: str | None = Query(default=None),
    baseline_scenario_id: str | None = Query(default=None),
) -> dict[str, Any]:
    user_id = _user_id_or_401(request)
    collection_uuid = _parse_collection_id(collection_id)

    settings = get_settings()
    if not settings.arch_docs_enabled:
        raise HTTPException(status_code=503, detail="Architecture docs are disabled")

    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        scenario = await _resolve_view_scenario(db, collection_uuid, scenario_id)
        baseline = await _resolve_baseline_scenario(
            db,
            collection_id=collection_uuid,
            scenario=scenario,
            baseline_scenario_id=baseline_scenario_id,
        )

        current_bundle = await _build_arch_bundle(
            db, collection_id=collection_uuid, scenario=scenario
        )
        baseline_bundle = None
        if baseline:
            baseline_bundle = await _build_arch_bundle(
                db,
                collection_id=collection_uuid,
                scenario=baseline,
            )

        report = compute_arc42_drift(
            current_bundle,
            baseline_bundle,
            baseline_scenario_id=baseline.id if baseline else None,
        )
        deltas = [asdict(delta) for delta in report.deltas]

        by_type: dict[str, int] = {}
        for delta in deltas:
            delta_type = str(delta.get("delta_type"))
            by_type[delta_type] = by_type.get(delta_type, 0) + 1

        severity = "low" if len(deltas) < 10 else "medium"
        return {
            "collection_id": str(collection_uuid),
            "scenario": _serialize_scenario(scenario),
            "baseline_scenario": _serialize_scenario(baseline) if baseline else None,
            "generated_at": report.generated_at.isoformat(),
            "current_hash": report.current_hash,
            "baseline_hash": report.baseline_hash,
            "summary": {
                "total": len(deltas),
                "by_type": by_type,
                "severity": severity,
            },
            "deltas": deltas,
            "warnings": report.warnings,
        }


@router.get("/collections/{collection_id}/views/ports-adapters")
async def ports_adapters_view(
    request: Request,
    collection_id: str,
    scenario_id: str | None = Query(default=None),
    direction: str | None = Query(default=None),
    container: str | None = Query(default=None),
) -> dict[str, Any]:
    user_id = _user_id_or_401(request)
    collection_uuid = _parse_collection_id(collection_id)

    direction_normalized = direction.strip().lower() if direction else None
    if direction_normalized and direction_normalized not in {"inbound", "outbound"}:
        raise HTTPException(status_code=400, detail="Invalid direction")

    settings = get_settings()
    if not settings.arch_docs_enabled:
        raise HTTPException(status_code=503, detail="Architecture docs are disabled")

    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        scenario = await _resolve_view_scenario(db, collection_uuid, scenario_id)
        bundle = await _build_arch_bundle(db, collection_id=collection_uuid, scenario=scenario)

        facts = bundle.ports_adapters
        if direction_normalized:
            facts = [fact for fact in facts if fact.direction == direction_normalized]
        if container:
            normalized_container = container.strip().lower()
            facts = [
                fact
                for fact in facts
                if (fact.container or "").strip().lower() == normalized_container
            ]

        inbound_count = sum(1 for fact in facts if fact.direction == "inbound")
        outbound_count = sum(1 for fact in facts if fact.direction == "outbound")

        return {
            "collection_id": str(collection_uuid),
            "scenario": _serialize_scenario(scenario),
            "summary": {
                "total": len(facts),
                "inbound": inbound_count,
                "outbound": outbound_count,
            },
            "filters": {
                "direction": direction_normalized,
                "container": container,
            },
            "items": [_serialize_port_adapter_fact(fact) for fact in facts],
            "warnings": bundle.warnings,
        }


@router.get("/collections/{collection_id}/views/erm")
async def erm_view(
    request: Request,
    collection_id: str,
    scenario_id: str | None = Query(default=None),
    include_mermaid: bool = Query(default=True),
) -> dict[str, Any]:
    user_id = _user_id_or_401(request)
    collection_uuid = _parse_collection_id(collection_id)

    settings = get_settings()
    if not settings.arch_docs_enabled:
        raise HTTPException(status_code=503, detail="Architecture docs are disabled")

    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        scenario = await _resolve_view_scenario(db, collection_uuid, scenario_id)

        table_nodes = (
            (
                await db.execute(
                    select(KnowledgeNode).where(
                        KnowledgeNode.collection_id == collection_uuid,
                        KnowledgeNode.kind == KnowledgeNodeKind.DB_TABLE,
                    )
                )
            )
            .scalars()
            .all()
        )
        column_nodes = (
            (
                await db.execute(
                    select(KnowledgeNode).where(
                        KnowledgeNode.collection_id == collection_uuid,
                        KnowledgeNode.kind == KnowledgeNodeKind.DB_COLUMN,
                    )
                )
            )
            .scalars()
            .all()
        )

        table_by_id = {node.id: node for node in table_nodes}
        table_by_name = {str(node.name).strip(): node for node in table_nodes}
        column_by_id = {node.id: node for node in column_nodes}

        edges = (
            (
                await db.execute(
                    select(KnowledgeEdge).where(
                        KnowledgeEdge.collection_id == collection_uuid,
                        KnowledgeEdge.kind.in_(
                            [
                                KnowledgeEdgeKind.TABLE_HAS_COLUMN,
                                KnowledgeEdgeKind.COLUMN_FK_TO_COLUMN,
                            ]
                        ),
                    )
                )
            )
            .scalars()
            .all()
        )

        columns_by_table: dict[uuid.UUID, list[KnowledgeNode]] = defaultdict(list)
        fk_rows: list[dict[str, Any]] = []

        for edge in edges:
            if edge.kind == KnowledgeEdgeKind.TABLE_HAS_COLUMN:
                source = table_by_id.get(edge.source_node_id)
                target = column_by_id.get(edge.target_node_id)
                if not source or not target:
                    continue
                columns_by_table[source.id].append(target)
                continue

            source_col = column_by_id.get(edge.source_node_id)
            target_col = column_by_id.get(edge.target_node_id)
            if not source_col or not target_col:
                continue
            source_meta = source_col.meta if isinstance(source_col.meta, dict) else {}
            target_meta = target_col.meta if isinstance(target_col.meta, dict) else {}
            source_table = (
                source_meta.get("table")
                or _parse_db_table_from_natural_key(source_col.natural_key)
                or "unknown"
            )
            target_table = (
                target_meta.get("table")
                or _parse_db_table_from_natural_key(target_col.natural_key)
                or "unknown"
            )
            fk_rows.append(
                {
                    "id": str(edge.id),
                    "fk_name": (edge.meta or {}).get("fk_name"),
                    "source_table": source_table,
                    "source_column": source_col.name,
                    "target_table": target_table,
                    "target_column": target_col.name,
                    "source_column_node_id": str(source_col.id),
                    "target_column_node_id": str(target_col.id),
                }
            )

        for column in column_nodes:
            meta = column.meta if isinstance(column.meta, dict) else {}
            table_name = str(meta.get("table") or "").strip()
            if not table_name:
                table_name = _parse_db_table_from_natural_key(column.natural_key) or ""
            if not table_name:
                continue
            maybe_table = table_by_name.get(table_name)
            if maybe_table is None:
                continue
            if column not in columns_by_table[maybe_table.id]:
                columns_by_table[maybe_table.id].append(column)

        serialized_tables: list[dict[str, Any]] = []
        total_columns = 0
        for table in sorted(table_nodes, key=lambda row: str(row.name).lower()):
            serialized_columns = sorted(
                [_serialize_erm_column(col) for col in columns_by_table.get(table.id, [])],
                key=lambda row: str(row.get("name") or "").lower(),
            )
            total_columns += len(serialized_columns)
            serialized_tables.append(_serialize_erm_table(table, serialized_columns))

        fk_rows = sorted(
            fk_rows,
            key=lambda row: (
                str(row.get("source_table") or ""),
                str(row.get("source_column") or ""),
                str(row.get("target_table") or ""),
                str(row.get("target_column") or ""),
            ),
        )

        mermaid_payload: dict[str, Any] | None = None
        if include_mermaid:
            erd_artifact = (
                await db.execute(
                    select(KnowledgeArtifact)
                    .where(
                        KnowledgeArtifact.collection_id == collection_uuid,
                        KnowledgeArtifact.kind == KnowledgeArtifactKind.MERMAID_ERD,
                    )
                    .order_by(KnowledgeArtifact.created_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            if erd_artifact:
                mermaid_payload = {
                    "artifact_id": str(erd_artifact.id),
                    "name": erd_artifact.name,
                    "content": erd_artifact.content,
                    "meta": erd_artifact.meta or {},
                }

        warnings: list[str] = []
        if not serialized_tables:
            warnings.append("No DB_TABLE nodes found; ERM extraction may be incomplete.")
        if include_mermaid and mermaid_payload is None:
            warnings.append("No MERMAID_ERD artifact found; showing relational fallback only.")

        return {
            "collection_id": str(collection_uuid),
            "scenario": _serialize_scenario(scenario),
            "summary": {
                "tables": len(serialized_tables),
                "columns": total_columns,
                "foreign_keys": len(fk_rows),
                "has_mermaid": mermaid_payload is not None,
            },
            "tables": serialized_tables,
            "foreign_keys": fk_rows,
            "mermaid": mermaid_payload,
            "warnings": warnings,
        }


@router.get("/collections/{collection_id}/views/topology")
async def topology_view(
    request: Request,
    collection_id: str,
    scenario_id: str | None = Query(default=None),
    layer: str | None = Query(default=TwinLayer.DOMAIN_CONTAINER.value),
    projection: str | None = Query(default=GraphProjection.ARCHITECTURE.value),
    entity_level: str | None = Query(default=None),
    include_kinds: str | None = Query(default=None),
    exclude_kinds: str | None = Query(default=None),
    page: int = Query(default=0, ge=0),
    limit: int = Query(default=1000, ge=1, le=5000),
) -> dict:
    user_id = _user_id_or_401(request)
    collection_uuid = _parse_collection_id(collection_id)
    layer_enum = _parse_layer(layer)
    projection_enum = _parse_projection(projection)
    resolved_entity_level = _topology_entity_level(layer_enum, entity_level)

    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        scenario = await _resolve_view_scenario(db, collection_uuid, scenario_id)
        graph = await get_scenario_graph(
            db,
            scenario.id,
            None if projection_enum == GraphProjection.ARCHITECTURE else layer_enum,
            page,
            limit,
            projection=projection_enum,
            entity_level=resolved_entity_level,
            include_kinds=_parse_kind_filter(include_kinds),
            exclude_kinds=_parse_kind_filter(exclude_kinds),
        )
        return {
            "collection_id": str(collection_uuid),
            "scenario": _serialize_scenario(scenario),
            "layer": layer_enum.value if layer_enum else None,
            "projection": graph["projection"],
            "entity_level": graph["entity_level"],
            "grouping_strategy": graph["grouping_strategy"],
            "excluded_kinds": graph["excluded_kinds"],
            "graph": graph,
        }


@router.get("/collections/{collection_id}/views/deep-dive")
async def deep_dive_view(
    request: Request,
    collection_id: str,
    scenario_id: str | None = Query(default=None),
    layer: str | None = Query(default=TwinLayer.CODE_CONTROLFLOW.value),
    projection: str | None = Query(default=GraphProjection.CODE_FILE.value),
    entity_level: str | None = Query(default=None),
    include_kinds: str | None = Query(default=None),
    exclude_kinds: str | None = Query(default=None),
    mode: str | None = Query(default="file_dependency"),
    page: int = Query(default=0, ge=0),
    limit: int = Query(default=3000, ge=1, le=10000),
) -> dict:
    user_id = _user_id_or_401(request)
    collection_uuid = _parse_collection_id(collection_id)
    layer_enum = _parse_layer(layer)
    projection_enum = _parse_projection(projection)

    edge_kinds: set[str] | None = None
    if mode == "symbol_callgraph":
        projection_enum = GraphProjection.CODE_SYMBOL
        edge_kinds = {
            "symbol_calls_symbol",
            "symbol_references_symbol",
            "symbol_imports_symbol",
            "symbol_extends_symbol",
            "symbol_implements_symbol",
        }
    elif mode == "contains_hierarchy":
        projection_enum = GraphProjection.CODE_SYMBOL
        edge_kinds = {"symbol_contains_symbol"}
    elif mode in (None, "file_dependency"):
        projection_enum = GraphProjection.CODE_FILE
    else:
        raise HTTPException(status_code=400, detail="Invalid deep dive mode")

    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        scenario = await _resolve_view_scenario(db, collection_uuid, scenario_id)
        graph = await get_scenario_graph(
            db,
            scenario.id,
            layer_enum,
            page,
            limit,
            projection=projection_enum,
            entity_level=entity_level
            or ("file" if projection_enum == GraphProjection.CODE_FILE else "symbol"),
            include_kinds=_parse_kind_filter(include_kinds),
            exclude_kinds=_parse_kind_filter(exclude_kinds),
            include_edge_kinds=edge_kinds,
        )
        return {
            "collection_id": str(collection_uuid),
            "scenario": _serialize_scenario(scenario),
            "layer": layer_enum.value if layer_enum else None,
            "projection": graph["projection"],
            "entity_level": graph["entity_level"],
            "grouping_strategy": graph["grouping_strategy"],
            "excluded_kinds": graph["excluded_kinds"],
            "graph": graph,
        }


@router.get("/collections/{collection_id}/views/ui-map")
async def ui_map_view(
    request: Request,
    collection_id: str,
    scenario_id: str | None = Query(default=None),
    page: int = Query(default=0, ge=0),
    limit: int = Query(default=1200, ge=1, le=10000),
) -> dict:
    user_id = _user_id_or_401(request)
    collection_uuid = _parse_collection_id(collection_id)
    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        scenario = await _resolve_view_scenario(db, collection_uuid, scenario_id)
        full_graph = await get_full_scenario_graph(
            db,
            scenario.id,
            layer=None,
            projection=GraphProjection.CODE_SYMBOL,
        )
        projection = build_ui_map_projection(full_graph["nodes"], full_graph["edges"])
        graph = _paginate_graph(
            projection["graph"]["nodes"],
            projection["graph"]["edges"],
            page=page,
            limit=limit,
        )
        warnings: list[str] = []
        if projection["summary"]["routes"] == 0:
            warnings.append("No UI route nodes found. UI extraction may still be pending.")
        return {
            "collection_id": str(collection_uuid),
            "scenario": _serialize_scenario(scenario),
            "projection": "ui_map",
            "entity_level": "ui",
            "summary": projection["summary"],
            "warnings": warnings,
            "graph": graph,
        }


@router.get("/collections/{collection_id}/views/test-matrix")
async def test_matrix_view(
    request: Request,
    collection_id: str,
    scenario_id: str | None = Query(default=None),
    page: int = Query(default=0, ge=0),
    limit: int = Query(default=1200, ge=1, le=10000),
) -> dict:
    user_id = _user_id_or_401(request)
    collection_uuid = _parse_collection_id(collection_id)
    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        scenario = await _resolve_view_scenario(db, collection_uuid, scenario_id)
        full_graph = await get_full_scenario_graph(
            db,
            scenario.id,
            layer=None,
            projection=GraphProjection.CODE_SYMBOL,
        )
        projection = build_test_matrix_projection(full_graph["nodes"], full_graph["edges"])
        graph = _paginate_graph(
            projection["graph"]["nodes"],
            projection["graph"]["edges"],
            page=page,
            limit=limit,
        )
        warnings: list[str] = []
        if projection["summary"]["test_cases"] == 0:
            warnings.append("No TEST_CASE nodes found. Test extraction may still be pending.")
        return {
            "collection_id": str(collection_uuid),
            "scenario": _serialize_scenario(scenario),
            "projection": "test_matrix",
            "entity_level": "test_case",
            "summary": projection["summary"],
            "matrix": projection["matrix"],
            "warnings": warnings,
            "graph": graph,
        }


@router.get("/collections/{collection_id}/views/user-flows")
async def user_flows_view(
    request: Request,
    collection_id: str,
    scenario_id: str | None = Query(default=None),
    page: int = Query(default=0, ge=0),
    limit: int = Query(default=1200, ge=1, le=10000),
) -> dict:
    user_id = _user_id_or_401(request)
    collection_uuid = _parse_collection_id(collection_id)
    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        scenario = await _resolve_view_scenario(db, collection_uuid, scenario_id)
        full_graph = await get_full_scenario_graph(
            db,
            scenario.id,
            layer=None,
            projection=GraphProjection.CODE_SYMBOL,
        )
        projection = build_user_flows_projection(full_graph["nodes"], full_graph["edges"])
        graph = _paginate_graph(
            projection["graph"]["nodes"],
            projection["graph"]["edges"],
            page=page,
            limit=limit,
        )
        warnings: list[str] = []
        if projection["summary"]["user_flows"] == 0:
            warnings.append("No USER_FLOW nodes found. Flow synthesis may still be pending.")
        return {
            "collection_id": str(collection_uuid),
            "scenario": _serialize_scenario(scenario),
            "projection": "user_flows",
            "entity_level": "user_flow",
            "summary": projection["summary"],
            "flows": projection["flows"],
            "warnings": warnings,
            "graph": graph,
        }


@router.get("/collections/{collection_id}/views/rebuild-readiness")
async def rebuild_readiness_view(
    request: Request,
    collection_id: str,
    scenario_id: str | None = Query(default=None),
) -> dict:
    user_id = _user_id_or_401(request)
    collection_uuid = _parse_collection_id(collection_id)
    scenario_uuid = _parse_optional_scenario_id(scenario_id)
    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        scenario = await _resolve_view_scenario(db, collection_uuid, scenario_id)
        full_graph = await get_full_scenario_graph(
            db,
            scenario.id,
            layer=None,
            projection=GraphProjection.CODE_SYMBOL,
        )
        readiness = compute_rebuild_readiness(full_graph["nodes"], full_graph["edges"])
        status = await get_collection_twin_status(
            db,
            collection_id=collection_uuid,
            scenario_id=scenario_uuid,
        )
        evidence_handles: list[dict[str, Any]] = []
        for item in readiness.get("critical_inferred_only") or []:
            for evidence_id in item.get("evidence_ids") or []:
                evidence_handles.append(
                    {
                        "kind": "evidence",
                        "ref": str(evidence_id),
                        "node_id": item.get("node_id"),
                    }
                )
        return {
            "collection_id": str(collection_uuid),
            "scenario": _serialize_scenario(scenario),
            "projection": "rebuild_readiness",
            "score": readiness["score"],
            "summary": readiness["summary"],
            "known_gaps": readiness["known_gaps"],
            "critical_inferred_only": readiness["critical_inferred_only"],
            "evidence_handles": evidence_handles[:200],
            "behavioral_layers_status": status.get("behavioral_layers_status"),
            "last_behavioral_materialized_at": status.get("last_behavioral_materialized_at"),
            "deep_warnings": status.get("deep_warnings") or [],
            "scip_status": status.get("scip_status"),
            "scip_projects_by_language": status.get("scip_projects_by_language") or {},
            "scip_failed_projects": status.get("scip_failed_projects") or [],
            "metrics_gate": status.get("metrics_gate") or {},
        }


@router.get("/collections/{collection_id}/views/graphrag")
async def graphrag_view(
    request: Request,
    collection_id: str,
    scenario_id: str | None = Query(default=None),
    include_kinds: str | None = Query(default=None),
    exclude_kinds: str | None = Query(default=None),
    edge_kinds: str | None = Query(default=None),
    community_mode: str | None = Query(default="color"),
    community_id: str | None = Query(default=None),
    page: int = Query(default=0, ge=0),
    limit: int = Query(default=800, ge=1, le=5000),
) -> dict:
    user_id = _user_id_or_401(request)
    collection_uuid = _parse_collection_id(collection_id)
    include_node_kinds = _parse_knowledge_node_kinds(include_kinds)
    exclude_node_kinds = _parse_knowledge_node_kinds(exclude_kinds)
    include_edge_kinds = _parse_knowledge_edge_kinds(edge_kinds)
    if include_node_kinds is None and exclude_node_kinds is None:
        exclude_node_kinds = set(GRAPHRAG_CODE_NOISE_NODE_KINDS)
    if include_edge_kinds is None:
        include_edge_kinds = set(GRAPHRAG_SEMANTIC_EDGE_KINDS)
    resolved_community_mode = _parse_graphrag_community_mode(community_mode)
    resolved_community_id = community_id.strip() if community_id and community_id.strip() else None

    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        scenario = await _resolve_view_scenario(db, collection_uuid, scenario_id)

        community_by_node_id: dict[uuid.UUID, str] = {}
        communities: dict[str, dict[str, Any]] = {}
        if resolved_community_mode != "none" or resolved_community_id:
            community_nodes, community_edges, _graph_source = await _load_community_graph(
                db, collection_uuid
            )
            community_by_node_id, communities = _compute_symbol_communities(
                community_nodes, community_edges
            )

        if resolved_community_id and resolved_community_id not in communities:
            raise HTTPException(status_code=404, detail="Community not found")

        node_query = select(KnowledgeNode).where(KnowledgeNode.collection_id == collection_uuid)
        if include_node_kinds:
            node_query = node_query.where(KnowledgeNode.kind.in_(include_node_kinds))
        if exclude_node_kinds:
            node_query = node_query.where(~KnowledgeNode.kind.in_(exclude_node_kinds))
        if resolved_community_id:
            member_ids = communities[resolved_community_id]["member_node_ids"]
            if not member_ids:
                status = {"status": "unavailable", "reason": "no_knowledge_graph"}
                return {
                    "collection_id": str(collection_uuid),
                    "scenario": _serialize_scenario(scenario),
                    "projection": "graphrag",
                    "entity_level": "knowledge_node",
                    "community_mode": resolved_community_mode,
                    "community_id": resolved_community_id,
                    "status": status,
                    "graph": {
                        "nodes": [],
                        "edges": [],
                        "page": page,
                        "limit": limit,
                        "total_nodes": 0,
                    },
                }
            node_query = node_query.where(KnowledgeNode.id.in_(member_ids))

        total_nodes = (
            await db.execute(select(func.count()).select_from(node_query.subquery()))
        ).scalar_one()

        paged_nodes = (
            (
                await db.execute(
                    node_query.order_by(
                        KnowledgeNode.kind,
                        KnowledgeNode.name,
                        KnowledgeNode.natural_key,
                    )
                    .offset(page * limit)
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
        page_node_ids = {node.id for node in paged_nodes}

        edges: list[KnowledgeEdge] = []
        if page_node_ids:
            edge_query = select(KnowledgeEdge).where(
                KnowledgeEdge.collection_id == collection_uuid,
                KnowledgeEdge.source_node_id.in_(page_node_ids),
                KnowledgeEdge.target_node_id.in_(page_node_ids),
            )
            if include_edge_kinds:
                edge_query = edge_query.where(KnowledgeEdge.kind.in_(include_edge_kinds))
            edges = (await db.execute(edge_query.order_by(KnowledgeEdge.kind))).scalars().all()

        status = {"status": "ready", "reason": "ok"}
        if total_nodes == 0:
            status = {"status": "unavailable", "reason": "no_graphrag_semantic_graph"}

        return {
            "collection_id": str(collection_uuid),
            "scenario": _serialize_scenario(scenario),
            "projection": "graphrag",
            "entity_level": "knowledge_node",
            "community_mode": resolved_community_mode,
            "community_id": resolved_community_id,
            "status": status,
            "graph": {
                "nodes": [
                    _serialize_graphrag_node(
                        node,
                        community_mode=resolved_community_mode,
                        focused_community_id=resolved_community_id
                        if resolved_community_mode == "focus"
                        else None,
                        community_by_node_id=community_by_node_id,
                        communities=communities,
                    )
                    for node in paged_nodes
                ],
                "edges": [
                    {
                        "id": str(edge.id),
                        "source_node_id": str(edge.source_node_id),
                        "target_node_id": str(edge.target_node_id),
                        "kind": edge.kind.value,
                        "meta": edge.meta or {},
                    }
                    for edge in edges
                ],
                "page": page,
                "limit": limit,
                "total_nodes": total_nodes,
            },
        }


@router.get("/collections/{collection_id}/views/graphrag/communities")
async def graphrag_communities_view(
    request: Request,
    collection_id: str,
    scenario_id: str | None = Query(default=None),
    page: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=1000),
) -> dict:
    user_id = _user_id_or_401(request)
    collection_uuid = _parse_collection_id(collection_id)

    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        scenario = await _resolve_view_scenario(db, collection_uuid, scenario_id)

        community_nodes, community_edges, _graph_source = await _load_community_graph(
            db, collection_uuid
        )
        _, communities = _compute_symbol_communities(community_nodes, community_edges)
        ordered = sorted(
            communities.values(), key=lambda item: (-int(item["size"]), str(item["label"]))
        )
        start = page * limit
        end = start + limit
        paged_items = ordered[start:end]

        return {
            "collection_id": str(collection_uuid),
            "scenario": _serialize_scenario(scenario),
            "items": [
                {
                    "id": item["id"],
                    "label": item["label"],
                    "size": item["size"],
                    "cohesion": item["cohesion"],
                    "top_kinds": item["top_kinds"],
                    "sample_nodes": item["sample_nodes"],
                }
                for item in paged_items
            ],
            "page": page,
            "limit": limit,
            "total": len(ordered),
        }


@router.get("/collections/{collection_id}/views/semantic-map")
async def semantic_map_view(
    request: Request,
    collection_id: str,
    scenario_id: str | None = Query(default=None),
    map_mode: str | None = Query(default="code_structure"),
    include_kinds: str | None = Query(default=None),
    exclude_kinds: str | None = Query(default=None),
    edge_kinds: str | None = Query(default=None),
    mixed_cluster_max_dominant_ratio: float = Query(default=0.55, ge=0.0, le=1.0),
    isolated_distance_multiplier: float = Query(default=1.2, ge=0.1, le=10.0),
    semantic_duplication_min_similarity: float | None = Query(default=None, ge=0.0, le=1.0),
    semantic_duplication_max_source_overlap: float | None = Query(default=None, ge=0.0, le=1.0),
    misplaced_min_dominant_ratio: float = Query(default=0.6, ge=0.0, le=1.0),
    page: int = Query(default=0, ge=0),
    limit: int = Query(default=500, ge=1, le=5000),
) -> dict:
    user_id = _user_id_or_401(request)
    collection_uuid = _parse_collection_id(collection_id)
    resolved_map_mode = _parse_semantic_map_mode(map_mode)
    include_node_kinds = _parse_knowledge_node_kinds(include_kinds)
    exclude_node_kinds = _parse_knowledge_node_kinds(exclude_kinds)
    include_edge_kinds = _parse_knowledge_edge_kinds(edge_kinds)

    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        scenario = await _resolve_view_scenario(db, collection_uuid, scenario_id)

        points: list[dict[str, Any]] = []
        warnings: list[str] = []

        if resolved_map_mode == "code_structure":
            points, structural_warnings = await _build_structural_community_points(
                db,
                collection_id=collection_uuid,
                include_node_kinds=include_node_kinds,
                exclude_node_kinds=exclude_node_kinds,
                include_edge_kinds=include_edge_kinds,
                page=page,
                limit=limit,
            )
            warnings.extend(structural_warnings)

        else:
            if (
                include_node_kinds and KnowledgeNodeKind.SEMANTIC_ENTITY not in include_node_kinds
            ) or (exclude_node_kinds and KnowledgeNodeKind.SEMANTIC_ENTITY in exclude_node_kinds):
                points = []
            else:
                community_rows = (
                    (
                        await db.execute(
                            select(KnowledgeCommunity).where(
                                KnowledgeCommunity.collection_id == collection_uuid
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
                if community_rows:
                    selected_level = min(community.level for community in community_rows)
                    selected_communities = [
                        community
                        for community in community_rows
                        if community.level == selected_level
                    ]

                    selected_communities.sort(
                        key=lambda community: (community.title, str(community.id))
                    )
                    if page > 0 or limit > 0:
                        start = page * limit
                        end = start + limit
                        selected_communities = selected_communities[start:end]

                    community_ids = [community.id for community in selected_communities]
                    members_by_community: dict[uuid.UUID, list[KnowledgeNode]] = defaultdict(list)
                    vectors_by_community: dict[uuid.UUID, list[float]] = {}

                    if community_ids:
                        member_rows = await db.execute(
                            select(CommunityMember, KnowledgeNode)
                            .join(KnowledgeNode, CommunityMember.node_id == KnowledgeNode.id)
                            .where(CommunityMember.community_id.in_(community_ids))
                            .order_by(
                                CommunityMember.community_id,
                                CommunityMember.score.desc(),
                                KnowledgeNode.name,
                            )
                        )
                        for member, node in member_rows.all():
                            members_by_community[member.community_id].append(node)

                        embedding_text = literal_column(
                            "knowledge_embeddings.embedding::text"
                        ).label("embedding_text")
                        embedding_rows = await db.execute(
                            select(
                                KnowledgeEmbedding.target_id,
                                KnowledgeEmbedding.updated_at,
                                embedding_text,
                            )
                            .where(
                                KnowledgeEmbedding.collection_id == collection_uuid,
                                KnowledgeEmbedding.target_type == EmbeddingTargetType.COMMUNITY,
                                KnowledgeEmbedding.target_id.in_(community_ids),
                            )
                            .order_by(
                                KnowledgeEmbedding.target_id,
                                KnowledgeEmbedding.updated_at.desc(),
                            )
                        )
                        for target_id, _updated_at, raw_embedding in embedding_rows.all():
                            if target_id in vectors_by_community:
                                continue
                            parsed = _parse_pgvector_text(
                                str(raw_embedding) if raw_embedding is not None else None
                            )
                            if parsed:
                                vectors_by_community[target_id] = parsed

                    for community in selected_communities:
                        member_nodes = members_by_community.get(community.id, [])
                        if not member_nodes:
                            continue
                        profile = _community_profile(member_nodes)
                        dominant_domain = profile["dominant_domain"]
                        dominant_ratio = float(profile["dominant_ratio"])
                        misplaced_nodes: list[dict[str, Any]] = []
                        if dominant_domain and dominant_ratio >= 0.6:
                            for member in member_nodes:
                                member_domain = profile["node_domain_by_id"].get(member.id)
                                if member_domain and member_domain != dominant_domain:
                                    misplaced_nodes.append(
                                        {
                                            "id": str(member.id),
                                            "name": member.name,
                                            "kind": member.kind.value,
                                            "domain": member_domain,
                                        }
                                    )

                        meta = _safe_meta(community.meta)
                        points.append(
                            {
                                "id": str(community.id),
                                "label": community.title or community.natural_key,
                                "x": 0.0,
                                "y": 0.0,
                                "member_count": len(member_nodes),
                                "cohesion": round(float(meta.get("modularity", 0.0)), 4),
                                "top_kinds": profile["top_kinds"],
                                "domain_counts": profile["domain_counts"],
                                "dominant_domain": dominant_domain,
                                "dominant_ratio": dominant_ratio,
                                "summary": community.summary,
                                "anchor_node_id": str(member_nodes[0].id),
                                "sample_nodes": [
                                    {
                                        "id": str(member.id),
                                        "name": member.name,
                                        "kind": member.kind.value,
                                        "natural_key": member.natural_key,
                                    }
                                    for member in member_nodes[:6]
                                ],
                                "member_node_ids": [str(member.id) for member in member_nodes],
                                "vector": vectors_by_community.get(community.id, []),
                                "source_refs": profile["source_refs"],
                                "name_tokens": profile["name_tokens"],
                                "misplaced_nodes": misplaced_nodes,
                            }
                        )

                    _project_vectors(points, "vector")
                    _normalize_xy(points)

                    if points and not any(point["vector"] for point in points):
                        warnings.append(
                            "No community embeddings found. Using fallback projection for semantic mode."
                        )
                else:
                    points, structural_warnings = await _build_structural_community_points(
                        db,
                        collection_id=collection_uuid,
                        include_node_kinds=include_node_kinds,
                        exclude_node_kinds=exclude_node_kinds,
                        include_edge_kinds=include_edge_kinds,
                        page=page,
                        limit=limit,
                    )
                    warnings.append(
                        "No semantic communities found. Falling back to structural communities."
                    )
                    warnings.extend(structural_warnings)

        signals = _build_semantic_map_signals(
            points,
            mode=resolved_map_mode,
            mixed_cluster_max_dominant_ratio=mixed_cluster_max_dominant_ratio,
            isolated_distance_multiplier=isolated_distance_multiplier,
            semantic_duplication_min_similarity=semantic_duplication_min_similarity,
            semantic_duplication_max_source_overlap=semantic_duplication_max_source_overlap,
            misplaced_min_dominant_ratio=misplaced_min_dominant_ratio,
        )

        status = {"status": "ready", "reason": "ok"}
        if not points:
            status = {
                "status": "unavailable",
                "reason": "no_symbol_communities"
                if resolved_map_mode == "code_structure"
                else "no_semantic_communities",
            }
        elif resolved_map_mode == "semantic" and not any(point["vector"] for point in points):
            status = {"status": "ready", "reason": "no_community_embeddings"}

        public_points = [
            {
                "id": point["id"],
                "label": point["label"],
                "x": point["x"],
                "y": point["y"],
                "member_count": point["member_count"],
                "cohesion": point["cohesion"],
                "top_kinds": point["top_kinds"],
                "domain_counts": point["domain_counts"],
                "dominant_domain": point["dominant_domain"],
                "dominant_ratio": point["dominant_ratio"],
                "summary": point["summary"],
                "anchor_node_id": point["anchor_node_id"],
                "sample_nodes": point["sample_nodes"],
                "member_node_ids": point.get("member_node_ids", []),
            }
            for point in points
        ]

        return {
            "collection_id": str(collection_uuid),
            "scenario": _serialize_scenario(scenario),
            "projection": "semantic_map",
            "map_mode": resolved_map_mode,
            "status": status,
            "thresholds": {
                "mixed_cluster_max_dominant_ratio": round(
                    float(mixed_cluster_max_dominant_ratio), 4
                ),
                "isolated_distance_multiplier": round(float(isolated_distance_multiplier), 4),
                "semantic_duplication_min_similarity": round(
                    float(semantic_duplication_min_similarity)
                    if semantic_duplication_min_similarity is not None
                    else (0.86 if resolved_map_mode == "semantic" else 0.35),
                    4,
                ),
                "semantic_duplication_max_source_overlap": round(
                    float(semantic_duplication_max_source_overlap)
                    if semantic_duplication_max_source_overlap is not None
                    else (0.30 if resolved_map_mode == "semantic" else 0.35),
                    4,
                ),
                "misplaced_min_dominant_ratio": round(float(misplaced_min_dominant_ratio), 4),
            },
            "summary": {
                "points": len(public_points),
                "mixed_clusters": len(signals["mixed_clusters"]),
                "isolated_points": len(signals["isolated_points"]),
                "semantic_duplication": len(signals["semantic_duplication"]),
                "misplaced_code": len(signals["misplaced_code"]),
            },
            "warnings": warnings,
            "signals": signals,
            "points": public_points,
        }


@router.get("/collections/{collection_id}/views/graphrag/path")
async def graphrag_path_view(
    request: Request,
    collection_id: str,
    from_node_id: str = Query(min_length=1),
    to_node_id: str = Query(min_length=1),
    scenario_id: str | None = Query(default=None),
    max_hops: int = Query(default=6, ge=1, le=20),
) -> dict:
    user_id = _user_id_or_401(request)
    collection_uuid = _parse_collection_id(collection_id)

    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        scenario = await _resolve_view_scenario(db, collection_uuid, scenario_id)

        from_node = await _resolve_knowledge_node(db, collection_uuid, from_node_id)
        if not from_node:
            raise HTTPException(status_code=404, detail="from_node_id not found")
        to_node = await _resolve_knowledge_node(db, collection_uuid, to_node_id)
        if not to_node:
            raise HTTPException(status_code=404, detail="to_node_id not found")

        context = await graphrag_trace_path(
            session=db,
            from_node_id=from_node.id,
            to_node_id=to_node.id,
            collection_id=collection_uuid,
            max_hops=max_hops,
        )

        status = "found" if context.entities else "not_found"
        if status == "not_found" and max_hops < 20:
            expanded = await graphrag_trace_path(
                session=db,
                from_node_id=from_node.id,
                to_node_id=to_node.id,
                collection_id=collection_uuid,
                max_hops=min(max_hops + 4, 20),
            )
            if expanded.entities:
                status = "truncated"
        path_nodes = [
            {
                "id": str(entity.node_id),
                "natural_key": entity.natural_key,
                "kind": entity.kind,
                "name": entity.name,
                "meta": {},
            }
            for entity in context.entities
        ]
        path_edges = [
            {
                "id": f"path-edge-{index + 1}",
                "source_node_id": edge.source_id,
                "target_node_id": edge.target_id,
                "kind": edge.kind,
                "meta": {
                    "path_order": index + 1,
                },
            }
            for index, edge in enumerate(context.edges)
        ]

        return {
            "collection_id": str(collection_uuid),
            "scenario": _serialize_scenario(scenario),
            "status": status,
            "from_node_id": str(from_node.id),
            "to_node_id": str(to_node.id),
            "max_hops": max_hops,
            "path": {
                "nodes": path_nodes,
                "edges": path_edges,
                "hops": max(len(path_nodes) - 1, 0),
            },
        }


@router.get("/collections/{collection_id}/views/graphrag/processes")
async def graphrag_processes_view(
    request: Request,
    collection_id: str,
    scenario_id: str | None = Query(default=None),
) -> dict:
    user_id = _user_id_or_401(request)
    collection_uuid = _parse_collection_id(collection_id)

    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        scenario = await _resolve_view_scenario(db, collection_uuid, scenario_id)

        process_nodes, graph_edges, _graph_source = await _load_community_graph(db, collection_uuid)
        community_by_node_id, _ = _compute_symbol_communities(process_nodes, graph_edges)
        processes = _detect_processes(process_nodes, graph_edges, community_by_node_id)
        return {
            "collection_id": str(collection_uuid),
            "scenario": _serialize_scenario(scenario),
            "items": [
                {
                    "id": process["id"],
                    "label": process["label"],
                    "process_type": process["process_type"],
                    "step_count": process["step_count"],
                    "community_ids": process["community_ids"],
                    "entry_node_id": process["entry_node_id"],
                    "terminal_node_id": process["terminal_node_id"],
                }
                for process in processes
            ],
            "total": len(processes),
        }


@router.get("/collections/{collection_id}/views/graphrag/processes/{process_id}")
async def graphrag_process_detail_view(
    request: Request,
    collection_id: str,
    process_id: str,
    scenario_id: str | None = Query(default=None),
) -> dict:
    user_id = _user_id_or_401(request)
    collection_uuid = _parse_collection_id(collection_id)

    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        scenario = await _resolve_view_scenario(db, collection_uuid, scenario_id)

        process_nodes, source_edges, _graph_source = await _load_community_graph(
            db, collection_uuid
        )
        community_by_node_id, _ = _compute_symbol_communities(process_nodes, source_edges)
        processes = _detect_processes(process_nodes, source_edges, community_by_node_id)
        process = next((item for item in processes if item["id"] == process_id), None)
        if not process:
            raise HTTPException(status_code=404, detail="Process not found")

        step_node_ids = [uuid.UUID(step["node_id"]) for step in process["steps"]]
        step_node_set = set(step_node_ids)
        process_edges = [
            {
                "id": f"{process['id']}-edge-{index + 1}",
                "source_node_id": process["steps"][index]["node_id"],
                "target_node_id": process["steps"][index + 1]["node_id"],
                "kind": KnowledgeEdgeKind.SYMBOL_CALLS_SYMBOL.value,
                "meta": {
                    "path_order": index + 1,
                },
            }
            for index in range(max(len(process["steps"]) - 1, 0))
        ]
        observed_edges = [
            edge
            for edge in source_edges
            if edge.source_node_id in step_node_set and edge.target_node_id in step_node_set
        ]
        observed_edge_ids = {
            (item["source_node_id"], item["target_node_id"]) for item in process_edges
        }
        for edge in observed_edges:
            pair = (str(edge.source_node_id), str(edge.target_node_id))
            if pair in observed_edge_ids:
                continue
            process_edges.append(
                {
                    "id": str(edge.id),
                    "source_node_id": str(edge.source_node_id),
                    "target_node_id": str(edge.target_node_id),
                    "kind": edge.kind.value,
                    "meta": edge.meta or {},
                }
            )

        return {
            "collection_id": str(collection_uuid),
            "scenario": _serialize_scenario(scenario),
            "process": {
                "id": process["id"],
                "label": process["label"],
                "process_type": process["process_type"],
                "step_count": process["step_count"],
                "community_ids": process["community_ids"],
                "entry_node_id": process["entry_node_id"],
                "terminal_node_id": process["terminal_node_id"],
            },
            "steps": process["steps"],
            "edges": process_edges,
        }


@router.get("/collections/{collection_id}/views/graphrag/evidence")
async def graphrag_evidence_view(
    request: Request,
    collection_id: str,
    node_id: str = Query(min_length=1),
    scenario_id: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
) -> dict:
    user_id = _user_id_or_401(request)
    collection_uuid = _parse_collection_id(collection_id)

    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        await _resolve_view_scenario(db, collection_uuid, scenario_id)

        node: KnowledgeNode | None = None
        try:
            node_uuid = uuid.UUID(node_id)
        except ValueError:
            node_uuid = None

        if node_uuid:
            node = (
                await db.execute(
                    select(KnowledgeNode).where(
                        KnowledgeNode.collection_id == collection_uuid,
                        KnowledgeNode.id == node_uuid,
                    )
                )
            ).scalar_one_or_none()

        if not node:
            node = (
                (
                    await db.execute(
                        select(KnowledgeNode)
                        .where(
                            KnowledgeNode.collection_id == collection_uuid,
                            KnowledgeNode.natural_key == node_id,
                        )
                        .limit(1)
                    )
                )
                .scalars()
                .first()
            )

        if not node:
            raise HTTPException(status_code=404, detail="Knowledge node not found")

        total = (
            await db.execute(
                select(func.count())
                .select_from(KnowledgeNodeEvidence)
                .where(KnowledgeNodeEvidence.node_id == node.id)
            )
        ).scalar_one()

        evidences = (
            (
                await db.execute(
                    select(KnowledgeEvidence)
                    .join(
                        KnowledgeNodeEvidence,
                        KnowledgeNodeEvidence.evidence_id == KnowledgeEvidence.id,
                    )
                    .where(KnowledgeNodeEvidence.node_id == node.id)
                    .order_by(KnowledgeEvidence.created_at.desc())
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )

        document_ids = {evidence.document_id for evidence in evidences if evidence.document_id}
        document_by_id: dict[uuid.UUID, Document] = {}
        if document_ids:
            docs = (
                (await db.execute(select(Document).where(Document.id.in_(document_ids))))
                .scalars()
                .all()
            )
            document_by_id = {doc.id: doc for doc in docs}

        document_by_path: dict[str, Document | None] = {}
        items: list[dict] = []
        for evidence in evidences:
            text = ""
            text_source = "unavailable"

            if evidence.snippet and evidence.snippet.strip():
                text = _truncate_text(evidence.snippet.strip())
                text_source = "snippet"
            else:
                document = (
                    document_by_id.get(evidence.document_id) if evidence.document_id else None
                )
                if not document and evidence.file_path:
                    cache_key = evidence.file_path.strip().lower()
                    if cache_key in document_by_path:
                        document = document_by_path[cache_key]
                    else:
                        escaped_path = _escape_like_pattern(evidence.file_path)
                        document = (
                            await db.execute(
                                select(Document)
                                .where(Document.uri.ilike(f"%{escaped_path}%", escape="\\"))
                                .order_by(Document.updated_at.desc())
                                .limit(1)
                            )
                        ).scalar_one_or_none()
                        document_by_path[cache_key] = document

                if document:
                    extracted = _extract_document_lines(
                        document.content_markdown,
                        evidence.start_line,
                        evidence.end_line,
                    )
                    if extracted:
                        text = extracted
                        text_source = "document_lines"

            items.append(
                {
                    "evidence_id": str(evidence.id),
                    "file_path": evidence.file_path,
                    "start_line": evidence.start_line,
                    "end_line": evidence.end_line,
                    "text": text,
                    "text_source": text_source,
                }
            )

        return {
            "collection_id": str(collection_uuid),
            "node_id": str(node.id),
            "node_name": node.name,
            "node_kind": node.kind.value,
            "items": items,
            "total": total,
        }


@router.get("/collections/{collection_id}/views/city")
async def city_view(
    request: Request,
    collection_id: str,
    scenario_id: str | None = Query(default=None),
    hotspots_limit: int = Query(default=50, ge=1, le=200),
) -> dict:
    user_id = _user_id_or_401(request)
    collection_uuid = _parse_collection_id(collection_id)

    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        scenario = await _resolve_view_scenario(db, collection_uuid, scenario_id)

        metrics = (
            (
                await db.execute(
                    select(MetricSnapshot).where(MetricSnapshot.scenario_id == scenario.id)
                )
            )
            .scalars()
            .all()
        )
        if not metrics:
            await refresh_metric_snapshots(db, scenario.id)
            await db.flush()
            metrics = (
                (
                    await db.execute(
                        select(MetricSnapshot).where(MetricSnapshot.scenario_id == scenario.id)
                    )
                )
                .scalars()
                .all()
            )

        settings = get_settings()
        metrics_ready = len(metrics) > 0
        metrics_reason = "ok" if metrics_ready else "no_real_metrics"
        if not metrics_ready:
            sources = (
                (await db.execute(select(Source).where(Source.collection_id == collection_uuid)))
                .scalars()
                .all()
            )
            github_source_ids = [
                source.id for source in sources if source.type == SourceType.GITHUB
            ]

            if github_source_ids:
                jobs = (
                    (
                        await db.execute(
                            select(CoverageIngestJob)
                            .where(CoverageIngestJob.source_id.in_(github_source_ids))
                            .order_by(CoverageIngestJob.created_at.desc())
                        )
                    )
                    .scalars()
                    .all()
                )
                has_pending = any(job.status in {"queued", "processing"} for job in jobs)
                latest_failed = next(
                    (job for job in jobs if job.status in {"failed", "rejected"}),
                    None,
                )
                if has_pending:
                    metrics_reason = "awaiting_ci_coverage"
                elif latest_failed is not None:
                    metrics_reason = "coverage_ingest_failed"
                else:
                    file_nodes = (
                        (
                            await db.execute(
                                select(TwinNode).where(
                                    TwinNode.scenario_id == scenario.id,
                                    TwinNode.kind == "file",
                                )
                            )
                        )
                        .scalars()
                        .all()
                    )
                    has_structural = any(
                        bool((node.meta or {}).get("metrics_structural_ready"))
                        for node in file_nodes
                    )
                    if has_structural:
                        metrics_reason = "awaiting_ci_coverage"

        metrics_status = {
            "status": "ready" if metrics_ready else "unavailable",
            "reason": metrics_reason,
            "strict_mode": bool(settings.metrics_strict_mode),
        }

        sorted_hotspots = sorted(
            metrics,
            key=lambda metric: (
                float(metric.complexity or 0.0),
                float(metric.coupling or 0.0),
                -float(metric.coverage or 0.0),
            ),
            reverse=True,
        )
        hotspots = sorted_hotspots[:hotspots_limit] if metrics_ready else []

        total = len(metrics)
        coverage_avg = (
            sum(float(metric.coverage or 0.0) for metric in metrics) / total
            if metrics_ready and total
            else None
        )
        complexity_avg = (
            sum(float(metric.complexity or 0.0) for metric in metrics) / total
            if metrics_ready and total
            else None
        )
        coupling_avg = (
            sum(float(metric.coupling or 0.0) for metric in metrics) / total
            if metrics_ready and total
            else None
        )
        cohesion_avg = (
            sum(float(metric.cohesion or 0.0) for metric in metrics) / total
            if metrics_ready and total
            else None
        )
        instability_avg = (
            sum(float(metric.instability or 0.0) for metric in metrics) / total
            if metrics_ready and total
            else None
        )
        duplication_ratio_avg = (
            sum(float(metric.duplication_ratio or 0.0) for metric in metrics) / total
            if metrics_ready and total
            else None
        )
        crap_score_avg = (
            sum(float(metric.crap_score or 0.0) for metric in metrics) / total
            if metrics_ready and total
            else None
        )
        fan_in_avg = (
            sum(float(metric.fan_in or 0.0) for metric in metrics) / total
            if metrics_ready and total
            else None
        )
        fan_out_avg = (
            sum(float(metric.fan_out or 0.0) for metric in metrics) / total
            if metrics_ready and total
            else None
        )
        cycle_participation_ratio = (
            sum(1.0 for metric in metrics if bool(metric.cycle_participation)) / total
            if metrics_ready and total
            else None
        )
        cycle_size_avg = (
            sum(float(metric.cycle_size or 0.0) for metric in metrics) / total
            if metrics_ready and total
            else None
        )
        change_frequency_avg = (
            sum(float(metric.change_frequency or 0.0) for metric in metrics) / total
            if metrics_ready and total
            else None
        )
        churn_avg = (
            sum(float((metric.meta or {}).get("churn", 0.0) or 0.0) for metric in metrics) / total
            if metrics_ready and total
            else None
        )

        cc_json_payload = json.loads(
            await export_codecharta_json(
                db,
                scenario.id,
                projection=GraphProjection.ARCHITECTURE,
                entity_level="container",
            )
        )

        return {
            "collection_id": str(collection_uuid),
            "scenario": _serialize_scenario(scenario),
            "summary": {
                "metric_nodes": total,
                "coverage_avg": coverage_avg,
                "complexity_avg": complexity_avg,
                "coupling_avg": coupling_avg,
                "cohesion_avg": cohesion_avg,
                "instability_avg": instability_avg,
                "duplication_ratio_avg": duplication_ratio_avg,
                "crap_score_avg": crap_score_avg,
                "fan_in_avg": fan_in_avg,
                "fan_out_avg": fan_out_avg,
                "cycle_participation_ratio": cycle_participation_ratio,
                "cycle_size_avg": cycle_size_avg,
                "change_frequency_avg": change_frequency_avg,
                "churn_avg": churn_avg,
            },
            "metrics_status": metrics_status,
            "hotspots": [
                {
                    "node_natural_key": metric.node_natural_key,
                    "loc": metric.loc,
                    "symbol_count": metric.symbol_count,
                    "coverage": metric.coverage,
                    "complexity": metric.complexity,
                    "coupling": metric.coupling,
                    "cohesion": float(metric.cohesion or 0.0),
                    "instability": float(metric.instability or 0.0),
                    "fan_in": int(metric.fan_in or 0),
                    "fan_out": int(metric.fan_out or 0),
                    "cycle_participation": bool(metric.cycle_participation),
                    "cycle_size": int(metric.cycle_size or 0),
                    "duplication_ratio": float(metric.duplication_ratio or 0.0),
                    "crap_score": (
                        float(metric.crap_score) if metric.crap_score is not None else None
                    ),
                    "change_frequency": metric.change_frequency,
                    "churn": float((metric.meta or {}).get("churn", 0.0) or 0.0),
                }
                for metric in hotspots
            ],
            "cc_json": cc_json_payload,
        }


@router.get("/collections/{collection_id}/views/evolution/investment-utilization")
async def evolution_investment_utilization_view(
    request: Request,
    collection_id: str,
    scenario_id: str | None = Query(default=None),
    entity_level: str = Query(default="container"),
    window_days: int = Query(default=DEFAULT_EVOLUTION_WINDOW_DAYS, ge=1, le=3650),
) -> dict:
    _ensure_evolution_enabled()
    user_id = _user_id_or_401(request)
    collection_uuid = _parse_collection_id(collection_id)

    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        scenario = await _resolve_view_scenario(db, collection_uuid, scenario_id)
        try:
            payload = await get_investment_utilization_payload(
                db,
                scenario_id=scenario.id,
                entity_level=entity_level.strip().lower(),
                window_days=window_days,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        return {
            "collection_id": str(collection_uuid),
            "scenario": _serialize_scenario(scenario),
            "entity_level": entity_level.strip().lower(),
            **payload,
        }


@router.get("/collections/{collection_id}/views/evolution/knowledge-islands")
async def evolution_knowledge_islands_view(
    request: Request,
    collection_id: str,
    scenario_id: str | None = Query(default=None),
    entity_level: str = Query(default="container"),
    window_days: int = Query(default=DEFAULT_EVOLUTION_WINDOW_DAYS, ge=1, le=3650),
    ownership_threshold: float = Query(default=0.7, ge=0.0, le=1.0),
) -> dict:
    _ensure_evolution_enabled()
    user_id = _user_id_or_401(request)
    collection_uuid = _parse_collection_id(collection_id)

    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        scenario = await _resolve_view_scenario(db, collection_uuid, scenario_id)
        try:
            payload = await get_knowledge_islands_payload(
                db,
                scenario_id=scenario.id,
                entity_level=entity_level.strip().lower(),
                window_days=window_days,
                ownership_threshold=ownership_threshold,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        return {
            "collection_id": str(collection_uuid),
            "scenario": _serialize_scenario(scenario),
            "entity_level": entity_level.strip().lower(),
            "ownership_threshold": ownership_threshold,
            **payload,
        }


@router.get("/collections/{collection_id}/views/evolution/temporal-coupling")
async def evolution_temporal_coupling_view(
    request: Request,
    collection_id: str,
    scenario_id: str | None = Query(default=None),
    entity_level: str = Query(default="component"),
    window_days: int = Query(default=DEFAULT_EVOLUTION_WINDOW_DAYS, ge=1, le=3650),
    min_jaccard: float = Query(default=DEFAULT_MIN_JACCARD, ge=0.0, le=1.0),
    max_edges: int = Query(default=DEFAULT_MAX_COUPLING_EDGES, ge=1, le=2000),
) -> dict:
    _ensure_evolution_enabled()
    user_id = _user_id_or_401(request)
    collection_uuid = _parse_collection_id(collection_id)

    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        scenario = await _resolve_view_scenario(db, collection_uuid, scenario_id)
        try:
            payload = await get_temporal_coupling_payload(
                db,
                scenario_id=scenario.id,
                entity_level=entity_level.strip().lower(),
                window_days=window_days,
                min_jaccard=min_jaccard,
                max_edges=max_edges,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        return {
            "collection_id": str(collection_uuid),
            "scenario": _serialize_scenario(scenario),
            "entity_level": entity_level.strip().lower(),
            "min_jaccard": min_jaccard,
            "max_edges": max_edges,
            **payload,
        }


@router.get("/collections/{collection_id}/views/evolution/fitness-functions")
async def evolution_fitness_functions_view(
    request: Request,
    collection_id: str,
    scenario_id: str | None = Query(default=None),
    window_days: int = Query(default=DEFAULT_EVOLUTION_WINDOW_DAYS, ge=1, le=3650),
    include_resolved: bool = Query(default=False),
) -> dict:
    _ensure_evolution_enabled()
    user_id = _user_id_or_401(request)
    collection_uuid = _parse_collection_id(collection_id)

    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        scenario = await _resolve_view_scenario(db, collection_uuid, scenario_id)
        payload = await get_fitness_functions_payload(
            db,
            scenario_id=scenario.id,
            window_days=window_days,
            include_resolved=include_resolved,
        )
        return {
            "collection_id": str(collection_uuid),
            "scenario": _serialize_scenario(scenario),
            "include_resolved": include_resolved,
            **payload,
        }


@router.get("/collections/{collection_id}/views/mermaid")
async def mermaid_view(
    request: Request,
    collection_id: str,
    scenario_id: str | None = Query(default=None),
    compare_with_base: bool = Query(default=True),
    c4_view: str | None = Query(default="container"),
    c4_scope: str | None = Query(default=None),
    max_nodes: int = Query(default=120, ge=10, le=5000),
) -> dict:
    user_id = _user_id_or_401(request)
    collection_uuid = _parse_collection_id(collection_id)
    selected_c4_view = _parse_c4_view(c4_view)
    selected_c4_scope = _parse_c4_scope(c4_scope)

    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        scenario = await _resolve_view_scenario(db, collection_uuid, scenario_id)

        if compare_with_base and scenario.base_scenario_id:
            as_is, to_be = await export_mermaid_asis_tobe_result(
                db,
                as_is_scenario_id=scenario.base_scenario_id,
                to_be_scenario_id=scenario.id,
                c4_view=selected_c4_view,
                c4_scope=selected_c4_scope,
                max_nodes=max_nodes,
            )
            return {
                "collection_id": str(collection_uuid),
                "scenario": _serialize_scenario(scenario),
                "mode": "compare",
                "c4_view": selected_c4_view,
                "c4_scope": selected_c4_scope,
                "max_nodes": max_nodes,
                "as_is_scenario_id": str(scenario.base_scenario_id),
                "as_is": as_is.content,
                "to_be": to_be.content,
                "as_is_warnings": as_is.warnings,
                "to_be_warnings": to_be.warnings,
                "warnings": sorted(set([*as_is.warnings, *to_be.warnings])),
            }

        result = await export_mermaid_c4_result(
            db,
            scenario.id,
            entity_level="container",
            c4_view=selected_c4_view,
            c4_scope=selected_c4_scope,
            max_nodes=max_nodes,
        )
        return {
            "collection_id": str(collection_uuid),
            "scenario": _serialize_scenario(scenario),
            "mode": "single",
            "c4_view": selected_c4_view,
            "c4_scope": selected_c4_scope,
            "max_nodes": max_nodes,
            "warnings": result.warnings,
            "content": result.content,
        }


@router.post("/scenarios/{scenario_id}/cypher")
async def query_cypher(request: Request, scenario_id: str, body: CypherRequest) -> dict:
    user_id = _user_id_or_401(request)
    async with get_db_session() as db:
        scenario = await _load_scenario(db, scenario_id)
        await _ensure_member(db, scenario.collection_id, user_id)
        await sync_scenario_to_age(db, scenario.id)
        try:
            rows = await run_read_only_cypher(db, scenario.id, body.query)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"rows": rows, "count": len(rows)}


@router.post("/scenarios/{scenario_id}/exports")
async def create_export(request: Request, scenario_id: str, body: ExportRequest) -> dict:
    user_id = _user_id_or_401(request)

    async with get_db_session() as db:
        scenario = await _load_scenario(db, scenario_id)
        await _ensure_member(db, scenario.collection_id, user_id)
        projection = (
            GraphProjection(body.projection) if body.projection else GraphProjection.ARCHITECTURE
        )

        if body.format == "lpg_jsonl":
            if projection == GraphProjection.CODE_SYMBOL:
                content = await export_lpg_jsonl(db, scenario.id)
            else:
                graph = await get_full_scenario_graph(
                    session=db,
                    scenario_id=scenario.id,
                    layer=None,
                    projection=projection,
                    entity_level=body.entity_level,
                )
                content = export_lpg_jsonl_from_graph(scenario.id, graph)
            kind = KnowledgeArtifactKind.LPG_JSONL
            name = f"{scenario.name}.lpg.jsonl"
        elif body.format == "cc_json":
            if projection == GraphProjection.CODE_SYMBOL:
                projection = GraphProjection.CODE_FILE
            content = await export_codecharta_json(
                db,
                scenario.id,
                projection=projection,
                entity_level=body.entity_level,
            )
            kind = KnowledgeArtifactKind.CC_JSON
            name = f"{scenario.name}.cc.json"
        elif body.format == "cx2":
            if projection == GraphProjection.CODE_SYMBOL:
                content = await export_cx2(db, scenario.id)
            else:
                graph = await get_full_scenario_graph(
                    session=db,
                    scenario_id=scenario.id,
                    layer=None,
                    projection=projection,
                    entity_level=body.entity_level,
                )
                content = export_cx2_from_graph(scenario.id, graph)
            kind = KnowledgeArtifactKind.CX2
            name = f"{scenario.name}.cx2.json"
        elif body.format == "jgf":
            if projection == GraphProjection.CODE_SYMBOL:
                content = await export_jgf(db, scenario.id)
            else:
                graph = await get_full_scenario_graph(
                    session=db,
                    scenario_id=scenario.id,
                    layer=None,
                    projection=projection,
                    entity_level=body.entity_level,
                )
                content = export_jgf_from_graph(scenario.id, graph)
            kind = KnowledgeArtifactKind.JGF
            name = f"{scenario.name}.jgf.json"
        elif body.format == "twin_manifest":
            content = await export_twin_manifest(db, scenario.id)
            kind = KnowledgeArtifactKind.TWIN_MANIFEST
            name = f"{scenario.name}.twin_manifest.json"
        else:
            selected_c4_view = _parse_c4_view(body.c4_view)
            selected_c4_scope = _parse_c4_scope(body.c4_scope)
            selected_max_nodes = body.max_nodes or 120
            if scenario.base_scenario_id:
                as_is_content, to_be_content = await export_mermaid_asis_tobe(
                    db,
                    as_is_scenario_id=scenario.base_scenario_id,
                    to_be_scenario_id=scenario.id,
                    c4_view=selected_c4_view,
                    c4_scope=selected_c4_scope,
                    max_nodes=selected_max_nodes,
                )
                as_is_artifact = await _upsert_artifact(
                    db,
                    collection_id=scenario.collection_id,
                    kind=KnowledgeArtifactKind.MERMAID_C4_ASIS,
                    name=f"{scenario.name}.asis.mmd",
                    content=as_is_content,
                    meta={
                        "scenario_id": str(scenario.base_scenario_id),
                        "c4_view": selected_c4_view,
                        "c4_scope": selected_c4_scope,
                        "max_nodes": selected_max_nodes,
                    },
                )
                to_be_artifact = await _upsert_artifact(
                    db,
                    collection_id=scenario.collection_id,
                    kind=KnowledgeArtifactKind.MERMAID_C4_TOBE,
                    name=f"{scenario.name}.tobe.mmd",
                    content=to_be_content,
                    meta={
                        "scenario_id": str(scenario.id),
                        "c4_view": selected_c4_view,
                        "c4_scope": selected_c4_scope,
                        "max_nodes": selected_max_nodes,
                    },
                )
                await db.commit()
                return {
                    "exports": [
                        {"id": str(as_is_artifact.id), "name": as_is_artifact.name},
                        {"id": str(to_be_artifact.id), "name": to_be_artifact.name},
                    ]
                }

            content = await export_mermaid_c4(
                db,
                scenario.id,
                entity_level=body.entity_level or "container",
                c4_view=selected_c4_view,
                c4_scope=selected_c4_scope,
                max_nodes=selected_max_nodes,
            )
            kind = (
                KnowledgeArtifactKind.MERMAID_C4_ASIS
                if scenario.is_as_is
                else KnowledgeArtifactKind.MERMAID_C4_TOBE
            )
            name = f"{scenario.name}.mmd"

        meta: dict = {
            "scenario_id": str(scenario.id),
            "format": body.format,
            "projection": projection.value,
            "entity_level": body.entity_level,
        }
        if body.format == "mermaid_c4":
            meta["c4_view"] = selected_c4_view
            meta["c4_scope"] = selected_c4_scope
            meta["max_nodes"] = selected_max_nodes
        artifact = await _upsert_artifact(
            db,
            collection_id=scenario.collection_id,
            kind=kind,
            name=name,
            content=content,
            meta=meta,
        )
        await db.commit()

        return {
            "id": str(artifact.id),
            "name": artifact.name,
            "kind": artifact.kind.value,
            "format": body.format,
        }


@router.get("/scenarios/{scenario_id}/exports/{export_id}")
async def get_export(request: Request, scenario_id: str, export_id: str) -> dict:
    user_id = _user_id_or_401(request)

    try:
        export_uuid = uuid.UUID(export_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid export_id") from e

    async with get_db_session() as db:
        scenario = await _load_scenario(db, scenario_id)
        await _ensure_member(db, scenario.collection_id, user_id)

        artifact = (
            await db.execute(
                select(KnowledgeArtifact).where(
                    KnowledgeArtifact.id == export_uuid,
                    KnowledgeArtifact.collection_id == scenario.collection_id,
                )
            )
        ).scalar_one_or_none()
        if not artifact:
            raise HTTPException(status_code=404, detail="Export not found")

        return {
            "id": str(artifact.id),
            "name": artifact.name,
            "kind": artifact.kind.value,
            "content": artifact.content,
            "meta": artifact.meta,
            "updated_at": artifact.updated_at,
        }


@router.get("/scenarios/{scenario_id}/exports/{export_id}/raw")
async def get_export_raw(request: Request, scenario_id: str, export_id: str) -> Response:
    user_id = _user_id_or_401(request)

    try:
        export_uuid = uuid.UUID(export_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid export_id") from e

    async with get_db_session() as db:
        scenario = await _load_scenario(db, scenario_id)
        await _ensure_member(db, scenario.collection_id, user_id)

        artifact = (
            await db.execute(
                select(KnowledgeArtifact).where(
                    KnowledgeArtifact.id == export_uuid,
                    KnowledgeArtifact.collection_id == scenario.collection_id,
                )
            )
        ).scalar_one_or_none()
        if not artifact:
            raise HTTPException(status_code=404, detail="Export not found")

        return Response(content=artifact.content, media_type="application/json")
