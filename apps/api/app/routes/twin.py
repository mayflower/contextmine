"""Digital twin and architecture intent routes."""

from __future__ import annotations

import json
import uuid
from typing import Literal

from contextmine_core import Collection, CollectionMember, get_settings
from contextmine_core import get_session as get_db_session
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
    export_mermaid_c4,
)
from contextmine_core.graph.age import run_read_only_cypher, sync_scenario_to_age
from contextmine_core.models import (
    CoverageIngestJob,
    Document,
    KnowledgeArtifact,
    KnowledgeArtifactKind,
    KnowledgeEdge,
    KnowledgeEdgeKind,
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
    GraphProjection,
    approve_and_execute_intent,
    create_to_be_scenario,
    get_full_scenario_graph,
    get_or_create_as_is_scenario,
    get_scenario_graph,
    list_scenario_patches,
    refresh_metric_snapshots,
    submit_intent,
)
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from app.middleware import get_session

router = APIRouter(prefix="/twin", tags=["twin"])


class CreateScenarioRequest(BaseModel):
    collection_id: str
    name: str = Field(min_length=1, max_length=255)


class CypherRequest(BaseModel):
    query: str = Field(min_length=1)


class ExportRequest(BaseModel):
    format: Literal["lpg_jsonl", "cc_json", "cx2", "jgf", "mermaid_c4"]
    projection: Literal["architecture", "code_file", "code_symbol"] | None = None
    entity_level: Literal["domain", "container", "component", "file", "symbol"] | None = None


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


def _parse_kind_filter(raw_value: str | None) -> set[str] | None:
    if not raw_value:
        return None
    values = {value.strip().lower() for value in raw_value.split(",") if value.strip()}
    return values or None


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
        edge_kinds = {"symbol_calls_symbol"}
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


@router.get("/collections/{collection_id}/views/graphrag")
async def graphrag_view(
    request: Request,
    collection_id: str,
    scenario_id: str | None = Query(default=None),
    include_kinds: str | None = Query(default=None),
    exclude_kinds: str | None = Query(default=None),
    edge_kinds: str | None = Query(default=None),
    page: int = Query(default=0, ge=0),
    limit: int = Query(default=800, ge=1, le=5000),
) -> dict:
    user_id = _user_id_or_401(request)
    collection_uuid = _parse_collection_id(collection_id)
    include_node_kinds = _parse_knowledge_node_kinds(include_kinds)
    exclude_node_kinds = _parse_knowledge_node_kinds(exclude_kinds)
    include_edge_kinds = _parse_knowledge_edge_kinds(edge_kinds)

    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        scenario = await _resolve_view_scenario(db, collection_uuid, scenario_id)

        node_query = select(KnowledgeNode).where(KnowledgeNode.collection_id == collection_uuid)
        if include_node_kinds:
            node_query = node_query.where(KnowledgeNode.kind.in_(include_node_kinds))
        if exclude_node_kinds:
            node_query = node_query.where(~KnowledgeNode.kind.in_(exclude_node_kinds))

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
            status = {"status": "unavailable", "reason": "no_knowledge_graph"}

        return {
            "collection_id": str(collection_uuid),
            "scenario": _serialize_scenario(scenario),
            "projection": "graphrag",
            "entity_level": "knowledge_node",
            "status": status,
            "graph": {
                "nodes": [
                    {
                        "id": str(node.id),
                        "natural_key": node.natural_key,
                        "kind": node.kind.value,
                        "name": node.name,
                        "meta": node.meta or {},
                    }
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
                }
                for metric in hotspots
            ],
            "cc_json": cc_json_payload,
        }


@router.get("/collections/{collection_id}/views/mermaid")
async def mermaid_view(
    request: Request,
    collection_id: str,
    scenario_id: str | None = Query(default=None),
    compare_with_base: bool = Query(default=True),
) -> dict:
    user_id = _user_id_or_401(request)
    collection_uuid = _parse_collection_id(collection_id)

    async with get_db_session() as db:
        await _ensure_member(db, collection_uuid, user_id)
        scenario = await _resolve_view_scenario(db, collection_uuid, scenario_id)

        if compare_with_base and scenario.base_scenario_id:
            as_is, to_be = await export_mermaid_asis_tobe(
                db,
                as_is_scenario_id=scenario.base_scenario_id,
                to_be_scenario_id=scenario.id,
            )
            return {
                "collection_id": str(collection_uuid),
                "scenario": _serialize_scenario(scenario),
                "mode": "compare",
                "as_is_scenario_id": str(scenario.base_scenario_id),
                "as_is": as_is,
                "to_be": to_be,
            }

        content = await export_mermaid_c4(db, scenario.id, entity_level="container")
        return {
            "collection_id": str(collection_uuid),
            "scenario": _serialize_scenario(scenario),
            "mode": "single",
            "content": content,
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
        else:
            if scenario.base_scenario_id:
                as_is_content, to_be_content = await export_mermaid_asis_tobe(
                    db,
                    as_is_scenario_id=scenario.base_scenario_id,
                    to_be_scenario_id=scenario.id,
                )
                as_is_artifact = await _upsert_artifact(
                    db,
                    collection_id=scenario.collection_id,
                    kind=KnowledgeArtifactKind.MERMAID_C4_ASIS,
                    name=f"{scenario.name}.asis.mmd",
                    content=as_is_content,
                    meta={"scenario_id": str(scenario.base_scenario_id)},
                )
                to_be_artifact = await _upsert_artifact(
                    db,
                    collection_id=scenario.collection_id,
                    kind=KnowledgeArtifactKind.MERMAID_C4_TOBE,
                    name=f"{scenario.name}.tobe.mmd",
                    content=to_be_content,
                    meta={"scenario_id": str(scenario.id)},
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
            )
            kind = (
                KnowledgeArtifactKind.MERMAID_C4_ASIS
                if scenario.is_as_is
                else KnowledgeArtifactKind.MERMAID_C4_TOBE
            )
            name = f"{scenario.name}.mmd"

        artifact = await _upsert_artifact(
            db,
            collection_id=scenario.collection_id,
            kind=kind,
            name=name,
            content=content,
            meta={
                "scenario_id": str(scenario.id),
                "format": body.format,
                "projection": projection.value,
                "entity_level": body.entity_level,
            },
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
