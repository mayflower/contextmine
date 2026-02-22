"""Digital twin service layer.

This module owns scenario lifecycle, snapshot ingestion, intent execution,
patch history, and graph retrieval for the twin APIs.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from contextmine_core.architecture_intents import (
    ArchitectureIntentV1,
    IntentRisk,
    classify_risk,
    compile_intent_patch,
)
from contextmine_core.models import (
    ArchitectureIntent,
    ArchitectureIntentAction,
    ArchitectureIntentRun,
    ArchitectureIntentStatus,
    IntentRiskLevel,
    KnowledgeEdge,
    KnowledgeNode,
    MetricSnapshot,
    TwinEdge,
    TwinEdgeLayer,
    TwinLayer,
    TwinNode,
    TwinNodeLayer,
    TwinPatch,
    TwinScenario,
)
from contextmine_core.semantic_snapshot.models import RelationKind, Snapshot, SymbolKind
from contextmine_core.twin.projections import (
    GraphProjection,
    build_architecture_projection,
    build_code_file_projection,
    build_code_symbol_projection,
)
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


def _compute_crap_score(
    complexity: float | None,
    coverage: float | None,
) -> float | None:
    """Compute CRAP score when both complexity and coverage are available."""
    if complexity is None or coverage is None:
        return None
    bounded_coverage = min(max(float(coverage), 0.0), 100.0) / 100.0
    cplx = float(complexity)
    return (cplx * cplx * ((1.0 - bounded_coverage) ** 3)) + cplx


def infer_node_layers(kind: str, meta: dict[str, Any] | None = None) -> set[TwinLayer]:
    """Infer layers for a node kind."""
    del meta
    norm = kind.lower()
    if norm in {"file", "module", "symbol", "function", "method", "class", "validator"}:
        return {TwinLayer.CODE_CONTROLFLOW}
    if norm in {"api_endpoint", "interface", "rpc", "service", "component"}:
        return {TwinLayer.COMPONENT_INTERFACE}
    if norm in {"bounded_context", "container", "db_table", "db_column"}:
        return {TwinLayer.DOMAIN_CONTAINER}
    return {TwinLayer.PORTFOLIO_SYSTEM}


def infer_edge_layers(kind: str) -> set[TwinLayer]:
    """Infer layers for edge kinds."""
    k = kind.lower()
    if k == "file_defines_symbol" or k.startswith("symbol_"):
        return {TwinLayer.CODE_CONTROLFLOW}
    if "calls" in k or "references" in k or "contains" in k:
        return {TwinLayer.CODE_CONTROLFLOW}
    if "interface" in k or "endpoint" in k or "rpc" in k:
        return {TwinLayer.COMPONENT_INTERFACE}
    if "context" in k or "domain" in k:
        return {TwinLayer.DOMAIN_CONTAINER}
    return {TwinLayer.PORTFOLIO_SYSTEM}


def _risk_to_model(risk: IntentRisk) -> IntentRiskLevel:
    if risk == IntentRisk.HIGH:
        return IntentRiskLevel.HIGH
    return IntentRiskLevel.LOW


async def get_or_create_as_is_scenario(
    session: AsyncSession,
    collection_id: UUID,
    user_id: UUID | None = None,
) -> TwinScenario:
    """Get AS-IS scenario for a collection, creating one if missing."""
    result = await session.execute(
        select(TwinScenario).where(
            TwinScenario.collection_id == collection_id,
            TwinScenario.is_as_is.is_(True),
        )
    )
    scenario = result.scalar_one_or_none()
    if scenario:
        return scenario

    scenario = TwinScenario(
        id=uuid.uuid4(),
        collection_id=collection_id,
        name="AS-IS",
        is_as_is=True,
        version=1,
        meta={"origin": "knowledge_graph"},
        created_by_user_id=user_id,
    )
    session.add(scenario)
    await session.flush()
    await seed_scenario_from_knowledge_graph(
        session, scenario.id, collection_id, clear_existing=True
    )
    await session.flush()
    return scenario


async def seed_scenario_from_knowledge_graph(
    session: AsyncSession,
    scenario_id: UUID,
    collection_id: UUID,
    clear_existing: bool,
) -> tuple[int, int]:
    """Seed a scenario from existing knowledge graph tables."""
    if clear_existing:
        await session.execute(
            delete(TwinEdgeLayer).where(
                TwinEdgeLayer.edge_id.in_(
                    select(TwinEdge.id).where(TwinEdge.scenario_id == scenario_id)
                )
            )
        )
        await session.execute(
            delete(TwinNodeLayer).where(
                TwinNodeLayer.node_id.in_(
                    select(TwinNode.id).where(TwinNode.scenario_id == scenario_id)
                )
            )
        )
        await session.execute(delete(TwinEdge).where(TwinEdge.scenario_id == scenario_id))
        await session.execute(delete(TwinNode).where(TwinNode.scenario_id == scenario_id))

    node_rows = (
        (
            await session.execute(
                select(KnowledgeNode).where(KnowledgeNode.collection_id == collection_id)
            )
        )
        .scalars()
        .all()
    )

    node_map: dict[UUID, UUID] = {}
    created_nodes = 0

    for node in node_rows:
        twin_node_id = await _upsert_twin_node(
            session=session,
            scenario_id=scenario_id,
            natural_key=node.natural_key,
            kind=node.kind.value,
            name=node.name,
            meta=node.meta or {},
            provenance_node_id=node.id,
        )
        node_map[node.id] = twin_node_id
        created_nodes += 1

    edge_rows = (
        (
            await session.execute(
                select(KnowledgeEdge).where(KnowledgeEdge.collection_id == collection_id)
            )
        )
        .scalars()
        .all()
    )

    created_edges = 0
    for edge in edge_rows:
        src = node_map.get(edge.source_node_id)
        dst = node_map.get(edge.target_node_id)
        if not src or not dst:
            continue
        await _upsert_twin_edge(
            session,
            scenario_id,
            src,
            dst,
            edge.kind.value,
            edge.meta or {},
        )
        created_edges += 1

    return created_nodes, created_edges


async def create_to_be_scenario(
    session: AsyncSession,
    collection_id: UUID,
    name: str,
    user_id: UUID | None,
) -> TwinScenario:
    """Create TO-BE scenario as a branch of AS-IS."""
    as_is = await get_or_create_as_is_scenario(session, collection_id, user_id)

    scenario = TwinScenario(
        id=uuid.uuid4(),
        collection_id=collection_id,
        name=name,
        base_scenario_id=as_is.id,
        is_as_is=False,
        version=as_is.version,
        meta={"base": str(as_is.id)},
        created_by_user_id=user_id,
    )
    session.add(scenario)
    await session.flush()

    await _clone_scenario_graph(session, from_scenario_id=as_is.id, to_scenario_id=scenario.id)
    return scenario


async def _clone_scenario_graph(
    session: AsyncSession,
    from_scenario_id: UUID,
    to_scenario_id: UUID,
) -> None:
    src_nodes = (
        (await session.execute(select(TwinNode).where(TwinNode.scenario_id == from_scenario_id)))
        .scalars()
        .all()
    )

    id_map: dict[UUID, UUID] = {}
    for node in src_nodes:
        new_id = await _upsert_twin_node(
            session=session,
            scenario_id=to_scenario_id,
            natural_key=node.natural_key,
            kind=node.kind,
            name=node.name,
            meta=node.meta or {},
            provenance_node_id=node.provenance_node_id,
        )
        id_map[node.id] = new_id

        src_layers = (
            (
                await session.execute(
                    select(TwinNodeLayer.layer).where(TwinNodeLayer.node_id == node.id)
                )
            )
            .scalars()
            .all()
        )
        for layer in src_layers:
            await _upsert_node_layer(session, new_id, layer)

    src_edges = (
        (await session.execute(select(TwinEdge).where(TwinEdge.scenario_id == from_scenario_id)))
        .scalars()
        .all()
    )
    for edge in src_edges:
        new_src = id_map.get(edge.source_node_id)
        new_dst = id_map.get(edge.target_node_id)
        if not new_src or not new_dst:
            continue

        new_edge_id = await _upsert_twin_edge(
            session=session,
            scenario_id=to_scenario_id,
            source_node_id=new_src,
            target_node_id=new_dst,
            kind=edge.kind,
            meta=edge.meta or {},
        )

        src_layers = (
            (
                await session.execute(
                    select(TwinEdgeLayer.layer).where(TwinEdgeLayer.edge_id == edge.id)
                )
            )
            .scalars()
            .all()
        )
        for layer in src_layers:
            await _upsert_edge_layer(session, new_edge_id, layer)


async def _upsert_twin_node(
    session: AsyncSession,
    scenario_id: UUID,
    natural_key: str,
    kind: str,
    name: str,
    meta: dict[str, Any],
    provenance_node_id: UUID | None,
) -> UUID:
    stmt = pg_insert(TwinNode).values(
        scenario_id=scenario_id,
        natural_key=natural_key,
        kind=kind,
        name=name,
        meta=meta,
        provenance_node_id=provenance_node_id,
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_twin_node_natural",
        set_={
            "kind": stmt.excluded.kind,
            "name": stmt.excluded.name,
            "meta": stmt.excluded.meta,
            "provenance_node_id": stmt.excluded.provenance_node_id,
        },
    ).returning(TwinNode.id)
    node_id = (await session.execute(stmt)).scalar_one()

    for layer in infer_node_layers(kind, meta):
        await _upsert_node_layer(session, node_id, layer)

    return node_id


async def _upsert_twin_edge(
    session: AsyncSession,
    scenario_id: UUID,
    source_node_id: UUID,
    target_node_id: UUID,
    kind: str,
    meta: dict[str, Any],
) -> UUID:
    stmt = pg_insert(TwinEdge).values(
        scenario_id=scenario_id,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        kind=kind,
        meta=meta,
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_twin_edge_unique",
        set_={"meta": stmt.excluded.meta},
    ).returning(TwinEdge.id)
    edge_id = (await session.execute(stmt)).scalar_one()

    for layer in infer_edge_layers(kind):
        await _upsert_edge_layer(session, edge_id, layer)

    return edge_id


async def _upsert_node_layer(session: AsyncSession, node_id: UUID, layer: TwinLayer) -> None:
    stmt = pg_insert(TwinNodeLayer).values(node_id=node_id, layer=layer)
    stmt = stmt.on_conflict_do_nothing(constraint="uq_twin_node_layer")
    await session.execute(stmt)


async def _upsert_edge_layer(session: AsyncSession, edge_id: UUID, layer: TwinLayer) -> None:
    stmt = pg_insert(TwinEdgeLayer).values(edge_id=edge_id, layer=layer)
    stmt = stmt.on_conflict_do_nothing(constraint="uq_twin_edge_layer")
    await session.execute(stmt)


async def ingest_snapshot_into_as_is(
    session: AsyncSession,
    collection_id: UUID,
    snapshot: Snapshot,
    source_id: UUID | None = None,
    user_id: UUID | None = None,
) -> tuple[TwinScenario, dict[str, int]]:
    """Upsert SCIP/LSIF snapshot entities into the AS-IS scenario."""
    scenario = await get_or_create_as_is_scenario(session, collection_id, user_id)

    file_node_ids: dict[str, UUID] = {}
    symbol_node_ids: dict[str, UUID] = {}

    node_count = 0
    edge_count = 0

    for file_info in snapshot.files:
        file_key = f"file:{file_info.path}"
        file_node_ids[file_info.path] = await _upsert_twin_node(
            session,
            scenario.id,
            natural_key=file_key,
            kind="file",
            name=file_info.path,
            meta={
                "language": file_info.language,
                "source_id": str(source_id) if source_id else None,
                **(snapshot.meta or {}),
            },
            provenance_node_id=None,
        )
        node_count += 1

    for symbol in snapshot.symbols:
        if symbol.kind == SymbolKind.UNKNOWN:
            continue
        symbol_key = f"symbol:{symbol.def_id}"
        symbol_node_ids[symbol.def_id] = await _upsert_twin_node(
            session,
            scenario.id,
            natural_key=symbol_key,
            kind=symbol.kind.value,
            name=symbol.name or symbol.def_id,
            meta={
                "file_path": symbol.file_path,
                "def_id": symbol.def_id,
                "symbol_kind": symbol.kind.value,
                "range": symbol.range.to_dict(),
            },
            provenance_node_id=None,
        )
        node_count += 1

        file_node_id = file_node_ids.get(symbol.file_path)
        if file_node_id:
            await _upsert_twin_edge(
                session,
                scenario.id,
                file_node_id,
                symbol_node_ids[symbol.def_id],
                "file_defines_symbol",
                {},
            )
            edge_count += 1

    for relation in snapshot.relations:
        src = symbol_node_ids.get(relation.src_def_id)
        dst = symbol_node_ids.get(relation.dst_def_id)
        if not src or not dst:
            continue

        edge_kind = _relation_to_edge_kind(relation.kind)
        await _upsert_twin_edge(
            session,
            scenario.id,
            src,
            dst,
            edge_kind,
            {"resolved": relation.resolved, "weight": relation.weight, **(relation.meta or {})},
        )
        edge_count += 1

    scenario.version += 1
    scenario.updated_at = datetime.now(UTC)

    return scenario, {"nodes_upserted": node_count, "edges_upserted": edge_count}


def _relation_to_edge_kind(kind: RelationKind) -> str:
    mapping = {
        RelationKind.CONTAINS: "symbol_contains_symbol",
        RelationKind.CALLS: "symbol_calls_symbol",
        RelationKind.REFERENCES: "symbol_references_symbol",
        RelationKind.EXTENDS: "symbol_extends_symbol",
        RelationKind.IMPLEMENTS: "symbol_implements_symbol",
        RelationKind.IMPORTS: "symbol_imports_symbol",
    }
    return mapping.get(kind, "symbol_references_symbol")


async def submit_intent(
    session: AsyncSession,
    scenario: TwinScenario,
    intent: ArchitectureIntentV1,
    requested_by: UUID | None,
    auto_execute: bool = True,
) -> ArchitectureIntent:
    """Persist and optionally execute an intent."""
    if scenario.id != intent.scenario_id:
        raise ValueError("scenario_id mismatch")
    if scenario.version != intent.expected_scenario_version:
        raise ValueError(
            f"version conflict: expected={intent.expected_scenario_version} actual={scenario.version}"
        )

    risk = classify_risk(intent.action)
    patch_ops = compile_intent_patch(intent)

    db_intent = ArchitectureIntent(
        id=intent.intent_id or uuid.uuid4(),
        scenario_id=scenario.id,
        intent_version=intent.intent_version,
        action=ArchitectureIntentAction(intent.action.value.lower()),
        target_type=intent.target.type,
        target_id=intent.target.id,
        params=intent.params,
        expected_scenario_version=intent.expected_scenario_version,
        status=ArchitectureIntentStatus.PENDING,
        risk_level=_risk_to_model(risk),
        requires_approval=risk == IntentRisk.HIGH,
        requested_by_user_id=requested_by,
    )
    session.add(db_intent)
    await session.flush()

    if db_intent.requires_approval:
        db_intent.status = ArchitectureIntentStatus.BLOCKED
        await _record_intent_run(
            session,
            db_intent.id,
            scenario_version_before=scenario.version,
            scenario_version_after=None,
            status="blocked",
            message="High-risk intent requires approval",
            error=None,
        )
        return db_intent

    if auto_execute:
        await execute_intent(session, db_intent, patch_ops)
    return db_intent


async def approve_and_execute_intent(
    session: AsyncSession,
    scenario: TwinScenario,
    intent_id: UUID,
) -> ArchitectureIntent:
    """Approve and execute a previously blocked intent."""
    intent = (
        await session.execute(
            select(ArchitectureIntent).where(
                ArchitectureIntent.id == intent_id,
                ArchitectureIntent.scenario_id == scenario.id,
            )
        )
    ).scalar_one()

    if intent.status != ArchitectureIntentStatus.BLOCKED:
        raise ValueError("intent is not blocked")

    payload = ArchitectureIntentV1(
        intent_version=intent.intent_version,
        scenario_id=intent.scenario_id,
        intent_id=intent.id,
        action=intent.action.value.upper(),
        target={"type": intent.target_type, "id": intent.target_id},
        params=intent.params,
        expected_scenario_version=scenario.version,
        requested_by=intent.requested_by_user_id,
    )
    patch_ops = compile_intent_patch(payload)
    intent.status = ArchitectureIntentStatus.APPROVED
    await execute_intent(session, intent, patch_ops)
    return intent


async def execute_intent(
    session: AsyncSession,
    intent: ArchitectureIntent,
    patch_ops: list[dict[str, Any]],
) -> None:
    """Apply compiled patch operations to a scenario graph."""
    scenario = (
        await session.execute(select(TwinScenario).where(TwinScenario.id == intent.scenario_id))
    ).scalar_one()
    before_version = scenario.version

    try:
        await _apply_patch_ops(session, scenario.id, patch_ops)
        scenario.version += 1
        scenario.updated_at = datetime.now(UTC)

        patch = TwinPatch(
            id=uuid.uuid4(),
            scenario_id=scenario.id,
            scenario_version=scenario.version,
            intent_id=intent.id,
            patch_ops=patch_ops,
            created_by_user_id=intent.requested_by_user_id,
        )
        session.add(patch)

        intent.status = ArchitectureIntentStatus.EXECUTED
        intent.last_error = None
        await _record_intent_run(
            session,
            intent.id,
            before_version,
            scenario.version,
            "executed",
            "Intent executed successfully",
            None,
        )
    except Exception as exc:  # noqa: BLE001
        intent.status = ArchitectureIntentStatus.FAILED
        intent.last_error = str(exc)
        await _record_intent_run(
            session,
            intent.id,
            before_version,
            None,
            "failed",
            "Intent execution failed",
            str(exc),
        )
        raise


async def _record_intent_run(
    session: AsyncSession,
    intent_id: UUID,
    scenario_version_before: int,
    scenario_version_after: int | None,
    status: str,
    message: str | None,
    error: str | None,
) -> None:
    session.add(
        ArchitectureIntentRun(
            id=uuid.uuid4(),
            intent_id=intent_id,
            scenario_version_before=scenario_version_before,
            scenario_version_after=scenario_version_after,
            status=status,
            message=message,
            error=error,
        )
    )


async def _apply_patch_ops(
    session: AsyncSession,
    scenario_id: UUID,
    patch_ops: list[dict[str, Any]],
) -> None:
    for op in patch_ops:
        operation = op.get("op")
        path = op.get("path", "")
        value = op.get("value")

        if operation == "add" and path == "/nodes/-":
            node_dict = value or {}
            await _upsert_twin_node(
                session=session,
                scenario_id=scenario_id,
                natural_key=node_dict["natural_key"],
                kind=node_dict.get("kind", "component"),
                name=node_dict.get("name", node_dict["natural_key"]),
                meta=node_dict.get("meta", {}),
                provenance_node_id=None,
            )
            continue

        if operation == "add" and path == "/edges/-":
            edge_dict = value or {}
            src = await _get_node_id_by_key(session, scenario_id, edge_dict["source_natural_key"])
            dst = await _get_node_id_by_key(session, scenario_id, edge_dict["target_natural_key"])
            if not src or not dst:
                raise ValueError("edge add failed: missing source/target node")

            await _upsert_twin_edge(
                session=session,
                scenario_id=scenario_id,
                source_node_id=src,
                target_node_id=dst,
                kind=edge_dict.get("kind", "relates_to"),
                meta=edge_dict.get("meta", {}),
            )
            continue

        if operation == "replace" and path.startswith("/nodes/by_natural_key/"):
            parts = path.split("/")
            if len(parts) < 6:
                raise ValueError(f"invalid replace path: {path}")
            natural_key = parts[3]
            meta_key = parts[5]
            node = (
                await session.execute(
                    select(TwinNode).where(
                        TwinNode.scenario_id == scenario_id,
                        TwinNode.natural_key == natural_key,
                    )
                )
            ).scalar_one_or_none()
            if not node:
                raise ValueError(f"node not found for patch: {natural_key}")
            node_meta = dict(node.meta or {})
            node_meta[meta_key] = value
            node.meta = node_meta
            node.updated_at = datetime.now(UTC)
            continue

        raise ValueError(f"unsupported patch operation: {json.dumps(op)}")


async def _get_node_id_by_key(
    session: AsyncSession,
    scenario_id: UUID,
    natural_key: str,
) -> UUID | None:
    return (
        await session.execute(
            select(TwinNode.id).where(
                TwinNode.scenario_id == scenario_id,
                TwinNode.natural_key == natural_key,
            )
        )
    ).scalar_one_or_none()


async def list_scenario_patches(session: AsyncSession, scenario_id: UUID) -> list[TwinPatch]:
    """List patch history for a scenario."""
    result = await session.execute(
        select(TwinPatch)
        .where(TwinPatch.scenario_id == scenario_id)
        .order_by(TwinPatch.scenario_version.asc())
    )
    return list(result.scalars().all())


async def get_scenario_graph(
    session: AsyncSession,
    scenario_id: UUID,
    layer: TwinLayer | None,
    page: int,
    limit: int,
    projection: GraphProjection = GraphProjection.CODE_SYMBOL,
    entity_level: str | None = None,
    include_kinds: set[str] | None = None,
    exclude_kinds: set[str] | None = None,
    include_edge_kinds: set[str] | None = None,
) -> dict[str, Any]:
    """Get paginated graph view for one scenario."""
    graph = await get_full_scenario_graph(
        session=session,
        scenario_id=scenario_id,
        layer=layer,
        projection=projection,
        entity_level=entity_level,
        include_kinds=include_kinds,
        exclude_kinds=exclude_kinds,
        include_edge_kinds=include_edge_kinds,
    )
    all_nodes = sorted(graph["nodes"], key=lambda n: str(n.get("natural_key") or n.get("id")))
    page_nodes = all_nodes[page * limit : (page + 1) * limit]
    page_ids = {str(n["id"]) for n in page_nodes}
    page_edges = [
        edge
        for edge in graph["edges"]
        if str(edge.get("source_node_id")) in page_ids
        and str(edge.get("target_node_id")) in page_ids
    ]

    return {
        "nodes": page_nodes,
        "edges": page_edges,
        "page": page,
        "limit": limit,
        "total_nodes": int(graph["total_nodes"]),
        "projection": graph["projection"],
        "entity_level": graph["entity_level"],
        "grouping_strategy": graph["grouping_strategy"],
        "excluded_kinds": graph["excluded_kinds"],
    }


async def get_full_scenario_graph(
    session: AsyncSession,
    scenario_id: UUID,
    layer: TwinLayer | None,
    projection: GraphProjection = GraphProjection.CODE_SYMBOL,
    entity_level: str | None = None,
    include_kinds: set[str] | None = None,
    exclude_kinds: set[str] | None = None,
    include_edge_kinds: set[str] | None = None,
) -> dict[str, Any]:
    """Get full (unpaged) graph view with optional projection."""
    node_stmt = select(TwinNode).where(TwinNode.scenario_id == scenario_id)
    if layer is not None:
        node_stmt = node_stmt.join(TwinNodeLayer, TwinNodeLayer.node_id == TwinNode.id).where(
            TwinNodeLayer.layer == layer
        )

    raw_nodes = (await session.execute(node_stmt)).scalars().all()
    raw_node_ids = {n.id for n in raw_nodes}
    edge_stmt = select(TwinEdge).where(TwinEdge.scenario_id == scenario_id)
    if raw_node_ids:
        edge_stmt = edge_stmt.where(
            TwinEdge.source_node_id.in_(raw_node_ids),
            TwinEdge.target_node_id.in_(raw_node_ids),
        )
    else:
        edge_stmt = edge_stmt.limit(0)
    raw_edges = (await session.execute(edge_stmt)).scalars().all()

    nodes = [
        {
            "id": str(n.id),
            "natural_key": n.natural_key,
            "kind": n.kind,
            "name": n.name,
            "meta": n.meta or {},
        }
        for n in raw_nodes
    ]
    edges = [
        {
            "id": str(e.id),
            "source_node_id": str(e.source_node_id),
            "target_node_id": str(e.target_node_id),
            "kind": e.kind,
            "meta": e.meta or {},
        }
        for e in raw_edges
    ]

    include_kinds_norm = {kind.lower() for kind in include_kinds} if include_kinds else None
    exclude_kinds_norm = {kind.lower() for kind in exclude_kinds} if exclude_kinds else None
    include_edge_kinds_norm = (
        {kind.lower() for kind in include_edge_kinds} if include_edge_kinds else None
    )

    grouping_strategy = "heuristic"
    effective_entity_level = entity_level
    effective_excluded_kinds = sorted(exclude_kinds_norm or set())

    if projection == GraphProjection.ARCHITECTURE:
        effective_entity_level = (entity_level or "container").lower()
        default_hidden = {
            "class",
            "method",
            "function",
            "property",
            "parameter",
            "variable",
            "constant",
        }
        effective_excluded = set(exclude_kinds_norm or set()) | default_hidden
        projected_nodes, projected_edges, grouping_strategy = build_architecture_projection(
            nodes=nodes,
            edges=edges,
            entity_level=effective_entity_level,
            include_kinds=include_kinds_norm,
            exclude_kinds=effective_excluded,
        )
        effective_excluded_kinds = sorted(effective_excluded)
    elif projection == GraphProjection.CODE_FILE:
        effective_entity_level = (entity_level or "file").lower()
        projected_nodes, projected_edges = build_code_file_projection(
            nodes=nodes,
            edges=edges,
            include_edge_kinds=include_edge_kinds_norm,
        )
    else:
        effective_entity_level = (entity_level or "symbol").lower()
        projected_nodes, projected_edges = build_code_symbol_projection(
            nodes=nodes,
            edges=edges,
            include_kinds=include_kinds_norm,
            exclude_kinds=exclude_kinds_norm,
            include_edge_kinds=include_edge_kinds_norm,
        )

    return {
        "nodes": projected_nodes,
        "edges": projected_edges,
        "total_nodes": len(projected_nodes),
        "projection": projection.value,
        "entity_level": effective_entity_level,
        "grouping_strategy": grouping_strategy,
        "excluded_kinds": effective_excluded_kinds,
    }


async def refresh_metric_snapshots(
    session: AsyncSession,
    scenario_id: UUID,
) -> int:
    """Refresh code city metric snapshots from real file metrics in twin graph."""
    await session.execute(delete(MetricSnapshot).where(MetricSnapshot.scenario_id == scenario_id))

    nodes = (
        (await session.execute(select(TwinNode).where(TwinNode.scenario_id == scenario_id)))
        .scalars()
        .all()
    )

    required_fields = ("loc", "complexity", "coupling")
    created = 0
    for node in nodes:
        if node.kind != "file":
            continue

        meta = node.meta or {}
        if not bool(meta.get("metrics_structural_ready")):
            continue

        if any(meta.get(field) is None for field in required_fields):
            continue

        snapshot = MetricSnapshot(
            id=uuid.uuid4(),
            scenario_id=scenario_id,
            node_natural_key=node.natural_key,
            loc=int(meta["loc"]),
            symbol_count=int(meta.get("symbol_count", 0) or 0),
            coupling=float(meta["coupling"]),
            coverage=float(meta.get("coverage", 0.0) or 0.0),
            complexity=float(meta["complexity"]),
            cohesion=float(meta.get("cohesion", 1.0) or 1.0),
            instability=float(meta.get("instability", 0.0) or 0.0),
            fan_in=int(meta.get("fan_in", 0) or 0),
            fan_out=int(meta.get("fan_out", 0) or 0),
            cycle_participation=bool(meta.get("cycle_participation", False)),
            cycle_size=int(meta.get("cycle_size", 0) or 0),
            duplication_ratio=float(meta.get("duplication_ratio", 0.0) or 0.0),
            crap_score=(float(meta["crap_score"]) if meta.get("crap_score") is not None else None),
            change_frequency=float(meta.get("change_frequency", 0.0) or 0.0),
            meta=meta,
        )
        session.add(snapshot)
        created += 1
    return created


async def apply_file_metrics_to_scenario(
    session: AsyncSession,
    scenario_id: UUID,
    file_metrics: list[dict[str, Any]],
) -> int:
    """Apply structural file-level metrics onto scenario file nodes."""
    if not file_metrics:
        return 0

    by_key: dict[str, dict[str, Any]] = {}
    for metric in file_metrics:
        file_path = str(metric.get("file_path", "")).strip()
        if not file_path:
            continue
        by_key[f"file:{file_path}"] = metric

    if not by_key:
        return 0

    nodes = (
        (
            await session.execute(
                select(TwinNode).where(
                    TwinNode.scenario_id == scenario_id,
                    TwinNode.natural_key.in_(list(by_key.keys())),
                )
            )
        )
        .scalars()
        .all()
    )

    updated = 0
    for node in nodes:
        metric = by_key.get(node.natural_key)
        if not metric:
            continue

        meta = dict(node.meta or {})
        meta.update(
            {
                "metrics_real": False,
                "metrics_structural_ready": True,
                "coverage_ready": False,
                "loc": int(metric["loc"]),
                "complexity": float(metric["complexity"]),
                "coupling_in": int(metric["coupling_in"]),
                "coupling_out": int(metric["coupling_out"]),
                "coupling": float(metric["coupling"]),
                "cohesion": float(metric.get("cohesion", 1.0) or 1.0),
                "instability": float(metric.get("instability", 0.0) or 0.0),
                "fan_in": int(metric.get("fan_in", 0) or 0),
                "fan_out": int(metric.get("fan_out", 0) or 0),
                "cycle_participation": bool(metric.get("cycle_participation", False)),
                "cycle_size": int(metric.get("cycle_size", 0) or 0),
                "duplication_ratio": float(metric.get("duplication_ratio", 0.0) or 0.0),
                "crap_score": (
                    float(metric["crap_score"]) if metric.get("crap_score") is not None else None
                ),
                "change_frequency": float(metric.get("change_frequency", 0.0) or 0.0),
                "churn": float(metric.get("churn", 0.0) or 0.0),
                "coverage": None,
                "metrics_sources": metric.get("sources", {}),
                "metrics_language": metric.get("language"),
            }
        )
        node.meta = meta
        updated += 1

    return updated


async def apply_coverage_metrics_to_scenario(
    session: AsyncSession,
    scenario_id: UUID,
    source_id: UUID,
    coverage_map: dict[str, float],
    coverage_sources: dict[str, dict[str, Any]] | None = None,
    commit_sha: str | None = None,
    ingest_job_id: UUID | None = None,
) -> int:
    """Apply coverage-only metrics onto existing file nodes for a source."""
    if not coverage_map:
        return 0

    coverage_sources = coverage_sources or {}
    nodes = (
        (await session.execute(select(TwinNode).where(TwinNode.scenario_id == scenario_id)))
        .scalars()
        .all()
    )

    updated = 0
    source_id_str = str(source_id)
    for node in nodes:
        if node.kind != "file":
            continue
        if not node.natural_key.startswith("file:"):
            continue

        meta = dict(node.meta or {})
        if str(meta.get("source_id") or "") != source_id_str:
            continue

        file_path = node.natural_key.removeprefix("file:")
        if file_path not in coverage_map:
            continue

        metrics_sources = dict(meta.get("metrics_sources") or {})
        metrics_sources["coverage"] = coverage_sources.get(file_path, {})
        complexity_value = float(meta["complexity"]) if meta.get("complexity") is not None else None
        coverage_value = float(coverage_map[file_path])
        meta.update(
            {
                "coverage": coverage_value,
                "coverage_ready": True,
                "metrics_real": bool(meta.get("metrics_structural_ready")),
                "metrics_sources": metrics_sources,
                "crap_score": _compute_crap_score(complexity_value, coverage_value),
            }
        )
        if commit_sha:
            meta["coverage_commit_sha"] = commit_sha
        if ingest_job_id:
            meta["coverage_ingest_job_id"] = str(ingest_job_id)
        node.meta = meta
        node.updated_at = datetime.now(UTC)
        updated += 1

    return updated
