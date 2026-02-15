"""Digital twin and architecture intent routes."""

from __future__ import annotations

import uuid
from typing import Literal

from contextmine_core import Collection, CollectionMember
from contextmine_core import get_session as get_db_session
from contextmine_core.architecture_intents import ArchitectureIntentV1
from contextmine_core.exports import (
    export_codecharta_json,
    export_cx2,
    export_jgf,
    export_lpg_jsonl,
    export_mermaid_asis_tobe,
    export_mermaid_c4,
)
from contextmine_core.graph.age import run_read_only_cypher, sync_scenario_to_age
from contextmine_core.models import (
    KnowledgeArtifact,
    KnowledgeArtifactKind,
    TwinLayer,
    TwinScenario,
)
from contextmine_core.twin import (
    approve_and_execute_intent,
    create_to_be_scenario,
    get_scenario_graph,
    list_scenario_patches,
    submit_intent,
)
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.middleware import get_session

router = APIRouter(prefix="/twin", tags=["twin"])


class CreateScenarioRequest(BaseModel):
    collection_id: str
    name: str = Field(min_length=1, max_length=255)


class CypherRequest(BaseModel):
    query: str = Field(min_length=1)


class ExportRequest(BaseModel):
    format: Literal["lpg_jsonl", "cc_json", "cx2", "jgf", "mermaid_c4"]


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
    page: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=5000),
) -> dict:
    user_id = _user_id_or_401(request)

    layer_enum: TwinLayer | None = None
    if layer:
        try:
            layer_enum = TwinLayer(layer)
        except ValueError as e:
            raise HTTPException(status_code=400, detail="Invalid layer") from e

    async with get_db_session() as db:
        scenario = await _load_scenario(db, scenario_id)
        await _ensure_member(db, scenario.collection_id, user_id)
        return await get_scenario_graph(db, scenario.id, layer_enum, page, limit)


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

        if body.format == "lpg_jsonl":
            content = await export_lpg_jsonl(db, scenario.id)
            kind = KnowledgeArtifactKind.LPG_JSONL
            name = f"{scenario.name}.lpg.jsonl"
        elif body.format == "cc_json":
            content = await export_codecharta_json(db, scenario.id)
            kind = KnowledgeArtifactKind.CC_JSON
            name = f"{scenario.name}.cc.json"
        elif body.format == "cx2":
            content = await export_cx2(db, scenario.id)
            kind = KnowledgeArtifactKind.CX2
            name = f"{scenario.name}.cx2.json"
        elif body.format == "jgf":
            content = await export_jgf(db, scenario.id)
            kind = KnowledgeArtifactKind.JGF
            name = f"{scenario.name}.jgf.json"
        else:
            if scenario.base_scenario_id:
                as_is_content, to_be_content = await export_mermaid_asis_tobe(
                    db,
                    as_is_scenario_id=scenario.base_scenario_id,
                    to_be_scenario_id=scenario.id,
                )
                as_is_artifact = KnowledgeArtifact(
                    id=uuid.uuid4(),
                    collection_id=scenario.collection_id,
                    kind=KnowledgeArtifactKind.MERMAID_C4_ASIS,
                    name=f"{scenario.name}.asis.mmd",
                    content=as_is_content,
                    meta={"scenario_id": str(scenario.base_scenario_id)},
                )
                to_be_artifact = KnowledgeArtifact(
                    id=uuid.uuid4(),
                    collection_id=scenario.collection_id,
                    kind=KnowledgeArtifactKind.MERMAID_C4_TOBE,
                    name=f"{scenario.name}.tobe.mmd",
                    content=to_be_content,
                    meta={"scenario_id": str(scenario.id)},
                )
                db.add(as_is_artifact)
                db.add(to_be_artifact)
                await db.commit()
                return {
                    "exports": [
                        {"id": str(as_is_artifact.id), "name": as_is_artifact.name},
                        {"id": str(to_be_artifact.id), "name": to_be_artifact.name},
                    ]
                }

            content = await export_mermaid_c4(db, scenario.id)
            kind = (
                KnowledgeArtifactKind.MERMAID_C4_ASIS
                if scenario.is_as_is
                else KnowledgeArtifactKind.MERMAID_C4_TOBE
            )
            name = f"{scenario.name}.mmd"

        artifact = KnowledgeArtifact(
            id=uuid.uuid4(),
            collection_id=scenario.collection_id,
            kind=kind,
            name=name,
            content=content,
            meta={"scenario_id": str(scenario.id), "format": body.format},
        )
        db.add(artifact)
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
