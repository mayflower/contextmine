"""Digital twin and export MCP tools."""

import uuid
from typing import Annotated, Any, Literal

from contextmine_core import Collection, user_can_access_collection
from contextmine_core import get_session as get_db_session
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.mcp_auth import get_current_user_id


class TwinGraphResult(BaseModel):
    """Bounded digital-twin graph projection."""

    scenario_id: str
    facet: Literal["code", "tests", "ui", "flows", "all"]
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    page: int
    limit: int
    total_nodes: int
    truncated: bool


class TwinQueryResult(BaseModel):
    """Rows returned by a read-only AGE query."""

    rows: list[dict[str, Any]]
    count: int


class TwinStatusResult(BaseModel):
    """Digital-twin freshness status from the shared core."""

    status: dict[str, Any]


class TwinTimelineResult(BaseModel):
    """Paginated digital-twin event timeline."""

    timeline: dict[str, Any]


class TwinRefreshResult(BaseModel):
    """Existing ContextMine source-version IDs queued for materialization."""

    collection_id: str
    created: int
    skipped: int
    items: list[dict[str, Any]]


class TwinExportResult(BaseModel):
    """Persisted export artifact metadata."""

    artifact_id: str
    scenario_id: str
    name: str
    kind: str
    format: str


def _uuid(value: str, field_name: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except ValueError as exc:
        raise ToolError(f"Invalid {field_name}: {value}") from exc


def _optional_uuid(value: str | None, field_name: str) -> uuid.UUID | None:
    return _uuid(value, field_name) if value else None


async def _require_collection(db, collection_id: str) -> Collection:
    collection_uuid = _uuid(collection_id, "collection_id")
    collection = (
        await db.execute(select(Collection).where(Collection.id == collection_uuid))
    ).scalar_one_or_none()
    if collection is None:
        raise ToolError("Collection not found.")
    if not await user_can_access_collection(db, collection, get_current_user_id()):
        raise ToolError("Access denied to this collection.")
    return collection


def _node_kind_in_scope(kind: str, scope: str) -> bool:
    normalized = kind.strip().lower()
    test_kinds = {"test_suite", "test_case", "test_fixture"}
    ui_kinds = {"ui_route", "ui_view", "ui_component", "interface_contract"}
    flow_kinds = {"user_flow", "flow_step"}
    if scope == "all":
        return True
    if scope == "tests":
        return normalized in test_kinds
    if scope == "ui":
        return normalized in ui_kinds
    if scope == "flows":
        return normalized in flow_kinds
    return normalized not in (test_kinds | ui_kinds | flow_kinds)


def _filter_graph_payload(
    graph: dict[str, Any],
    *,
    scope: str,
    provenance_mode: str | None,
    include_test_links: bool | None,
    include_ui_links: bool | None,
) -> dict[str, Any]:
    nodes = list(graph.get("nodes") or [])
    edges = list(graph.get("edges") or [])
    allowed_ids = {
        str(node.get("id"))
        for node in nodes
        if _node_kind_in_scope(str(node.get("kind") or ""), scope)
    }
    nodes = [node for node in nodes if str(node.get("id")) in allowed_ids]
    edges = [
        edge
        for edge in edges
        if str(edge.get("source_node_id")) in allowed_ids
        and str(edge.get("target_node_id")) in allowed_ids
    ]

    if provenance_mode:
        mode = provenance_mode.strip().lower()
        if mode not in {"deterministic", "inferred"}:
            raise ToolError("Invalid provenance mode. Use deterministic or inferred.")
        allowed_ids = {
            str(node.get("id"))
            for node in nodes
            if str(((node.get("meta") or {}).get("provenance") or {}).get("mode") or "")
            .strip()
            .lower()
            == mode
        }
        nodes = [node for node in nodes if str(node.get("id")) in allowed_ids]
        edges = [
            edge
            for edge in edges
            if str(edge.get("source_node_id")) in allowed_ids
            and str(edge.get("target_node_id")) in allowed_ids
        ]

    if include_test_links is False:
        edges = [
            edge for edge in edges if not str(edge.get("kind") or "").lower().startswith("test_")
        ]
    if include_ui_links is False:
        edges = [
            edge for edge in edges if not str(edge.get("kind") or "").lower().startswith("ui_")
        ]
    return {**graph, "nodes": nodes, "edges": edges, "total_nodes": len(nodes)}


async def get_twin_graph(
    scenario_id: Annotated[str, Field(description="Scenario UUID")],
    layer: Annotated[str | None, Field(description="Optional twin layer")] = None,
    page: Annotated[int, Field(description="Page index", ge=0)] = 0,
    limit: Annotated[int, Field(description="Nodes per page", ge=1, le=5000)] = 200,
    facet: Annotated[
        Literal["code", "tests", "ui", "flows", "all"],
        Field(description="Semantic facet filter"),
    ] = "all",
    include_provenance_mode: Annotated[
        Literal["deterministic", "inferred"] | None,
        Field(description="Optional provenance filter"),
    ] = None,
    include_test_links: Annotated[bool | None, Field(description="Include test links")] = None,
    include_ui_links: Annotated[bool | None, Field(description="Include UI links")] = None,
) -> TwinGraphResult:
    """Read a bounded digital-twin graph page."""
    from contextmine_core.models import TwinLayer
    from contextmine_core.twin import get_scenario_graph

    scenario_uuid = _uuid(scenario_id, "scenario_id")
    try:
        layer_value = TwinLayer(layer) if layer else None
    except ValueError as exc:
        raise ToolError(f"Invalid layer: {layer}") from exc
    try:
        async with get_db_session() as db:
            graph = await get_scenario_graph(db, scenario_uuid, layer_value, page, limit)
    except Exception as exc:
        raise ToolError(f"Failed to read twin graph: {exc}") from exc
    graph = _filter_graph_payload(
        graph,
        scope=facet,
        provenance_mode=include_provenance_mode,
        include_test_links=include_test_links,
        include_ui_links=include_ui_links,
    )
    nodes = list(graph.get("nodes") or [])
    edges = list(graph.get("edges") or [])
    return TwinGraphResult(
        scenario_id=scenario_id,
        facet=facet,
        nodes=nodes,
        edges=edges,
        page=page,
        limit=limit,
        total_nodes=int(graph.get("total_nodes") or len(nodes)),
        truncated=len(nodes) >= limit,
    )


async def query_twin_cypher(
    scenario_id: Annotated[str, Field(description="Scenario UUID")],
    query: Annotated[
        str, Field(description="Read-only Cypher query", min_length=1, max_length=20000)
    ],
) -> TwinQueryResult:
    """Run the existing guarded read-only Cypher implementation."""
    from contextmine_core.graph.age import run_read_only_cypher, sync_scenario_to_age

    scenario_uuid = _uuid(scenario_id, "scenario_id")
    try:
        async with get_db_session() as db:
            await sync_scenario_to_age(db, scenario_uuid)
            rows = await run_read_only_cypher(db, scenario_uuid, query)
    except Exception as exc:
        raise ToolError(f"Cypher query failed: {exc}") from exc
    return TwinQueryResult(rows=rows, count=len(rows))


async def get_twin_status(
    collection_id: Annotated[str, Field(description="Collection UUID")],
    scenario_id: Annotated[str | None, Field(description="Optional scenario UUID")] = None,
) -> TwinStatusResult:
    """Get digital-twin freshness status using the shared REST/core implementation."""
    from contextmine_core.twin import get_collection_twin_status

    scenario_uuid = _optional_uuid(scenario_id, "scenario_id")
    async with get_db_session() as db:
        collection = await _require_collection(db, collection_id)
        try:
            payload = await get_collection_twin_status(
                db, collection_id=collection.id, scenario_id=scenario_uuid
            )
        except Exception as exc:
            raise ToolError(f"Failed to fetch twin status: {exc}") from exc
    return TwinStatusResult(status=payload)


async def get_twin_timeline(
    collection_id: Annotated[str, Field(description="Collection UUID")],
    source_id: Annotated[str | None, Field(description="Optional source UUID")] = None,
    event_type: Annotated[
        str | None, Field(description="Optional event type", max_length=100)
    ] = None,
    status: Annotated[str | None, Field(description="Optional status", max_length=100)] = None,
    page: Annotated[int, Field(description="Page index", ge=0)] = 0,
    limit: Annotated[int, Field(description="Items per page", ge=1, le=200)] = 50,
) -> TwinTimelineResult:
    """Read a bounded digital-twin event timeline."""
    from contextmine_core.twin import list_collection_twin_events

    source_uuid = _optional_uuid(source_id, "source_id")
    async with get_db_session() as db:
        collection = await _require_collection(db, collection_id)
        try:
            payload = await list_collection_twin_events(
                db,
                collection_id=collection.id,
                page=page,
                limit=limit,
                source_id=source_uuid,
                event_type=event_type,
                status=status,
            )
        except Exception as exc:
            raise ToolError(f"Failed to fetch twin timeline: {exc}") from exc
    return TwinTimelineResult(timeline=payload)


async def refresh_twin(
    collection_id: Annotated[str, Field(description="Collection UUID")],
    source_ids: Annotated[
        list[str] | None, Field(description="Optional source UUIDs", max_length=100)
    ] = None,
    force: Annotated[bool, Field(description="Refresh unchanged revisions too")] = False,
) -> TwinRefreshResult:
    """Queue source versions through the existing twin refresh implementation."""
    from contextmine_core.twin import coerce_source_ids, trigger_collection_refresh

    try:
        parsed_source_ids = coerce_source_ids(source_ids)
    except ValueError as exc:
        raise ToolError("Invalid source_ids") from exc
    async with get_db_session() as db:
        collection = await _require_collection(db, collection_id)
        try:
            payload = await trigger_collection_refresh(
                db,
                collection_id=collection.id,
                source_ids=parsed_source_ids or None,
                force=force,
            )
            await db.commit()
        except Exception as exc:
            raise ToolError(f"Failed to refresh twin: {exc}") from exc
    return TwinRefreshResult.model_validate(payload)


async def _run_export(db, scenario, format_name: str):
    from contextmine_core.exports import (
        export_codecharta_json,
        export_cx2,
        export_jgf,
        export_lpg_jsonl,
        export_mermaid_c4,
        export_twin_manifest,
    )
    from contextmine_core.models import KnowledgeArtifactKind
    from contextmine_core.twin import GraphProjection

    dispatch = {
        "lpg_jsonl": (export_lpg_jsonl, KnowledgeArtifactKind.LPG_JSONL, "lpg.jsonl"),
        "cx2": (export_cx2, KnowledgeArtifactKind.CX2, "cx2.json"),
        "jgf": (export_jgf, KnowledgeArtifactKind.JGF, "jgf.json"),
        "twin_manifest": (
            export_twin_manifest,
            KnowledgeArtifactKind.TWIN_MANIFEST,
            "twin_manifest.json",
        ),
    }
    if format_name in dispatch:
        export_fn, kind, suffix = dispatch[format_name]
        return await export_fn(db, scenario.id), kind, f"{scenario.name}.{suffix}"
    if format_name == "cc_json":
        content = await export_codecharta_json(
            db,
            scenario.id,
            projection=GraphProjection.ARCHITECTURE,
            entity_level="container",
        )
        return content, KnowledgeArtifactKind.CC_JSON, f"{scenario.name}.cc.json"
    content = await export_mermaid_c4(db, scenario.id)
    kind = (
        KnowledgeArtifactKind.MERMAID_C4_ASIS
        if scenario.is_as_is
        else KnowledgeArtifactKind.MERMAID_C4_TOBE
    )
    return content, kind, f"{scenario.name}.mmd"


async def export_twin_view(
    scenario_id: Annotated[str, Field(description="Scenario UUID")],
    format: Annotated[
        Literal["lpg_jsonl", "cc_json", "cx2", "jgf", "mermaid_c4", "twin_manifest"],
        Field(description="Export format"),
    ],
) -> TwinExportResult:
    """Generate and persist a twin export through the existing exporters."""
    from contextmine_core.models import KnowledgeArtifact, TwinScenario

    scenario_uuid = _uuid(scenario_id, "scenario_id")
    async with get_db_session() as db:
        scenario = (
            await db.execute(select(TwinScenario).where(TwinScenario.id == scenario_uuid))
        ).scalar_one_or_none()
        if scenario is None:
            raise ToolError("Scenario not found.")
        try:
            content, kind, name = await _run_export(db, scenario, format)
            artifact = KnowledgeArtifact(
                id=uuid.uuid4(),
                collection_id=scenario.collection_id,
                kind=kind,
                name=name,
                content=content,
                meta={"scenario_id": str(scenario.id), "format": format},
            )
            db.add(artifact)
            await db.commit()
        except Exception as exc:
            raise ToolError(f"Export failed: {exc}") from exc
    return TwinExportResult(
        artifact_id=str(artifact.id),
        scenario_id=scenario_id,
        name=artifact.name,
        kind=artifact.kind.value,
        format=format,
    )


def register_twin_tools(mcp: FastMCP) -> None:
    """Register digital-twin and export tools."""
    mcp.tool(name="get_twin_graph")(get_twin_graph)
    mcp.tool(name="query_twin_cypher")(query_twin_cypher)
    mcp.tool(name="get_twin_status")(get_twin_status)
    mcp.tool(name="get_twin_timeline")(get_twin_timeline)
    mcp.tool(name="refresh_twin")(refresh_twin)
    mcp.tool(name="export_twin_view")(export_twin_view)
