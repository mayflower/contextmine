"""Twin manifest export for rebuild-oriented agent workflows."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import UUID

from contextmine_core.twin import GraphProjection, get_full_scenario_graph
from contextmine_core.twin.projections import (
    build_test_matrix_projection,
    build_ui_map_projection,
    build_user_flows_projection,
    compute_rebuild_readiness,
)
from sqlalchemy.ext.asyncio import AsyncSession


async def export_twin_manifest(
    session: AsyncSession,
    scenario_id: UUID,
) -> str:
    """Export normalized twin manifest JSON for rebuild agents."""
    full_graph = await get_full_scenario_graph(
        session=session,
        scenario_id=scenario_id,
        layer=None,
        projection=GraphProjection.CODE_SYMBOL,
    )
    architecture_graph = await get_full_scenario_graph(
        session=session,
        scenario_id=scenario_id,
        layer=None,
        projection=GraphProjection.ARCHITECTURE,
        entity_level="container",
    )

    nodes = list(full_graph.get("nodes") or [])
    edges = list(full_graph.get("edges") or [])
    ui_map = build_ui_map_projection(nodes, edges)
    test_matrix = build_test_matrix_projection(nodes, edges)
    user_flows = build_user_flows_projection(nodes, edges)
    readiness = compute_rebuild_readiness(nodes, edges)

    interfaces = []
    for node in nodes:
        kind = str(node.get("kind") or "")
        if kind not in {"api_endpoint", "graphql_operation", "service_rpc", "interface_contract"}:
            continue
        interfaces.append(
            {
                "id": str(node.get("id")),
                "kind": kind,
                "name": str(node.get("name") or ""),
                "natural_key": str(node.get("natural_key") or ""),
                "meta": node.get("meta") or {},
            }
        )

    manifest = {
        "manifest_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "scenario_id": str(scenario_id),
        "sections": {
            "architecture": {
                "projection": "architecture/container",
                "graph": architecture_graph,
            },
            "interfaces": interfaces,
            "ui_map": {
                "summary": ui_map.get("summary") or {},
                "graph": ui_map.get("graph") or {},
            },
            "user_flows": {
                "summary": user_flows.get("summary") or {},
                "flows": user_flows.get("flows") or [],
                "graph": user_flows.get("graph") or {},
            },
            "test_obligations": {
                "summary": test_matrix.get("summary") or {},
                "matrix": test_matrix.get("matrix") or [],
                "graph": test_matrix.get("graph") or {},
            },
            "readiness": readiness,
        },
        "known_gaps": readiness.get("known_gaps") or [],
    }
    return json.dumps(manifest, indent=2, sort_keys=True)
