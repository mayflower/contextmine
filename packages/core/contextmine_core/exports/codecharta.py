"""CodeCharta cc.json export for twin scenarios."""

from __future__ import annotations

import json
from uuid import UUID

from contextmine_core.models import MetricSnapshot, TwinEdge
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


async def export_codecharta_json(session: AsyncSession, scenario_id: UUID) -> str:
    """Export scenario metrics in a CodeCharta-friendly structure."""
    metrics = (
        (
            await session.execute(
                select(MetricSnapshot).where(MetricSnapshot.scenario_id == scenario_id)
            )
        )
        .scalars()
        .all()
    )

    nodes = []
    for metric in metrics:
        nodes.append(
            {
                "name": metric.node_natural_key,
                "attributes": {
                    "loc": metric.loc or 0,
                    "symbol_count": metric.symbol_count or 0,
                    "coupling": metric.coupling or 0.0,
                    "coverage": metric.coverage or 0.0,
                    "complexity": metric.complexity or 0.0,
                    "change_frequency": metric.change_frequency or 0.0,
                },
            }
        )

    edges = (
        (await session.execute(select(TwinEdge).where(TwinEdge.scenario_id == scenario_id)))
        .scalars()
        .all()
    )
    cc_edges = [
        {
            "from": str(edge.source_node_id),
            "to": str(edge.target_node_id),
            "type": edge.kind,
        }
        for edge in edges
    ]

    payload = {
        "projectName": f"scenario-{scenario_id}",
        "nodes": nodes,
        "edges": cc_edges,
    }
    return json.dumps(payload, indent=2)
