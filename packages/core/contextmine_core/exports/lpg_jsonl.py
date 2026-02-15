"""LPG JSONL export for twin scenarios."""

from __future__ import annotations

import json
from uuid import UUID

from contextmine_core.models import TwinEdge, TwinNode
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


async def export_lpg_jsonl(session: AsyncSession, scenario_id: UUID) -> str:
    """Export scenario graph as JSONL (nodes + edges)."""
    lines: list[str] = []

    nodes = (
        (await session.execute(select(TwinNode).where(TwinNode.scenario_id == scenario_id)))
        .scalars()
        .all()
    )
    for node in nodes:
        lines.append(
            json.dumps(
                {
                    "type": "node",
                    "id": str(node.id),
                    "scenario_id": str(node.scenario_id),
                    "labels": [node.kind],
                    "natural_key": node.natural_key,
                    "name": node.name,
                    "properties": node.meta or {},
                }
            )
        )

    edges = (
        (await session.execute(select(TwinEdge).where(TwinEdge.scenario_id == scenario_id)))
        .scalars()
        .all()
    )
    for edge in edges:
        lines.append(
            json.dumps(
                {
                    "type": "edge",
                    "id": str(edge.id),
                    "scenario_id": str(edge.scenario_id),
                    "label": edge.kind,
                    "source": str(edge.source_node_id),
                    "target": str(edge.target_node_id),
                    "properties": edge.meta or {},
                }
            )
        )

    return "\n".join(lines)
