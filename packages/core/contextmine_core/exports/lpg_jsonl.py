"""LPG JSONL export for twin scenarios."""

from __future__ import annotations

import json
from uuid import UUID

from contextmine_core.models import TwinEdge, TwinNode
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


def export_lpg_jsonl_from_graph(scenario_id: UUID, graph: dict) -> str:
    """Export already projected graph payload as LPG JSONL."""
    lines: list[str] = []
    for node in graph.get("nodes", []):
        lines.append(
            json.dumps(
                {
                    "type": "node",
                    "id": str(node["id"]),
                    "scenario_id": str(scenario_id),
                    "labels": [str(node.get("kind") or "node")],
                    "natural_key": node.get("natural_key"),
                    "name": node.get("name"),
                    "properties": node.get("meta") or {},
                }
            )
        )
    for edge in graph.get("edges", []):
        lines.append(
            json.dumps(
                {
                    "type": "edge",
                    "id": str(edge["id"]),
                    "scenario_id": str(scenario_id),
                    "label": edge.get("kind"),
                    "source": str(edge["source_node_id"]),
                    "target": str(edge["target_node_id"]),
                    "properties": edge.get("meta") or {},
                }
            )
        )
    return "\n".join(lines)


async def export_lpg_jsonl(session: AsyncSession, scenario_id: UUID) -> str:
    """Export scenario graph as JSONL (nodes + edges)."""
    nodes = (
        (await session.execute(select(TwinNode).where(TwinNode.scenario_id == scenario_id)))
        .scalars()
        .all()
    )
    edges = (
        (await session.execute(select(TwinEdge).where(TwinEdge.scenario_id == scenario_id)))
        .scalars()
        .all()
    )
    return export_lpg_jsonl_from_graph(
        scenario_id,
        {
            "nodes": [
                {
                    "id": str(node.id),
                    "natural_key": node.natural_key,
                    "kind": node.kind,
                    "name": node.name,
                    "meta": node.meta or {},
                }
                for node in nodes
            ],
            "edges": [
                {
                    "id": str(edge.id),
                    "source_node_id": str(edge.source_node_id),
                    "target_node_id": str(edge.target_node_id),
                    "kind": edge.kind,
                    "meta": edge.meta or {},
                }
                for edge in edges
            ],
        },
    )
