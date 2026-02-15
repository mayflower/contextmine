"""CX2 and JGF graph exports for twin scenarios."""

from __future__ import annotations

import json
from uuid import UUID

from contextmine_core.models import TwinEdge, TwinNode
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


async def export_cx2(session: AsyncSession, scenario_id: UUID) -> str:
    """Export scenario as Cytoscape CX2-like JSON."""
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

    payload = {
        "CXVersion": "2.0",
        "metaData": [{"name": "nodes"}, {"name": "edges"}],
        "nodes": [
            {
                "id": str(node.id),
                "name": node.name,
                "represents": node.natural_key,
                "attributes": {"kind": node.kind, **(node.meta or {})},
            }
            for node in nodes
        ],
        "edges": [
            {
                "id": str(edge.id),
                "s": str(edge.source_node_id),
                "t": str(edge.target_node_id),
                "i": edge.kind,
                "attributes": edge.meta or {},
            }
            for edge in edges
        ],
    }
    return json.dumps(payload, indent=2)


async def export_jgf(session: AsyncSession, scenario_id: UUID) -> str:
    """Export scenario as JSON Graph Format (JGF)."""
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

    payload = {
        "graph": {
            "id": str(scenario_id),
            "type": "digraph",
            "nodes": [
                {
                    "id": str(node.id),
                    "label": node.name,
                    "metadata": {
                        "kind": node.kind,
                        "natural_key": node.natural_key,
                        **(node.meta or {}),
                    },
                }
                for node in nodes
            ],
            "edges": [
                {
                    "id": str(edge.id),
                    "source": str(edge.source_node_id),
                    "target": str(edge.target_node_id),
                    "relation": edge.kind,
                    "metadata": edge.meta or {},
                }
                for edge in edges
            ],
        }
    }
    return json.dumps(payload, indent=2)
