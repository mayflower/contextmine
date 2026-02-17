"""CX2 and JGF graph exports for twin scenarios."""

from __future__ import annotations

import json
from uuid import UUID

from contextmine_core.models import TwinEdge, TwinNode
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


def export_cx2_from_graph(scenario_id: UUID, graph: dict) -> str:
    """Export already projected graph payload as Cytoscape CX2-like JSON."""
    payload = {
        "CXVersion": "2.0",
        "metaData": [{"name": "nodes"}, {"name": "edges"}],
        "nodes": [
            {
                "id": str(node["id"]),
                "name": node.get("name"),
                "represents": node.get("natural_key"),
                "attributes": {"kind": node.get("kind"), **(node.get("meta") or {})},
            }
            for node in graph.get("nodes", [])
        ],
        "edges": [
            {
                "id": str(edge["id"]),
                "s": str(edge["source_node_id"]),
                "t": str(edge["target_node_id"]),
                "i": edge.get("kind"),
                "attributes": edge.get("meta") or {},
            }
            for edge in graph.get("edges", [])
        ],
    }
    return json.dumps(payload, indent=2)


def export_jgf_from_graph(scenario_id: UUID, graph: dict) -> str:
    """Export already projected graph payload as JSON Graph Format (JGF)."""
    payload = {
        "graph": {
            "id": str(scenario_id),
            "type": "digraph",
            "nodes": [
                {
                    "id": str(node["id"]),
                    "label": node.get("name"),
                    "metadata": {
                        "kind": node.get("kind"),
                        "natural_key": node.get("natural_key"),
                        **(node.get("meta") or {}),
                    },
                }
                for node in graph.get("nodes", [])
            ],
            "edges": [
                {
                    "id": str(edge["id"]),
                    "source": str(edge["source_node_id"]),
                    "target": str(edge["target_node_id"]),
                    "relation": edge.get("kind"),
                    "metadata": edge.get("meta") or {},
                }
                for edge in graph.get("edges", [])
            ],
        }
    }
    return json.dumps(payload, indent=2)


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

    return export_cx2_from_graph(
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

    return export_jgf_from_graph(
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
