"""Mermaid C4 export for AS-IS/TO-BE twin scenarios."""

from __future__ import annotations

from uuid import UUID

from contextmine_core.models import TwinScenario
from contextmine_core.twin import GraphProjection, get_full_scenario_graph
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


def _safe_id(value: str) -> str:
    return value.replace("-", "_").replace(":", "_").replace("/", "_")


async def export_mermaid_c4(
    session: AsyncSession,
    scenario_id: UUID,
    entity_level: str = "container",
) -> str:
    """Render a scenario as Mermaid C4 container diagram."""
    scenario = (
        await session.execute(select(TwinScenario).where(TwinScenario.id == scenario_id))
    ).scalar_one()
    graph = await get_full_scenario_graph(
        session=session,
        scenario_id=scenario_id,
        layer=None,
        projection=GraphProjection.ARCHITECTURE,
        entity_level=entity_level,
    )
    nodes = graph["nodes"]
    edges = graph["edges"]

    lines = ["C4Container", f'title "{scenario.name}"']
    lines.append('System_Boundary(system_boundary, "System") {')
    for node in nodes:
        node_id = _safe_id(str(node["id"]))
        kind = str(node.get("kind") or "container")
        natural_key = str(node.get("natural_key") or "")
        meta = node.get("meta") or {}
        label = str(node.get("name") or natural_key)
        description = f"{kind} | members={meta.get('member_count', 0)}"
        lines.append(f'  Container({node_id}, "{label}", "{kind}", "{description}")')
    lines.append("}")

    for edge in edges:
        src = _safe_id(str(edge["source_node_id"]))
        dst = _safe_id(str(edge["target_node_id"]))
        meta = edge.get("meta") or {}
        weight = meta.get("weight", 1)
        lines.append(f'Rel({src}, {dst}, "depends_on (w={weight})")')

    return "\n".join(lines)


async def export_mermaid_asis_tobe(
    session: AsyncSession,
    as_is_scenario_id: UUID,
    to_be_scenario_id: UUID,
) -> tuple[str, str]:
    """Render two Mermaid C4 artifacts (AS-IS and TO-BE)."""
    as_is = await export_mermaid_c4(session, as_is_scenario_id)
    to_be = await export_mermaid_c4(session, to_be_scenario_id)
    return as_is, to_be
