"""Mermaid C4 export for AS-IS/TO-BE twin scenarios."""

from __future__ import annotations

from uuid import UUID

from contextmine_core.models import TwinNode, TwinScenario
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


def _safe_id(value: str) -> str:
    return value.replace("-", "_").replace(":", "_").replace("/", "_")


async def export_mermaid_c4(
    session: AsyncSession,
    scenario_id: UUID,
) -> str:
    """Render a scenario as Mermaid C4 container diagram."""
    scenario = (
        await session.execute(select(TwinScenario).where(TwinScenario.id == scenario_id))
    ).scalar_one()
    nodes = (
        (await session.execute(select(TwinNode).where(TwinNode.scenario_id == scenario_id)))
        .scalars()
        .all()
    )

    lines = ["C4Container", f'title "{scenario.name}"']

    for node in nodes:
        node_id = _safe_id(str(node.id))
        tech = node.kind
        lines.append(f'Container({node_id}, "{node.name}", "{tech}", "{node.natural_key}")')

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
