"""Static user-flow synthesizer and graph materializer."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

from contextmine_core.analyzer.extractors.tests import TestsExtraction
from contextmine_core.analyzer.extractors.traceability import (
    build_endpoint_symbol_index,
    symbol_token_variants,
)
from contextmine_core.analyzer.extractors.ui import UIExtraction
from contextmine_core.models import (
    KnowledgeEdge,
    KnowledgeEdgeKind,
    KnowledgeNode,
    KnowledgeNodeKind,
)
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]


def _provenance(
    *,
    mode: str,
    extractor: str,
    confidence: float,
    evidence_ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "provenance": {
            "mode": mode,
            "extractor": extractor,
            "confidence": round(max(0.0, min(confidence, 1.0)), 4),
            "evidence_ids": list(dict.fromkeys(evidence_ids or [])),
        }
    }


@dataclass
class FlowStepDef:
    """One step in a synthesized user flow."""

    name: str
    order: int
    endpoint_path: str | None = None
    symbol_hints: list[str] = field(default_factory=list)
    test_case_refs: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    natural_key: str = ""


@dataclass
class UserFlowDef:
    """Synthesized user flow."""

    name: str
    route_path: str
    evidence_ids: list[str] = field(default_factory=list)
    steps: list[FlowStepDef] = field(default_factory=list)
    natural_key: str = ""


@dataclass
class FlowSynthesis:
    """Synthesis output."""

    flows: list[UserFlowDef] = field(default_factory=list)


def synthesize_user_flows(
    ui_extractions: list[UIExtraction],
    test_extractions: list[TestsExtraction],
) -> FlowSynthesis:
    """Build deterministic user flows from explicit UI and test symbol evidence."""
    route_to_symbol_hints: dict[str, list[str]] = {}
    route_to_navigation_hints: dict[str, list[str]] = {}
    symbol_to_test_refs: dict[str, list[str]] = {}

    for ui in ui_extractions:
        if not ui.routes:
            continue
        views_by_name = {view.name: view for view in ui.views}
        default_view_name = ui.views[0].name if len(ui.views) == 1 else None
        for route in ui.routes:
            route_to_symbol_hints.setdefault(route.path, [])
            route_to_navigation_hints.setdefault(route.path, [])

            candidate_view_names: list[str] = []
            if route.view_name_hint:
                candidate_view_names.append(route.view_name_hint)
            elif default_view_name:
                candidate_view_names.append(default_view_name)

            for view_name in candidate_view_names:
                view = views_by_name.get(view_name)
                if view is None:
                    continue
                route_to_symbol_hints[route.path].extend(view.symbol_hints)
                route_to_navigation_hints[route.path].extend(view.navigation_targets)

    for test_file in test_extractions:
        for case in test_file.cases:
            for symbol_hint in case.symbol_hints:
                symbol_to_test_refs.setdefault(symbol_hint.lower(), [])
                symbol_to_test_refs[symbol_hint.lower()].append(case.natural_key)

    synthesis = FlowSynthesis()
    for route, symbol_hints in sorted(route_to_symbol_hints.items()):
        deduped_hints = list(dict.fromkeys(symbol_hints))
        deduped_navigation = list(dict.fromkeys(route_to_navigation_hints.get(route, [])))
        flow_name = f"Flow {route}"
        flow = UserFlowDef(
            name=flow_name,
            route_path=route,
            natural_key=f"user_flow:{route}:{_hash(route)}",
        )
        flow.steps.append(
            FlowStepDef(
                name=f"Open {route}",
                order=1,
                endpoint_path=None,
                natural_key=f"flow_step:{flow.natural_key}:1",
            )
        )

        next_order = 2
        for symbol in deduped_hints:
            flow.steps.append(
                FlowStepDef(
                    name=f"Invoke {symbol}",
                    order=next_order,
                    endpoint_path=None,
                    symbol_hints=[symbol],
                    test_case_refs=symbol_to_test_refs.get(symbol.lower(), []),
                    natural_key=f"flow_step:{flow.natural_key}:{next_order}:{_hash(symbol)}",
                )
            )
            next_order += 1

        for target in deduped_navigation:
            flow.steps.append(
                FlowStepDef(
                    name=f"Navigate to {target}",
                    order=next_order,
                    endpoint_path=None,
                    natural_key=f"flow_step:{flow.natural_key}:{next_order}:{_hash(target)}",
                )
            )
            next_order += 1

        synthesis.flows.append(flow)

    return synthesis


async def _upsert_node(
    session: AsyncSession,
    *,
    collection_id: UUID,
    kind: KnowledgeNodeKind,
    natural_key: str,
    name: str,
    meta: dict[str, Any],
) -> UUID:
    stmt = pg_insert(KnowledgeNode).values(
        collection_id=collection_id,
        kind=kind,
        natural_key=natural_key,
        name=name,
        meta=meta,
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_knowledge_node_natural",
        set_={"name": stmt.excluded.name, "meta": stmt.excluded.meta},
    ).returning(KnowledgeNode.id)
    return (await session.execute(stmt)).scalar_one()


async def _upsert_edge(
    session: AsyncSession,
    *,
    collection_id: UUID,
    source_node_id: UUID,
    target_node_id: UUID,
    kind: KnowledgeEdgeKind,
    meta: dict[str, Any],
) -> UUID:
    stmt = pg_insert(KnowledgeEdge).values(
        collection_id=collection_id,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        kind=kind,
        meta=meta,
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_knowledge_edge_unique",
        set_={"meta": stmt.excluded.meta},
    ).returning(KnowledgeEdge.id)
    return (await session.execute(stmt)).scalar_one()


async def build_flows_graph(
    session: AsyncSession,
    collection_id: UUID,
    synthesis: FlowSynthesis,
    *,
    source_id: UUID | None = None,
) -> dict[str, int]:
    """Persist user-flow nodes and edges."""
    del source_id
    stats = {
        "user_flows": 0,
        "flow_steps": 0,
        "flow_edges": 0,
        "flow_endpoint_edges": 0,
        "flow_test_edges": 0,
    }
    if not synthesis.flows:
        return stats

    endpoint_symbol_index = await build_endpoint_symbol_index(
        session=session,
        collection_id=collection_id,
    )

    test_case_rows = (
        (
            await session.execute(
                select(KnowledgeNode).where(
                    KnowledgeNode.collection_id == collection_id,
                    KnowledgeNode.kind == KnowledgeNodeKind.TEST_CASE,
                )
            )
        )
        .scalars()
        .all()
    )
    test_case_by_key = {row.natural_key: row.id for row in test_case_rows}

    for flow in synthesis.flows:
        flow_meta = {
            "route_path": flow.route_path,
            **_provenance(mode="inferred", extractor="flows.v1", confidence=0.8),
        }
        flow_id = await _upsert_node(
            session,
            collection_id=collection_id,
            kind=KnowledgeNodeKind.USER_FLOW,
            natural_key=flow.natural_key,
            name=flow.name,
            meta=flow_meta,
        )
        stats["user_flows"] += 1

        for step in flow.steps:
            step_confidence = 0.78 if step.symbol_hints else 0.84
            step_meta = {
                "order": step.order,
                "endpoint_path": step.endpoint_path,
                "symbol_hints": step.symbol_hints,
                "test_case_refs": step.test_case_refs,
                **_provenance(
                    mode="inferred" if step.symbol_hints else "deterministic",
                    extractor="flows.v1",
                    confidence=step_confidence,
                    evidence_ids=step.evidence_ids,
                ),
            }
            step_id = await _upsert_node(
                session,
                collection_id=collection_id,
                kind=KnowledgeNodeKind.FLOW_STEP,
                natural_key=step.natural_key,
                name=step.name,
                meta=step_meta,
            )
            stats["flow_steps"] += 1

            await _upsert_edge(
                session,
                collection_id=collection_id,
                source_node_id=flow_id,
                target_node_id=step_id,
                kind=KnowledgeEdgeKind.USER_FLOW_HAS_STEP,
                meta=_provenance(
                    mode="deterministic",
                    extractor="flows.v1",
                    confidence=0.9,
                ),
            )
            stats["flow_edges"] += 1

            endpoint_ids: set[UUID] = set()
            for symbol_hint in step.symbol_hints:
                for token in symbol_token_variants(symbol_hint):
                    endpoint_ids.update(endpoint_symbol_index.get(token, set()))
            for endpoint_id in endpoint_ids:
                await _upsert_edge(
                    session,
                    collection_id=collection_id,
                    source_node_id=step_id,
                    target_node_id=endpoint_id,
                    kind=KnowledgeEdgeKind.FLOW_STEP_CALLS_ENDPOINT,
                    meta=_provenance(
                        mode="inferred",
                        extractor="flows.v1.symbol_trace",
                        confidence=0.78,
                    ),
                )
                stats["flow_endpoint_edges"] += 1

            for test_case_ref in step.test_case_refs:
                test_case_id = test_case_by_key.get(test_case_ref)
                if not test_case_id:
                    continue
                await _upsert_edge(
                    session,
                    collection_id=collection_id,
                    source_node_id=test_case_id,
                    target_node_id=flow_id,
                    kind=KnowledgeEdgeKind.TEST_CASE_VERIFIES_FLOW,
                    meta=_provenance(
                        mode="inferred",
                        extractor="flows.v1",
                        confidence=0.7,
                    ),
                )
                stats["flow_test_edges"] += 1

    return stats
