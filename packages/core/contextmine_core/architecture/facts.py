"""Architecture fact extraction from twin and knowledge graph data."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import PurePosixPath
from typing import Any
from uuid import UUID

from contextmine_core.exports import export_mermaid_c4_result
from contextmine_core.models import (
    KnowledgeEdge,
    KnowledgeEvidence,
    KnowledgeNode,
    KnowledgeNodeEvidence,
    KnowledgeNodeKind,
    MetricSnapshot,
    TwinScenario,
)
from contextmine_core.twin import GraphProjection, get_full_scenario_graph
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .schemas import ArchitectureFact, ArchitectureFactsBundle, EvidenceRef, PortAdapterFact

DETERMINISTIC_CONFIDENCE = 0.9
HYBRID_CONFIDENCE = 0.75
LLM_ONLY_CONFIDENCE = 0.55

INBOUND_PORT_KINDS = {
    KnowledgeNodeKind.API_ENDPOINT,
    KnowledgeNodeKind.GRAPHQL_OPERATION,
    KnowledgeNodeKind.SERVICE_RPC,
}
OUTBOUND_TARGET_KINDS = {
    KnowledgeNodeKind.DB_TABLE,
    KnowledgeNodeKind.MESSAGE_SCHEMA,
    KnowledgeNodeKind.SERVICE_RPC,
    KnowledgeNodeKind.API_ENDPOINT,
    KnowledgeNodeKind.GRAPHQL_OPERATION,
}

INBOUND_PROTOCOL_BY_KIND: dict[KnowledgeNodeKind, str] = {
    KnowledgeNodeKind.API_ENDPOINT: "http",
    KnowledgeNodeKind.GRAPHQL_OPERATION: "graphql",
    KnowledgeNodeKind.SERVICE_RPC: "rpc",
}
OUTBOUND_PROTOCOL_BY_KIND: dict[KnowledgeNodeKind, str] = {
    KnowledgeNodeKind.DB_TABLE: "sql",
    KnowledgeNodeKind.MESSAGE_SCHEMA: "message",
    KnowledgeNodeKind.SERVICE_RPC: "rpc",
    KnowledgeNodeKind.API_ENDPOINT: "http",
    KnowledgeNodeKind.GRAPHQL_OPERATION: "graphql",
}


def _canonical_file_path(path: str | None, meta: dict[str, Any] | None = None) -> str | None:
    if path:
        cleaned = str(path).strip()
        if cleaned:
            return cleaned
    meta = meta or {}
    file_path = meta.get("file_path")
    if isinstance(file_path, str) and file_path.strip():
        return file_path.strip()
    return None


def _derive_arch_group(path: str | None, meta: dict[str, Any]) -> tuple[str, str, str] | None:
    architecture_meta = meta.get("architecture")
    if isinstance(architecture_meta, dict):
        explicit_domain = str(architecture_meta.get("domain") or "").strip()
        explicit_container = str(architecture_meta.get("container") or "").strip()
        explicit_component = str(architecture_meta.get("component") or "").strip()
        if explicit_domain and explicit_container:
            component = explicit_component or explicit_container
            return explicit_domain, explicit_container, component

    if not path:
        return None

    normalized = path.strip("/")
    parts = [part for part in normalized.split("/") if part]
    if not parts:
        return None

    if parts[0] == "services" and len(parts) >= 3:
        domain = parts[1]
        container = parts[2]
    elif parts[0] == "apps" and len(parts) >= 2:
        domain = parts[1]
        container = parts[1]
    else:
        domain = parts[0]
        container = parts[1] if len(parts) > 1 else parts[0]

    component = PurePosixPath(normalized).stem or container
    return domain, container, component


def _evidence_from_symbol_meta(node: KnowledgeNode) -> tuple[EvidenceRef, ...]:
    meta = node.meta or {}
    file_path = _canonical_file_path(None, meta)
    if not file_path:
        return ()
    start_line = meta.get("start_line")
    end_line = meta.get("end_line")
    return (
        EvidenceRef(
            kind="file",
            ref=file_path,
            start_line=int(start_line) if isinstance(start_line, int) else None,
            end_line=int(end_line) if isinstance(end_line, int) else None,
        ),
    )


async def _load_node_evidence(
    session: AsyncSession,
    node_ids: set[UUID],
) -> dict[UUID, tuple[EvidenceRef, ...]]:
    if not node_ids:
        return {}

    rows = await session.execute(
        select(
            KnowledgeNodeEvidence.node_id,
            KnowledgeEvidence.file_path,
            KnowledgeEvidence.start_line,
            KnowledgeEvidence.end_line,
        )
        .join(KnowledgeEvidence, KnowledgeEvidence.id == KnowledgeNodeEvidence.evidence_id)
        .where(KnowledgeNodeEvidence.node_id.in_(node_ids))
        .order_by(KnowledgeEvidence.created_at.desc())
    )

    evidence_by_node: dict[UUID, list[EvidenceRef]] = defaultdict(list)
    for node_id, file_path, start_line, end_line in rows.all():
        evidence_by_node[node_id].append(
            EvidenceRef(
                kind="file",
                ref=file_path,
                start_line=start_line,
                end_line=end_line,
            )
        )

    return {node_id: tuple(items[:3]) for node_id, items in evidence_by_node.items()}


def _metric_averages(metrics: list[MetricSnapshot]) -> dict[str, float | None]:
    if not metrics:
        return {
            "coverage_avg": None,
            "complexity_avg": None,
            "coupling_avg": None,
            "change_frequency_avg": None,
        }

    total = float(len(metrics))
    return {
        "coverage_avg": round(sum(float(m.coverage or 0.0) for m in metrics) / total, 4),
        "complexity_avg": round(sum(float(m.complexity or 0.0) for m in metrics) / total, 4),
        "coupling_avg": round(sum(float(m.coupling or 0.0) for m in metrics) / total, 4),
        "change_frequency_avg": round(
            sum(float(m.change_frequency or 0.0) for m in metrics) / total,
            4,
        ),
    }


def _extract_inbound_ports(
    nodes: list[KnowledgeNode],
    evidence_by_node: dict[UUID, tuple[EvidenceRef, ...]],
) -> list[PortAdapterFact]:
    facts: list[PortAdapterFact] = []

    for node in sorted(nodes, key=lambda row: (row.kind.value, row.natural_key)):
        if node.kind not in INBOUND_PORT_KINDS:
            continue

        evidence = evidence_by_node.get(node.id, ())
        file_path = evidence[0].ref if evidence else None
        group = _derive_arch_group(file_path, node.meta or {}) if file_path else None
        container = group[1] if group else None
        component = group[2] if group else None
        adapter_name = PurePosixPath(file_path).stem if file_path else None

        source = "deterministic" if evidence else "hybrid"
        confidence = DETERMINISTIC_CONFIDENCE if evidence else HYBRID_CONFIDENCE

        facts.append(
            PortAdapterFact(
                fact_id=f"inbound:{node.kind.value}:{node.natural_key}",
                direction="inbound",
                port_name=node.name,
                adapter_name=adapter_name,
                container=container,
                component=component,
                protocol=INBOUND_PROTOCOL_BY_KIND.get(node.kind),
                source=source,
                confidence=confidence,
                attributes={
                    "node_kind": node.kind.value,
                    "natural_key": node.natural_key,
                    "path": (node.meta or {}).get("path"),
                    "method": (node.meta or {}).get("method"),
                },
                evidence=evidence,
            )
        )

    return facts


def _extract_outbound_ports(
    symbol_nodes: dict[UUID, KnowledgeNode],
    target_nodes: dict[UUID, KnowledgeNode],
    edges: list[KnowledgeEdge],
) -> list[PortAdapterFact]:
    facts: list[PortAdapterFact] = []

    for edge in edges:
        source_symbol = symbol_nodes.get(edge.source_node_id)
        target_node = target_nodes.get(edge.target_node_id)
        if not source_symbol or not target_node:
            continue

        file_path = _canonical_file_path(None, source_symbol.meta or {})
        group = _derive_arch_group(file_path, source_symbol.meta or {}) if file_path else None
        container = group[1] if group else None
        component = group[2] if group else None

        source = "deterministic" if group else "hybrid"
        confidence = DETERMINISTIC_CONFIDENCE if group else HYBRID_CONFIDENCE

        facts.append(
            PortAdapterFact(
                fact_id=(
                    f"outbound:{source_symbol.natural_key}:{target_node.natural_key}:{edge.kind.value}"
                ),
                direction="outbound",
                port_name=target_node.name,
                adapter_name=source_symbol.name,
                container=container,
                component=component,
                protocol=OUTBOUND_PROTOCOL_BY_KIND.get(target_node.kind),
                source=source,
                confidence=confidence,
                attributes={
                    "edge_kind": edge.kind.value,
                    "target_kind": target_node.kind.value,
                    "target_natural_key": target_node.natural_key,
                    "source_natural_key": source_symbol.natural_key,
                },
                evidence=_evidence_from_symbol_meta(source_symbol),
            )
        )

    return facts


def _dedupe_ports(facts: list[PortAdapterFact]) -> list[PortAdapterFact]:
    by_id: dict[str, PortAdapterFact] = {}
    for fact in facts:
        existing = by_id.get(fact.fact_id)
        if existing is None:
            by_id[fact.fact_id] = fact
            continue
        if float(fact.confidence) > float(existing.confidence):
            by_id[fact.fact_id] = fact
            continue
        if len(fact.evidence) > len(existing.evidence):
            by_id[fact.fact_id] = fact
    return sorted(by_id.values(), key=lambda row: row.fact_id)


async def build_architecture_facts(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID,
    enable_llm_enrich: bool = False,
    llm_provider: Any | None = None,
) -> ArchitectureFactsBundle:
    """Build architecture facts from twin, metrics, and knowledge-graph signals."""

    scenario = (
        await session.execute(
            select(TwinScenario).where(
                TwinScenario.id == scenario_id,
                TwinScenario.collection_id == collection_id,
            )
        )
    ).scalar_one_or_none()
    if not scenario:
        raise ValueError("Scenario not found in collection")

    bundle = ArchitectureFactsBundle(
        collection_id=collection_id,
        scenario_id=scenario_id,
        scenario_name=scenario.name,
    )

    if enable_llm_enrich and llm_provider is None:
        bundle.warnings.append(
            "ARCH_DOCS_LLM_ENRICH is enabled but no LLM provider is available; using deterministic fallback."
        )

    container_graph = await get_full_scenario_graph(
        session=session,
        scenario_id=scenario_id,
        layer=None,
        projection=GraphProjection.ARCHITECTURE,
        entity_level="container",
        include_kinds={"file"},
    )
    component_graph = await get_full_scenario_graph(
        session=session,
        scenario_id=scenario_id,
        layer=None,
        projection=GraphProjection.ARCHITECTURE,
        entity_level="component",
        include_kinds={"file"},
    )

    for node in container_graph["nodes"]:
        meta = node.get("meta") or {}
        bundle.facts.append(
            ArchitectureFact(
                fact_id=f"container:{node.get('natural_key')}",
                fact_type="container",
                title=f"Container {node.get('name')}",
                description=(
                    f"Container '{node.get('name')}' with {meta.get('member_count', 0)} members"
                ),
                source="deterministic",
                confidence=DETERMINISTIC_CONFIDENCE,
                tags=("c4", "container"),
                attributes={
                    "natural_key": node.get("natural_key"),
                    "domain": meta.get("domain"),
                    "container": meta.get("container"),
                    "member_count": meta.get("member_count", 0),
                    "grouping_strategy": container_graph.get("grouping_strategy"),
                },
                evidence=(),
            )
        )

    for node in component_graph["nodes"]:
        meta = node.get("meta") or {}
        component_name = str(meta.get("component") or node.get("name") or "component")
        bundle.facts.append(
            ArchitectureFact(
                fact_id=f"component:{component_name}",
                fact_type="component",
                title=f"Component {component_name}",
                description=(
                    f"Component '{component_name}' in container '{meta.get('container')}'"
                ),
                source="deterministic",
                confidence=DETERMINISTIC_CONFIDENCE,
                tags=("c4", "component"),
                attributes={
                    "natural_key": node.get("natural_key"),
                    "domain": meta.get("domain"),
                    "container": meta.get("container"),
                    "component": component_name,
                    "member_count": meta.get("member_count", 0),
                },
                evidence=(),
            )
        )

    for edge in component_graph["edges"]:
        source_id = str(edge.get("source_node_id") or "")
        target_id = str(edge.get("target_node_id") or "")
        weight = (edge.get("meta") or {}).get("weight", 1)
        bundle.facts.append(
            ArchitectureFact(
                fact_id=f"component_dep:{source_id}:{target_id}",
                fact_type="component_dependency",
                title="Component dependency",
                description=f"Component dependency {source_id} -> {target_id}",
                source="deterministic",
                confidence=DETERMINISTIC_CONFIDENCE,
                tags=("c4", "dependency"),
                attributes={
                    "source_node_id": source_id,
                    "target_node_id": target_id,
                    "weight": weight,
                    "sample_edge_kinds": (edge.get("meta") or {}).get("sample_edge_kinds", []),
                },
                evidence=(EvidenceRef(kind="edge", ref=str(edge.get("id"))),),
            )
        )

    context_view = await export_mermaid_c4_result(
        session,
        scenario_id,
        c4_view="context",
    )
    deployment_view = await export_mermaid_c4_result(
        session,
        scenario_id,
        c4_view="deployment",
    )

    if context_view.warnings:
        bundle.warnings.extend([f"c4_context: {warning}" for warning in context_view.warnings])
    if deployment_view.warnings:
        bundle.warnings.extend(
            [f"c4_deployment: {warning}" for warning in deployment_view.warnings]
        )

    bundle.facts.append(
        ArchitectureFact(
            fact_id=f"c4_context:{scenario_id}",
            fact_type="c4_context",
            title="C4 context view",
            description="Rendered C4 context view for system-level boundaries",
            source="deterministic" if not context_view.warnings else "hybrid",
            confidence=(
                DETERMINISTIC_CONFIDENCE if not context_view.warnings else HYBRID_CONFIDENCE
            ),
            tags=("c4", "context"),
            attributes={
                "warnings": list(context_view.warnings),
                "mermaid": context_view.content,
            },
            evidence=(),
        )
    )
    bundle.facts.append(
        ArchitectureFact(
            fact_id=f"c4_deployment:{scenario_id}",
            fact_type="c4_deployment",
            title="C4 deployment view",
            description="Rendered C4 deployment view for runtime/deployment mapping",
            source="deterministic" if not deployment_view.warnings else "hybrid",
            confidence=(
                DETERMINISTIC_CONFIDENCE if not deployment_view.warnings else HYBRID_CONFIDENCE
            ),
            tags=("c4", "deployment"),
            attributes={
                "warnings": list(deployment_view.warnings),
                "mermaid": deployment_view.content,
            },
            evidence=(),
        )
    )

    metrics = (
        (
            await session.execute(
                select(MetricSnapshot).where(MetricSnapshot.scenario_id == scenario_id)
            )
        )
        .scalars()
        .all()
    )
    averages = _metric_averages(metrics)
    bundle.facts.append(
        ArchitectureFact(
            fact_id=f"quality:{scenario_id}",
            fact_type="quality_summary",
            title="Code quality summary",
            description="Aggregated quality metrics for this scenario",
            source="deterministic",
            confidence=DETERMINISTIC_CONFIDENCE,
            tags=("quality", "metrics"),
            attributes={
                "metric_nodes": len(metrics),
                **averages,
            },
            evidence=(),
        )
    )

    kg_nodes = (
        (
            await session.execute(
                select(KnowledgeNode).where(KnowledgeNode.collection_id == collection_id)
            )
        )
        .scalars()
        .all()
    )

    inbound_nodes = [node for node in kg_nodes if node.kind in INBOUND_PORT_KINDS]
    db_tables = [node for node in kg_nodes if node.kind == KnowledgeNodeKind.DB_TABLE]
    business_rules = [node for node in kg_nodes if node.kind == KnowledgeNodeKind.BUSINESS_RULE]

    for table in sorted(db_tables, key=lambda row: row.name):
        bundle.facts.append(
            ArchitectureFact(
                fact_id=f"erm:{table.natural_key}",
                fact_type="erm_table",
                title=f"Data table {table.name}",
                description=f"Table '{table.name}' in the extracted ER model",
                source="deterministic",
                confidence=DETERMINISTIC_CONFIDENCE,
                tags=("erm", "data"),
                attributes={
                    "natural_key": table.natural_key,
                    "column_count": (table.meta or {}).get("column_count", 0),
                },
                evidence=(),
            )
        )

    bundle.facts.append(
        ArchitectureFact(
            fact_id=f"rules:{collection_id}",
            fact_type="business_rules",
            title="Business rule coverage",
            description=f"{len(business_rules)} business rules extracted from source code",
            source="deterministic",
            confidence=DETERMINISTIC_CONFIDENCE,
            tags=("rules", "domain"),
            attributes={"count": len(business_rules)},
            evidence=(),
        )
    )

    inbound_ids = {node.id for node in inbound_nodes}
    evidence_by_node = await _load_node_evidence(session, inbound_ids)

    ports = _extract_inbound_ports(inbound_nodes, evidence_by_node)

    symbol_nodes = {node.id: node for node in kg_nodes if node.kind == KnowledgeNodeKind.SYMBOL}
    target_nodes = {node.id: node for node in kg_nodes if node.kind in OUTBOUND_TARGET_KINDS}

    kg_edges = (
        (
            await session.execute(
                select(KnowledgeEdge).where(KnowledgeEdge.collection_id == collection_id)
            )
        )
        .scalars()
        .all()
    )
    outbound_edges = [
        edge
        for edge in kg_edges
        if edge.source_node_id in symbol_nodes and edge.target_node_id in target_nodes
    ]
    ports.extend(_extract_outbound_ports(symbol_nodes, target_nodes, outbound_edges))

    if enable_llm_enrich and llm_provider is not None and ports:
        # Keep V1 deterministic by default, but mark low-confidence unresolved entries
        # as LLM-enrichable candidates for downstream refinement.
        enriched: list[PortAdapterFact] = []
        for row in ports:
            if row.container or row.component:
                enriched.append(row)
                continue
            enriched.append(
                replace(
                    row,
                    source="llm",
                    confidence=LLM_ONLY_CONFIDENCE,
                    attributes={**row.attributes, "llm_enrichment": "required"},
                )
            )
        ports = enriched

    bundle.ports_adapters = _dedupe_ports(ports)

    return bundle
