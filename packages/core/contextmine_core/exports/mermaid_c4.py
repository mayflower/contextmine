"""Mermaid C4 export for AS-IS/TO-BE twin scenarios."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from contextmine_core.architecture.recovery import recover_architecture_model
from contextmine_core.architecture.recovery_docs import load_recovery_docs
from contextmine_core.architecture.recovery_model import RecoveredArchitectureModel
from contextmine_core.models import (
    KnowledgeEdge,
    KnowledgeEvidence,
    KnowledgeNode,
    KnowledgeNodeEvidence,
    KnowledgeNodeKind,
    TwinScenario,
)
from contextmine_core.twin import (
    GraphProjection,
    get_full_scenario_graph,
    get_scenario_provenance_node_ids,
)
from contextmine_core.twin.grouping import canonical_file_path_from_node, derive_arch_group
from contextmine_core.twin.projections import build_inferred_architecture_projection
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

SUPPORTED_C4_VIEWS = {"context", "container", "component", "code", "deployment"}
DEFAULT_C4_VIEW = "container"
DEFAULT_MAX_NODES = 120


@dataclass(frozen=True)
class C4ExportResult:
    """Rendered C4 content with metadata for UI/API consumers."""

    content: str
    c4_view: str
    c4_scope: str | None
    warnings: list[str]


def _safe_id(value: str) -> str:
    return value.replace("-", "_").replace(":", "_").replace("/", "_")


def _safe_text(value: str | None, fallback: str = "") -> str:
    text = str(value or fallback).strip()
    if not text:
        text = fallback
    return text.replace('"', "'").replace("\n", " ").strip()


_canonical_file_path = canonical_file_path_from_node
_derive_arch_group = derive_arch_group


def _apply_scope_filter(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    c4_scope: str,
    match_fields: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    """Filter nodes/edges by scope. Returns (nodes, edges, matched)."""
    scope = c4_scope.lower()
    scoped_ids: set[str] = set()
    for node in nodes:
        match_values = {str(node.get("name") or "").lower()}
        meta = node.get("meta") or {}
        for field in match_fields:
            if field != "name":
                match_values.add(str(meta.get(field) or "").lower())
        if scope in match_values:
            scoped_ids.add(str(node["id"]))
    if not scoped_ids:
        return nodes, edges, False
    filtered_nodes = [node for node in nodes if str(node.get("id")) in scoped_ids]
    filtered_edges = [
        edge
        for edge in edges
        if str(edge.get("source_node_id")) in scoped_ids
        and str(edge.get("target_node_id")) in scoped_ids
    ]
    return filtered_nodes, filtered_edges, True


def _limit_nodes_by_degree(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    max_nodes: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    if max_nodes <= 0 or len(nodes) <= max_nodes:
        return nodes, edges, False

    degree: dict[str, int] = defaultdict(int)
    for edge in edges:
        src = str(edge.get("source_node_id"))
        dst = str(edge.get("target_node_id"))
        degree[src] += 1
        degree[dst] += 1

    ranked_ids = [
        str(node.get("id"))
        for node in sorted(
            nodes,
            key=lambda node: (
                degree.get(str(node.get("id")), 0),
                str(node.get("name") or ""),
            ),
            reverse=True,
        )
    ]
    keep_ids = set(ranked_ids[:max_nodes])

    limited_nodes = [node for node in nodes if str(node.get("id")) in keep_ids]
    limited_edges = [
        edge
        for edge in edges
        if str(edge.get("source_node_id")) in keep_ids
        and str(edge.get("target_node_id")) in keep_ids
    ]
    return limited_nodes, limited_edges, True


def _build_relation_lines(edges: list[dict[str, Any]], label_key: str = "kind") -> list[str]:
    lines: list[str] = []
    for edge in edges:
        src = _safe_id(str(edge.get("source_node_id") or ""))
        dst = _safe_id(str(edge.get("target_node_id") or ""))
        if not src or not dst:
            continue
        meta = edge.get("meta") or {}
        weight = meta.get("weight")
        base_label = str(edge.get(label_key) or edge.get("kind") or "depends_on")
        label = f"{base_label} (w={weight})" if weight is not None else base_label
        lines.append(f'Rel({src}, {dst}, "{_safe_text(label, "depends_on")}")')
    return lines


def _normalize_c4_view(value: str | None) -> str:
    candidate = str(value or DEFAULT_C4_VIEW).strip().lower()
    if candidate in SUPPORTED_C4_VIEWS:
        return candidate
    return DEFAULT_C4_VIEW


def _normalize_scope(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _kind_value(kind: Any) -> str:
    if hasattr(kind, "value"):
        return str(kind.value)
    return str(kind)


def _kg_nodes_to_recovery_inputs(
    nodes: list[KnowledgeNode],
) -> tuple[list[dict[str, Any]], dict[UUID, str]]:
    subject_ref_by_id: dict[UUID, str] = {}
    recovery_nodes: list[dict[str, Any]] = []
    for node in nodes:
        subject_ref = str(node.natural_key or node.id)
        subject_ref_by_id[node.id] = subject_ref
        recovery_nodes.append(
            {
                "id": subject_ref,
                "kind": node.kind,
                "name": node.name,
                "natural_key": node.natural_key,
                "meta": node.meta or {},
            }
        )
    return recovery_nodes, subject_ref_by_id


def _kg_edges_to_recovery_inputs(
    edges: list[KnowledgeEdge],
    subject_ref_by_id: dict[UUID, str],
) -> list[dict[str, Any]]:
    recovery_edges: list[dict[str, Any]] = []
    for edge in edges:
        source_ref = subject_ref_by_id.get(edge.source_node_id)
        target_ref = subject_ref_by_id.get(edge.target_node_id)
        if not source_ref or not target_ref:
            continue
        recovery_edges.append(
            {
                "source_node_id": source_ref,
                "target_node_id": target_ref,
                "kind": edge.kind,
                "meta": {},
            }
        )
    return recovery_edges


async def _load_recovered_architecture_model(
    session: AsyncSession,
    scenario_id: UUID,
    collection_id: UUID,
) -> RecoveredArchitectureModel:
    scenario_knowledge_node_ids = await get_scenario_provenance_node_ids(session, scenario_id)
    if not scenario_knowledge_node_ids:
        return RecoveredArchitectureModel()

    kg_nodes = (
        (
            await session.execute(
                select(KnowledgeNode).where(
                    KnowledgeNode.collection_id == collection_id,
                    KnowledgeNode.id.in_(scenario_knowledge_node_ids),
                )
            )
        )
        .scalars()
        .all()
    )
    kg_edges = (
        (
            await session.execute(
                select(KnowledgeEdge).where(
                    KnowledgeEdge.collection_id == collection_id,
                    KnowledgeEdge.source_node_id.in_(scenario_knowledge_node_ids),
                    KnowledgeEdge.target_node_id.in_(scenario_knowledge_node_ids),
                )
            )
        )
        .scalars()
        .all()
    )

    recovery_nodes, subject_ref_by_id = _kg_nodes_to_recovery_inputs(kg_nodes)
    recovery_edges = _kg_edges_to_recovery_inputs(kg_edges, subject_ref_by_id)
    recovery_docs = await load_recovery_docs(session, kg_nodes)
    return recover_architecture_model(recovery_nodes, recovery_edges, docs=recovery_docs)


def _append_recovery_diagnostics(
    warnings: list[str],
    model: RecoveredArchitectureModel,
) -> None:
    ambiguous_count = sum(1 for item in model.hypotheses if item.status == "ambiguous")
    unresolved_count = sum(1 for item in model.hypotheses if item.status == "unresolved")
    if ambiguous_count:
        warnings.append(f"Ambiguous recovered memberships: {ambiguous_count}.")
    if unresolved_count:
        warnings.append(f"Unresolved recovered memberships: {unresolved_count}.")
    warnings.extend(model.warnings)


def _node_sort_key(node: dict[str, Any]) -> tuple[str, str]:
    return (str(node.get("kind") or ""), str(node.get("name") or node.get("id") or ""))


def _render_recovered_container_node(node: dict[str, Any]) -> str:
    node_id = _safe_id(str(node.get("id") or ""))
    kind = _safe_text(str(node.get("kind") or "container"), "container")
    label = _safe_text(str(node.get("name") or kind), kind)
    confidence = (node.get("meta") or {}).get("confidence")
    description = f"{kind} | confidence={confidence}" if confidence is not None else kind
    return f'  Container({node_id}, "{label}", "{kind}", "{_safe_text(description, kind)}")'


def _render_recovered_supporting_node(node: dict[str, Any]) -> str:
    node_id = _safe_id(str(node.get("id") or ""))
    kind = _safe_text(str(node.get("kind") or "dependency"), "dependency")
    label = _safe_text(str(node.get("name") or kind), kind)
    if kind == "external_system":
        return f'System_Ext({node_id}, "{label}", "{kind}")'
    return f'Container({node_id}, "{label}", "{kind}", "{kind}")'


async def _render_container_view(
    session: AsyncSession,
    scenario_id: UUID,
    collection_id: UUID,
    scenario_name: str,
    c4_scope: str | None,
) -> C4ExportResult:
    warnings: list[str] = []
    recovered_model = await _load_recovered_architecture_model(session, scenario_id, collection_id)
    _append_recovery_diagnostics(warnings, recovered_model)
    projection = build_inferred_architecture_projection(recovered_model, entity_level="container")

    nodes = list(projection["nodes"])
    edges = list(projection["edges"])
    used_recovered = bool(nodes)

    if not nodes:
        warnings.append(
            "Recovered architecture unavailable for container view; used file projection."
        )
        graph = await get_full_scenario_graph(
            session=session,
            scenario_id=scenario_id,
            layer=None,
            projection=GraphProjection.ARCHITECTURE,
            entity_level="container",
            include_kinds={"file"},
        )
        nodes = list(graph["nodes"])
        edges = list(graph["edges"])

    if c4_scope:
        match_fields = ("name", "container", "domain") if not used_recovered else ("name",)
        nodes, edges, matched = _apply_scope_filter(nodes, edges, c4_scope, match_fields)
        if not matched:
            scope_source = "recovered container" if used_recovered else "container"
            warnings.append(
                f'No {scope_source} matched scope "{c4_scope}"; rendered all containers.'
            )

    lines = ["C4Container", f'title "{_safe_text(scenario_name, "Scenario")}"']
    lines.append('System_Boundary(system_boundary, "System") {')
    internal_nodes = [node for node in nodes if str(node.get("kind")) == "container"]
    supporting_nodes = [node for node in nodes if str(node.get("kind")) != "container"]
    for node in sorted(internal_nodes, key=_node_sort_key):
        if used_recovered:
            lines.append(_render_recovered_container_node(node))
            continue
        node_id = _safe_id(str(node.get("id") or ""))
        kind = _safe_text(str(node.get("kind") or "container"), "container")
        natural_key = _safe_text(str(node.get("natural_key") or ""), "")
        meta = node.get("meta") or {}
        label = _safe_text(str(node.get("name") or natural_key), natural_key or "container")
        description = _safe_text(f"{kind} | members={meta.get('member_count', 0)}", kind)
        lines.append(f'  Container({node_id}, "{label}", "{kind}", "{description}")')
    lines.append("}")
    if used_recovered:
        for node in sorted(supporting_nodes, key=_node_sort_key):
            lines.append(_render_recovered_supporting_node(node))
    lines.extend(_build_relation_lines(edges))

    if not nodes:
        warnings.append("No architecture containers available; returning minimal diagram.")

    return C4ExportResult(
        content="\n".join(lines),
        c4_view="container",
        c4_scope=c4_scope,
        warnings=warnings,
    )


def _render_component_container_lines(
    container_groups: dict[str, list[dict[str, Any]]],
) -> list[str]:
    """Render Mermaid C4 lines for component containers."""
    lines: list[str] = []
    for container_name in sorted(container_groups):
        boundary_id = _safe_id(f"container:{container_name}")
        lines.append(
            f'  Container_Boundary({boundary_id}, "{_safe_text(container_name, "container")}") {{'
        )
        for node in sorted(
            container_groups[container_name], key=lambda item: str(item.get("name") or "")
        ):
            node_id = _safe_id(str(node.get("id") or ""))
            kind = _safe_text(str(node.get("kind") or "component"), "component")
            label = _safe_text(str(node.get("name") or "component"), "component")
            members = int((node.get("meta") or {}).get("member_count", 0) or 0)
            description = _safe_text(f"members={members}", "")
            lines.append(f'    Component({node_id}, "{label}", "{kind}", "{description}")')
        lines.append("  }")
    return lines


async def _render_component_view(
    session: AsyncSession,
    scenario_id: UUID,
    collection_id: UUID,
    scenario_name: str,
    c4_scope: str | None,
) -> C4ExportResult:
    warnings: list[str] = []
    recovered_model = await _load_recovered_architecture_model(session, scenario_id, collection_id)
    _append_recovery_diagnostics(warnings, recovered_model)
    projection = build_inferred_architecture_projection(recovered_model, entity_level="component")

    nodes = list(projection["nodes"])
    edges = list(projection["edges"])
    used_recovered = bool(nodes)

    if not nodes:
        warnings.append(
            "Recovered architecture unavailable for component view; used file projection."
        )
        graph = await get_full_scenario_graph(
            session=session,
            scenario_id=scenario_id,
            layer=None,
            projection=GraphProjection.ARCHITECTURE,
            entity_level="component",
            include_kinds={"file"},
        )
        nodes = list(graph["nodes"])
        edges = list(graph["edges"])

    if c4_scope:
        scoped_source_nodes = list(nodes)
        scoped_source_edges = list(edges)
        nodes, edges, matched = _apply_scope_filter(
            nodes,
            edges,
            c4_scope,
            ("name", "component", "container", "container_context"),
        )
        if not matched:
            scope_source = "recovered component" if used_recovered else "component/container"
            warnings.append(
                f'No {scope_source} matched scope "{c4_scope}"; rendered all components.'
            )
        elif used_recovered:
            scoped_ids = {str(node.get("id")) for node in nodes}
            edges = [
                edge
                for edge in scoped_source_edges
                if str(edge.get("source_node_id")) in scoped_ids
                or str(edge.get("target_node_id")) in scoped_ids
            ]
            related_ids = (
                scoped_ids
                | {str(edge.get("source_node_id")) for edge in edges}
                | {str(edge.get("target_node_id")) for edge in edges}
            )
            nodes = [node for node in scoped_source_nodes if str(node.get("id")) in related_ids]

    container_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    supporting_nodes: list[dict[str, Any]] = []
    for node in nodes:
        if str(node.get("kind")) != "component":
            supporting_nodes.append(node)
            continue
        meta = node.get("meta") or {}
        container = _safe_text(
            str(meta.get("container_context") or meta.get("container") or "shared"),
            "shared",
        )
        container_groups[container].append(node)

    lines = ["C4Component", f'title "{_safe_text(scenario_name, "Scenario")}"']
    lines.append('Container_Boundary(system_container, "System") {')
    lines.extend(_render_component_container_lines(container_groups))
    lines.append("}")
    if used_recovered:
        for node in sorted(supporting_nodes, key=_node_sort_key):
            lines.append(_render_recovered_supporting_node(node))
    lines.extend(_build_relation_lines(edges))

    if not nodes:
        warnings.append("No architecture components available; returning minimal diagram.")

    return C4ExportResult(
        content="\n".join(lines),
        c4_view="component",
        c4_scope=c4_scope,
        warnings=warnings,
    )


def _build_component_path_map(
    file_graph: dict[str, Any],
) -> tuple[dict[str, set[str]], dict[str, str]]:
    """Build component-to-paths and component-to-container maps from a file graph."""
    component_to_paths: dict[str, set[str]] = defaultdict(set)
    component_to_container: dict[str, str] = {}
    for node in file_graph["nodes"]:
        file_path = _canonical_file_path(node)
        if not file_path:
            continue
        group = _derive_arch_group(file_path, node.get("meta") or {})
        if not group:
            continue
        domain, container, component = group
        key = f"{domain}/{container}/{component}"
        component_to_paths[key].add(file_path)
        component_to_container[key] = container
    return component_to_paths, component_to_container


def _select_recovered_code_scope_paths(
    model: RecoveredArchitectureModel,
    c4_scope: str | None,
    warnings: list[str],
) -> tuple[set[str], str | None]:
    projection = build_inferred_architecture_projection(model, entity_level="component")
    component_nodes = [node for node in projection["nodes"] if str(node.get("kind")) == "component"]
    if not component_nodes:
        return set(), None

    def score(node: dict[str, Any]) -> tuple[float, str]:
        meta = node.get("meta") or {}
        return (
            float(meta.get("confidence") or 0.0),
            str(node.get("id") or ""),
        )

    matches = component_nodes
    if c4_scope:
        scope = c4_scope.lower()
        matches = []
        for node in component_nodes:
            meta = node.get("meta") or {}
            values = {
                str(node.get("name") or "").lower(),
                str(node.get("id") or "").lower(),
                str(meta.get("container_context") or "").lower(),
                str(meta.get("container_id") or "").lower(),
                str(meta.get("entity_id") or "").lower(),
            }
            if scope in values:
                matches.append(node)
        if not matches:
            warnings.append(
                f'No recovered code scope matched "{c4_scope}"; falling back to file heuristics.'
            )
            return set(), None
        if len(matches) > 1:
            warnings.append(
                f'Ambiguous recovered code scope "{c4_scope}" matched {len(matches)} components; defaulted to the strongest recovered candidate.'
            )
    elif len(component_nodes) > 1:
        warnings.append(
            "No recovered code scope provided; defaulted to the strongest recovered component."
        )

    selected = sorted(matches, key=score, reverse=True)[0]
    evidence_paths = {
        str(path)
        for path in (selected.get("meta") or {}).get("evidence_summary") or []
        if isinstance(path, str) and path.strip()
    }
    if not evidence_paths:
        warnings.append(
            "Recovered code scope lacked file evidence; falling back to file heuristics."
        )
    return evidence_paths, str(selected.get("name") or "")


def _select_code_scope_component(
    component_to_paths: dict[str, set[str]],
    c4_scope: str | None,
    warnings: list[str],
) -> str | None:
    """Select the component to scope the code view to."""
    if not component_to_paths:
        return None
    selected: str | None = None
    if c4_scope:
        scope = c4_scope.lower()
        for key in component_to_paths:
            _, container, component = key.split("/", 2)
            if scope in {key.lower(), container.lower(), component.lower()}:
                selected = key
                break
        if selected is None:
            warnings.append(
                f'No component/file scope matched "{c4_scope}"; defaulted to largest component.'
            )
    if selected is None:
        selected = max(component_to_paths.items(), key=lambda item: len(item[1]))[0]
    return selected


def _filter_edges_by_node_ids(
    edges: list[dict[str, Any]],
    node_ids: set[str],
) -> list[dict[str, Any]]:
    """Filter edges to only those connecting nodes in node_ids."""
    return [
        edge
        for edge in edges
        if str(edge.get("source_node_id")) in node_ids
        and str(edge.get("target_node_id")) in node_ids
    ]


def _check_edge_kind_warnings(edges: list[dict[str, Any]], warnings: list[str]) -> None:
    """Add warnings about missing edge kinds in code view."""
    has_call_edges = any(str(edge.get("kind") or "") == "symbol_calls_symbol" for edge in edges)
    has_reference_edges = any(
        str(edge.get("kind") or "") == "symbol_references_symbol" for edge in edges
    )
    if not has_call_edges and has_reference_edges:
        warnings.append(
            "No symbol_calls_symbol edges found in scope; using references/contains relationships as fallback."
        )


def _render_code_symbol_lines(nodes: list[dict[str, Any]]) -> list[str]:
    """Render C4 Component lines for symbol nodes."""
    lines: list[str] = []
    for node in sorted(nodes, key=lambda item: str(item.get("name") or "")):
        node_id = _safe_id(str(node.get("id") or ""))
        kind = _safe_text(str(node.get("kind") or "symbol"), "symbol")
        label = _safe_text(str(node.get("name") or kind), kind)
        file_path = _safe_text(_canonical_file_path(node) or "", "")
        description = _safe_text(f"{kind} | {file_path}" if file_path else kind, kind)
        lines.append(f'    Component({node_id}, "{label}", "{kind}", "{description}")')
    return lines


async def _render_code_view(
    session: AsyncSession,
    scenario_id: UUID,
    collection_id: UUID,
    scenario_name: str,
    c4_scope: str | None,
    max_nodes: int,
) -> C4ExportResult:
    warnings: list[str] = []
    recovered_model = await _load_recovered_architecture_model(session, scenario_id, collection_id)
    _append_recovery_diagnostics(warnings, recovered_model)
    recovered_scope_paths, recovered_scope_name = _select_recovered_code_scope_paths(
        recovered_model,
        c4_scope,
        warnings,
    )

    file_graph = await get_full_scenario_graph(
        session=session,
        scenario_id=scenario_id,
        layer=None,
        projection=GraphProjection.CODE_FILE,
        entity_level="file",
    )
    component_to_paths, component_to_container = _build_component_path_map(file_graph)
    selected_component = None
    if not recovered_scope_paths:
        selected_component = _select_code_scope_component(component_to_paths, c4_scope, warnings)
    scope_paths = component_to_paths.get(selected_component, set()) if selected_component else set()
    if recovered_scope_paths:
        scope_paths = recovered_scope_paths

    symbol_graph = await get_full_scenario_graph(
        session=session,
        scenario_id=scenario_id,
        layer=None,
        projection=GraphProjection.CODE_SYMBOL,
        entity_level="symbol",
        include_edge_kinds={
            "symbol_contains_symbol",
            "symbol_references_symbol",
            "symbol_calls_symbol",
            "symbol_extends_symbol",
            "symbol_implements_symbol",
        },
    )

    nodes = list(symbol_graph["nodes"])
    if scope_paths:
        scoped_nodes = [node for node in nodes if (_canonical_file_path(node) or "") in scope_paths]
        if scoped_nodes:
            nodes = scoped_nodes
        else:
            warnings.append("Scoped code selection had no symbols; rendering global code view.")

    node_ids = {str(node.get("id")) for node in nodes}
    edges = _filter_edges_by_node_ids(symbol_graph["edges"], node_ids)
    _check_edge_kind_warnings(edges, warnings)

    nodes, edges, was_limited = _limit_nodes_by_degree(nodes, edges, max_nodes)
    if was_limited:
        warnings.append(f"Code view limited to top {max_nodes} symbols by degree for readability.")

    node_ids = {str(node.get("id")) for node in nodes}
    edges = _filter_edges_by_node_ids(edges, node_ids)

    container_name = "code-scope"
    if recovered_scope_name:
        container_name = recovered_scope_name
    elif selected_component:
        container_name = component_to_container.get(selected_component, container_name)

    lines = ["C4Component", f'title "{_safe_text(scenario_name, "Scenario")} - Code"']
    lines.append('Container_Boundary(code_boundary, "Code") {')
    lines.append(
        f'  Container_Boundary(scope_boundary, "{_safe_text(container_name, "scope")}") {{'
    )
    lines.extend(_render_code_symbol_lines(nodes))
    lines.append("  }")
    lines.append("}")
    lines.extend(_build_relation_lines(edges))

    if not nodes:
        warnings.append("No symbol-level nodes available for code view; returning minimal diagram.")

    return C4ExportResult(
        content="\n".join(lines),
        c4_view="code",
        c4_scope=c4_scope,
        warnings=warnings,
    )


async def _surface_counts(session: AsyncSession, collection_id: UUID) -> dict[str, int]:
    kinds = [
        KnowledgeNodeKind.API_ENDPOINT,
        KnowledgeNodeKind.GRAPHQL_OPERATION,
        KnowledgeNodeKind.SERVICE_RPC,
        KnowledgeNodeKind.JOB,
        KnowledgeNodeKind.MESSAGE_SCHEMA,
        KnowledgeNodeKind.DB_TABLE,
    ]

    rows = await session.execute(
        select(KnowledgeNode.kind, func.count(KnowledgeNode.id))
        .where(
            KnowledgeNode.collection_id == collection_id,
            KnowledgeNode.kind.in_(kinds),
        )
        .group_by(KnowledgeNode.kind)
    )

    counts: dict[str, int] = {_kind_value(kind): 0 for kind in kinds}
    for kind, count in rows.all():
        counts[_kind_value(kind)] = int(count or 0)
    return counts


async def _render_context_view(
    session: AsyncSession,
    collection_id: UUID,
    scenario_id: UUID,
    scenario_name: str,
    c4_scope: str | None,
) -> C4ExportResult:
    del c4_scope
    warnings: list[str] = []
    recovered_model = await _load_recovered_architecture_model(session, scenario_id, collection_id)
    _append_recovery_diagnostics(warnings, recovered_model)

    external_entities = [
        entity for entity in recovered_model.entities if entity.kind == "external_system"
    ]
    supporting_entities = [
        entity
        for entity in recovered_model.entities
        if entity.kind in {"data_store", "message_channel"}
    ]
    if external_entities or supporting_entities:
        lines = ["C4Context", f'title "{_safe_text(scenario_name, "Scenario")}"']
        lines.append('Person(user_actor, "User", "Application user")')
        lines.append(
            f'System(system_target, "{_safe_text(scenario_name, "System")}", "Primary application")'
        )
        for index, entity in enumerate(
            sorted(external_entities, key=lambda item: item.name.lower()),
            start=1,
        ):
            node_id = _safe_id(f"external:{index}:{entity.entity_id}")
            lines.append(
                f'System_Ext({node_id}, "{_safe_text(entity.name, "External System")}", "{entity.kind}")'
            )
            lines.append(f'Rel(system_target, {node_id}, "uses")')
        for index, entity in enumerate(
            sorted(supporting_entities, key=lambda item: item.name.lower()),
            start=1,
        ):
            node_id = _safe_id(f"supporting:{index}:{entity.entity_id}")
            lines.append(
                f'System_Ext({node_id}, "{_safe_text(entity.name, entity.kind)}", "{entity.kind}")'
            )
            lines.append(f'Rel(system_target, {node_id}, "depends_on")')
        lines.append('Rel(user_actor, system_target, "Uses")')
        return C4ExportResult(
            content="\n".join(lines),
            c4_view="context",
            c4_scope=None,
            warnings=warnings,
        )

    counts = await _surface_counts(session, collection_id)

    endpoint_count = (
        counts.get("api_endpoint", 0)
        + counts.get("graphql_operation", 0)
        + counts.get("service_rpc", 0)
    )
    job_count = counts.get("job", 0)
    data_count = counts.get("message_schema", 0) + counts.get("db_table", 0)

    lines = ["C4Context", f'title "{_safe_text(scenario_name, "Scenario")}"']
    lines.append('Person(user_actor, "User", "Application user")')
    lines.append(
        f'System(system_target, "{_safe_text(scenario_name, "System")}", "Primary application")'
    )

    if endpoint_count > 0:
        lines.append('System_Ext(client_system, "Client Apps", "Calls public APIs")')
        lines.append(
            f'Rel(client_system, system_target, "API usage ({endpoint_count} endpoints/ops)")'
        )
    if job_count > 0:
        lines.append('System_Ext(scheduler_system, "Scheduler/CI", "Triggers jobs/workflows")')
        lines.append(f'Rel(scheduler_system, system_target, "Scheduled flows ({job_count} jobs)")')
    if data_count > 0:
        lines.append('System_Ext(data_system, "Data Platform", "Schemas and data stores")')
        lines.append(f'Rel(system_target, data_system, "Data contracts ({data_count} entities)")')

    lines.append('Rel(user_actor, system_target, "Uses")')

    if endpoint_count == 0 and job_count == 0 and data_count == 0:
        warnings.append(
            "No recovered systems or surface graph signals found; rendered minimal context diagram."
        )

    return C4ExportResult(
        content="\n".join(lines),
        c4_view="context",
        c4_scope=None,
        warnings=warnings,
    )


def _infer_job_container(
    job: KnowledgeNode,
    file_path_by_job: dict[UUID, str],
    container_names: list[str],
    container_name_set: set[str],
) -> str:
    """Infer the container for a job based on file path or name matching."""
    file_path = file_path_by_job.get(job.id)
    if file_path:
        group = _derive_arch_group(file_path, {})
        if group:
            candidate = group[1]
            if not container_name_set or candidate.lower() in container_name_set:
                return candidate

    name_l = str(job.name or "").lower()
    for container_name in container_names:
        if container_name and container_name.lower() in name_l:
            return container_name

    return "shared"


def _scope_deployment_jobs(
    job_nodes: list[KnowledgeNode],
    file_path_by_job: dict[UUID, str],
    c4_scope: str | None,
    warnings: list[str],
) -> list[KnowledgeNode]:
    """Filter deployment jobs by scope, returning empty list if no scope or no match."""
    if not c4_scope:
        return []
    scope_l = c4_scope.lower()
    scoped: list[KnowledgeNode] = []
    for job in job_nodes:
        job_file_path = file_path_by_job.get(job.id, "")
        if scope_l in str(job.name or "").lower() or scope_l in job_file_path.lower():
            scoped.append(job)
    if not scoped:
        warnings.append(f'No deployment jobs matched scope "{c4_scope}"; rendered all jobs.')
    return scoped


def _render_deployment_container_lines(
    jobs_by_container: dict[str, list[KnowledgeNode]],
) -> list[str]:
    """Render Mermaid deployment node lines grouped by container."""
    lines: list[str] = []
    for container_name in sorted(jobs_by_container):
        deployment_id = _safe_id(f"deploy:{container_name}")
        lines.append(
            f'  Deployment_Node({deployment_id}, "{_safe_text(container_name, "shared")}", "deployment slice") {{'
        )
        for job in jobs_by_container[container_name]:
            job_id = _safe_id(str(job.id))
            job_meta = job.meta or {}
            framework = _safe_text(str(job_meta.get("framework") or "job"), "job")
            image = _safe_text(str(job_meta.get("container_image") or ""), "")
            description = f"framework={framework}"
            if image:
                description = f"{description} | image={image}"
            lines.append(
                f'    Container({job_id}, "{_safe_text(job.name, "job")}", "{framework}", "{_safe_text(description, framework)}")'
            )
        lines.append("  }")
    return lines


def _render_deployment_trigger_lines(
    jobs_by_container: dict[str, list[KnowledgeNode]],
) -> list[str]:
    """Render trigger relationship lines for deployment jobs."""
    lines: list[str] = ['System_Ext(trigger_engine, "Scheduler", "CI / cron / workflow triggers")']
    for jobs in jobs_by_container.values():
        for job in jobs:
            job_id = _safe_id(str(job.id))
            schedule = _safe_text(
                str((job.meta or {}).get("schedule") or "manual/event"), "manual/event"
            )
            lines.append(f'Rel(trigger_engine, {job_id}, "{schedule}")')
    return lines


def _recovered_job_container_names(
    model: RecoveredArchitectureModel,
    job: KnowledgeNode,
) -> list[str]:
    entity_name_by_id = {entity.entity_id: entity.name for entity in model.entities}
    container_ids = {
        membership.entity_id
        for membership in model.memberships_for(job.natural_key)
        if membership.entity_id.startswith("container:")
    }
    names = [
        entity_name_by_id.get(container_id, container_id.split(":", 1)[1])
        for container_id in sorted(container_ids)
    ]
    return [name for name in names if name]


async def _render_deployment_view(
    session: AsyncSession,
    collection_id: UUID,
    scenario_id: UUID,
    scenario_name: str,
    c4_scope: str | None,
    max_nodes: int,
) -> C4ExportResult:
    warnings: list[str] = []
    recovered_model = await _load_recovered_architecture_model(session, scenario_id, collection_id)
    _append_recovery_diagnostics(warnings, recovered_model)

    job_nodes = (
        (
            await session.execute(
                select(KnowledgeNode).where(
                    KnowledgeNode.collection_id == collection_id,
                    KnowledgeNode.kind == KnowledgeNodeKind.JOB,
                )
            )
        )
        .scalars()
        .all()
    )

    if not job_nodes:
        lines = [
            "C4Deployment",
            f'title "{_safe_text(scenario_name, "Scenario")}"',
            'Deployment_Node(runtime_cluster, "Runtime", "cluster") {',
            '  Container(no_jobs, "No jobs detected", "deployment", "No deployment metadata found")',
            "}",
        ]
        warnings.append("No job nodes found; rendered minimal deployment diagram.")
        return C4ExportResult(
            content="\n".join(lines),
            c4_view="deployment",
            c4_scope=c4_scope,
            warnings=warnings,
        )

    job_ids = [node.id for node in job_nodes]
    evidence_rows = await session.execute(
        select(KnowledgeNodeEvidence.node_id, KnowledgeEvidence.file_path)
        .join(KnowledgeEvidence, KnowledgeEvidence.id == KnowledgeNodeEvidence.evidence_id)
        .where(KnowledgeNodeEvidence.node_id.in_(job_ids))
    )
    file_path_by_job: dict[UUID, str] = {}
    for node_id, file_path in evidence_rows.all():
        if node_id not in file_path_by_job and isinstance(file_path, str) and file_path.strip():
            file_path_by_job[node_id] = file_path.strip()

    container_graph = await get_full_scenario_graph(
        session=session,
        scenario_id=scenario_id,
        layer=None,
        projection=GraphProjection.ARCHITECTURE,
        entity_level="container",
        include_kinds={"file"},
    )
    container_names = sorted(
        {
            str(node.get("name") or "").strip()
            for node in container_graph["nodes"]
            if str(node.get("name") or "").strip()
        }
    )
    container_name_set = {name.lower() for name in container_names}

    jobs_by_container: dict[str, list[KnowledgeNode]] = defaultdict(list)
    heuristic_mapping_count = 0
    shared_mapping_count = 0

    def infer_container(job: KnowledgeNode) -> str:
        return _infer_job_container(
            job,
            file_path_by_job,
            container_names,
            container_name_set,
        )

    scoped_jobs = _scope_deployment_jobs(job_nodes, file_path_by_job, c4_scope, warnings)
    source_jobs = sorted(scoped_jobs or list(job_nodes), key=lambda node: str(node.name or ""))
    if len(source_jobs) > max_nodes:
        source_jobs = source_jobs[:max_nodes]
        warnings.append(f"Deployment view limited to {max_nodes} jobs for readability.")

    for job in source_jobs:
        recovered_container_names = _recovered_job_container_names(recovered_model, job)
        if recovered_container_names:
            for container_name in recovered_container_names:
                jobs_by_container[container_name].append(job)
            continue
        container_name = infer_container(job)
        if container_name == "shared":
            shared_mapping_count += 1
        else:
            heuristic_mapping_count += 1
        jobs_by_container[container_name].append(job)

    lines = ["C4Deployment", f'title "{_safe_text(scenario_name, "Scenario")}"']
    lines.append('Deployment_Node(runtime_cluster, "Runtime", "cluster") {')
    lines.extend(_render_deployment_container_lines(jobs_by_container))
    lines.append("}")
    lines.extend(_render_deployment_trigger_lines(jobs_by_container))

    if heuristic_mapping_count > 0:
        warnings.append(
            f"Missing recovered runtime evidence for {heuristic_mapping_count} jobs; used file-path/runtime heuristics."
        )
    if shared_mapping_count > 0:
        warnings.append(
            f"{shared_mapping_count} jobs could not be mapped to a concrete container and were placed in shared deployment."
        )

    return C4ExportResult(
        content="\n".join(lines),
        c4_view="deployment",
        c4_scope=c4_scope,
        warnings=warnings,
    )


async def export_mermaid_c4_result(
    session: AsyncSession,
    scenario_id: UUID,
    entity_level: str = "container",
    c4_view: str = DEFAULT_C4_VIEW,
    c4_scope: str | None = None,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> C4ExportResult:
    """Render a scenario as a Mermaid C4 view with metadata."""
    del entity_level

    scenario = (
        await session.execute(select(TwinScenario).where(TwinScenario.id == scenario_id))
    ).scalar_one()

    normalized_view = _normalize_c4_view(c4_view)
    normalized_scope = _normalize_scope(c4_scope)
    effective_max_nodes = max(10, int(max_nodes or DEFAULT_MAX_NODES))

    if normalized_view == "context":
        return await _render_context_view(
            session=session,
            collection_id=scenario.collection_id,
            scenario_id=scenario_id,
            scenario_name=scenario.name,
            c4_scope=normalized_scope,
        )
    if normalized_view == "component":
        return await _render_component_view(
            session=session,
            scenario_id=scenario_id,
            collection_id=scenario.collection_id,
            scenario_name=scenario.name,
            c4_scope=normalized_scope,
        )
    if normalized_view == "code":
        return await _render_code_view(
            session=session,
            scenario_id=scenario_id,
            collection_id=scenario.collection_id,
            scenario_name=scenario.name,
            c4_scope=normalized_scope,
            max_nodes=effective_max_nodes,
        )
    if normalized_view == "deployment":
        return await _render_deployment_view(
            session=session,
            collection_id=scenario.collection_id,
            scenario_id=scenario_id,
            scenario_name=scenario.name,
            c4_scope=normalized_scope,
            max_nodes=effective_max_nodes,
        )

    return await _render_container_view(
        session=session,
        scenario_id=scenario_id,
        collection_id=scenario.collection_id,
        scenario_name=scenario.name,
        c4_scope=normalized_scope,
    )


async def export_mermaid_c4(
    session: AsyncSession,
    scenario_id: UUID,
    entity_level: str = "container",
    c4_view: str = DEFAULT_C4_VIEW,
    c4_scope: str | None = None,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> str:
    """Render a scenario as Mermaid C4 content (backward-compatible string API)."""
    result = await export_mermaid_c4_result(
        session=session,
        scenario_id=scenario_id,
        entity_level=entity_level,
        c4_view=c4_view,
        c4_scope=c4_scope,
        max_nodes=max_nodes,
    )
    return result.content


async def export_mermaid_asis_tobe_result(
    session: AsyncSession,
    as_is_scenario_id: UUID,
    to_be_scenario_id: UUID,
    c4_view: str = DEFAULT_C4_VIEW,
    c4_scope: str | None = None,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> tuple[C4ExportResult, C4ExportResult]:
    """Render two Mermaid C4 results (AS-IS and TO-BE)."""
    as_is = await export_mermaid_c4_result(
        session=session,
        scenario_id=as_is_scenario_id,
        c4_view=c4_view,
        c4_scope=c4_scope,
        max_nodes=max_nodes,
    )
    to_be = await export_mermaid_c4_result(
        session=session,
        scenario_id=to_be_scenario_id,
        c4_view=c4_view,
        c4_scope=c4_scope,
        max_nodes=max_nodes,
    )
    return as_is, to_be


async def export_mermaid_asis_tobe(
    session: AsyncSession,
    as_is_scenario_id: UUID,
    to_be_scenario_id: UUID,
    c4_view: str = DEFAULT_C4_VIEW,
    c4_scope: str | None = None,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> tuple[str, str]:
    """Render two Mermaid C4 artifacts (AS-IS and TO-BE)."""
    as_is, to_be = await export_mermaid_asis_tobe_result(
        session=session,
        as_is_scenario_id=as_is_scenario_id,
        to_be_scenario_id=to_be_scenario_id,
        c4_view=c4_view,
        c4_scope=c4_scope,
        max_nodes=max_nodes,
    )
    return as_is.content, to_be.content
