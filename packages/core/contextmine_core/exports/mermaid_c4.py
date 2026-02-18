"""Mermaid C4 export for AS-IS/TO-BE twin scenarios."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any
from uuid import UUID

from contextmine_core.models import (
    KnowledgeEvidence,
    KnowledgeNode,
    KnowledgeNodeEvidence,
    KnowledgeNodeKind,
    TwinScenario,
)
from contextmine_core.twin import GraphProjection, get_full_scenario_graph
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


def _canonical_file_path(node: dict[str, Any]) -> str | None:
    kind = str(node.get("kind") or "").lower()
    natural_key = str(node.get("natural_key") or "")
    if kind == "file" and natural_key.startswith("file:"):
        return natural_key.split(":", 1)[1].strip() or None

    meta = node.get("meta") or {}
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


async def _render_container_view(
    session: AsyncSession,
    scenario_id: UUID,
    scenario_name: str,
    c4_scope: str | None,
) -> C4ExportResult:
    warnings: list[str] = []
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
        scope = c4_scope.lower()
        scoped_ids = {
            str(node["id"])
            for node in nodes
            if scope
            in {
                str(node.get("name") or "").lower(),
                str((node.get("meta") or {}).get("container") or "").lower(),
                str((node.get("meta") or {}).get("domain") or "").lower(),
            }
        }
        if scoped_ids:
            nodes = [node for node in nodes if str(node.get("id")) in scoped_ids]
            edges = [
                edge
                for edge in edges
                if str(edge.get("source_node_id")) in scoped_ids
                and str(edge.get("target_node_id")) in scoped_ids
            ]
        else:
            warnings.append(f'No container matched scope "{c4_scope}"; rendered all containers.')

    lines = ["C4Container", f'title "{_safe_text(scenario_name, "Scenario")}"']
    lines.append('System_Boundary(system_boundary, "System") {')
    for node in nodes:
        node_id = _safe_id(str(node.get("id") or ""))
        kind = _safe_text(str(node.get("kind") or "container"), "container")
        natural_key = _safe_text(str(node.get("natural_key") or ""), "")
        meta = node.get("meta") or {}
        label = _safe_text(str(node.get("name") or natural_key), natural_key or "container")
        description = _safe_text(f"{kind} | members={meta.get('member_count', 0)}", kind)
        lines.append(f'  Container({node_id}, "{label}", "{kind}", "{description}")')
    lines.append("}")
    lines.extend(_build_relation_lines(edges))

    if not nodes:
        warnings.append("No architecture containers available; returning minimal diagram.")

    return C4ExportResult(
        content="\n".join(lines),
        c4_view="container",
        c4_scope=c4_scope,
        warnings=warnings,
    )


async def _render_component_view(
    session: AsyncSession,
    scenario_id: UUID,
    scenario_name: str,
    c4_scope: str | None,
) -> C4ExportResult:
    warnings: list[str] = []
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
        scope = c4_scope.lower()
        scoped = [
            node
            for node in nodes
            if scope
            in {
                str(node.get("name") or "").lower(),
                str((node.get("meta") or {}).get("component") or "").lower(),
                str((node.get("meta") or {}).get("container") or "").lower(),
            }
        ]
        if scoped:
            scoped_ids = {str(node.get("id")) for node in scoped}
            nodes = scoped
            edges = [
                edge
                for edge in edges
                if str(edge.get("source_node_id")) in scoped_ids
                and str(edge.get("target_node_id")) in scoped_ids
            ]
        else:
            warnings.append(
                f'No component/container matched scope "{c4_scope}"; rendered all components.'
            )

    container_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        container = _safe_text(str((node.get("meta") or {}).get("container") or "shared"), "shared")
        container_groups[container].append(node)

    lines = ["C4Component", f'title "{_safe_text(scenario_name, "Scenario")}"']
    lines.append('Container_Boundary(system_container, "System") {')
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
    lines.append("}")
    lines.extend(_build_relation_lines(edges))

    if not nodes:
        warnings.append("No architecture components available; returning minimal diagram.")

    return C4ExportResult(
        content="\n".join(lines),
        c4_view="component",
        c4_scope=c4_scope,
        warnings=warnings,
    )


async def _render_code_view(
    session: AsyncSession,
    scenario_id: UUID,
    scenario_name: str,
    c4_scope: str | None,
    max_nodes: int,
) -> C4ExportResult:
    warnings: list[str] = []

    file_graph = await get_full_scenario_graph(
        session=session,
        scenario_id=scenario_id,
        layer=None,
        projection=GraphProjection.CODE_FILE,
        entity_level="file",
    )

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

    selected_component: str | None = None
    if component_to_paths:
        if c4_scope:
            scope = c4_scope.lower()
            for key, _paths in component_to_paths.items():
                _, container, component = key.split("/", 2)
                if scope in {key.lower(), container.lower(), component.lower()}:
                    selected_component = key
                    break
            if selected_component is None:
                warnings.append(
                    f'No component/file scope matched "{c4_scope}"; defaulted to largest component.'
                )
        if selected_component is None:
            selected_component = max(component_to_paths.items(), key=lambda item: len(item[1]))[0]

    scope_paths = component_to_paths.get(selected_component, set()) if selected_component else set()

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
    edges = [
        edge
        for edge in symbol_graph["edges"]
        if str(edge.get("source_node_id")) in node_ids
        and str(edge.get("target_node_id")) in node_ids
    ]

    has_call_edges = any(str(edge.get("kind") or "") == "symbol_calls_symbol" for edge in edges)
    has_reference_edges = any(
        str(edge.get("kind") or "") == "symbol_references_symbol" for edge in edges
    )
    if not has_call_edges and has_reference_edges:
        warnings.append(
            "No symbol_calls_symbol edges found in scope; using references/contains relationships as fallback."
        )

    nodes, edges, was_limited = _limit_nodes_by_degree(nodes, edges, max_nodes)
    if was_limited:
        warnings.append(f"Code view limited to top {max_nodes} symbols by degree for readability.")

    node_ids = {str(node.get("id")) for node in nodes}
    edges = [
        edge
        for edge in edges
        if str(edge.get("source_node_id")) in node_ids
        and str(edge.get("target_node_id")) in node_ids
    ]

    container_name = "code-scope"
    if selected_component:
        container_name = component_to_container.get(selected_component, container_name)

    lines = ["C4Component", f'title "{_safe_text(scenario_name, "Scenario")} - Code"']
    lines.append('Container_Boundary(code_boundary, "Code") {')
    lines.append(
        f'  Container_Boundary(scope_boundary, "{_safe_text(container_name, "scope")}") {{'
    )
    for node in sorted(nodes, key=lambda item: str(item.get("name") or "")):
        node_id = _safe_id(str(node.get("id") or ""))
        kind = _safe_text(str(node.get("kind") or "symbol"), "symbol")
        label = _safe_text(str(node.get("name") or kind), kind)
        file_path = _safe_text(_canonical_file_path(node) or "", "")
        description = _safe_text(f"{kind} | {file_path}" if file_path else kind, kind)
        lines.append(f'    Component({node_id}, "{label}", "{kind}", "{description}")')
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
    scenario_name: str,
    c4_scope: str | None,
) -> C4ExportResult:
    del c4_scope
    warnings: list[str] = []
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
            "No surface graph signals (API/jobs/data) found; rendered minimal context diagram."
        )

    return C4ExportResult(
        content="\n".join(lines),
        c4_view="context",
        c4_scope=None,
        warnings=warnings,
    )


async def _render_deployment_view(
    session: AsyncSession,
    collection_id: UUID,
    scenario_id: UUID,
    scenario_name: str,
    c4_scope: str | None,
    max_nodes: int,
) -> C4ExportResult:
    warnings: list[str] = []

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
    weak_mapping_count = 0

    def infer_container(job: KnowledgeNode) -> str:
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

    scoped_jobs: list[KnowledgeNode] = []
    if c4_scope:
        scope_l = c4_scope.lower()
        for job in job_nodes:
            job_file_path = file_path_by_job.get(job.id, "")
            if scope_l in str(job.name or "").lower() or scope_l in job_file_path.lower():
                scoped_jobs.append(job)
        if not scoped_jobs:
            warnings.append(f'No deployment jobs matched scope "{c4_scope}"; rendered all jobs.')

    source_jobs = scoped_jobs or list(job_nodes)
    source_jobs = sorted(source_jobs, key=lambda node: str(node.name or ""))
    if len(source_jobs) > max_nodes:
        source_jobs = source_jobs[:max_nodes]
        warnings.append(f"Deployment view limited to {max_nodes} jobs for readability.")

    for job in source_jobs:
        container_name = infer_container(job)
        if container_name == "shared":
            weak_mapping_count += 1
        jobs_by_container[container_name].append(job)

    lines = ["C4Deployment", f'title "{_safe_text(scenario_name, "Scenario")}"']
    lines.append('Deployment_Node(runtime_cluster, "Runtime", "cluster") {')
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
    lines.append("}")

    lines.append('System_Ext(trigger_engine, "Scheduler", "CI / cron / workflow triggers")')
    for jobs in jobs_by_container.values():
        for job in jobs:
            job_id = _safe_id(str(job.id))
            schedule = _safe_text(
                str((job.meta or {}).get("schedule") or "manual/event"), "manual/event"
            )
            lines.append(f'Rel(trigger_engine, {job_id}, "{schedule}")')

    if weak_mapping_count > 0:
        warnings.append(
            f"{weak_mapping_count} jobs could not be mapped to a concrete container and were placed in shared deployment."
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
            scenario_name=scenario.name,
            c4_scope=normalized_scope,
        )
    if normalized_view == "component":
        return await _render_component_view(
            session=session,
            scenario_id=scenario_id,
            scenario_name=scenario.name,
            c4_scope=normalized_scope,
        )
    if normalized_view == "code":
        return await _render_code_view(
            session=session,
            scenario_id=scenario_id,
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
