"""Deterministic UI extractor and graph materializer."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

from contextmine_core.analyzer.extractors.ast_utils import (
    first_child,
    is_pascal_case,
    line_number,
    node_text,
    unquote,
    walk,
)
from contextmine_core.analyzer.extractors.traceability import (
    build_endpoint_symbol_index,
    resolve_symbol_refs_for_calls,
    symbol_token_variants,
)
from contextmine_core.models import (
    KnowledgeEdge,
    KnowledgeEdgeKind,
    KnowledgeNode,
    KnowledgeNodeEvidence,
    KnowledgeNodeKind,
)
from contextmine_core.treesitter.languages import TreeSitterLanguage, detect_language
from contextmine_core.treesitter.manager import get_treesitter_manager
from sqlalchemy.dialects.postgresql import insert as pg_insert

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


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


UI_FILE_SUFFIXES = (
    ".tsx",
    ".jsx",
    ".vue",
    ".svelte",
    ".html",
    ".ts",
    ".js",
)
HTTP_METHOD_NAMES = {"get", "post", "put", "patch", "delete"}
HTTP_CLIENT_NAMES = {"axios", "client", "api", "http", "request", "agent"}
ROUTER_METHOD_NAMES = {"get", "post", "put", "patch", "delete"}
NAV_METHOD_NAMES = {"navigate", "push", "replace", "to"}
JS_UI_LANGUAGES = {
    TreeSitterLanguage.JAVASCRIPT,
    TreeSitterLanguage.TYPESCRIPT,
    TreeSitterLanguage.TSX,
}


@dataclass
class UIRouteDef:
    """Extracted UI route."""

    path: str
    file_path: str
    line: int
    view_name_hint: str | None = None


@dataclass
class UIViewDef:
    """Extracted UI view/screen."""

    name: str
    file_path: str
    line: int
    components: list[str] = field(default_factory=list)
    symbol_hints: list[str] = field(default_factory=list)
    endpoint_hints: list[str] = field(default_factory=list)
    navigation_targets: list[str] = field(default_factory=list)
    call_sites: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class UIExtraction:
    """Result of UI extraction for a single file."""

    file_path: str
    routes: list[UIRouteDef] = field(default_factory=list)
    views: list[UIViewDef] = field(default_factory=list)


def looks_like_ui_file(file_path: str) -> bool:
    """Best-effort UI-file detector."""
    lower = file_path.lower()
    if not lower.endswith(UI_FILE_SUFFIXES):
        return False
    return any(token in lower for token in ("/src/", "/ui/", "/pages/", "/app/", "/components/"))


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(v.strip() for v in values if v and v.strip()))


def extract_ui_from_file(file_path: str, content: str) -> UIExtraction:
    """Extract routes, views, and UI composition hints from one file."""
    extraction = UIExtraction(file_path=file_path)
    language = detect_language(file_path)
    if language not in JS_UI_LANGUAGES:
        return extraction

    manager = get_treesitter_manager()
    try:
        tree = manager.parse(file_path, content)
    except Exception as exc:
        logger.debug("ui extraction skipped for %s: %s", file_path, exc)
        return extraction

    root = tree.root_node
    extraction.routes = _extract_routes(file_path, content, root)
    extraction.views = _extract_views(file_path, content, root)
    return extraction


def extract_ui_from_files(files: list[tuple[str, str]]) -> list[UIExtraction]:
    """Extract UI semantics from a set of repository files."""
    results: list[UIExtraction] = []
    for file_path, content in files:
        if not looks_like_ui_file(file_path):
            continue
        if not content.strip():
            continue
        extraction = extract_ui_from_file(file_path, content)
        if extraction.routes or extraction.views:
            results.append(extraction)
    return results


def _extract_routes(file_path: str, content: str, root: Any) -> list[UIRouteDef]:
    routes: list[UIRouteDef] = []
    for node in walk(root):
        if node.type in {"jsx_self_closing_element", "jsx_opening_element"}:
            route = _route_from_jsx(file_path, content, node)
            if route is not None:
                routes.append(route)
            continue

        if node.type == "object":
            route = _route_from_object(file_path, content, node)
            if route is not None:
                routes.append(route)
            continue

        if node.type == "call_expression":
            route = _route_from_router_call(file_path, content, node)
            if route is not None:
                routes.append(route)

    deduped: list[UIRouteDef] = []
    seen: set[tuple[str, int, str | None]] = set()
    for route in routes:
        key = (route.path, route.line, route.view_name_hint)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(route)
    return deduped


def _route_from_jsx(file_path: str, content: str, node: Any) -> UIRouteDef | None:
    tag_name = _jsx_tag_name(content, node)
    if tag_name != "Route":
        return None

    path_value = ""
    view_hint: str | None = None
    for attr in [child for child in node.children if child.type == "jsx_attribute"]:
        name = node_text(content, first_child(attr, "property_identifier")).strip()
        if not name:
            continue
        if name == "path":
            path_value = _jsx_attribute_string(content, attr) or ""
        elif name in {"element", "component", "render"}:
            view_hint = _jsx_attribute_view_hint(content, attr) or view_hint
    if not path_value:
        return None
    return UIRouteDef(
        path=path_value, file_path=file_path, line=line_number(node), view_name_hint=view_hint
    )


def _route_from_object(file_path: str, content: str, node: Any) -> UIRouteDef | None:
    path_value: str | None = None
    view_hint: str | None = None
    for pair in [child for child in node.children if child.type == "pair"]:
        key_node = pair.child_by_field_name("key") or first_child(pair, "property_identifier")
        key = node_text(content, key_node).strip()
        if key == "path":
            value_node = pair.child_by_field_name("value")
            if value_node is not None:
                path_value = _string_literal(content, value_node)
        elif key in {"element", "component", "screen", "view"}:
            value_node = pair.child_by_field_name("value")
            if value_node is not None:
                view_hint = _view_name_from_value(content, value_node) or view_hint
    if not path_value:
        return None
    return UIRouteDef(
        path=path_value, file_path=file_path, line=line_number(node), view_name_hint=view_hint
    )


def _route_from_router_call(file_path: str, content: str, node: Any) -> UIRouteDef | None:
    _, base_name, method_name = _js_call_name(content, node)
    if method_name.lower() not in ROUTER_METHOD_NAMES:
        return None
    if "router" not in base_name.lower():
        return None
    route_path = _first_string_argument(content, node)
    if not route_path:
        return None
    return UIRouteDef(path=route_path, file_path=file_path, line=line_number(node))


def _extract_views(file_path: str, content: str, root: Any) -> list[UIViewDef]:
    views: dict[str, UIViewDef] = {}
    for node in walk(root):
        view_name: str | None = None
        signal_node: Any | None = None
        if node.type == "function_declaration":
            ident = first_child(node, "identifier")
            candidate = node_text(content, ident).strip()
            if is_pascal_case(candidate):
                view_name = candidate
                signal_node = node
        elif node.type == "class_declaration":
            ident = first_child(node, "type_identifier") or first_child(node, "identifier")
            candidate = node_text(content, ident).strip()
            if is_pascal_case(candidate):
                view_name = candidate
                signal_node = node
        elif node.type == "variable_declarator":
            ident = first_child(node, "identifier")
            value = node.child_by_field_name("value")
            candidate = node_text(content, ident).strip()
            if (
                is_pascal_case(candidate)
                and value is not None
                and value.type
                in {
                    "arrow_function",
                    "function_expression",
                }
            ):
                view_name = candidate
                signal_node = value

        if not view_name or signal_node is None:
            continue
        if not _contains_jsx(signal_node):
            continue

        components, symbol_hints, endpoint_hints, navigation_targets, call_sites = (
            _collect_view_signals(content, signal_node, view_name)
        )
        views[view_name] = UIViewDef(
            name=view_name,
            file_path=file_path,
            line=line_number(node),
            components=components[:40],
            symbol_hints=symbol_hints[:40],
            endpoint_hints=endpoint_hints[:20],
            navigation_targets=navigation_targets[:20],
            call_sites=call_sites[:120],
        )
    return list(views.values())


def _collect_view_signals(
    content: str,
    root: Any,
    view_name: str,
) -> tuple[list[str], list[str], list[str], list[str], list[dict[str, Any]]]:
    components: list[str] = []
    symbol_hints: list[str] = []
    endpoint_hints: list[str] = []
    navigation_targets: list[str] = []
    call_sites: list[dict[str, Any]] = []
    for node in walk(root):
        if node.type in {"jsx_self_closing_element", "jsx_opening_element"}:
            name = _jsx_tag_name(content, node)
            if name and is_pascal_case(name) and name != view_name and name != "Route":
                components.append(name)
            continue

        if node.type != "call_expression":
            continue
        _, base_name, method_name = _js_call_name(content, node)
        callee = method_name or base_name
        if callee:
            call_sites.append(
                {
                    "line": line_number(node),
                    "column": int(node.start_point[1]),
                    "callee": callee,
                }
            )
            if len(callee) >= 3:
                symbol_hints.append(callee)
        endpoint = _endpoint_from_call(content, node, base_name=base_name, method_name=method_name)
        if endpoint:
            endpoint_hints.append(endpoint)
        navigation = _navigation_target(content, node, base_name=base_name, method_name=method_name)
        if navigation:
            navigation_targets.append(navigation)
    return (
        _dedupe(components),
        _dedupe(symbol_hints),
        _dedupe(endpoint_hints),
        _dedupe(navigation_targets),
        call_sites,
    )


def _contains_jsx(node: Any) -> bool:
    return any(child.type in {"jsx_element", "jsx_self_closing_element"} for child in walk(node))


def _js_call_name(content: str, call_node: Any) -> tuple[str, str, str]:
    function = call_node.child_by_field_name("function")
    if function is None:
        return "", "", ""
    if function.type == "identifier":
        base = node_text(content, function).strip()
        return base, base, ""
    if function.type == "member_expression":
        obj = node_text(content, function.child_by_field_name("object")).strip()
        prop = node_text(content, function.child_by_field_name("property")).strip()
        full = f"{obj}.{prop}" if obj and prop else obj or prop
        return full, obj, prop
    token = node_text(content, function).strip()
    return token, token, ""


def _endpoint_from_call(
    content: str,
    call_node: Any,
    *,
    base_name: str,
    method_name: str,
) -> str | None:
    endpoint = _first_string_argument(content, call_node)
    if not endpoint:
        return None

    lower_base = base_name.lower()
    lower_method = method_name.lower()
    if lower_base == "fetch":
        return endpoint
    if lower_method in HTTP_METHOD_NAMES and (
        lower_base in HTTP_CLIENT_NAMES or "." in lower_base or lower_base.endswith("client")
    ):
        return endpoint
    return None


def _navigation_target(
    content: str,
    call_node: Any,
    *,
    base_name: str,
    method_name: str,
) -> str | None:
    first_arg = _first_string_argument(content, call_node)
    if not first_arg:
        return None
    lower_base = base_name.lower()
    lower_method = method_name.lower()
    if lower_base in NAV_METHOD_NAMES or lower_method in NAV_METHOD_NAMES:
        return first_arg
    return None


def _first_string_argument(content: str, call_node: Any) -> str | None:
    args = call_node.child_by_field_name("arguments")
    if args is None:
        return None
    for child in args.children:
        literal = _string_literal(content, child)
        if literal:
            return literal
    return None


def _string_literal(content: str, node: Any) -> str | None:
    if node.type == "string":
        return unquote(node_text(content, node))
    if node.type == "template_string":
        raw = node_text(content, node).strip()
        if "${" in raw:
            return None
        return unquote(raw)
    return None


def _jsx_tag_name(content: str, node: Any) -> str:
    ident = first_child(node, "identifier") or first_child(node, "nested_identifier")
    return node_text(content, ident).strip()


def _jsx_attribute_string(content: str, attr_node: Any) -> str | None:
    for child in attr_node.children:
        literal = _string_literal(content, child)
        if literal:
            return literal
        if child.type == "jsx_expression":
            for nested in walk(child):
                literal = _string_literal(content, nested)
                if literal:
                    return literal
    return None


def _jsx_attribute_view_hint(content: str, attr_node: Any) -> str | None:
    for child in attr_node.children:
        if child.type == "identifier":
            name = node_text(content, child).strip()
            if is_pascal_case(name):
                return name
        if child.type == "jsx_expression":
            for nested in walk(child):
                if nested.type in {"jsx_self_closing_element", "jsx_opening_element"}:
                    tag = _jsx_tag_name(content, nested)
                    if is_pascal_case(tag):
                        return tag
                if nested.type == "identifier":
                    token = node_text(content, nested).strip()
                    if is_pascal_case(token):
                        return token
    return None


def _view_name_from_value(content: str, value_node: Any) -> str | None:
    if value_node.type == "identifier":
        candidate = node_text(content, value_node).strip()
        if is_pascal_case(candidate):
            return candidate
    if value_node.type in {"jsx_self_closing_element", "jsx_opening_element"}:
        candidate = _jsx_tag_name(content, value_node)
        if is_pascal_case(candidate):
            return candidate
    for nested in walk(value_node):
        if nested.type == "identifier":
            candidate = node_text(content, nested).strip()
            if is_pascal_case(candidate):
                return candidate
    return None


async def _create_node_evidence(
    session: AsyncSession,
    *,
    node_id: UUID,
    file_path: str,
    line: int,
    snippet: str | None = None,
) -> str:
    from contextmine_core.models import KnowledgeEvidence

    evidence = KnowledgeEvidence(
        file_path=file_path,
        start_line=max(1, line),
        end_line=max(1, line),
        snippet=snippet,
    )
    session.add(evidence)
    await session.flush()
    session.add(KnowledgeNodeEvidence(node_id=node_id, evidence_id=evidence.id))
    return str(evidence.id)


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


async def build_ui_graph(
    session: AsyncSession,
    collection_id: UUID,
    extractions: list[UIExtraction],
    *,
    source_id: UUID | None = None,
) -> dict[str, int]:
    """Persist extracted UI semantics into the knowledge graph."""
    stats = {
        "ui_routes": 0,
        "ui_views": 0,
        "ui_components": 0,
        "interface_contracts": 0,
        "ui_edges": 0,
        "contract_edges": 0,
    }
    if not extractions:
        return stats

    endpoint_symbol_index = await build_endpoint_symbol_index(
        session=session,
        collection_id=collection_id,
    )

    route_ids_by_path: dict[str, UUID] = {}
    view_ids_by_name: dict[str, UUID] = {}
    component_ids_by_name: dict[str, UUID] = {}

    for extraction in extractions:
        for route in extraction.routes:
            route_key = f"ui_route:{route.path}"
            route_meta = {
                "file_path": route.file_path,
                "path": route.path,
                "view_name_hint": route.view_name_hint,
                **_provenance(mode="deterministic", extractor="ui.v1", confidence=0.98),
            }
            route_id = await _upsert_node(
                session,
                collection_id=collection_id,
                kind=KnowledgeNodeKind.UI_ROUTE,
                natural_key=route_key,
                name=route.path,
                meta=route_meta,
            )
            evidence_id = await _create_node_evidence(
                session,
                node_id=route_id,
                file_path=route.file_path,
                line=route.line,
                snippet=f"route {route.path}",
            )
            route_meta["provenance"]["evidence_ids"] = [evidence_id]  # type: ignore[index]
            await _upsert_node(
                session,
                collection_id=collection_id,
                kind=KnowledgeNodeKind.UI_ROUTE,
                natural_key=route_key,
                name=route.path,
                meta=route_meta,
            )
            route_ids_by_path[route.path] = route_id
            stats["ui_routes"] += 1

        for view in extraction.views:
            view_key = f"ui_view:{view.file_path}:{view.name}"
            view_meta = {
                "file_path": view.file_path,
                "components": view.components,
                "symbol_hints": view.symbol_hints,
                "endpoint_hints": view.endpoint_hints,
                "navigation_targets": view.navigation_targets,
                "call_sites": view.call_sites,
                **_provenance(mode="deterministic", extractor="ui.v1", confidence=0.94),
            }
            view_id = await _upsert_node(
                session,
                collection_id=collection_id,
                kind=KnowledgeNodeKind.UI_VIEW,
                natural_key=view_key,
                name=view.name,
                meta=view_meta,
            )
            evidence_id = await _create_node_evidence(
                session,
                node_id=view_id,
                file_path=view.file_path,
                line=view.line,
                snippet=f"view {view.name}",
            )
            view_meta["provenance"]["evidence_ids"] = [evidence_id]  # type: ignore[index]
            await _upsert_node(
                session,
                collection_id=collection_id,
                kind=KnowledgeNodeKind.UI_VIEW,
                natural_key=view_key,
                name=view.name,
                meta=view_meta,
            )
            view_ids_by_name[view.name] = view_id
            stats["ui_views"] += 1

            for component_name in view.components:
                component_key = f"ui_component:{view.file_path}:{component_name}"
                component_meta = {
                    "file_path": view.file_path,
                    **_provenance(mode="deterministic", extractor="ui.v1", confidence=0.88),
                }
                component_id = await _upsert_node(
                    session,
                    collection_id=collection_id,
                    kind=KnowledgeNodeKind.UI_COMPONENT,
                    natural_key=component_key,
                    name=component_name,
                    meta=component_meta,
                )
                component_ids_by_name[component_name] = component_id
                stats["ui_components"] += 1

                await _upsert_edge(
                    session,
                    collection_id=collection_id,
                    source_node_id=view_id,
                    target_node_id=component_id,
                    kind=KnowledgeEdgeKind.UI_VIEW_COMPOSES_COMPONENT,
                    meta=_provenance(
                        mode="deterministic",
                        extractor="ui.v1",
                        confidence=0.88,
                        evidence_ids=[evidence_id],
                    ),
                )
                stats["ui_edges"] += 1

            resolved_symbol_refs = await resolve_symbol_refs_for_calls(
                session=session,
                collection_id=collection_id,
                source_id=source_id,
                file_path=view.file_path,
                call_sites=view.call_sites,
                fallback_symbol_hints=view.symbol_hints,
            )
            view_meta["resolved_symbol_refs"] = [
                {
                    "symbol_node_id": str(ref.symbol_node_id),
                    "symbol_name": ref.symbol_name,
                    "engine": ref.engine,
                    "confidence": ref.confidence,
                }
                for ref in resolved_symbol_refs
            ]
            await _upsert_node(
                session,
                collection_id=collection_id,
                kind=KnowledgeNodeKind.UI_VIEW,
                natural_key=view_key,
                name=view.name,
                meta=view_meta,
            )

            for ref in resolved_symbol_refs:
                contract_key = f"interface_contract:{view.name}:{ref.symbol_node_id}"
                contract_meta = {
                    "source_view": view.name,
                    "symbol_node_id": str(ref.symbol_node_id),
                    "symbol_name": ref.symbol_name,
                    "resolution_engine": ref.engine,
                    **_provenance(
                        mode="deterministic" if ref.engine.startswith("scip") else "inferred",
                        extractor=f"ui.v1.{ref.engine}",
                        confidence=ref.confidence,
                    ),
                }
                contract_id = await _upsert_node(
                    session,
                    collection_id=collection_id,
                    kind=KnowledgeNodeKind.INTERFACE_CONTRACT,
                    natural_key=contract_key,
                    name=f"{view.name} contract",
                    meta=contract_meta,
                )
                stats["interface_contracts"] += 1
                token_candidates = symbol_token_variants(ref.symbol_name)
                for token in token_candidates:
                    endpoint_ids = endpoint_symbol_index.get(token, set())
                    for endpoint_id in endpoint_ids:
                        await _upsert_edge(
                            session,
                            collection_id=collection_id,
                            source_node_id=contract_id,
                            target_node_id=endpoint_id,
                            kind=KnowledgeEdgeKind.CONTRACT_GOVERNS_ENDPOINT,
                            meta=_provenance(
                                mode="inferred",
                                extractor=f"ui.v1.endpoint.{ref.engine}",
                                confidence=max(0.66, ref.confidence - 0.08),
                                evidence_ids=[evidence_id],
                            ),
                        )
                        stats["contract_edges"] += 1

        if extraction.routes and extraction.views:
            default_view_name = extraction.views[0].name if len(extraction.views) == 1 else None
            for route in extraction.routes:
                route_id = route_ids_by_path.get(route.path)
                if not route_id:
                    continue

                target_view_name = route.view_name_hint or default_view_name
                target_view_id = (
                    view_ids_by_name.get(target_view_name) if target_view_name else None
                )
                if target_view_id is None:
                    continue

                inferred = route.view_name_hint is None
                await _upsert_edge(
                    session,
                    collection_id=collection_id,
                    source_node_id=route_id,
                    target_node_id=target_view_id,
                    kind=KnowledgeEdgeKind.UI_ROUTE_RENDERS_VIEW,
                    meta=_provenance(
                        mode="inferred" if inferred else "deterministic",
                        extractor="ui.v1",
                        confidence=0.78 if inferred else 0.94,
                    ),
                )
                stats["ui_edges"] += 1

    # Keep dictionaries available for synth layers in calling code if needed.
    del component_ids_by_name
    return stats
