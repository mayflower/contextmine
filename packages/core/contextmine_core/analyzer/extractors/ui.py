"""Deterministic UI extractor and graph materializer."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
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
from sqlalchemy import select
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


JS_UI_FILE_SUFFIXES = (".tsx", ".jsx", ".vue", ".svelte", ".ts", ".js")
UI_TEMPLATE_FILE_SUFFIXES = (
    ".html",
    ".htm",
    ".twig",
    ".phtml",
    ".blade.php",
    ".erb",
    ".haml",
    ".jinja",
    ".jinja2",
    ".mustache",
    ".hbs",
    ".ejs",
    ".liquid",
    ".gohtml",
)
UI_ROUTE_FILE_SUFFIXES = (
    ".php",
    ".py",
    ".rb",
    ".go",
    ".java",
    ".kt",
    ".kts",
    ".cs",
    ".js",
    ".ts",
)
UI_JS_PATH_HINTS = ("/src/", "/ui/", "/pages/", "/app/", "/components/", "/assets/")
UI_TEMPLATE_PATH_HINTS = (
    "/templates/",
    "/template/",
    "/views/",
    "/view/",
    "/resources/views/",
    "/frontend/",
    "/front/",
    "/theme/",
    "/themes/",
    "/web/",
    "/public/",
)
UI_ROUTE_PATH_HINTS = (
    "/routes/",
    "/route/",
    "/router/",
    "/controllers/",
    "/controller/",
    "/routing/",
    "/urls.py",
)
UI_ROUTE_FILE_BASENAMES = {
    "routes.php",
    "web.php",
    "urls.py",
    "routes.rb",
    "router.go",
    "routes.ts",
    "routes.js",
}
HTTP_METHOD_NAMES = {"get", "post", "put", "patch", "delete"}
HTTP_CLIENT_NAMES = {"axios", "client", "api", "http", "request", "agent"}
ROUTER_METHOD_NAMES = {"get", "post", "put", "patch", "delete"}
NAV_METHOD_NAMES = {"navigate", "push", "replace", "to"}
JS_UI_LANGUAGES = {
    TreeSitterLanguage.JAVASCRIPT,
    TreeSitterLanguage.TYPESCRIPT,
    TreeSitterLanguage.TSX,
}
UI_TEST_PATH_TOKENS = ("/__tests__/", "/tests/", "/test/", "/cypress/", "/playwright/")
UI_TEST_FILE_MARKERS = (".test.", ".spec.", "_test.", "_spec.")
UI_ROUTE_SEGMENT_HINTS = {
    "admin",
    "app",
    "content",
    "dashboard",
    "page",
    "pages",
    "route",
    "routes",
    "screen",
    "screens",
    "settings",
    "view",
    "views",
}
UI_GENERIC_VIEW_STEMS = {
    "api",
    "client",
    "config",
    "configuration",
    "constants",
    "helper",
    "helpers",
    "service",
    "services",
    "types",
    "util",
    "utils",
}
UI_STRIP_ROUTE_SEGMENTS = {"pages", "page", "routes", "route", "views", "view", "screens", "screen"}
UI_ROUTE_EXCLUDE_PREFIXES = ("/api", "/graphql", "/rpc", "/internal", "/v1", "/v2")
UI_RENDER_CALL_PATTERN = re.compile(
    r"(?i)(?:\bview|render(?:_template|Template)?|template|TemplateResponse)\s*\(\s*['\"](?P<name>[^'\"]+)['\"]"
)
UI_ROUTE_PATTERNS = (
    re.compile(
        r"(?i)\bRoute::(?:get|post|put|patch|delete|match|any)\s*\(\s*['\"](?P<path>[^'\"]+)['\"]"
    ),
    re.compile(
        r"(?i)\b(?:\$[A-Za-z_][\w]*|app|router|route)\s*->\s*(?:get|post|put|patch|delete|match|any|map)\s*\(\s*['\"](?P<path>[^'\"]+)['\"]"
    ),
    re.compile(r"(?i)#\[\s*Route\s*\(\s*['\"](?P<path>[^'\"]+)['\"]"),
    re.compile(r"(?i)@Route\s*\(\s*['\"](?P<path>[^'\"]+)['\"]"),
    re.compile(r"(?i)\b(?:path|re_path)\s*\(\s*r?['\"](?P<path>[^'\"]+)['\"]"),
    re.compile(r"(?i)\b(?:get|post|put|patch|delete)\s+['\"](?P<path>/[^'\"]+)['\"]"),
)
UI_ENDPOINT_HINT_PATTERNS = (
    re.compile(
        r"(?i)\b(?:fetch|axios\.(?:get|post|put|patch|delete)|api\.(?:get|post|put|patch|delete)|client\.(?:get|post|put|patch|delete))\s*\(\s*['\"](?P<path>[^'\"]+)['\"]"
    ),
    re.compile(r"(?i)\baction\s*=\s*['\"](?P<path>[^'\"]+)['\"]"),
    re.compile(r"(?i)\bdata-(?:url|endpoint)\s*=\s*['\"](?P<path>[^'\"]+)['\"]"),
)
UI_NAVIGATION_HINT_PATTERN = re.compile(r"(?i)\bhref\s*=\s*['\"](?P<path>[^'\"]+)['\"]")
UI_COMPONENT_TAG_PATTERN = re.compile(r"<(?P<tag>[A-Za-z][A-Za-z0-9:_-]{2,})\b")
UI_SYMBOL_HINT_PATTERN = re.compile(
    r"\b(?P<symbol>[A-Z][A-Za-z0-9_]+Controller::[A-Za-z0-9_]+|[A-Z][A-Za-z0-9_]*(?:Service|Controller|Handler|Action)\b)"
)


@dataclass
class UIRouteDef:
    """Extracted UI route."""

    path: str
    file_path: str
    line: int
    view_name_hint: str | None = None
    inferred: bool = False


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
    inferred: bool = False


@dataclass
class UIExtraction:
    """Result of UI extraction for a single file."""

    file_path: str
    routes: list[UIRouteDef] = field(default_factory=list)
    views: list[UIViewDef] = field(default_factory=list)


def looks_like_ui_file(file_path: str) -> bool:
    """Best-effort UI-file detector."""
    path = file_path.replace("\\", "/").lower()
    basename = PurePosixPath(path).name
    if looks_like_ui_test_file(path):
        return False
    if path.endswith(JS_UI_FILE_SUFFIXES):
        return any(token in path for token in UI_JS_PATH_HINTS)
    if path.endswith(UI_TEMPLATE_FILE_SUFFIXES):
        return any(token in path for token in UI_TEMPLATE_PATH_HINTS + UI_JS_PATH_HINTS)
    if path.endswith(UI_ROUTE_FILE_SUFFIXES):
        return basename in UI_ROUTE_FILE_BASENAMES or any(
            token in path for token in UI_ROUTE_PATH_HINTS
        )
    return False


def looks_like_ui_test_file(file_path: str) -> bool:
    """Return True when a file path resembles a UI test fixture."""
    normalized = file_path.replace("\\", "/").lower()
    if any(token in normalized for token in UI_TEST_PATH_TOKENS):
        return True
    return any(marker in normalized for marker in UI_TEST_FILE_MARKERS)


def _to_pascal_case(raw: str) -> str:
    tokens = [part for part in re.split(r"[^A-Za-z0-9]+", raw or "") if part]
    if not tokens:
        return ""
    return "".join(token[:1].upper() + token[1:] for token in tokens)


def _view_name_from_file_path(file_path: str) -> str | None:
    path = PurePosixPath(file_path.replace("\\", "/"))
    stem = path.stem
    while True:
        normalized = re.sub(
            r"\.(html|twig|blade|php|phtml|erb|haml|jinja2?|mustache|hbs|ejs)$",
            "",
            stem,
            flags=re.IGNORECASE,
        )
        if normalized == stem:
            break
        stem = normalized
    if stem.lower() in {"index", "main", "app"} and path.parent.name:
        stem = path.parent.name
    if stem.lower() in UI_GENERIC_VIEW_STEMS and path.parent.name:
        stem = path.parent.name
    candidate = _to_pascal_case(stem)
    return candidate if is_pascal_case(candidate) else None


def _looks_like_route_module(file_path: str) -> bool:
    normalized = file_path.replace("\\", "/").lower()
    parts = [part for part in normalized.split("/") if part]
    if any(part in UI_ROUTE_SEGMENT_HINTS for part in parts):
        return True
    return "/admin/assets/src/" in normalized


def _route_path_from_file(file_path: str) -> str | None:
    normalized = file_path.replace("\\", "/")
    lower = normalized.lower()
    base_prefix = ""
    relative = ""
    if "/admin/assets/src/" in lower:
        split_idx = lower.index("/admin/assets/src/")
        relative = normalized[split_idx + len("/admin/assets/src/") :]
        base_prefix = "/admin"
    elif "/assets/src/" in lower:
        split_idx = lower.index("/assets/src/")
        relative = normalized[split_idx + len("/assets/src/") :]
        base_prefix = "/admin" if "/admin/" in lower[:split_idx] else ""
    elif "/src/" in lower:
        split_idx = lower.index("/src/")
        relative = normalized[split_idx + len("/src/") :]
    elif "/ui/" in lower:
        split_idx = lower.index("/ui/")
        relative = normalized[split_idx + len("/ui/") :]
    else:
        return None

    relative_path = PurePosixPath(relative)
    segments = [segment for segment in relative_path.parts if segment and segment != "."]
    if not segments:
        return base_prefix or "/"

    if (
        segments[-1]
        .lower()
        .endswith(JS_UI_FILE_SUFFIXES + UI_TEMPLATE_FILE_SUFFIXES + UI_ROUTE_FILE_SUFFIXES)
    ):
        file_stem = PurePosixPath(segments[-1]).stem
        segments = segments[:-1] + [file_stem]

    filtered: list[str] = []
    for idx, segment in enumerate(segments):
        token = segment.strip().lower()
        if not token:
            continue
        if idx == 0 and token in UI_STRIP_ROUTE_SEGMENTS:
            continue
        if token in {"index", "."}:
            continue
        filtered.append(token)

    if not filtered:
        return base_prefix or "/"

    route = "/".join(filtered)
    if base_prefix:
        route = f"{base_prefix.rstrip('/')}/{route}".rstrip("/")
    return f"/{route.strip('/')}"


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(v.strip() for v in values if v and v.strip()))


def _line_number_for_offset(content: str, offset: int) -> int:
    return max(1, content.count("\n", 0, max(0, offset)) + 1)


def _normalize_ui_route_path(raw: str) -> str | None:
    candidate = (raw or "").strip()
    if not candidate:
        return None
    if "://" in candidate:
        return None
    candidate = candidate.split("?", 1)[0].split("#", 1)[0]
    candidate = candidate.strip().strip("^$").strip()
    if not candidate:
        return None
    if "<" in candidate and ">" in candidate:
        candidate = re.sub(r"<[^>]+>", ":param", candidate)
    if "{" in candidate and "}" in candidate:
        candidate = re.sub(r"\{[^}]+\}", ":param", candidate)
    if not candidate.startswith("/"):
        candidate = f"/{candidate}"
    normalized = "/" + "/".join(part for part in candidate.split("/") if part)
    if not normalized:
        return "/"
    lower = normalized.lower()
    if any(
        lower == prefix or lower.startswith(f"{prefix}/") for prefix in UI_ROUTE_EXCLUDE_PREFIXES
    ):
        return None
    return normalized


def _normalize_view_hint(raw: str) -> str | None:
    candidate = (raw or "").strip()
    if not candidate:
        return None
    candidate = candidate.split("::", 1)[-1]
    candidate = candidate.replace(".", "/").replace(":", "/")
    parts = [part for part in re.split(r"[/\\]+", candidate) if part]
    if not parts:
        return None
    tail = parts[-1]
    if tail.lower() in {"index", "main", "default"} and len(parts) > 1:
        tail = parts[-2]
    tail = re.sub(r"\.(html|twig|blade|php|phtml|erb|haml|jinja2?|mustache|hbs|ejs)$", "", tail)
    value = _to_pascal_case(tail)
    return value if is_pascal_case(value) else None


def _is_template_like_file(file_path: str) -> bool:
    normalized = file_path.replace("\\", "/").lower()
    return normalized.endswith(UI_TEMPLATE_FILE_SUFFIXES) and any(
        token in normalized for token in UI_TEMPLATE_PATH_HINTS + UI_JS_PATH_HINTS
    )


def _extract_ui_heuristic(file_path: str, content: str) -> UIExtraction:
    extraction = UIExtraction(file_path=file_path)
    routes: list[UIRouteDef] = []
    route_keys: set[tuple[str, int, str | None]] = set()

    for pattern in UI_ROUTE_PATTERNS:
        for match in pattern.finditer(content):
            raw_path = match.groupdict().get("path") or ""
            path = _normalize_ui_route_path(raw_path)
            if not path:
                continue
            window = content[match.end() : match.end() + 600]
            render_match = UI_RENDER_CALL_PATTERN.search(window)
            view_hint = (
                _normalize_view_hint(render_match.groupdict().get("name") or "")
                if render_match
                else None
            )
            line = _line_number_for_offset(content, match.start())
            key = (path, line, view_hint)
            if key in route_keys:
                continue
            route_keys.add(key)
            routes.append(
                UIRouteDef(path=path, file_path=file_path, line=line, view_name_hint=view_hint)
            )

    if not routes and _looks_like_route_module(file_path):
        inferred_route = _route_path_from_file(file_path)
        normalized_route = _normalize_ui_route_path(inferred_route or "")
        if normalized_route:
            routes.append(
                UIRouteDef(
                    path=normalized_route,
                    file_path=file_path,
                    line=1,
                    view_name_hint=_view_name_from_file_path(file_path),
                    inferred=True,
                )
            )

    view_candidates: dict[str, UIViewDef] = {}
    template_like = _is_template_like_file(file_path)

    if template_like:
        name = _view_name_from_file_path(file_path)
        if name:
            view_candidates[name] = UIViewDef(
                name=name,
                file_path=file_path,
                line=1,
                inferred=True,
            )

    for render_match in UI_RENDER_CALL_PATTERN.finditer(content):
        view_name = _normalize_view_hint(render_match.groupdict().get("name") or "")
        if not view_name:
            continue
        if view_name not in view_candidates:
            view_candidates[view_name] = UIViewDef(
                name=view_name,
                file_path=file_path,
                line=_line_number_for_offset(content, render_match.start()),
                inferred=True,
            )

    components: list[str] = []
    symbol_hints: list[str] = []
    endpoint_hints: list[str] = []
    navigation_targets: list[str] = []

    for tag_match in UI_COMPONENT_TAG_PATTERN.finditer(content):
        tag = tag_match.groupdict().get("tag") or ""
        if ":" in tag:
            tag = tag.split(":", 1)[1]
        if "-" in tag:
            components.append(_to_pascal_case(tag))
        elif is_pascal_case(tag):
            components.append(tag)

    for symbol_match in UI_SYMBOL_HINT_PATTERN.finditer(content):
        symbol_hints.append(symbol_match.groupdict().get("symbol") or "")

    for pattern in UI_ENDPOINT_HINT_PATTERNS:
        for endpoint_match in pattern.finditer(content):
            hint = _normalize_endpoint_path(endpoint_match.groupdict().get("path") or "")
            if hint:
                endpoint_hints.append(hint)

    for navigation_match in UI_NAVIGATION_HINT_PATTERN.finditer(content):
        nav = _normalize_endpoint_path(navigation_match.groupdict().get("path") or "")
        if not nav:
            continue
        if any(nav.startswith(prefix) for prefix in UI_ROUTE_EXCLUDE_PREFIXES):
            endpoint_hints.append(nav)
        else:
            navigation_targets.append(nav)

    if not view_candidates and routes:
        hinted_names = [route.view_name_hint for route in routes if route.view_name_hint]
        if len(hinted_names) == 1:
            only_name = hinted_names[0]
            if only_name:
                view_candidates[only_name] = UIViewDef(
                    name=only_name,
                    file_path=file_path,
                    line=1,
                    inferred=True,
                )

    if view_candidates:
        deduped_components = _dedupe(components)[:40]
        deduped_symbols = _dedupe(symbol_hints)[:40]
        deduped_endpoints = _dedupe(endpoint_hints)[:20]
        deduped_navigation = _dedupe(navigation_targets)[:20]
        for view in view_candidates.values():
            view.components = deduped_components
            view.symbol_hints = deduped_symbols
            view.endpoint_hints = deduped_endpoints
            view.navigation_targets = deduped_navigation

    extraction.routes = routes
    extraction.views = list(view_candidates.values())
    return extraction


def extract_ui_from_file(file_path: str, content: str) -> UIExtraction:
    """Extract routes, views, and UI composition hints from one file."""
    extraction = UIExtraction(file_path=file_path)
    if looks_like_ui_test_file(file_path):
        return extraction

    language = detect_language(file_path)
    if language not in JS_UI_LANGUAGES:
        return _extract_ui_heuristic(file_path, content)

    manager = get_treesitter_manager()
    try:
        tree = manager.parse(file_path, content)
    except Exception as exc:
        logger.debug("ui extraction skipped for %s: %s", file_path, exc)
        return extraction

    root = tree.root_node
    extraction.routes = _extract_routes(file_path, content, root)
    extraction.views = _extract_views(file_path, content, root)
    if not extraction.views:
        inferred = _infer_module_view(file_path, content, root)
        if inferred is not None:
            extraction.views = [inferred]

    if not extraction.routes and extraction.views:
        for view in extraction.views:
            inferred_route = _infer_route_for_view(file_path, view)
            if inferred_route is not None:
                extraction.routes.append(inferred_route)
                break

    if not extraction.routes and not extraction.views:
        return _extract_ui_heuristic(file_path, content)

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


def _infer_module_view(file_path: str, content: str, root: Any) -> UIViewDef | None:
    view_name = _view_name_from_file_path(file_path)
    if not view_name:
        return None

    components, symbol_hints, endpoint_hints, navigation_targets, call_sites = (
        _collect_view_signals(
            content,
            root,
            view_name,
        )
    )
    if not (
        endpoint_hints
        or navigation_targets
        or components
        or (symbol_hints and _looks_like_route_module(file_path))
    ):
        return None

    return UIViewDef(
        name=view_name,
        file_path=file_path,
        line=1,
        components=components[:40],
        symbol_hints=symbol_hints[:40],
        endpoint_hints=endpoint_hints[:20],
        navigation_targets=navigation_targets[:20],
        call_sites=call_sites[:120],
        inferred=True,
    )


def _infer_route_for_view(file_path: str, view: UIViewDef) -> UIRouteDef | None:
    if not _looks_like_route_module(file_path) and not view.navigation_targets:
        return None
    route_path = _route_path_from_file(file_path)
    if not route_path:
        return None
    return UIRouteDef(
        path=route_path,
        file_path=file_path,
        line=1,
        view_name_hint=view.name,
        inferred=True,
    )


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
        full_name, base_name, method_name = _js_call_name(content, node)
        callee = full_name or method_name or base_name
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
            if method_name and method_name != callee and len(method_name) >= 3:
                symbol_hints.append(method_name)
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
    endpoint = _first_string_or_url_argument(content, call_node)
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


def _first_string_or_url_argument(content: str, call_node: Any) -> str | None:
    args = call_node.child_by_field_name("arguments")
    if args is None:
        return None
    for child in args.children:
        literal = _string_literal(content, child)
        if literal:
            return literal
        if child.type == "object":
            url_literal = _object_field_string(
                content, child, field_names={"url", "path", "endpoint"}
            )
            if url_literal:
                return url_literal
    return None


def _object_field_string(
    content: str,
    object_node: Any,
    *,
    field_names: set[str],
) -> str | None:
    for pair in [child for child in object_node.children if child.type == "pair"]:
        key_node = pair.child_by_field_name("key") or first_child(pair, "property_identifier")
        key = unquote(node_text(content, key_node)).strip().lower()
        if key not in field_names:
            continue
        value_node = pair.child_by_field_name("value")
        if value_node is None:
            continue
        literal = _string_literal(content, value_node)
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


def _normalize_endpoint_path(value: str) -> str | None:
    raw = (value or "").strip()
    if not raw:
        return None
    if "://" in raw:
        _, _, remainder = raw.partition("://")
        slash = remainder.find("/")
        raw = remainder[slash:] if slash >= 0 else "/"
    raw = raw.split("?", 1)[0].split("#", 1)[0].strip()
    if not raw:
        return None
    if not raw.startswith("/"):
        raw = f"/{raw}"
    return "/" + "/".join(part for part in raw.split("/") if part)


def _endpoint_path_from_name(name: str) -> tuple[str | None, str | None]:
    value = (name or "").strip()
    if not value:
        return None, None
    method = None
    path = value
    if " " in value:
        first, remainder = value.split(" ", 1)
        if first.strip().lower() in HTTP_METHOD_NAMES:
            method = first.strip().lower()
            path = remainder.strip()
    return method, _normalize_endpoint_path(path)


def _parse_endpoint_hint(hint: str) -> tuple[str | None, str | None]:
    value = (hint or "").strip()
    if not value:
        return None, None
    method = None
    path = value
    if " " in value:
        first, remainder = value.split(" ", 1)
        if first.strip().lower() in HTTP_METHOD_NAMES:
            method = first.strip().lower()
            path = remainder.strip()
    return method, _normalize_endpoint_path(path)


async def _build_endpoint_path_indexes(
    session: AsyncSession,
    *,
    collection_id: UUID,
) -> tuple[dict[str, set[UUID]], dict[tuple[str, str], set[UUID]]]:
    endpoint_rows = (
        (
            await session.execute(
                select(KnowledgeNode).where(
                    KnowledgeNode.collection_id == collection_id,
                    KnowledgeNode.kind == KnowledgeNodeKind.API_ENDPOINT,
                )
            )
        )
        .scalars()
        .all()
    )
    by_path: dict[str, set[UUID]] = {}
    by_method_path: dict[tuple[str, str], set[UUID]] = {}
    for endpoint in endpoint_rows:
        meta = endpoint.meta or {}
        method = str(meta.get("method") or "").strip().lower() or None
        path = _normalize_endpoint_path(str(meta.get("path") or ""))
        if not path:
            name_method, name_path = _endpoint_path_from_name(str(endpoint.name or ""))
            method = method or name_method
            path = name_path
        if not path:
            continue
        by_path.setdefault(path, set()).add(endpoint.id)
        if method:
            by_method_path.setdefault((method, path), set()).add(endpoint.id)
    return by_path, by_method_path


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
    endpoint_path_index, endpoint_method_path_index = await _build_endpoint_path_indexes(
        session,
        collection_id=collection_id,
    )

    for extraction in extractions:
        route_ids_for_extraction: dict[str, UUID] = {}
        view_ids_for_extraction: dict[str, UUID] = {}
        for route in extraction.routes:
            route_key = f"ui_route:{route.path}"
            route_meta = {
                "file_path": route.file_path,
                "path": route.path,
                "view_name_hint": route.view_name_hint,
                **_provenance(
                    mode="inferred" if route.inferred else "deterministic",
                    extractor="ui.v1",
                    confidence=0.79 if route.inferred else 0.98,
                ),
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
            route_ids_for_extraction[route.path] = route_id
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
                **_provenance(
                    mode="inferred" if view.inferred else "deterministic",
                    extractor="ui.v1",
                    confidence=0.8 if view.inferred else 0.94,
                ),
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
            view_ids_for_extraction[view.name] = view_id
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

            linked_endpoint_ids: set[UUID] = set()
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
                        linked_endpoint_ids.add(endpoint_id)
                        stats["contract_edges"] += 1

            for endpoint_hint in view.endpoint_hints:
                method_hint, path_hint = _parse_endpoint_hint(endpoint_hint)
                if not path_hint:
                    continue
                endpoint_ids: set[UUID] = set(endpoint_path_index.get(path_hint, set()))
                if method_hint:
                    endpoint_ids.update(
                        endpoint_method_path_index.get((method_hint, path_hint), set())
                    )
                endpoint_ids.difference_update(linked_endpoint_ids)
                if not endpoint_ids:
                    continue

                method_token = method_hint or "any"
                contract_key = f"interface_contract:{view.name}:endpoint:{method_token}:{path_hint}"
                contract_meta = {
                    "source_view": view.name,
                    "endpoint_hint": endpoint_hint,
                    "endpoint_path": path_hint,
                    "endpoint_method": method_hint,
                    **_provenance(
                        mode="inferred", extractor="ui.v1.endpoint_hint", confidence=0.83
                    ),
                }
                contract_id = await _upsert_node(
                    session,
                    collection_id=collection_id,
                    kind=KnowledgeNodeKind.INTERFACE_CONTRACT,
                    natural_key=contract_key,
                    name=f"{view.name} endpoint contract",
                    meta=contract_meta,
                )
                stats["interface_contracts"] += 1
                for endpoint_id in endpoint_ids:
                    await _upsert_edge(
                        session,
                        collection_id=collection_id,
                        source_node_id=contract_id,
                        target_node_id=endpoint_id,
                        kind=KnowledgeEdgeKind.CONTRACT_GOVERNS_ENDPOINT,
                        meta=_provenance(
                            mode="inferred",
                            extractor="ui.v1.endpoint_hint",
                            confidence=0.82,
                            evidence_ids=[evidence_id],
                        ),
                    )
                    stats["contract_edges"] += 1

        if extraction.routes and extraction.views:
            default_view_name = extraction.views[0].name if len(extraction.views) == 1 else None
            views_by_name = {view.name: view for view in extraction.views}
            for route in extraction.routes:
                route_id = route_ids_for_extraction.get(route.path)
                if not route_id:
                    continue

                target_view_name = route.view_name_hint or default_view_name
                target_view_id = (
                    view_ids_for_extraction.get(target_view_name) if target_view_name else None
                )
                if target_view_id is None:
                    continue

                target_view = views_by_name.get(target_view_name) if target_view_name else None
                inferred = (
                    route.inferred
                    or route.view_name_hint is None
                    or bool(target_view and target_view.inferred)
                )
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

    return stats
