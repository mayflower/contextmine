"""Deterministic UI extractor and graph materializer."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any
from uuid import UUID

from contextmine_core.analyzer.extractors.ast_utils import (
    HTTP_CLIENT_NAMES,
    HTTP_METHOD_NAMES,
    first_child,
    first_string_argument,
    is_pascal_case,
    js_call_name,
    line_number,
    node_text,
    ruby_first_string_arg,
    string_literal,
    unquote,
    walk,
)
from contextmine_core.analyzer.extractors.graph_helpers import (
    create_node_evidence as _create_node_evidence,
)
from contextmine_core.analyzer.extractors.graph_helpers import (
    dedupe_strings as _dedupe,
)
from contextmine_core.analyzer.extractors.graph_helpers import (
    provenance as _provenance,
)
from contextmine_core.analyzer.extractors.graph_helpers import (
    upsert_edge as _upsert_edge,
)
from contextmine_core.analyzer.extractors.graph_helpers import (
    upsert_node as _upsert_node,
)
from contextmine_core.analyzer.extractors.traceability import (
    build_endpoint_symbol_index,
    resolve_symbol_refs_for_calls,
    symbol_token_variants,
)
from contextmine_core.models import (
    KnowledgeEdgeKind,
    KnowledgeNode,
    KnowledgeNodeKind,
)
from contextmine_core.treesitter.languages import TreeSitterLanguage, detect_language
from contextmine_core.treesitter.manager import get_treesitter_manager
from sqlalchemy import select

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


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
_TEMPLATE_EXT_RE = re.compile(
    r"\.(html|twig|blade|php|phtml|erb|haml|jinja2?|mustache|hbs|ejs)$",
    re.IGNORECASE,
)
_PATH_SRC = "/src/"
_PATH_ADMIN_ASSETS_SRC = "/admin/assets/src/"
_PATH_ASSETS_SRC = "/assets/src/"
_EXTRACTOR_UI_V1 = "ui.v1"
UI_RENDER_CALL_PATTERN = re.compile(
    r"(?i)(?:\bview|render(?:_template|Template)?|template|TemplateResponse)\s*\(\s*['\"](?P<name>[^'\"]+)['\"]"
)
_HTTP_METHODS = r"get|post|put|patch|delete|match|any"
_HTTP_METHODS_WITH_MAP = r"get|post|put|patch|delete|match|any|map"
_PHP_VAR_OR_ROUTER = r"\$[a-z_]\w*|app|router|route"
_QUOTED_PATH = r"['\"](?P<path>[^'\"]+)['\"]"
UI_ROUTE_PATTERNS = (
    re.compile(rf"(?i)\bRoute::(?:{_HTTP_METHODS})\s*\(\s*{_QUOTED_PATH}"),
    re.compile(
        rf"(?i)\b(?:{_PHP_VAR_OR_ROUTER})\s*->\s*(?:{_HTTP_METHODS_WITH_MAP})\s*\(\s*{_QUOTED_PATH}"
    ),
    re.compile(r"(?i)#\[\s*Route\s*\(\s*(?:path\s*[:=]\s*)?['\"](?P<path>[^'\"]+)['\"]"),
    re.compile(r"(?i)@Route\s*\(\s*(?:path\s*[:=]\s*)?['\"](?P<path>[^'\"]+)['\"]"),
    re.compile(r"(?i)\b(?:path|re_path)\s*\(\s*r?['\"](?P<path>[^'\"]+)['\"]"),
    re.compile(r"(?i)\b(?:get|post|put|patch|delete)\s+['\"](?P<path>/[^'\"]+)['\"]"),
)
UI_ENDPOINT_HINT_PATTERNS = (
    re.compile(
        r"(?i)\b(?:fetch|(?:axios|api|client)\.(?:get|post|put|patch|delete))\s*\(\s*['\"](?P<path>[^'\"]+)['\"]"
    ),
    re.compile(r"(?i)\baction\s*=\s*['\"](?P<path>[^'\"]+)['\"]"),
    re.compile(r"(?i)\bdata-(?:url|endpoint)\s*=\s*['\"](?P<path>[^'\"]+)['\"]"),
)
UI_NAVIGATION_HINT_PATTERN = re.compile(r"(?i)\bhref\s*=\s*['\"](?P<path>[^'\"]+)['\"]")
UI_COMPONENT_TAG_PATTERN = re.compile(r"<(?P<tag>[A-Za-z][A-Za-z0-9:_-]{2,})\b")
UI_SYMBOL_HINT_PATTERN = re.compile(
    r"\b(?P<symbol>[A-Z]\w+Controller::\w+|[A-Z]\w*(?:Service|Controller|Handler|Action)\b)"
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
        normalized = _TEMPLATE_EXT_RE.sub("", stem)
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
    return _PATH_ADMIN_ASSETS_SRC in normalized


def _split_relative_path(normalized: str, lower: str) -> tuple[str, str] | None:
    """Extract (base_prefix, relative) from a normalised file path, or None if not UI."""
    if _PATH_ADMIN_ASSETS_SRC in lower:
        idx = lower.index(_PATH_ADMIN_ASSETS_SRC)
        return "/admin", normalized[idx + len(_PATH_ADMIN_ASSETS_SRC) :]
    if _PATH_ASSETS_SRC in lower:
        idx = lower.index(_PATH_ASSETS_SRC)
        prefix = "/admin" if "/admin/" in lower[:idx] else ""
        return prefix, normalized[idx + len(_PATH_ASSETS_SRC) :]
    if _PATH_SRC in lower:
        idx = lower.index(_PATH_SRC)
        return "", normalized[idx + len(_PATH_SRC) :]
    if "/ui/" in lower:
        idx = lower.index("/ui/")
        return "", normalized[idx + len("/ui/") :]
    return None


def _route_path_from_file(file_path: str) -> str | None:
    normalized = file_path.replace("\\", "/")
    lower = normalized.lower()
    split_result = _split_relative_path(normalized, lower)
    if split_result is None:
        return None
    base_prefix, relative = split_result

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

    filtered = [
        token
        for idx, segment in enumerate(segments)
        if (token := segment.strip().lower())
        and not (idx == 0 and token in UI_STRIP_ROUTE_SEGMENTS)
        and token not in {"index", "."}
    ]

    if not filtered:
        return base_prefix or "/"

    route = "/".join(filtered)
    if base_prefix:
        route = f"{base_prefix.rstrip('/')}/{route}".rstrip("/")
    return f"/{route.strip('/')}"


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
    candidate = _TEMPLATE_EXT_RE.sub("", candidate)
    candidate = candidate.replace(".", "/").replace(":", "/")
    parts = [part for part in re.split(r"[/\\]+", candidate) if part]
    if not parts:
        return None
    tail = parts[-1]
    if tail.lower() in {"index", "main", "default"} and len(parts) > 1:
        tail = parts[-2]
    tail = _TEMPLATE_EXT_RE.sub("", tail)
    value = _to_pascal_case(tail)
    return value if is_pascal_case(value) else None


def _is_template_like_file(file_path: str) -> bool:
    normalized = file_path.replace("\\", "/").lower()
    return normalized.endswith(UI_TEMPLATE_FILE_SUFFIXES) and any(
        token in normalized for token in UI_TEMPLATE_PATH_HINTS + UI_JS_PATH_HINTS
    )


def _scan_routes_from_patterns(file_path: str, content: str) -> list[UIRouteDef]:
    """Scan content for explicit route definitions from regex patterns."""
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
    return routes


def _infer_fallback_route(file_path: str) -> UIRouteDef | None:
    """Create an inferred route from the file path if it looks like a route module."""
    if not _looks_like_route_module(file_path):
        return None
    inferred_route = _route_path_from_file(file_path)
    normalized_route = _normalize_ui_route_path(inferred_route or "")
    if not normalized_route:
        return None
    return UIRouteDef(
        path=normalized_route,
        file_path=file_path,
        line=1,
        view_name_hint=_view_name_from_file_path(file_path),
        inferred=True,
    )


def _scan_view_candidates(file_path: str, content: str) -> dict[str, UIViewDef]:
    """Scan for template and render-call view candidates."""
    view_candidates: dict[str, UIViewDef] = {}
    if _is_template_like_file(file_path):
        name = _view_name_from_file_path(file_path)
        if name:
            view_candidates[name] = UIViewDef(name=name, file_path=file_path, line=1, inferred=True)
    for render_match in UI_RENDER_CALL_PATTERN.finditer(content):
        view_name = _normalize_view_hint(render_match.groupdict().get("name") or "")
        if view_name and view_name not in view_candidates:
            view_candidates[view_name] = UIViewDef(
                name=view_name,
                file_path=file_path,
                line=_line_number_for_offset(content, render_match.start()),
                inferred=True,
            )
    return view_candidates


def _scan_component_tags(content: str) -> list[str]:
    """Extract component names from HTML/JSX tags in content."""
    components: list[str] = []
    for tag_match in UI_COMPONENT_TAG_PATTERN.finditer(content):
        tag = tag_match.groupdict().get("tag") or ""
        if ":" in tag:
            tag = tag.split(":", 1)[1]
        if "-" in tag:
            components.append(_to_pascal_case(tag))
        elif is_pascal_case(tag):
            components.append(tag)
    return components


def _scan_navigation_and_endpoints(
    content: str,
) -> tuple[list[str], list[str]]:
    """Extract endpoint hints and navigation targets from content."""
    endpoint_hints: list[str] = []
    navigation_targets: list[str] = []
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
    return endpoint_hints, navigation_targets


def _extract_ui_heuristic(file_path: str, content: str) -> UIExtraction:
    extraction = UIExtraction(file_path=file_path)

    routes = _scan_routes_from_patterns(file_path, content)
    if not routes:
        fallback = _infer_fallback_route(file_path)
        if fallback:
            routes.append(fallback)

    view_candidates = _scan_view_candidates(file_path, content)

    components = _scan_component_tags(content)
    symbol_hints = [
        m.groupdict().get("symbol") or "" for m in UI_SYMBOL_HINT_PATTERN.finditer(content)
    ]
    endpoint_hints, navigation_targets = _scan_navigation_and_endpoints(content)

    if not view_candidates and routes:
        hinted_names = [route.view_name_hint for route in routes if route.view_name_hint]
        if len(hinted_names) == 1 and hinted_names[0]:
            only_name = hinted_names[0]
            view_candidates[only_name] = UIViewDef(
                name=only_name, file_path=file_path, line=1, inferred=True
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


# Languages that support AST-based route extraction via decorators/annotations
_BACKEND_ROUTE_LANGUAGES = {
    TreeSitterLanguage.PYTHON,
    TreeSitterLanguage.JAVA,
    TreeSitterLanguage.GO,
    TreeSitterLanguage.PHP,
    TreeSitterLanguage.RUBY,
    TreeSitterLanguage.CSHARP,
}


def extract_ui_from_file(file_path: str, content: str) -> UIExtraction:
    """Extract routes, views, and UI composition hints from one file."""
    if looks_like_ui_test_file(file_path):
        return UIExtraction(file_path=file_path)

    language = detect_language(file_path)

    if language in _BACKEND_ROUTE_LANGUAGES:
        return _extract_ui_from_backend(file_path, content, language)

    if language not in JS_UI_LANGUAGES:
        return _extract_ui_heuristic(file_path, content)

    return _extract_ui_from_js(file_path, content)


def _extract_ui_from_backend(
    file_path: str, content: str, language: TreeSitterLanguage
) -> UIExtraction:
    """Extract UI from a backend language file."""
    ast_routes = _extract_backend_routes_ast(file_path, content, language)
    heuristic = _extract_ui_heuristic(file_path, content)
    if ast_routes:
        extraction = UIExtraction(file_path=file_path)
        extraction.routes = ast_routes
        extraction.views = heuristic.views
        return extraction
    return heuristic


def _extract_ui_from_js(file_path: str, content: str) -> UIExtraction:
    """Extract UI from a JS/TS/TSX file using AST."""
    extraction = UIExtraction(file_path=file_path)
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
        inferred_route = _infer_first_route_for_views(file_path, extraction.views)
        if inferred_route is not None:
            extraction.routes.append(inferred_route)

    if not extraction.routes and not extraction.views:
        return _extract_ui_heuristic(file_path, content)

    return extraction


def _infer_first_route_for_views(file_path: str, views: list[UIViewDef]) -> UIRouteDef | None:
    """Try to infer a route from the first view that yields one."""
    for view in views:
        inferred_route = _infer_route_for_view(file_path, view)
        if inferred_route is not None:
            return inferred_route
    return None


# ---------------------------------------------------------------------------
# AST-based route extraction for backend languages
# ---------------------------------------------------------------------------


_BACKEND_ROUTE_EXTRACTORS: dict[TreeSitterLanguage, Any] = {}


def _init_backend_route_extractors() -> dict[TreeSitterLanguage, Any]:
    """Lazily initialize the backend route extractor dispatch map."""
    if not _BACKEND_ROUTE_EXTRACTORS:
        _BACKEND_ROUTE_EXTRACTORS.update(
            {
                TreeSitterLanguage.JAVA: _extract_java_routes,
                TreeSitterLanguage.PYTHON: _extract_python_routes,
                TreeSitterLanguage.GO: _extract_go_routes,
                TreeSitterLanguage.PHP: _extract_php_routes,
                TreeSitterLanguage.RUBY: _extract_ruby_routes,
                TreeSitterLanguage.CSHARP: _extract_csharp_routes,
            }
        )
    return _BACKEND_ROUTE_EXTRACTORS


def _extract_backend_routes_ast(
    file_path: str,
    content: str,
    language: TreeSitterLanguage,
) -> list[UIRouteDef]:
    """Extract routes from backend code using tree-sitter AST."""
    manager = get_treesitter_manager()
    try:
        tree = manager.parse(file_path, content)
    except Exception:
        return []

    extractors = _init_backend_route_extractors()
    extractor = extractors.get(language)
    if extractor is None:
        return []
    return extractor(file_path, content, tree.root_node)


def _extract_java_routes(
    file_path: str,
    content: str,
    root: Any,
) -> list[UIRouteDef]:
    """Extract Spring Boot routes from Java annotations.

    Handles: @RequestMapping, @GetMapping, @PostMapping, @PutMapping,
    @DeleteMapping, @PatchMapping, @Route (JAX-RS/Vaadin)
    """
    routes: list[UIRouteDef] = []
    spring_mappings = {
        "requestmapping",
        "getmapping",
        "postmapping",
        "putmapping",
        "deletemapping",
        "patchmapping",
        "route",
    }
    # Class-level prefix from @RequestMapping
    class_prefix = ""
    for node in walk(root):
        if node.type == "class_declaration":
            class_prefix = _java_class_route_prefix(content, node) or ""
        if node.type not in {"marker_annotation", "annotation"}:
            continue
        route = _try_java_annotation_route(content, node, file_path, class_prefix, spring_mappings)
        if route is not None:
            routes.append(route)
    return routes


def _try_java_annotation_route(
    content: str,
    node: Any,
    file_path: str,
    class_prefix: str,
    spring_mappings: set[str],
) -> UIRouteDef | None:
    """Try to extract a route from a Java annotation node."""
    ann_name_node = node.child_by_field_name("name")
    if ann_name_node is None:
        for child in node.children:
            if child.type == "identifier":
                ann_name_node = child
                break
    if ann_name_node is None:
        return None
    ann_name = node_text(content, ann_name_node).strip().lower()
    if ann_name not in spring_mappings:
        return None
    path = _java_annotation_string_value(content, node) or ""
    full_path = f"{class_prefix.rstrip('/')}/{path.lstrip('/')}".rstrip("/") or "/"
    normalized = _normalize_ui_route_path(full_path)
    if not normalized:
        return None
    return UIRouteDef(path=normalized, file_path=file_path, line=line_number(node))


def _java_class_route_prefix(content: str, class_node: Any) -> str | None:
    """Extract @RequestMapping value from a class declaration's modifiers."""
    for child in class_node.children:
        if child.type != "modifiers":
            continue
        for mod in child.children:
            if mod.type not in {"marker_annotation", "annotation"}:
                continue
            if _is_request_mapping_annotation(content, mod):
                return _java_annotation_string_value(content, mod) or ""
    return None


def _is_request_mapping_annotation(content: str, annotation_node: Any) -> bool:
    """Check if a Java annotation is @RequestMapping."""
    name_node = annotation_node.child_by_field_name("name")
    if name_node is None:
        for c in annotation_node.children:
            if c.type == "identifier":
                name_node = c
                break
    return (
        name_node is not None and node_text(content, name_node).strip().lower() == "requestmapping"
    )


def _java_annotation_string_value(content: str, annotation_node: Any) -> str | None:
    """Extract the first string literal value from a Java annotation."""
    args = annotation_node.child_by_field_name("arguments")
    if args is None:
        # Check for annotation_argument_list
        for child in annotation_node.children:
            if child.type == "annotation_argument_list":
                args = child
                break
    if args is None:
        return None
    for child in walk(args):
        if child.type == "string_literal":
            text = node_text(content, child).strip()
            return text.strip('"')
    return None


_FLASK_FASTAPI_METHODS = {"route", "get", "post", "put", "delete", "patch", "head", "options"}


def _extract_python_routes(
    file_path: str,
    content: str,
    root: Any,
) -> list[UIRouteDef]:
    """Extract Flask/FastAPI/Django routes from Python decorators.

    Handles: @app.route(), @app.get(), @router.post(), url patterns
    """
    routes: list[UIRouteDef] = []
    for node in walk(root):
        if node.type != "decorator":
            continue
        route = _try_python_decorator_route(content, node, file_path)
        if route is not None:
            routes.append(route)
    return routes


def _try_python_decorator_route(
    content: str, decorator_node: Any, file_path: str
) -> UIRouteDef | None:
    """Try to extract a route from a Python decorator node."""
    for child in decorator_node.children:
        if child.type != "call":
            continue
        func = child.child_by_field_name("function")
        if func is None or func.type != "attribute":
            continue
        attr = func.child_by_field_name("attribute")
        attr_name = node_text(content, attr).strip().lower() if attr else ""
        if attr_name not in _FLASK_FASTAPI_METHODS:
            continue
        path = _python_first_string_arg(content, child)
        if not path:
            continue
        normalized = _normalize_ui_route_path(path)
        if normalized:
            return UIRouteDef(
                path=normalized, file_path=file_path, line=line_number(decorator_node)
            )
    return None


def _python_first_string_arg(content: str, call_node: Any) -> str | None:
    """Extract the first string argument from a Python call."""
    args = call_node.child_by_field_name("arguments")
    if args is None:
        return None
    for child in args.children:
        if child.type == "string":
            text = node_text(content, child).strip()
            return text.strip("'\"")
    return None


_GO_HTTP_METHODS = {
    "get",
    "post",
    "put",
    "delete",
    "patch",
    "head",
    "options",
    "handle",
    "handlefunc",
    "any",
    "match",
}


def _extract_go_routes(
    file_path: str,
    content: str,
    root: Any,
) -> list[UIRouteDef]:
    """Extract Go HTTP routes from common frameworks.

    Handles: http.HandleFunc(), e.GET(), r.Get(), router.Handle(), mux patterns
    """
    routes: list[UIRouteDef] = []
    for node in walk(root):
        if node.type != "call_expression":
            continue
        route = _try_go_call_route(content, node, file_path)
        if route is not None:
            routes.append(route)
    return routes


def _try_go_call_route(content: str, node: Any, file_path: str) -> UIRouteDef | None:
    """Try to extract a route from a Go call_expression."""
    func = node.child_by_field_name("function")
    if func is None:
        return None
    method_name = ""
    if func.type == "selector_expression":
        field = func.child_by_field_name("field")
        method_name = node_text(content, field).strip() if field else ""
    elif func.type == "identifier":
        method_name = node_text(content, func).strip()
    if method_name.lower() not in _GO_HTTP_METHODS:
        return None
    args = node.child_by_field_name("arguments")
    if args is None:
        return None
    return _go_first_string_route(content, args, file_path, node)


def _go_first_string_route(content: str, args: Any, file_path: str, node: Any) -> UIRouteDef | None:
    """Extract the first string literal argument as a route path."""
    for child in args.children:
        if child.type == "interpreted_string_literal":
            path = node_text(content, child).strip().strip('"')
            if path.startswith("/"):
                normalized = _normalize_ui_route_path(path)
                if normalized:
                    return UIRouteDef(path=normalized, file_path=file_path, line=line_number(node))
            break
    return None


_PHP_ROUTE_ATTRS = {"route", "get", "post", "put", "delete", "patch"}
_PHP_ROUTE_STATIC_METHODS = {"get", "post", "put", "delete", "patch", "match", "any"}


def _extract_php_routes(
    file_path: str,
    content: str,
    root: Any,
) -> list[UIRouteDef]:
    """Extract PHP routes from Laravel/Symfony patterns.

    Handles: Route::get(), $router->get(), #[Route()] attributes
    """
    routes: list[UIRouteDef] = []
    for node in walk(root):
        if node.type == "attribute":
            route = _try_php_attribute_route(content, node, file_path)
            if route is not None:
                routes.append(route)
        if node.type == "scoped_call_expression":
            route = _try_php_scoped_route(content, node, file_path)
            if route is not None:
                routes.append(route)
    return routes


def _try_php_attribute_route(content: str, node: Any, file_path: str) -> UIRouteDef | None:
    """Try to extract a route from a PHP 8 attribute."""
    attr_name = ""
    for child in node.children:
        if child.type in {"name", "identifier"}:
            attr_name = node_text(content, child).strip()
            break
    if attr_name.lower() not in _PHP_ROUTE_ATTRS:
        return None
    path = _php_attribute_string_value(content, node)
    if not path:
        return None
    normalized = _normalize_ui_route_path(path)
    if not normalized:
        return None
    return UIRouteDef(path=normalized, file_path=file_path, line=line_number(node))


def _try_php_scoped_route(content: str, node: Any, file_path: str) -> UIRouteDef | None:
    """Try to extract a route from a Route::method() static call."""
    scope = node.child_by_field_name("scope")
    name = node.child_by_field_name("name")
    scope_text = node_text(content, scope).strip() if scope else ""
    name_text = node_text(content, name).strip().lower() if name else ""
    if scope_text != "Route" or name_text not in _PHP_ROUTE_STATIC_METHODS:
        return None
    args = node.child_by_field_name("arguments")
    if not args:
        return None
    path = _php_first_string_arg(content, args)
    if not path:
        return None
    normalized = _normalize_ui_route_path(path)
    if not normalized:
        return None
    view_hint = _php_find_view_hint(content, node)
    return UIRouteDef(
        path=normalized, file_path=file_path, line=line_number(node), view_name_hint=view_hint
    )


def _php_find_view_hint(content: str, route_node: Any) -> str | None:
    """Search for view('template.name') calls within a PHP route handler."""
    for child in walk(route_node):
        if child.type == "function_call_expression":
            fn_name_node = child.child_by_field_name("function")
            if fn_name_node is None:
                for c in child.children:
                    if c.type in {"name", "identifier"}:
                        fn_name_node = c
                        break
            fn_name = node_text(content, fn_name_node).strip() if fn_name_node else ""
            if fn_name in {"view", "render"}:
                args = child.child_by_field_name("arguments")
                if args:
                    template_name = _php_first_string_arg(content, args)
                    if template_name:
                        return _normalize_view_hint(template_name)
    return None


def _php_attribute_string_value(content: str, attr_node: Any) -> str | None:
    """Extract first string from a PHP attribute's arguments."""
    for child in walk(attr_node):
        if child.type in {"string", "encapsed_string"}:
            text = node_text(content, child).strip()
            return text.strip("'\"")
    return None


def _php_first_string_arg(content: str, args_node: Any) -> str | None:
    """Extract first string argument from PHP function arguments."""
    for child in args_node.children:
        if child.type in {"string", "encapsed_string"}:
            text = node_text(content, child).strip()
            return text.strip("'\"")
        if child.type == "argument":
            for inner in child.children:
                if inner.type in {"string", "encapsed_string"}:
                    text = node_text(content, inner).strip()
                    return text.strip("'\"")
    return None


_RAILS_METHODS = {"get", "post", "put", "patch", "delete", "match", "root"}


def _extract_ruby_routes(
    file_path: str,
    content: str,
    root: Any,
) -> list[UIRouteDef]:
    """Extract Ruby on Rails routes from DSL.

    Handles: get '/path', post '/path', resources :name, match '/path'
    """
    routes: list[UIRouteDef] = []
    for node in walk(root):
        if node.type != "call":
            continue
        route = _try_ruby_call_route(content, node, file_path)
        if route is not None:
            routes.append(route)
    return routes


def _ruby_call_method_name(content: str, node: Any) -> str:
    """Extract the method name from a Ruby call node."""
    method_node = node.child_by_field_name("method")
    if method_node is None:
        for child in node.children:
            if child.type == "identifier":
                method_node = child
                break
    return node_text(content, method_node).strip() if method_node else ""


def _try_ruby_call_route(content: str, node: Any, file_path: str) -> UIRouteDef | None:
    """Try to extract a route from a Ruby call node."""
    method_name = _ruby_call_method_name(content, node)

    if method_name in _RAILS_METHODS:
        path = _ruby_first_string_arg(content, node)
        if method_name == "root" and not path:
            path = "/"
        if not path:
            return None
        normalized = _normalize_ui_route_path(path)
        if normalized:
            return UIRouteDef(path=normalized, file_path=file_path, line=line_number(node))
    elif method_name in {"resources", "resource"}:
        symbol = _ruby_first_symbol_arg(content, node)
        if symbol:
            normalized = _normalize_ui_route_path(f"/{symbol}")
            if normalized:
                return UIRouteDef(path=normalized, file_path=file_path, line=line_number(node))
    return None


def _ruby_first_string_arg(content: str, call_node: Any) -> str | None:
    return ruby_first_string_arg(content, call_node)


def _ruby_first_symbol_arg(content: str, call_node: Any) -> str | None:
    """Extract first symbol argument from a Ruby call (:users -> 'users')."""
    args = call_node.child_by_field_name("arguments")
    if args is None:
        for child in call_node.children:
            if child.type == "argument_list":
                args = child
                break
    if args is None:
        return None
    for child in args.children:
        if child.type in {"simple_symbol", "symbol"}:
            text = node_text(content, child).strip()
            return text.lstrip(":")
    return None


_CSHARP_ROUTE_ATTRS = {
    "route",
    "httpget",
    "httppost",
    "httpput",
    "httpdelete",
    "httppatch",
    "httphead",
}


def _extract_csharp_routes(
    file_path: str,
    content: str,
    root: Any,
) -> list[UIRouteDef]:
    """Extract ASP.NET routes from C# attributes.

    Handles: [Route], [HttpGet], [HttpPost], [ApiController] with route templates
    """
    routes: list[UIRouteDef] = []
    controller_prefix = ""
    for node in walk(root):
        if node.type == "class_declaration":
            controller_prefix = _csharp_class_route_prefix(content, node) or ""
        if node.type != "attribute":
            continue
        route = _try_csharp_attribute_route(content, node, file_path, controller_prefix)
        if route is not None:
            routes.append(route)
    return routes


def _try_csharp_attribute_route(
    content: str, node: Any, file_path: str, controller_prefix: str
) -> UIRouteDef | None:
    """Try to extract a route from a C# attribute node."""
    attr_name_node = None
    for child in node.children:
        if child.type in {"identifier", "name"}:
            attr_name_node = child
            break
    if attr_name_node is None:
        return None
    attr_name = node_text(content, attr_name_node).strip().lower()
    if attr_name.endswith("attribute"):
        attr_name = attr_name[: -len("attribute")]
    if attr_name not in _CSHARP_ROUTE_ATTRS:
        return None
    path = _csharp_attribute_string_value(content, node) or ""
    if controller_prefix and not path.startswith("/"):
        full_path = f"{controller_prefix.rstrip('/')}/{path.lstrip('/')}"
    else:
        full_path = path
    full_path = full_path.rstrip("/") or "/"
    full_path = full_path.replace("[controller]", ":controller").replace("[action]", ":action")
    normalized = _normalize_ui_route_path(full_path)
    if not normalized:
        return None
    return UIRouteDef(path=normalized, file_path=file_path, line=line_number(node))


def _csharp_class_route_prefix(content: str, class_node: Any) -> str | None:
    """Extract [Route] prefix from a C# class."""
    for child in class_node.children:
        if child.type != "attribute_list":
            continue
        for attr in walk(child):
            if attr.type != "attribute":
                continue
            if _is_csharp_route_attribute(content, attr):
                return _csharp_attribute_string_value(content, attr)
    return None


def _is_csharp_route_attribute(content: str, attr_node: Any) -> bool:
    """Check if a C# attribute is a Route or RoutePrefix attribute."""
    for c in attr_node.children:
        if c.type in {"identifier", "name"}:
            return node_text(content, c).strip().lower() in {"route", "routeprefix"}
    return False


def _csharp_attribute_string_value(content: str, attr_node: Any) -> str | None:
    """Extract first string from a C# attribute's arguments."""
    for child in walk(attr_node):
        if child.type == "string_literal":
            text = node_text(content, child).strip()
            return text.strip('"')
    return None


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


_ROUTE_EXTRACTOR_DISPATCH: dict[str, Any] = {
    "jsx_self_closing_element": "_route_from_jsx",
    "jsx_opening_element": "_route_from_jsx",
    "object": "_route_from_object",
    "call_expression": "_route_from_router_call",
}


def _try_extract_route_from_node(file_path: str, content: str, node: Any) -> UIRouteDef | None:
    """Dispatch to the right route extractor based on AST node type."""
    if node.type in {"jsx_self_closing_element", "jsx_opening_element"}:
        return _route_from_jsx(file_path, content, node)
    if node.type == "object":
        return _route_from_object(file_path, content, node)
    if node.type == "call_expression":
        return _route_from_router_call(file_path, content, node)
    return None


def _extract_routes(file_path: str, content: str, root: Any) -> list[UIRouteDef]:
    routes: list[UIRouteDef] = []
    for node in walk(root):
        route = _try_extract_route_from_node(file_path, content, node)
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
    if method_name.lower() not in HTTP_METHOD_NAMES:
        return None
    if "router" not in base_name.lower():
        return None
    route_path = _first_string_argument(content, node)
    if not route_path:
        return None
    return UIRouteDef(path=route_path, file_path=file_path, line=line_number(node))


def _try_extract_view_candidate(
    content: str,
    node: Any,
) -> tuple[str | None, Any | None]:
    """Try to extract a view name and signal node from an AST node."""
    if node.type == "function_declaration":
        ident = first_child(node, "identifier")
        candidate = node_text(content, ident).strip()
        if is_pascal_case(candidate):
            return candidate, node
    elif node.type == "class_declaration":
        ident = first_child(node, "type_identifier") or first_child(node, "identifier")
        candidate = node_text(content, ident).strip()
        if is_pascal_case(candidate):
            return candidate, node
    elif node.type == "variable_declarator":
        ident = first_child(node, "identifier")
        value = node.child_by_field_name("value")
        candidate = node_text(content, ident).strip()
        if (
            is_pascal_case(candidate)
            and value is not None
            and value.type in {"arrow_function", "function_expression"}
        ):
            return candidate, value
    return None, None


def _extract_views(file_path: str, content: str, root: Any) -> list[UIViewDef]:
    views: dict[str, UIViewDef] = {}
    for node in walk(root):
        view_name, signal_node = _try_extract_view_candidate(content, node)
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


def _process_call_expression_signals(
    content: str,
    node: Any,
    symbol_hints: list[str],
    endpoint_hints: list[str],
    navigation_targets: list[str],
    call_sites: list[dict[str, Any]],
) -> None:
    """Extract symbol, endpoint, and navigation signals from a call_expression node."""
    full_name, base_name, method_name = _js_call_name(content, node)
    callee = full_name or method_name or base_name
    if callee:
        call_sites.append(
            {"line": line_number(node), "column": int(node.start_point[1]), "callee": callee}
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
        _process_call_expression_signals(
            content, node, symbol_hints, endpoint_hints, navigation_targets, call_sites
        )
    return (
        _dedupe(components),
        _dedupe(symbol_hints),
        _dedupe(endpoint_hints),
        _dedupe(navigation_targets),
        call_sites,
    )


def _contains_jsx(node: Any) -> bool:
    return any(child.type in {"jsx_element", "jsx_self_closing_element"} for child in walk(node))


_js_call_name = js_call_name


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


_first_string_argument = first_string_argument
_string_literal = string_literal


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


def _pascal_from_jsx_expression(content: str, expr_node: Any) -> str | None:
    """Search a jsx_expression subtree for a PascalCase identifier or JSX tag."""
    for nested in walk(expr_node):
        if nested.type in {"jsx_self_closing_element", "jsx_opening_element"}:
            tag = _jsx_tag_name(content, nested)
            if is_pascal_case(tag):
                return tag
        if nested.type == "identifier":
            token = node_text(content, nested).strip()
            if is_pascal_case(token):
                return token
    return None


def _jsx_attribute_view_hint(content: str, attr_node: Any) -> str | None:
    for child in attr_node.children:
        if child.type == "identifier":
            name = node_text(content, child).strip()
            if is_pascal_case(name):
                return name
        if child.type == "jsx_expression":
            result = _pascal_from_jsx_expression(content, child)
            if result:
                return result
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


_parse_endpoint_hint = _endpoint_path_from_name


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
        route_ids_for_extraction = await _persist_routes(session, collection_id, extraction, stats)
        view_ids_for_extraction = await _persist_views(
            session,
            collection_id,
            extraction,
            source_id,
            endpoint_symbol_index,
            endpoint_path_index,
            endpoint_method_path_index,
            stats,
        )
        await _link_routes_to_views(
            session,
            collection_id,
            extraction,
            route_ids_for_extraction,
            view_ids_for_extraction,
            stats,
        )

    return stats


async def _persist_routes(
    session: AsyncSession,
    collection_id: UUID,
    extraction: UIExtraction,
    stats: dict[str, int],
) -> dict[str, UUID]:
    """Persist UI route nodes. Returns a mapping of route path to node ID."""
    route_ids: dict[str, UUID] = {}

    for route in extraction.routes:
        route_key = f"ui_route:{route.path}"
        route_meta = {
            "file_path": route.file_path,
            "path": route.path,
            "view_name_hint": route.view_name_hint,
            **_provenance(
                mode="inferred" if route.inferred else "deterministic",
                extractor=_EXTRACTOR_UI_V1,
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
            start_line=route.line,
            snippet=f"route {route.path}",
        )
        route_meta["provenance"]["evidence_ids"] = [evidence_id]
        await _upsert_node(
            session,
            collection_id=collection_id,
            kind=KnowledgeNodeKind.UI_ROUTE,
            natural_key=route_key,
            name=route.path,
            meta=route_meta,
        )
        route_ids[route.path] = route_id
        stats["ui_routes"] += 1

    return route_ids


async def _persist_views(
    session: AsyncSession,
    collection_id: UUID,
    extraction: UIExtraction,
    source_id: UUID | None,
    endpoint_symbol_index: dict[str, set[UUID]],
    endpoint_path_index: dict[str, set[UUID]],
    endpoint_method_path_index: dict[tuple[str, str], set[UUID]],
    stats: dict[str, int],
) -> dict[str, UUID]:
    """Persist UI view nodes, components, contracts, and endpoint links.

    Returns a mapping of view name to node ID.
    """
    view_ids: dict[str, UUID] = {}

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
                extractor=_EXTRACTOR_UI_V1,
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
            start_line=view.line,
            snippet=f"view {view.name}",
        )
        view_meta["provenance"]["evidence_ids"] = [evidence_id]
        await _upsert_node(
            session,
            collection_id=collection_id,
            kind=KnowledgeNodeKind.UI_VIEW,
            natural_key=view_key,
            name=view.name,
            meta=view_meta,
        )
        view_ids[view.name] = view_id
        stats["ui_views"] += 1

        for component_name in view.components:
            component_key = f"ui_component:{view.file_path}:{component_name}"
            component_meta = {
                "file_path": view.file_path,
                **_provenance(mode="deterministic", extractor=_EXTRACTOR_UI_V1, confidence=0.88),
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
                    extractor=_EXTRACTOR_UI_V1,
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

        linked_endpoint_ids = await _link_symbol_contracts(
            session,
            collection_id=collection_id,
            view=view,
            resolved_symbol_refs=resolved_symbol_refs,
            endpoint_symbol_index=endpoint_symbol_index,
            evidence_id=evidence_id,
            stats=stats,
        )

        await _link_endpoint_hint_contracts(
            session,
            collection_id=collection_id,
            view=view,
            endpoint_path_index=endpoint_path_index,
            endpoint_method_path_index=endpoint_method_path_index,
            linked_endpoint_ids=linked_endpoint_ids,
            evidence_id=evidence_id,
            stats=stats,
        )

    return view_ids


async def _link_symbol_contracts(
    session: AsyncSession,
    *,
    collection_id: UUID,
    view: UIViewDef,
    resolved_symbol_refs: list[Any],
    endpoint_symbol_index: dict[str, set[UUID]],
    evidence_id: Any,
    stats: dict[str, int],
) -> set[UUID]:
    """Create INTERFACE_CONTRACT nodes from resolved symbol refs and link to endpoints."""
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
                extractor=f"{_EXTRACTOR_UI_V1}.{ref.engine}",
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
        for token in symbol_token_variants(ref.symbol_name):
            for endpoint_id in endpoint_symbol_index.get(token, set()):
                await _upsert_edge(
                    session,
                    collection_id=collection_id,
                    source_node_id=contract_id,
                    target_node_id=endpoint_id,
                    kind=KnowledgeEdgeKind.CONTRACT_GOVERNS_ENDPOINT,
                    meta=_provenance(
                        mode="inferred",
                        extractor=f"{_EXTRACTOR_UI_V1}.endpoint.{ref.engine}",
                        confidence=max(0.66, ref.confidence - 0.08),
                        evidence_ids=[evidence_id],
                    ),
                )
                linked_endpoint_ids.add(endpoint_id)
                stats["contract_edges"] += 1
    return linked_endpoint_ids


async def _link_endpoint_hint_contracts(
    session: AsyncSession,
    *,
    collection_id: UUID,
    view: UIViewDef,
    endpoint_path_index: dict[str, set[UUID]],
    endpoint_method_path_index: dict[tuple[str, str], set[UUID]],
    linked_endpoint_ids: set[UUID],
    evidence_id: Any,
    stats: dict[str, int],
) -> None:
    """Create INTERFACE_CONTRACT nodes from endpoint hints and link to endpoints."""
    for endpoint_hint in view.endpoint_hints:
        method_hint, path_hint = _parse_endpoint_hint(endpoint_hint)
        if not path_hint:
            continue
        endpoint_ids: set[UUID] = set(endpoint_path_index.get(path_hint, set()))
        if method_hint:
            endpoint_ids.update(endpoint_method_path_index.get((method_hint, path_hint), set()))
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
                mode="inferred",
                extractor=f"{_EXTRACTOR_UI_V1}.endpoint_hint",
                confidence=0.83,
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
                    extractor=f"{_EXTRACTOR_UI_V1}.endpoint_hint",
                    confidence=0.82,
                    evidence_ids=[evidence_id],
                ),
            )
            stats["contract_edges"] += 1


def _is_inferred_route_link(route: UIRouteDef, target_view: UIViewDef | None) -> bool:
    """Determine if a route-to-view link is inferred."""
    return (
        route.inferred or route.view_name_hint is None or bool(target_view and target_view.inferred)
    )


async def _link_routes_to_views(
    session: AsyncSession,
    collection_id: UUID,
    extraction: UIExtraction,
    route_ids: dict[str, UUID],
    view_ids: dict[str, UUID],
    stats: dict[str, int],
) -> None:
    """Create edges linking UI routes to the views they render."""
    if not extraction.routes or not extraction.views:
        return

    default_view_name = extraction.views[0].name if len(extraction.views) == 1 else None
    views_by_name = {view.name: view for view in extraction.views}
    for route in extraction.routes:
        route_id = route_ids.get(route.path)
        if not route_id:
            continue

        target_view_name = route.view_name_hint or default_view_name
        if not target_view_name:
            continue
        target_view_id = view_ids.get(target_view_name)
        if target_view_id is None:
            continue

        target_view = views_by_name.get(target_view_name)
        inferred = _is_inferred_route_link(route, target_view)
        await _upsert_edge(
            session,
            collection_id=collection_id,
            source_node_id=route_id,
            target_node_id=target_view_id,
            kind=KnowledgeEdgeKind.UI_ROUTE_RENDERS_VIEW,
            meta=_provenance(
                mode="inferred" if inferred else "deterministic",
                extractor=_EXTRACTOR_UI_V1,
                confidence=0.78 if inferred else 0.94,
            ),
        )
        stats["ui_edges"] += 1
