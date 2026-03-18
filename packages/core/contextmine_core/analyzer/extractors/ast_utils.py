"""Tree-sitter AST helpers for deterministic extractors."""

from __future__ import annotations

from typing import Any


def parse_with_language(language: str, content: str) -> Any | None:
    """Parse content with the requested tree-sitter grammar."""
    try:
        from tree_sitter_language_pack import get_parser
    except ImportError:
        return None

    try:
        parser = get_parser(language)
    except Exception:
        return None

    try:
        return parser.parse(content.encode("utf-8")).root_node
    except Exception:
        return None


def walk(node: Any) -> Any:
    """Yield node and all descendants in preorder."""
    yield node
    for child in node.children:
        yield from walk(child)


def node_text(content: str, node: Any | None) -> str:
    """Return node text slice from source content."""
    if node is None:
        return ""
    return content[node.start_byte : node.end_byte]


def line_number(node: Any | None) -> int:
    """Return 1-based node start line."""
    if node is None:
        return 1
    return int(node.start_point[0]) + 1


def first_child(node: Any, node_type: str) -> Any | None:
    """Return first direct child with the given type."""
    for child in node.children:
        if child.type == node_type:
            return child
    return None


def children_of(node: Any, node_type: str) -> list[Any]:
    """Return all direct children with the given type."""
    return [child for child in node.children if child.type == node_type]


def unquote(value: str) -> str:
    """Strip matching quote wrappers around literal text."""
    text = value.strip()
    if len(text) >= 6 and text.startswith('"""') and text.endswith('"""'):
        return text[3:-3].strip()
    if len(text) >= 6 and text.startswith("'''") and text.endswith("'''"):
        return text[3:-3].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"', "`"}:
        return text[1:-1]
    return text


def is_pascal_case(value: str) -> bool:
    """Best-effort PascalCase detector."""
    if not value or not value[0].isupper():
        return False
    return all(ch.isalnum() or ch == "_" for ch in value)


# ---------------------------------------------------------------------------
# JS/TS AST helpers shared between the tests and UI extractors
# ---------------------------------------------------------------------------

HTTP_METHOD_NAMES = {"get", "post", "put", "patch", "delete"}
HTTP_CLIENT_NAMES = {"axios", "client", "api", "http", "request", "agent", "supertest"}


def js_call_name(content: str, call_node: Any) -> tuple[str, str, str]:
    """Extract (full_name, base_name, method_name) from a JS/TS call_expression."""
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


def string_literal(content: str, node: Any) -> str | None:
    """Extract a string value from a JS/TS string or template_string node."""
    if node.type == "string":
        return unquote(node_text(content, node))
    if node.type == "template_string":
        raw = node_text(content, node).strip()
        if "${" in raw:
            return None
        return unquote(raw)
    return None


def first_string_argument(content: str, call_node: Any) -> str | None:
    """Return the first string literal argument of a JS/TS call."""
    args = call_node.child_by_field_name("arguments")
    if args is None:
        return None
    for child in args.children:
        value = string_literal(content, child)
        if value:
            return value
    return None


def endpoint_from_call(
    content: str,
    call_node: Any,
    *,
    base_name: str,
    method_name: str,
) -> str | None:
    """Detect an HTTP endpoint string from a JS/TS call expression."""
    first_literal = first_string_argument(content, call_node)
    if not first_literal:
        return None

    lower_base = base_name.lower()
    lower_method = method_name.lower()
    if lower_base == "fetch":
        return first_literal
    if lower_method in HTTP_METHOD_NAMES and (
        lower_base in HTTP_CLIENT_NAMES or "." in lower_base or lower_base.endswith("client")
    ):
        return first_literal
    return None
