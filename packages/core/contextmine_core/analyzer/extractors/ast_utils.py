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
