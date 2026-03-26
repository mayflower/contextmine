"""GraphQL schema extractor using AST parsing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from contextmine_core.analyzer.extractors.ast_utils import (
    children_of,
    first_child,
    node_text,
    parse_with_language,
    unquote,
    walk,
)

logger = logging.getLogger(__name__)


@dataclass
class GraphQLFieldDef:
    """Extracted GraphQL field definition."""

    name: str
    field_type: str
    arguments: list[tuple[str, str]] = field(default_factory=list)  # (name, type)
    description: str | None = None


@dataclass
class GraphQLTypeDef:
    """Extracted GraphQL type definition."""

    name: str
    kind: str  # type, input, interface, enum, union, scalar
    fields: list[GraphQLFieldDef] = field(default_factory=list)
    implements: list[str] = field(default_factory=list)
    enum_values: list[str] = field(default_factory=list)
    union_types: list[str] = field(default_factory=list)
    description: str | None = None


@dataclass
class GraphQLOperationDef:
    """Extracted GraphQL operation (query/mutation/subscription root)."""

    name: str
    kind: str  # Query, Mutation, Subscription
    fields: list[GraphQLFieldDef] = field(default_factory=list)


@dataclass
class GraphQLExtraction:
    """Result of parsing a GraphQL schema file."""

    file_path: str
    types: list[GraphQLTypeDef] = field(default_factory=list)
    operations: list[GraphQLOperationDef] = field(default_factory=list)


_GRAPHQL_KIND_MAP = {
    "object_type_definition": "type",
    "interface_type_definition": "interface",
    "input_object_type_definition": "input",
    "enum_type_definition": "enum",
    "union_type_definition": "union",
    "scalar_type_definition": "scalar",
}


def _extract_enum_values(content: str, node: object) -> list[str]:
    """Extract enum value names from an enum type definition node."""
    enum_values_node = first_child(node, "enum_values_definition")
    if enum_values_node is None:
        return []
    values: list[str] = []
    for enum_value in children_of(enum_values_node, "enum_value_definition"):
        for sub in walk(enum_value):
            if sub.type == "name":
                enum_name = node_text(content, sub).strip()
                if enum_name:
                    values.append(enum_name)
                    break
    return values


def _populate_type_def_fields(
    content: str, kind: str, node: object, type_def: GraphQLTypeDef
) -> None:
    """Populate fields or enum values on a type definition based on its kind."""
    if kind == "enum":
        type_def.enum_values = _extract_enum_values(content, node)
    elif kind in {"type", "interface"}:
        fields_node = first_child(node, "fields_definition")
        if fields_node is not None:
            type_def.fields = _parse_field_definitions(
                content, fields_node, field_type="field_definition"
            )
    elif kind == "input":
        fields_node = first_child(node, "input_fields_definition")
        if fields_node is not None:
            type_def.fields = _parse_field_definitions(
                content,
                fields_node,
                field_type="input_value_definition",
            )


def _build_graphql_type_def(
    content: str,
    kind: str,
    node: object,
    name: str,
) -> GraphQLTypeDef:
    """Build a complete GraphQLTypeDef from a parsed AST node."""
    description = _extract_description(content, node)
    type_def = GraphQLTypeDef(name=name, kind=kind, description=description)
    type_def.implements = _extract_named_types(content, first_child(node, "implements_interfaces"))
    type_def.union_types = _extract_named_types(content, first_child(node, "union_member_types"))
    _populate_type_def_fields(content, kind, node, type_def)
    return type_def


def extract_from_graphql(file_path: str, content: str) -> GraphQLExtraction:
    """Extract type and operation definitions from a GraphQL schema.

    Args:
        file_path: Path to the schema file
        content: File content

    Returns:
        GraphQLExtraction with extracted definitions
    """
    result = GraphQLExtraction(file_path=file_path)
    root = parse_with_language("graphql", content)
    if root is None:
        logger.warning("GraphQL parser unavailable; skipping AST extraction for %s", file_path)
        return result

    for node in walk(root):
        kind = _GRAPHQL_KIND_MAP.get(node.type)
        if kind is None:
            continue

        name = node_text(content, first_child(node, "name")).strip()
        if not name:
            continue

        type_def = _build_graphql_type_def(content, kind, node, name)
        result.types.append(type_def)
        if name in {"Query", "Mutation", "Subscription"}:
            result.operations.append(
                GraphQLOperationDef(name=name, kind=name, fields=type_def.fields)
            )
    return result


def _extract_description(content: str, node: object) -> str | None:
    description_node = first_child(node, "description")
    if description_node is None:
        return None
    return unquote(node_text(content, description_node))


def _extract_named_types(content: str, node: object | None) -> list[str]:
    if node is None:
        return []
    values: list[str] = []
    for sub in walk(node):
        if sub.type != "named_type":
            continue
        type_name = node_text(content, first_child(sub, "name")).strip()
        if type_name:
            values.append(type_name)
    return list(dict.fromkeys(values))


def _parse_field_definitions(
    content: str, container: object, field_type: str
) -> list[GraphQLFieldDef]:
    fields: list[GraphQLFieldDef] = []
    for field_node in children_of(container, field_type):
        name = node_text(content, first_child(field_node, "name")).strip()
        out_type = node_text(content, first_child(field_node, "type")).strip()
        if not name or not out_type:
            continue

        field_def = GraphQLFieldDef(
            name=name,
            field_type=out_type,
            description=_extract_description(content, field_node),
        )

        args_node = first_child(field_node, "arguments_definition")
        if args_node is not None:
            for arg in children_of(args_node, "input_value_definition"):
                arg_name = node_text(content, first_child(arg, "name")).strip()
                arg_type = node_text(content, first_child(arg, "type")).strip()
                if arg_name and arg_type:
                    field_def.arguments.append((arg_name, arg_type))
        fields.append(field_def)

    return fields


def extract_from_graphql_file(file_path: Path | str) -> GraphQLExtraction:
    """Extract from a GraphQL schema file on disk.

    Args:
        file_path: Path to the schema file

    Returns:
        GraphQLExtraction with extracted definitions
    """
    path = Path(file_path)
    if not path.exists():
        return GraphQLExtraction(file_path=str(file_path))

    content = path.read_text(encoding="utf-8", errors="replace")
    return extract_from_graphql(str(file_path), content)
