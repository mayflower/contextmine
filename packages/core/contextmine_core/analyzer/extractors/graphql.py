"""GraphQL schema extractor.

Parses GraphQL schema files (.graphql/.gql) to extract:
- Types (object, input, enum, interface, union)
- Operations (query, mutation, subscription)
- Field definitions
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

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


# Regex patterns for GraphQL schema parsing
# These handle the most common SDL constructs
TYPE_PATTERN = re.compile(
    r'(?:"""([^"]*?)"""\s*)?'  # Optional description
    r"(type|interface|input|enum|union|scalar)\s+"
    r"(\w+)"  # Type name
    r"(?:\s+implements\s+([\w\s&]+))?"  # Optional implements clause
    r"(?:\s*=\s*([\w\s|]+))?"  # Optional union types
    r"(?:\s*\{([^}]*)\})?",  # Optional field block
    re.MULTILINE | re.DOTALL,
)

FIELD_PATTERN = re.compile(
    r'(?:"""([^"]*?)"""\s*)?'  # Optional description
    r"(\w+)"  # Field name
    r"(?:\(([^)]*)\))?"  # Optional arguments
    r"\s*:\s*"  # Colon
    r"([\w\[\]!]+)",  # Type
    re.MULTILINE,
)

ENUM_VALUE_PATTERN = re.compile(r"^\s*(\w+)\s*$", re.MULTILINE)

ARGUMENT_PATTERN = re.compile(r"(\w+)\s*:\s*([\w\[\]!]+)")


def extract_from_graphql(file_path: str, content: str) -> GraphQLExtraction:
    """Extract type and operation definitions from a GraphQL schema.

    Uses regex-based parsing for SDL (Schema Definition Language).
    For production use, consider using graphql-core library.

    Args:
        file_path: Path to the schema file
        content: File content

    Returns:
        GraphQLExtraction with extracted definitions
    """
    result = GraphQLExtraction(file_path=file_path)

    # Remove comments (lines starting with #)
    content_clean = "\n".join(
        line for line in content.split("\n") if not line.strip().startswith("#")
    )

    for match in TYPE_PATTERN.finditer(content_clean):
        description = match.group(1)
        kind = match.group(2)
        name = match.group(3)
        implements_str = match.group(4)
        union_str = match.group(5)
        body = match.group(6)

        type_def = GraphQLTypeDef(
            name=name,
            kind=kind,
            description=description.strip() if description else None,
        )

        # Parse implements
        if implements_str:
            type_def.implements = [i.strip() for i in implements_str.replace("&", ",").split(",")]

        # Parse union types
        if union_str and kind == "union":
            type_def.union_types = [t.strip() for t in union_str.split("|")]

        # Parse fields or enum values
        if body:
            if kind == "enum":
                type_def.enum_values = [
                    m.group(1) for m in ENUM_VALUE_PATTERN.finditer(body) if m.group(1)
                ]
            elif kind in ("type", "interface", "input"):
                type_def.fields = _parse_fields(body)

        result.types.append(type_def)

        # Check if this is a root operation type
        if name in ("Query", "Mutation", "Subscription"):
            op = GraphQLOperationDef(name=name, kind=name, fields=type_def.fields)
            result.operations.append(op)

    return result


def _parse_fields(body: str) -> list[GraphQLFieldDef]:
    """Parse field definitions from a type body."""
    fields = []

    for match in FIELD_PATTERN.finditer(body):
        description = match.group(1)
        name = match.group(2)
        args_str = match.group(3)
        field_type = match.group(4)

        field_def = GraphQLFieldDef(
            name=name,
            field_type=field_type,
            description=description.strip() if description else None,
        )

        # Parse arguments
        if args_str:
            for arg_match in ARGUMENT_PATTERN.finditer(args_str):
                field_def.arguments.append((arg_match.group(1), arg_match.group(2)))

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
