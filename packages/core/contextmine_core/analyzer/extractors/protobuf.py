"""Protobuf (.proto) file extractor.

Parses Protocol Buffer definition files to extract:
- Messages
- Services and RPCs
- Enums
- Field definitions
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ProtoFieldDef:
    """Extracted protobuf field definition."""

    name: str
    field_type: str
    number: int
    repeated: bool = False
    optional: bool = False
    map_key_type: str | None = None
    map_value_type: str | None = None


@dataclass
class ProtoMessageDef:
    """Extracted protobuf message definition."""

    name: str
    fields: list[ProtoFieldDef] = field(default_factory=list)
    nested_messages: list[str] = field(default_factory=list)
    nested_enums: list[str] = field(default_factory=list)


@dataclass
class ProtoEnumDef:
    """Extracted protobuf enum definition."""

    name: str
    values: list[tuple[str, int]] = field(default_factory=list)  # (name, number)


@dataclass
class ProtoRPCDef:
    """Extracted protobuf RPC definition."""

    name: str
    request_type: str
    response_type: str
    request_stream: bool = False
    response_stream: bool = False


@dataclass
class ProtoServiceDef:
    """Extracted protobuf service definition."""

    name: str
    rpcs: list[ProtoRPCDef] = field(default_factory=list)


@dataclass
class ProtobufExtraction:
    """Result of parsing a protobuf file."""

    file_path: str
    package: str | None = None
    syntax: str = "proto3"
    imports: list[str] = field(default_factory=list)
    messages: list[ProtoMessageDef] = field(default_factory=list)
    enums: list[ProtoEnumDef] = field(default_factory=list)
    services: list[ProtoServiceDef] = field(default_factory=list)


# Regex patterns for protobuf parsing
SYNTAX_PATTERN = re.compile(r'syntax\s*=\s*"(proto[23])"\s*;')
PACKAGE_PATTERN = re.compile(r"package\s+([\w.]+)\s*;")
IMPORT_PATTERN = re.compile(r'import\s+"([^"]+)"\s*;')
MESSAGE_PATTERN = re.compile(r"message\s+(\w+)\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}")
ENUM_PATTERN = re.compile(r"enum\s+(\w+)\s*\{([^}]*)\}")
SERVICE_PATTERN = re.compile(r"service\s+(\w+)\s*\{([^}]*)\}")
FIELD_PATTERN = re.compile(
    r"(repeated\s+|optional\s+)?"  # Optional modifiers
    r"(map<(\w+),\s*(\w+)>|[\w.]+)"  # Type (including map)
    r"\s+(\w+)"  # Field name
    r"\s*=\s*(\d+)"  # Field number
)
ENUM_VALUE_PATTERN = re.compile(r"(\w+)\s*=\s*(-?\d+)")
RPC_PATTERN = re.compile(
    r"rpc\s+(\w+)\s*\("  # RPC name
    r"\s*(stream\s+)?(\w+)\s*\)"  # Request type
    r"\s*returns\s*\("
    r"\s*(stream\s+)?(\w+)\s*\)"  # Response type
)


def extract_from_protobuf(file_path: str, content: str) -> ProtobufExtraction:
    """Extract message, service, and enum definitions from a protobuf file.

    Uses regex-based parsing for common protobuf constructs.

    Args:
        file_path: Path to the proto file
        content: File content

    Returns:
        ProtobufExtraction with extracted definitions
    """
    result = ProtobufExtraction(file_path=file_path)

    # Remove comments
    content_clean = _remove_comments(content)

    # Extract syntax
    syntax_match = SYNTAX_PATTERN.search(content_clean)
    if syntax_match:
        result.syntax = syntax_match.group(1)

    # Extract package
    package_match = PACKAGE_PATTERN.search(content_clean)
    if package_match:
        result.package = package_match.group(1)

    # Extract imports
    for import_match in IMPORT_PATTERN.finditer(content_clean):
        result.imports.append(import_match.group(1))

    # Extract enums (before messages to handle nested)
    for enum_match in ENUM_PATTERN.finditer(content_clean):
        result.enums.append(_parse_enum(enum_match.group(1), enum_match.group(2)))

    # Extract messages
    for msg_match in MESSAGE_PATTERN.finditer(content_clean):
        result.messages.append(_parse_message(msg_match.group(1), msg_match.group(2)))

    # Extract services
    for svc_match in SERVICE_PATTERN.finditer(content_clean):
        result.services.append(_parse_service(svc_match.group(1), svc_match.group(2)))

    return result


def _remove_comments(content: str) -> str:
    """Remove single-line and multi-line comments from protobuf content."""
    # Remove multi-line comments
    content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
    # Remove single-line comments
    content = re.sub(r"//.*$", "", content, flags=re.MULTILINE)
    return content


def _parse_message(name: str, body: str) -> ProtoMessageDef:
    """Parse a message definition body."""
    msg = ProtoMessageDef(name=name)

    # Find nested messages
    for nested_match in re.finditer(r"message\s+(\w+)", body):
        msg.nested_messages.append(nested_match.group(1))

    # Find nested enums
    for nested_match in re.finditer(r"enum\s+(\w+)", body):
        msg.nested_enums.append(nested_match.group(1))

    # Parse fields (excluding nested message/enum blocks)
    # Simple approach: find fields by pattern
    for field_match in FIELD_PATTERN.finditer(body):
        modifier = field_match.group(1)
        type_full = field_match.group(2)
        map_key = field_match.group(3)
        map_value = field_match.group(4)
        field_name = field_match.group(5)
        field_number = int(field_match.group(6))

        field_def = ProtoFieldDef(
            name=field_name,
            field_type=type_full,
            number=field_number,
        )

        if modifier:
            modifier = modifier.strip()
            field_def.repeated = modifier == "repeated"
            field_def.optional = modifier == "optional"

        if map_key and map_value:
            field_def.map_key_type = map_key
            field_def.map_value_type = map_value

        msg.fields.append(field_def)

    return msg


def _parse_enum(name: str, body: str) -> ProtoEnumDef:
    """Parse an enum definition body."""
    enum_def = ProtoEnumDef(name=name)

    for value_match in ENUM_VALUE_PATTERN.finditer(body):
        enum_def.values.append((value_match.group(1), int(value_match.group(2))))

    return enum_def


def _parse_service(name: str, body: str) -> ProtoServiceDef:
    """Parse a service definition body."""
    service = ProtoServiceDef(name=name)

    for rpc_match in RPC_PATTERN.finditer(body):
        rpc = ProtoRPCDef(
            name=rpc_match.group(1),
            request_type=rpc_match.group(3),
            response_type=rpc_match.group(5),
            request_stream=rpc_match.group(2) is not None,
            response_stream=rpc_match.group(4) is not None,
        )
        service.rpcs.append(rpc)

    return service


def extract_from_protobuf_file(file_path: Path | str) -> ProtobufExtraction:
    """Extract from a protobuf file on disk.

    Args:
        file_path: Path to the proto file

    Returns:
        ProtobufExtraction with extracted definitions
    """
    path = Path(file_path)
    if not path.exists():
        return ProtobufExtraction(file_path=str(file_path))

    content = path.read_text(encoding="utf-8", errors="replace")
    return extract_from_protobuf(str(file_path), content)
