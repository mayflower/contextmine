"""Protobuf (.proto) extractor using AST parsing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from contextmine_core.analyzer.extractors.ast_utils import (
    children_of,
    first_child,
    node_text,
    parse_with_language,
    unquote,
)

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


def extract_from_protobuf(file_path: str, content: str) -> ProtobufExtraction:
    """Extract message, service, and enum definitions from a protobuf file.

    Args:
        file_path: Path to the proto file
        content: File content

    Returns:
        ProtobufExtraction with extracted definitions
    """
    result = ProtobufExtraction(file_path=file_path)
    root = parse_with_language("proto", content)
    if root is None:
        logger.warning("Protobuf parser unavailable; skipping AST extraction for %s", file_path)
        return result

    for node in root.children:
        if node.type == "syntax":
            syntax_value = unquote(node_text(content, first_child(node, "string")))
            if syntax_value:
                result.syntax = syntax_value
        elif node.type == "package":
            package_name = node_text(content, first_child(node, "full_ident")).strip()
            if package_name:
                result.package = package_name
        elif node.type == "import":
            imported = unquote(node_text(content, first_child(node, "string")))
            if imported:
                result.imports.append(imported)
        elif node.type == "enum":
            parsed = _parse_enum(content, node)
            if parsed is not None:
                result.enums.append(parsed)
        elif node.type == "message":
            parsed = _parse_message(content, node)
            if parsed is not None:
                result.messages.append(parsed)
        elif node.type == "service":
            parsed = _parse_service(content, node)
            if parsed is not None:
                result.services.append(parsed)
    return result


def _parse_message(content: str, message_node: object) -> ProtoMessageDef | None:
    name = node_text(
        content, first_child(first_child(message_node, "message_name"), "identifier")
    ).strip()
    if not name:
        return None

    message = ProtoMessageDef(name=name)
    body = first_child(message_node, "message_body")
    if body is None:
        return message

    for child in body.children:
        if child.type == "field":
            field = _parse_field(content, child)
            if field is not None:
                message.fields.append(field)
        elif child.type == "map_field":
            field = _parse_map_field(content, child)
            if field is not None:
                message.fields.append(field)
        elif child.type == "message":
            nested_name = node_text(
                content,
                first_child(first_child(child, "message_name"), "identifier"),
            ).strip()
            if nested_name:
                message.nested_messages.append(nested_name)
        elif child.type == "enum":
            nested_enum_name = node_text(
                content,
                first_child(first_child(child, "enum_name"), "identifier"),
            ).strip()
            if nested_enum_name:
                message.nested_enums.append(nested_enum_name)

    return message


def _parse_field(content: str, node: object) -> ProtoFieldDef | None:
    field_name = node_text(content, first_child(node, "identifier")).strip()
    type_node = first_child(node, "type")
    number_node = first_child(node, "field_number")
    if not field_name or type_node is None or number_node is None:
        return None

    try:
        number = int(node_text(content, number_node).strip())
    except ValueError:
        return None

    return ProtoFieldDef(
        name=field_name,
        field_type=node_text(content, type_node).strip(),
        number=number,
        repeated=first_child(node, "repeated") is not None,
        optional=first_child(node, "optional") is not None,
    )


def _parse_map_field(content: str, node: object) -> ProtoFieldDef | None:
    field_name = node_text(content, first_child(node, "identifier")).strip()
    key_type = node_text(content, first_child(node, "key_type")).strip()
    value_type = node_text(content, first_child(node, "type")).strip()
    number_node = first_child(node, "field_number")
    if not field_name or not key_type or not value_type or number_node is None:
        return None

    try:
        number = int(node_text(content, number_node).strip())
    except ValueError:
        return None

    return ProtoFieldDef(
        name=field_name,
        field_type=f"map<{key_type},{value_type}>",
        number=number,
        map_key_type=key_type,
        map_value_type=value_type,
    )


def _parse_enum(content: str, enum_node: object) -> ProtoEnumDef | None:
    name = node_text(
        content, first_child(first_child(enum_node, "enum_name"), "identifier")
    ).strip()
    if not name:
        return None

    enum_def = ProtoEnumDef(name=name)
    enum_body = first_child(enum_node, "enum_body")
    if enum_body is None:
        return enum_def

    for enum_field in children_of(enum_body, "enum_field"):
        ident = node_text(content, first_child(enum_field, "identifier")).strip()
        value_node = first_child(enum_field, "int_lit")
        if not ident or value_node is None:
            continue
        try:
            value = int(node_text(content, value_node).strip())
        except ValueError:
            continue
        enum_def.values.append((ident, value))
    return enum_def


def _parse_service(content: str, service_node: object) -> ProtoServiceDef | None:
    name = node_text(
        content,
        first_child(first_child(service_node, "service_name"), "identifier"),
    ).strip()
    if not name:
        return None

    service = ProtoServiceDef(name=name)
    for rpc_node in children_of(service_node, "rpc"):
        rpc = _parse_rpc(content, rpc_node)
        if rpc is not None:
            service.rpcs.append(rpc)
    return service


def _parse_rpc(content: str, rpc_node: Any) -> ProtoRPCDef | None:
    name = node_text(content, first_child(first_child(rpc_node, "rpc_name"), "identifier")).strip()
    if not name:
        return None

    request_type = ""
    response_type = ""
    request_stream = False
    response_stream = False
    stage = "seek_request"

    for child in rpc_node.children:
        if stage == "seek_request":
            if child.type == "(":
                stage = "request"
            continue
        if stage == "request":
            if child.type == "stream":
                request_stream = True
            elif child.type == "message_or_enum_type":
                request_type = node_text(content, child).strip()
            elif child.type == ")":
                stage = "seek_response"
            continue
        if stage == "seek_response":
            if child.type == "(":
                stage = "response"
            continue
        if stage == "response":
            if child.type == "stream":
                response_stream = True
            elif child.type == "message_or_enum_type":
                response_type = node_text(content, child).strip()
            elif child.type == ")":
                break

    if not request_type or not response_type:
        return None
    return ProtoRPCDef(
        name=name,
        request_type=request_type,
        response_type=response_type,
        request_stream=request_stream,
        response_stream=response_stream,
    )


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
