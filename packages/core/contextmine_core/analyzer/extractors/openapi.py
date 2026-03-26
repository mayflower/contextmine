"""OpenAPI specification extractor.

Parses OpenAPI 3.x YAML/JSON files to extract:
- API endpoints (path + method)
- Request/response schemas
- Security definitions
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class EndpointDef:
    """Extracted API endpoint definition."""

    path: str
    method: str
    operation_id: str | None = None
    summary: str | None = None
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    request_body_ref: str | None = None
    response_refs: dict[str, str] = field(default_factory=dict)  # status -> schema ref
    parameters: list[dict[str, Any]] = field(default_factory=list)
    security: list[dict[str, list[str]]] = field(default_factory=list)
    handler_hints: list[str] = field(default_factory=list)


@dataclass
class SchemaDef:
    """Extracted schema definition."""

    name: str
    schema_type: str  # object, array, string, etc.
    properties: dict[str, str] = field(default_factory=dict)  # prop name -> type
    required: list[str] = field(default_factory=list)


@dataclass
class OpenAPIExtraction:
    """Result of parsing an OpenAPI specification."""

    file_path: str
    title: str | None = None
    version: str | None = None
    base_path: str | None = None
    endpoints: list[EndpointDef] = field(default_factory=list)
    schemas: dict[str, SchemaDef] = field(default_factory=dict)


def extract_from_openapi(file_path: str, content: str) -> OpenAPIExtraction:
    """Extract endpoint and schema definitions from an OpenAPI spec.

    Supports OpenAPI 3.x format (YAML or JSON).

    Args:
        file_path: Path to the specification file
        content: File content

    Returns:
        OpenAPIExtraction with extracted definitions
    """
    try:
        spec = yaml.safe_load(content)
        if not isinstance(spec, dict):
            return OpenAPIExtraction(file_path=file_path)
        return extract_from_openapi_document(file_path, spec)

    except (yaml.YAMLError, json.JSONDecodeError) as e:
        logger.warning("Failed to parse OpenAPI spec %s: %s", file_path, e)
    return OpenAPIExtraction(file_path=file_path)


def _extract_openapi_info(spec: dict[str, Any], result: OpenAPIExtraction) -> None:
    """Extract title, version, and base_path from an OpenAPI/Swagger spec."""
    info = spec.get("info", {})
    if isinstance(info, dict):
        result.title = info.get("title")
        result.version = info.get("version")

    servers = spec.get("servers", [])
    if isinstance(servers, list) and servers and isinstance(servers[0], dict):
        result.base_path = servers[0].get("url", "")
    else:
        base_path = spec.get("basePath")
        if isinstance(base_path, str) and base_path.strip():
            result.base_path = base_path


def _extract_openapi_endpoints(
    spec: dict[str, Any], result: OpenAPIExtraction, *, is_swagger2: bool
) -> None:
    """Extract endpoint definitions from paths in the spec."""
    paths = spec.get("paths", {})
    if not isinstance(paths, dict):
        return
    for path, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue
        for method in ["get", "post", "put", "patch", "delete", "options", "head"]:
            operation = path_item.get(method)
            if not isinstance(operation, dict):
                continue
            endpoint = _parse_endpoint(path, method, operation, is_swagger2=is_swagger2)
            result.endpoints.append(endpoint)


def _extract_openapi_schemas(spec: dict[str, Any], result: OpenAPIExtraction) -> None:
    """Extract schema definitions from components or definitions."""
    schema_defs: dict[str, Any] = {}
    components = spec.get("components", {})
    if isinstance(components, dict):
        schemas = components.get("schemas", {})
        if isinstance(schemas, dict):
            schema_defs.update(schemas)
    definitions = spec.get("definitions", {})
    if isinstance(definitions, dict):
        schema_defs.update(definitions)

    for name, schema in schema_defs.items():
        if isinstance(schema, dict):
            result.schemas[name] = _parse_schema(name, schema)


def extract_from_openapi_document(file_path: str, spec: dict[str, Any]) -> OpenAPIExtraction:
    """Extract endpoint/schema definitions from an already parsed OpenAPI/Swagger document."""
    result = OpenAPIExtraction(file_path=file_path)
    if not isinstance(spec, dict):
        return result

    _extract_openapi_info(spec, result)
    is_swagger2 = bool(spec.get("swagger")) and not bool(spec.get("openapi"))
    _extract_openapi_endpoints(spec, result, is_swagger2=is_swagger2)
    _extract_openapi_schemas(spec, result)
    return result


def _parse_endpoint(
    path: str,
    method: str,
    operation: dict[str, Any],
    *,
    is_swagger2: bool,
) -> EndpointDef:
    """Parse an endpoint operation definition."""
    endpoint = EndpointDef(
        path=path,
        method=method.upper(),
        operation_id=operation.get("operationId"),
        summary=operation.get("summary"),
        description=operation.get("description"),
        tags=operation.get("tags", []),
        handler_hints=_extract_handler_hints(operation),
    )

    endpoint.request_body_ref = _parse_request_body_ref(operation, is_swagger2=is_swagger2)
    endpoint.response_refs = _parse_response_refs(operation, is_swagger2=is_swagger2)

    # Parse parameters
    params = operation.get("parameters", [])
    endpoint.parameters = params if isinstance(params, list) else []

    # Parse security
    security = operation.get("security", [])
    endpoint.security = security if isinstance(security, list) else []

    return endpoint


def _parse_request_body_ref(operation: dict[str, Any], *, is_swagger2: bool) -> str | None:
    """Extract request body schema ref from an operation."""
    if is_swagger2:
        return _parse_swagger2_body_ref(operation)
    return _parse_openapi3_body_ref(operation)


def _parse_swagger2_body_ref(operation: dict[str, Any]) -> str | None:
    parameters = operation.get("parameters", [])
    if not isinstance(parameters, list):
        return None
    for param in parameters:
        if not isinstance(param, dict) or param.get("in") != "body":
            continue
        schema = param.get("schema")
        if isinstance(schema, dict) and "$ref" in schema:
            return _extract_ref_name(schema["$ref"])
    return None


def _parse_openapi3_body_ref(operation: dict[str, Any]) -> str | None:
    request_body = operation.get("requestBody", {})
    if not isinstance(request_body, dict):
        return None
    content = request_body.get("content", {})
    if not isinstance(content, dict):
        return None
    for _media_type, media_def in content.items():
        if not isinstance(media_def, dict) or "schema" not in media_def:
            continue
        schema = media_def["schema"]
        if isinstance(schema, dict) and "$ref" in schema:
            return _extract_ref_name(schema["$ref"])
        break
    return None


def _parse_response_refs(operation: dict[str, Any], *, is_swagger2: bool) -> dict[str, str]:
    """Extract response schema refs from an operation."""
    refs: dict[str, str] = {}
    responses = operation.get("responses", {})
    if not isinstance(responses, dict):
        return refs
    for status, response in responses.items():
        if not isinstance(response, dict):
            continue
        ref = _parse_single_response_ref(response, is_swagger2=is_swagger2)
        if ref:
            refs[str(status)] = ref
    return refs


def _parse_single_response_ref(response: dict[str, Any], *, is_swagger2: bool) -> str | None:
    if is_swagger2:
        schema = response.get("schema")
        if isinstance(schema, dict) and "$ref" in schema:
            return _extract_ref_name(schema["$ref"])
        return None
    content = response.get("content", {})
    if not isinstance(content, dict):
        return None
    for _media_type, media_def in content.items():
        if not isinstance(media_def, dict) or "schema" not in media_def:
            continue
        schema = media_def["schema"]
        if isinstance(schema, dict) and "$ref" in schema:
            return _extract_ref_name(schema["$ref"])
        break
    return None


def _extract_handler_hints(operation: dict[str, Any]) -> list[str]:
    """Extract implementation symbol hints from vendor extension fields."""
    hints: list[str] = []
    for key in (
        "x-handler",
        "x-handler-name",
        "x-handler-symbol",
        "x-operation-handler",
        "x-controller",
    ):
        value = operation.get(key)
        if isinstance(value, str) and value.strip():
            hints.append(value.strip())
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    hints.append(item.strip())
    return list(dict.fromkeys(hints))


def _resolve_property_type(prop_def: dict[str, Any]) -> str | None:
    """Resolve the type string for a single schema property definition."""
    if "$ref" in prop_def:
        return _extract_ref_name(prop_def["$ref"])
    if "type" not in prop_def:
        return None
    prop_type = prop_def["type"]
    if prop_type != "array" or "items" not in prop_def:
        return prop_type
    items = prop_def["items"]
    if not isinstance(items, dict):
        return prop_type
    if "$ref" in items:
        return f"array<{_extract_ref_name(items['$ref'])}>"
    return f"array<{items.get('type', 'any')}>"


def _parse_schema(name: str, schema: dict[str, Any]) -> SchemaDef:
    """Parse a schema definition."""
    schema_def = SchemaDef(
        name=name,
        schema_type=schema.get("type", "object"),
        required=schema.get("required", []),
    )

    properties = schema.get("properties", {})
    for prop_name, prop_def in properties.items():
        if not isinstance(prop_def, dict):
            continue
        resolved = _resolve_property_type(prop_def)
        if resolved is not None:
            schema_def.properties[prop_name] = resolved

    return schema_def


def _extract_ref_name(ref: str) -> str:
    """Extract the schema name from a $ref string."""
    # Format: "#/components/schemas/SchemaName"
    if ref.startswith("#/"):
        parts = ref.split("/")
        return parts[-1]
    return ref


def extract_from_openapi_file(file_path: Path | str) -> OpenAPIExtraction:
    """Extract from an OpenAPI specification file on disk.

    Args:
        file_path: Path to the spec file

    Returns:
        OpenAPIExtraction with extracted definitions
    """
    path = Path(file_path)
    if not path.exists():
        return OpenAPIExtraction(file_path=str(file_path))

    content = path.read_text(encoding="utf-8", errors="replace")
    return extract_from_openapi(str(file_path), content)
