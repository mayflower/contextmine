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


def extract_from_openapi_document(file_path: str, spec: dict[str, Any]) -> OpenAPIExtraction:
    """Extract endpoint/schema definitions from an already parsed OpenAPI/Swagger document."""
    result = OpenAPIExtraction(file_path=file_path)
    if not isinstance(spec, dict):
        return result

    # Extract basic info
    info = spec.get("info", {})
    if isinstance(info, dict):
        result.title = info.get("title")
        result.version = info.get("version")

    # Prefer OpenAPI server URL, fallback to Swagger 2 basePath.
    servers = spec.get("servers", [])
    if isinstance(servers, list) and servers and isinstance(servers[0], dict):
        result.base_path = servers[0].get("url", "")
    else:
        base_path = spec.get("basePath")
        if isinstance(base_path, str) and base_path.strip():
            result.base_path = base_path

    is_swagger2 = bool(spec.get("swagger")) and not bool(spec.get("openapi"))

    # Extract paths/endpoints
    paths = spec.get("paths", {})
    if isinstance(paths, dict):
        for path, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue
            for method in ["get", "post", "put", "patch", "delete", "options", "head"]:
                operation = path_item.get(method)
                if not isinstance(operation, dict):
                    continue
                endpoint = _parse_endpoint(path, method, operation, is_swagger2=is_swagger2)
                result.endpoints.append(endpoint)

    # Extract schemas from components (OpenAPI 3) or definitions (Swagger 2)
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

    if is_swagger2:
        parameters = operation.get("parameters", [])
        if isinstance(parameters, list):
            for param in parameters:
                if not isinstance(param, dict):
                    continue
                if param.get("in") != "body":
                    continue
                schema = param.get("schema")
                if isinstance(schema, dict) and "$ref" in schema:
                    endpoint.request_body_ref = _extract_ref_name(schema["$ref"])
                    break
    else:
        # Parse OpenAPI 3 request body
        request_body = operation.get("requestBody", {})
        if isinstance(request_body, dict):
            content = request_body.get("content", {})
            if isinstance(content, dict):
                for _media_type, media_def in content.items():
                    if isinstance(media_def, dict) and "schema" in media_def:
                        schema = media_def["schema"]
                        if isinstance(schema, dict) and "$ref" in schema:
                            endpoint.request_body_ref = _extract_ref_name(schema["$ref"])
                        break

    # Parse responses
    responses = operation.get("responses", {})
    if isinstance(responses, dict):
        for status, response in responses.items():
            if not isinstance(response, dict):
                continue
            if is_swagger2:
                schema = response.get("schema")
                if isinstance(schema, dict) and "$ref" in schema:
                    endpoint.response_refs[str(status)] = _extract_ref_name(schema["$ref"])
                continue

            content = response.get("content", {})
            if not isinstance(content, dict):
                continue
            for _media_type, media_def in content.items():
                if isinstance(media_def, dict) and "schema" in media_def:
                    schema = media_def["schema"]
                    if isinstance(schema, dict) and "$ref" in schema:
                        endpoint.response_refs[str(status)] = _extract_ref_name(schema["$ref"])
                    break

    # Parse parameters
    params = operation.get("parameters", [])
    endpoint.parameters = params if isinstance(params, list) else []

    # Parse security
    security = operation.get("security", [])
    endpoint.security = security if isinstance(security, list) else []

    return endpoint


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


def _parse_schema(name: str, schema: dict[str, Any]) -> SchemaDef:
    """Parse a schema definition."""
    schema_def = SchemaDef(
        name=name,
        schema_type=schema.get("type", "object"),
        required=schema.get("required", []),
    )

    # Extract property types
    properties = schema.get("properties", {})
    for prop_name, prop_def in properties.items():
        if isinstance(prop_def, dict):
            if "$ref" in prop_def:
                schema_def.properties[prop_name] = _extract_ref_name(prop_def["$ref"])
            elif "type" in prop_def:
                prop_type = prop_def["type"]
                if prop_type == "array" and "items" in prop_def:
                    items = prop_def["items"]
                    if isinstance(items, dict):
                        if "$ref" in items:
                            schema_def.properties[prop_name] = (
                                f"array<{_extract_ref_name(items['$ref'])}>"
                            )
                        else:
                            schema_def.properties[prop_name] = f"array<{items.get('type', 'any')}>"
                else:
                    schema_def.properties[prop_name] = prop_type

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
