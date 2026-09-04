"""Repair structured LLM output that arrives with JSON-encoded fields.

Anthropic's tool-calling transport occasionally serializes a nested array or
object into a JSON *string* instead of returning it structurally. The value is
well-formed JSON, just one encoding level too deep, and strict schema
validation rejects it:

    rules
      Input should be a valid list [type=list_type,
       input_value='[\\n  {\\n    "name": ...', input_type=str]

Decoding such fields before validation keeps the extracted data instead of
discarding the whole response.
"""

from __future__ import annotations

import json
import logging
import types
import typing
from typing import Any

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

__all__ = [
    "coerce_json_string_fields",
    "field_expects_structure",
    "repair_structured_payload",
]


def _unwrap_annotation(annotation: Any) -> list[Any]:
    """Return the candidate types of an annotation, flattening unions."""
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        return [arg for arg in typing.get_args(annotation) if arg is not type(None)]
    return [annotation]


def field_expects_structure(annotation: Any) -> bool:
    """Report whether a field wants a list, dict or model rather than a string.

    A field that legitimately accepts a string is never decoded, so a
    ``reasoning`` field holding the text ``"[1, 2]"`` keeps its exact value.
    """
    for candidate in _unwrap_annotation(annotation):
        if candidate is str:
            return False

        origin = typing.get_origin(candidate) or candidate
        if origin in (list, tuple, set, frozenset, dict):
            return True
        if isinstance(origin, type) and issubclass(origin, BaseModel):
            return True
    return False


def coerce_json_string_fields(data: Any, schema: type[BaseModel]) -> Any:
    """Decode JSON strings sitting in fields that expect structured values.

    Returns ``data`` unchanged when it is not a mapping. Fields are left alone
    unless the schema wants a structure there *and* the string parses as JSON,
    so malformed output still reaches validation and reports a real error.
    """
    if not isinstance(data, dict):
        return data

    coerced: dict[str, Any] = dict(data)
    for name, field in schema.model_fields.items():
        for key in {name, field.alias} - {None}:
            raw = coerced.get(key)
            if not isinstance(raw, str) or not field_expects_structure(field.annotation):
                continue
            try:
                decoded = json.loads(raw)
            except TypeError, ValueError:
                # Not JSON after all - let validation report the real problem.
                continue
            if isinstance(decoded, str):
                # A plain quoted string is not the structure we are looking for.
                continue
            coerced[key] = decoded
    return coerced


def repair_structured_payload[T: BaseModel](
    raw: Any,
    output_schema: type[T],
) -> T | None:
    """Rebuild a schema instance from the raw tool-call arguments of a response.

    Used when the provider returned a complete answer that only failed schema
    validation because a nested value was JSON-encoded. Returns None when the
    payload needed no repair or cannot be salvaged, so the caller still reports
    the original error instead of hiding it.
    """
    tool_calls = getattr(raw, "tool_calls", None) or []
    for tool_call in tool_calls:
        arguments = tool_call.get("args") if isinstance(tool_call, dict) else None
        if not isinstance(arguments, dict):
            continue
        coerced = coerce_json_string_fields(arguments, output_schema)
        if coerced == arguments:
            continue
        try:
            repaired = output_schema.model_validate(coerced)
        except ValidationError:
            continue
        logger.warning("Repaired JSON-encoded fields in %s response", output_schema.__name__)
        return repaired
    return None
