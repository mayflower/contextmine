"""Compile architecture intents into RFC6902 patch operations."""

from __future__ import annotations

import uuid
from typing import Any

from contextmine_core.architecture_intents.schema import ArchitectureIntentV1, IntentAction


def _default_node(kind: str, natural_key: str, name: str, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "kind": kind,
        "natural_key": natural_key,
        "name": name,
        "meta": params,
    }


def compile_intent_patch(intent: ArchitectureIntentV1) -> list[dict[str, Any]]:
    """Compile a strict intent into JSON patch operations.

    The patch document uses a canonical twin shape with root keys `nodes` and `edges`.
    """
    action = intent.action
    target_id = intent.target.id
    params = intent.params or {}

    if action == IntentAction.EXTRACT_DOMAIN:
        node = _default_node(
            kind="bounded_context",
            natural_key=f"context:{target_id}",
            name=params.get("name", f"Domain {target_id}"),
            params=params,
        )
        return [{"op": "add", "path": "/nodes/-", "value": node}]

    if action == IntentAction.SPLIT_CONTAINER:
        container_a = _default_node(
            kind="container",
            natural_key=f"container:{target_id}:a",
            name=params.get("left_name", f"{target_id}-a"),
            params=params,
        )
        container_b = _default_node(
            kind="container",
            natural_key=f"container:{target_id}:b",
            name=params.get("right_name", f"{target_id}-b"),
            params=params,
        )
        return [
            {"op": "add", "path": "/nodes/-", "value": container_a},
            {"op": "add", "path": "/nodes/-", "value": container_b},
        ]

    if action == IntentAction.MOVE_COMPONENT:
        return [
            {
                "op": "replace",
                "path": f"/nodes/by_natural_key/{target_id}/meta/domain",
                "value": params.get("domain", "unassigned"),
            }
        ]

    if action == IntentAction.DEFINE_INTERFACE:
        interface_key = f"interface:{target_id}:{params.get('name', 'api')}"
        interface_node = _default_node(
            kind="interface",
            natural_key=interface_key,
            name=params.get("name", "interface"),
            params=params,
        )
        edge = {
            "id": str(uuid.uuid4()),
            "kind": "defines_interface",
            "source_natural_key": target_id,
            "target_natural_key": interface_key,
            "meta": params,
        }
        return [
            {"op": "add", "path": "/nodes/-", "value": interface_node},
            {"op": "add", "path": "/edges/-", "value": edge},
        ]

    if action == IntentAction.SET_VALIDATOR:
        validator_key = f"validator:{target_id}:{params.get('name', 'validator')}"
        validator_node = _default_node(
            kind="validator",
            natural_key=validator_key,
            name=params.get("name", "validator"),
            params=params,
        )
        return [{"op": "add", "path": "/nodes/-", "value": validator_node}]

    if action == IntentAction.APPLY_DATA_BOUNDARY:
        return [
            {
                "op": "replace",
                "path": f"/nodes/by_natural_key/{target_id}/meta/data_boundary",
                "value": params.get("boundary", "isolated"),
            }
        ]

    raise ValueError(f"Unsupported intent action: {action}")
