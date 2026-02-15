"""Tests for architecture intent schema and patch compiler."""

from __future__ import annotations

import uuid

import pytest
from contextmine_core.architecture_intents import (
    ArchitectureIntentV1,
    IntentAction,
    IntentRisk,
    classify_risk,
    compile_intent_patch,
)
from pydantic import ValidationError


def _base_payload(action: str) -> dict:
    scenario_id = str(uuid.uuid4())
    return {
        "intent_version": "1.0",
        "scenario_id": scenario_id,
        "action": action,
        "target": {"type": "node", "id": "payments"},
        "params": {"name": "Payments"},
        "expected_scenario_version": 3,
    }


def test_schema_rejects_unknown_fields() -> None:
    payload = _base_payload("EXTRACT_DOMAIN")
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        ArchitectureIntentV1(**payload)


def test_risk_classification() -> None:
    assert classify_risk(IntentAction.SPLIT_CONTAINER) == IntentRisk.HIGH
    assert classify_risk(IntentAction.EXTRACT_DOMAIN) == IntentRisk.LOW


def test_compile_extract_domain_patch() -> None:
    intent = ArchitectureIntentV1(**_base_payload("EXTRACT_DOMAIN"))
    ops = compile_intent_patch(intent)
    assert len(ops) == 1
    assert ops[0]["op"] == "add"
    assert ops[0]["path"] == "/nodes/-"


def test_compile_define_interface_patch() -> None:
    intent = ArchitectureIntentV1(**_base_payload("DEFINE_INTERFACE"))
    ops = compile_intent_patch(intent)
    assert len(ops) == 2
    assert any(op["path"] == "/edges/-" for op in ops)


def test_compile_data_boundary_patch() -> None:
    payload = _base_payload("APPLY_DATA_BOUNDARY")
    payload["params"] = {"boundary": "strict"}
    intent = ArchitectureIntentV1(**payload)
    ops = compile_intent_patch(intent)
    assert ops[0]["op"] == "replace"
    assert "data_boundary" in ops[0]["path"]
