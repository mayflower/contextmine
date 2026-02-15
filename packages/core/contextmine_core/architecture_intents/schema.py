"""Schema definitions for versioned architecture intents."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class IntentAction(str, Enum):
    """Supported intent actions for v1."""

    EXTRACT_DOMAIN = "EXTRACT_DOMAIN"
    SPLIT_CONTAINER = "SPLIT_CONTAINER"
    MOVE_COMPONENT = "MOVE_COMPONENT"
    DEFINE_INTERFACE = "DEFINE_INTERFACE"
    SET_VALIDATOR = "SET_VALIDATOR"
    APPLY_DATA_BOUNDARY = "APPLY_DATA_BOUNDARY"


class IntentTarget(BaseModel):
    """Intent target descriptor."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["node", "edge", "context", "service"]
    id: str = Field(min_length=1, max_length=2048)


class ArchitectureIntentV1(BaseModel):
    """Strict intent schema consumed by API and MCP."""

    model_config = ConfigDict(extra="forbid")

    intent_version: Literal["1.0"]
    scenario_id: UUID
    intent_id: UUID | None = None
    action: IntentAction
    target: IntentTarget
    params: dict[str, Any] = Field(default_factory=dict)
    expected_scenario_version: int = Field(ge=1)
    requested_by: UUID | None = None


class IntentRisk(str, Enum):
    """Execution risk class used by gate logic."""

    LOW = "low"
    HIGH = "high"


HIGH_RISK_ACTIONS: set[IntentAction] = {
    IntentAction.SPLIT_CONTAINER,
    IntentAction.APPLY_DATA_BOUNDARY,
}


def classify_risk(action: IntentAction) -> IntentRisk:
    """Return risk level for a given action."""
    if action in HIGH_RISK_ACTIONS:
        return IntentRisk.HIGH
    return IntentRisk.LOW
