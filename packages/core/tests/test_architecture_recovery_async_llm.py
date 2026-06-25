"""Tests for the async LLM adjudication path that wires a real provider.

The sync ``recover_architecture_model`` adjudicator path only accepted a callable /
``.adjudicate`` object, which the real (async) ``LLMProvider`` is not. These tests
cover ``adjudicate_architecture_model_async`` + ``adjudicate_packet``, including the
closed-world rejection of hallucinated selections.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from contextmine_core.architecture.recovery import (
    adjudicate_architecture_model_async,
    recover_architecture_model,
)
from contextmine_core.architecture.recovery_llm import (
    AdjudicationOutput,
    adjudicate_packet,
    build_adjudication_packet,
)
from contextmine_core.research.llm.provider import LLMProvider
from pydantic import BaseModel

from .models.architecture_recovery_fixture import build_architecture_recovery_fixture

pytestmark = pytest.mark.anyio


def _deterministic_model():
    fixture = build_architecture_recovery_fixture()
    return recover_architecture_model(fixture["nodes"], fixture["edges"], docs=fixture["docs"])


def _session(model):
    return next(row for row in model.hypotheses if row.subject_ref == "symbol:session_manager")


class FakeProvider(LLMProvider):
    """Returns a fixed AdjudicationOutput for every packet."""

    def __init__(self, output: AdjudicationOutput):
        self._output = output
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "fake"

    async def generate_text(
        self,
        *,
        system: str,
        messages: Sequence[dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        return ""

    async def generate_structured(
        self,
        *,
        system: str,
        messages: Sequence[dict[str, str]],
        output_schema: type[BaseModel],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> BaseModel:
        self.calls += 1
        return self._output


async def test_async_adjudication_selects_packet_local_candidate() -> None:
    model = _deterministic_model()
    provider = FakeProvider(
        AdjudicationOutput(
            selected_entity_ids=["container:api"],
            rationale="API runtime entrypoint evidence favours the API path.",
            evidence_ids=["ev-1"],
        )
    )

    updated = await adjudicate_architecture_model_async(model, provider)

    selected = _session(updated)
    assert selected.status == "selected"
    assert selected.selected_entity_ids == ("container:api",)
    assert [m.entity_id for m in updated.memberships_for("symbol:session_manager")] == [
        "container:api"
    ]
    assert provider.calls >= 1


async def test_async_adjudication_rejects_hallucinated_entity_closed_world() -> None:
    model = _deterministic_model()
    provider = FakeProvider(
        AdjudicationOutput(
            selected_entity_ids=["container:admin"],  # never a candidate
            rationale="Invent a new runtime.",
            evidence_ids=["ev-1"],
        )
    )

    updated = await adjudicate_architecture_model_async(model, provider)

    preserved = _session(updated)
    # Deterministic selection is preserved; nothing invented.
    assert preserved.selected_entity_ids == ("container:api", "container:worker")
    assert any("candidate" in warning.lower() for warning in updated.warnings)


async def test_adjudicate_packet_returns_validated_dict_shape() -> None:
    model = _deterministic_model()
    packet = build_adjudication_packet(model=model, hypothesis=_session(model))
    provider = FakeProvider(
        AdjudicationOutput(
            selected_entity_ids=["container:api"],
            rationale="grounded",
            evidence_ids=["ev-1"],
        )
    )

    result = await adjudicate_packet(provider, packet)

    assert result["selected_entity_ids"] == ["container:api"]
    assert result["evidence_ids"] == ["ev-1"]
    assert "rationale" in result
