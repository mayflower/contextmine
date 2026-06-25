"""Tests for grounded bounded-context detection.

The point of this detector is to group files by behaviour, not by folder path, and
to do so under the grounding invariants: closed-world (no invented files),
evidence-required, and abstention when there is no coherent domain.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from contextmine_core.grounding import FileSignal, detect_bounded_contexts
from contextmine_core.grounding.bounded_context import (
    BoundedContextFinding,
    BoundedContextOutput,
)
from contextmine_core.research.llm.provider import LLMProvider
from pydantic import BaseModel

pytestmark = pytest.mark.anyio


class ScriptedProvider(LLMProvider):
    def __init__(self, responses: list[BaseModel]):
        self._responses = responses
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "scripted"

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
        index = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return self._responses[index]


# Files in DIFFERENT folders that belong to the same business domain.
_SIGNALS = [
    FileSignal(
        path="apps/web/src/checkout.ts",
        symbols=["submitOrder", "applyCoupon"],
        dependencies=["payments", "cart"],
    ),
    FileSignal(
        path="services/billing/charge.py",
        symbols=["charge_card", "refund"],
        dependencies=["stripe"],
        tables=["payments"],
    ),
    FileSignal(
        path="lib/notify/email.py",
        symbols=["send_receipt"],
        dependencies=["smtp"],
    ),
]


async def test_groups_files_by_behaviour_across_folders() -> None:
    provider = ScriptedProvider(
        [
            BoundedContextOutput(
                findings=[
                    BoundedContextFinding(
                        name="Payments",
                        candidate_ids=["apps/web/src/checkout.ts", "services/billing/charge.py"],
                        evidence_ids=["ev-1", "ev-2"],
                        rationale="Both drive the order/charge flow.",
                    )
                ]
            )
        ]
    )

    contexts = await detect_bounded_contexts(provider=provider, signals=_SIGNALS)

    assert len(contexts) == 1
    payments = contexts[0]
    assert payments.name == "Payments"
    # Grouped two files from different folders (apps/ vs services/), not by path prefix.
    assert set(payments.file_paths) == {
        "apps/web/src/checkout.ts",
        "services/billing/charge.py",
    }
    assert payments.evidence_refs == ["apps/web/src/checkout.ts", "services/billing/charge.py"]


async def test_strips_hallucinated_file_path() -> None:
    provider = ScriptedProvider(
        [
            BoundedContextOutput(
                findings=[
                    BoundedContextFinding(
                        name="Ghost",
                        candidate_ids=["services/billing/charge.py", "does/not/exist.py"],
                        evidence_ids=["ev-2"],
                        rationale="x",
                    )
                ]
            )
        ]
    )

    contexts = await detect_bounded_contexts(provider=provider, signals=_SIGNALS)

    assert contexts[0].file_paths == ["services/billing/charge.py"]  # hallucinated path dropped


async def test_abstains_when_no_coherent_domain() -> None:
    provider = ScriptedProvider([BoundedContextOutput(findings=[])])
    contexts = await detect_bounded_contexts(provider=provider, signals=_SIGNALS)
    assert contexts == []


async def test_empty_signals_returns_empty() -> None:
    provider = ScriptedProvider([BoundedContextOutput(findings=[])])
    contexts = await detect_bounded_contexts(provider=provider, signals=[])
    assert contexts == []
    assert provider.calls == 0  # no LLM call for empty input


async def test_self_consistency_assigns_confidence() -> None:
    payments = BoundedContextFinding(
        name="Payments",
        candidate_ids=["apps/web/src/checkout.ts", "services/billing/charge.py"],
        evidence_ids=["ev-1", "ev-2"],
        rationale="charge flow",
    )
    notify = BoundedContextFinding(
        name="Notifications",
        candidate_ids=["lib/notify/email.py"],
        evidence_ids=["ev-3"],
        rationale="email",
    )
    provider = ScriptedProvider(
        [
            BoundedContextOutput(findings=[payments]),
            BoundedContextOutput(findings=[payments]),
            BoundedContextOutput(findings=[notify]),
        ]
    )

    contexts = await detect_bounded_contexts(provider=provider, signals=_SIGNALS, samples=3)

    by_name = {c.name: c for c in contexts}
    assert by_name["Payments"].confidence == pytest.approx(2 / 3)
    assert by_name["Notifications"].confidence == pytest.approx(1 / 3)
