"""Tests for the grounded LLM detection primitives.

These assert the anti-guessing invariants directly: hallucinated ids are stripped,
uncited findings are dropped, abstention is a valid outcome, confidence is derived
from agreement, and adversarial verification drops refuted findings.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from contextmine_core.grounding import (
    Candidate,
    Evidence,
    GroundedFinding,
    adversarial_verify,
    judge,
    self_consistency,
)
from contextmine_core.grounding.confidence import _Verdict
from contextmine_core.research.llm.provider import LLMProvider
from pydantic import BaseModel

pytestmark = pytest.mark.anyio


class DemoFinding(GroundedFinding):
    name: str = ""


class DemoOutput(BaseModel):
    findings: list[DemoFinding] = []


class ScriptedProvider(LLMProvider):
    """Returns pre-scripted structured outputs, one per call."""

    def __init__(self, responses: list[BaseModel] | None = None, *, raises: bool = False):
        self._responses = list(responses or [])
        self._raises = raises
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
        if self._raises:
            raise RuntimeError("scripted failure")
        index = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return self._responses[index]


def _candidates() -> list[Candidate]:
    return [Candidate(id="c1", label="alpha"), Candidate(id="c2", label="beta")]


def _evidence() -> list[Evidence]:
    return [Evidence(id="ev-1", ref="a.py:1", snippet="x"), Evidence(id="ev-2", ref="b.py:2")]


async def test_judge_strips_hallucinated_candidate_ids() -> None:
    provider = ScriptedProvider(
        [DemoOutput(findings=[DemoFinding(candidate_ids=["c1", "c99"], evidence_ids=["ev-1"])])]
    )
    result = await judge(
        provider=provider,
        task="detect",
        candidates=_candidates(),
        evidence=_evidence(),
        output_schema=DemoOutput,
    )
    assert len(result.findings) == 1
    # c99 was never supplied -> stripped; c1 kept.
    assert result.findings[0].candidate_ids == ["c1"]
    assert result.findings[0].evidence_ids == ["ev-1"]


async def test_judge_drops_uncited_finding() -> None:
    provider = ScriptedProvider(
        [DemoOutput(findings=[DemoFinding(candidate_ids=["c1"], evidence_ids=["ev-99"])])]
    )
    result = await judge(
        provider=provider,
        task="detect",
        candidates=_candidates(),
        evidence=_evidence(),
        output_schema=DemoOutput,
    )
    assert result.findings == []
    assert result.abstained is True
    assert len(result.dropped) == 1
    assert "evidence" in result.dropped[0][1]


async def test_judge_abstains_on_empty_output() -> None:
    provider = ScriptedProvider([DemoOutput(findings=[])])
    result = await judge(
        provider=provider,
        task="detect",
        candidates=_candidates(),
        evidence=_evidence(),
        output_schema=DemoOutput,
    )
    assert result.abstained is True


async def test_judge_abstains_on_provider_error() -> None:
    provider = ScriptedProvider(raises=True)
    result = await judge(
        provider=provider,
        task="detect",
        candidates=_candidates(),
        evidence=_evidence(),
        output_schema=DemoOutput,
    )
    assert result.abstained is True
    assert result.raw is None


async def test_require_candidate_drops_finding_without_candidate() -> None:
    provider = ScriptedProvider(
        [DemoOutput(findings=[DemoFinding(candidate_ids=[], evidence_ids=["ev-1"])])]
    )
    result = await judge(
        provider=provider,
        task="detect",
        candidates=_candidates(),
        evidence=_evidence(),
        output_schema=DemoOutput,
        require_candidate=True,
    )
    assert result.findings == []
    assert "candidate" in result.dropped[0][1]


async def test_self_consistency_scores_by_agreement() -> None:
    # Finding A appears in 2 of 3 samples; finding B in 1 of 3.
    finding_a = DemoFinding(candidate_ids=["c1"], evidence_ids=["ev-1"], name="A")
    finding_b = DemoFinding(candidate_ids=["c2"], evidence_ids=["ev-2"], name="B")
    provider = ScriptedProvider(
        [
            DemoOutput(findings=[finding_a]),
            DemoOutput(findings=[finding_a]),
            DemoOutput(findings=[finding_b]),
        ]
    )

    async def make_call() -> object:
        return await judge(
            provider=provider,
            task="detect",
            candidates=_candidates(),
            evidence=_evidence(),
            output_schema=DemoOutput,
            temperature=0.4,
        )

    scored = await self_consistency(make_call, samples=3)
    confidences = sorted((round(s.confidence, 3) for s in scored), reverse=True)
    assert confidences == [round(2 / 3, 3), round(1 / 3, 3)]


async def test_adversarial_verify_keeps_supported_and_drops_refuted() -> None:
    finding = DemoFinding(candidate_ids=["c1"], evidence_ids=["ev-1"])

    kept_provider = ScriptedProvider([_Verdict(refuted=False)])
    survived = await adversarial_verify(
        provider=kept_provider,
        claim="alpha is a thing",
        finding=finding,
        evidence=_evidence(),
    )
    assert survived is True

    refute_provider = ScriptedProvider([_Verdict(refuted=True, reason="no support")])
    survived = await adversarial_verify(
        provider=refute_provider,
        claim="alpha is a thing",
        finding=finding,
        evidence=_evidence(),
    )
    assert survived is False


async def test_adversarial_verify_drops_when_no_cited_evidence() -> None:
    finding = DemoFinding(candidate_ids=["c1"], evidence_ids=[])
    provider = ScriptedProvider([_Verdict(refuted=False)])
    survived = await adversarial_verify(
        provider=provider,
        claim="alpha",
        finding=finding,
        evidence=_evidence(),
    )
    assert survived is False
