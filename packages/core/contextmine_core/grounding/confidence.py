"""Derived confidence for grounded findings.

Confidence is *earned*, never a literal. Two mechanisms:

- ``self_consistency``: run the same judgement N times at a non-zero temperature and
  score each finding by how often it recurs (agreement fraction).
- ``adversarial_verify``: a second judge prompted to *refute* a finding from its own
  cited evidence; a finding that survives is kept, one that is refuted is dropped.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import anyio
from pydantic import BaseModel, Field

from .judge import Evidence, GroundedFinding, JudgeResult, render_evidence

if TYPE_CHECKING:
    from contextmine_core.research.llm.provider import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class ScoredFinding:
    """A finding with confidence derived from agreement across samples."""

    finding: Any
    confidence: float
    support: int
    samples: int


def default_signature(finding: GroundedFinding) -> str:
    """Identity used to group equivalent findings across samples."""
    candidates = "|".join(sorted(getattr(finding, "candidate_ids", []) or []))
    evidence = "|".join(sorted(getattr(finding, "evidence_ids", []) or []))
    return f"{candidates}::{evidence}"


async def self_consistency(
    make_call: Callable[[], Awaitable[JudgeResult]],
    *,
    samples: int = 3,
    signature_fn: Callable[[Any], str] = default_signature,
) -> list[ScoredFinding]:
    """Run a judgement ``samples`` times and score findings by recurrence.

    Args:
        make_call: Zero-arg coroutine factory that runs one judgement. It should use
            a non-zero temperature so independent samples can disagree.
        samples: Number of independent runs.
        signature_fn: Maps a finding to a grouping key.

    Returns:
        Scored findings sorted by descending confidence; ``confidence`` is the
        fraction of samples that produced the finding.
    """
    if samples < 1:
        raise ValueError("samples must be >= 1")

    # Backend-agnostic concurrency (works under asyncio and trio).
    results: list[JudgeResult | None] = [None] * samples

    async def _run(index: int) -> None:
        results[index] = await make_call()

    async with anyio.create_task_group() as task_group:
        for sample_index in range(samples):
            task_group.start_soon(_run, sample_index)

    buckets: dict[str, list[Any]] = {}
    for result in results:
        if result is None:
            continue
        for finding in result.findings:
            buckets.setdefault(signature_fn(finding), []).append(finding)

    scored = [
        ScoredFinding(
            finding=group[0],
            confidence=len(group) / samples,
            support=len(group),
            samples=samples,
        )
        for group in buckets.values()
    ]
    scored.sort(key=lambda item: item.confidence, reverse=True)
    return scored


class _Verdict(BaseModel):
    refuted: bool = Field(description="True if the claim is NOT supported by the cited evidence")
    reason: str = Field(default="", description="Brief justification")


_VERIFY_SYSTEM = (
    "You are a skeptical reviewer. Decide only from the cited evidence whether a claim "
    "is supported. If the evidence does not clearly support it, set refuted=true. "
    "Default to refuted=true when uncertain."
)


async def adversarial_verify(
    *,
    provider: LLMProvider,
    claim: str,
    finding: GroundedFinding,
    evidence: Sequence[Evidence],
    temperature: float = 0.0,
) -> bool:
    """Return True if ``claim`` survives an adversarial refutation pass.

    Only the evidence the finding actually cited is shown to the verifier. On any
    provider failure the finding is conservatively treated as refuted (dropped).
    """
    cited_ids = set(getattr(finding, "evidence_ids", []) or [])
    cited = [item for item in evidence if item.id in cited_ids]
    if not cited:
        return False

    prompt = (
        f"Claim: {claim}\n\n"
        f"Cited evidence:\n{render_evidence(cited)}\n\n"
        "Is this claim refuted by, or unsupported by, the cited evidence?"
    )
    try:
        verdict = await provider.generate_structured(
            system=_VERIFY_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_schema=_Verdict,
            temperature=temperature,
            max_tokens=512,
        )
    except Exception as exc:
        logger.warning("Adversarial verify failed; treating as refuted: %s", exc)
        return False
    return not verdict.refuted
