"""GroundedJudge: constrained, evidence-validated LLM detection.

Wraps ``LLMProvider.generate_structured`` so detectors only ever *select / label*
over real candidates and real evidence, and the model output is validated back
against the supplied sets (closed-world + evidence-required) before it is trusted.

This generalizes the ``analyzer/extractors/triage.py`` pattern (propose -> validate
against the input set -> fall back). See ``docs/design/grounded-llm-detection.md``.

The two enforced anti-guessing rules here:

- Closed-world: any candidate/evidence id the model returns that was not supplied is
  dropped, never stored.
- Evidence-required: a finding with no surviving evidence id is dropped.

Confidence is intentionally *not* part of the model schema; it is derived separately
(see ``confidence.py``) so detectors never write a hardcoded confidence literal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Sequence

    from contextmine_core.research.llm.provider import LLMProvider

logger = logging.getLogger(__name__)


class Candidate(BaseModel):
    """A real thing the model is allowed to select (file, symbol, node, ...)."""

    id: str = Field(description="Stable id the model must reference to select this candidate")
    label: str = Field(default="", description="Human-readable label")
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Features shown to the model for this candidate"
    )


class Evidence(BaseModel):
    """A real, resolvable piece of supporting evidence."""

    id: str = Field(description="Evidence id the model cites, e.g. 'ev-1'")
    ref: str = Field(description="Concrete reference: file:line, node id, or edge id")
    snippet: str = Field(default="", description="Short excerpt or description")


class GroundedFinding(BaseModel):
    """Base class for detector outputs. Subclass to add task-specific fields."""

    candidate_ids: list[str] = Field(
        default_factory=list,
        description="Selected candidate ids (must come from the supplied candidates)",
    )
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="Evidence ids that support this finding (must come from the supplied evidence)",
    )
    rationale: str = Field(
        default="", description="Why this finding follows from the cited evidence"
    )


@dataclass
class JudgeResult:
    """Validated output of a single grounded judgement."""

    findings: list[Any] = field(default_factory=list)
    dropped: list[tuple[Any, str]] = field(default_factory=list)
    raw: Any = None

    @property
    def abstained(self) -> bool:
        """True when no finding survived validation (an explicit, valid outcome)."""
        return not self.findings


GROUNDED_SYSTEM = (
    "You are a precise code-analysis judge. You may only select from the candidates "
    "and cite the evidence provided. Never invent ids. If the evidence does not "
    "support a conclusion, return no findings rather than guessing."
)


def render_candidates(candidates: Sequence[Candidate]) -> str:
    """Render candidates as an id-prefixed list for the prompt."""
    if not candidates:
        return "(none)"
    lines: list[str] = []
    for candidate in candidates:
        payload = (
            ", ".join(f"{key}={value}" for key, value in candidate.payload.items())
            if candidate.payload
            else ""
        )
        suffix = f" [{payload}]" if payload else ""
        lines.append(f"- {candidate.id}: {candidate.label}{suffix}")
    return "\n".join(lines)


def render_evidence(evidence: Sequence[Evidence]) -> str:
    """Render evidence as an id-prefixed list for the prompt."""
    if not evidence:
        return "(none)"
    return "\n".join(f"- {item.id} ({item.ref}): {item.snippet}" for item in evidence)


def _build_prompt(
    *,
    task: str,
    candidates: Sequence[Candidate],
    evidence: Sequence[Evidence],
) -> str:
    return (
        f"{task}\n\n"
        f"CANDIDATES (select only by these ids):\n{render_candidates(candidates)}\n\n"
        f"EVIDENCE (cite only these ids):\n{render_evidence(evidence)}\n\n"
        "Each finding MUST list candidate_ids drawn only from the candidates above and "
        "evidence_ids drawn only from the evidence above. Do not invent ids. "
        "If nothing is supported by the evidence, return an empty list of findings."
    )


async def judge(
    *,
    provider: LLMProvider,
    task: str,
    candidates: Sequence[Candidate],
    evidence: Sequence[Evidence],
    output_schema: type[BaseModel],
    system: str = GROUNDED_SYSTEM,
    findings_attr: str = "findings",
    require_evidence: bool = True,
    require_candidate: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> JudgeResult:
    """Run one grounded judgement.

    Args:
        provider: LLM provider used for structured generation.
        task: The instruction describing what to detect/label.
        candidates: Closed set the model may select from.
        evidence: Closed set the model may cite from.
        output_schema: Pydantic model with a ``findings`` list of ``GroundedFinding``.
        system: System prompt.
        findings_attr: Attribute on ``output_schema`` holding the findings list.
        require_evidence: Drop findings with no surviving evidence id.
        require_candidate: Drop findings with no surviving candidate id.
        temperature: Sampling temperature (0 = deterministic; >0 for self-consistency).
        max_tokens: Max output tokens.

    Returns:
        ``JudgeResult`` with validated findings (hallucinated ids stripped, uncited
        findings dropped). On provider failure, returns an empty (abstained) result.
    """
    candidate_ids = {candidate.id for candidate in candidates}
    evidence_ids = {item.id for item in evidence}
    prompt = _build_prompt(task=task, candidates=candidates, evidence=evidence)

    try:
        raw = await provider.generate_structured(
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_schema=output_schema,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        logger.warning("Grounded judge failed; abstaining: %s", exc)
        return JudgeResult(findings=[], raw=None)

    raw_findings = list(getattr(raw, findings_attr, None) or [])
    kept: list[Any] = []
    dropped: list[tuple[Any, str]] = []

    for finding in raw_findings:
        supplied_candidate_ids = list(getattr(finding, "candidate_ids", []) or [])
        supplied_evidence_ids = list(getattr(finding, "evidence_ids", []) or [])

        valid_candidate_ids = [cid for cid in supplied_candidate_ids if cid in candidate_ids]
        valid_evidence_ids = [eid for eid in supplied_evidence_ids if eid in evidence_ids]
        hallucinated = (set(supplied_candidate_ids) - candidate_ids) | (
            set(supplied_evidence_ids) - evidence_ids
        )

        # Closed-world: keep only ids that were actually supplied.
        finding.candidate_ids = valid_candidate_ids
        finding.evidence_ids = valid_evidence_ids

        if require_evidence and not valid_evidence_ids:
            dropped.append((finding, "no valid evidence ids"))
            continue
        if require_candidate and not valid_candidate_ids:
            dropped.append((finding, "no valid candidate ids"))
            continue
        if hallucinated:
            logger.debug("Grounded judge stripped hallucinated ids: %s", sorted(hallucinated))
        kept.append(finding)

    return JudgeResult(findings=kept, dropped=dropped, raw=raw)
