"""Grounded bounded-context (domain) detection.

Replaces path-prefix "domain / container" derivation
(``twin/grouping.derive_arch_group`` = ``path.split("/")[1:3]``) with a grounded LLM
decomposition: files are clustered into named bounded contexts using real signals
(symbols, dependency edges, shared data) with the file path as ONE weak feature.

Every assignment is validated closed-world against the supplied files and must cite
evidence; the detector abstains (returns nothing) when the files do not form a
coherent domain instead of bucketing them by folder. Confidence is derived from
self-consistency when ``samples > 1``, never a literal.

See ``docs/design/grounded-llm-detection.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from .confidence import self_consistency
from .judge import Candidate, Evidence, GroundedFinding, JudgeResult, judge

if TYPE_CHECKING:
    from collections.abc import Sequence

    from contextmine_core.research.llm.provider import LLMProvider


@dataclass
class FileSignal:
    """Behavioural signals for one file, assembled from the knowledge graph."""

    path: str
    symbols: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)


class BoundedContextFinding(GroundedFinding):
    """One detected bounded context (the LLM's structured output)."""

    name: str = Field(default="", description="Short bounded-context / domain name")


class BoundedContextOutput(BaseModel):
    findings: list[BoundedContextFinding] = Field(default_factory=list)


@dataclass
class BoundedContext:
    """A validated bounded context: member files + the evidence that grounds it."""

    name: str
    file_paths: list[str]
    evidence_refs: list[str]
    rationale: str
    confidence: float | None = None


_SYSTEM = (
    "You are a software-architecture domain analyst. Group files into bounded contexts "
    "(business domains) based on what the code DOES - its symbols, dependencies, and "
    "shared data - NOT merely its folder path. Only reference the files provided; never "
    "invent paths. If the files do not form coherent domains, return no findings."
)

_TASK = (
    "Group the candidate files into bounded contexts. Each finding is ONE bounded "
    "context: set candidate_ids to the member file ids, evidence_ids to the file "
    "evidence that supports grouping them, name to a short domain name, and rationale "
    "to why these files belong together. A file should appear in at most one context. "
    "Prefer behaviour (symbols, dependencies, shared tables) over folder location."
)


def _candidates(signals: Sequence[FileSignal]) -> list[Candidate]:
    return [
        Candidate(
            id=signal.path,
            label=PurePosixPath(signal.path).name,
            payload={
                "symbols": ", ".join(signal.symbols[:8]) or "-",
                "dependencies": ", ".join(signal.dependencies[:8]) or "-",
                "tables": ", ".join(signal.tables[:8]) or "-",
            },
        )
        for signal in signals
    ]


def _evidence(signals: Sequence[FileSignal]) -> list[Evidence]:
    evidence: list[Evidence] = []
    for index, signal in enumerate(signals, start=1):
        detail = "; ".join(
            part
            for part in (
                f"symbols: {', '.join(signal.symbols[:6])}" if signal.symbols else "",
                f"deps: {', '.join(signal.dependencies[:6])}" if signal.dependencies else "",
                f"tables: {', '.join(signal.tables[:6])}" if signal.tables else "",
            )
            if part
        )
        evidence.append(
            Evidence(id=f"ev-{index}", ref=signal.path, snippet=detail or "(no signals)")
        )
    return evidence


def _to_context(finding: BoundedContextFinding, ref_by_eid: dict[str, str]) -> BoundedContext:
    return BoundedContext(
        name=(finding.name or "").strip() or "unnamed",
        file_paths=list(finding.candidate_ids),
        evidence_refs=[ref_by_eid[eid] for eid in finding.evidence_ids if eid in ref_by_eid],
        rationale=finding.rationale,
    )


async def detect_bounded_contexts(
    *,
    provider: LLMProvider,
    signals: Sequence[FileSignal],
    samples: int = 1,
) -> list[BoundedContext]:
    """Detect bounded contexts from file signals.

    Args:
        provider: LLM provider.
        signals: Per-file behavioural signals (path is one weak feature among them).
        samples: When > 1, run the judgement repeatedly and derive a confidence from
            agreement across runs; otherwise a single deterministic pass.

    Returns:
        Validated bounded contexts (closed-world, evidence-required). Empty when the
        files do not form coherent domains.
    """
    if not signals:
        return []

    candidates = _candidates(signals)
    evidence = _evidence(signals)
    ref_by_eid = {item.id: item.ref for item in evidence}

    async def _run(temperature: float) -> JudgeResult:
        return await judge(
            provider=provider,
            task=_TASK,
            candidates=candidates,
            evidence=evidence,
            output_schema=BoundedContextOutput,
            system=_SYSTEM,
            require_candidate=True,
            temperature=temperature,
        )

    if samples > 1:
        scored = await self_consistency(lambda: _run(0.4), samples=samples)
        contexts: list[BoundedContext] = []
        for item in scored:
            context = _to_context(item.finding, ref_by_eid)
            context.confidence = item.confidence
            contexts.append(context)
        return contexts

    result = await _run(0.0)
    return [_to_context(finding, ref_by_eid) for finding in result.findings]
