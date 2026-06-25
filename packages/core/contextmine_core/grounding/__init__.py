"""Grounded LLM detection primitives.

Constrained, evidence-validated LLM judgement used to replace keyword/string-matching
heuristics with grounded detection. See ``docs/design/grounded-llm-detection.md``.
"""

from __future__ import annotations

from .confidence import ScoredFinding, adversarial_verify, default_signature, self_consistency
from .judge import (
    GROUNDED_SYSTEM,
    Candidate,
    Evidence,
    GroundedFinding,
    JudgeResult,
    judge,
    render_candidates,
    render_evidence,
)

__all__ = [
    "GROUNDED_SYSTEM",
    "Candidate",
    "Evidence",
    "GroundedFinding",
    "JudgeResult",
    "ScoredFinding",
    "adversarial_verify",
    "default_signature",
    "judge",
    "render_candidates",
    "render_evidence",
    "self_consistency",
]
