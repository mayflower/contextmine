"""Verification models for answer validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class VerificationStatus(str, Enum):
    """Status of answer verification."""

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class CitationVerification:
    """Verification result for a single citation.

    Checks whether a citation ID referenced in the answer
    corresponds to an actual evidence item in the run.
    """

    citation_id: str
    """The citation ID found in the answer."""

    found: bool
    """Whether the citation references an existing evidence item."""

    evidence_snippet: str | None = None
    """First 100 chars of evidence content if found."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "citation_id": self.citation_id,
            "found": self.found,
            "evidence_snippet": self.evidence_snippet,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CitationVerification:
        """Create from dictionary."""
        return cls(
            citation_id=data["citation_id"],
            found=data["found"],
            evidence_snippet=data.get("evidence_snippet"),
        )


@dataclass
class EvidenceSupportScore:
    """Score for how well evidence supports the answer.

    Uses heuristics based on evidence quantity, relevance scores,
    and coverage.
    """

    score: float
    """Support score from 0.0 to 1.0."""

    reasoning: str
    """Explanation of how the score was calculated."""

    supporting_evidence_ids: list[str] = field(default_factory=list)
    """IDs of evidence items that support the answer."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "reasoning": self.reasoning,
            "supporting_evidence_ids": self.supporting_evidence_ids,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceSupportScore:
        """Create from dictionary."""
        return cls(
            score=data["score"],
            reasoning=data["reasoning"],
            supporting_evidence_ids=data.get("supporting_evidence_ids", []),
        )


@dataclass
class SemanticGrounding:
    """Result of semantic grounding check.

    Verifies that claims in the answer are actually supported
    by the content of the collected evidence.
    """

    is_grounded: bool
    """Whether the answer is grounded in evidence."""

    grounding_score: float
    """Score from 0.0 to 1.0 indicating how well grounded."""

    ungrounded_claims: list[str]
    """Claims in the answer not supported by evidence."""

    reasoning: str
    """Explanation of the grounding assessment."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_grounded": self.is_grounded,
            "grounding_score": self.grounding_score,
            "ungrounded_claims": self.ungrounded_claims,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SemanticGrounding:
        """Create from dictionary."""
        return cls(
            is_grounded=data["is_grounded"],
            grounding_score=data["grounding_score"],
            ungrounded_claims=data.get("ungrounded_claims", []),
            reasoning=data["reasoning"],
        )


@dataclass
class ConfidenceCalibration:
    """Comparison of stated confidence vs evidence quality.

    Checks whether the confidence level in the answer is
    appropriate given the evidence collected.
    """

    stated_confidence: float
    """Confidence stated in the answer (0.0 to 1.0)."""

    evidence_confidence: float
    """Confidence derived from evidence quality (0.0 to 1.0)."""

    calibration_delta: float
    """Absolute difference between stated and evidence confidence."""

    is_calibrated: bool
    """Whether the delta is within acceptable tolerance."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stated_confidence": self.stated_confidence,
            "evidence_confidence": self.evidence_confidence,
            "calibration_delta": self.calibration_delta,
            "is_calibrated": self.is_calibrated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConfidenceCalibration:
        """Create from dictionary."""
        return cls(
            stated_confidence=data["stated_confidence"],
            evidence_confidence=data["evidence_confidence"],
            calibration_delta=data["calibration_delta"],
            is_calibrated=data["is_calibrated"],
        )


@dataclass
class VerificationResult:
    """Complete verification result for a research run.

    Aggregates citation verification, evidence support scoring,
    semantic grounding, and confidence calibration into a final verdict.
    """

    status: VerificationStatus
    """Overall verification status."""

    citations: list[CitationVerification]
    """Verification results for each citation."""

    evidence_support: EvidenceSupportScore
    """Score for evidence support."""

    semantic_grounding: SemanticGrounding | None
    """Semantic grounding check result (None if not performed)."""

    confidence_calibration: ConfidenceCalibration
    """Confidence calibration result."""

    issues: list[str]
    """List of issues found during verification."""

    verified_at: str
    """ISO timestamp of verification."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "citations": [c.to_dict() for c in self.citations],
            "evidence_support": self.evidence_support.to_dict(),
            "semantic_grounding": self.semantic_grounding.to_dict()
            if self.semantic_grounding
            else None,
            "confidence_calibration": self.confidence_calibration.to_dict(),
            "issues": self.issues,
            "verified_at": self.verified_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerificationResult:
        """Create from dictionary."""
        sg_data = data.get("semantic_grounding")
        return cls(
            status=VerificationStatus(data["status"]),
            citations=[CitationVerification.from_dict(c) for c in data["citations"]],
            evidence_support=EvidenceSupportScore.from_dict(data["evidence_support"]),
            semantic_grounding=SemanticGrounding.from_dict(sg_data) if sg_data else None,
            confidence_calibration=ConfidenceCalibration.from_dict(data["confidence_calibration"]),
            issues=data["issues"],
            verified_at=data["verified_at"],
        )

    @property
    def citation_validity_rate(self) -> float:
        """Calculate percentage of valid citations."""
        if not self.citations:
            return 1.0  # No citations = no invalid citations
        valid = sum(1 for c in self.citations if c.found)
        return valid / len(self.citations)

    @classmethod
    def create_passed(
        cls,
        citations: list[CitationVerification],
        evidence_support: EvidenceSupportScore,
        confidence_calibration: ConfidenceCalibration,
        semantic_grounding: SemanticGrounding | None = None,
    ) -> VerificationResult:
        """Create a passing verification result."""
        return cls(
            status=VerificationStatus.PASSED,
            citations=citations,
            evidence_support=evidence_support,
            semantic_grounding=semantic_grounding,
            confidence_calibration=confidence_calibration,
            issues=[],
            verified_at=datetime.now(UTC).isoformat(),
        )

    @classmethod
    def create_failed(
        cls,
        citations: list[CitationVerification],
        evidence_support: EvidenceSupportScore,
        confidence_calibration: ConfidenceCalibration,
        issues: list[str],
        semantic_grounding: SemanticGrounding | None = None,
    ) -> VerificationResult:
        """Create a failed verification result."""
        return cls(
            status=VerificationStatus.FAILED,
            citations=citations,
            evidence_support=evidence_support,
            semantic_grounding=semantic_grounding,
            confidence_calibration=confidence_calibration,
            issues=issues,
            verified_at=datetime.now(UTC).isoformat(),
        )
