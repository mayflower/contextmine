"""Claim and traceability data model for architecture evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .schemas import EvidenceRef

ArchitectureClaimStatus = Literal[
    "confirmed",
    "ambiguous",
    "conflicting",
    "unknown",
    "hypothesis",
]
ClaimDerivation = Literal["parser", "graph_fusion", "llm_adjudicated"]


def _evidence_payload(evidence: tuple[EvidenceRef, ...]) -> list[dict[str, Any]]:
    return [asdict(ref) for ref in evidence]


@dataclass(frozen=True)
class ArchitectureClaim:
    """Evidence-backed architecture claim."""

    claim_id: str
    claim_kind: str
    summary: str
    status: ArchitectureClaimStatus
    confidence: float
    entity_ids: tuple[str, ...] = ()
    relationship_ids: tuple[str, ...] = ()
    decision_ids: tuple[str, ...] = ()
    evidence: tuple[EvidenceRef, ...] = ()
    counter_evidence: tuple[EvidenceRef, ...] = ()
    derived_from: tuple[ClaimDerivation, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_kind": self.claim_kind,
            "summary": self.summary,
            "status": self.status,
            "confidence": float(self.confidence),
            "entity_ids": list(sorted(self.entity_ids)),
            "relationship_ids": list(sorted(self.relationship_ids)),
            "decision_ids": list(sorted(self.decision_ids)),
            "evidence": _evidence_payload(self.evidence),
            "counter_evidence": _evidence_payload(self.counter_evidence),
            "derived_from": list(sorted(self.derived_from)),
            "attributes": dict(sorted(self.attributes.items())),
        }


@dataclass(frozen=True)
class Arc42ClaimTraceability:
    """Section-to-claim traceability for arc42 documents."""

    section_key: str
    claim_ids: tuple[str, ...]
    summary: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "section_key": self.section_key,
            "claim_ids": list(sorted(self.claim_ids)),
            "summary": self.summary,
            "attributes": dict(sorted(self.attributes.items())),
        }


def claim_supporting_evidence(claim: ArchitectureClaim) -> tuple[EvidenceRef, ...]:
    """Return evidence that supports one claim."""

    return claim.evidence


def claim_counter_evidence(claim: ArchitectureClaim) -> tuple[EvidenceRef, ...]:
    """Return counter-evidence attached to one claim."""

    return claim.counter_evidence
