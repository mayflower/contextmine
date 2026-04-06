"""Evidence-backed recovered architecture model types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .schemas import EvidenceRef

RecoveredHypothesisStatus = Literal["selected", "ambiguous", "unresolved"]
RecoveredDecisionStatus = Literal["confirmed", "hypothesis"]


@dataclass(frozen=True)
class RecoveredArchitectureEntity:
    """Recovered architecture entity such as a container or external system."""

    entity_id: str
    kind: str
    name: str
    confidence: float
    evidence: tuple[EvidenceRef, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecoveredArchitectureRelationship:
    """Recovered relationship between two architecture entities."""

    source_entity_id: str
    target_entity_id: str
    kind: str
    confidence: float
    evidence: tuple[EvidenceRef, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecoveredArchitectureMembership:
    """Recovered subject-to-entity membership."""

    subject_ref: str
    entity_id: str
    relationship_kind: str
    confidence: float
    evidence: tuple[EvidenceRef, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecoveredArchitectureHypothesis:
    """Ambiguous or unresolved membership/adjudication hypothesis."""

    subject_ref: str
    candidate_entity_ids: tuple[str, ...]
    selected_entity_ids: tuple[str, ...]
    rationale: str
    status: RecoveredHypothesisStatus
    confidence: float
    evidence: tuple[EvidenceRef, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecoveredArchitectureDecision:
    """Recovered design decision backed by local evidence."""

    title: str
    summary: str
    status: RecoveredDecisionStatus
    affected_entity_ids: tuple[str, ...]
    confidence: float
    evidence: tuple[EvidenceRef, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)


def _json_ready_items(rows: tuple[Any, ...], key: str) -> list[dict[str, Any]]:
    """Convert dataclass rows into stable JSON-ready dictionaries."""

    items: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: tuple(getattr(item, name) for name in key.split(","))):
        payload = asdict(row)
        payload["evidence"] = [asdict(item) for item in row.evidence]
        items.append(payload)
    return items


@dataclass(frozen=True)
class RecoveredArchitectureModel:
    """Recovered architecture model used by facts, exporters, and agent prompts."""

    entities: tuple[RecoveredArchitectureEntity, ...] = ()
    relationships: tuple[RecoveredArchitectureRelationship, ...] = ()
    memberships: tuple[RecoveredArchitectureMembership, ...] = ()
    hypotheses: tuple[RecoveredArchitectureHypothesis, ...] = ()
    decisions: tuple[RecoveredArchitectureDecision, ...] = ()
    warnings: tuple[str, ...] = ()

    def entity_names(self, kind: str | None = None) -> list[str]:
        rows = self.entities
        if kind is not None:
            rows = tuple(entity for entity in rows if entity.kind == kind)
        return sorted(entity.name for entity in rows)

    def memberships_for(self, subject_ref: str) -> list[RecoveredArchitectureMembership]:
        return sorted(
            (
                membership
                for membership in self.memberships
                if membership.subject_ref == subject_ref
            ),
            key=lambda membership: (
                membership.subject_ref,
                membership.entity_id,
                membership.relationship_kind,
                membership.confidence,
            ),
            reverse=False,
        )

    def relationship_tuples(self, kind: str | None = None) -> list[tuple[str, str, str]]:
        rows = self.relationships
        if kind is not None:
            rows = tuple(relationship for relationship in rows if relationship.kind == kind)
        return sorted(
            (relationship.source_entity_id, relationship.target_entity_id, relationship.kind)
            for relationship in rows
        )

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "entities": _json_ready_items(self.entities, "entity_id,kind,name"),
            "relationships": _json_ready_items(
                self.relationships, "source_entity_id,target_entity_id,kind"
            ),
            "memberships": _json_ready_items(
                self.memberships, "subject_ref,entity_id,relationship_kind"
            ),
            "hypotheses": _json_ready_items(self.hypotheses, "subject_ref,status,rationale"),
            "decisions": _json_ready_items(self.decisions, "title,status,summary"),
            "warnings": list(sorted(self.warnings)),
        }
