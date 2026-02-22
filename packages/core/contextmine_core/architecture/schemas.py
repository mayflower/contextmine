"""Schemas for architecture facts, arc42 documents, and drift deltas."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

FactSource = Literal["deterministic", "hybrid", "llm"]
DeltaType = Literal[
    "added",
    "removed",
    "changed_confidence",
    "moved_component",
    "new_port",
    "removed_adapter",
]
PortDirection = Literal["inbound", "outbound"]


@dataclass(frozen=True)
class EvidenceRef:
    """Evidence pointer for a fact."""

    kind: Literal["file", "node", "edge", "artifact"]
    ref: str
    start_line: int | None = None
    end_line: int | None = None


@dataclass(frozen=True)
class ArchitectureFact:
    """One architecture-level fact extracted from twin/KG data."""

    fact_id: str
    fact_type: str
    title: str
    description: str
    source: FactSource
    confidence: float
    tags: tuple[str, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)
    evidence: tuple[EvidenceRef, ...] = ()


@dataclass(frozen=True)
class PortAdapterFact:
    """A Ports-and-Adapters mapping fact."""

    fact_id: str
    direction: PortDirection
    port_name: str
    adapter_name: str | None
    container: str | None
    component: str | None
    protocol: str | None
    source: FactSource
    confidence: float
    attributes: dict[str, Any] = field(default_factory=dict)
    evidence: tuple[EvidenceRef, ...] = ()


@dataclass
class ArchitectureFactsBundle:
    """Aggregated architecture facts for a collection/scenario."""

    collection_id: UUID
    scenario_id: UUID
    scenario_name: str
    facts: list[ArchitectureFact] = field(default_factory=list)
    ports_adapters: list[PortAdapterFact] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def canonical_payload(self) -> dict[str, Any]:
        facts = [
            {
                **asdict(fact),
                "tags": list(fact.tags),
                "evidence": [asdict(ev) for ev in fact.evidence],
            }
            for fact in sorted(self.facts, key=lambda row: row.fact_id)
        ]
        ports = [
            {
                **asdict(fact),
                "evidence": [asdict(ev) for ev in fact.evidence],
            }
            for fact in sorted(self.ports_adapters, key=lambda row: row.fact_id)
        ]
        return {
            "collection_id": str(self.collection_id),
            "scenario_id": str(self.scenario_id),
            "facts": facts,
            "ports_adapters": ports,
        }

    def facts_hash(self) -> str:
        encoded = json.dumps(self.canonical_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass
class Arc42Document:
    """Rendered arc42 document plus machine-readable metadata."""

    collection_id: UUID
    scenario_id: UUID
    scenario_name: str
    title: str
    generated_at: datetime
    sections: dict[str, str]
    markdown: str
    warnings: list[str] = field(default_factory=list)
    confidence_summary: dict[str, Any] = field(default_factory=dict)
    section_coverage: dict[str, bool] = field(default_factory=dict)

    @classmethod
    def empty(cls, *, collection_id: UUID, scenario_id: UUID, scenario_name: str) -> Arc42Document:
        now = datetime.now(UTC)
        return cls(
            collection_id=collection_id,
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            title=f"arc42 - {scenario_name}",
            generated_at=now,
            sections={},
            markdown="",
        )


@dataclass(frozen=True)
class DriftDelta:
    """One delta between baseline and target architecture states."""

    delta_type: DeltaType
    subject: str
    detail: str
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    confidence: float = 0.9


@dataclass
class Arc42DriftReport:
    """Drift report for arc42-relevant facts."""

    collection_id: UUID
    scenario_id: UUID
    baseline_scenario_id: UUID | None
    generated_at: datetime
    current_hash: str
    baseline_hash: str | None
    deltas: list[DriftDelta] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def summarize_confidence(
    facts: list[ArchitectureFact], ports: list[PortAdapterFact]
) -> dict[str, Any]:
    """Compute confidence summary used in artifact metadata and API outputs."""

    buckets: dict[str, list[float]] = {"deterministic": [], "hybrid": [], "llm": []}
    for row in facts:
        buckets.setdefault(row.source, []).append(float(row.confidence))
    for row in ports:
        buckets.setdefault(row.source, []).append(float(row.confidence))

    by_source: dict[str, Any] = {}
    for source, values in buckets.items():
        if not values:
            by_source[source] = {"count": 0, "avg": None}
            continue
        by_source[source] = {
            "count": len(values),
            "avg": round(sum(values) / len(values), 4),
        }

    all_values = [value for values in buckets.values() for value in values]
    return {
        "total": len(all_values),
        "avg": round(sum(all_values) / len(all_values), 4) if all_values else None,
        "by_source": by_source,
    }
