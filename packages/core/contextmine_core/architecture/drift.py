"""Advisory drift reporting for architecture facts."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from .schemas import Arc42DriftReport, ArchitectureFact, ArchitectureFactsBundle, DriftDelta


def _fact_index(facts: list[ArchitectureFact]) -> dict[str, ArchitectureFact]:
    return {fact.fact_id: fact for fact in facts}


def _port_key(row: dict[str, Any]) -> tuple[str, str, str | None]:
    return (
        str(row.get("direction")),
        str(row.get("port_name")),
        row.get("adapter_name"),
    )


def compute_arc42_drift(
    current: ArchitectureFactsBundle,
    baseline: ArchitectureFactsBundle | None,
    *,
    baseline_scenario_id: UUID | None = None,
) -> Arc42DriftReport:
    """Compute architecture drift deltas between current and baseline fact bundles."""

    report = Arc42DriftReport(
        collection_id=current.collection_id,
        scenario_id=current.scenario_id,
        baseline_scenario_id=baseline_scenario_id,
        generated_at=datetime.now(UTC),
        current_hash=current.facts_hash(),
        baseline_hash=baseline.facts_hash() if baseline else None,
        deltas=[],
        warnings=[],
    )

    if baseline is None:
        report.warnings.append("No baseline scenario provided; drift comparison skipped.")
        return report

    current_facts = _fact_index(current.facts)
    baseline_facts = _fact_index(baseline.facts)

    current_ids = set(current_facts)
    baseline_ids = set(baseline_facts)

    for fact_id in sorted(current_ids - baseline_ids):
        fact = current_facts[fact_id]
        report.deltas.append(
            DriftDelta(
                delta_type="added",
                subject=fact.fact_id,
                detail=f"Added {fact.fact_type}: {fact.title}",
                after={
                    "fact_type": fact.fact_type,
                    "title": fact.title,
                    "confidence": fact.confidence,
                },
                confidence=float(fact.confidence),
            )
        )

    for fact_id in sorted(baseline_ids - current_ids):
        fact = baseline_facts[fact_id]
        report.deltas.append(
            DriftDelta(
                delta_type="removed",
                subject=fact.fact_id,
                detail=f"Removed {fact.fact_type}: {fact.title}",
                before={
                    "fact_type": fact.fact_type,
                    "title": fact.title,
                    "confidence": fact.confidence,
                },
                confidence=float(fact.confidence),
            )
        )

    for fact_id in sorted(current_ids & baseline_ids):
        current_fact = current_facts[fact_id]
        baseline_fact = baseline_facts[fact_id]
        if abs(float(current_fact.confidence) - float(baseline_fact.confidence)) >= 0.05:
            report.deltas.append(
                DriftDelta(
                    delta_type="changed_confidence",
                    subject=fact_id,
                    detail="Confidence changed",
                    before={"confidence": baseline_fact.confidence},
                    after={"confidence": current_fact.confidence},
                    confidence=max(float(current_fact.confidence), float(baseline_fact.confidence)),
                )
            )

    current_ports = {row.fact_id: row for row in current.ports_adapters}
    baseline_ports = {row.fact_id: row for row in baseline.ports_adapters}

    current_port_ids = set(current_ports)
    baseline_port_ids = set(baseline_ports)

    for port_id in sorted(current_port_ids - baseline_port_ids):
        port = current_ports[port_id]
        report.deltas.append(
            DriftDelta(
                delta_type="new_port",
                subject=port.fact_id,
                detail=f"New {port.direction} port: {port.port_name}",
                after={
                    "direction": port.direction,
                    "port_name": port.port_name,
                    "adapter_name": port.adapter_name,
                    "container": port.container,
                    "component": port.component,
                },
                confidence=float(port.confidence),
            )
        )

    for port_id in sorted(baseline_port_ids - current_port_ids):
        port = baseline_ports[port_id]
        report.deltas.append(
            DriftDelta(
                delta_type="removed_adapter",
                subject=port.fact_id,
                detail=f"Removed adapter for {port.direction} port: {port.port_name}",
                before={
                    "direction": port.direction,
                    "port_name": port.port_name,
                    "adapter_name": port.adapter_name,
                    "container": port.container,
                    "component": port.component,
                },
                confidence=float(port.confidence),
            )
        )

    # Heuristic moved-component detection: same port tuple but ownership changed.
    current_by_key = {_port_key(asdict(row)): row for row in current.ports_adapters}
    baseline_by_key = {_port_key(asdict(row)): row for row in baseline.ports_adapters}

    for key in sorted(set(current_by_key) & set(baseline_by_key)):
        cur = current_by_key[key]
        prev = baseline_by_key[key]
        if cur.container != prev.container or cur.component != prev.component:
            report.deltas.append(
                DriftDelta(
                    delta_type="moved_component",
                    subject=cur.port_name,
                    detail="Port ownership moved between components",
                    before={
                        "container": prev.container,
                        "component": prev.component,
                    },
                    after={
                        "container": cur.container,
                        "component": cur.component,
                    },
                    confidence=max(float(cur.confidence), float(prev.confidence)),
                )
            )

    return report
