"""Unit tests for arc42 drift reports."""

from uuid import uuid4

from contextmine_core.architecture.drift import compute_arc42_drift
from contextmine_core.architecture.schemas import (
    ArchitectureFact,
    ArchitectureFactsBundle,
    PortAdapterFact,
)


def test_compute_arc42_drift_detects_added_and_new_port() -> None:
    collection_id = uuid4()
    baseline_scenario_id = uuid4()
    scenario_id = uuid4()

    baseline = ArchitectureFactsBundle(
        collection_id=collection_id,
        scenario_id=baseline_scenario_id,
        scenario_name="baseline",
        facts=[
            ArchitectureFact(
                fact_id="container:orders",
                fact_type="container",
                title="Container orders",
                description="",
                source="deterministic",
                confidence=0.9,
            )
        ],
        ports_adapters=[],
    )

    current = ArchitectureFactsBundle(
        collection_id=collection_id,
        scenario_id=scenario_id,
        scenario_name="current",
        facts=[
            ArchitectureFact(
                fact_id="container:orders",
                fact_type="container",
                title="Container orders",
                description="",
                source="deterministic",
                confidence=0.8,
            ),
            ArchitectureFact(
                fact_id="component:orders_handler",
                fact_type="component",
                title="Component orders_handler",
                description="",
                source="deterministic",
                confidence=0.9,
            ),
        ],
        ports_adapters=[
            PortAdapterFact(
                fact_id="inbound:orders:create",
                direction="inbound",
                port_name="CreateOrder",
                adapter_name="orders_handler",
                container="orders",
                component="handler",
                protocol="http",
                source="deterministic",
                confidence=0.9,
            )
        ],
    )

    report = compute_arc42_drift(
        current,
        baseline,
        baseline_scenario_id=baseline_scenario_id,
    )

    delta_types = {delta.delta_type for delta in report.deltas}
    assert "added" in delta_types
    assert "new_port" in delta_types
    assert "changed_confidence" in delta_types
    assert report.baseline_scenario_id == baseline_scenario_id
