"""Unit tests for arc42 generator."""

from uuid import uuid4

from contextmine_core.architecture.arc42 import (
    generate_arc42_from_facts,
    normalize_arc42_section_key,
)
from contextmine_core.architecture.schemas import (
    ArchitectureFact,
    ArchitectureFactsBundle,
    PortAdapterFact,
)


class _Scenario:
    def __init__(self, name: str) -> None:
        self.name = name


def test_normalize_arc42_section_key_supports_aliases() -> None:
    assert normalize_arc42_section_key("quality") == "10_quality_requirements"
    assert normalize_arc42_section_key("5") == "5_building_block_view"
    assert normalize_arc42_section_key("unknown") is None


def test_generate_arc42_from_facts_renders_selected_section() -> None:
    collection_id = uuid4()
    scenario_id = uuid4()
    bundle = ArchitectureFactsBundle(
        collection_id=collection_id,
        scenario_id=scenario_id,
        scenario_name="AS-IS",
        facts=[
            ArchitectureFact(
                fact_id="container:orders",
                fact_type="container",
                title="Container orders",
                description="Orders container",
                source="deterministic",
                confidence=0.9,
                tags=("c4", "container"),
                attributes={"container": "orders", "member_count": 5},
            ),
            ArchitectureFact(
                fact_id="quality:1",
                fact_type="quality_summary",
                title="Quality",
                description="summary",
                source="deterministic",
                confidence=0.9,
                tags=("quality",),
                attributes={
                    "metric_nodes": 3,
                    "coverage_avg": 80.0,
                    "complexity_avg": 4.0,
                    "coupling_avg": 2.0,
                    "change_frequency_avg": 1.2,
                },
            ),
        ],
        ports_adapters=[
            PortAdapterFact(
                fact_id="inbound:1",
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

    document = generate_arc42_from_facts(
        bundle,
        _Scenario("AS-IS"),
        options={"section": "quality"},
    )

    assert "10_quality_requirements" in document.sections
    assert len(document.sections) == 1
    assert "Average test coverage" in document.markdown
    assert document.section_coverage["10_quality_requirements"] is True
