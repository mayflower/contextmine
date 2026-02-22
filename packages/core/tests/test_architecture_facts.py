"""Unit tests for architecture fact helpers."""

from contextmine_core.architecture.facts import _dedupe_ports, _derive_arch_group
from contextmine_core.architecture.schemas import PortAdapterFact


def test_derive_arch_group_from_services_path() -> None:
    group = _derive_arch_group("services/billing/api/routes.py", {})
    assert group == ("billing", "api", "routes")


def test_derive_arch_group_prefers_explicit_architecture_meta() -> None:
    group = _derive_arch_group(
        "apps/web/main.ts",
        {
            "architecture": {
                "domain": "payments",
                "container": "checkout",
                "component": "payment-form",
            }
        },
    )
    assert group == ("payments", "checkout", "payment-form")


def test_dedupe_ports_prefers_higher_confidence() -> None:
    low = PortAdapterFact(
        fact_id="inbound:api:orders",
        direction="inbound",
        port_name="CreateOrder",
        adapter_name="orders_handler",
        container="orders",
        component="handler",
        protocol="http",
        source="hybrid",
        confidence=0.75,
    )
    high = PortAdapterFact(
        fact_id="inbound:api:orders",
        direction="inbound",
        port_name="CreateOrder",
        adapter_name="orders_handler",
        container="orders",
        component="handler",
        protocol="http",
        source="deterministic",
        confidence=0.9,
    )

    deduped = _dedupe_ports([low, high])
    assert len(deduped) == 1
    assert deduped[0].confidence == 0.9
    assert deduped[0].source == "deterministic"
