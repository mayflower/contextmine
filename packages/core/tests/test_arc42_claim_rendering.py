"""Tests for claim-backed arc42 rendering."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

from contextmine_core.architecture.arc42 import render_claim_backed_arc42
from contextmine_core.architecture.claim_model import Arc42ClaimTraceability, ArchitectureClaim
from contextmine_core.architecture.schemas import (
    ArchitectureFact,
    ArchitectureFactsBundle,
    EvidenceRef,
)


def _evidence(ref: str, *, kind: str = "file") -> EvidenceRef:
    return EvidenceRef(kind=kind, ref=ref)


def _bundle() -> ArchitectureFactsBundle:
    return ArchitectureFactsBundle(
        collection_id=uuid4(),
        scenario_id=uuid4(),
        scenario_name="AS-IS",
        facts=[
            ArchitectureFact(
                fact_id="architecture_decision:adr-001",
                fact_type="architecture_decision",
                title="ADR-001 async embedding workers",
                description="Embeddings generation runs in the worker runtime.",
                source="deterministic",
                confidence=0.94,
                attributes={"affected_entity_ids": ["container:api", "container:worker"]},
                evidence=(_evidence("docs/adr/001-async-embedding-workers.md"),),
            ),
            ArchitectureFact(
                fact_id="recovered_hypothesis:symbol:session_manager",
                fact_type="recovered_hypothesis",
                title="Recovered hypothesis",
                description="Shared session manager code remains ambiguous across API and worker.",
                source="hybrid",
                confidence=0.88,
                attributes={
                    "subject_ref": "symbol:session_manager",
                    "candidate_entity_ids": ["container:api", "container:worker"],
                    "selected_entity_ids": ["container:api", "container:worker"],
                    "status": "ambiguous",
                },
                evidence=(_evidence("packages/core/session_manager.py"),),
            ),
        ],
        warnings=["Ambiguous recovered memberships: 1."],
    )


def _claims() -> list[ArchitectureClaim]:
    return [
        ArchitectureClaim(
            claim_id="claim:runtime:worker",
            claim_kind="runtime_assignment",
            summary="Embeddings run in the worker runtime.",
            status="confirmed",
            confidence=0.93,
            entity_ids=("container:worker",),
            decision_ids=("ADR-001 async embedding workers",),
            evidence=(_evidence("docs/adr/001-async-embedding-workers.md"),),
            counter_evidence=(_evidence("services/contextmine/api/embeddings.py"),),
            derived_from=("parser", "graph_fusion"),
        ),
        ArchitectureClaim(
            claim_id="claim:session:shared",
            claim_kind="ownership",
            summary="Session manager code is shared between API and worker.",
            status="ambiguous",
            confidence=0.72,
            entity_ids=("container:api", "container:worker"),
            evidence=(_evidence("packages/core/session_manager.py"),),
            counter_evidence=(),
            derived_from=("graph_fusion",),
        ),
    ]


def _traceability() -> list[Arc42ClaimTraceability]:
    return [
        Arc42ClaimTraceability(
            section_key="5_building_block_view",
            claim_ids=("claim:runtime:worker", "claim:session:shared"),
            summary="Building block view relies on runtime assignment and shared ownership claims.",
        ),
        Arc42ClaimTraceability(
            section_key="9_architecture_decisions",
            claim_ids=("claim:runtime:worker",),
            summary="Decision section is grounded in ADR-backed worker runtime claim.",
        ),
    ]


def test_render_claim_backed_arc42_carries_claims_and_traceability() -> None:
    document = render_claim_backed_arc42(
        _bundle(),
        SimpleNamespace(name="AS-IS"),
        claims=_claims(),
        claim_traceability=_traceability(),
    )

    assert document.claims
    assert document.claim_traceability
    assert document.claim_ids_for_section("5_building_block_view") == [
        "claim:runtime:worker",
        "claim:session:shared",
    ]
    assert "claim:runtime:worker" in document.markdown


def test_render_claim_backed_arc42_makes_ambiguity_and_traceability_visible() -> None:
    document = render_claim_backed_arc42(
        _bundle(),
        SimpleNamespace(name="AS-IS"),
        claims=_claims(),
        claim_traceability=_traceability(),
    )

    assert "UNKNOWN: ambiguous claim" in document.sections["5_building_block_view"]
    assert "Ambiguous recovered memberships: 1." in document.warnings
    assert any(trace.section_key == "9_architecture_decisions" for trace in document.claim_traceability)


def test_render_claim_backed_arc42_does_not_invent_unbacked_claims() -> None:
    document = render_claim_backed_arc42(
        _bundle(),
        SimpleNamespace(name="AS-IS"),
        claims=_claims()[:1],
        claim_traceability=_traceability()[:1],
    )

    assert [claim.claim_id for claim in document.claims] == ["claim:runtime:worker"]
    assert "claim:session:shared" not in document.markdown
