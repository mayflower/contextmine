"""Tests for architecture claims and claim traceability."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4

from contextmine_core.architecture.claim_model import (
    Arc42ClaimTraceability,
    ArchitectureClaim,
    claim_counter_evidence,
    claim_supporting_evidence,
)
from contextmine_core.architecture.schemas import Arc42Document, EvidenceRef


def _evidence(ref: str, *, kind: str = "file") -> EvidenceRef:
    return EvidenceRef(kind=kind, ref=ref)


def test_architecture_claim_exposes_required_fields_and_statuses() -> None:
    claim = ArchitectureClaim(
        claim_id="claim:runtime:worker",
        claim_kind="runtime_assignment",
        summary="Embeddings run in the worker runtime.",
        status="confirmed",
        confidence=0.93,
        entity_ids=("container:worker",),
        relationship_ids=("container:api:publishes_to:message_channel:embeddings",),
        decision_ids=("decision:adr-001",),
        evidence=(_evidence("docs/adr/001-worker.md"),),
        counter_evidence=(_evidence("services/api/embeddings.py"),),
        derived_from=("parser", "graph_fusion"),
    )

    assert claim.claim_id == "claim:runtime:worker"
    assert claim.claim_kind == "runtime_assignment"
    assert claim.status == "confirmed"
    assert claim.entity_ids == ("container:worker",)
    assert claim.relationship_ids == ("container:api:publishes_to:message_channel:embeddings",)
    assert claim.decision_ids == ("decision:adr-001",)
    assert claim.evidence
    assert claim.counter_evidence
    assert claim.derived_from == ("parser", "graph_fusion")


def test_claim_supporting_and_counter_evidence_helpers_remain_explicit() -> None:
    claim = ArchitectureClaim(
        claim_id="claim:session:ownership",
        claim_kind="ownership",
        summary="API owns session issuance.",
        status="hypothesis",
        confidence=0.61,
        entity_ids=("container:api",),
        evidence=(
            _evidence("docs/adr/002-session-ownership.md"),
            _evidence("services/api/session_handler.py"),
        ),
        counter_evidence=(_evidence("services/worker/session_repair.py"),),
        derived_from=("graph_fusion",),
    )

    assert claim_supporting_evidence(claim) == claim.evidence
    assert claim_counter_evidence(claim) == claim.counter_evidence


def test_claim_canonical_payload_is_stable_and_json_serializable() -> None:
    claim = ArchitectureClaim(
        claim_id="claim:integration:github",
        claim_kind="external_integration",
        summary="API invokes GitHub OAuth.",
        status="confirmed",
        confidence=0.9,
        entity_ids=("container:api", "external_system:github-oauth"),
        evidence=(
            _evidence("docs/integrations/github-oauth.md"),
            _evidence("services/api/oauth.py"),
        ),
        counter_evidence=(),
        derived_from=("parser", "llm_adjudicated"),
    )

    payload = claim.canonical_payload()

    assert payload["claim_id"] == "claim:integration:github"
    assert payload["status"] == "confirmed"
    assert payload["derived_from"] == ["llm_adjudicated", "parser"]
    assert payload["evidence"][0]["ref"] == "docs/integrations/github-oauth.md"
    json.dumps(payload)


def test_claim_traceability_can_be_attached_to_arc42_document() -> None:
    collection_id = uuid4()
    scenario_id = uuid4()
    claim = ArchitectureClaim(
        claim_id="claim:runtime:worker",
        claim_kind="runtime_assignment",
        summary="Embeddings run in the worker runtime.",
        status="confirmed",
        confidence=0.93,
        entity_ids=("container:worker",),
        evidence=(_evidence("docs/adr/001-worker.md"),),
        derived_from=("parser",),
    )
    trace = Arc42ClaimTraceability(
        section_key="5_building_block_view",
        claim_ids=("claim:runtime:worker",),
        summary="Worker runtime claim is used in the building block view.",
    )

    document = Arc42Document(
        collection_id=collection_id,
        scenario_id=scenario_id,
        scenario_name="AS-IS",
        title="arc42 - AS-IS",
        generated_at=datetime.now(UTC),
        sections={"5_building_block_view": "Worker runtime"},
        markdown="# arc42 - AS-IS\n",
        claim_traceability=[trace],
        claims=[claim],
    )

    assert document.claim_traceability == [trace]
    assert document.claims == [claim]
    assert document.claim_ids_for_section("5_building_block_view") == ["claim:runtime:worker"]


def test_arc42_document_traceability_payload_is_serializable() -> None:
    collection_id = uuid4()
    scenario_id = uuid4()
    claim = ArchitectureClaim(
        claim_id="claim:queue:embeddings",
        claim_kind="message_flow",
        summary="API publishes embeddings jobs to a queue.",
        status="ambiguous",
        confidence=0.72,
        entity_ids=("container:api", "message_channel:embeddings"),
        evidence=(_evidence("docs/records/queueing.md"),),
        counter_evidence=(_evidence("services/api/embeddings.py"),),
        derived_from=("graph_fusion",),
    )
    trace = Arc42ClaimTraceability(
        section_key="8_concepts",
        claim_ids=("claim:queue:embeddings",),
        summary="Concept section references queue-based dispatch.",
    )
    document = Arc42Document(
        collection_id=collection_id,
        scenario_id=scenario_id,
        scenario_name="TO-BE",
        title="arc42 - TO-BE",
        generated_at=datetime.now(UTC),
        sections={"8_concepts": "Queue-based dispatch"},
        markdown="# arc42 - TO-BE\n",
        claim_traceability=[trace],
        claims=[claim],
    )

    payload = document.canonical_payload()

    assert payload["claims"][0]["claim_id"] == "claim:queue:embeddings"
    assert payload["claim_traceability"][0]["section_key"] == "8_concepts"
    json.dumps(payload)
