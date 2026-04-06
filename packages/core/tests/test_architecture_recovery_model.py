"""Tests for the evidence-backed recovered architecture model."""

from __future__ import annotations

import json

from contextmine_core.architecture.recovery_model import (
    RecoveredArchitectureDecision,
    RecoveredArchitectureEntity,
    RecoveredArchitectureHypothesis,
    RecoveredArchitectureMembership,
    RecoveredArchitectureModel,
    RecoveredArchitectureRelationship,
)
from contextmine_core.architecture.schemas import EvidenceRef


def _evidence(ref: str) -> tuple[EvidenceRef, ...]:
    return (EvidenceRef(kind="file", ref=ref),)


def test_entity_names_returns_sorted_names_and_filters_by_kind() -> None:
    model = RecoveredArchitectureModel(
        entities=(
            RecoveredArchitectureEntity(
                entity_id="component:session-manager",
                kind="component",
                name="Session Manager",
                confidence=0.88,
                evidence=_evidence("packages/core/session_manager.py"),
            ),
            RecoveredArchitectureEntity(
                entity_id="container:worker",
                kind="container",
                name="Worker Runtime",
                confidence=0.95,
                evidence=_evidence("services/worker/main.py"),
            ),
            RecoveredArchitectureEntity(
                entity_id="container:api",
                kind="container",
                name="API Runtime",
                confidence=0.97,
                evidence=_evidence("services/api/main.py"),
            ),
        )
    )

    assert model.entity_names() == ["API Runtime", "Session Manager", "Worker Runtime"]
    assert model.entity_names(kind="container") == ["API Runtime", "Worker Runtime"]
    assert model.entity_names(kind="component") == ["Session Manager"]


def test_memberships_for_returns_multiple_memberships_in_deterministic_order() -> None:
    membership_api = RecoveredArchitectureMembership(
        subject_ref="symbol:session_manager",
        entity_id="container:api",
        relationship_kind="implemented_by",
        confidence=0.94,
        evidence=_evidence("services/api/session_manager.py"),
    )
    membership_worker = RecoveredArchitectureMembership(
        subject_ref="symbol:session_manager",
        entity_id="container:worker",
        relationship_kind="implemented_by",
        confidence=0.91,
        evidence=_evidence("services/worker/session_manager.py"),
    )
    model = RecoveredArchitectureModel(memberships=(membership_worker, membership_api))

    assert model.memberships_for("symbol:session_manager") == [membership_api, membership_worker]


def test_relationship_tuples_returns_comparable_sorted_tuples() -> None:
    model = RecoveredArchitectureModel(
        relationships=(
            RecoveredArchitectureRelationship(
                source_entity_id="external:openai",
                target_entity_id="container:worker",
                kind="invokes",
                confidence=0.86,
                evidence=_evidence("services/worker/embeddings.py"),
            ),
            RecoveredArchitectureRelationship(
                source_entity_id="container:api",
                target_entity_id="data_store:postgres",
                kind="reads_writes",
                confidence=0.93,
                evidence=_evidence("services/api/repository.py"),
            ),
        )
    )

    assert model.relationship_tuples() == [
        ("container:api", "data_store:postgres", "reads_writes"),
        ("external:openai", "container:worker", "invokes"),
    ]
    assert model.relationship_tuples(kind="invokes") == [
        ("external:openai", "container:worker", "invokes")
    ]


def test_canonical_payload_is_stable_sorted_and_json_serializable() -> None:
    model = RecoveredArchitectureModel(
        entities=(
            RecoveredArchitectureEntity(
                entity_id="container:worker",
                kind="container",
                name="Worker Runtime",
                confidence=0.91,
                evidence=_evidence("services/worker/main.py"),
            ),
            RecoveredArchitectureEntity(
                entity_id="container:api",
                kind="container",
                name="API Runtime",
                confidence=0.96,
                evidence=_evidence("services/api/main.py"),
            ),
        ),
        relationships=(
            RecoveredArchitectureRelationship(
                source_entity_id="container:api",
                target_entity_id="container:worker",
                kind="publishes_to",
                confidence=0.84,
                evidence=_evidence("services/api/events.py"),
            ),
        ),
        memberships=(
            RecoveredArchitectureMembership(
                subject_ref="symbol:session_manager",
                entity_id="container:worker",
                relationship_kind="implemented_by",
                confidence=0.83,
                evidence=_evidence("services/worker/session_manager.py"),
            ),
        ),
    )

    payload = model.canonical_payload()

    assert [entity["entity_id"] for entity in payload["entities"]] == [
        "container:api",
        "container:worker",
    ]
    assert payload["relationships"][0]["kind"] == "publishes_to"
    assert payload["memberships"][0]["subject_ref"] == "symbol:session_manager"
    json.dumps(payload)


def test_hypothesis_stores_expected_fields() -> None:
    hypothesis = RecoveredArchitectureHypothesis(
        subject_ref="symbol:session_manager",
        candidate_entity_ids=("container:api", "container:worker"),
        selected_entity_ids=("container:api", "container:worker"),
        rationale="Shared session lifecycle code is referenced by both runtimes.",
        status="ambiguous",
        confidence=0.74,
        evidence=_evidence("packages/core/session_manager.py"),
    )

    assert hypothesis.subject_ref == "symbol:session_manager"
    assert hypothesis.candidate_entity_ids == ("container:api", "container:worker")
    assert hypothesis.selected_entity_ids == ("container:api", "container:worker")
    assert hypothesis.status == "ambiguous"
    assert hypothesis.confidence == 0.74
    assert hypothesis.evidence


def test_decision_stores_expected_fields() -> None:
    decision = RecoveredArchitectureDecision(
        title="Use async job worker for embeddings",
        summary="ADR documents background embedding generation in the worker runtime.",
        status="confirmed",
        affected_entity_ids=("container:worker", "external:openai"),
        confidence=0.92,
        evidence=_evidence("docs/adr/001-embeddings-worker.md"),
    )

    assert decision.title == "Use async job worker for embeddings"
    assert decision.summary.startswith("ADR documents")
    assert decision.status == "confirmed"
    assert decision.affected_entity_ids == ("container:worker", "external:openai")
    assert decision.confidence == 0.92
    assert decision.evidence


def test_every_object_carries_confidence_and_evidence() -> None:
    entity = RecoveredArchitectureEntity(
        entity_id="container:api",
        kind="container",
        name="API Runtime",
        confidence=0.97,
        evidence=_evidence("services/api/main.py"),
    )
    relationship = RecoveredArchitectureRelationship(
        source_entity_id="container:api",
        target_entity_id="data_store:postgres",
        kind="reads_writes",
        confidence=0.88,
        evidence=_evidence("services/api/repository.py"),
    )
    membership = RecoveredArchitectureMembership(
        subject_ref="symbol:session_manager",
        entity_id="container:api",
        relationship_kind="implemented_by",
        confidence=0.82,
        evidence=_evidence("services/api/session_manager.py"),
    )
    hypothesis = RecoveredArchitectureHypothesis(
        subject_ref="symbol:session_manager",
        candidate_entity_ids=("container:api",),
        selected_entity_ids=("container:api",),
        rationale="Explicit architecture metadata.",
        status="selected",
        confidence=0.9,
        evidence=_evidence("services/api/session_manager.py"),
    )
    decision = RecoveredArchitectureDecision(
        title="API owns session issuance",
        summary="ADR confirms API-issued sessions.",
        status="hypothesis",
        affected_entity_ids=("container:api",),
        confidence=0.67,
        evidence=_evidence("docs/adr/002-session-ownership.md"),
    )

    for row in (entity, relationship, membership, hypothesis, decision):
        assert row.confidence > 0
        assert row.evidence


def test_single_subject_can_have_two_memberships_simultaneously() -> None:
    model = RecoveredArchitectureModel(
        memberships=(
            RecoveredArchitectureMembership(
                subject_ref="symbol:session_manager",
                entity_id="container:api",
                relationship_kind="implemented_by",
                confidence=0.93,
                evidence=_evidence("services/api/session_manager.py"),
            ),
            RecoveredArchitectureMembership(
                subject_ref="symbol:session_manager",
                entity_id="container:worker",
                relationship_kind="implemented_by",
                confidence=0.9,
                evidence=_evidence("services/worker/session_manager.py"),
            ),
        )
    )

    memberships = model.memberships_for("symbol:session_manager")
    assert [membership.entity_id for membership in memberships] == [
        "container:api",
        "container:worker",
    ]
