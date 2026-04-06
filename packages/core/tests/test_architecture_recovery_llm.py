"""Tests for constrained LLM adjudication over recovered architecture facts."""

from __future__ import annotations

from contextmine_core.architecture.recovery import recover_architecture_model
from contextmine_core.architecture.recovery_llm import (
    apply_adjudication,
    build_adjudication_packet,
)

from .models.architecture_recovery_fixture import build_architecture_recovery_fixture


def _deterministic_model():
    fixture = build_architecture_recovery_fixture()
    return recover_architecture_model(
        fixture["nodes"],
        fixture["edges"],
        docs=fixture["docs"],
    )


def _session_manager_hypothesis(model):
    return next(row for row in model.hypotheses if row.subject_ref == "symbol:session_manager")


def test_build_adjudication_packet_includes_only_local_evidence_candidates_and_snippets() -> None:
    model = _deterministic_model()
    hypothesis = _session_manager_hypothesis(model)

    packet = build_adjudication_packet(model=model, hypothesis=hypothesis)

    assert packet["subject_ref"] == "symbol:session_manager"
    assert packet["candidate_entity_ids"] == ["container:api", "container:worker"]
    assert packet["selected_entity_ids"] == ["container:api", "container:worker"]
    assert packet["evidence"]
    assert all(
        sorted(item) == ["evidence_id", "kind", "ref", "snippet"] for item in packet["evidence"]
    )
    assert all(item["evidence_id"].startswith("ev-") for item in packet["evidence"])


def test_apply_adjudication_rejects_outputs_referencing_unknown_evidence_ids() -> None:
    model = _deterministic_model()
    hypothesis = _session_manager_hypothesis(model)
    packet = build_adjudication_packet(model=model, hypothesis=hypothesis)

    updated = apply_adjudication(
        model=model,
        hypothesis=hypothesis,
        packet=packet,
        adjudication={
            "selected_entity_ids": ["container:api"],
            "rationale": "Prefer the API runtime.",
            "evidence_ids": ["ev-999"],
        },
    )

    preserved = _session_manager_hypothesis(updated)
    assert preserved.selected_entity_ids == ("container:api", "container:worker")
    assert any("unknown evidence" in warning.lower() for warning in updated.warnings)


def test_llm_adjudication_can_select_between_candidates_but_cannot_invent_new_entities() -> None:
    model = _deterministic_model()
    hypothesis = _session_manager_hypothesis(model)
    packet = build_adjudication_packet(model=model, hypothesis=hypothesis)
    evidence_id = packet["evidence"][0]["evidence_id"]

    updated = apply_adjudication(
        model=model,
        hypothesis=hypothesis,
        packet=packet,
        adjudication={
            "selected_entity_ids": ["container:api"],
            "rationale": "Runtime entrypoint evidence favors the API path.",
            "evidence_ids": [evidence_id],
        },
    )

    selected = _session_manager_hypothesis(updated)
    assert selected.status == "selected"
    assert selected.selected_entity_ids == ("container:api",)
    assert [
        membership.entity_id for membership in updated.memberships_for("symbol:session_manager")
    ] == ["container:api"]

    rejected = apply_adjudication(
        model=model,
        hypothesis=hypothesis,
        packet=packet,
        adjudication={
            "selected_entity_ids": ["container:admin"],
            "rationale": "Invent a new runtime.",
            "evidence_ids": [evidence_id],
        },
    )

    preserved = _session_manager_hypothesis(rejected)
    assert preserved.selected_entity_ids == ("container:api", "container:worker")
    assert any("candidate" in warning.lower() for warning in rejected.warnings)


def test_malformed_adjudication_preserves_deterministic_result_and_adds_warning() -> None:
    fixture = build_architecture_recovery_fixture()

    class MalformedAdjudicator:
        def adjudicate(self, packet):  # noqa: ANN001
            return "not-a-dict"

    model = recover_architecture_model(
        fixture["nodes"],
        fixture["edges"],
        docs=fixture["docs"],
        llm_adjudicator=MalformedAdjudicator(),
    )

    hypothesis = _session_manager_hypothesis(model)
    assert hypothesis.selected_entity_ids == ("container:api", "container:worker")
    assert any("malformed" in warning.lower() for warning in model.warnings)


def test_adjudication_enriches_rationale_without_erasing_evidence_or_confidence() -> None:
    model = _deterministic_model()
    hypothesis = _session_manager_hypothesis(model)
    packet = build_adjudication_packet(model=model, hypothesis=hypothesis)
    evidence_id = packet["evidence"][0]["evidence_id"]

    updated = apply_adjudication(
        model=model,
        hypothesis=hypothesis,
        packet=packet,
        adjudication={
            "selected_entity_ids": ["container:api"],
            "rationale": "API entrypoint and handler evidence outweigh the worker path.",
            "evidence_ids": [evidence_id],
        },
    )

    adjudicated = _session_manager_hypothesis(updated)
    assert "Multiple runtime candidates remain plausible" in adjudicated.rationale
    assert "API entrypoint and handler evidence outweigh the worker path." in adjudicated.rationale
    assert adjudicated.confidence == hypothesis.confidence
    assert adjudicated.evidence == hypothesis.evidence
