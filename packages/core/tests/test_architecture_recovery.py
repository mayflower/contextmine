"""Outside-in tests for deterministic architecture recovery."""

from __future__ import annotations

from copy import deepcopy

from contextmine_core.architecture.recovery import recover_architecture_model

from .models.architecture_recovery_fixture import build_architecture_recovery_fixture


def test_recovery_identifies_api_worker_and_shared_entities() -> None:
    fixture = build_architecture_recovery_fixture()

    model = recover_architecture_model(
        fixture["nodes"],
        fixture["edges"],
        docs=fixture["docs"],
    )

    assert model.entity_names(kind="container") == ["API Runtime", "Worker Runtime"]
    assert "Session Manager" in model.entity_names(kind="component")
    assert "Shared Core" not in model.entity_names(kind="container")


def test_recovery_preserves_multi_membership_for_shared_session_manager() -> None:
    fixture = build_architecture_recovery_fixture()

    model = recover_architecture_model(
        fixture["nodes"],
        fixture["edges"],
        docs=fixture["docs"],
    )

    memberships = [
        membership.entity_id
        for membership in model.memberships_for("symbol:session_manager")
        if membership.relationship_kind == "contained_in"
    ]
    assert memberships == [
        "container:api",
        "container:worker",
    ]


def test_recovery_infers_real_relationships_for_db_topic_and_external_calls() -> None:
    fixture = build_architecture_recovery_fixture()

    model = recover_architecture_model(
        fixture["nodes"],
        fixture["edges"],
        docs=fixture["docs"],
    )

    assert ("container:api", "data_store:customer_sessions", "reads_writes") in (
        model.relationship_tuples()
    )
    assert ("container:api", "message_channel:user-events", "publishes_to") in (
        model.relationship_tuples()
    )
    assert ("container:api", "external_system:github-oauth", "invokes") in (
        model.relationship_tuples()
    )
    assert ("container:worker", "external_system:openai-embeddings", "invokes") in (
        model.relationship_tuples()
    )


def test_recovery_attaches_evidence_and_confidence_to_every_inferred_object() -> None:
    fixture = build_architecture_recovery_fixture()

    model = recover_architecture_model(
        fixture["nodes"],
        fixture["edges"],
        docs=fixture["docs"],
    )

    for row in (
        list(model.entities)
        + list(model.relationships)
        + list(model.memberships)
        + list(model.hypotheses)
    ):
        assert row.confidence > 0
        assert row.evidence


def test_recovery_emits_hypothesis_instead_of_random_single_assignment_when_ambiguous() -> None:
    fixture = build_architecture_recovery_fixture()

    model = recover_architecture_model(
        fixture["nodes"],
        fixture["edges"],
        docs=fixture["docs"],
    )

    hypothesis = next(
        row for row in model.hypotheses if row.subject_ref == "symbol:session_manager"
    )
    assert hypothesis.status == "ambiguous"
    assert hypothesis.selected_entity_ids == ("container:api", "container:worker")
    assert hypothesis.candidate_entity_ids == ("container:api", "container:worker")


def _build_scoring_fixture() -> dict[str, object]:
    fixture = deepcopy(build_architecture_recovery_fixture())
    nodes = fixture["nodes"]
    edges = fixture["edges"]

    for node in nodes:
        if node["id"] == "symbol:event_publisher":
            node["meta"] = {
                "file_path": "services/contextmine/api/events.py",
            }
            break

    nodes.extend(
        [
            {
                "id": "symbol:metadata_wins",
                "kind": "symbol",
                "name": "Metadata Wins",
                "natural_key": "symbol:metadata_wins",
                "meta": {
                    "file_path": "services/contextmine/worker/metadata_wins.py",
                    "architecture": {
                        "domain": "contextmine",
                        "container": "api",
                        "component": "metadata-wins",
                    },
                },
            },
            {
                "id": "symbol:near_tie_helper",
                "kind": "symbol",
                "name": "Near Tie Helper",
                "natural_key": "symbol:near_tie_helper",
                "meta": {
                    "file_path": "packages/core/near_tie_helper.py",
                },
            },
            {
                "id": "symbol:orphan_helper",
                "kind": "symbol",
                "name": "Orphan Helper",
                "natural_key": "symbol:orphan_helper",
                "meta": {
                    "file_path": "packages/core/orphan_helper.py",
                },
            },
            {
                "id": "symbol:path_hint_only",
                "kind": "symbol",
                "name": "Path Hint Only",
                "natural_key": "symbol:path_hint_only",
                "meta": {
                    "file_path": "services/contextmine/api/path_hint_only.py",
                },
            },
        ]
    )
    edges.extend(
        [
            {
                "source_node_id": "symbol:api_session_handler",
                "target_node_id": "symbol:near_tie_helper",
                "kind": "symbol_calls_symbol",
                "meta": {},
            },
            {
                "source_node_id": "symbol:event_publisher",
                "target_node_id": "symbol:near_tie_helper",
                "kind": "symbol_calls_symbol",
                "meta": {},
            },
            {
                "source_node_id": "symbol:embedding_job_runner",
                "target_node_id": "symbol:near_tie_helper",
                "kind": "symbol_calls_symbol",
                "meta": {},
            },
        ]
    )
    return fixture


def test_explicit_architecture_metadata_outranks_path_heuristics_but_keeps_counter_evidence() -> (
    None
):
    fixture = _build_scoring_fixture()

    model = recover_architecture_model(
        fixture["nodes"],
        fixture["edges"],
        docs=fixture["docs"],
    )

    metadata_hypothesis = next(
        row for row in model.hypotheses if row.subject_ref == "symbol:metadata_wins"
    )

    assert metadata_hypothesis.status == "selected"
    assert metadata_hypothesis.selected_entity_ids == ("container:api",)
    assert metadata_hypothesis.candidate_entity_ids == ("container:api", "container:worker")


def test_if_two_memberships_are_close_the_weaker_one_is_retained() -> None:
    fixture = _build_scoring_fixture()

    model = recover_architecture_model(
        fixture["nodes"],
        fixture["edges"],
        docs=fixture["docs"],
    )

    near_tie_hypothesis = next(
        row for row in model.hypotheses if row.subject_ref == "symbol:near_tie_helper"
    )
    near_tie_memberships = model.memberships_for("symbol:near_tie_helper")

    assert near_tie_hypothesis.status == "ambiguous"
    assert near_tie_hypothesis.selected_entity_ids == ("container:api", "container:worker")
    assert [membership.entity_id for membership in near_tie_memberships] == [
        "container:api",
        "container:worker",
    ]


def test_if_no_candidate_clears_threshold_emit_unresolved_hypothesis() -> None:
    fixture = _build_scoring_fixture()

    model = recover_architecture_model(
        fixture["nodes"],
        fixture["edges"],
        docs=fixture["docs"],
    )

    unresolved = next(row for row in model.hypotheses if row.subject_ref == "symbol:orphan_helper")

    assert unresolved.status == "unresolved"
    assert unresolved.selected_entity_ids == ()
    assert model.memberships_for("symbol:orphan_helper") == []
    assert "Orphan Helper" not in model.entity_names(kind="component")


def test_hypotheses_cite_same_evidence_refs_used_for_competing_candidates() -> None:
    fixture = _build_scoring_fixture()

    model = recover_architecture_model(
        fixture["nodes"],
        fixture["edges"],
        docs=fixture["docs"],
    )

    memberships = model.memberships_for("symbol:near_tie_helper")
    membership_evidence_refs = {
        evidence.ref for membership in memberships for evidence in membership.evidence
    }
    hypothesis = next(
        row for row in model.hypotheses if row.subject_ref == "symbol:near_tie_helper"
    )

    assert membership_evidence_refs.issubset({evidence.ref for evidence in hypothesis.evidence})


def test_confidence_is_monotonic_explicit_metadata_gt_structural_gt_path_only() -> None:
    fixture = _build_scoring_fixture()

    model = recover_architecture_model(
        fixture["nodes"],
        fixture["edges"],
        docs=fixture["docs"],
    )

    explicit = next(
        membership
        for membership in model.memberships
        if membership.subject_ref == "symbol:metadata_wins"
        and membership.entity_id == "container:api"
    )
    structural = next(
        membership
        for membership in model.memberships
        if membership.subject_ref == "symbol:near_tie_helper"
        and membership.entity_id == "container:api"
    )
    path_only = next(
        hypothesis
        for hypothesis in model.hypotheses
        if hypothesis.subject_ref == "symbol:path_hint_only"
    )

    assert path_only.status in {"ambiguous", "unresolved"}
    assert explicit.confidence > structural.confidence > path_only.confidence
