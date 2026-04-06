"""Tests for inferred architecture projection helpers."""

from __future__ import annotations

from contextmine_core.architecture.recovery_model import (
    RecoveredArchitectureEntity,
    RecoveredArchitectureHypothesis,
    RecoveredArchitectureMembership,
    RecoveredArchitectureModel,
    RecoveredArchitectureRelationship,
)
from contextmine_core.architecture.schemas import EvidenceRef
from contextmine_core.twin.projections import build_inferred_architecture_projection


def _evidence(ref: str) -> tuple[EvidenceRef, ...]:
    return (EvidenceRef(kind="file", ref=ref),)


def _projection_model() -> RecoveredArchitectureModel:
    return RecoveredArchitectureModel(
        entities=(
            RecoveredArchitectureEntity(
                entity_id="container:api",
                kind="container",
                name="API Runtime",
                confidence=0.96,
                evidence=_evidence("services/contextmine/api/routes.py"),
            ),
            RecoveredArchitectureEntity(
                entity_id="container:worker",
                kind="container",
                name="Worker Runtime",
                confidence=0.91,
                evidence=_evidence("services/contextmine/worker/jobs.py"),
            ),
            RecoveredArchitectureEntity(
                entity_id="component:session-manager",
                kind="component",
                name="Session Manager",
                confidence=0.87,
                evidence=_evidence("packages/core/session_manager.py"),
            ),
            RecoveredArchitectureEntity(
                entity_id="data_store:sessions",
                kind="data_store",
                name="sessions",
                confidence=0.93,
                evidence=_evidence("db/schema/sessions.sql"),
            ),
        ),
        relationships=(
            RecoveredArchitectureRelationship(
                source_entity_id="component:session-manager",
                target_entity_id="data_store:sessions",
                kind="reads_writes",
                confidence=0.89,
                evidence=_evidence("packages/core/session_manager.py"),
            ),
        ),
        memberships=(
            RecoveredArchitectureMembership(
                subject_ref="symbol:session_manager",
                entity_id="container:api",
                relationship_kind="contained_in",
                confidence=0.92,
                evidence=_evidence("services/contextmine/api/routes.py"),
            ),
            RecoveredArchitectureMembership(
                subject_ref="symbol:session_manager",
                entity_id="container:worker",
                relationship_kind="contained_in",
                confidence=0.88,
                evidence=_evidence("services/contextmine/worker/jobs.py"),
            ),
            RecoveredArchitectureMembership(
                subject_ref="symbol:session_manager",
                entity_id="component:session-manager",
                relationship_kind="implements",
                confidence=0.9,
                evidence=_evidence("packages/core/session_manager.py"),
            ),
        ),
        hypotheses=(
            RecoveredArchitectureHypothesis(
                subject_ref="symbol:session_manager",
                candidate_entity_ids=("container:api", "container:worker"),
                selected_entity_ids=("container:api", "container:worker"),
                rationale="Shared component belongs to both runtimes.",
                status="ambiguous",
                confidence=0.88,
                evidence=_evidence("packages/core/session_manager.py"),
            ),
            RecoveredArchitectureHypothesis(
                subject_ref="symbol:orphan",
                candidate_entity_ids=(),
                selected_entity_ids=(),
                rationale="No runtime evidence.",
                status="unresolved",
                confidence=0.0,
                evidence=_evidence("packages/core/orphan.py"),
            ),
        ),
    )


def test_recovered_entities_and_relationships_project_without_collapsing_memberships() -> None:
    projection = build_inferred_architecture_projection(
        _projection_model(), entity_level="component"
    )

    node_ids = {node["id"] for node in projection["nodes"]}
    assert "component:session-manager@container:api" in node_ids
    assert "component:session-manager@container:worker" in node_ids


def test_relationship_kinds_are_preserved() -> None:
    projection = build_inferred_architecture_projection(
        _projection_model(), entity_level="component"
    )

    assert {edge["kind"] for edge in projection["edges"]} == {"reads_writes"}


def test_projected_nodes_preserve_evidence_summaries_and_confidence() -> None:
    projection = build_inferred_architecture_projection(
        _projection_model(), entity_level="component"
    )

    api_node = next(
        node
        for node in projection["nodes"]
        if node["id"] == "component:session-manager@container:api"
    )
    assert api_node["meta"]["confidence"] == 0.9
    assert api_node["meta"]["evidence_summary"] == ["packages/core/session_manager.py"]


def test_shared_components_appear_in_more_than_one_runtime_context() -> None:
    projection = build_inferred_architecture_projection(
        _projection_model(), entity_level="component"
    )

    container_contexts = {
        node["meta"]["container_context"]
        for node in projection["nodes"]
        if node["id"].startswith("component:session-manager@")
    }
    assert container_contexts == {"API Runtime", "Worker Runtime"}


def test_projection_exposes_ambiguity_and_unresolved_counts_for_ui_consumers() -> None:
    projection = build_inferred_architecture_projection(
        _projection_model(), entity_level="component"
    )

    assert projection["summary"]["ambiguous_hypotheses"] == 1
    assert projection["summary"]["unresolved_hypotheses"] == 1
