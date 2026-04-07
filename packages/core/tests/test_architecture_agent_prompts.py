"""Tests for agent-facing recovered architecture prompt helpers."""

from __future__ import annotations

from contextmine_core.architecture.agent_sdk import (
    _arc42_prompt,
    _prompt_claim_sections,
    _recovered_architecture_payload,
)
from contextmine_core.architecture.claim_model import ArchitectureClaim
from contextmine_core.architecture.recovery_model import (
    RecoveredArchitectureDecision,
    RecoveredArchitectureEntity,
    RecoveredArchitectureHypothesis,
    RecoveredArchitectureModel,
    RecoveredArchitectureRelationship,
)
from contextmine_core.architecture.schemas import EvidenceRef


def _evidence(ref: str) -> tuple[EvidenceRef, ...]:
    return (EvidenceRef(kind="file", ref=ref),)


def _model() -> RecoveredArchitectureModel:
    return RecoveredArchitectureModel(
        entities=(
            RecoveredArchitectureEntity(
                entity_id="container:api",
                kind="container",
                name="API Runtime",
                confidence=0.96,
                evidence=_evidence("services/contextmine/api/routes.py"),
            ),
        ),
        relationships=(
            RecoveredArchitectureRelationship(
                source_entity_id="container:api",
                target_entity_id="external_system:github-oauth",
                kind="invokes",
                confidence=0.9,
                evidence=_evidence("docs/integrations/github-oauth.md"),
            ),
        ),
        hypotheses=(
            RecoveredArchitectureHypothesis(
                subject_ref="symbol:session_manager",
                candidate_entity_ids=("container:api", "container:worker"),
                selected_entity_ids=("container:api", "container:worker"),
                rationale="Shared code spans API and worker runtimes.",
                status="ambiguous",
                confidence=0.88,
                evidence=_evidence("packages/core/session_manager.py"),
            ),
        ),
        decisions=(
            RecoveredArchitectureDecision(
                title="ADR-001 async embedding workers",
                summary="Embeddings generation runs in the worker runtime.",
                status="confirmed",
                affected_entity_ids=("container:api", "container:worker"),
                confidence=0.92,
                evidence=_evidence("docs/adr/001-async-embedding-workers.md"),
            ),
        ),
    )


def test_recovered_architecture_payload_accepts_model_instances() -> None:
    payload = _recovered_architecture_payload(_model())

    assert payload is not None
    assert payload["entities"][0]["entity_id"] == "container:api"
    assert payload["decisions"][0]["title"] == "ADR-001 async embedding workers"


def test_arc42_prompt_embeds_recovered_architecture_contract() -> None:
    prompt = _arc42_prompt(
        scenario_name="AS-IS",
        section=None,
        recovered_architecture=_model(),
    )

    assert "Recovered architecture payload is provided below" in prompt
    assert "entities, relationships, hypotheses, and decisions" in prompt
    assert "Do not invent facts" in prompt
    assert "ambiguous or unresolved" in prompt
    assert "ADR-001 async embedding workers" in prompt
    assert "symbol:session_manager" in prompt


def test_prompt_claim_sections_extracts_claims_open_questions_and_evidence_hints() -> None:
    claims = [
        ArchitectureClaim(
            claim_id="claim:runtime:worker",
            claim_kind="runtime_assignment",
            summary="Embeddings run in the worker runtime.",
            status="confirmed",
            confidence=0.93,
            entity_ids=("container:worker",),
            evidence=(EvidenceRef(kind="file", ref="docs/adr/001-async-embedding-workers.md"),),
            counter_evidence=(),
            derived_from=("parser",),
        ),
        ArchitectureClaim(
            claim_id="claim:session:shared",
            claim_kind="ownership",
            summary="Session manager code is shared between API and worker.",
            status="ambiguous",
            confidence=0.72,
            entity_ids=("container:api", "container:worker"),
            evidence=(EvidenceRef(kind="file", ref="packages/core/session_manager.py"),),
            counter_evidence=(),
            derived_from=("graph_fusion",),
        ),
    ]

    sections = _prompt_claim_sections(claims=claims, recovered_architecture=_model())

    assert "Structured claims" in sections
    assert "Open questions" in sections
    assert "Evidence hints" in sections
    assert "claim:runtime:worker" in sections
    assert "claim:session:shared" in sections
    assert "symbol:session_manager" in sections
