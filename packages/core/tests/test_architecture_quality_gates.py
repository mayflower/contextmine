"""Hard quality gates for architecture recovery and export behavior."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest
from contextmine_core.architecture.arc42 import render_claim_backed_arc42
from contextmine_core.architecture.claim_model import Arc42ClaimTraceability, ArchitectureClaim
from contextmine_core.architecture.facts import build_architecture_facts
from contextmine_core.architecture.recovery import recover_architecture_model
from contextmine_core.architecture.recovery_decisions import recover_architecture_decisions
from contextmine_core.architecture.recovery_docs import load_recovery_docs
from contextmine_core.architecture.recovery_llm import apply_adjudication, build_adjudication_packet
from contextmine_core.architecture.recovery_model import (
    RecoveredArchitectureEntity,
    RecoveredArchitectureHypothesis,
    RecoveredArchitectureMembership,
    RecoveredArchitectureModel,
)
from contextmine_core.architecture.schemas import (
    ArchitectureFact,
    ArchitectureFactsBundle,
    EvidenceRef,
)
from contextmine_core.exports import mermaid_c4 as mermaid_export
from contextmine_core.models import (
    KnowledgeEdge,
    KnowledgeEdgeKind,
    KnowledgeNode,
    KnowledgeNodeKind,
)

from .models.architecture_recovery_fixture import build_architecture_recovery_fixture


class _ScalarResult:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value

    def scalar_one(self):
        return self._value

    def scalars(self):
        return self

    def all(self):
        return self._value


class _FakeSession:
    def __init__(self, scenario=None, metrics=None, nodes=None, edges=None, documents=None):
        self.scenario = scenario
        self.metrics = metrics or []
        self.nodes = nodes or []
        self.edges = edges or []
        self.documents = documents or []

    async def execute(self, stmt):  # noqa: ANN001
        statement = str(stmt)
        if "FROM twin_scenarios" in statement:
            return _ScalarResult(self.scenario)
        if "FROM metric_snapshots" in statement:
            return _ScalarResult(self.metrics)
        if "FROM knowledge_nodes" in statement:
            return _ScalarResult(self.nodes)
        if "FROM knowledge_edges" in statement:
            return _ScalarResult(self.edges)
        if "FROM documents" in statement:
            return _ScalarResult(self.documents)
        if "knowledge_node_evidence" in statement:
            return _ScalarResult([])
        return _ScalarResult([])


def _evidence(ref: str, *, kind: str = "file") -> tuple[EvidenceRef, ...]:
    return (EvidenceRef(kind=kind, ref=ref),)


def _arc42_bundle() -> ArchitectureFactsBundle:
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
                evidence=_evidence("docs/adr/001-async-embedding-workers.md"),
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
                evidence=_evidence("packages/core/session_manager.py"),
            ),
        ],
        warnings=["Ambiguous recovered memberships: 1."],
    )


def _recovered_model() -> RecoveredArchitectureModel:
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
                confidence=0.9,
                evidence=_evidence("services/contextmine/worker/jobs.py"),
            ),
            RecoveredArchitectureEntity(
                entity_id="component:session-manager",
                kind="component",
                name="Session Manager",
                confidence=0.88,
                evidence=_evidence("packages/core/session_manager.py"),
            ),
        ),
        memberships=(
            RecoveredArchitectureMembership(
                subject_ref="endpoint:post:/sessions",
                entity_id="container:api",
                relationship_kind="contained_in",
                confidence=0.96,
                evidence=_evidence("services/contextmine/api/routes.py"),
            ),
            RecoveredArchitectureMembership(
                subject_ref="symbol:session_manager",
                entity_id="container:api",
                relationship_kind="contained_in",
                confidence=0.93,
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
                confidence=0.91,
                evidence=_evidence("packages/core/session_manager.py"),
            ),
        ),
        hypotheses=(
            RecoveredArchitectureHypothesis(
                subject_ref="symbol:session_manager",
                candidate_entity_ids=("container:api", "container:worker"),
                selected_entity_ids=("container:api", "container:worker"),
                rationale="Shared session manager code remains multi-homed.",
                status="ambiguous",
                confidence=0.88,
                evidence=_evidence("packages/core/session_manager.py"),
            ),
        ),
    )


def test_no_confirmed_architecture_claim_without_evidence() -> None:
    document = render_claim_backed_arc42(
        _arc42_bundle(),
        SimpleNamespace(name="AS-IS"),
        claims=[
            ArchitectureClaim(
                claim_id="claim:unsupported",
                claim_kind="runtime_assignment",
                summary="This should not survive without evidence.",
                status="confirmed",
                confidence=0.95,
                entity_ids=("container:worker",),
                evidence=(),
                derived_from=("parser",),
            )
        ],
        claim_traceability=[
            Arc42ClaimTraceability(
                section_key="5_building_block_view",
                claim_ids=("claim:unsupported",),
                summary="Unsupported claim",
            )
        ],
    )

    assert not document.claims
    assert any("claim:unsupported" in warning for warning in document.warnings)


def test_no_confirmed_container_only_from_path_heuristic() -> None:
    model = recover_architecture_model(
        [
            {
                "id": "symbol:event_publisher",
                "kind": "symbol",
                "name": "Event Publisher",
                "natural_key": "symbol:event_publisher",
                "meta": {"file_path": "services/contextmine/api/events.py"},
            }
        ],
        [],
        docs=[],
    )

    memberships = model.memberships_for("symbol:event_publisher")
    assert not memberships
    hypothesis = next(
        row for row in model.hypotheses if row.subject_ref == "symbol:event_publisher"
    )
    assert hypothesis.status in {"ambiguous", "unresolved"}


def test_no_decision_without_documentable_source() -> None:
    entities = (
        RecoveredArchitectureEntity(
            entity_id="container:worker",
            kind="container",
            name="Worker Runtime",
            confidence=0.96,
            evidence=_evidence("services/contextmine/worker/jobs.py"),
            attributes={"container": "worker"},
        ),
    )
    decisions = recover_architecture_decisions(
        [
            {
                "title": "Worker runtime",
                "text": "Use the worker runtime.",
                "structured_data": {
                    "decision": "Use the worker runtime.",
                    "affected_entity_ids": ["container:worker"],
                },
            }
        ],
        entities,
    )

    assert decisions == ()


def test_no_llm_adjudication_with_unknown_evidence_ids() -> None:
    fixture = build_architecture_recovery_fixture()
    model = recover_architecture_model(fixture["nodes"], fixture["edges"], docs=fixture["docs"])
    hypothesis = next(
        row for row in model.hypotheses if row.subject_ref == "symbol:session_manager"
    )
    packet = build_adjudication_packet(model=model, hypothesis=hypothesis)

    updated = apply_adjudication(
        model=model,
        hypothesis=hypothesis,
        packet=packet,
        adjudication={
            "selected_entity_ids": ["container:api"],
            "rationale": "Prefer API runtime.",
            "evidence_ids": ["ev-does-not-exist"],
        },
    )

    assert any("unknown evidence" in warning.lower() for warning in updated.warnings)


@pytest.mark.anyio
async def test_no_silent_reduction_of_multi_membership_to_single_membership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection_id = uuid4()
    scenario_id = uuid4()
    endpoint = KnowledgeNode(
        id=uuid4(),
        collection_id=collection_id,
        kind=KnowledgeNodeKind.API_ENDPOINT,
        natural_key="endpoint:post:/sessions",
        name="POST /sessions",
        meta={"method": "POST", "path": "/sessions"},
    )
    symbol = KnowledgeNode(
        id=uuid4(),
        collection_id=collection_id,
        kind=KnowledgeNodeKind.SYMBOL,
        natural_key="symbol:session_manager",
        name="SessionManager",
        meta={"file_path": "packages/core/session_manager.py"},
    )
    table = KnowledgeNode(
        id=uuid4(),
        collection_id=collection_id,
        kind=KnowledgeNodeKind.DB_TABLE,
        natural_key="db:sessions",
        name="sessions",
        meta={},
    )
    edge = KnowledgeEdge(
        id=uuid4(),
        collection_id=collection_id,
        source_node_id=symbol.id,
        target_node_id=table.id,
        kind=KnowledgeEdgeKind.SYMBOL_REFERENCES_SYMBOL,
    )
    scenario = SimpleNamespace(id=scenario_id, collection_id=collection_id, name="AS-IS")
    session = _FakeSession(scenario=scenario, nodes=[endpoint, symbol, table], edges=[edge])

    async def _fake_c4(*_args, **_kwargs):
        return SimpleNamespace(content="C4Context", warnings=[])

    async def _fake_node_ids(*_args, **_kwargs):
        return {endpoint.id, symbol.id, table.id}

    async def _fake_evidence(*_args, **_kwargs):
        return {endpoint.id: _evidence("services/contextmine/api/routes.py")}

    monkeypatch.setattr(mermaid_export, "export_mermaid_c4_result", _fake_c4)
    monkeypatch.setattr("contextmine_core.architecture.facts.export_mermaid_c4_result", _fake_c4)
    monkeypatch.setattr(
        "contextmine_core.architecture.facts.get_scenario_provenance_node_ids", _fake_node_ids
    )
    monkeypatch.setattr("contextmine_core.architecture.facts._load_node_evidence", _fake_evidence)
    monkeypatch.setattr(
        "contextmine_core.architecture.facts.recover_architecture_model",
        lambda *_args, **_kwargs: _recovered_model(),
    )

    bundle = await build_architecture_facts(
        session,
        collection_id=collection_id,
        scenario_id=scenario_id,
    )

    outbound = next(fact for fact in bundle.ports_adapters if fact.direction == "outbound")
    assert outbound.attributes["candidate_container_ids"] == ["container:api", "container:worker"]
    assert outbound.attributes["legacy_mapping_lossy"] is True


def test_no_arc42_claim_without_claim_and_evidence_anchoring() -> None:
    document = render_claim_backed_arc42(
        _arc42_bundle(),
        SimpleNamespace(name="AS-IS"),
        claims=[
            ArchitectureClaim(
                claim_id="claim:unanchored",
                claim_kind="runtime_assignment",
                summary="Unanchored claim.",
                status="confirmed",
                confidence=0.92,
                entity_ids=("container:worker",),
                evidence=_evidence("docs/adr/001-async-embedding-workers.md"),
                derived_from=("parser",),
            )
        ],
        claim_traceability=[],
    )

    assert not document.claims
    assert "claim:unanchored" not in document.markdown


def test_missing_or_conflicting_artifacts_surface_unknown_not_smooth_prose() -> None:
    document = render_claim_backed_arc42(
        _arc42_bundle(),
        SimpleNamespace(name="AS-IS"),
        claims=[
            ArchitectureClaim(
                claim_id="claim:session:shared",
                claim_kind="ownership",
                summary="Session manager code is shared between API and worker.",
                status="ambiguous",
                confidence=0.72,
                entity_ids=("container:api", "container:worker"),
                evidence=_evidence("packages/core/session_manager.py"),
                derived_from=("graph_fusion",),
            )
        ],
        claim_traceability=[
            Arc42ClaimTraceability(
                section_key="5_building_block_view",
                claim_ids=("claim:session:shared",),
                summary="Shared ownership remains unresolved.",
            )
        ],
    )

    assert "UNKNOWN: ambiguous claim" in document.sections["5_building_block_view"]


@pytest.mark.anyio
async def test_repo_wide_docs_count_even_without_document_rows() -> None:
    file_node = KnowledgeNode(
        id=uuid4(),
        collection_id=uuid4(),
        kind=KnowledgeNodeKind.FILE,
        natural_key="docs/records/async-worker-rollout.md",
        name="Async worker rollout",
        meta={
            "uri": "docs/records/async-worker-rollout.md",
            "file_path": "docs/records/async-worker-rollout.md",
            "content_markdown": (
                "---\n"
                "affected_entity_ids:\n"
                "  - container:api\n"
                "  - container:worker\n"
                "---\n"
                "# Async worker rollout\n\n"
                "## Decision\nEmbeddings execute in the worker runtime.\n"
            ),
        },
    )

    docs = await load_recovery_docs(_FakeSession(documents=[]), [file_node])
    assert docs


@pytest.mark.anyio
async def test_fallback_modes_warn_explicitly_and_avoid_generic_best_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario_id = uuid4()
    scenario = SimpleNamespace(id=scenario_id, collection_id=uuid4(), name="AS-IS")

    async def _fake_recovered(*_args, **_kwargs):
        return mermaid_export.RecoveredArchitectureModel()

    async def _fake_graph(**_kwargs):  # noqa: ANN003
        return {
            "nodes": [
                {
                    "id": "largest-component",
                    "natural_key": "component|contextmine|api|session-manager",
                    "kind": "component",
                    "name": "Session Manager",
                    "meta": {
                        "component": "session-manager",
                        "container": "api",
                        "member_count": 12,
                    },
                }
            ],
            "edges": [],
            "total_nodes": 1,
            "projection": "architecture",
            "entity_level": "component",
            "grouping_strategy": "heuristic",
            "excluded_kinds": [],
        }

    monkeypatch.setattr(mermaid_export, "_load_recovered_architecture_model", _fake_recovered)
    monkeypatch.setattr(mermaid_export, "get_full_scenario_graph", _fake_graph)

    result = await mermaid_export.export_mermaid_c4_result(
        _FakeSession(scenario=scenario),
        scenario_id,
        c4_view="component",
    )

    assert any("degraded" in warning.lower() for warning in result.warnings)
    assert not any(
        "best effort due sparse source signals" in warning.lower() for warning in result.warnings
    )
