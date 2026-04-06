"""Tests for integrating recovered architecture into build_architecture_facts."""

from __future__ import annotations

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from contextmine_core.architecture import facts as architecture_facts
from contextmine_core.architecture.recovery_model import (
    RecoveredArchitectureEntity,
    RecoveredArchitectureHypothesis,
    RecoveredArchitectureMembership,
    RecoveredArchitectureModel,
    RecoveredArchitectureRelationship,
)
from contextmine_core.architecture.schemas import EvidenceRef
from contextmine_core.models import (
    KnowledgeEdge,
    KnowledgeEdgeKind,
    KnowledgeNode,
    KnowledgeNodeKind,
)


class _ScalarResult:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value

    def scalars(self):
        return self

    def all(self):
        return self._value


class _FakeSession:
    def __init__(self, scenario, metrics, nodes, edges, documents=None):
        self.scenario = scenario
        self.metrics = metrics
        self.nodes = nodes
        self.edges = edges
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
        raise AssertionError(f"Unexpected statement: {statement}")


def _evidence(ref: str) -> tuple[EvidenceRef, ...]:
    return (EvidenceRef(kind="file", ref=ref),)


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
        relationships=(
            RecoveredArchitectureRelationship(
                source_entity_id="container:api",
                target_entity_id="data_store:sessions",
                kind="reads_writes",
                confidence=0.91,
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
            RecoveredArchitectureHypothesis(
                subject_ref="symbol:orphan",
                candidate_entity_ids=(),
                selected_entity_ids=(),
                rationale="No candidate cleared the threshold.",
                status="unresolved",
                confidence=0.0,
                evidence=_evidence("packages/core/orphan.py"),
            ),
        ),
        warnings=(
            "Rejected adjudication for symbol:session_manager: unknown evidence IDs referenced.",
            "Missing evidence packet for symbol:orphan.",
        ),
    )


def _knowledge_fixture(collection_id):
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
    return endpoint, symbol, table, edge


def _install_common_patches(
    monkeypatch: pytest.MonkeyPatch,
    collection_id,
    endpoint,
    symbol,
    table,
):
    async def _fake_c4(*_args, **_kwargs):
        return SimpleNamespace(content="C4Context", warnings=[])

    async def _fake_projection(**_kwargs):
        return {"nodes": [], "edges": [], "grouping_strategy": "heuristic"}

    async def _fake_node_ids(*_args, **_kwargs):
        return {endpoint.id, symbol.id, table.id}

    async def _fake_evidence(*_args, **_kwargs):
        return {endpoint.id: _evidence("services/contextmine/api/routes.py")}

    monkeypatch.setattr(architecture_facts, "export_mermaid_c4_result", _fake_c4)
    monkeypatch.setattr(architecture_facts, "get_full_scenario_graph", _fake_projection)
    monkeypatch.setattr(architecture_facts, "get_scenario_provenance_node_ids", _fake_node_ids)
    monkeypatch.setattr(architecture_facts, "_load_node_evidence", _fake_evidence)


@pytest.mark.anyio
async def test_build_architecture_facts_calls_recovery_and_emits_recovered_entities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection_id = uuid4()
    scenario_id = uuid4()
    endpoint, symbol, table, edge = _knowledge_fixture(collection_id)
    scenario = SimpleNamespace(id=scenario_id, collection_id=collection_id, name="AS-IS")
    session = _FakeSession(scenario, [], [endpoint, symbol, table], [edge])
    called = {"count": 0}

    def _fake_recover(*_args, **_kwargs):
        called["count"] += 1
        return _recovered_model()

    _install_common_patches(monkeypatch, collection_id, endpoint, symbol, table)
    monkeypatch.setattr(architecture_facts, "recover_architecture_model", _fake_recover)

    bundle = await architecture_facts.build_architecture_facts(
        session,
        collection_id=collection_id,
        scenario_id=scenario_id,
    )

    assert called["count"] == 1
    recovered_entity_ids = {
        fact.attributes.get("entity_id")
        for fact in bundle.facts
        if fact.fact_type in {"container", "component"}
    }
    assert "container:api" in recovered_entity_ids
    assert "component:session-manager" in recovered_entity_ids


@pytest.mark.anyio
async def test_existing_ports_adapters_extraction_still_works_with_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection_id = uuid4()
    scenario_id = uuid4()
    endpoint, symbol, table, edge = _knowledge_fixture(collection_id)
    scenario = SimpleNamespace(id=scenario_id, collection_id=collection_id, name="AS-IS")
    session = _FakeSession(scenario, [], [endpoint, symbol, table], [edge])
    _install_common_patches(monkeypatch, collection_id, endpoint, symbol, table)
    monkeypatch.setattr(
        architecture_facts,
        "recover_architecture_model",
        lambda *_args, **_kwargs: _recovered_model(),
    )

    bundle = await architecture_facts.build_architecture_facts(
        session,
        collection_id=collection_id,
        scenario_id=scenario_id,
    )

    assert {fact.direction for fact in bundle.ports_adapters} == {"inbound", "outbound"}
    assert {fact.protocol for fact in bundle.ports_adapters} == {"http", "sql"}


@pytest.mark.anyio
async def test_multi_membership_keeps_best_legacy_fields_and_preserves_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection_id = uuid4()
    scenario_id = uuid4()
    endpoint, symbol, table, edge = _knowledge_fixture(collection_id)
    scenario = SimpleNamespace(id=scenario_id, collection_id=collection_id, name="AS-IS")
    session = _FakeSession(scenario, [], [endpoint, symbol, table], [edge])
    _install_common_patches(monkeypatch, collection_id, endpoint, symbol, table)
    monkeypatch.setattr(
        architecture_facts,
        "recover_architecture_model",
        lambda *_args, **_kwargs: _recovered_model(),
    )

    bundle = await architecture_facts.build_architecture_facts(
        session,
        collection_id=collection_id,
        scenario_id=scenario_id,
    )

    outbound = next(fact for fact in bundle.ports_adapters if fact.direction == "outbound")
    assert outbound.container == "api"
    assert outbound.component == "Session Manager"
    assert outbound.attributes["candidate_memberships"] == [
        "component:session-manager",
        "container:api",
        "container:worker",
    ]


@pytest.mark.anyio
async def test_recovery_warnings_become_precise_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection_id = uuid4()
    scenario_id = uuid4()
    endpoint, symbol, table, edge = _knowledge_fixture(collection_id)
    scenario = SimpleNamespace(id=scenario_id, collection_id=collection_id, name="AS-IS")
    session = _FakeSession(scenario, [], [endpoint, symbol, table], [edge])
    _install_common_patches(monkeypatch, collection_id, endpoint, symbol, table)
    monkeypatch.setattr(
        architecture_facts,
        "recover_architecture_model",
        lambda *_args, **_kwargs: _recovered_model(),
    )

    bundle = await architecture_facts.build_architecture_facts(
        session,
        collection_id=collection_id,
        scenario_id=scenario_id,
    )

    assert "unresolved_hypotheses=1" in bundle.warnings
    assert "rejected_llm_adjudications=1" in bundle.warnings
    assert "missing_evidence_packets=1" in bundle.warnings
    assert not any("best-effort" in warning for warning in bundle.warnings)


@pytest.mark.anyio
async def test_bundle_canonical_payload_remains_stable_and_serializable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection_id = uuid4()
    scenario_id = uuid4()
    endpoint, symbol, table, edge = _knowledge_fixture(collection_id)
    scenario = SimpleNamespace(id=scenario_id, collection_id=collection_id, name="AS-IS")
    session = _FakeSession(scenario, [], [endpoint, symbol, table], [edge])
    _install_common_patches(monkeypatch, collection_id, endpoint, symbol, table)
    monkeypatch.setattr(
        architecture_facts,
        "recover_architecture_model",
        lambda *_args, **_kwargs: _recovered_model(),
    )

    bundle = await architecture_facts.build_architecture_facts(
        session,
        collection_id=collection_id,
        scenario_id=scenario_id,
    )

    payload_a = bundle.canonical_payload()
    payload_b = bundle.canonical_payload()
    assert payload_a == payload_b
    json.dumps(payload_a)
