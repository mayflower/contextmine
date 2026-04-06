"""Tests for ADR-like decision recovery."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from contextmine_core.architecture import facts as architecture_facts
from contextmine_core.architecture.recovery import recover_architecture_model
from contextmine_core.models import (
    Document,
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

    def scalars(self):
        return self

    def all(self):
        return self._value


class _FakeSession:
    def __init__(self, scenario, nodes, edges, documents=None):
        self.scenario = scenario
        self.nodes = nodes
        self.edges = edges
        self.documents = documents or []

    async def execute(self, stmt):  # noqa: ANN001
        statement = str(stmt)
        if "FROM twin_scenarios" in statement:
            return _ScalarResult(self.scenario)
        if "FROM metric_snapshots" in statement:
            return _ScalarResult([])
        if "FROM knowledge_nodes" in statement:
            return _ScalarResult(self.nodes)
        if "FROM knowledge_edges" in statement:
            return _ScalarResult(self.edges)
        if "FROM documents" in statement:
            return _ScalarResult(self.documents)
        raise AssertionError(f"Unexpected statement: {statement}")


def test_explicit_adr_docs_produce_confirmed_decisions() -> None:
    fixture = build_architecture_recovery_fixture()
    model = recover_architecture_model(fixture["nodes"], fixture["edges"], docs=fixture["docs"])

    assert len(model.decisions) == 1
    decision = model.decisions[0]
    assert decision.status == "confirmed"
    assert decision.affected_entity_ids == ("container:api", "container:worker")
    assert any(ref.ref == "docs/adr/001-async-embedding-workers.md" for ref in decision.evidence)


def test_indirect_doc_inference_creates_hypothesis_decision() -> None:
    fixture = build_architecture_recovery_fixture()
    docs = [
        {
            "id": "doc:note:shared-session-manager",
            "title": "Shared session manager note",
            "text": "Session Manager remains shared between API Runtime and Worker Runtime.",
            "meta": {"file_path": "docs/notes/shared-session-manager.md"},
        }
    ]
    model = recover_architecture_model(fixture["nodes"], fixture["edges"], docs=docs)

    assert len(model.decisions) == 1
    decision = model.decisions[0]
    assert decision.status == "hypothesis"
    assert "container:api" in decision.affected_entity_ids
    assert "container:worker" in decision.affected_entity_ids


@pytest.mark.anyio
async def test_recovered_decisions_flow_through_architecture_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection_id = uuid4()
    scenario_id = uuid4()
    fixture = build_architecture_recovery_fixture()
    model = recover_architecture_model(fixture["nodes"], fixture["edges"], docs=fixture["docs"])

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
    edge = KnowledgeEdge(
        id=uuid4(),
        collection_id=collection_id,
        source_node_id=symbol.id,
        target_node_id=endpoint.id,
        kind=KnowledgeEdgeKind.SYMBOL_CALLS_SYMBOL,
    )

    async def _fake_c4(*_args, **_kwargs):
        return SimpleNamespace(content="C4Context", warnings=[])

    async def _fake_node_ids(*_args, **_kwargs):
        return {endpoint.id, symbol.id}

    async def _fake_evidence(*_args, **_kwargs):
        return {}

    monkeypatch.setattr(architecture_facts, "export_mermaid_c4_result", _fake_c4)
    monkeypatch.setattr(architecture_facts, "get_scenario_provenance_node_ids", _fake_node_ids)
    monkeypatch.setattr(architecture_facts, "_load_node_evidence", _fake_evidence)
    monkeypatch.setattr(
        architecture_facts, "recover_architecture_model", lambda *_args, **_kwargs: model
    )

    bundle = await architecture_facts.build_architecture_facts(
        _FakeSession(
            SimpleNamespace(id=scenario_id, collection_id=collection_id, name="AS-IS"),
            [endpoint, symbol],
            [edge],
        ),
        collection_id=collection_id,
        scenario_id=scenario_id,
    )

    decision_facts = [fact for fact in bundle.facts if fact.fact_type == "architecture_decision"]
    assert len(decision_facts) == 1
    assert decision_facts[0].attributes["status"] == "confirmed"


@pytest.mark.anyio
async def test_default_pipeline_loads_persisted_adr_docs_into_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection_id = uuid4()
    scenario_id = uuid4()
    doc_id = uuid4()

    adr_file = KnowledgeNode(
        id=uuid4(),
        collection_id=collection_id,
        kind=KnowledgeNodeKind.FILE,
        natural_key="docs/adr/001-async-embedding-workers.md",
        name="ADR-001 async embedding workers",
        meta={
            "document_id": str(doc_id),
            "file_path": "docs/adr/001-async-embedding-workers.md",
            "uri": "docs/adr/001-async-embedding-workers.md",
        },
    )
    api_symbol = KnowledgeNode(
        id=uuid4(),
        collection_id=collection_id,
        kind=KnowledgeNodeKind.SYMBOL,
        natural_key="symbol:api_session_handler",
        name="Api Session Handler",
        meta={
            "file_path": "services/contextmine/api/session_handler.py",
            "architecture": {"domain": "contextmine", "container": "api"},
        },
    )
    worker_job = KnowledgeNode(
        id=uuid4(),
        collection_id=collection_id,
        kind=KnowledgeNodeKind.JOB,
        natural_key="job:embeddings_sync",
        name="Embeddings Sync",
        meta={
            "file_path": "services/contextmine/worker/jobs/embeddings.py",
            "architecture": {"domain": "contextmine", "container": "worker"},
        },
    )
    edge = KnowledgeEdge(
        id=uuid4(),
        collection_id=collection_id,
        source_node_id=api_symbol.id,
        target_node_id=worker_job.id,
        kind=KnowledgeEdgeKind.SYMBOL_CALLS_SYMBOL,
    )
    document = Document(
        id=doc_id,
        source_id=uuid4(),
        uri="docs/adr/001-async-embedding-workers.md",
        title="ADR-001 async embedding workers",
        content_markdown=(
            "Embeddings generation runs in the worker runtime.\n"
            "Session Manager remains shared between API Runtime and Worker Runtime."
        ),
        content_hash="hash",
        meta={"file_path": "docs/adr/001-async-embedding-workers.md"},
        last_seen_at=datetime.now(UTC),
    )

    async def _fake_c4(*_args, **_kwargs):
        return SimpleNamespace(content="C4Context", warnings=[])

    async def _fake_node_ids(*_args, **_kwargs):
        return {adr_file.id, api_symbol.id, worker_job.id}

    async def _fake_evidence(*_args, **_kwargs):
        return {}

    session = _FakeSession(
        SimpleNamespace(id=scenario_id, collection_id=collection_id, name="AS-IS"),
        [adr_file, api_symbol, worker_job],
        [edge],
        documents=[document],
    )

    monkeypatch.setattr(architecture_facts, "export_mermaid_c4_result", _fake_c4)
    monkeypatch.setattr(architecture_facts, "get_scenario_provenance_node_ids", _fake_node_ids)
    monkeypatch.setattr(architecture_facts, "_load_node_evidence", _fake_evidence)
    bundle = await architecture_facts.build_architecture_facts(
        session,
        collection_id=collection_id,
        scenario_id=scenario_id,
    )

    decision_facts = [fact for fact in bundle.facts if fact.fact_type == "architecture_decision"]
    assert len(decision_facts) == 1
    assert decision_facts[0].attributes["status"] == "hypothesis"
    assert decision_facts[0].evidence[0].ref == "docs/adr/001-async-embedding-workers.md"
