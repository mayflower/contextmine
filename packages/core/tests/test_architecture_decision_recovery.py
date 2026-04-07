"""Tests for ADR-like decision recovery."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from contextmine_core.architecture import facts as architecture_facts
from contextmine_core.architecture.recovery import recover_architecture_model
from contextmine_core.architecture.recovery_decisions import recover_architecture_decisions
from contextmine_core.architecture.recovery_docs import load_recovery_docs
from contextmine_core.architecture.recovery_model import RecoveredArchitectureEntity
from contextmine_core.architecture.schemas import EvidenceRef
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


def test_structured_adr_fields_produce_confirmed_decision_with_supersedes_and_section_evidence() -> (
    None
):
    entities = (
        RecoveredArchitectureEntity(
            entity_id="container:worker",
            kind="container",
            name="Worker Runtime",
            confidence=0.96,
            evidence=(EvidenceRef(kind="file", ref="services/contextmine/worker/jobs.py"),),
            attributes={"container": "worker"},
        ),
    )
    docs = [
        {
            "id": "artifact:docs/adr/0007.md",
            "title": "Placeholder title",
            "text": (
                "# Async embedding workers\n\n"
                "## Decision\nMove embeddings generation to the worker runtime.\n\n"
                "## Consequences\nRequests remain responsive.\n"
            ),
            "summary": "Move embeddings generation to the worker runtime.",
            "meta": {"file_path": "docs/adr/0007-async-embedding-workers.md"},
            "structured_data": {
                "title": "Async embedding workers",
                "status": "accepted",
                "decision": "Move embeddings generation to the worker runtime.",
                "consequences": "Requests remain responsive.",
                "affected_entity_ids": ["container:worker"],
                "supersedes": "ADR-0004",
                "replaces": "Synchronous API embeddings",
            },
        }
    ]

    decisions = recover_architecture_decisions(docs, entities)

    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.title == "Async embedding workers"
    assert decision.status == "confirmed"
    assert decision.summary == "Move embeddings generation to the worker runtime."
    assert decision.affected_entity_ids == ("container:worker",)
    assert decision.attributes["supersedes"] == "ADR-0004"
    assert decision.attributes["replaces"] == "Synchronous API embeddings"
    assert any(ref.kind == "section" for ref in decision.evidence)


def test_decision_entity_linking_uses_entity_id_aliases_and_repo_paths() -> None:
    entities = (
        RecoveredArchitectureEntity(
            entity_id="container:api",
            kind="container",
            name="API Runtime",
            confidence=0.96,
            evidence=(EvidenceRef(kind="file", ref="services/contextmine/api/routes.py"),),
            attributes={"container": "api", "aliases": ["public api", "rest api"]},
        ),
        RecoveredArchitectureEntity(
            entity_id="component:session-service",
            kind="component",
            name="Session Service",
            confidence=0.92,
            evidence=(EvidenceRef(kind="file", ref="services/contextmine/api/session_service.py"),),
        ),
    )
    docs = [
        {
            "id": "artifact:docs/adr/0012.md",
            "title": "Session API boundary",
            "text": (
                "The public API keeps session creation in container:api and the "
                "services/contextmine/api/session_service.py implementation."
            ),
            "summary": "Clarify session API boundary.",
            "meta": {"file_path": "docs/adr/0012-session-api-boundary.md"},
            "structured_data": {
                "decision": (
                    "The public API keeps session creation in container:api and the "
                    "services/contextmine/api/session_service.py implementation."
                ),
                "affected_entity_refs": [
                    "container:api",
                    "public api",
                    "services/contextmine/api/session_service.py",
                ],
            },
        }
    ]

    decisions = recover_architecture_decisions(docs, entities)

    assert len(decisions) == 1
    assert decisions[0].status == "confirmed"
    assert decisions[0].affected_entity_ids == (
        "component:session-service",
        "container:api",
    )


def test_ambiguous_entity_linking_stays_hypothesis_with_reason_and_counter_evidence() -> None:
    entities = (
        RecoveredArchitectureEntity(
            entity_id="container:api",
            kind="container",
            name="API Runtime",
            confidence=0.96,
            evidence=(EvidenceRef(kind="file", ref="services/contextmine/api/routes.py"),),
            attributes={"container": "api", "aliases": ["request runtime"]},
        ),
        RecoveredArchitectureEntity(
            entity_id="container:worker",
            kind="container",
            name="Worker Runtime",
            confidence=0.95,
            evidence=(EvidenceRef(kind="file", ref="services/contextmine/worker/jobs.py"),),
            attributes={"container": "worker", "aliases": ["processing runtime"]},
        ),
    )
    docs = [
        {
            "id": "artifact:docs/notes/runtime-boundary.md",
            "title": "Runtime boundary note",
            "text": "The processing runtime should move out of the request runtime path.",
            "summary": "Keep long-running work off the request path.",
            "meta": {"file_path": "docs/notes/runtime-boundary.md"},
        }
    ]

    decisions = recover_architecture_decisions(docs, entities)

    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.status == "hypothesis"
    assert decision.affected_entity_ids == ("container:api", "container:worker")
    assert decision.attributes["linking_reason"]
    assert decision.attributes["counter_evidence_entity_ids"] == [
        "container:api",
        "container:worker",
    ]


@pytest.mark.anyio
async def test_repo_artifact_without_document_row_can_drive_confirmed_decision_recovery() -> None:
    file_node = KnowledgeNode(
        id=uuid4(),
        collection_id=uuid4(),
        kind=KnowledgeNodeKind.FILE,
        natural_key="docs/records/0014.md",
        name="Record 14",
        meta={
            "uri": "docs/records/0014.md",
            "file_path": "docs/records/0014.md",
            "content_markdown": (
                "---\n"
                "affected_entity_ids:\n"
                "  - container:worker\n"
                "supersedes: ADR-0006\n"
                "---\n"
                "# Worker-only embeddings\n\n"
                "## Decision\nUse container:worker for embeddings generation.\n"
            ),
        },
    )
    docs = await load_recovery_docs(_FakeSession(None, [], [], documents=[]), [file_node])
    entities = (
        RecoveredArchitectureEntity(
            entity_id="container:worker",
            kind="container",
            name="Worker Runtime",
            confidence=0.96,
            evidence=(EvidenceRef(kind="file", ref="services/contextmine/worker/jobs.py"),),
            attributes={"container": "worker"},
        ),
    )

    decisions = recover_architecture_decisions(docs, entities)

    assert len(decisions) == 1
    assert decisions[0].status == "confirmed"
    assert decisions[0].affected_entity_ids == ("container:worker",)
    assert any(ref.kind == "section" for ref in decisions[0].evidence)
