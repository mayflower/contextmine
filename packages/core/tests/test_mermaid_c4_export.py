from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from contextmine_core.architecture.recovery_model import (
    RecoveredArchitectureEntity,
    RecoveredArchitectureHypothesis,
    RecoveredArchitectureMembership,
    RecoveredArchitectureModel,
    RecoveredArchitectureRelationship,
)
from contextmine_core.architecture.schemas import EvidenceRef
from contextmine_core.exports import mermaid_c4 as mermaid_export
from contextmine_core.models import Document, KnowledgeNode, KnowledgeNodeKind
from contextmine_core.twin import GraphProjection


class _Result:
    def __init__(self, scenario_name: str, collection_id=None) -> None:
        self._scenario = SimpleNamespace(name=scenario_name, collection_id=collection_id or uuid4())

    def scalar_one(self) -> SimpleNamespace:
        return self._scenario


class _Scalars:
    def __init__(self, values):
        self._values = values

    def all(self):
        return self._values


class _ScalarResult:
    def __init__(self, values):
        self._values = values

    def all(self):
        return self._values

    def scalars(self):
        return _Scalars(self._values)


class _FakeSession:
    def __init__(self, jobs=None, evidence_rows=None, documents=None):
        self.jobs = jobs or []
        self.evidence_rows = evidence_rows or []
        self.documents = documents or []

    async def execute(self, stmt):  # noqa: ANN001
        statement = str(stmt)
        if "knowledge_node_evidence" in statement:
            return _ScalarResult(self.evidence_rows)
        if "knowledge_nodes" in statement:
            return _ScalarResult(self.jobs)
        if "knowledge_edges" in statement:
            return _ScalarResult([])
        if "FROM documents" in statement:
            return _ScalarResult(self.documents)
        return _Result("AS-IS", collection_id=uuid4())


def _evidence(ref: str) -> tuple[EvidenceRef, ...]:
    return (EvidenceRef(kind="file", ref=ref),)


def _empty_recovered_model() -> RecoveredArchitectureModel:
    return RecoveredArchitectureModel()


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
            RecoveredArchitectureEntity(
                entity_id="external_system:github-oauth",
                kind="external_system",
                name="GitHub OAuth",
                confidence=0.92,
                evidence=_evidence("docs/integrations/github-oauth.md"),
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
            RecoveredArchitectureRelationship(
                source_entity_id="container:api",
                target_entity_id="external_system:github-oauth",
                kind="invokes",
                confidence=0.9,
                evidence=_evidence("docs/integrations/github-oauth.md"),
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
            RecoveredArchitectureMembership(
                subject_ref="job:embeddings_sync",
                entity_id="container:worker",
                relationship_kind="runs_in",
                confidence=0.95,
                evidence=_evidence("services/contextmine/worker/jobs.py"),
            ),
        ),
        hypotheses=(
            RecoveredArchitectureHypothesis(
                subject_ref="symbol:session_manager",
                candidate_entity_ids=("container:api", "container:worker"),
                selected_entity_ids=("container:api", "container:worker"),
                rationale="Shared session manager code spans API and worker.",
                status="ambiguous",
                confidence=0.88,
                evidence=_evidence("packages/core/session_manager.py"),
            ),
        ),
    )


@pytest.mark.anyio
async def test_mermaid_c4_container_uses_architecture_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_recovered(*_args, **_kwargs):
        return _recovered_model()

    async def _should_not_fallback(**_kwargs):  # noqa: ANN003
        raise AssertionError("container view should not use file-bucket architecture projection")

    monkeypatch.setattr(mermaid_export, "_load_recovered_architecture_model", _fake_recovered)
    monkeypatch.setattr(mermaid_export, "get_full_scenario_graph", _should_not_fallback)

    content = await mermaid_export.export_mermaid_c4(_FakeSession(), uuid4(), c4_view="container")

    assert "C4Container" in content
    assert "API Runtime" in content


@pytest.mark.anyio
async def test_component_view_renders_components(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_recovered(*_args, **_kwargs):
        return _recovered_model()

    monkeypatch.setattr(mermaid_export, "_load_recovered_architecture_model", _fake_recovered)
    result = await mermaid_export.export_mermaid_c4_result(
        _FakeSession(),
        uuid4(),
        c4_view="component",
        c4_scope="Session Manager",
    )

    assert result.c4_view == "component"
    assert "C4Component" in result.content
    assert "reads_writes" in result.content
    assert "Session Manager" in result.content
    assert any("Ambiguous recovered memberships" in warning for warning in result.warnings)


@pytest.mark.anyio
async def test_component_view_warns_explicitly_before_largest_component_degraded_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_recovered(*_args, **_kwargs):
        return _empty_recovered_model()

    async def _fake_graph(**_kwargs):  # noqa: ANN003
        return {
            "nodes": [
                {
                    "id": "largest-component",
                    "natural_key": "component|contextmine|api|session-manager",
                    "kind": "component",
                    "name": "Session Manager",
                    "meta": {"component": "session-manager", "container": "api", "member_count": 12},
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
        _FakeSession(),
        uuid4(),
        c4_view="component",
    )

    assert "C4Component" in result.content
    assert any("degraded" in warning.lower() for warning in result.warnings)
    assert any("largest component" in warning.lower() for warning in result.warnings)


@pytest.mark.anyio
async def test_code_view_warns_on_calls_fallback_and_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    symbol_nodes = [
        {
            "id": f"s{index}",
            "natural_key": f"symbol:Node{index}",
            "kind": "method",
            "name": f"node_{index}",
            "meta": {"file_path": "services/billing/api/invoice.py"},
        }
        for index in range(1, 13)
    ]
    symbol_edges = [
        {
            "id": f"e{index}",
            "source_node_id": f"s{index}",
            "target_node_id": f"s{index + 1}",
            "kind": "symbol_references_symbol",
            "meta": {},
        }
        for index in range(1, 12)
    ]

    async def _fake_get_full_scenario_graph(**kwargs):  # noqa: ANN003
        projection = kwargs.get("projection")
        if projection == GraphProjection.CODE_FILE:
            return {
                "nodes": [
                    {
                        "id": "f1",
                        "natural_key": "file:services/billing/api/invoice.py",
                        "kind": "file",
                        "name": "services/billing/api/invoice.py",
                        "meta": {},
                    }
                ],
                "edges": [],
                "total_nodes": 1,
                "projection": "code_file",
                "entity_level": "file",
                "grouping_strategy": "heuristic",
                "excluded_kinds": [],
            }

        return {
            "nodes": symbol_nodes,
            "edges": symbol_edges,
            "total_nodes": len(symbol_nodes),
            "projection": "code_symbol",
            "entity_level": "symbol",
            "grouping_strategy": "heuristic",
            "excluded_kinds": [],
        }

    async def _fake_recovered(*_args, **_kwargs):
        return _recovered_model()

    monkeypatch.setattr(mermaid_export, "_load_recovered_architecture_model", _fake_recovered)
    monkeypatch.setattr(mermaid_export, "get_full_scenario_graph", _fake_get_full_scenario_graph)

    result = await mermaid_export.export_mermaid_c4_result(
        _FakeSession(),
        uuid4(),
        c4_view="code",
        c4_scope="Session Manager",
        max_nodes=10,
    )

    assert "C4Component" in result.content
    assert any("Ambiguous recovered code scope" in warning for warning in result.warnings)
    assert any(
        "symbol_calls_symbol" in warning or "fallback" in warning for warning in result.warnings
    )
    assert any("limited" in warning.lower() for warning in result.warnings)


@pytest.mark.anyio
async def test_context_view_sparse_emits_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_surface_counts(_session, _collection_id):
        return {
            "api_endpoint": 0,
            "graphql_operation": 0,
            "service_rpc": 0,
            "job": 0,
            "message_schema": 0,
            "db_table": 0,
        }

    async def _fake_recovered(*_args, **_kwargs):
        return _empty_recovered_model()

    monkeypatch.setattr(mermaid_export, "_surface_counts", _fake_surface_counts)
    monkeypatch.setattr(mermaid_export, "_load_recovered_architecture_model", _fake_recovered)

    result = await mermaid_export._render_context_view(
        session=_FakeSession(),
        collection_id=uuid4(),
        scenario_id=uuid4(),
        scenario_name="AS-IS",
        c4_scope=None,
    )

    assert "C4Context" in result.content
    assert result.warnings


@pytest.mark.anyio
async def test_context_view_uses_recovered_external_systems(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_recovered(*_args, **_kwargs):
        return _recovered_model()

    monkeypatch.setattr(mermaid_export, "_load_recovered_architecture_model", _fake_recovered)

    result = await mermaid_export._render_context_view(
        session=_FakeSession(),
        collection_id=uuid4(),
        scenario_id=uuid4(),
        scenario_name="AS-IS",
        c4_scope=None,
    )

    assert "GitHub OAuth" in result.content
    assert "System_Ext" in result.content


@pytest.mark.anyio
async def test_load_recovered_architecture_model_includes_persisted_docs(
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

    async def _fake_node_ids(*_args, **_kwargs):
        return {adr_file.id, api_symbol.id, worker_job.id}

    monkeypatch.setattr(mermaid_export, "get_scenario_provenance_node_ids", _fake_node_ids)

    model = await mermaid_export._load_recovered_architecture_model(
        _FakeSession(
            jobs=[adr_file, api_symbol, worker_job],
            documents=[document],
        ),
        scenario_id=scenario_id,
        collection_id=collection_id,
    )

    assert len(model.decisions) == 1
    assert model.decisions[0].title == "ADR-001 async embedding workers"


@pytest.mark.anyio
async def test_deployment_view_uses_recovered_runtime_memberships(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_worker = SimpleNamespace(
        id=uuid4(),
        name="Embeddings Sync",
        natural_key="job:embeddings_sync",
        meta={"schedule": "0 * * * *", "framework": "job"},
    )
    job_unknown = SimpleNamespace(
        id=uuid4(),
        name="Orphan Job",
        natural_key="job:orphan",
        meta={"framework": "job"},
    )

    async def _fake_recovered(*_args, **_kwargs):
        return _recovered_model()

    async def _fake_get_full_scenario_graph(**kwargs):  # noqa: ANN003
        assert kwargs["projection"] == GraphProjection.ARCHITECTURE
        return {
            "nodes": [
                {
                    "id": "arch:container:1",
                    "name": "Worker Runtime",
                    "kind": "container",
                    "meta": {},
                }
            ],
            "edges": [],
            "total_nodes": 1,
            "projection": "architecture",
            "entity_level": "container",
            "grouping_strategy": "heuristic",
            "excluded_kinds": [],
        }

    monkeypatch.setattr(mermaid_export, "_load_recovered_architecture_model", _fake_recovered)
    monkeypatch.setattr(mermaid_export, "get_full_scenario_graph", _fake_get_full_scenario_graph)

    result = await mermaid_export._render_deployment_view(
        session=_FakeSession(jobs=[job_worker, job_unknown]),
        collection_id=uuid4(),
        scenario_id=uuid4(),
        scenario_name="AS-IS",
        c4_scope=None,
        max_nodes=50,
    )

    assert "Worker Runtime" in result.content
    assert "shared" in result.content.lower()
    assert any("shared deployment" in warning.lower() for warning in result.warnings)


@pytest.mark.anyio
async def test_deployment_view_sparse_emits_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_recovered(*_args, **_kwargs):
        return _empty_recovered_model()

    monkeypatch.setattr(mermaid_export, "_load_recovered_architecture_model", _fake_recovered)

    result = await mermaid_export._render_deployment_view(
        session=_FakeSession(jobs=[]),
        collection_id=uuid4(),
        scenario_id=uuid4(),
        scenario_name="AS-IS",
        c4_scope=None,
        max_nodes=50,
    )

    assert "C4Deployment" in result.content
    assert result.warnings
