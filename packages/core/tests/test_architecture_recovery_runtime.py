"""Focused runtime/context recovery tests."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest
from contextmine_core.architecture.recovery import recover_architecture_model
from contextmine_core.exports import mermaid_c4 as mermaid_export

from .models.architecture_recovery_fixture import build_architecture_recovery_fixture


class _ScalarResult:
    def __init__(self, values):
        self._values = values

    def all(self):
        return self._values

    def scalars(self):
        return self


class _FakeSession:
    def __init__(self, jobs=None, evidence_rows=None):
        self.jobs = jobs or []
        self.evidence_rows = evidence_rows or []

    async def execute(self, stmt):  # noqa: ANN001
        statement = str(stmt)
        if "knowledge_node_evidence" in statement:
            return _ScalarResult(self.evidence_rows)
        if "knowledge_nodes" in statement:
            return _ScalarResult(self.jobs)
        raise AssertionError(f"Unexpected statement: {statement}")


def test_external_system_entities_are_recovered_from_local_evidence() -> None:
    fixture = build_architecture_recovery_fixture()
    model = recover_architecture_model(fixture["nodes"], fixture["edges"], docs=fixture["docs"])

    external_names = {entity.name for entity in model.entities if entity.kind == "external_system"}
    assert external_names == {"GitHub OAuth", "OpenAI Embeddings"}


def test_job_runtime_membership_is_recovered_from_job_evidence() -> None:
    fixture = build_architecture_recovery_fixture()
    model = recover_architecture_model(fixture["nodes"], fixture["edges"], docs=fixture["docs"])

    job_memberships = {
        membership.entity_id
        for membership in model.memberships_for("job:embeddings_sync")
        if membership.entity_id.startswith("container:")
    }
    assert job_memberships == {"container:worker"}


@pytest.mark.anyio
async def test_context_view_renders_named_recovered_systems(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = build_architecture_recovery_fixture()
    model = recover_architecture_model(fixture["nodes"], fixture["edges"], docs=fixture["docs"])

    async def _fake_recovered(*_args, **_kwargs):
        return model

    monkeypatch.setattr(mermaid_export, "_load_recovered_architecture_model", _fake_recovered)

    result = await mermaid_export._render_context_view(
        session=_FakeSession(),
        collection_id=uuid4(),
        scenario_id=uuid4(),
        scenario_name="AS-IS",
        c4_scope=None,
    )

    assert "GitHub OAuth" in result.content
    assert "OpenAI Embeddings" in result.content
    assert "customer_sessions" in result.content


@pytest.mark.anyio
async def test_shared_deployment_is_only_used_when_runtime_evidence_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = build_architecture_recovery_fixture()
    model = recover_architecture_model(fixture["nodes"], fixture["edges"], docs=fixture["docs"])
    worker_job = SimpleNamespace(
        id=uuid4(),
        name="Embeddings Sync",
        natural_key="job:embeddings_sync",
        meta={"schedule": "0 * * * *", "framework": "job"},
    )
    orphan_job = SimpleNamespace(
        id=uuid4(),
        name="Orphan Job",
        natural_key="job:orphan",
        meta={"framework": "job"},
    )

    async def _fake_recovered(*_args, **_kwargs):
        return model

    async def _fake_container_graph(**_kwargs):  # noqa: ANN003
        return {
            "nodes": [
                {"id": "container:api", "name": "API Runtime", "kind": "container", "meta": {}},
                {
                    "id": "container:worker",
                    "name": "Worker Runtime",
                    "kind": "container",
                    "meta": {},
                },
            ],
            "edges": [],
            "total_nodes": 2,
            "projection": "architecture",
            "entity_level": "container",
            "grouping_strategy": "recovered",
            "excluded_kinds": [],
        }

    monkeypatch.setattr(mermaid_export, "_load_recovered_architecture_model", _fake_recovered)
    monkeypatch.setattr(mermaid_export, "get_full_scenario_graph", _fake_container_graph)

    result = await mermaid_export._render_deployment_view(
        session=_FakeSession(jobs=[worker_job, orphan_job]),
        collection_id=uuid4(),
        scenario_id=uuid4(),
        scenario_name="AS-IS",
        c4_scope=None,
        max_nodes=50,
    )

    assert "Worker Runtime" in result.content
    assert "shared" in result.content.lower()
    assert any("shared deployment" in warning.lower() for warning in result.warnings)
