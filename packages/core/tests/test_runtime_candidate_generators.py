"""Tests for artifact-driven runtime and system candidate generation."""

from __future__ import annotations

from contextmine_core.architecture.recovery import (
    generate_data_store_candidates,
    generate_external_system_candidates,
    generate_message_channel_candidates,
    generate_runtime_candidates,
    recover_architecture_model,
)


def _artifact_doc(
    *,
    artifact_id: str,
    file_path: str,
    text: str,
    parser: str | None = None,
    artifact_kind: str | None = None,
    structured_data: dict[str, object] | None = None,
) -> dict[str, object]:
    meta: dict[str, object] = {"file_path": file_path}
    if parser is not None:
        meta["parser"] = parser
        meta["parser_hint"] = parser
    if artifact_kind is not None:
        meta["artifact_kind"] = artifact_kind
    return {
        "id": artifact_id,
        "title": file_path.rsplit("/", 1)[-1],
        "text": text,
        "summary": text.strip() or file_path,
        "meta": meta,
        "structured_data": structured_data or {},
    }


def test_runtime_candidates_can_come_from_entrypoints_specs_and_deployments() -> None:
    nodes = [
        {
            "id": "api:list_sessions",
            "kind": "api_endpoint",
            "name": "GET /sessions",
            "natural_key": "api:list_sessions",
            "meta": {"file_path": "services/contextmine/worker/routes.py"},
        },
        {
            "id": "job:reindex",
            "kind": "job",
            "name": "Reindex Search",
            "natural_key": "job:reindex",
            "meta": {
                "file_path": "jobs/reindex.py",
            },
        },
        {
            "id": "job:nightly_sync",
            "kind": "job",
            "name": "Nightly Sync",
            "natural_key": "job:nightly_sync",
            "meta": {
                "file_path": "jobs/nightly_sync.py",
                "schedule": "0 3 * * *",
            },
        },
        {
            "id": "symbol:cli_main",
            "kind": "symbol",
            "name": "CLI Main",
            "natural_key": "symbol:cli_main",
            "meta": {"file_path": "tools/reindex_cli.py"},
        },
    ]
    docs = [
        _artifact_doc(
            artifact_id="artifact:openapi",
            file_path="specs/public-api.yaml",
            parser="openapi",
            artifact_kind="api_spec",
            text=(
                "openapi: 3.1.0\n"
                "info:\n"
                "  title: ContextMine Public API\n"
                "paths:\n"
                "  /sessions:\n"
                "    get:\n"
                "      summary: List sessions\n"
            ),
        ),
        _artifact_doc(
            artifact_id="artifact:k8s",
            file_path="infra/runtime.yaml",
            parser="kubernetes_manifest",
            artifact_kind="deployment",
            text=(
                "apiVersion: apps/v1\n"
                "kind: Deployment\n"
                "metadata:\n"
                "  name: embeddings-worker\n"
                "spec:\n"
                "  template:\n"
                "    spec:\n"
                "      containers:\n"
                "        - name: embeddings-worker\n"
                "          image: contextmine/worker:latest\n"
                "---\n"
                "apiVersion: batch/v1\n"
                "kind: CronJob\n"
                "metadata:\n"
                "  name: nightly-sync\n"
                "spec:\n"
                "  jobTemplate:\n"
                "    spec:\n"
                "      template:\n"
                "        spec:\n"
                "          containers:\n"
                "            - name: scheduler\n"
                "              image: contextmine/scheduler:latest\n"
            ),
        ),
    ]

    candidates = {entity.entity_id: entity for entity in generate_runtime_candidates(nodes, docs)}

    assert {
        "container:api",
        "container:worker",
        "container:job",
        "container:scheduler",
        "container:cli",
    } <= set(candidates)
    for entity in candidates.values():
        assert entity.confidence > 0
        assert entity.evidence


def test_data_store_candidates_can_come_from_sql_and_orm_artifacts() -> None:
    docs = [
        _artifact_doc(
            artifact_id="artifact:schema",
            file_path="db/schema/customer_sessions.sql",
            parser="sql",
            artifact_kind="sql",
            text=("create table customer_sessions (\n  id uuid primary key\n);\n"),
        ),
        _artifact_doc(
            artifact_id="artifact:orm",
            file_path="app/models/audit_event.py",
            parser="plain_text",
            artifact_kind="documentation",
            text=('class AuditEvent(Base):\n    __tablename__ = "audit_events"\n'),
        ),
    ]

    candidates = {entity.entity_id: entity for entity in generate_data_store_candidates([], docs)}

    assert "data_store:customer_sessions" in candidates
    assert "data_store:audit_events" in candidates
    assert all(entity.evidence and entity.confidence > 0 for entity in candidates.values())


def test_message_channel_candidates_can_come_from_asyncapi() -> None:
    docs = [
        _artifact_doc(
            artifact_id="artifact:asyncapi",
            file_path="specs/events.yaml",
            parser="asyncapi",
            artifact_kind="api_spec",
            text=(
                "asyncapi: 2.6.0\n"
                "info:\n"
                "  title: User Events\n"
                "channels:\n"
                "  user-events:\n"
                "    publish:\n"
                "      message:\n"
                "        name: UserEvent\n"
            ),
        ),
    ]

    candidates = {
        entity.entity_id: entity for entity in generate_message_channel_candidates([], docs)
    }

    assert "message_channel:user-events" in candidates
    assert candidates["message_channel:user-events"].evidence
    assert candidates["message_channel:user-events"].confidence > 0


def test_external_system_candidates_can_come_from_config_client_libraries_and_api_artifacts() -> (
    None
):
    docs = [
        _artifact_doc(
            artifact_id="artifact:config",
            file_path="config/production.env",
            parser="plain_text",
            artifact_kind="documentation",
            text="STRIPE_BASE_URL=https://api.stripe.com/v1\n",
        ),
        _artifact_doc(
            artifact_id="artifact:client",
            file_path="services/contextmine/worker/openai_client.py",
            parser="plain_text",
            artifact_kind="documentation",
            text="from openai import OpenAI\nclient = OpenAI()\n",
        ),
        _artifact_doc(
            artifact_id="artifact:github-openapi",
            file_path="specs/github-api.yaml",
            parser="openapi",
            artifact_kind="api_spec",
            text=(
                "openapi: 3.0.0\n"
                "info:\n"
                "  title: GitHub API\n"
                "servers:\n"
                "  - url: https://api.github.com\n"
                "paths: {}\n"
            ),
        ),
    ]

    candidates = {
        entity.entity_id: entity for entity in generate_external_system_candidates([], docs)
    }

    assert "external_system:stripe-api" in candidates
    assert "external_system:openai" in candidates
    assert "external_system:github-api" in candidates
    assert all(entity.evidence and entity.confidence > 0 for entity in candidates.values())


def test_recovery_uses_stronger_runtime_evidence_over_path_priors() -> None:
    nodes = [
        {
            "id": "api:list_sessions",
            "kind": "api_endpoint",
            "name": "GET /sessions",
            "natural_key": "api:list_sessions",
            "meta": {"file_path": "services/contextmine/worker/routes.py"},
        }
    ]
    docs = [
        _artifact_doc(
            artifact_id="artifact:openapi",
            file_path="specs/public-api.yaml",
            parser="openapi",
            artifact_kind="api_spec",
            text=(
                "openapi: 3.1.0\n"
                "info:\n"
                "  title: ContextMine Public API\n"
                "paths:\n"
                "  /sessions:\n"
                "    get:\n"
                "      summary: List sessions\n"
            ),
        ),
        _artifact_doc(
            artifact_id="artifact:worker-runtime",
            file_path="infra/worker-runtime.yaml",
            parser="kubernetes_manifest",
            artifact_kind="deployment",
            text=(
                "apiVersion: apps/v1\n"
                "kind: Deployment\n"
                "metadata:\n"
                "  name: embeddings-worker\n"
                "spec:\n"
                "  template:\n"
                "    spec:\n"
                "      containers:\n"
                "        - name: embeddings-worker\n"
                "          image: contextmine/worker:latest\n"
            ),
        ),
    ]

    model = recover_architecture_model(nodes, [], docs=docs)

    memberships = model.memberships_for("api:list_sessions")
    hypothesis = next(row for row in model.hypotheses if row.subject_ref == "api:list_sessions")

    assert [membership.entity_id for membership in memberships] == ["container:api"]
    assert hypothesis.selected_entity_ids == ("container:api",)
    assert "container:worker" in hypothesis.candidate_entity_ids


def test_recovery_includes_artifact_backed_runtime_and_system_entities() -> None:
    nodes = [
        {
            "id": "api:list_sessions",
            "kind": "api_endpoint",
            "name": "GET /sessions",
            "natural_key": "api:list_sessions",
            "meta": {"file_path": "services/contextmine/api/routes.py"},
        }
    ]
    docs = [
        _artifact_doc(
            artifact_id="artifact:openapi",
            file_path="specs/public-api.yaml",
            parser="openapi",
            artifact_kind="api_spec",
            text=(
                "openapi: 3.1.0\n"
                "info:\n"
                "  title: ContextMine Public API\n"
                "paths:\n"
                "  /sessions:\n"
                "    get:\n"
                "      summary: List sessions\n"
            ),
        ),
        _artifact_doc(
            artifact_id="artifact:schema",
            file_path="db/schema/customer_sessions.sql",
            parser="sql",
            artifact_kind="sql",
            text="create table customer_sessions (id uuid primary key);",
        ),
        _artifact_doc(
            artifact_id="artifact:asyncapi",
            file_path="specs/events.yaml",
            parser="asyncapi",
            artifact_kind="api_spec",
            text=(
                "asyncapi: 2.6.0\n"
                "info:\n"
                "  title: User Events\n"
                "channels:\n"
                "  user-events:\n"
                "    publish:\n"
                "      message:\n"
                "        name: UserEvent\n"
            ),
        ),
        _artifact_doc(
            artifact_id="artifact:config",
            file_path="config/production.env",
            parser="plain_text",
            artifact_kind="documentation",
            text="STRIPE_BASE_URL=https://api.stripe.com/v1\n",
        ),
    ]

    model = recover_architecture_model(nodes, [], docs=docs)
    entity_ids = {entity.entity_id for entity in model.entities}

    assert "container:api" in entity_ids
    assert "data_store:customer_sessions" in entity_ids
    assert "message_channel:user-events" in entity_ids
    assert "external_system:stripe-api" in entity_ids
    assert all(entity.evidence and entity.confidence > 0 for entity in model.entities)
