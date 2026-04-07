"""Tests for evidence-backed component clustering."""

from __future__ import annotations

from contextmine_core.architecture.recovery import recover_architecture_model
from contextmine_core.twin.projections import build_inferred_architecture_projection


def _symbol(
    *,
    node_id: str,
    name: str,
    file_path: str,
    container: str | None = None,
    component: str | None = None,
) -> dict[str, object]:
    meta: dict[str, object] = {"file_path": file_path}
    if container is not None:
        architecture: dict[str, str] = {"domain": "contextmine", "container": container}
        if component is not None:
            architecture["component"] = component
        meta["architecture"] = architecture
    return {
        "id": node_id,
        "kind": "symbol",
        "name": name,
        "natural_key": node_id,
        "meta": meta,
    }


def _edge(source: str, target: str, kind: str) -> dict[str, object]:
    return {
        "source_node_id": source,
        "target_node_id": target,
        "kind": kind,
        "meta": {},
    }


def test_related_symbols_can_be_clustered_into_one_component() -> None:
    nodes = [
        _symbol(
            node_id="symbol:create_session",
            name="Create Session",
            file_path="services/contextmine/api/session_service.py",
            container="api",
        ),
        _symbol(
            node_id="symbol:load_session",
            name="Load Session",
            file_path="services/contextmine/api/session_service.py",
            container="api",
        ),
        {
            "id": "db:customer_sessions",
            "kind": "db_table",
            "name": "customer_sessions",
            "natural_key": "db:customer_sessions",
            "meta": {"file_path": "db/schema/customer_sessions.sql"},
        },
        {
            "id": "ext:github_oauth",
            "kind": "external_system",
            "name": "GitHub OAuth",
            "natural_key": "ext:github_oauth",
            "meta": {"file_path": "docs/integrations/github-oauth.md"},
        },
    ]
    edges = [
        _edge("symbol:create_session", "symbol:load_session", "symbol_calls_symbol"),
        _edge("symbol:create_session", "db:customer_sessions", "reads_writes"),
        _edge("symbol:load_session", "db:customer_sessions", "reads_writes"),
        _edge("symbol:create_session", "ext:github_oauth", "invokes"),
        _edge("symbol:load_session", "ext:github_oauth", "invokes"),
    ]
    docs = [
        {
            "id": "doc:session-service",
            "title": "Session Service",
            "text": (
                "The Session Service handles create session and load session flows "
                "for customer sessions and GitHub OAuth."
            ),
            "meta": {"file_path": "docs/architecture/session-service.md"},
        }
    ]

    model = recover_architecture_model(nodes, edges, docs=docs)

    components = [entity for entity in model.entities if entity.kind == "component"]
    assert [entity.entity_id for entity in components] == ["component:session-service"]

    create_memberships = [
        membership.entity_id
        for membership in model.memberships_for("symbol:create_session")
        if membership.relationship_kind == "implements"
    ]
    load_memberships = [
        membership.entity_id
        for membership in model.memberships_for("symbol:load_session")
        if membership.relationship_kind == "implements"
    ]

    assert create_memberships == ["component:session-service"]
    assert load_memberships == ["component:session-service"]


def test_single_adapter_wrapper_symbol_does_not_become_component_by_default() -> None:
    nodes = [
        _symbol(
            node_id="symbol:github_oauth_client",
            name="GitHub OAuth Client",
            file_path="services/contextmine/api/github_oauth_client.py",
            container="api",
        ),
        {
            "id": "ext:github_oauth",
            "kind": "external_system",
            "name": "GitHub OAuth",
            "natural_key": "ext:github_oauth",
            "meta": {"file_path": "docs/integrations/github-oauth.md"},
        },
    ]
    edges = [_edge("symbol:github_oauth_client", "ext:github_oauth", "invokes")]

    model = recover_architecture_model(nodes, edges, docs=[])

    assert [entity for entity in model.entities if entity.kind == "component"] == []
    assert [
        membership
        for membership in model.memberships_for("symbol:github_oauth_client")
        if membership.relationship_kind == "implements"
    ] == []


def test_shared_component_can_belong_to_multiple_containers_and_projection_preserves_it() -> None:
    nodes = [
        _symbol(
            node_id="symbol:session_manager",
            name="Session Manager",
            file_path="packages/core/session_manager.py",
        ),
        _symbol(
            node_id="symbol:api_session_handler",
            name="API Session Handler",
            file_path="services/contextmine/api/session_handler.py",
            container="api",
            component="session-handler",
        ),
        _symbol(
            node_id="symbol:embedding_job_runner",
            name="Embedding Job Runner",
            file_path="services/contextmine/worker/jobs/embeddings.py",
            container="worker",
            component="embedding-job",
        ),
        {
            "id": "db:customer_sessions",
            "kind": "db_table",
            "name": "customer_sessions",
            "natural_key": "db:customer_sessions",
            "meta": {"file_path": "db/schema/customer_sessions.sql"},
        },
    ]
    edges = [
        _edge("symbol:api_session_handler", "symbol:session_manager", "symbol_calls_symbol"),
        _edge("symbol:embedding_job_runner", "symbol:session_manager", "symbol_calls_symbol"),
        _edge("symbol:session_manager", "db:customer_sessions", "reads_writes"),
    ]
    docs = [
        {
            "id": "doc:shared-core",
            "title": "Shared Session Management",
            "text": "Session Manager remains shared between API and worker runtimes.",
            "meta": {"file_path": "docs/adr/shared-session-management.md"},
        }
    ]

    model = recover_architecture_model(nodes, edges, docs=docs)

    component_memberships = [
        membership.entity_id
        for membership in model.memberships_for("symbol:session_manager")
        if membership.relationship_kind == "implements"
    ]
    container_memberships = [
        membership.entity_id
        for membership in model.memberships_for("symbol:session_manager")
        if membership.relationship_kind == "contained_in"
    ]

    assert component_memberships == ["component:session-manager"]
    assert container_memberships == ["container:api", "container:worker"]

    projection = build_inferred_architecture_projection(model, entity_level="component")
    node_ids = {node["id"] for node in projection["nodes"]}
    assert "component:session-manager@container:api" in node_ids
    assert "component:session-manager@container:worker" in node_ids


def test_component_boundary_ambiguity_is_preserved_when_two_clusters_are_equally_plausible() -> None:
    nodes = [
        _symbol(
            node_id="symbol:create_session",
            name="Create Session",
            file_path="services/contextmine/api/session_service.py",
            container="api",
            component="session-service",
        ),
        _symbol(
            node_id="symbol:refresh_session",
            name="Refresh Session",
            file_path="services/contextmine/api/session_service.py",
            container="api",
            component="session-service",
        ),
        _symbol(
            node_id="symbol:create_audit_log",
            name="Create Audit Log",
            file_path="services/contextmine/api/audit_service.py",
            container="api",
            component="audit-service",
        ),
        _symbol(
            node_id="symbol:load_audit_log",
            name="Load Audit Log",
            file_path="services/contextmine/api/audit_service.py",
            container="api",
            component="audit-service",
        ),
        _symbol(
            node_id="symbol:normalize_payload",
            name="Normalize Payload",
            file_path="packages/core/normalize_payload.py",
        ),
    ]
    edges = [
        _edge("symbol:create_session", "symbol:normalize_payload", "symbol_calls_symbol"),
        _edge("symbol:create_audit_log", "symbol:normalize_payload", "symbol_calls_symbol"),
    ]

    model = recover_architecture_model(nodes, edges, docs=[])

    hypothesis = next(
        row for row in model.hypotheses if row.subject_ref == "symbol:normalize_payload"
    )

    assert hypothesis.status == "ambiguous"
    assert hypothesis.candidate_entity_ids == (
        "component:audit-service",
        "component:session-service",
    )
    assert hypothesis.selected_entity_ids == ()
