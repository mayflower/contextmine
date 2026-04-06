"""Synthetic evidence graph for architecture recovery tests."""

from __future__ import annotations

from typing import Any


def build_architecture_recovery_fixture() -> dict[str, Any]:
    """Return a synthetic monorepo-like architecture evidence graph."""

    nodes = [
        {
            "id": "symbol:api_session_handler",
            "kind": "symbol",
            "name": "Api Session Handler",
            "natural_key": "symbol:api_session_handler",
            "meta": {
                "file_path": "services/contextmine/api/session_handler.py",
                "architecture": {
                    "domain": "contextmine",
                    "container": "api",
                    "component": "session-handler",
                },
            },
        },
        {
            "id": "symbol:event_publisher",
            "kind": "symbol",
            "name": "Event Publisher",
            "natural_key": "symbol:event_publisher",
            "meta": {
                "file_path": "services/contextmine/api/events.py",
                "architecture": {
                    "domain": "contextmine",
                    "container": "api",
                    "component": "event-publisher",
                },
            },
        },
        {
            "id": "symbol:embedding_job_runner",
            "kind": "symbol",
            "name": "Embedding Job Runner",
            "natural_key": "symbol:embedding_job_runner",
            "meta": {
                "file_path": "services/contextmine/worker/jobs/embeddings.py",
                "architecture": {
                    "domain": "contextmine",
                    "container": "worker",
                    "component": "embedding-job",
                },
            },
        },
        {
            "id": "symbol:session_manager",
            "kind": "symbol",
            "name": "Session Manager",
            "natural_key": "symbol:session_manager",
            "meta": {
                "file_path": "packages/core/session_manager.py",
            },
        },
        {
            "id": "api:endpoint:create_session",
            "kind": "api_endpoint",
            "name": "POST /sessions",
            "natural_key": "api:endpoint:create_session",
            "meta": {
                "path": "/sessions",
                "method": "POST",
                "file_path": "services/contextmine/api/routes.py",
                "architecture": {
                    "domain": "contextmine",
                    "container": "api",
                    "component": "public-api",
                },
            },
        },
        {
            "id": "job:embeddings_sync",
            "kind": "job",
            "name": "Embeddings Sync",
            "natural_key": "job:embeddings_sync",
            "meta": {
                "file_path": "services/contextmine/worker/jobs/embeddings.py",
                "schedule": "0 * * * *",
                "architecture": {
                    "domain": "contextmine",
                    "container": "worker",
                    "component": "embedding-job",
                },
            },
        },
        {
            "id": "db:customer_sessions",
            "kind": "db_table",
            "name": "customer_sessions",
            "natural_key": "db:customer_sessions",
            "meta": {
                "file_path": "db/schema/customer_sessions.sql",
            },
        },
        {
            "id": "msg:user_events",
            "kind": "message_schema",
            "name": "user-events",
            "natural_key": "msg:user_events",
            "meta": {
                "file_path": "schemas/user-events.avsc",
            },
        },
        {
            "id": "ext:github_oauth",
            "kind": "external_system",
            "name": "GitHub OAuth",
            "natural_key": "ext:github_oauth",
            "meta": {
                "provider": "github",
                "file_path": "docs/integrations/github-oauth.md",
            },
        },
        {
            "id": "ext:openai_embeddings",
            "kind": "external_system",
            "name": "OpenAI Embeddings",
            "natural_key": "ext:openai_embeddings",
            "meta": {
                "provider": "openai",
                "file_path": "docs/integrations/openai-embeddings.md",
            },
        },
    ]

    edges = [
        {
            "source_node_id": "api:endpoint:create_session",
            "target_node_id": "symbol:api_session_handler",
            "kind": "handled_by",
            "meta": {},
        },
        {
            "source_node_id": "symbol:api_session_handler",
            "target_node_id": "symbol:session_manager",
            "kind": "symbol_calls_symbol",
            "meta": {},
        },
        {
            "source_node_id": "symbol:session_manager",
            "target_node_id": "db:customer_sessions",
            "kind": "reads_writes",
            "meta": {},
        },
        {
            "source_node_id": "symbol:api_session_handler",
            "target_node_id": "ext:github_oauth",
            "kind": "invokes",
            "meta": {},
        },
        {
            "source_node_id": "symbol:api_session_handler",
            "target_node_id": "symbol:event_publisher",
            "kind": "symbol_calls_symbol",
            "meta": {},
        },
        {
            "source_node_id": "symbol:event_publisher",
            "target_node_id": "msg:user_events",
            "kind": "publishes_to",
            "meta": {},
        },
        {
            "source_node_id": "job:embeddings_sync",
            "target_node_id": "symbol:embedding_job_runner",
            "kind": "runs",
            "meta": {},
        },
        {
            "source_node_id": "symbol:embedding_job_runner",
            "target_node_id": "symbol:session_manager",
            "kind": "symbol_calls_symbol",
            "meta": {},
        },
        {
            "source_node_id": "symbol:embedding_job_runner",
            "target_node_id": "ext:openai_embeddings",
            "kind": "invokes",
            "meta": {},
        },
    ]

    docs = [
        {
            "id": "doc:adr:001",
            "kind": "adr",
            "title": "ADR-001 async embedding workers",
            "text": (
                "Embeddings generation runs in the worker runtime. "
                "Session Manager remains shared between API and worker."
            ),
            "meta": {
                "file_path": "docs/adr/001-async-embedding-workers.md",
                "affected_entity_ids": ["container:api", "container:worker"],
            },
        }
    ]

    return {"nodes": nodes, "edges": edges, "docs": docs}
