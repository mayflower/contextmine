"""End-to-end model-free repository sync gate.

This runs inside the production worker image against a fresh pg4ai database and
a real Prefect server. Only the remote GitHub URL is redirected to a read-only,
pinned local clone so that upstream branch movement cannot change the fixture.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from http.client import HTTPConnection
from pathlib import Path
from typing import Any
from unittest.mock import patch

import prefect
from contextmine_core import (
    Chunk,
    Collection,
    CollectionVisibility,
    Document,
    Source,
    SourceType,
    Symbol,
    TwinNode,
    TwinScenario,
    User,
    get_session,
    get_settings,
    hybrid_search,
)
from contextmine_core.lsp import shutdown_lsp_manager
from contextmine_core.models import KnowledgeNode
from contextmine_worker import flows
from git import Repo
from sqlalchemy import func, select

FIXTURE_OWNER = "fastapi"
FIXTURE_REPOSITORY = "full-stack-fastapi-template"
FIXTURE_BRANCH = "contextmine-smoke"
FIXTURE_URL = f"https://github.com/{FIXTURE_OWNER}/{FIXTURE_REPOSITORY}"
EXPECTED_DOCUMENT_SUFFIXES = {
    "README.md",
    "backend/app/main.py",
    "backend/app/api/routes/items.py",
    "frontend/src/main.tsx",
    "frontend/src/components/Items/AddItem.tsx",
    "compose.yml",
}


def _require_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


FIXTURE_PATH = Path(_require_environment("CONTEXTMINE_SMOKE_FIXTURE_DIR")).resolve()
FIXTURE_REVISION = _require_environment("CONTEXTMINE_SMOKE_FIXTURE_REVISION")
PREFECT_SERVER_VERSION = _require_environment("CONTEXTMINE_SMOKE_PREFECT_SERVER_VERSION")


def _assert_fixture() -> None:
    repository = Repo(FIXTURE_PATH)
    actual_revision = repository.head.commit.hexsha
    if actual_revision != FIXTURE_REVISION:
        raise AssertionError(
            f"Fixture revision mismatch: expected {FIXTURE_REVISION}, got {actual_revision}"
        )
    if repository.is_dirty(untracked_files=True):
        raise AssertionError("Pinned fixture must be clean")


def _assert_api_health() -> None:
    connection = HTTPConnection("api", 8000, timeout=10)
    try:
        connection.request("GET", "/api/health")
        response = connection.getresponse()
        if response.status != 200:
            raise AssertionError(f"API health returned HTTP {response.status}")
        payload = json.loads(response.read())
    finally:
        connection.close()
    if payload != {"status": "ok"}:
        raise AssertionError(f"Unexpected API health response: {payload}")


def _assert_prefect_versions() -> str:
    client_version = prefect.__version__
    if client_version != PREFECT_SERVER_VERSION:
        raise AssertionError(
            "Prefect client/server version mismatch: "
            f"client={client_version}, server={PREFECT_SERVER_VERSION}"
        )
    return client_version


def _forbid_model_provider(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("A model provider was initialized while MODEL_CALLS_ENABLED=false")


def _clone_pinned_fixture(
    repo_path: Path,
    clone_url: str,
    branch: str | None = None,
    token: str | None = None,
    ssh_private_key: str | None = None,
) -> Repo:
    expected_url = f"{FIXTURE_URL}.git"
    if clone_url != expected_url:
        raise AssertionError(f"Unexpected clone URL: {clone_url}")
    if token or ssh_private_key:
        raise AssertionError("The public smoke fixture must not require credentials")

    repository = _ORIGINAL_CLONE_OR_PULL(
        repo_path,
        str(FIXTURE_PATH),
        FIXTURE_BRANCH,
    )
    actual_revision = repository.head.commit.hexsha
    if actual_revision != FIXTURE_REVISION:
        raise AssertionError(
            f"Indexed revision mismatch: expected {FIXTURE_REVISION}, got {actual_revision}"
        )
    return repository


_ORIGINAL_CLONE_OR_PULL = flows.clone_or_pull_repo


async def _seed_source() -> tuple[uuid.UUID, uuid.UUID]:
    user_id = uuid.uuid4()
    collection_id = uuid.uuid4()
    source_id = uuid.uuid4()

    async with get_session() as session:
        session.add(
            User(
                id=user_id,
                github_user_id=9_000_000_001,
                github_login="contextmine-smoke",
                name="ContextMine Smoke",
            )
        )
        session.add(
            Collection(
                id=collection_id,
                slug="model-free-system-smoke",
                name="Model-free system smoke",
                visibility=CollectionVisibility.GLOBAL,
                owner_user_id=user_id,
                config={},
            )
        )
        session.add(
            Source(
                id=source_id,
                collection_id=collection_id,
                type=SourceType.GITHUB,
                url=FIXTURE_URL,
                config={
                    "owner": FIXTURE_OWNER,
                    "repo": FIXTURE_REPOSITORY,
                    "branch": FIXTURE_BRANCH,
                },
                enabled=True,
                schedule_interval_minutes=1440,
            )
        )

    return collection_id, source_id


async def _database_snapshot(collection_id: uuid.UUID, source_id: uuid.UUID) -> dict[str, Any]:
    async with get_session() as session:
        document_count = await session.scalar(
            select(func.count(Document.id)).where(Document.source_id == source_id)
        )
        chunk_count = await session.scalar(
            select(func.count(Chunk.id))
            .join(Document, Chunk.document_id == Document.id)
            .where(Document.source_id == source_id)
        )
        symbol_count = await session.scalar(
            select(func.count(Symbol.id))
            .join(Document, Symbol.document_id == Document.id)
            .where(Document.source_id == source_id)
        )
        knowledge_node_count = await session.scalar(
            select(func.count(KnowledgeNode.id)).where(KnowledgeNode.collection_id == collection_id)
        )
        twin_node_count = await session.scalar(
            select(func.count(TwinNode.id)).where(
                TwinNode.scenario_id.in_(
                    select(TwinScenario.id).where(TwinScenario.collection_id == collection_id)
                ),
                TwinNode.is_active.is_(True),
            )
        )
        source = await session.scalar(select(Source).where(Source.id == source_id))
        uris = set(
            (await session.execute(select(Document.uri).where(Document.source_id == source_id)))
            .scalars()
            .all()
        )

    return {
        "documents": int(document_count or 0),
        "chunks": int(chunk_count or 0),
        "symbols": int(symbol_count or 0),
        "knowledge_nodes": int(knowledge_node_count or 0),
        "twin_nodes": int(twin_node_count or 0),
        "cursor": source.cursor if source else None,
        "uris": uris,
    }


def _assert_first_sync(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("status") != "success":
        raise AssertionError(f"Initial sync did not succeed: {result}")

    stats = dict(result.get("stats") or {})
    expected_equal = {
        "commit_sha": FIXTURE_REVISION,
        "docs_processing_failures": 0,
        "docs_processing_timeouts": 0,
        "docs_chunk_deferred": 0,
        "embedding_tokens_used": 0,
        "chunks_embedded": 0,
        "metrics_gate": "pass",
        "scip_degraded": False,
    }
    for key, expected in expected_equal.items():
        actual = stats.get(key)
        if actual != expected:
            raise AssertionError(f"Unexpected {key}: expected {expected!r}, got {actual!r}")

    expected_counts = {
        "files_indexed": 210,
        "docs_created": 210,
        "chunks_created": 691,
        "symbols_created": 553,
        "scip_projects_detected": 4,
        "scip_projects_indexed": 4,
        "scip_symbols": 2652,
        "scip_relations": 5998,
        "structural_metric_files": 124,
        "metrics_requested_files": 124,
        "kg_file_nodes": 210,
        "kg_symbol_nodes": 553,
        "twin_nodes_upserted": 6391,
    }
    for key, expected in expected_counts.items():
        actual = int(stats.get(key, 0) or 0)
        if actual != expected:
            raise AssertionError(f"Unexpected {key}: expected {expected}, got {actual}")

    detected_languages = set(stats.get("scip_languages_detected") or [])
    if not {"python", "typescript"}.issubset(detected_languages):
        raise AssertionError(f"Expected Python and TypeScript, got {sorted(detected_languages)}")
    if stats.get("kg_errors"):
        raise AssertionError(f"Knowledge graph errors: {stats['kg_errors']}")

    return stats


def _assert_database_snapshot(snapshot: dict[str, Any]) -> None:
    expected_counts = {
        "documents": 210,
        "chunks": 691,
        "symbols": 553,
        "knowledge_nodes": 4665,
        "twin_nodes": 4621,
    }
    for key, expected in expected_counts.items():
        actual = int(snapshot[key])
        if actual != expected:
            raise AssertionError(f"Unexpected persisted {key}: expected {expected}, got {actual}")
    if snapshot["cursor"] != FIXTURE_REVISION:
        raise AssertionError(f"Expected source cursor {FIXTURE_REVISION}, got {snapshot['cursor']}")

    uris = set(snapshot["uris"])
    missing = {
        suffix
        for suffix in EXPECTED_DOCUMENT_SUFFIXES
        if not any(uri.split("?", 1)[0].endswith(f"/{suffix}") for uri in uris)
    }
    if missing:
        raise AssertionError(f"Expected fixture documents are missing: {sorted(missing)}")


def _assert_noop_sync(result: dict[str, Any]) -> None:
    if result.get("status") != "success":
        raise AssertionError(f"No-op sync did not succeed: {result}")
    stats = dict(result.get("stats") or {})
    expected_zero = (
        "docs_created",
        "docs_updated",
        "docs_deleted",
        "chunks_created",
        "chunks_deleted",
        "symbols_created",
        "symbols_deleted",
        "embedding_tokens_used",
    )
    for key in expected_zero:
        if int(stats.get(key, 0) or 0) != 0:
            raise AssertionError(f"No-op sync changed {key}: {stats.get(key)}")
    if stats.get("commit_sha") != FIXTURE_REVISION:
        raise AssertionError(f"No-op sync moved revision: {stats.get('commit_sha')}")


async def _run() -> None:
    _assert_fixture()
    _assert_api_health()
    prefect_client_version = _assert_prefect_versions()
    settings = get_settings()
    if settings.model_calls_enabled:
        raise AssertionError("MODEL_CALLS_ENABLED must be false for this gate")

    collection_id, source_id = await _seed_source()
    provider_patches = (
        patch.object(flows, "clone_or_pull_repo", side_effect=_clone_pinned_fixture),
        patch.object(flows, "get_embedder", side_effect=_forbid_model_provider),
        patch(
            "contextmine_core.research.llm.get_llm_provider",
            side_effect=_forbid_model_provider,
        ),
        patch(
            "contextmine_core.research.llm.get_research_llm_provider",
            side_effect=_forbid_model_provider,
        ),
    )

    with provider_patches[0], provider_patches[1], provider_patches[2], provider_patches[3]:
        first_result = await flows.sync_single_source(str(source_id), FIXTURE_URL)
        first_stats = _assert_first_sync(first_result)
        first_snapshot = await _database_snapshot(collection_id, source_id)
        _assert_database_snapshot(first_snapshot)

        search_response = await hybrid_search(
            query="FastAPI",
            query_embedding=None,
            user_id=None,
            collection_id=collection_id,
            top_k=10,
        )
        if not search_response.results:
            raise AssertionError("Model-free full-text search returned no results")
        if search_response.total_vector_matches != 0:
            raise AssertionError("Model-free search unexpectedly returned vector matches")

        second_result = await flows.sync_single_source(str(source_id), FIXTURE_URL)
        _assert_noop_sync(second_result)
        second_snapshot = await _database_snapshot(collection_id, source_id)

    if {key: value for key, value in first_snapshot.items() if key != "uris"} != {
        key: value for key, value in second_snapshot.items() if key != "uris"
    }:
        raise AssertionError("No-op sync changed persisted object counts")

    summary = {
        "fixture": f"{FIXTURE_OWNER}/{FIXTURE_REPOSITORY}@{FIXTURE_REVISION}",
        "model_calls_enabled": False,
        "prefect": {
            "client": prefect_client_version,
            "server": PREFECT_SERVER_VERSION,
        },
        "first_sync": {
            key: first_stats.get(key)
            for key in (
                "files_indexed",
                "docs_created",
                "chunks_created",
                "symbols_created",
                "scip_projects_indexed",
                "scip_symbols",
                "scip_relations",
                "structural_metric_files",
                "metrics_gate",
                "kg_file_nodes",
                "kg_symbol_nodes",
                "twin_nodes_upserted",
            )
        },
        "persisted": {key: value for key, value in first_snapshot.items() if key != "uris"},
        "search_results": len(search_response.results),
        "noop_sync": "pass",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


async def main() -> None:
    """Run the gate and close any language server started by traceability."""
    try:
        await _run()
    finally:
        await shutdown_lsp_manager()


if __name__ == "__main__":
    asyncio.run(main())
