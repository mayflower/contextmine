"""Tests for repo-wide architecture artifact inventory."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from contextmine_core.architecture.artifact_inventory import (
    build_document_artifact_inventory,
    build_repo_artifact_inventory,
    merge_document_and_repo_artifacts,
)
from contextmine_core.architecture.recovery import recover_architecture_model
from contextmine_core.architecture.recovery_docs import artifact_inventory_to_recovery_docs
from contextmine_core.models import Document

from .models.architecture_recovery_fixture import build_architecture_recovery_fixture


def _write(tmp_path: Path, repo_path: str, content: str) -> None:
    path = tmp_path / repo_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_repo_artifact_inventory_reads_repo_files_without_document_rows(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "docs/architecture/decision-record.md",
        "# Decision\n\nEmbeddings run in a worker runtime.\n",
    )

    artifacts = build_repo_artifact_inventory(
        repo_root=tmp_path,
        repo_paths=["docs/architecture/decision-record.md"],
    )

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.artifact_id
    assert artifact.artifact_kind == "documentation"
    assert artifact.repo_path == "docs/architecture/decision-record.md"
    assert artifact.media_type == "text/markdown"
    assert artifact.parser_hint == "markdown"
    assert artifact.raw_text.startswith("# Decision")
    assert artifact.evidence


def test_build_repo_artifact_inventory_classifies_supported_artifact_types(
    tmp_path: Path,
) -> None:
    samples = [
        ("docs/guide.md", "# Guide", "documentation", "markdown"),
        ("docs/guide.mdx", "# Guide", "documentation", "mdx"),
        ("docs/guide.rst", "Guide\n=====", "documentation", "rst"),
        ("docs/guide.txt", "guide", "documentation", "plain_text"),
        ("diagrams/system.puml", "@startuml\n@enduml\n", "diagram", "plantuml"),
        ("diagrams/system.mmd", "graph TD\nA-->B\n", "diagram", "mermaid"),
        ("architecture/context.c4.dsl", "workspace { model {} }", "diagram", "c4_dsl"),
        ("specs/openapi.yaml", "openapi: 3.0.0\npaths: {}\n", "api_spec", "openapi"),
        ("specs/asyncapi.yaml", "asyncapi: 2.6.0\nchannels: {}\n", "api_spec", "asyncapi"),
        ("db/schema.sql", "create table users(id int);", "sql", "sql"),
        (
            "deploy/k8s/deployment.yaml",
            "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: api\n",
            "deployment",
            "kubernetes_manifest",
        ),
        (
            "charts/contextmine/Chart.yaml",
            "apiVersion: v2\nname: contextmine\nversion: 0.1.0\n",
            "deployment",
            "helm_chart",
        ),
        (
            "deploy/docker-compose.yml",
            "services:\n  api:\n    image: app:latest\n",
            "deployment",
            "docker_compose",
        ),
        ("infra/main.tf", 'resource "aws_s3_bucket" "logs" {}', "infrastructure", "terraform"),
        (
            ".github/workflows/ci.yml",
            "name: CI\non: [push]\njobs:\n  test:\n    runs-on: ubuntu-latest\n",
            "ci_workflow",
            "github_actions_workflow",
        ),
    ]

    for repo_path, content, _artifact_kind, _parser_hint in samples:
        _write(tmp_path, repo_path, content)

    artifacts = build_repo_artifact_inventory(
        repo_root=tmp_path,
        repo_paths=[repo_path for repo_path, *_rest in samples],
    )

    by_path = {artifact.repo_path: artifact for artifact in artifacts}
    assert set(by_path) == {repo_path for repo_path, *_rest in samples}
    for repo_path, _content, artifact_kind, parser_hint in samples:
        artifact = by_path[repo_path]
        assert artifact.artifact_kind == artifact_kind
        assert artifact.parser_hint == parser_hint
        assert artifact.media_type
        assert artifact.raw_text


def test_merge_document_and_repo_artifacts_dedupes_on_repo_path_and_preserves_evidence(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "docs/adr/001-worker-runtime.md",
        "# Worker Runtime\n\nUse a background worker for embeddings.\n",
    )
    document = Document(
        id=uuid4(),
        source_id=uuid4(),
        uri="docs/adr/001-worker-runtime.md",
        title="ADR-001 worker runtime",
        content_markdown="# Worker Runtime\n\nUse a background worker for embeddings.\n",
        content_hash="hash",
        meta={"file_path": "docs/adr/001-worker-runtime.md"},
        last_seen_at=datetime.now(UTC),
    )

    repo_artifacts = build_repo_artifact_inventory(
        repo_root=tmp_path,
        repo_paths=["docs/adr/001-worker-runtime.md"],
    )
    document_artifacts = build_document_artifact_inventory([document])

    merged = merge_document_and_repo_artifacts(document_artifacts, repo_artifacts)

    assert len(merged) == 1
    artifact = merged[0]
    assert artifact.repo_path == "docs/adr/001-worker-runtime.md"
    assert {ref.kind for ref in artifact.evidence} >= {"artifact", "file"}


def test_artifact_inventory_can_feed_recovery_without_scenario_file_refs(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "docs/records/embedding-worker.md",
        (
            "---\n"
            "affected_entity_ids:\n"
            "  - container:api\n"
            "  - container:worker\n"
            "---\n"
            "# Worker runtime\n\n"
            "Embeddings generation runs in the worker runtime.\n"
        ),
    )
    fixture = build_architecture_recovery_fixture()

    repo_artifacts = build_repo_artifact_inventory(
        repo_root=tmp_path,
        repo_paths=["docs/records/embedding-worker.md"],
    )
    recovery_docs = artifact_inventory_to_recovery_docs(repo_artifacts)
    model = recover_architecture_model(
        fixture["nodes"],
        fixture["edges"],
        docs=recovery_docs,
    )

    assert len(model.decisions) == 1
    assert model.decisions[0].affected_entity_ids == ("container:api", "container:worker")
