"""Tests for typed architecture artifact parsers."""

from __future__ import annotations

from contextmine_core.architecture.artifact_inventory import ArtifactInventoryEntry
from contextmine_core.architecture.artifact_parsers import (
    parse_artifact,
    parse_deployment_manifest,
    parse_markdown_adr,
    parse_openapi_spec,
    parse_sql_schema,
)
from contextmine_core.architecture.schemas import EvidenceRef


def _artifact(
    *,
    repo_path: str,
    parser_hint: str,
    raw_text: str,
    artifact_kind: str = "documentation",
) -> ArtifactInventoryEntry:
    return ArtifactInventoryEntry(
        artifact_id=f"artifact:{repo_path}",
        artifact_kind=artifact_kind,
        repo_path=repo_path,
        media_type="text/plain",
        parser_hint=parser_hint,
        raw_text=raw_text,
        raw_data=None,
        evidence=(EvidenceRef(kind="file", ref=repo_path),),
    )


def test_parse_markdown_adr_extracts_structured_sections_and_frontmatter() -> None:
    artifact = _artifact(
        repo_path="docs/records/0007.md",
        parser_hint="markdown",
        raw_text=(
            "---\n"
            "status: accepted\n"
            "affected_entity_ids:\n"
            "  - container:api\n"
            "  - container:worker\n"
            "supersedes: ADR-0004\n"
            "---\n"
            "# Async embedding workers\n\n"
            "## Context\nThe API path was blocking requests.\n\n"
            "## Decision\nMove embeddings generation to a worker runtime.\n\n"
            "## Consequences\nRequests stay responsive.\n\n"
            "## Alternatives\nKeep everything synchronous.\n"
        ),
    )

    parsed = parse_markdown_adr(artifact)

    assert parsed.parser_name == "markdown_adr"
    assert parsed.confidence >= 0.9
    assert parsed.structured_data["title"] == "Async embedding workers"
    assert parsed.structured_data["status"] == "accepted"
    assert parsed.structured_data["context"] == "The API path was blocking requests."
    assert parsed.structured_data["decision"] == "Move embeddings generation to a worker runtime."
    assert parsed.structured_data["consequences"] == "Requests stay responsive."
    assert parsed.structured_data["alternatives"] == "Keep everything synchronous."
    assert parsed.structured_data["supersedes"] == "ADR-0004"
    assert parsed.structured_data["affected_entity_ids"] == ["container:api", "container:worker"]


def test_parse_artifact_recognizes_adr_structure_without_typical_filename_tokens() -> None:
    artifact = _artifact(
        repo_path="docs/records/0008.md",
        parser_hint="markdown",
        raw_text=(
            "# Session runtime split\n\n"
            "## Context\nThe sync path couples API and embedding execution.\n\n"
            "## Decision\nUse a worker for embeddings.\n\n"
            "## Consequences\nAdds queueing complexity.\n"
        ),
    )

    parsed = parse_artifact(artifact)

    assert parsed.parser_name == "markdown_adr"
    assert parsed.structured_data["decision"] == "Use a worker for embeddings."


def test_parse_openapi_spec_extracts_services_servers_paths_and_operations() -> None:
    artifact = _artifact(
        repo_path="specs/openapi.yaml",
        parser_hint="openapi",
        artifact_kind="api_spec",
        raw_text=(
            "openapi: 3.0.0\n"
            "info:\n"
            "  title: Contextmine API\n"
            "servers:\n"
            "  - url: https://api.example.test\n"
            "paths:\n"
            "  /sessions:\n"
            "    post:\n"
            "      operationId: createSession\n"
            "  /sessions/{id}:\n"
            "    get:\n"
            "      operationId: getSession\n"
        ),
    )

    parsed = parse_openapi_spec(artifact)

    assert parsed.parser_name == "openapi"
    assert parsed.structured_data["service_name"] == "Contextmine API"
    assert parsed.structured_data["server_urls"] == ["https://api.example.test"]
    assert parsed.structured_data["paths"] == ["/sessions", "/sessions/{id}"]
    assert parsed.structured_data["operations"] == [
        "POST /sessions",
        "GET /sessions/{id}",
    ]


def test_parse_artifact_supports_asyncapi_channels_and_messages() -> None:
    artifact = _artifact(
        repo_path="specs/events.yaml",
        parser_hint="asyncapi",
        artifact_kind="api_spec",
        raw_text=(
            "asyncapi: 2.6.0\n"
            "info:\n"
            "  title: Contextmine Events\n"
            "channels:\n"
            "  embeddings.created:\n"
            "    publish:\n"
            "      message:\n"
            "        name: EmbeddingsCreated\n"
            "  sessions.updated:\n"
            "    subscribe:\n"
            "      message:\n"
            "        name: SessionUpdated\n"
        ),
    )

    parsed = parse_artifact(artifact)

    assert parsed.parser_name == "asyncapi"
    assert parsed.structured_data["service_name"] == "Contextmine Events"
    assert parsed.structured_data["channels"] == ["embeddings.created", "sessions.updated"]
    assert parsed.structured_data["message_names"] == ["EmbeddingsCreated", "SessionUpdated"]


def test_parse_deployment_manifest_extracts_deployables_images_ports_jobs_and_bindings() -> None:
    artifact = _artifact(
        repo_path="deploy/runtime.yaml",
        parser_hint="kubernetes_manifest",
        artifact_kind="deployment",
        raw_text=(
            "apiVersion: apps/v1\n"
            "kind: Deployment\n"
            "metadata:\n"
            "  name: api\n"
            "spec:\n"
            "  template:\n"
            "    spec:\n"
            "      containers:\n"
            "        - name: api\n"
            "          image: contextmine/api:latest\n"
            "          ports:\n"
            "            - containerPort: 8080\n"
            "---\n"
            "apiVersion: batch/v1\n"
            "kind: CronJob\n"
            "metadata:\n"
            "  name: embeddings-sync\n"
            "spec:\n"
            "  jobTemplate:\n"
            "    spec:\n"
            "      template:\n"
            "        spec:\n"
            "          containers:\n"
            "            - name: worker\n"
            "              image: contextmine/worker:latest\n"
            "---\n"
            "apiVersion: v1\n"
            "kind: Service\n"
            "metadata:\n"
            "  name: api\n"
            "spec:\n"
            "  selector:\n"
            "    app: api\n"
            "  ports:\n"
            "    - port: 80\n"
            "      targetPort: 8080\n"
        ),
    )

    parsed = parse_deployment_manifest(artifact)

    assert parsed.parser_name == "kubernetes_manifest"
    assert parsed.structured_data["deployables"] == ["api"]
    assert parsed.structured_data["container_names"] == ["api", "worker"]
    assert parsed.structured_data["images"] == ["contextmine/api:latest", "contextmine/worker:latest"]
    assert parsed.structured_data["ports"] == [80, 8080]
    assert parsed.structured_data["jobs"] == ["embeddings-sync"]
    assert parsed.structured_data["service_bindings"] == [{"service": "api", "target_port": 8080}]


def test_parse_artifact_supports_compose_and_helm_manifests() -> None:
    compose = parse_artifact(
        _artifact(
            repo_path="deploy/docker-compose.yml",
            parser_hint="docker_compose",
            artifact_kind="deployment",
            raw_text=(
                "services:\n"
                "  api:\n"
                "    image: contextmine/api:latest\n"
                "    ports:\n"
                "      - \"8080:8080\"\n"
                "  worker:\n"
                "    image: contextmine/worker:latest\n"
            ),
        )
    )
    helm = parse_artifact(
        _artifact(
            repo_path="charts/contextmine/Chart.yaml",
            parser_hint="helm_chart",
            artifact_kind="deployment",
            raw_text="apiVersion: v2\nname: contextmine\nversion: 0.1.0\n",
        )
    )

    assert compose.parser_name == "docker_compose"
    assert compose.structured_data["deployables"] == ["api", "worker"]
    assert helm.parser_name == "helm_chart"
    assert helm.structured_data["deployables"] == ["contextmine"]


def test_parse_sql_schema_extracts_tables_views_and_owner_hints() -> None:
    artifact = _artifact(
        repo_path="db/schema.sql",
        parser_hint="sql",
        artifact_kind="sql",
        raw_text=(
            "CREATE TABLE sessions(id bigint primary key, owner_id bigint);\n"
            "CREATE VIEW active_sessions AS SELECT * FROM sessions;\n"
            "ALTER TABLE sessions OWNER TO contextmine_app;\n"
        ),
    )

    parsed = parse_sql_schema(artifact)

    assert parsed.parser_name == "sql"
    assert parsed.structured_data["tables"] == ["sessions"]
    assert parsed.structured_data["views"] == ["active_sessions"]
    assert parsed.structured_data["owner_hints"] == [{"object_name": "sessions", "owner": "contextmine_app"}]


def test_parse_artifact_treats_mermaid_plantuml_and_c4dsl_as_typed_diagrams() -> None:
    mermaid = parse_artifact(
        _artifact(
            repo_path="docs/diagram.mmd",
            parser_hint="mermaid",
            artifact_kind="diagram",
            raw_text="graph TD\nA-->B\n",
        )
    )
    plantuml = parse_artifact(
        _artifact(
            repo_path="docs/diagram.puml",
            parser_hint="plantuml",
            artifact_kind="diagram",
            raw_text="@startuml\nA --> B\n@enduml\n",
        )
    )
    c4dsl = parse_artifact(
        _artifact(
            repo_path="architecture/context.c4.dsl",
            parser_hint="c4_dsl",
            artifact_kind="diagram",
            raw_text="workspace { model { softwareSystem = softwareSystem \"Contextmine\" } }",
        )
    )

    assert mermaid.parser_name == "mermaid"
    assert mermaid.structured_data["diagram_type"] == "mermaid"
    assert plantuml.parser_name == "plantuml"
    assert plantuml.structured_data["diagram_type"] == "plantuml"
    assert c4dsl.parser_name == "c4_dsl"
    assert c4dsl.structured_data["diagram_type"] == "c4_dsl"


def test_parse_artifact_uses_low_confidence_fallback_only_when_no_parser_matches() -> None:
    artifact = _artifact(
        repo_path="notes/random.log",
        parser_hint="plain_text",
        raw_text="just a loose note without structured sections",
    )

    parsed = parse_artifact(artifact)

    assert parsed.parser_name == "fallback_heuristic"
    assert parsed.confidence < 0.5
    assert parsed.structured_data["summary"] == "just a loose note without structured sections"

