"""Typed parsers for architecture-relevant repo artifacts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

import yaml

from .artifact_inventory import ArtifactInventoryEntry
from .schemas import EvidenceRef

_SECTION_PATTERN = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_CREATE_TABLE_PATTERN = re.compile(r"\bcreate\s+table\s+([a-zA-Z_][\w.]*)", re.IGNORECASE)
_CREATE_VIEW_PATTERN = re.compile(r"\bcreate\s+view\s+([a-zA-Z_][\w.]*)", re.IGNORECASE)
_OWNER_PATTERN = re.compile(
    r"\balter\s+(?:table|view)\s+([a-zA-Z_][\w.]*)\s+owner\s+to\s+([a-zA-Z_][\w]*)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedArtifact:
    """Structured parsing result for an inventory entry."""

    artifact_id: str
    artifact_kind: str
    repo_path: str
    parser_name: str
    confidence: float
    structured_data: dict[str, Any]
    evidence: tuple[EvidenceRef, ...] = ()


def _frontmatter_and_body(raw_text: str) -> tuple[dict[str, Any], str]:
    text = str(raw_text or "")
    stripped = text.lstrip()
    if not stripped.startswith("---\n"):
        return {}, text

    lines = stripped.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    collected: list[str] = []
    end_index = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_index = index
            break
        collected.append(line)

    if end_index is None:
        return {}, text

    body = "\n".join(lines[end_index + 1 :])
    try:
        frontmatter = yaml.safe_load("\n".join(collected)) or {}
    except yaml.YAMLError:
        frontmatter = {}
    return frontmatter if isinstance(frontmatter, dict) else {}, body


def _markdown_title(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return None


def _markdown_sections(text: str) -> dict[str, str]:
    matches = list(_SECTION_PATTERN.finditer(text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        title = match.group(1).strip().lower()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            sections[title] = content
    return sections


def _fallback_result(artifact: ArtifactInventoryEntry) -> ParsedArtifact:
    return ParsedArtifact(
        artifact_id=artifact.artifact_id,
        artifact_kind=artifact.artifact_kind,
        repo_path=artifact.repo_path,
        parser_name="fallback_heuristic",
        confidence=0.2,
        structured_data={"summary": (artifact.raw_text or "").strip()},
        evidence=artifact.evidence,
    )


def parse_markdown_adr(artifact: ArtifactInventoryEntry) -> ParsedArtifact:
    """Parse a markdown-like ADR into structured sections."""

    frontmatter, body = _frontmatter_and_body(artifact.raw_text or "")
    sections = _markdown_sections(body)
    title = str(frontmatter.get("title") or _markdown_title(body) or PurePosixPath(artifact.repo_path).stem)
    status = frontmatter.get("status")
    supersedes = frontmatter.get("supersedes") or frontmatter.get("replaces")
    affected_entity_ids = frontmatter.get("affected_entity_ids") or []

    if not isinstance(affected_entity_ids, list):
        affected_entity_ids = []
    affected_entity_ids = [str(item).strip() for item in affected_entity_ids if str(item).strip()]

    structured = {
        "title": title.strip(),
        "status": str(status).strip().lower() if status is not None else None,
        "context": sections.get("context"),
        "decision": sections.get("decision"),
        "consequences": sections.get("consequences"),
        "alternatives": sections.get("alternatives"),
        "supersedes": str(supersedes).strip() if supersedes is not None else None,
        "replaces": str(frontmatter.get("replaces")).strip()
        if frontmatter.get("replaces") is not None
        else None,
        "affected_entity_ids": affected_entity_ids,
    }

    signals = sum(
        1
        for key in ("context", "decision", "consequences")
        if structured.get(key)
    )
    has_structured_frontmatter = bool(
        structured.get("status")
        or structured.get("supersedes")
        or structured.get("replaces")
        or structured.get("affected_entity_ids")
    )
    if structured.get("decision") and (signals >= 2 or has_structured_frontmatter):
        return ParsedArtifact(
            artifact_id=artifact.artifact_id,
            artifact_kind=artifact.artifact_kind,
            repo_path=artifact.repo_path,
            parser_name="markdown_adr",
            confidence=0.95,
            structured_data=structured,
            evidence=artifact.evidence,
        )
    return _fallback_result(artifact)


def parse_openapi_spec(artifact: ArtifactInventoryEntry) -> ParsedArtifact:
    """Parse an OpenAPI spec into a compact structured view."""

    try:
        payload = yaml.safe_load(artifact.raw_text or "") or {}
    except yaml.YAMLError:
        return _fallback_result(artifact)
    if not isinstance(payload, dict):
        return _fallback_result(artifact)

    title = payload.get("info", {}).get("title") if isinstance(payload.get("info"), dict) else None
    servers = payload.get("servers") if isinstance(payload.get("servers"), list) else []
    server_urls = [
        str(server.get("url")).strip()
        for server in servers
        if isinstance(server, dict) and str(server.get("url") or "").strip()
    ]
    paths = payload.get("paths") if isinstance(payload.get("paths"), dict) else {}
    operations: list[str] = []
    path_names: list[str] = []
    for path_name, path_item in paths.items():
        path_names.append(str(path_name))
        if not isinstance(path_item, dict):
            continue
        for method, _operation in path_item.items():
            if str(method).lower() not in {"get", "post", "put", "patch", "delete", "options", "head"}:
                continue
            operations.append(f"{str(method).upper()} {path_name}")

    return ParsedArtifact(
        artifact_id=artifact.artifact_id,
        artifact_kind=artifact.artifact_kind,
        repo_path=artifact.repo_path,
        parser_name="openapi",
        confidence=0.95,
        structured_data={
            "service_name": str(title or PurePosixPath(artifact.repo_path).stem),
            "server_urls": server_urls,
            "paths": path_names,
            "operations": operations,
        },
        evidence=artifact.evidence,
    )


def _parse_asyncapi_spec(artifact: ArtifactInventoryEntry) -> ParsedArtifact:
    try:
        payload = yaml.safe_load(artifact.raw_text or "") or {}
    except yaml.YAMLError:
        return _fallback_result(artifact)
    if not isinstance(payload, dict):
        return _fallback_result(artifact)

    title = payload.get("info", {}).get("title") if isinstance(payload.get("info"), dict) else None
    channels = payload.get("channels") if isinstance(payload.get("channels"), dict) else {}
    channel_names: list[str] = []
    message_names: list[str] = []
    for channel_name, channel_payload in channels.items():
        channel_names.append(str(channel_name))
        if not isinstance(channel_payload, dict):
            continue
        for operation_name in ("publish", "subscribe"):
            operation = channel_payload.get(operation_name)
            if not isinstance(operation, dict):
                continue
            message = operation.get("message")
            if isinstance(message, dict) and str(message.get("name") or "").strip():
                message_names.append(str(message.get("name")).strip())

    return ParsedArtifact(
        artifact_id=artifact.artifact_id,
        artifact_kind=artifact.artifact_kind,
        repo_path=artifact.repo_path,
        parser_name="asyncapi",
        confidence=0.95,
        structured_data={
            "service_name": str(title or PurePosixPath(artifact.repo_path).stem),
            "channels": channel_names,
            "message_names": message_names,
        },
        evidence=artifact.evidence,
    )


def parse_deployment_manifest(artifact: ArtifactInventoryEntry) -> ParsedArtifact:
    """Parse Kubernetes, Compose, or Helm deployment manifests."""

    parser_hint = artifact.parser_hint
    if parser_hint == "docker_compose":
        try:
            payload = yaml.safe_load(artifact.raw_text or "") or {}
        except yaml.YAMLError:
            return _fallback_result(artifact)
        if not isinstance(payload, dict):
            return _fallback_result(artifact)
        services = payload.get("services") if isinstance(payload.get("services"), dict) else {}
        deployables: list[str] = []
        images: list[str] = []
        ports: list[int] = []
        for service_name, service in services.items():
            deployables.append(str(service_name))
            if not isinstance(service, dict):
                continue
            image = str(service.get("image") or "").strip()
            if image:
                images.append(image)
            for mapping in service.get("ports") or []:
                if isinstance(mapping, str) and ":" in mapping:
                    _, container_port = mapping.split(":", 1)
                    container_port = container_port.strip().strip("\"'")
                    if container_port.isdigit():
                        ports.append(int(container_port))
        return ParsedArtifact(
            artifact_id=artifact.artifact_id,
            artifact_kind=artifact.artifact_kind,
            repo_path=artifact.repo_path,
            parser_name="docker_compose",
            confidence=0.92,
            structured_data={
                "deployables": deployables,
                "container_names": deployables,
                "images": images,
                "ports": sorted(set(ports)),
                "jobs": [],
                "service_bindings": [],
            },
            evidence=artifact.evidence,
        )

    if parser_hint == "helm_chart":
        try:
            payload = yaml.safe_load(artifact.raw_text or "") or {}
        except yaml.YAMLError:
            return _fallback_result(artifact)
        if not isinstance(payload, dict):
            return _fallback_result(artifact)
        name = str(payload.get("name") or PurePosixPath(artifact.repo_path).parent.name).strip()
        return ParsedArtifact(
            artifact_id=artifact.artifact_id,
            artifact_kind=artifact.artifact_kind,
            repo_path=artifact.repo_path,
            parser_name="helm_chart",
            confidence=0.9,
            structured_data={
                "deployables": [name] if name else [],
                "container_names": [],
                "images": [],
                "ports": [],
                "jobs": [],
                "service_bindings": [],
            },
            evidence=artifact.evidence,
        )

    try:
        payloads = list(yaml.safe_load_all(artifact.raw_text or ""))
    except yaml.YAMLError:
        return _fallback_result(artifact)

    deployables: list[str] = []
    container_names: list[str] = []
    images: list[str] = []
    ports: list[int] = []
    jobs: list[str] = []
    service_bindings: list[dict[str, Any]] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        kind = str(payload.get("kind") or "").strip()
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        name = str(metadata.get("name") or "").strip()
        lowered_kind = kind.lower()
        if lowered_kind in {"deployment", "statefulset", "daemonset"} and name:
            deployables.append(name)
        if lowered_kind in {"job", "cronjob"} and name:
            jobs.append(name)
        if lowered_kind == "service" and name:
            spec = payload.get("spec") if isinstance(payload.get("spec"), dict) else {}
            for port_row in spec.get("ports") or []:
                if not isinstance(port_row, dict):
                    continue
                port = port_row.get("port")
                target_port = port_row.get("targetPort")
                if isinstance(port, int):
                    ports.append(port)
                if isinstance(target_port, int):
                    ports.append(target_port)
                service_bindings.append(
                    {"service": name, "target_port": target_port if isinstance(target_port, int) else None}
                )
        template = payload.get("spec") if isinstance(payload.get("spec"), dict) else {}
        if lowered_kind == "cronjob":
            template = (
                template.get("jobTemplate", {}).get("spec", {}).get("template", {})
                if isinstance(template.get("jobTemplate"), dict)
                else {}
            )
        else:
            template = template.get("template", {}) if isinstance(template.get("template"), dict) else {}
        pod_spec = template.get("spec") if isinstance(template.get("spec"), dict) else {}
        for container in pod_spec.get("containers") or []:
            if not isinstance(container, dict):
                continue
            container_name = str(container.get("name") or "").strip()
            if container_name:
                container_names.append(container_name)
            image = str(container.get("image") or "").strip()
            if image:
                images.append(image)
            for port_row in container.get("ports") or []:
                if isinstance(port_row, dict) and isinstance(port_row.get("containerPort"), int):
                    ports.append(int(port_row["containerPort"]))

    return ParsedArtifact(
        artifact_id=artifact.artifact_id,
        artifact_kind=artifact.artifact_kind,
        repo_path=artifact.repo_path,
        parser_name="kubernetes_manifest",
        confidence=0.94,
        structured_data={
            "deployables": deployables,
            "container_names": sorted(set(container_names)),
            "images": sorted(set(images)),
            "ports": sorted(set(ports)),
            "jobs": jobs,
            "service_bindings": [binding for binding in service_bindings if binding["target_port"] is not None],
        },
        evidence=artifact.evidence,
    )


def parse_sql_schema(artifact: ArtifactInventoryEntry) -> ParsedArtifact:
    """Parse table/view declarations from SQL schema text."""

    raw_text = artifact.raw_text or ""
    tables = [match.split(".")[-1] for match in _CREATE_TABLE_PATTERN.findall(raw_text)]
    views = [match.split(".")[-1] for match in _CREATE_VIEW_PATTERN.findall(raw_text)]
    owner_hints = [
        {"object_name": object_name.split(".")[-1], "owner": owner}
        for object_name, owner in _OWNER_PATTERN.findall(raw_text)
    ]
    return ParsedArtifact(
        artifact_id=artifact.artifact_id,
        artifact_kind=artifact.artifact_kind,
        repo_path=artifact.repo_path,
        parser_name="sql",
        confidence=0.9 if tables or views else 0.4,
        structured_data={
            "tables": tables,
            "views": views,
            "owner_hints": owner_hints,
        },
        evidence=artifact.evidence,
    )


def _parse_diagram_artifact(artifact: ArtifactInventoryEntry) -> ParsedArtifact:
    return ParsedArtifact(
        artifact_id=artifact.artifact_id,
        artifact_kind=artifact.artifact_kind,
        repo_path=artifact.repo_path,
        parser_name=artifact.parser_hint,
        confidence=0.92,
        structured_data={"diagram_type": artifact.parser_hint},
        evidence=artifact.evidence,
    )


def parse_artifact(artifact: ArtifactInventoryEntry) -> ParsedArtifact:
    """Dispatch to the smallest matching typed parser for one inventory entry."""

    parser_hint = artifact.parser_hint
    if parser_hint in {"markdown", "mdx", "rst", "plain_text"}:
        parsed = parse_markdown_adr(artifact)
        if parsed.parser_name != "fallback_heuristic":
            return parsed
        return _fallback_result(artifact)
    if parser_hint == "openapi":
        return parse_openapi_spec(artifact)
    if parser_hint == "asyncapi":
        return _parse_asyncapi_spec(artifact)
    if parser_hint in {"kubernetes_manifest", "docker_compose", "helm_chart"}:
        return parse_deployment_manifest(artifact)
    if parser_hint == "sql":
        return parse_sql_schema(artifact)
    if parser_hint in {"mermaid", "plantuml", "c4_dsl"}:
        return _parse_diagram_artifact(artifact)
    return _fallback_result(artifact)
