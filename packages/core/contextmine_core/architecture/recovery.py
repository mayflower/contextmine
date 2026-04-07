"""Deterministic in-memory architecture recovery from local evidence."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import replace
from typing import Any
from urllib.parse import urlparse

from contextmine_core.twin.grouping import derive_arch_group

from .artifact_inventory import ArtifactInventoryEntry
from .artifact_parsers import ParsedArtifact, parse_artifact
from .recovery_decisions import recover_architecture_decisions
from .recovery_llm import apply_adjudication, build_adjudication_packet
from .recovery_model import (
    RecoveredArchitectureEntity,
    RecoveredArchitectureHypothesis,
    RecoveredArchitectureMembership,
    RecoveredArchitectureModel,
    RecoveredArchitectureRelationship,
)
from .schemas import EvidenceRef

EXPLICIT_METADATA_SCORE = 0.98
STRUCTURAL_EDGE_BASE_SCORE = 0.78
STRUCTURAL_EDGE_BONUS = 0.04
STRUCTURAL_EDGE_CAP = 0.86
RUNTIME_ENTRYPOINT_SCORE = 0.91
JOB_RUNTIME_SCORE = 0.88
SCHEDULER_RUNTIME_SCORE = 0.9
CLI_RUNTIME_SCORE = 0.84
PATH_HEURISTIC_SCORE = 0.45
SELECTION_THRESHOLD = 0.6
CLOSE_ENOUGH_DELTA = 0.06
COMPONENT_EXPLICIT_SCORE = 0.97
COMPONENT_FILE_CLUSTER_SCORE = 0.9
COMPONENT_SINGLE_SYMBOL_SCORE = 0.87
COMPONENT_STRUCTURAL_SCORE = 0.79
COMPONENT_STRUCTURAL_BONUS = 0.03
COMPONENT_THRESHOLD = 0.72
COMPONENT_CLOSE_ENOUGH_DELTA = 0.05

_ENTITY_CONFIDENCE = 0.9
_RELATIONSHIP_CONFIDENCE = 0.86
_ARTIFACT_ENTITY_CONFIDENCE = 0.88

_ORM_TABLE_PATTERNS = (
    re.compile(r"__tablename__\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE),
    re.compile(r"db_table\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE),
)
_URL_PATTERN = re.compile(r"https?://[^\s\"')>]+", re.IGNORECASE)
_CLIENT_LIBRARY_HINTS = {
    "openai": "OpenAI",
    "github": "GitHub",
    "stripe": "Stripe",
    "slack": "Slack",
    "sentry": "Sentry",
    "twilio": "Twilio",
}
_LOCAL_EXTERNAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0"}
_ADAPTER_TOKENS = ("adapter", "wrapper", "client", "proxy", "gateway")

_NODE_KIND_TO_ENTITY_KIND = {
    "db_table": "data_store",
    "message_schema": "message_channel",
    "external_system": "external_system",
}


def _slug(value: str) -> str:
    text = str(value or "").strip().lower()
    return "-".join(part for part in text.replace("_", "-").split() if part)


def _title_from_slug(value: str) -> str:
    return " ".join(part.capitalize() for part in value.split("-") if part)


def _container_display_name(container: str) -> str:
    if container == "api":
        return "API Runtime"
    if container == "worker":
        return "Worker Runtime"
    return f"{_title_from_slug(container)} Runtime"


def _component_display_name(component_slug: str) -> str:
    return _title_from_slug(component_slug)


def _node_meta(node: dict[str, Any]) -> dict[str, Any]:
    meta = node.get("meta")
    return meta if isinstance(meta, dict) else {}


def _node_kind(node: dict[str, Any]) -> str:
    kind = node.get("kind")
    return str(kind.value if hasattr(kind, "value") else kind or "").lower()


def _node_name(node: dict[str, Any]) -> str:
    return str(node.get("name") or node.get("natural_key") or node.get("id") or "").strip()


def _node_file_path(node: dict[str, Any]) -> str | None:
    file_path = _node_meta(node).get("file_path")
    if isinstance(file_path, str) and file_path.strip():
        return file_path.strip()
    return None


def _node_evidence(node: dict[str, Any]) -> tuple[EvidenceRef, ...]:
    refs: list[EvidenceRef] = []
    file_path = _node_file_path(node)
    if file_path:
        refs.append(EvidenceRef(kind="file", ref=file_path))
    node_id = str(node.get("id") or "").strip()
    if node_id:
        refs.append(EvidenceRef(kind="node", ref=node_id))
    return tuple(refs)


def _doc_meta(doc: dict[str, Any] | Any) -> dict[str, Any]:
    meta = doc.get("meta") if isinstance(doc, dict) else getattr(doc, "meta", None)
    return meta if isinstance(meta, dict) else {}


def _doc_value(doc: dict[str, Any] | Any, key: str) -> Any:
    if isinstance(doc, dict):
        return doc.get(key)
    return getattr(doc, key, None)


def _doc_id(doc: dict[str, Any] | Any) -> str:
    return str(_doc_value(doc, "id") or "").strip()


def _doc_text(doc: dict[str, Any] | Any) -> str:
    for key in ("text", "content_markdown", "summary"):
        value = _doc_value(doc, key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _doc_title(doc: dict[str, Any] | Any) -> str:
    for key in ("title", "name", "uri"):
        value = _doc_value(doc, key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return _doc_id(doc)


def _doc_file_path(doc: dict[str, Any] | Any) -> str | None:
    file_path = _doc_meta(doc).get("file_path")
    if isinstance(file_path, str) and file_path.strip():
        return file_path.strip()
    return None


def _doc_structured_data(doc: dict[str, Any] | Any) -> dict[str, Any]:
    data = _doc_value(doc, "structured_data")
    return data if isinstance(data, dict) else {}


def _doc_evidence(doc: dict[str, Any] | Any) -> tuple[EvidenceRef, ...]:
    refs: list[EvidenceRef] = []
    file_path = _doc_file_path(doc)
    if file_path:
        refs.append(EvidenceRef(kind="file", ref=file_path))
    doc_id = _doc_id(doc)
    if doc_id:
        refs.append(EvidenceRef(kind="artifact", ref=doc_id))
    return _dedupe_evidence(refs)


def _dedupe_evidence(
    evidence: list[EvidenceRef] | tuple[EvidenceRef, ...],
) -> tuple[EvidenceRef, ...]:
    seen: set[tuple[str, str, int | None, int | None]] = set()
    deduped: list[EvidenceRef] = []
    for ref in evidence:
        key = (ref.kind, ref.ref, ref.start_line, ref.end_line)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return tuple(deduped)


def _parser_hint_for_doc(doc: dict[str, Any] | Any) -> str:
    meta = _doc_meta(doc)
    for key in ("parser", "parser_hint"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    text = _doc_text(doc).lower()
    file_path = str(_doc_file_path(doc) or "").lower()
    if file_path.endswith(".sql"):
        return "sql"
    if "asyncapi:" in text:
        return "asyncapi"
    if "openapi:" in text or "swagger:" in text:
        return "openapi"
    if "apiversion:" in text and "\nkind:" in text:
        return "kubernetes_manifest"
    if file_path.endswith((".md", ".mdx")):
        return "markdown"
    if file_path.endswith(".rst"):
        return "rst"
    return "plain_text"


def _parsed_artifacts(docs: list[dict[str, Any]] | None) -> list[ParsedArtifact]:
    parsed: list[ParsedArtifact] = []
    for doc in docs or []:
        parser_name = _parser_hint_for_doc(doc)
        structured_data = _doc_structured_data(doc)
        evidence = _doc_evidence(doc)
        if structured_data:
            parsed.append(
                ParsedArtifact(
                    artifact_id=_doc_id(doc) or f"artifact:{_doc_file_path(doc) or 'unknown'}",
                    artifact_kind=str(_doc_meta(doc).get("artifact_kind") or "documentation"),
                    repo_path=str(_doc_file_path(doc) or _doc_title(doc) or "unknown"),
                    parser_name=parser_name,
                    confidence=_ARTIFACT_ENTITY_CONFIDENCE,
                    structured_data=structured_data,
                    evidence=evidence,
                )
            )
            continue

        artifact = ArtifactInventoryEntry(
            artifact_id=_doc_id(doc) or f"artifact:{_doc_file_path(doc) or 'unknown'}",
            artifact_kind=str(_doc_meta(doc).get("artifact_kind") or "documentation"),
            repo_path=str(_doc_file_path(doc) or _doc_title(doc) or "unknown"),
            media_type="text/plain",
            parser_hint=parser_name,
            raw_text=_doc_text(doc),
            raw_data=None,
            evidence=evidence,
        )
        parsed.append(parse_artifact(artifact))
    return parsed


def _merge_entity_candidate(
    store: dict[str, RecoveredArchitectureEntity],
    *,
    entity_id: str,
    kind: str,
    name: str,
    confidence: float,
    evidence: tuple[EvidenceRef, ...],
    attributes: dict[str, Any] | None = None,
) -> None:
    existing = store.get(entity_id)
    if existing is None:
        store[entity_id] = RecoveredArchitectureEntity(
            entity_id=entity_id,
            kind=kind,
            name=name,
            confidence=confidence,
            evidence=_dedupe_evidence(evidence),
            attributes=attributes or {},
        )
        return
    store[entity_id] = replace(
        existing,
        name=existing.name if len(existing.name) >= len(name) else name,
        confidence=max(existing.confidence, confidence),
        evidence=_dedupe_evidence(list(existing.evidence) + list(evidence)),
        attributes={**existing.attributes, **(attributes or {})},
    )


def _runtime_kind_from_text(value: str) -> str | None:
    lowered = str(value or "").strip().lower()
    if not lowered:
        return None
    if any(token in lowered for token in ("scheduler", "cron", "schedule")):
        return "scheduler"
    if any(token in lowered for token in ("cli", "command", "console")):
        return "cli"
    if any(token in lowered for token in ("worker", "consumer", "background", "queue")):
        return "worker"
    if any(token in lowered for token in ("job", "batch")):
        return "job"
    if any(token in lowered for token in ("api", "http", "web", "server", "gateway")):
        return "api"
    return None


def _looks_like_cli_path(file_path: str | None) -> bool:
    lowered = str(file_path or "").lower()
    return any(token in lowered for token in ("/cli", "_cli", "/bin/", "/cmd/", "console"))


def _title_from_identifier(value: str) -> str:
    if value in _CLIENT_LIBRARY_HINTS:
        return _CLIENT_LIBRARY_HINTS[value]
    return _title_from_slug(value)


def _is_external_host(hostname: str) -> bool:
    host = hostname.strip().lower()
    if not host or host in _LOCAL_EXTERNAL_HOSTS:
        return False
    return not (
        host.endswith(".svc")
        or host.endswith(".cluster.local")
        or host.endswith(".local")
        or host.endswith(".internal")
    )


def _external_system_from_url(url: str) -> tuple[str, str] | None:
    hostname = str(urlparse(url).hostname or "").strip().lower()
    if not _is_external_host(hostname):
        return None
    labels = [label for label in hostname.split(".") if label]
    if not labels:
        return None
    if labels[0] == "api" and len(labels) >= 2:
        slug = f"{labels[1]}-api"
        return slug, f"{_title_from_identifier(labels[1])} API"
    base = labels[-2] if len(labels) >= 2 else labels[0]
    slug = _slug(base)
    if not slug:
        return None
    return slug, _title_from_identifier(slug)


def _orm_table_names(text: str) -> list[str]:
    names: list[str] = []
    for pattern in _ORM_TABLE_PATTERNS:
        names.extend(match.strip() for match in pattern.findall(text or "") if match.strip())
    return sorted(set(names))


def _edge_ref(edge: dict[str, Any]) -> str:
    source = str(edge.get("source_node_id") or "")
    target = str(edge.get("target_node_id") or "")
    kind = str(edge.get("kind") or "edge")
    return f"{source}:{kind}:{target}"


def _edge_evidence(
    edge: dict[str, Any],
    source_node: dict[str, Any] | None,
    target_node: dict[str, Any] | None,
) -> tuple[EvidenceRef, ...]:
    refs = [EvidenceRef(kind="edge", ref=_edge_ref(edge))]
    if source_node is not None:
        refs.extend(_node_evidence(source_node))
    if target_node is not None:
        refs.extend(_node_evidence(target_node))
    return _dedupe_evidence(refs)


def _has_explicit_architecture(node: dict[str, Any]) -> bool:
    architecture_meta = _node_meta(node).get("architecture")
    if not isinstance(architecture_meta, dict):
        return False
    return bool(
        str(architecture_meta.get("domain") or "").strip()
        and str(architecture_meta.get("container") or "").strip()
    )


def _explicit_container_from_node(node: dict[str, Any]) -> str | None:
    if not _has_explicit_architecture(node):
        return None
    architecture_meta = _node_meta(node).get("architecture") or {}
    container = str(architecture_meta.get("container") or "").strip()
    return container or None


def _explicit_component_from_node(node: dict[str, Any]) -> str | None:
    architecture_meta = _node_meta(node).get("architecture")
    if not isinstance(architecture_meta, dict):
        return None
    component = str(architecture_meta.get("component") or "").strip()
    return _slug(component) or None


def _path_container_from_node(node: dict[str, Any]) -> str | None:
    file_path = _node_file_path(node)
    if not file_path or not (file_path.startswith("services/") or file_path.startswith("apps/")):
        return None
    group = derive_arch_group(file_path, {})
    if group is None:
        return None
    return group[1]


def _eligible_for_membership(node: dict[str, Any] | None) -> bool:
    if node is None:
        return False
    return _node_kind(node) in {"symbol", "file", "api_endpoint", "job"}


def _entity_id_for_node(node: dict[str, Any]) -> str | None:
    kind = _node_kind(node)
    if kind == "db_table":
        return f"data_store:{_node_name(node)}"
    if kind == "message_schema":
        return f"message_channel:{_slug(_node_name(node))}"
    if kind == "external_system":
        return f"external_system:{_slug(_node_name(node))}"
    return None


def _entity_kind_for_node(node: dict[str, Any]) -> str | None:
    kind = _node_kind(node)
    return _NODE_KIND_TO_ENTITY_KIND.get(kind)


def generate_runtime_candidates(
    nodes: list[dict[str, Any]],
    docs: list[dict[str, Any]] | None = None,
) -> list[RecoveredArchitectureEntity]:
    """Generate runtime/container entities from entrypoints, jobs, and deployment artifacts."""

    entities: dict[str, RecoveredArchitectureEntity] = {}
    for node in nodes:
        explicit_container = _explicit_container_from_node(node)
        evidence = _node_evidence(node)
        if explicit_container is not None:
            _merge_entity_candidate(
                entities,
                entity_id=f"container:{explicit_container}",
                kind="container",
                name=_container_display_name(explicit_container),
                confidence=EXPLICIT_METADATA_SCORE,
                evidence=evidence,
                attributes={"source": "explicit_metadata", "container": explicit_container},
            )
            continue

        kind = _node_kind(node)
        if kind == "api_endpoint":
            _merge_entity_candidate(
                entities,
                entity_id="container:api",
                kind="container",
                name=_container_display_name("api"),
                confidence=RUNTIME_ENTRYPOINT_SCORE,
                evidence=evidence,
                attributes={"source": "entrypoint", "container": "api"},
            )
            continue

        if kind == "job":
            job_meta = _node_meta(node)
            if job_meta.get("schedule"):
                _merge_entity_candidate(
                    entities,
                    entity_id="container:scheduler",
                    kind="container",
                    name=_container_display_name("scheduler"),
                    confidence=SCHEDULER_RUNTIME_SCORE,
                    evidence=evidence,
                    attributes={"source": "job_schedule", "container": "scheduler"},
                )
                continue
            _merge_entity_candidate(
                entities,
                entity_id="container:job",
                kind="container",
                name=_container_display_name("job"),
                confidence=JOB_RUNTIME_SCORE,
                evidence=evidence,
                attributes={"source": "job_definition", "container": "job"},
            )
            continue

        if kind in {"symbol", "file"} and _looks_like_cli_path(_node_file_path(node)):
            _merge_entity_candidate(
                entities,
                entity_id="container:cli",
                kind="container",
                name=_container_display_name("cli"),
                confidence=CLI_RUNTIME_SCORE,
                evidence=evidence,
                attributes={"source": "cli_entrypoint", "container": "cli"},
            )

    for artifact in _parsed_artifacts(docs):
        evidence = artifact.evidence
        if artifact.parser_name == "openapi":
            _merge_entity_candidate(
                entities,
                entity_id="container:api",
                kind="container",
                name=_container_display_name("api"),
                confidence=max(artifact.confidence, RUNTIME_ENTRYPOINT_SCORE),
                evidence=evidence,
                attributes={"source": "openapi", "container": "api"},
            )
            continue

        if artifact.parser_name not in {"kubernetes_manifest", "docker_compose", "helm_chart"}:
            continue

        structured = artifact.structured_data
        if structured.get("ports"):
            _merge_entity_candidate(
                entities,
                entity_id="container:api",
                kind="container",
                name=_container_display_name("api"),
                confidence=artifact.confidence,
                evidence=evidence,
                attributes={"source": artifact.parser_name, "container": "api"},
            )

        names: list[str] = []
        for key in ("deployables", "container_names", "images", "jobs"):
            values = structured.get(key)
            if isinstance(values, list):
                names.extend(str(value) for value in values if str(value).strip())

        for name in names:
            runtime = _runtime_kind_from_text(name)
            if runtime is None:
                continue
            _merge_entity_candidate(
                entities,
                entity_id=f"container:{runtime}",
                kind="container",
                name=_container_display_name(runtime),
                confidence=artifact.confidence,
                evidence=evidence,
                attributes={"source": artifact.parser_name, "container": runtime},
            )

    return [entities[key] for key in sorted(entities)]


def generate_data_store_candidates(
    nodes: list[dict[str, Any]],
    docs: list[dict[str, Any]] | None = None,
) -> list[RecoveredArchitectureEntity]:
    """Generate data-store entities from graph nodes and schema artifacts."""

    entities: dict[str, RecoveredArchitectureEntity] = {}
    for node in nodes:
        if _node_kind(node) != "db_table":
            continue
        name = _node_name(node)
        _merge_entity_candidate(
            entities,
            entity_id=f"data_store:{name}",
            kind="data_store",
            name=name,
            confidence=_ENTITY_CONFIDENCE,
            evidence=_node_evidence(node),
            attributes={"source_node_id": str(node.get("id") or "")},
        )

    for artifact in _parsed_artifacts(docs):
        structured = artifact.structured_data
        names: list[str] = []
        if artifact.parser_name == "sql":
            names.extend(
                str(name).strip() for name in structured.get("tables") or [] if str(name).strip()
            )
            names.extend(
                str(name).strip() for name in structured.get("views") or [] if str(name).strip()
            )
        else:
            names.extend(
                _orm_table_names(
                    _doc_text(
                        next(
                            (doc for doc in docs or [] if _doc_id(doc) == artifact.artifact_id), {}
                        )
                    )
                )
            )
        for name in sorted(set(names)):
            _merge_entity_candidate(
                entities,
                entity_id=f"data_store:{name}",
                kind="data_store",
                name=name,
                confidence=max(artifact.confidence, 0.84),
                evidence=artifact.evidence,
                attributes={"source": artifact.parser_name},
            )

    return [entities[key] for key in sorted(entities)]


def generate_message_channel_candidates(
    nodes: list[dict[str, Any]],
    docs: list[dict[str, Any]] | None = None,
) -> list[RecoveredArchitectureEntity]:
    """Generate message-channel entities from graph nodes and AsyncAPI artifacts."""

    entities: dict[str, RecoveredArchitectureEntity] = {}
    for node in nodes:
        if _node_kind(node) != "message_schema":
            continue
        slug = _slug(_node_name(node))
        _merge_entity_candidate(
            entities,
            entity_id=f"message_channel:{slug}",
            kind="message_channel",
            name=_node_name(node),
            confidence=_ENTITY_CONFIDENCE,
            evidence=_node_evidence(node),
            attributes={"source_node_id": str(node.get("id") or "")},
        )

    for artifact in _parsed_artifacts(docs):
        if artifact.parser_name != "asyncapi":
            continue
        channels = artifact.structured_data.get("channels") or []
        for channel in channels:
            slug = _slug(str(channel))
            if not slug:
                continue
            _merge_entity_candidate(
                entities,
                entity_id=f"message_channel:{slug}",
                kind="message_channel",
                name=str(channel),
                confidence=artifact.confidence,
                evidence=artifact.evidence,
                attributes={"source": "asyncapi"},
            )

    return [entities[key] for key in sorted(entities)]


def generate_external_system_candidates(
    nodes: list[dict[str, Any]],
    docs: list[dict[str, Any]] | None = None,
) -> list[RecoveredArchitectureEntity]:
    """Generate external-system entities from graph nodes, configs, clients, and API artifacts."""

    entities: dict[str, RecoveredArchitectureEntity] = {}
    for node in nodes:
        if _node_kind(node) != "external_system":
            continue
        slug = _slug(_node_name(node))
        _merge_entity_candidate(
            entities,
            entity_id=f"external_system:{slug}",
            kind="external_system",
            name=_node_name(node),
            confidence=_ENTITY_CONFIDENCE,
            evidence=_node_evidence(node),
            attributes={"source_node_id": str(node.get("id") or "")},
        )

    for doc in docs or []:
        text = _doc_text(doc)
        evidence = _doc_evidence(doc)
        structured = _doc_structured_data(doc)

        listed_external_systems = structured.get("external_systems")
        if isinstance(listed_external_systems, list):
            for item in listed_external_systems:
                slug = _slug(str(item))
                if not slug:
                    continue
                _merge_entity_candidate(
                    entities,
                    entity_id=f"external_system:{slug}",
                    kind="external_system",
                    name=str(item),
                    confidence=0.86,
                    evidence=evidence,
                    attributes={"source": "structured_data"},
                )

        for url in _URL_PATTERN.findall(text):
            resolved = _external_system_from_url(url)
            if resolved is None:
                continue
            slug, name = resolved
            _merge_entity_candidate(
                entities,
                entity_id=f"external_system:{slug}",
                kind="external_system",
                name=name,
                confidence=0.9,
                evidence=evidence,
                attributes={"source": "url_reference"},
            )

        lowered = text.lower()
        for provider, display_name in _CLIENT_LIBRARY_HINTS.items():
            if provider not in lowered:
                continue
            _merge_entity_candidate(
                entities,
                entity_id=f"external_system:{provider}",
                kind="external_system",
                name=display_name,
                confidence=0.84,
                evidence=evidence,
                attributes={"source": "client_library"},
            )

    for artifact in _parsed_artifacts(docs):
        if artifact.parser_name != "openapi":
            continue
        for url in artifact.structured_data.get("server_urls") or []:
            resolved = _external_system_from_url(str(url))
            if resolved is None:
                continue
            slug, name = resolved
            _merge_entity_candidate(
                entities,
                entity_id=f"external_system:{slug}",
                kind="external_system",
                name=name,
                confidence=artifact.confidence,
                evidence=artifact.evidence,
                attributes={"source": "openapi_server"},
            )

    return [entities[key] for key in sorted(entities)]


def _component_candidate_score(
    count: int,
) -> float:
    return min(
        COMPONENT_STRUCTURAL_SCORE + COMPONENT_STRUCTURAL_BONUS * max(count - 1, 0),
        STRUCTURAL_EDGE_CAP,
    )


def _symbol_interaction_targets(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    node_by_id = {str(node.get("id")): node for node in nodes}
    symbol_links: dict[str, set[str]] = defaultdict(set)
    resource_links: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        source_id = str(edge.get("source_node_id") or "")
        target_id = str(edge.get("target_node_id") or "")
        source_node = node_by_id.get(source_id)
        target_node = node_by_id.get(target_id)
        if source_node is None or target_node is None:
            continue
        source_kind = _node_kind(source_node)
        target_kind = _node_kind(target_node)
        if source_kind == "symbol" and target_kind == "symbol":
            symbol_links[source_id].add(target_id)
            symbol_links[target_id].add(source_id)
            continue
        if source_kind == "symbol" and target_kind in {
            "db_table",
            "message_schema",
            "external_system",
        }:
            resource_links[source_id].add(_entity_id_for_node(target_node) or target_id)
        if target_kind == "symbol" and source_kind in {
            "db_table",
            "message_schema",
            "external_system",
        }:
            resource_links[target_id].add(_entity_id_for_node(source_node) or source_id)
    return symbol_links, resource_links


def _docs_lower(docs: list[dict[str, Any]] | None) -> list[str]:
    return [f"{_doc_title(doc)}\n{_doc_text(doc)}".lower() for doc in docs or []]


def _doc_mentions_any(docs_lower: list[str], phrases: list[str]) -> bool:
    lowered = [phrase.strip().lower() for phrase in phrases if phrase and phrase.strip()]
    return any(phrase in doc_text for doc_text in docs_lower for phrase in lowered)


def _component_slug_for_node(node: dict[str, Any]) -> str:
    explicit = _explicit_component_from_node(node)
    if explicit is not None:
        return explicit
    file_path = _node_file_path(node)
    if file_path:
        stem = file_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        slug = _slug(stem)
        if slug:
            return slug
    return _slug(_node_name(node))


def _looks_like_adapter_symbol(node: dict[str, Any]) -> bool:
    file_path = (_node_file_path(node) or "").lower()
    name = _node_name(node).lower()
    return any(token in file_path or token in name for token in _ADAPTER_TOKENS)


def _component_candidate_key(component_id: str) -> str:
    return component_id.split(":", 1)[1] if ":" in component_id else component_id


def _add_component_candidate(
    candidate_map: dict[str, dict[str, dict[str, Any]]],
    *,
    subject_ref: str,
    component_id: str,
    score: float,
    evidence: list[EvidenceRef] | tuple[EvidenceRef, ...],
    signal: str,
) -> None:
    component_slug = _component_candidate_key(component_id)
    existing = candidate_map[subject_ref].get(component_slug)
    payload = {
        "score": score,
        "evidence": list(evidence),
        "signal": signal,
        "component_id": component_id,
    }
    if existing is None or score > float(existing["score"]):
        candidate_map[subject_ref][component_slug] = payload
        return
    if score == float(existing["score"]):
        existing["evidence"] = list(existing["evidence"]) + list(evidence)


def _select_component_candidates(
    candidate_map: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, tuple[tuple[str, ...], str]]:
    selections: dict[str, tuple[tuple[str, ...], str]] = {}
    for subject_ref, candidates in candidate_map.items():
        if not candidates:
            selections[subject_ref] = ((), "unresolved")
            continue
        ranked = sorted(candidates.items(), key=lambda item: (-float(item[1]["score"]), item[0]))
        top_score = float(ranked[0][1]["score"])
        selected = tuple(
            payload["component_id"]
            for _key, payload in ranked
            if float(payload["score"]) >= COMPONENT_THRESHOLD
            and (top_score - float(payload["score"])) <= COMPONENT_CLOSE_ENOUGH_DELTA
        )
        if not selected:
            selections[subject_ref] = ((), "unresolved")
        elif len(selected) == 1:
            selections[subject_ref] = (selected, "selected")
        else:
            selections[subject_ref] = ((), "ambiguous")
    return selections


def cluster_component_candidates(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    docs: list[dict[str, Any]] | None,
    memberships: list[RecoveredArchitectureMembership],
) -> tuple[
    list[RecoveredArchitectureEntity],
    list[RecoveredArchitectureMembership],
    list[RecoveredArchitectureHypothesis],
]:
    """Cluster related symbols into evidence-backed components."""

    node_by_id = {str(node.get("id")): node for node in nodes}
    symbol_nodes = {str(node.get("id")): node for node in nodes if _node_kind(node) == "symbol"}
    docs_lower = _docs_lower(docs)
    symbol_links, resource_links = _symbol_interaction_targets(nodes, edges)
    container_memberships_by_subject: dict[str, list[RecoveredArchitectureMembership]] = (
        defaultdict(list)
    )
    for membership in memberships:
        if membership.relationship_kind == "contained_in":
            container_memberships_by_subject[membership.subject_ref].append(membership)

    component_candidate_map: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    component_evidence_by_id: dict[str, list[EvidenceRef]] = defaultdict(list)

    symbols_by_file: dict[tuple[str, tuple[str, ...]], list[str]] = defaultdict(list)
    for subject_ref, node in symbol_nodes.items():
        file_path = _node_file_path(node)
        if not file_path:
            continue
        container_ids = tuple(
            membership.entity_id
            for membership in sorted(
                container_memberships_by_subject.get(subject_ref, []),
                key=lambda row: row.entity_id,
            )
        )
        symbols_by_file[(file_path, container_ids)].append(subject_ref)

    for subject_ref, node in symbol_nodes.items():
        explicit_component = _explicit_component_from_node(node)
        evidence = list(_node_evidence(node))
        if explicit_component is not None:
            component_id = f"component:{explicit_component}"
            _add_component_candidate(
                component_candidate_map,
                subject_ref=subject_ref,
                component_id=component_id,
                score=COMPONENT_EXPLICIT_SCORE,
                evidence=evidence,
                signal="explicit_component",
            )
            component_evidence_by_id[component_id].extend(evidence)

    for (file_path, _container_ids), subject_refs in symbols_by_file.items():
        if len(subject_refs) < 2:
            continue
        file_slug = _slug(file_path.rsplit("/", 1)[-1].rsplit(".", 1)[0])
        if not file_slug:
            continue
        call_connected = any(
            other in symbol_links.get(subject_ref, set())
            for idx, subject_ref in enumerate(subject_refs)
            for other in subject_refs[idx + 1 :]
        )
        shared_resources = (
            set.intersection(
                *[
                    set(resource_links.get(subject_ref, set()))
                    for subject_ref in subject_refs
                    if resource_links.get(subject_ref)
                ]
            )
            if all(resource_links.get(subject_ref) for subject_ref in subject_refs)
            else set()
        )
        symbol_names = [_node_name(symbol_nodes[subject_ref]) for subject_ref in subject_refs]
        doc_hint = _doc_mentions_any(
            docs_lower, [_component_display_name(file_slug), *symbol_names]
        )
        if not (call_connected or shared_resources or doc_hint):
            continue
        component_id = f"component:{file_slug}"
        shared_evidence: list[EvidenceRef] = []
        for subject_ref in subject_refs:
            shared_evidence.extend(_node_evidence(symbol_nodes[subject_ref]))
        for subject_ref in subject_refs:
            _add_component_candidate(
                component_candidate_map,
                subject_ref=subject_ref,
                component_id=component_id,
                score=COMPONENT_FILE_CLUSTER_SCORE,
                evidence=shared_evidence,
                signal="file_cluster",
            )
        component_evidence_by_id[component_id].extend(shared_evidence)

    for subject_ref, node in symbol_nodes.items():
        if component_candidate_map.get(subject_ref):
            continue
        resource_targets = resource_links.get(subject_ref, set())
        container_count = len(container_memberships_by_subject.get(subject_ref, []))
        doc_hint = _doc_mentions_any(
            docs_lower,
            [_node_name(node), _component_display_name(_component_slug_for_node(node))],
        )
        strong_signals = 0
        if container_count > 1:
            strong_signals += 1
        if any(
            target.startswith("data_store:") or target.startswith("message_channel:")
            for target in resource_targets
        ):
            strong_signals += 1
        if doc_hint:
            strong_signals += 1
        if not _looks_like_adapter_symbol(node) and strong_signals >= 2:
            component_id = f"component:{_component_slug_for_node(node)}"
            evidence = list(_node_evidence(node))
            for target in resource_targets:
                target_entity = next(
                    (entity for entity in nodes if _entity_id_for_node(entity) == target), None
                )
                if target_entity is not None:
                    evidence.extend(_node_evidence(target_entity))
            _add_component_candidate(
                component_candidate_map,
                subject_ref=subject_ref,
                component_id=component_id,
                score=COMPONENT_SINGLE_SYMBOL_SCORE,
                evidence=evidence,
                signal="single_symbol",
            )
            component_evidence_by_id[component_id].extend(evidence)

    selected_seed_components_by_subject: dict[str, set[str]] = defaultdict(set)
    for subject_ref, candidates in component_candidate_map.items():
        for payload in candidates.values():
            if float(payload["score"]) >= COMPONENT_THRESHOLD:
                selected_seed_components_by_subject[subject_ref].add(str(payload["component_id"]))

    structural_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    structural_evidence: dict[str, dict[str, list[EvidenceRef]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for edge in edges:
        source_id = str(edge.get("source_node_id") or "")
        target_id = str(edge.get("target_node_id") or "")
        source_node = node_by_id.get(source_id)
        target_node = node_by_id.get(target_id)
        if _node_kind(source_node or {}) != "symbol" or _node_kind(target_node or {}) != "symbol":
            continue
        evidence = list(_edge_evidence(edge, source_node, target_node))
        for component_id in selected_seed_components_by_subject.get(source_id, set()):
            if component_id not in selected_seed_components_by_subject.get(target_id, set()):
                structural_counts[target_id][component_id] += 1
                structural_evidence[target_id][component_id].extend(evidence)
        for component_id in selected_seed_components_by_subject.get(target_id, set()):
            if component_id not in selected_seed_components_by_subject.get(source_id, set()):
                structural_counts[source_id][component_id] += 1
                structural_evidence[source_id][component_id].extend(evidence)

    for subject_ref, candidate_counts in structural_counts.items():
        if component_candidate_map.get(subject_ref):
            continue
        for component_id, count in candidate_counts.items():
            _add_component_candidate(
                component_candidate_map,
                subject_ref=subject_ref,
                component_id=component_id,
                score=_component_candidate_score(count),
                evidence=structural_evidence[subject_ref][component_id],
                signal="structural_component",
            )
            component_evidence_by_id[component_id].extend(
                structural_evidence[subject_ref][component_id]
            )

    selections = _select_component_candidates(component_candidate_map)
    component_entities: dict[str, RecoveredArchitectureEntity] = {}
    component_memberships: list[RecoveredArchitectureMembership] = []
    component_hypotheses: list[RecoveredArchitectureHypothesis] = []

    for subject_ref, node in symbol_nodes.items():
        candidates = component_candidate_map.get(subject_ref, {})
        if not candidates:
            continue
        selected, status = selections.get(subject_ref, ((), "unresolved"))
        for component_id in selected:
            payload = next(
                payload
                for payload in candidates.values()
                if payload["component_id"] == component_id
            )
            component_memberships.append(
                RecoveredArchitectureMembership(
                    subject_ref=subject_ref,
                    entity_id=component_id,
                    relationship_kind="implements",
                    confidence=float(payload["score"]),
                    evidence=_dedupe_evidence(list(payload["evidence"])),
                    attributes={"signal": str(payload["signal"])},
                )
            )
            component_evidence_by_id[component_id].extend(payload["evidence"])
            _merge_entity_candidate(
                component_entities,
                entity_id=component_id,
                kind="component",
                name=_component_display_name(_component_candidate_key(component_id)),
                confidence=max(float(payload["score"]), _ENTITY_CONFIDENCE),
                evidence=_dedupe_evidence(component_evidence_by_id[component_id]),
                attributes={"component": _component_candidate_key(component_id)},
            )

        if status == "selected":
            continue
        if status == "ambiguous":
            rationale = (
                "Multiple component boundaries remain plausible from local cohesion signals."
            )
        else:
            rationale = "No component candidate cleared the threshold from local evidence."
        component_hypotheses.append(
            RecoveredArchitectureHypothesis(
                subject_ref=subject_ref,
                candidate_entity_ids=tuple(
                    sorted(str(payload["component_id"]) for payload in candidates.values())
                ),
                selected_entity_ids=(),
                rationale=rationale,
                status=status,
                confidence=max(float(payload["score"]) for payload in candidates.values()),
                evidence=_dedupe_evidence(
                    [ref for payload in candidates.values() for ref in list(payload["evidence"])]
                )
                or _node_evidence(node),
            )
        )

    return (
        [component_entities[key] for key in sorted(component_entities)],
        _dedupe_memberships(component_memberships),
        component_hypotheses,
    )


def _direct_candidate_map(
    nodes: list[dict[str, Any]],
    runtime_entities: list[RecoveredArchitectureEntity],
) -> dict[str, dict[str, dict[str, Any]]]:
    candidates: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    runtime_by_container = {
        entity.entity_id.split(":", 1)[1]: entity
        for entity in runtime_entities
        if entity.entity_id.startswith("container:")
    }
    for node in nodes:
        if not _eligible_for_membership(node):
            continue

        node_id = str(node.get("id") or "")
        evidence = list(_node_evidence(node))
        explicit_container = _explicit_container_from_node(node)
        if explicit_container is not None:
            runtime_entity = runtime_by_container.get(explicit_container)
            candidates[node_id][explicit_container] = {
                "score": EXPLICIT_METADATA_SCORE,
                "evidence": evidence + list(runtime_entity.evidence)
                if runtime_entity
                else evidence,
                "signal": "explicit",
            }

        kind = _node_kind(node)
        if kind == "api_endpoint" and "api" in runtime_by_container:
            runtime_entity = runtime_by_container["api"]
            candidates[node_id]["api"] = {
                "score": RUNTIME_ENTRYPOINT_SCORE,
                "evidence": evidence + list(runtime_entity.evidence),
                "signal": "entrypoint",
            }

        if kind == "job":
            job_meta = _node_meta(node)
            if job_meta.get("schedule") and "scheduler" in runtime_by_container:
                runtime_entity = runtime_by_container["scheduler"]
                candidates[node_id]["scheduler"] = {
                    "score": SCHEDULER_RUNTIME_SCORE,
                    "evidence": evidence + list(runtime_entity.evidence),
                    "signal": "job_schedule",
                }
            elif "job" in runtime_by_container:
                runtime_entity = runtime_by_container["job"]
                candidates[node_id]["job"] = {
                    "score": JOB_RUNTIME_SCORE,
                    "evidence": evidence + list(runtime_entity.evidence),
                    "signal": "job_definition",
                }
            elif "worker" in runtime_by_container:
                runtime_entity = runtime_by_container["worker"]
                candidates[node_id]["worker"] = {
                    "score": JOB_RUNTIME_SCORE,
                    "evidence": evidence + list(runtime_entity.evidence),
                    "signal": "job_definition",
                }

        if (
            kind in {"symbol", "file"}
            and _looks_like_cli_path(_node_file_path(node))
            and "cli" in runtime_by_container
        ):
            runtime_entity = runtime_by_container["cli"]
            candidates[node_id]["cli"] = {
                "score": CLI_RUNTIME_SCORE,
                "evidence": evidence + list(runtime_entity.evidence),
                "signal": "cli_entrypoint",
            }

        path_container = _path_container_from_node(node)
        if path_container is None or path_container not in runtime_by_container:
            continue
        existing = candidates[node_id].get(path_container)
        if existing is None or float(existing["score"]) < PATH_HEURISTIC_SCORE:
            candidates[node_id][path_container] = {
                "score": PATH_HEURISTIC_SCORE,
                "evidence": evidence + list(runtime_by_container[path_container].evidence),
                "signal": "path",
            }
    return candidates


def _select_candidates(
    candidate_map: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, tuple[tuple[str, ...], str]]:
    selections: dict[str, tuple[tuple[str, ...], str]] = {}
    for subject_ref, candidates in candidate_map.items():
        if not candidates:
            selections[subject_ref] = ((), "unresolved")
            continue

        ranked = sorted(
            candidates.items(),
            key=lambda item: (-float(item[1]["score"]), item[0]),
        )
        top_score = float(ranked[0][1]["score"])
        selected = tuple(
            container
            for container, payload in ranked
            if float(payload["score"]) >= SELECTION_THRESHOLD
            and (top_score - float(payload["score"])) <= CLOSE_ENOUGH_DELTA
        )

        if not selected:
            selections[subject_ref] = ((), "unresolved")
        elif len(selected) == 1:
            selections[subject_ref] = (selected, "selected")
        else:
            selections[subject_ref] = (selected, "ambiguous")
    return selections


def _dedupe_memberships(
    memberships: list[RecoveredArchitectureMembership],
) -> list[RecoveredArchitectureMembership]:
    deduped: dict[tuple[str, str, str], RecoveredArchitectureMembership] = {}
    for membership in memberships:
        key = (membership.subject_ref, membership.entity_id, membership.relationship_kind)
        existing = deduped.get(key)
        if existing is None or membership.confidence > existing.confidence:
            deduped[key] = membership
        elif membership.confidence == existing.confidence:
            deduped[key] = replace(
                existing,
                evidence=_dedupe_evidence(list(existing.evidence) + list(membership.evidence)),
            )
    return [deduped[key] for key in sorted(deduped)]


def _infer_memberships(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    runtime_entities: list[RecoveredArchitectureEntity],
) -> tuple[list[RecoveredArchitectureMembership], list[RecoveredArchitectureHypothesis]]:
    node_by_id = {str(node.get("id")): node for node in nodes}
    candidate_map = _direct_candidate_map(nodes, runtime_entities)
    initial_selections = _select_candidates(candidate_map)

    anchor_containers_by_subject = {
        subject_ref: selected
        for subject_ref, (selected, _status) in initial_selections.items()
        if selected
    }
    structural_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    structural_evidence: dict[str, dict[str, list[EvidenceRef]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for edge in edges:
        source_id = str(edge.get("source_node_id") or "")
        target_id = str(edge.get("target_node_id") or "")
        source_node = node_by_id.get(source_id)
        target_node = node_by_id.get(target_id)
        evidence = list(_edge_evidence(edge, source_node, target_node))

        for container in anchor_containers_by_subject.get(source_id, ()):
            if _eligible_for_membership(
                target_node
            ) and container not in anchor_containers_by_subject.get(target_id, ()):
                structural_counts[target_id][container] += 1
                structural_evidence[target_id][container].extend(evidence)

        for container in anchor_containers_by_subject.get(target_id, ()):
            if _eligible_for_membership(
                source_node
            ) and container not in anchor_containers_by_subject.get(source_id, ()):
                structural_counts[source_id][container] += 1
                structural_evidence[source_id][container].extend(evidence)

    for subject_ref, counts_by_container in structural_counts.items():
        subject_candidates = candidate_map.setdefault(subject_ref, {})
        for container, count in counts_by_container.items():
            score = min(
                STRUCTURAL_EDGE_BASE_SCORE + STRUCTURAL_EDGE_BONUS * max(count - 1, 0),
                STRUCTURAL_EDGE_CAP,
            )
            existing = subject_candidates.get(container)
            evidence = structural_evidence[subject_ref][container]
            if existing is None or score > float(existing["score"]):
                subject_candidates[container] = {
                    "score": score,
                    "evidence": evidence,
                    "signal": "structural",
                }
            elif score == float(existing["score"]):
                existing["evidence"] = list(existing["evidence"]) + evidence

    memberships: list[RecoveredArchitectureMembership] = []
    hypotheses: list[RecoveredArchitectureHypothesis] = []
    selections = _select_candidates(candidate_map)

    for node in nodes:
        subject_ref = str(node.get("id") or "")
        if not _eligible_for_membership(node):
            continue

        candidates = candidate_map.get(subject_ref, {})
        selected, status = selections.get(subject_ref, ((), "unresolved"))

        for container in selected:
            payload = candidates[container]
            memberships.append(
                RecoveredArchitectureMembership(
                    subject_ref=subject_ref,
                    entity_id=f"container:{container}",
                    relationship_kind="contained_in",
                    confidence=float(payload["score"]),
                    evidence=_dedupe_evidence(list(payload["evidence"])),
                    attributes={"signal": str(payload["signal"])},
                )
            )

        if status == "selected" and len(candidates) <= 1:
            continue

        if status == "selected":
            rationale = "Explicit architecture metadata outranks weaker heuristic evidence."
        elif status == "ambiguous":
            rationale = (
                "Multiple runtime candidates remain plausible after applying the scoring policy."
            )
        else:
            rationale = "No candidate cleared the selection threshold from local evidence."

        hypothesis_evidence = _dedupe_evidence(
            [ref for payload in candidates.values() for ref in list(payload["evidence"])]
        )
        hypotheses.append(
            RecoveredArchitectureHypothesis(
                subject_ref=subject_ref,
                candidate_entity_ids=tuple(
                    f"container:{container}" for container in sorted(candidates)
                ),
                selected_entity_ids=tuple(f"container:{container}" for container in selected),
                rationale=rationale,
                status=status,
                confidence=max(
                    (float(payload["score"]) for payload in candidates.values()), default=0.0
                ),
                evidence=hypothesis_evidence or _node_evidence(node),
            )
        )

    return _dedupe_memberships(memberships), hypotheses


def _collect_container_entities(
    nodes: list[dict[str, Any]],
    memberships: list[RecoveredArchitectureMembership],
    runtime_entities: list[RecoveredArchitectureEntity],
) -> list[RecoveredArchitectureEntity]:
    node_by_id = {str(node.get("id")): node for node in nodes}
    entities: dict[str, RecoveredArchitectureEntity] = {
        entity.entity_id: entity for entity in runtime_entities if entity.kind == "container"
    }
    evidence_by_container: dict[str, list[EvidenceRef]] = defaultdict(list)
    for membership in memberships:
        if not membership.entity_id.startswith("container:"):
            continue
        container = membership.entity_id.split(":", 1)[1]
        evidence_by_container[container].extend(membership.evidence)
        node = node_by_id.get(membership.subject_ref)
        if node is not None:
            evidence_by_container[container].extend(_node_evidence(node))

    for container, evidence in sorted(evidence_by_container.items()):
        _merge_entity_candidate(
            entities,
            entity_id=f"container:{container}",
            kind="container",
            name=_container_display_name(container),
            confidence=_ENTITY_CONFIDENCE,
            evidence=_dedupe_evidence(evidence),
            attributes={"container": container},
        )
    return [entities[key] for key in sorted(entities)]


def _merge_entities(
    *entity_groups: list[RecoveredArchitectureEntity],
) -> list[RecoveredArchitectureEntity]:
    merged: dict[str, RecoveredArchitectureEntity] = {}
    for group in entity_groups:
        for entity in group:
            _merge_entity_candidate(
                merged,
                entity_id=entity.entity_id,
                kind=entity.kind,
                name=entity.name,
                confidence=entity.confidence,
                evidence=entity.evidence,
                attributes=entity.attributes,
            )
    return [merged[key] for key in sorted(merged)]


def _collect_direct_entities(
    nodes: list[dict[str, Any]],
    component_subject_ids: set[str],
) -> list[RecoveredArchitectureEntity]:
    entities: dict[str, RecoveredArchitectureEntity] = {}
    for node in nodes:
        entity_id = _entity_id_for_node(node)
        entity_kind = _entity_kind_for_node(node)
        if entity_id is None or entity_kind is None:
            continue
        entities[entity_id] = RecoveredArchitectureEntity(
            entity_id=entity_id,
            kind=entity_kind,
            name=_node_name(node),
            confidence=_ENTITY_CONFIDENCE,
            evidence=_node_evidence(node),
            attributes={"source_node_id": str(node.get("id") or "")},
        )
    return [entities[key] for key in sorted(entities)]


def _relationship_sources_for_node(
    node: dict[str, Any],
    memberships_by_subject: dict[str, list[RecoveredArchitectureMembership]],
) -> list[str]:
    node_id = str(node.get("id") or "")
    direct_entity_id = _entity_id_for_node(node)
    if direct_entity_id is not None and _entity_kind_for_node(node) != "component":
        return [direct_entity_id]
    memberships = memberships_by_subject.get(node_id, [])
    if memberships:
        return [membership.entity_id for membership in memberships]
    if direct_entity_id is not None:
        return [direct_entity_id]
    return []


def _lift_relationships(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    memberships: list[RecoveredArchitectureMembership],
) -> list[RecoveredArchitectureRelationship]:
    node_by_id = {str(node.get("id")): node for node in nodes}
    memberships_by_subject: dict[str, list[RecoveredArchitectureMembership]] = defaultdict(list)
    for membership in memberships:
        memberships_by_subject[membership.subject_ref].append(membership)

    relationships: dict[tuple[str, str, str], RecoveredArchitectureRelationship] = {}
    for edge in edges:
        source_id = str(edge.get("source_node_id") or "")
        target_id = str(edge.get("target_node_id") or "")
        source_node = node_by_id.get(source_id)
        target_node = node_by_id.get(target_id)
        if source_node is None or target_node is None:
            continue

        source_entity_ids = _relationship_sources_for_node(source_node, memberships_by_subject)
        target_entity_ids = _relationship_sources_for_node(target_node, memberships_by_subject)
        if not source_entity_ids or not target_entity_ids:
            continue

        relation_kind = str(edge.get("kind") or "depends_on")
        evidence = _edge_evidence(edge, source_node, target_node)
        for source_entity_id in source_entity_ids:
            for target_entity_id in target_entity_ids:
                if source_entity_id == target_entity_id:
                    continue
                key = (source_entity_id, target_entity_id, relation_kind)
                existing = relationships.get(key)
                if existing is None:
                    relationships[key] = RecoveredArchitectureRelationship(
                        source_entity_id=source_entity_id,
                        target_entity_id=target_entity_id,
                        kind=relation_kind,
                        confidence=_RELATIONSHIP_CONFIDENCE,
                        evidence=evidence,
                    )
                    continue
                relationships[key] = replace(
                    existing,
                    evidence=_dedupe_evidence(list(existing.evidence) + list(evidence)),
                    confidence=max(existing.confidence, _RELATIONSHIP_CONFIDENCE),
                )

    return [relationships[key] for key in sorted(relationships)]


def _run_adjudicator(llm_adjudicator: Any, packet: dict[str, Any]) -> Any:
    if hasattr(llm_adjudicator, "adjudicate"):
        return llm_adjudicator.adjudicate(packet)
    if callable(llm_adjudicator):
        return llm_adjudicator(packet)
    raise TypeError("llm_adjudicator must be callable or expose adjudicate(packet).")


def recover_architecture_model(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    docs: list[dict[str, Any]] | None = None,
    llm_adjudicator: Any | None = None,
    llm_hypothesis_limit: int | None = None,
) -> RecoveredArchitectureModel:
    """Recover a deterministic architecture model from local graph facts only."""

    runtime_entities = generate_runtime_candidates(nodes, docs)
    generated_entities = _merge_entities(
        generate_data_store_candidates(nodes, docs),
        generate_message_channel_candidates(nodes, docs),
        generate_external_system_candidates(nodes, docs),
    )
    runtime_memberships, runtime_hypotheses = _infer_memberships(nodes, edges, runtime_entities)
    component_entities, component_memberships, component_hypotheses = cluster_component_candidates(
        nodes,
        edges,
        docs,
        runtime_memberships,
    )
    memberships = list(runtime_memberships) + list(component_memberships)
    hypotheses = list(runtime_hypotheses) + list(component_hypotheses)
    component_subject_ids = {
        membership.subject_ref
        for membership in component_memberships
        if membership.relationship_kind == "implements"
    }
    container_entities = _collect_container_entities(nodes, runtime_memberships, runtime_entities)
    direct_entities = _collect_direct_entities(nodes, component_subject_ids)
    relationships = _lift_relationships(nodes, edges, memberships)

    merged_entities = _merge_entities(
        container_entities,
        generated_entities,
        component_entities,
        direct_entities,
    )
    entity_by_id = {entity.entity_id: entity for entity in merged_entities}
    decisions = recover_architecture_decisions(
        docs,
        tuple(entity_by_id[key] for key in sorted(entity_by_id)),
    )

    model = RecoveredArchitectureModel(
        entities=tuple(entity_by_id[key] for key in sorted(entity_by_id)),
        relationships=tuple(relationships),
        memberships=tuple(memberships),
        hypotheses=tuple(hypotheses),
        decisions=decisions,
    )
    if llm_adjudicator is None:
        return model

    adjudication_candidates = [
        hypothesis
        for hypothesis in model.hypotheses
        if hypothesis.status in {"ambiguous", "unresolved"}
    ]
    if llm_hypothesis_limit is not None:
        limit = max(0, int(llm_hypothesis_limit))
        if limit < len(adjudication_candidates):
            skipped = len(adjudication_candidates) - limit
            model = replace(
                model,
                warnings=tuple(
                    list(model.warnings)
                    + [
                        "Skipped LLM adjudication for "
                        f"{skipped} architecture hypotheses due to llm_hypothesis_limit={limit}."
                    ]
                ),
            )
            adjudication_candidates = adjudication_candidates[:limit]

    for hypothesis in tuple(adjudication_candidates):
        if hypothesis.status not in {"ambiguous", "unresolved"}:
            continue
        packet = build_adjudication_packet(model=model, hypothesis=hypothesis)
        try:
            adjudication = _run_adjudicator(llm_adjudicator, packet)
        except Exception as exc:  # noqa: BLE001
            model = replace(
                model,
                warnings=tuple(
                    list(model.warnings)
                    + [f"Malformed adjudication for {hypothesis.subject_ref}: {exc}"]
                ),
            )
            continue
        model = apply_adjudication(
            model=model,
            hypothesis=hypothesis,
            packet=packet,
            adjudication=adjudication,
        )
    return model
