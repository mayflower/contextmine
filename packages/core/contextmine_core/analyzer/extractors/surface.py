"""System Surface Catalog builder.

Integrates all spec-driven extractors (OpenAPI, GraphQL, Protobuf, Jobs)
and builds knowledge graph nodes and edges.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

import yaml
from contextmine_core.analyzer.extractors.graphql import (
    GraphQLExtraction,
    extract_from_graphql,
)
from contextmine_core.analyzer.extractors.jobs import JobsExtraction, extract_jobs
from contextmine_core.analyzer.extractors.openapi import (
    EndpointDef,
    OpenAPIExtraction,
    extract_from_openapi_document,
)
from contextmine_core.analyzer.extractors.protobuf import (
    ProtobufExtraction,
    extract_from_protobuf,
)
from contextmine_core.analyzer.extractors.traceability import symbol_token_variants
from contextmine_core.models import (
    KnowledgeEdge,
    KnowledgeEdgeKind,
    KnowledgeEvidence,
    KnowledgeNode,
    KnowledgeNodeEvidence,
    KnowledgeNodeKind,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass
class SurfaceCatalog:
    """Consolidated system surface catalog from multiple sources."""

    openapi_specs: list[OpenAPIExtraction] = field(default_factory=list)
    graphql_schemas: list[GraphQLExtraction] = field(default_factory=list)
    protobuf_files: list[ProtobufExtraction] = field(default_factory=list)
    job_definitions: list[JobsExtraction] = field(default_factory=list)


@dataclass(frozen=True)
class _SymbolCandidate:
    node_id: UUID
    name: str
    natural_key: str


def _provenance(*, mode: str, extractor: str, confidence: float) -> dict[str, Any]:
    return {
        "provenance": {
            "mode": mode,
            "extractor": extractor,
            "confidence": round(max(0.0, min(confidence, 1.0)), 4),
            "evidence_ids": [],
        }
    }


class SurfaceCatalogExtractor:
    """Extracts system surface definitions from various spec files."""

    def __init__(self) -> None:
        self.catalog = SurfaceCatalog()

    def add_file(self, file_path: str, content: str) -> bool:
        """Add a file to the catalog if it's a recognized spec type.

        Args:
            file_path: Path to the file
            content: File content

        Returns:
            True if the file was processed, False otherwise
        """
        # OpenAPI/Swagger specs
        openapi_doc = self._load_openapi_document(file_path, content)
        if openapi_doc is not None:
            extraction = extract_from_openapi_document(file_path, openapi_doc)
            if extraction.endpoints:
                self.catalog.openapi_specs.append(extraction)
                return True

        # GraphQL schemas
        if file_path.endswith((".graphql", ".gql")):
            extraction = extract_from_graphql(file_path, content)
            if extraction.types:
                self.catalog.graphql_schemas.append(extraction)
                return True

        # Protobuf files
        if file_path.endswith(".proto"):
            extraction = extract_from_protobuf(file_path, content)
            if extraction.messages or extraction.services:
                self.catalog.protobuf_files.append(extraction)
                return True

        # Job definitions
        if self._is_job_file(file_path):
            extraction = extract_jobs(file_path, content)
            if extraction.jobs:
                self.catalog.job_definitions.append(extraction)
                return True

        return False

    def _load_openapi_document(self, file_path: str, content: str) -> dict[str, Any] | None:
        """Parse and validate OpenAPI/Swagger docs deterministically."""
        if not file_path.endswith((".yml", ".yaml", ".json")):
            return None

        try:
            parsed = yaml.safe_load(content)
        except (yaml.YAMLError, json.JSONDecodeError):
            return None
        if not isinstance(parsed, dict):
            return None

        # OpenAPI 3.x / Swagger 2.0 identity + path table are required.
        has_spec_version = bool(parsed.get("openapi")) or bool(parsed.get("swagger"))
        has_paths = isinstance(parsed.get("paths"), dict)
        if not (has_spec_version and has_paths):
            return None
        return parsed

    def _is_job_file(self, file_path: str) -> bool:
        """Check if file is a job definition file."""
        if ".github/workflows" in file_path and file_path.endswith((".yml", ".yaml")):
            return True
        if file_path.endswith(("cronjob.yaml", "cronjob.yml")):
            return True
        return "prefect" in file_path.lower() and file_path.endswith((".yml", ".yaml"))


async def build_surface_graph(
    session: AsyncSession,
    collection_id: UUID,
    catalog: SurfaceCatalog,
) -> dict:
    """Build knowledge graph nodes and edges from surface catalog.

    Creates:
    - API_ENDPOINT nodes for OpenAPI endpoints
    - GRAPHQL_OPERATION nodes for GraphQL operations
    - GRAPHQL_TYPE nodes for GraphQL types
    - MESSAGE_SCHEMA nodes for Protobuf messages
    - SERVICE_RPC nodes for Protobuf services
    - JOB nodes for workflow/cron jobs

    Args:
        session: Database session
        collection_id: Collection UUID
        catalog: The surface catalog

    Returns:
        Stats dict
    """
    from sqlalchemy import select
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    stats = {
        "endpoint_nodes": 0,
        "endpoint_handler_links": 0,
        "graphql_nodes": 0,
        "proto_nodes": 0,
        "job_nodes": 0,
        "edges_created": 0,
        "evidence_created": 0,
    }
    symbol_candidates = await _load_symbol_candidates(session, collection_id)

    # Process OpenAPI specs
    for spec in catalog.openapi_specs:
        for endpoint in spec.endpoints:
            natural_key = f"api:{endpoint.method}:{endpoint.path}"
            resolved_handler_symbols = _resolve_endpoint_handler_symbols(
                endpoint=endpoint,
                symbol_candidates=symbol_candidates,
            )
            endpoint_meta = {
                "method": endpoint.method,
                "path": endpoint.path,
                "operation_id": endpoint.operation_id,
                "summary": endpoint.summary,
                "tags": endpoint.tags,
                "request_body_ref": endpoint.request_body_ref,
                "response_refs": endpoint.response_refs,
                "handler_hints": endpoint.handler_hints,
                "handler_symbol_names": [item["symbol_name"] for item in resolved_handler_symbols],
                "handler_symbol_node_ids": [
                    item["symbol_node_id"] for item in resolved_handler_symbols
                ],
                # Backward-compatible key expected by existing endpoint symbol index helper.
                "handler_symbols": [item["symbol_name"] for item in resolved_handler_symbols],
                "spec_file_path": spec.file_path,
                **_provenance(
                    mode="deterministic" if endpoint.operation_id else "inferred",
                    extractor="surface.openapi.v2",
                    confidence=0.96 if endpoint.operation_id else 0.8,
                ),
            }

            stmt = pg_insert(KnowledgeNode).values(
                collection_id=collection_id,
                kind=KnowledgeNodeKind.API_ENDPOINT,
                natural_key=natural_key,
                name=f"{endpoint.method} {endpoint.path}",
                meta=endpoint_meta,
            )
            stmt = stmt.on_conflict_do_update(
                constraint="uq_knowledge_node_natural",
                set_={
                    "name": stmt.excluded.name,
                    "meta": stmt.excluded.meta,
                },
            ).returning(KnowledgeNode.id)

            result = await session.execute(stmt)
            node_id = result.scalar_one()
            stats["endpoint_nodes"] += 1
            stats["endpoint_handler_links"] += len(resolved_handler_symbols)

            # Create evidence
            evidence_id = await _create_evidence(session, node_id, spec.file_path)
            stats["evidence_created"] += 1
            if evidence_id:
                endpoint_meta["provenance"]["evidence_ids"] = [evidence_id]  # type: ignore[index]
            stmt = pg_insert(KnowledgeNode).values(
                collection_id=collection_id,
                kind=KnowledgeNodeKind.API_ENDPOINT,
                natural_key=natural_key,
                name=f"{endpoint.method} {endpoint.path}",
                meta=endpoint_meta,
            )
            stmt = stmt.on_conflict_do_update(
                constraint="uq_knowledge_node_natural",
                set_={"name": stmt.excluded.name, "meta": stmt.excluded.meta},
            )
            await session.execute(stmt)

    # Process GraphQL schemas
    for schema in catalog.graphql_schemas:
        type_node_ids: dict[str, UUID] = {}

        # Create type nodes
        for type_def in schema.types:
            natural_key = f"graphql:type:{type_def.name}"

            stmt = pg_insert(KnowledgeNode).values(
                collection_id=collection_id,
                kind=KnowledgeNodeKind.GRAPHQL_TYPE,
                natural_key=natural_key,
                name=type_def.name,
                meta={
                    "kind": type_def.kind,
                    "fields": [f.name for f in type_def.fields],
                    "implements": type_def.implements,
                    "enum_values": type_def.enum_values,
                    "union_types": type_def.union_types,
                    "spec_file_path": schema.file_path,
                    **_provenance(
                        mode="deterministic",
                        extractor="surface.graphql.v1",
                        confidence=0.96,
                    ),
                },
            )
            stmt = stmt.on_conflict_do_update(
                constraint="uq_knowledge_node_natural",
                set_={
                    "name": stmt.excluded.name,
                    "meta": stmt.excluded.meta,
                },
            ).returning(KnowledgeNode.id)

            result = await session.execute(stmt)
            node_id = result.scalar_one()
            type_node_ids[type_def.name] = node_id
            stats["graphql_nodes"] += 1

        # Create operation nodes
        for op in schema.operations:
            natural_key = f"graphql:op:{op.name}"

            stmt = pg_insert(KnowledgeNode).values(
                collection_id=collection_id,
                kind=KnowledgeNodeKind.GRAPHQL_OPERATION,
                natural_key=natural_key,
                name=op.name,
                meta={
                    "kind": op.kind,
                    "fields": [{"name": f.name, "type": f.field_type} for f in op.fields],
                    "spec_file_path": schema.file_path,
                    **_provenance(
                        mode="deterministic",
                        extractor="surface.graphql.v1",
                        confidence=0.95,
                    ),
                },
            )
            stmt = stmt.on_conflict_do_update(
                constraint="uq_knowledge_node_natural",
                set_={
                    "name": stmt.excluded.name,
                    "meta": stmt.excluded.meta,
                },
            ).returning(KnowledgeNode.id)

            result = await session.execute(stmt)
            node_id = result.scalar_one()
            stats["graphql_nodes"] += 1

            # Create evidence
            await _create_evidence(session, node_id, schema.file_path)
            stats["evidence_created"] += 1

    # Process Protobuf files
    for proto in catalog.protobuf_files:
        message_node_ids: dict[str, UUID] = {}

        # Create message nodes
        for msg in proto.messages:
            qualified_name = f"{proto.package}.{msg.name}" if proto.package else msg.name
            natural_key = f"proto:msg:{qualified_name}"

            stmt = pg_insert(KnowledgeNode).values(
                collection_id=collection_id,
                kind=KnowledgeNodeKind.MESSAGE_SCHEMA,
                natural_key=natural_key,
                name=msg.name,
                meta={
                    "package": proto.package,
                    "fields": [
                        {"name": f.name, "type": f.field_type, "number": f.number}
                        for f in msg.fields
                    ],
                    "nested_messages": msg.nested_messages,
                    "nested_enums": msg.nested_enums,
                    "spec_file_path": proto.file_path,
                    **_provenance(
                        mode="deterministic",
                        extractor="surface.protobuf.v1",
                        confidence=0.95,
                    ),
                },
            )
            stmt = stmt.on_conflict_do_update(
                constraint="uq_knowledge_node_natural",
                set_={
                    "name": stmt.excluded.name,
                    "meta": stmt.excluded.meta,
                },
            ).returning(KnowledgeNode.id)

            result = await session.execute(stmt)
            node_id = result.scalar_one()
            message_node_ids[msg.name] = node_id
            stats["proto_nodes"] += 1

            # Create evidence
            await _create_evidence(session, node_id, proto.file_path)
            stats["evidence_created"] += 1

        # Create service/RPC nodes
        for service in proto.services:
            qualified_name = f"{proto.package}.{service.name}" if proto.package else service.name
            natural_key = f"proto:svc:{qualified_name}"

            stmt = pg_insert(KnowledgeNode).values(
                collection_id=collection_id,
                kind=KnowledgeNodeKind.SERVICE_RPC,
                natural_key=natural_key,
                name=service.name,
                meta={
                    "package": proto.package,
                    "rpcs": [
                        {
                            "name": r.name,
                            "request_type": r.request_type,
                            "response_type": r.response_type,
                            "request_stream": r.request_stream,
                            "response_stream": r.response_stream,
                        }
                        for r in service.rpcs
                    ],
                    "spec_file_path": proto.file_path,
                    **_provenance(
                        mode="deterministic",
                        extractor="surface.protobuf.v1",
                        confidence=0.94,
                    ),
                },
            )
            stmt = stmt.on_conflict_do_update(
                constraint="uq_knowledge_node_natural",
                set_={
                    "name": stmt.excluded.name,
                    "meta": stmt.excluded.meta,
                },
            ).returning(KnowledgeNode.id)

            result = await session.execute(stmt)
            node_id = result.scalar_one()
            stats["proto_nodes"] += 1

            # Create edges to request/response message types
            for rpc in service.rpcs:
                for msg_name, msg_node_id in message_node_ids.items():
                    if msg_name == rpc.request_type or msg_name == rpc.response_type:
                        edge_exists = await session.execute(
                            select(KnowledgeEdge.id).where(
                                KnowledgeEdge.collection_id == collection_id,
                                KnowledgeEdge.source_node_id == node_id,
                                KnowledgeEdge.target_node_id == msg_node_id,
                                KnowledgeEdge.kind == KnowledgeEdgeKind.RPC_USES_MESSAGE,
                            )
                        )
                        if not edge_exists.scalar_one_or_none():
                            session.add(
                                KnowledgeEdge(
                                    collection_id=collection_id,
                                    source_node_id=node_id,
                                    target_node_id=msg_node_id,
                                    kind=KnowledgeEdgeKind.RPC_USES_MESSAGE,
                                    meta={"rpc_name": rpc.name},
                                )
                            )
                            stats["edges_created"] += 1

    # Process job definitions
    for job_extraction in catalog.job_definitions:
        for job in job_extraction.jobs:
            natural_key = f"job:{job.framework}:{job.name}"

            stmt = pg_insert(KnowledgeNode).values(
                collection_id=collection_id,
                kind=KnowledgeNodeKind.JOB,
                natural_key=natural_key,
                name=job.name,
                meta={
                    "framework": job.framework,
                    "schedule": job.schedule,
                    "triggers": [{"type": t.trigger_type, "cron": t.cron} for t in job.triggers],
                    "container_image": job.container_image,
                    **_provenance(
                        mode="deterministic",
                        extractor="surface.jobs.v1",
                        confidence=0.93,
                    ),
                    **job.meta,
                },
            )
            stmt = stmt.on_conflict_do_update(
                constraint="uq_knowledge_node_natural",
                set_={
                    "name": stmt.excluded.name,
                    "meta": stmt.excluded.meta,
                },
            ).returning(KnowledgeNode.id)

            result = await session.execute(stmt)
            node_id = result.scalar_one()
            stats["job_nodes"] += 1

            # Create evidence with file path
            await _create_evidence(session, node_id, job.file_path)
            stats["evidence_created"] += 1

    return stats


async def _create_evidence(session: AsyncSession, node_id: UUID, file_path: str) -> str | None:
    """Create evidence record linking node to source file."""
    from sqlalchemy import select

    # Check if evidence link already exists
    existing = await session.execute(
        select(KnowledgeNodeEvidence.evidence_id).where(KnowledgeNodeEvidence.node_id == node_id)
    )
    existing_id = existing.scalar_one_or_none()
    if existing_id:
        return str(existing_id)

    evidence = KnowledgeEvidence(
        file_path=file_path,
        start_line=1,
        end_line=1,
    )
    session.add(evidence)
    await session.flush()

    link = KnowledgeNodeEvidence(
        node_id=node_id,
        evidence_id=evidence.id,
    )
    session.add(link)
    return str(evidence.id)


async def _load_symbol_candidates(
    session: AsyncSession,
    collection_id: UUID,
) -> dict[str, list[_SymbolCandidate]]:
    from sqlalchemy import select

    rows = (
        (
            await session.execute(
                select(KnowledgeNode).where(
                    KnowledgeNode.collection_id == collection_id,
                    KnowledgeNode.kind == KnowledgeNodeKind.SYMBOL,
                )
            )
        )
        .scalars()
        .all()
    )
    indexed: dict[str, list[_SymbolCandidate]] = {}
    for row in rows:
        candidate = _SymbolCandidate(node_id=row.id, name=row.name, natural_key=row.natural_key)
        for token in symbol_token_variants(row.name):
            indexed.setdefault(token, []).append(candidate)
    return indexed


def _resolve_endpoint_handler_symbols(
    *,
    endpoint: EndpointDef,
    symbol_candidates: dict[str, list[_SymbolCandidate]],
) -> list[dict[str, str | float]]:
    hints: list[tuple[str, str, float]] = []
    if endpoint.operation_id:
        hints.append((endpoint.operation_id, "operation_id", 0.84))
    for handler_hint in endpoint.handler_hints:
        hints.append((handler_hint, "handler_hint", 0.94))

    by_symbol_id: dict[UUID, dict[str, str | float]] = {}
    for hint_value, source, base_confidence in hints:
        for token in symbol_token_variants(hint_value):
            for candidate in symbol_candidates.get(token, []):
                confidence = base_confidence
                if candidate.name.strip().lower() == hint_value.strip().lower():
                    confidence = min(0.99, confidence + 0.04)
                record = by_symbol_id.get(candidate.node_id)
                if record and float(record["confidence"]) >= confidence:
                    continue
                by_symbol_id[candidate.node_id] = {
                    "symbol_node_id": str(candidate.node_id),
                    "symbol_name": candidate.name,
                    "symbol_natural_key": candidate.natural_key,
                    "match_source": source,
                    "confidence": round(confidence, 4),
                }
    return sorted(
        by_symbol_id.values(),
        key=lambda item: (
            -float(item["confidence"]),
            str(item["symbol_name"]).lower(),
        ),
    )
