"""System Surface Catalog builder.

Integrates all spec-driven extractors (OpenAPI, GraphQL, Protobuf, Jobs)
and builds knowledge graph nodes and edges.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID

from contextmine_core.analyzer.extractors.graphql import (
    GraphQLExtraction,
    extract_from_graphql,
)
from contextmine_core.analyzer.extractors.jobs import JobsExtraction, extract_jobs
from contextmine_core.analyzer.extractors.openapi import (
    OpenAPIExtraction,
    extract_from_openapi,
)
from contextmine_core.analyzer.extractors.protobuf import (
    ProtobufExtraction,
    extract_from_protobuf,
)
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
        # OpenAPI specs
        if self._is_openapi(file_path, content):
            extraction = extract_from_openapi(file_path, content)
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

    def _is_openapi(self, file_path: str, content: str) -> bool:
        """Check if file is likely an OpenAPI spec."""
        if not file_path.endswith((".yml", ".yaml", ".json")):
            return False

        # Quick content check
        content_lower = content[:2000].lower()
        return (
            "openapi" in content_lower or "swagger" in content_lower or '"openapi"' in content_lower
        )

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
        "graphql_nodes": 0,
        "proto_nodes": 0,
        "job_nodes": 0,
        "edges_created": 0,
        "evidence_created": 0,
    }

    # Process OpenAPI specs
    for spec in catalog.openapi_specs:
        for endpoint in spec.endpoints:
            natural_key = f"api:{endpoint.method}:{endpoint.path}"

            stmt = pg_insert(KnowledgeNode).values(
                collection_id=collection_id,
                kind=KnowledgeNodeKind.API_ENDPOINT,
                natural_key=natural_key,
                name=f"{endpoint.method} {endpoint.path}",
                meta={
                    "method": endpoint.method,
                    "path": endpoint.path,
                    "operation_id": endpoint.operation_id,
                    "summary": endpoint.summary,
                    "tags": endpoint.tags,
                    "request_body_ref": endpoint.request_body_ref,
                    "response_refs": endpoint.response_refs,
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
            stats["endpoint_nodes"] += 1

            # Create evidence
            await _create_evidence(session, node_id, spec.file_path)
            stats["evidence_created"] += 1

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
            natural_key = f"job:{job.kind.value}:{job.name}"

            stmt = pg_insert(KnowledgeNode).values(
                collection_id=collection_id,
                kind=KnowledgeNodeKind.JOB,
                natural_key=natural_key,
                name=job.name,
                meta={
                    "kind": job.kind.value,
                    "schedule": job.schedule,
                    "triggers": [{"type": t.trigger_type, "cron": t.cron} for t in job.triggers],
                    "runs_on": job.runs_on,
                    "container_image": job.container_image,
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


async def _create_evidence(session: AsyncSession, node_id: UUID, file_path: str) -> None:
    """Create evidence record linking node to source file."""
    from sqlalchemy import select

    # Check if evidence link already exists
    existing = await session.execute(
        select(KnowledgeNodeEvidence.evidence_id).where(KnowledgeNodeEvidence.node_id == node_id)
    )
    if existing.scalar_one_or_none():
        return

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
