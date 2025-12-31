"""Extractors for various file types.

Each extractor takes file content and returns extracted entities (nodes/edges).

LLM-based extractors (framework-agnostic):
- rules: Business rule extraction from any OO language
- jobs: Job/workflow extraction from any framework
- schema: Database schema extraction from any format

Standard spec extractors (inherently generic):
- openapi: OpenAPI/Swagger specs
- graphql: GraphQL schemas
- protobuf: Protocol Buffer definitions

Legacy extractors (kept for backwards compatibility):
- alembic: Alembic migration files (Python/SQLAlchemy specific)
- erm: ERM graph building (wraps alembic or new schema)
"""

# LLM-based generic extractors
# Legacy extractors (backwards compatibility)
from contextmine_core.analyzer.extractors.alembic import extract_from_alembic

# Standard spec extractors (inherently generic)
from contextmine_core.analyzer.extractors.graphql import (
    GraphQLExtraction,
    extract_from_graphql,
)
from contextmine_core.analyzer.extractors.jobs import (
    JobDef,
    JobKind,
    JobsExtraction,
    JobStepDef,
    JobTriggerDef,
    build_jobs_graph,
    extract_jobs,  # Sync extraction (backwards compat)
    extract_jobs_from_file,  # Async LLM-based (single file)
    extract_jobs_from_files,  # Async LLM-based (batch with triage)
)
from contextmine_core.analyzer.extractors.openapi import (
    EndpointDef,
    OpenAPIExtraction,
    extract_from_openapi,
)
from contextmine_core.analyzer.extractors.protobuf import (
    ProtobufExtraction,
    extract_from_protobuf,
)
from contextmine_core.analyzer.extractors.rules import (
    ExtractedRule,
    RulesExtraction,
    build_rules_graph,
    extract_rules_from_file,
)
from contextmine_core.analyzer.extractors.schema import (
    AggregatedSchema,
    ColumnDef,
    ForeignKeyDef,
    SchemaExtraction,
    TableDef,
    aggregate_schema_extractions,
    build_schema_graph,
    extract_schema_from_file,  # Async LLM-based (single file)
    extract_schema_from_files,  # Async LLM-based (batch with triage)
    generate_mermaid_erd,
    save_erd_artifact,
)

# Surface catalog (aggregates multiple spec types)
from contextmine_core.analyzer.extractors.surface import (
    SurfaceCatalog,
    SurfaceCatalogExtractor,
    build_surface_graph,
)

# Re-exports for backwards compatibility
# The old erm module used ERMExtractor and ERMSchema - map to new schema module
ERMExtractor = None  # Deprecated - use extract_schema_from_file instead
ERMSchema = AggregatedSchema  # Alias for backwards compatibility
build_erm_graph = build_schema_graph  # Alias for backwards compatibility


__all__ = [
    # LLM-based extractors
    "extract_rules_from_file",
    "RulesExtraction",
    "ExtractedRule",
    "build_rules_graph",
    "extract_jobs",  # Sync (backwards compat)
    "extract_jobs_from_file",  # Async LLM-based (single file)
    "extract_jobs_from_files",  # Async LLM-based (batch with triage)
    "JobsExtraction",
    "JobDef",
    "JobTriggerDef",
    "JobStepDef",
    "JobKind",
    "build_jobs_graph",
    "extract_schema_from_file",  # Async LLM-based (single file)
    "extract_schema_from_files",  # Async LLM-based (batch with triage)
    "SchemaExtraction",
    "AggregatedSchema",
    "TableDef",
    "ColumnDef",
    "ForeignKeyDef",
    "aggregate_schema_extractions",
    "build_schema_graph",
    "generate_mermaid_erd",
    "save_erd_artifact",
    # Standard spec extractors
    "extract_from_openapi",
    "OpenAPIExtraction",
    "EndpointDef",
    "extract_from_graphql",
    "GraphQLExtraction",
    "extract_from_protobuf",
    "ProtobufExtraction",
    # Surface catalog
    "SurfaceCatalog",
    "SurfaceCatalogExtractor",
    "build_surface_graph",
    # Legacy (backwards compatibility)
    "extract_from_alembic",
    "ERMExtractor",
    "ERMSchema",
    "build_erm_graph",
]
