"""Extractors for various file types.

Each extractor takes file content and returns extracted entities (nodes/edges).
"""

from contextmine_core.analyzer.extractors.alembic import extract_from_alembic
from contextmine_core.analyzer.extractors.erm import (
    ERMExtractor,
    build_erm_graph,
    generate_mermaid_erd,
)
from contextmine_core.analyzer.extractors.graphql import (
    GraphQLExtraction,
    extract_from_graphql,
)
from contextmine_core.analyzer.extractors.jobs import (
    JobDef,
    JobKind,
    JobsExtraction,
    extract_jobs,
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
    FailureKind,
    RuleCandidateDef,
    RulesExtraction,
    build_rule_candidates_graph,
    extract_rule_candidates,
)
from contextmine_core.analyzer.extractors.surface import (
    SurfaceCatalog,
    SurfaceCatalogExtractor,
    build_surface_graph,
)

__all__ = [
    # Alembic/ERM
    "extract_from_alembic",
    "ERMExtractor",
    "build_erm_graph",
    "generate_mermaid_erd",
    # OpenAPI
    "extract_from_openapi",
    "OpenAPIExtraction",
    "EndpointDef",
    # GraphQL
    "extract_from_graphql",
    "GraphQLExtraction",
    # Protobuf
    "extract_from_protobuf",
    "ProtobufExtraction",
    # Jobs
    "extract_jobs",
    "JobsExtraction",
    "JobDef",
    "JobKind",
    # Surface catalog
    "SurfaceCatalog",
    "SurfaceCatalogExtractor",
    "build_surface_graph",
    # Rules
    "extract_rule_candidates",
    "RulesExtraction",
    "RuleCandidateDef",
    "FailureKind",
    "build_rule_candidates_graph",
]
