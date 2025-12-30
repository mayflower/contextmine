"""Pydantic schemas for knowledge graph entities.

These schemas are used for:
- API responses
- LLM structured output validation
- MCP tool outputs
"""

from datetime import datetime
from uuid import UUID

from contextmine_core.models import (
    KnowledgeArtifactKind,
    KnowledgeEdgeKind,
    KnowledgeNodeKind,
)
from pydantic import BaseModel, ConfigDict, Field


class EvidenceSchema(BaseModel):
    """Schema for knowledge evidence."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    document_id: UUID | None = None
    chunk_id: UUID | None = None
    file_path: str
    start_line: int
    end_line: int
    snippet: str | None = None
    created_at: datetime


class KnowledgeNodeSchema(BaseModel):
    """Schema for knowledge graph nodes."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    collection_id: UUID
    kind: KnowledgeNodeKind
    natural_key: str
    name: str
    meta: dict = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class KnowledgeNodeWithEvidenceSchema(KnowledgeNodeSchema):
    """Schema for knowledge node with evidence."""

    evidence: list[EvidenceSchema] = Field(default_factory=list)


class KnowledgeEdgeSchema(BaseModel):
    """Schema for knowledge graph edges."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    collection_id: UUID
    source_node_id: UUID
    target_node_id: UUID
    kind: KnowledgeEdgeKind
    meta: dict = Field(default_factory=dict)
    created_at: datetime


class KnowledgeEdgeWithNodesSchema(KnowledgeEdgeSchema):
    """Schema for edge with source and target node details."""

    source_node: KnowledgeNodeSchema
    target_node: KnowledgeNodeSchema
    evidence: list[EvidenceSchema] = Field(default_factory=list)


class KnowledgeArtifactSchema(BaseModel):
    """Schema for knowledge artifacts (ERD, arc42, etc.)."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    collection_id: UUID
    kind: KnowledgeArtifactKind
    name: str
    content: str
    meta: dict = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class KnowledgeArtifactWithEvidenceSchema(KnowledgeArtifactSchema):
    """Schema for artifact with contributing evidence."""

    evidence: list[EvidenceSchema] = Field(default_factory=list)


# --- Schemas for creating/updating entities ---


class KnowledgeNodeCreate(BaseModel):
    """Schema for creating a knowledge node."""

    kind: KnowledgeNodeKind
    natural_key: str
    name: str
    meta: dict = Field(default_factory=dict)


class KnowledgeEdgeCreate(BaseModel):
    """Schema for creating a knowledge edge."""

    source_node_id: UUID
    target_node_id: UUID
    kind: KnowledgeEdgeKind
    meta: dict = Field(default_factory=dict)


class EvidenceCreate(BaseModel):
    """Schema for creating evidence."""

    document_id: UUID | None = None
    chunk_id: UUID | None = None
    file_path: str
    start_line: int
    end_line: int
    snippet: str | None = None


class KnowledgeArtifactCreate(BaseModel):
    """Schema for creating a knowledge artifact."""

    kind: KnowledgeArtifactKind
    name: str
    content: str
    meta: dict = Field(default_factory=dict)


# --- Schemas for graph queries ---


class GraphNeighborhoodRequest(BaseModel):
    """Request for graph neighborhood query."""

    node_id: UUID
    depth: int = Field(default=1, ge=1, le=3)
    edge_kinds: list[KnowledgeEdgeKind] | None = None


class GraphNeighborhoodResponse(BaseModel):
    """Response for graph neighborhood query."""

    center_node: KnowledgeNodeSchema
    nodes: list[KnowledgeNodeSchema]
    edges: list[KnowledgeEdgeSchema]


class GraphPathRequest(BaseModel):
    """Request for finding path between nodes."""

    from_node_id: UUID
    to_node_id: UUID
    max_hops: int = Field(default=6, ge=1, le=10)


class GraphPathResponse(BaseModel):
    """Response for path query."""

    found: bool
    path_length: int | None = None
    nodes: list[KnowledgeNodeSchema] = Field(default_factory=list)
    edges: list[KnowledgeEdgeSchema] = Field(default_factory=list)


# --- Schemas for business rules ---


class RuleCandidateMetadata(BaseModel):
    """Metadata for a rule candidate node."""

    predicate_snippet: str
    failure_kind: str
    container_symbol_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    language: str


class BusinessRuleMetadata(BaseModel):
    """Metadata for a labeled business rule."""

    rule_name: str
    description: str
    category: str | None = None
    severity: str | None = None
    source_candidate_ids: list[UUID] = Field(default_factory=list)
    llm_model: str | None = None
    labeled_at: datetime | None = None


# --- Schemas for GraphRAG ---


class GraphRAGRequest(BaseModel):
    """Request for GraphRAG retrieval."""

    query: str
    collection_id: UUID | None = None
    max_depth: int = Field(default=2, ge=1, le=3)
    max_nodes: int = Field(default=50, ge=1, le=200)


class GraphRAGBundle(BaseModel):
    """Response bundle from GraphRAG."""

    query: str
    nodes: list[KnowledgeNodeWithEvidenceSchema]
    edges: list[KnowledgeEdgeSchema]
    snippets: list[str] = Field(default_factory=list)
    markdown_summary: str
