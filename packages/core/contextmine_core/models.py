"""Database models."""

import enum
import uuid
from datetime import datetime

from contextmine_core.database import Base
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship


class CollectionVisibility(enum.Enum):
    """Visibility options for collections."""

    GLOBAL = "global"
    PRIVATE = "private"


class SourceType(enum.Enum):
    """Type of source."""

    GITHUB = "github"
    WEB = "web"


class SyncRunStatus(enum.Enum):
    """Status of a sync run."""

    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class EmbeddingProvider(enum.Enum):
    """Embedding model providers."""

    OPENAI = "openai"
    GEMINI = "gemini"


class SymbolKind(enum.Enum):
    """Kind of code symbol."""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    VARIABLE = "variable"
    CONSTANT = "constant"
    MODULE = "module"
    INTERFACE = "interface"
    ENUM = "enum"
    PROPERTY = "property"
    TYPE_ALIAS = "type_alias"


class SymbolEdgeType(enum.Enum):
    """Type of relationship between symbols."""

    CALLS = "calls"  # Function calls another function
    IMPORTS = "imports"  # Symbol imports another symbol
    INHERITS = "inherits"  # Class inherits from another class
    IMPLEMENTS = "implements"  # Class implements an interface
    CONTAINS = "contains"  # Class contains a method
    REFERENCES = "references"  # General reference to another symbol


class KnowledgeNodeKind(enum.Enum):
    """Kind of knowledge graph node."""

    # Code entities
    FILE = "file"
    SYMBOL = "symbol"
    # Semantic entities (LLM-extracted for GraphRAG)
    SEMANTIC_ENTITY = "semantic_entity"
    # Database entities
    DB_TABLE = "db_table"
    DB_COLUMN = "db_column"
    DB_CONSTRAINT = "db_constraint"
    # API entities
    API_ENDPOINT = "api_endpoint"
    GRAPHQL_OPERATION = "graphql_operation"
    GRAPHQL_TYPE = "graphql_type"
    MESSAGE_SCHEMA = "message_schema"
    SERVICE_RPC = "service_rpc"
    # Job entities
    JOB = "job"
    # Business rules
    RULE_CANDIDATE = "rule_candidate"
    BUSINESS_RULE = "business_rule"
    # Architecture
    BOUNDED_CONTEXT = "bounded_context"


class KnowledgeEdgeKind(enum.Enum):
    """Kind of knowledge graph edge."""

    # Code relationships
    FILE_DEFINES_SYMBOL = "file_defines_symbol"
    SYMBOL_CONTAINS_SYMBOL = "symbol_contains_symbol"
    FILE_IMPORTS_FILE = "file_imports_file"
    SYMBOL_CALLS_SYMBOL = "symbol_calls_symbol"
    SYMBOL_REFERENCES_SYMBOL = "symbol_references_symbol"
    # Semantic relationships (LLM-extracted for GraphRAG)
    SEMANTIC_RELATIONSHIP = "semantic_relationship"
    # Connect semantic entities to source files (for graph connectivity)
    FILE_MENTIONS_ENTITY = "file_mentions_entity"
    # Database relationships
    TABLE_HAS_COLUMN = "table_has_column"
    COLUMN_FK_TO_COLUMN = "column_fk_to_column"
    TABLE_HAS_CONSTRAINT = "table_has_constraint"
    # API relationships
    SYSTEM_EXPOSES_ENDPOINT = "system_exposes_endpoint"
    ENDPOINT_USES_SCHEMA = "endpoint_uses_schema"
    RPC_USES_MESSAGE = "rpc_uses_message"
    # Job relationships
    JOB_DEFINED_IN_FILE = "job_defined_in_file"
    JOB_DEPENDS_ON = "job_depends_on"
    # Business rule relationships
    RULE_DERIVED_FROM_CANDIDATE = "rule_derived_from_candidate"
    RULE_EVIDENCED_BY = "rule_evidenced_by"
    # Architecture relationships
    DOCUMENTED_BY = "documented_by"
    BELONGS_TO_CONTEXT = "belongs_to_context"


class KnowledgeArtifactKind(enum.Enum):
    """Kind of knowledge artifact."""

    MERMAID_ERD = "mermaid_erd"
    RULE_CATALOG = "rule_catalog"
    SURFACE_CATALOG = "surface_catalog"
    LPG_JSONL = "lpg_jsonl"
    CC_JSON = "cc_json"
    CX2 = "cx2"
    JGF = "jgf"
    MERMAID_C4_ASIS = "mermaid_c4_asis"
    MERMAID_C4_TOBE = "mermaid_c4_tobe"


class EmbeddingTargetType(enum.Enum):
    """Target type for knowledge embeddings."""

    NODE = "node"
    COMMUNITY = "community"


class TwinLayer(enum.Enum):
    """Architecture layers for twin nodes and edges."""

    PORTFOLIO_SYSTEM = "portfolio_system"
    DOMAIN_CONTAINER = "domain_container"
    COMPONENT_INTERFACE = "component_interface"
    CODE_CONTROLFLOW = "code_controlflow"


class ArchitectureIntentAction(enum.Enum):
    """Supported architecture intent actions."""

    EXTRACT_DOMAIN = "extract_domain"
    SPLIT_CONTAINER = "split_container"
    MOVE_COMPONENT = "move_component"
    DEFINE_INTERFACE = "define_interface"
    SET_VALIDATOR = "set_validator"
    APPLY_DATA_BOUNDARY = "apply_data_boundary"


class ArchitectureIntentStatus(enum.Enum):
    """Lifecycle status of architecture intents."""

    PENDING = "pending"
    BLOCKED = "blocked"
    APPROVED = "approved"
    EXECUTED = "executed"
    FAILED = "failed"


class IntentRiskLevel(enum.Enum):
    """Risk levels used by auto execution gates."""

    LOW = "low"
    HIGH = "high"


class ValidationSourceKind(enum.Enum):
    """Sources for validation and orchestration data."""

    TEKTON = "tekton"
    ARGO = "argo"
    TEMPORAL = "temporal"


class AppKV(Base):
    """Simple key-value store for app settings and metadata."""

    __tablename__ = "app_kv"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    value: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class User(Base):
    """User account linked to GitHub OAuth."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    github_user_id: Mapped[int] = mapped_column(BigInteger, unique=True, nullable=False)
    github_login: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    avatar_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    oauth_tokens: Mapped[list["OAuthToken"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    owned_collections: Mapped[list["Collection"]] = relationship(
        back_populates="owner", cascade="all, delete-orphan"
    )
    collection_memberships: Mapped[list["CollectionMember"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class OAuthToken(Base):
    """OAuth tokens for external providers."""

    __tablename__ = "oauth_tokens"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    access_token_encrypted: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    user: Mapped["User"] = relationship(back_populates="oauth_tokens")


class Collection(Base):
    """A collection of documentation sources."""

    __tablename__ = "collections"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    slug: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    visibility: Mapped[CollectionVisibility] = mapped_column(
        Enum(
            CollectionVisibility,
            name="collection_visibility",
            create_type=False,
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
    )
    owner_user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    config: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    owner: Mapped["User"] = relationship(back_populates="owned_collections")
    members: Mapped[list["CollectionMember"]] = relationship(
        back_populates="collection", cascade="all, delete-orphan"
    )
    invites: Mapped[list["CollectionInvite"]] = relationship(
        back_populates="collection", cascade="all, delete-orphan"
    )
    sources: Mapped[list["Source"]] = relationship(
        back_populates="collection", cascade="all, delete-orphan"
    )


class CollectionMember(Base):
    """Membership link between users and collections."""

    __tablename__ = "collection_members"

    collection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        primary_key=True,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    collection: Mapped["Collection"] = relationship(back_populates="members")
    user: Mapped["User"] = relationship(back_populates="collection_memberships")


class CollectionInvite(Base):
    """Pending invite to a collection for a GitHub user."""

    __tablename__ = "collection_invites"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
    )
    github_login: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    collection: Mapped["Collection"] = relationship(back_populates="invites")


class Source(Base):
    """A source of documentation (GitHub repo or web URL)."""

    __tablename__ = "sources"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
    )
    type: Mapped[SourceType] = mapped_column(
        Enum(
            SourceType,
            name="source_type",
            create_type=False,
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
    )
    url: Mapped[str] = mapped_column(String(2048), nullable=False)
    config: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    schedule_interval_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=60)
    next_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    cursor: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Deploy key for SSH access (GitHub sources only)
    deploy_key_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    deploy_key_fingerprint: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    collection: Mapped["Collection"] = relationship(back_populates="sources")
    runs: Mapped[list["SyncRun"]] = relationship(
        back_populates="source", cascade="all, delete-orphan"
    )
    documents: Mapped[list["Document"]] = relationship(
        back_populates="source", cascade="all, delete-orphan"
    )
    ingest_token: Mapped["SourceIngestToken | None"] = relationship(
        back_populates="source", cascade="all, delete-orphan", uselist=False
    )
    coverage_ingest_jobs: Mapped[list["CoverageIngestJob"]] = relationship(
        back_populates="source", cascade="all, delete-orphan"
    )


class SourceIngestToken(Base):
    """Single active CI ingest token for one source."""

    __tablename__ = "source_ingest_tokens"
    __table_args__ = (UniqueConstraint("source_id", name="uq_source_ingest_token_source"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
    )
    token_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    token_preview: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    rotated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    source: Mapped["Source"] = relationship(back_populates="ingest_token")


class CoverageIngestJob(Base):
    """Coverage ingest job pushed from CI and processed asynchronously."""

    __tablename__ = "coverage_ingest_jobs"
    __table_args__ = (
        Index("ix_coverage_ingest_job_source_created", "source_id", "created_at"),
        Index("ix_coverage_ingest_job_status", "status", "updated_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
    )
    collection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
    )
    scenario_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("twin_scenarios.id", ondelete="SET NULL"),
        nullable=True,
    )
    commit_sha: Mapped[str] = mapped_column(String(64), nullable=False)
    branch: Mapped[str | None] = mapped_column(String(255), nullable=True)
    provider: Mapped[str] = mapped_column(String(64), nullable=False, default="github_actions")
    workflow_run_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    error_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    stats: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    source: Mapped["Source"] = relationship(back_populates="coverage_ingest_jobs")
    collection: Mapped["Collection"] = relationship()
    scenario: Mapped["TwinScenario | None"] = relationship()
    reports: Mapped[list["CoverageIngestReport"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )


class CoverageIngestReport(Base):
    """Raw uploaded report payload attached to one ingest job."""

    __tablename__ = "coverage_ingest_reports"
    __table_args__ = (Index("ix_coverage_ingest_report_job", "job_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("coverage_ingest_jobs.id", ondelete="CASCADE"),
        nullable=False,
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    protocol_detected: Mapped[str | None] = mapped_column(String(64), nullable=True)
    report_bytes: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    diagnostics: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    job: Mapped["CoverageIngestJob"] = relationship(back_populates="reports")


class SyncRun(Base):
    """A sync run for a source."""

    __tablename__ = "sync_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[SyncRunStatus] = mapped_column(
        Enum(
            SyncRunStatus,
            name="sync_run_status",
            create_type=False,
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
    )
    stats: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    source: Mapped["Source"] = relationship(back_populates="runs")


class Document(Base):
    """A document extracted from a source."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
    )
    uri: Mapped[str] = mapped_column(String(2048), unique=True, nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    content_markdown: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    meta: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    source: Mapped["Source"] = relationship(back_populates="documents")
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    symbols: Mapped[list["Symbol"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class EmbeddingModel(Base):
    """An embedding model configuration."""

    __tablename__ = "embedding_models"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    provider: Mapped[EmbeddingProvider] = mapped_column(
        Enum(
            EmbeddingProvider,
            name="embedding_provider",
            create_type=False,
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
    )
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="embedding_model")


class Chunk(Base):
    """A chunk of a document for retrieval."""

    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # tsv is a generated column, not mapped in Python
    meta: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    # Embedding columns
    embedding_model_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("embedding_models.id", ondelete="SET NULL"),
        nullable=True,
    )
    embedded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Note: embedding vector column is not mapped as SQLAlchemy type
    # It's accessed via raw SQL for pgvector operations

    # Relationships
    document: Mapped["Document"] = relationship(back_populates="chunks")
    embedding_model: Mapped["EmbeddingModel | None"] = relationship(back_populates="chunks")


class Symbol(Base):
    """A code symbol (function, class, method, etc.) extracted from a document.

    Symbols are extracted via tree-sitter during sync and stored for fast lookup.
    They cascade delete when the parent Document is deleted.
    """

    __tablename__ = "symbols"
    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "qualified_name",
            "start_line",
            "end_line",
            "kind",
            name="uq_symbol_identity",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    # Qualified name: e.g., "ResearchAgent.research" or just "run_research"
    qualified_name: Mapped[str] = mapped_column(String(1024), nullable=False)
    # Simple name: e.g., "research" or "run_research"
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    kind: Mapped[SymbolKind] = mapped_column(
        Enum(
            SymbolKind,
            name="symbol_kind",
            create_type=False,
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
    )
    # Line numbers (1-indexed)
    start_line: Mapped[int] = mapped_column(Integer, nullable=False)
    end_line: Mapped[int] = mapped_column(Integer, nullable=False)
    # Optional signature for functions/methods
    signature: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Parent symbol qualified name (for nested symbols like methods in classes)
    parent_name: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    # Additional metadata (docstring, decorators, etc.)
    meta: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    document: Mapped["Document"] = relationship(back_populates="symbols")
    # Edges where this symbol is the source
    outgoing_edges: Mapped[list["SymbolEdge"]] = relationship(
        back_populates="source_symbol",
        foreign_keys="SymbolEdge.source_symbol_id",
        cascade="all, delete-orphan",
    )
    # Edges where this symbol is the target
    incoming_edges: Mapped[list["SymbolEdge"]] = relationship(
        back_populates="target_symbol",
        foreign_keys="SymbolEdge.target_symbol_id",
        cascade="all, delete-orphan",
    )


class SymbolEdge(Base):
    """A relationship between two symbols (calls, imports, inherits, etc.).

    Edges are extracted during sync using tree-sitter and/or LSP.
    They cascade delete when either the source or target Symbol is deleted.
    """

    __tablename__ = "symbol_edges"
    __table_args__ = (
        Index(
            "uq_symbol_edge_identity",
            "source_symbol_id",
            "target_symbol_id",
            "edge_type",
            text("coalesce(source_line, -1)"),
            unique=True,
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_symbol_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("symbols.id", ondelete="CASCADE"),
        nullable=False,
    )
    target_symbol_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("symbols.id", ondelete="CASCADE"),
        nullable=False,
    )
    edge_type: Mapped[SymbolEdgeType] = mapped_column(
        Enum(
            SymbolEdgeType,
            name="symbol_edge_type",
            create_type=False,
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
    )
    # Line number in source where the reference occurs
    source_line: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Additional metadata
    meta: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    source_symbol: Mapped["Symbol"] = relationship(
        back_populates="outgoing_edges",
        foreign_keys=[source_symbol_id],
    )
    target_symbol: Mapped["Symbol"] = relationship(
        back_populates="incoming_edges",
        foreign_keys=[target_symbol_id],
    )


# -----------------------------------------------------------------------------
# Knowledge Graph Tables
# -----------------------------------------------------------------------------


class KnowledgeEvidence(Base):
    """Evidence linking knowledge nodes/edges to source locations.

    Evidence provides traceability back to the original source: file path,
    line numbers, and optionally document/chunk references.
    """

    __tablename__ = "knowledge_evidence"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # Optional link to indexed document
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
    )
    # Optional link to specific chunk
    chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chunks.id", ondelete="SET NULL"),
        nullable=True,
    )
    # File path (may differ from document URI for non-indexed files)
    file_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    # Line span (1-indexed)
    start_line: Mapped[int] = mapped_column(Integer, nullable=False)
    end_line: Mapped[int] = mapped_column(Integer, nullable=False)
    # Optional code snippet for context
    snippet: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    document: Mapped["Document | None"] = relationship()
    chunk: Mapped["Chunk | None"] = relationship()


class KnowledgeNode(Base):
    """A node in the knowledge graph representing an extracted entity.

    Nodes are typed (FILE, SYMBOL, DB_TABLE, API_ENDPOINT, etc.) and have
    a natural key for idempotent upserts. Metadata is stored as JSON.
    """

    __tablename__ = "knowledge_nodes"
    __table_args__ = (
        UniqueConstraint("collection_id", "kind", "natural_key", name="uq_knowledge_node_natural"),
        Index("ix_knowledge_node_collection_kind", "collection_id", "kind"),
        Index("ix_knowledge_node_meta", "meta", postgresql_using="gin"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
    )
    kind: Mapped[KnowledgeNodeKind] = mapped_column(
        Enum(
            KnowledgeNodeKind,
            name="knowledge_node_kind",
            create_type=False,
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
    )
    # Stable natural key for upserts (e.g., file path, qualified symbol name)
    natural_key: Mapped[str] = mapped_column(String(2048), nullable=False)
    # Display name
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    # Flexible metadata (schema depends on kind)
    meta: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    collection: Mapped["Collection"] = relationship()
    evidence_links: Mapped[list["KnowledgeNodeEvidence"]] = relationship(
        back_populates="node", cascade="all, delete-orphan"
    )
    outgoing_edges: Mapped[list["KnowledgeEdge"]] = relationship(
        back_populates="source_node",
        foreign_keys="KnowledgeEdge.source_node_id",
        cascade="all, delete-orphan",
    )
    incoming_edges: Mapped[list["KnowledgeEdge"]] = relationship(
        back_populates="target_node",
        foreign_keys="KnowledgeEdge.target_node_id",
        cascade="all, delete-orphan",
    )


class KnowledgeEdge(Base):
    """An edge in the knowledge graph representing a relationship between nodes.

    Edges are typed (FILE_DEFINES_SYMBOL, TABLE_HAS_COLUMN, etc.) and can
    have metadata and evidence links.
    """

    __tablename__ = "knowledge_edges"
    __table_args__ = (
        UniqueConstraint(
            "collection_id",
            "source_node_id",
            "target_node_id",
            "kind",
            name="uq_knowledge_edge_unique",
        ),
        Index("ix_knowledge_edge_source", "source_node_id", "kind"),
        Index("ix_knowledge_edge_target", "target_node_id", "kind"),
        Index("ix_knowledge_edge_collection", "collection_id", "kind"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
    )
    source_node_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_nodes.id", ondelete="CASCADE"),
        nullable=False,
    )
    target_node_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_nodes.id", ondelete="CASCADE"),
        nullable=False,
    )
    kind: Mapped[KnowledgeEdgeKind] = mapped_column(
        Enum(
            KnowledgeEdgeKind,
            name="knowledge_edge_kind",
            create_type=False,
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
    )
    # Flexible metadata
    meta: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    collection: Mapped["Collection"] = relationship()
    source_node: Mapped["KnowledgeNode"] = relationship(
        back_populates="outgoing_edges",
        foreign_keys=[source_node_id],
    )
    target_node: Mapped["KnowledgeNode"] = relationship(
        back_populates="incoming_edges",
        foreign_keys=[target_node_id],
    )
    evidence_links: Mapped[list["KnowledgeEdgeEvidence"]] = relationship(
        back_populates="edge", cascade="all, delete-orphan"
    )


class KnowledgeNodeEvidence(Base):
    """Link table between knowledge nodes and their evidence."""

    __tablename__ = "knowledge_node_evidence"

    node_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_nodes.id", ondelete="CASCADE"),
        primary_key=True,
    )
    evidence_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_evidence.id", ondelete="CASCADE"),
        primary_key=True,
    )

    # Relationships
    node: Mapped["KnowledgeNode"] = relationship(back_populates="evidence_links")
    evidence: Mapped["KnowledgeEvidence"] = relationship()


class KnowledgeEdgeEvidence(Base):
    """Link table between knowledge edges and their evidence."""

    __tablename__ = "knowledge_edge_evidence"

    edge_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_edges.id", ondelete="CASCADE"),
        primary_key=True,
    )
    evidence_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_evidence.id", ondelete="CASCADE"),
        primary_key=True,
    )

    # Relationships
    edge: Mapped["KnowledgeEdge"] = relationship(back_populates="evidence_links")
    evidence: Mapped["KnowledgeEvidence"] = relationship()


class KnowledgeArtifact(Base):
    """A generated artifact from the knowledge graph (ERD, arc42 doc, etc.).

    Artifacts are derived from extracted knowledge and stored for retrieval.
    They link back to evidence showing what contributed to their generation.
    """

    __tablename__ = "knowledge_artifacts"
    __table_args__ = (
        UniqueConstraint("collection_id", "kind", "name", name="uq_knowledge_artifact_name"),
        Index("ix_knowledge_artifact_collection_kind", "collection_id", "kind"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
    )
    kind: Mapped[KnowledgeArtifactKind] = mapped_column(
        Enum(
            KnowledgeArtifactKind,
            name="knowledge_artifact_kind",
            create_type=False,
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    # The artifact content (Mermaid diagram, Markdown doc, etc.)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Flexible metadata
    meta: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    collection: Mapped["Collection"] = relationship()
    evidence_links: Mapped[list["KnowledgeArtifactEvidence"]] = relationship(
        back_populates="artifact", cascade="all, delete-orphan"
    )


class KnowledgeArtifactEvidence(Base):
    """Link table between knowledge artifacts and their contributing evidence."""

    __tablename__ = "knowledge_artifact_evidence"

    artifact_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_artifacts.id", ondelete="CASCADE"),
        primary_key=True,
    )
    evidence_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_evidence.id", ondelete="CASCADE"),
        primary_key=True,
    )

    # Relationships
    artifact: Mapped["KnowledgeArtifact"] = relationship(back_populates="evidence_links")
    evidence: Mapped["KnowledgeEvidence"] = relationship()


# -----------------------------------------------------------------------------
# GraphRAG Community Tables
# -----------------------------------------------------------------------------


class KnowledgeCommunity(Base):
    """A community in the knowledge graph hierarchy.

    Communities are detected via deterministic label propagation and form
    a two-level hierarchy. Level 1 communities contain nodes, level 2
    communities contain level 1 communities.
    """

    __tablename__ = "knowledge_communities"
    __table_args__ = (
        UniqueConstraint(
            "collection_id", "level", "natural_key", name="uq_knowledge_community_natural"
        ),
        Index("ix_knowledge_community_collection_level", "collection_id", "level"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
    )
    # Hierarchy level: 1 = node communities, 2 = meta-communities
    level: Mapped[int] = mapped_column(Integer, nullable=False)
    # Stable natural key for idempotent upserts
    natural_key: Mapped[str] = mapped_column(String(2048), nullable=False)
    # Human-readable title (derived or LLM-generated)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    # Summary text (extractive or LLM-generated)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Flexible metadata (member count, top symbols, etc.)
    meta: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    collection: Mapped["Collection"] = relationship()
    members: Mapped[list["CommunityMember"]] = relationship(
        back_populates="community", cascade="all, delete-orphan"
    )


class CommunityMember(Base):
    """Membership of a node in a community with score.

    Score represents the strength of membership (e.g., normalized internal
    degree for graph communities).
    """

    __tablename__ = "community_members"
    __table_args__ = (UniqueConstraint("community_id", "node_id", name="uq_community_member"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    community_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_communities.id", ondelete="CASCADE"),
        nullable=False,
    )
    node_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_nodes.id", ondelete="CASCADE"),
        nullable=False,
    )
    # Membership strength (0-1, higher = stronger membership)
    score: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    community: Mapped["KnowledgeCommunity"] = relationship(back_populates="members")
    node: Mapped["KnowledgeNode"] = relationship()


class KnowledgeEmbedding(Base):
    """Embedding vector for a knowledge node or community.

    Used for semantic similarity search in GraphRAG retrieval.
    Content hash enables idempotent updates (skip if unchanged).
    """

    __tablename__ = "knowledge_embeddings"
    __table_args__ = (
        UniqueConstraint(
            "collection_id",
            "target_type",
            "target_id",
            "model_name",
            name="uq_knowledge_embedding_target",
        ),
        Index("ix_knowledge_embedding_collection", "collection_id", "target_type"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
    )
    target_type: Mapped[EmbeddingTargetType] = mapped_column(
        Enum(
            EmbeddingTargetType,
            name="embedding_target_type",
            create_type=False,
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
    )
    # UUID of the node or community
    target_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    # Model identifier (e.g., "text-embedding-3-small")
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    # Provider (e.g., "openai", "gemini")
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    # Content hash for idempotency (hash of text that was embedded)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    # Note: embedding vector column added via raw SQL (pgvector)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    collection: Mapped["Collection"] = relationship()


# -----------------------------------------------------------------------------
# Digital Twin / Intent Engine Tables
# -----------------------------------------------------------------------------


class TwinScenario(Base):
    """Versioned twin scenario (AS-IS baseline or TO-BE branch)."""

    __tablename__ = "twin_scenarios"
    __table_args__ = (
        Index("ix_twin_scenario_collection", "collection_id"),
        Index("ix_twin_scenario_parent", "base_scenario_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    base_scenario_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("twin_scenarios.id", ondelete="SET NULL"),
        nullable=True,
    )
    is_as_is: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    meta: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    collection: Mapped["Collection"] = relationship()
    base_scenario: Mapped["TwinScenario | None"] = relationship(remote_side=[id])
    created_by: Mapped["User | None"] = relationship()


class TwinNode(Base):
    """Node in a scenario graph."""

    __tablename__ = "twin_nodes"
    __table_args__ = (
        UniqueConstraint("scenario_id", "natural_key", name="uq_twin_node_natural"),
        Index("ix_twin_node_scenario_kind", "scenario_id", "kind"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scenario_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("twin_scenarios.id", ondelete="CASCADE"),
        nullable=False,
    )
    natural_key: Mapped[str] = mapped_column(String(2048), nullable=False)
    kind: Mapped[str] = mapped_column(String(128), nullable=False)
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    meta: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    provenance_node_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_nodes.id", ondelete="SET NULL"),
        nullable=True,
    )
    source_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sources.id", ondelete="SET NULL"),
        nullable=True,
    )
    source_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("twin_source_versions.id", ondelete="SET NULL"),
        nullable=True,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    scenario: Mapped["TwinScenario"] = relationship()
    provenance_node: Mapped["KnowledgeNode | None"] = relationship()
    source: Mapped["Source | None"] = relationship()
    source_version: Mapped["TwinSourceVersion | None"] = relationship()


class TwinEdge(Base):
    """Edge in a scenario graph."""

    __tablename__ = "twin_edges"
    __table_args__ = (
        UniqueConstraint(
            "scenario_id",
            "source_node_id",
            "target_node_id",
            "kind",
            name="uq_twin_edge_unique",
        ),
        Index("ix_twin_edge_scenario_kind", "scenario_id", "kind"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scenario_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("twin_scenarios.id", ondelete="CASCADE"),
        nullable=False,
    )
    source_node_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("twin_nodes.id", ondelete="CASCADE"),
        nullable=False,
    )
    target_node_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("twin_nodes.id", ondelete="CASCADE"),
        nullable=False,
    )
    kind: Mapped[str] = mapped_column(String(128), nullable=False)
    meta: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    source_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sources.id", ondelete="SET NULL"),
        nullable=True,
    )
    source_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("twin_source_versions.id", ondelete="SET NULL"),
        nullable=True,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    scenario: Mapped["TwinScenario"] = relationship()
    source_node: Mapped["TwinNode"] = relationship(foreign_keys=[source_node_id])
    target_node: Mapped["TwinNode"] = relationship(foreign_keys=[target_node_id])
    source: Mapped["Source | None"] = relationship()
    source_version: Mapped["TwinSourceVersion | None"] = relationship()


class TwinSourceVersion(Base):
    """Materialized source revision metadata for one collection/source."""

    __tablename__ = "twin_source_versions"
    __table_args__ = (
        UniqueConstraint(
            "source_id",
            "revision_key",
            "extractor_version",
            name="uq_twin_source_version_revision",
        ),
        Index("ix_twin_source_version_collection", "collection_id", "finished_at"),
        Index("ix_twin_source_version_source_status", "source_id", "status"),
        CheckConstraint(
            "status IN ('queued','materializing','ready','failed','stale','loading','generating')",
            name="ck_twin_source_version_status",
        ),
        CheckConstraint(
            "joern_status IN ('pending','generating','loading','ready','failed')",
            name="ck_twin_source_version_joern_status",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
    )
    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
    )
    revision_key: Mapped[str] = mapped_column(String(255), nullable=False)
    extractor_version: Mapped[str] = mapped_column(String(64), nullable=False)
    language_profile: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    stats: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    joern_status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    joern_project: Mapped[str | None] = mapped_column(String(255), nullable=True)
    joern_cpg_path: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    joern_server_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    collection: Mapped["Collection"] = relationship()
    source: Mapped["Source"] = relationship()


class TwinEvent(Base):
    """Append-only event log for twin materialization lifecycle."""

    __tablename__ = "twin_events"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_twin_event_idempotency"),
        Index("ix_twin_event_collection_ts", "collection_id", "event_ts"),
        Index("ix_twin_event_source_ts", "source_id", "event_ts"),
        Index("ix_twin_event_scenario_ts", "scenario_id", "event_ts"),
        CheckConstraint(
            "status IN ('queued','materializing','ready','failed','stale','loading','generating','degraded')",
            name="ck_twin_event_status",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
    )
    scenario_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("twin_scenarios.id", ondelete="SET NULL"),
        nullable=True,
    )
    source_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sources.id", ondelete="SET NULL"),
        nullable=True,
    )
    source_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("twin_source_versions.id", ondelete="SET NULL"),
        nullable=True,
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    event_ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    collection: Mapped["Collection"] = relationship()
    scenario: Mapped["TwinScenario | None"] = relationship()
    source: Mapped["Source | None"] = relationship()
    source_version: Mapped["TwinSourceVersion | None"] = relationship()


class TwinAnalysisCache(Base):
    """Cache for expensive twin analysis queries."""

    __tablename__ = "twin_analysis_cache"
    __table_args__ = (
        UniqueConstraint(
            "scenario_id",
            "engine",
            "tool_name",
            "params_hash",
            "cache_key",
            name="uq_twin_analysis_cache_key",
        ),
        Index(
            "ix_twin_analysis_cache_lookup",
            "scenario_id",
            "engine",
            "tool_name",
            "expires_at",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    cache_key: Mapped[str] = mapped_column(String(128), nullable=False)
    scenario_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("twin_scenarios.id", ondelete="CASCADE"),
        nullable=False,
    )
    engine: Mapped[str] = mapped_column(String(32), nullable=False, default="graphrag")
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    params_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    scenario: Mapped["TwinScenario"] = relationship()


class TwinFinding(Base):
    """Normalized findings attached to twin scenario/source-version snapshots."""

    __tablename__ = "twin_findings"
    __table_args__ = (
        UniqueConstraint("scenario_id", "fingerprint", name="uq_twin_finding_fingerprint"),
        Index("ix_twin_findings_scenario_created", "scenario_id", "created_at"),
        Index("ix_twin_findings_status", "scenario_id", "status"),
        Index("ix_twin_findings_type", "scenario_id", "finding_type"),
        CheckConstraint(
            "status IN ('open','triaged','resolved','false_positive','suppressed')",
            name="ck_twin_findings_status",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scenario_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("twin_scenarios.id", ondelete="CASCADE"),
        nullable=False,
    )
    source_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("twin_source_versions.id", ondelete="SET NULL"),
        nullable=True,
    )
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    finding_type: Mapped[str] = mapped_column(String(128), nullable=False)
    severity: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="open")
    filename: Mapped[str] = mapped_column(String(2048), nullable=False)
    line_number: Mapped[int] = mapped_column(Integer, nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    flow_data: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    meta: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    scenario: Mapped["TwinScenario"] = relationship()
    source_version: Mapped["TwinSourceVersion | None"] = relationship()


class TwinNodeLayer(Base):
    """Many-to-many layer assignment for twin nodes."""

    __tablename__ = "twin_node_layers"
    __table_args__ = (UniqueConstraint("node_id", "layer", name="uq_twin_node_layer"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    node_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("twin_nodes.id", ondelete="CASCADE"),
        nullable=False,
    )
    layer: Mapped[TwinLayer] = mapped_column(
        Enum(
            TwinLayer,
            name="twin_layer",
            create_type=False,
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    node: Mapped["TwinNode"] = relationship()


class TwinEdgeLayer(Base):
    """Many-to-many layer assignment for twin edges."""

    __tablename__ = "twin_edge_layers"
    __table_args__ = (UniqueConstraint("edge_id", "layer", name="uq_twin_edge_layer"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    edge_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("twin_edges.id", ondelete="CASCADE"),
        nullable=False,
    )
    layer: Mapped[TwinLayer] = mapped_column(
        Enum(
            TwinLayer,
            name="twin_layer",
            create_type=False,
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    edge: Mapped["TwinEdge"] = relationship()


class ArchitectureIntent(Base):
    """Architecture intent request attached to a scenario."""

    __tablename__ = "architecture_intents"
    __table_args__ = (
        Index("ix_arch_intent_scenario", "scenario_id", "status"),
        Index("ix_arch_intent_requested_by", "requested_by_user_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scenario_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("twin_scenarios.id", ondelete="CASCADE"),
        nullable=False,
    )
    intent_version: Mapped[str] = mapped_column(String(16), nullable=False)
    action: Mapped[ArchitectureIntentAction] = mapped_column(
        Enum(
            ArchitectureIntentAction,
            name="architecture_intent_action",
            create_type=False,
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
    )
    target_type: Mapped[str] = mapped_column(String(64), nullable=False)
    target_id: Mapped[str] = mapped_column(String(2048), nullable=False)
    params: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    expected_scenario_version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[ArchitectureIntentStatus] = mapped_column(
        Enum(
            ArchitectureIntentStatus,
            name="architecture_intent_status",
            create_type=False,
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
        default=ArchitectureIntentStatus.PENDING,
    )
    risk_level: Mapped[IntentRiskLevel] = mapped_column(
        Enum(
            IntentRiskLevel,
            name="intent_risk_level",
            create_type=False,
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
        default=IntentRiskLevel.LOW,
    )
    requires_approval: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    requested_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    scenario: Mapped["TwinScenario"] = relationship()
    requested_by: Mapped["User | None"] = relationship()


class ArchitectureIntentRun(Base):
    """Execution journal entry for an intent."""

    __tablename__ = "architecture_intent_runs"
    __table_args__ = (Index("ix_arch_intent_run_intent", "intent_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    intent_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("architecture_intents.id", ondelete="CASCADE"),
        nullable=False,
    )
    scenario_version_before: Mapped[int] = mapped_column(Integer, nullable=False)
    scenario_version_after: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    intent: Mapped["ArchitectureIntent"] = relationship()


class TwinPatch(Base):
    """RFC6902 patch history for a scenario."""

    __tablename__ = "twin_patches"
    __table_args__ = (
        UniqueConstraint("scenario_id", "scenario_version", name="uq_twin_patch_version"),
        Index("ix_twin_patch_scenario", "scenario_id", "scenario_version"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scenario_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("twin_scenarios.id", ondelete="CASCADE"),
        nullable=False,
    )
    scenario_version: Mapped[int] = mapped_column(Integer, nullable=False)
    intent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("architecture_intents.id", ondelete="SET NULL"),
        nullable=True,
    )
    patch_ops: Mapped[list[dict]] = mapped_column(JSON, nullable=False, default=list)
    created_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    scenario: Mapped["TwinScenario"] = relationship()
    intent: Mapped["ArchitectureIntent | None"] = relationship()
    created_by: Mapped["User | None"] = relationship()


class ValidationSnapshot(Base):
    """Normalized validation/orchestration metrics snapshot."""

    __tablename__ = "validation_snapshots"
    __table_args__ = (
        Index("ix_validation_snapshot_collection", "collection_id", "captured_at"),
        Index("ix_validation_snapshot_source", "source_kind", "metric_key"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=True,
    )
    source_kind: Mapped[ValidationSourceKind] = mapped_column(
        Enum(
            ValidationSourceKind,
            name="validation_source_kind",
            create_type=False,
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
    )
    metric_key: Mapped[str] = mapped_column(String(128), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str | None] = mapped_column(String(64), nullable=True)
    meta: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    captured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    collection: Mapped["Collection | None"] = relationship()


class MetricSnapshot(Base):
    """Code city metrics per scenario/node."""

    __tablename__ = "metric_snapshots"
    __table_args__ = (
        Index("ix_metric_snapshot_scenario", "scenario_id", "captured_at"),
        Index("ix_metric_snapshot_node", "scenario_id", "node_natural_key"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scenario_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("twin_scenarios.id", ondelete="CASCADE"),
        nullable=False,
    )
    node_natural_key: Mapped[str] = mapped_column(String(2048), nullable=False)
    loc: Mapped[int | None] = mapped_column(Integer, nullable=True)
    symbol_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    coupling: Mapped[float | None] = mapped_column(Float, nullable=True)
    coverage: Mapped[float | None] = mapped_column(Float, nullable=True)
    complexity: Mapped[float | None] = mapped_column(Float, nullable=True)
    cohesion: Mapped[float | None] = mapped_column(Float, nullable=True)
    instability: Mapped[float | None] = mapped_column(Float, nullable=True)
    fan_in: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fan_out: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cycle_participation: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    cycle_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duplication_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    crap_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    change_frequency: Mapped[float | None] = mapped_column(Float, nullable=True)
    meta: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    captured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    scenario: Mapped["TwinScenario"] = relationship()
