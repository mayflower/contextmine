"""Database models."""

import enum
import uuid
from datetime import datetime

from contextmine_core.database import Base
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
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
    ARC42_SECTION = "arc42_section"


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
    ARC42 = "arc42"
    RULE_CATALOG = "rule_catalog"
    SURFACE_CATALOG = "surface_catalog"


class EmbeddingTargetType(enum.Enum):
    """Target type for knowledge embeddings."""

    NODE = "node"
    COMMUNITY = "community"


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
