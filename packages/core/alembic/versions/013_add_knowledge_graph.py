"""Add knowledge graph tables.

Revision ID: 013
Revises: 012
Create Date: 2024-12-30

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "013"
down_revision: str | None = "012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Create enum types
    op.execute(
        """
        CREATE TYPE knowledge_node_kind AS ENUM (
            'file', 'symbol', 'db_table', 'db_column', 'db_constraint',
            'api_endpoint', 'graphql_operation', 'message_schema', 'job',
            'rule_candidate', 'business_rule', 'bounded_context', 'arc42_section'
        )
        """
    )
    op.execute(
        """
        CREATE TYPE knowledge_edge_kind AS ENUM (
            'file_defines_symbol', 'symbol_contains_symbol', 'file_imports_file',
            'symbol_calls_symbol', 'symbol_references_symbol',
            'table_has_column', 'column_fk_to_column', 'table_has_constraint',
            'system_exposes_endpoint', 'endpoint_uses_schema', 'job_defined_in_file',
            'rule_derived_from_candidate', 'rule_evidenced_by',
            'documented_by', 'belongs_to_context'
        )
        """
    )
    op.execute(
        """
        CREATE TYPE knowledge_artifact_kind AS ENUM (
            'erd_mermaid', 'arc42', 'rule_catalog', 'surface_catalog'
        )
        """
    )

    # Create knowledge_evidence table
    op.create_table(
        "knowledge_evidence",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("document_id", sa.UUID(), nullable=True),
        sa.Column("chunk_id", sa.UUID(), nullable=True),
        sa.Column("file_path", sa.String(length=2048), nullable=False),
        sa.Column("start_line", sa.Integer(), nullable=False),
        sa.Column("end_line", sa.Integer(), nullable=False),
        sa.Column("snippet", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["chunk_id"], ["chunks.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create knowledge_nodes table
    op.create_table(
        "knowledge_nodes",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("collection_id", sa.UUID(), nullable=False),
        sa.Column(
            "kind",
            postgresql.ENUM(
                "file",
                "symbol",
                "db_table",
                "db_column",
                "db_constraint",
                "api_endpoint",
                "graphql_operation",
                "message_schema",
                "job",
                "rule_candidate",
                "business_rule",
                "bounded_context",
                "arc42_section",
                name="knowledge_node_kind",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column("natural_key", sa.String(length=2048), nullable=False),
        sa.Column("name", sa.String(length=512), nullable=False),
        sa.Column("meta", postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "collection_id", "kind", "natural_key", name="uq_knowledge_node_natural"
        ),
    )
    op.create_index(
        "ix_knowledge_node_collection_kind",
        "knowledge_nodes",
        ["collection_id", "kind"],
    )
    op.create_index(
        "ix_knowledge_node_meta",
        "knowledge_nodes",
        ["meta"],
        postgresql_using="gin",
    )

    # Create knowledge_edges table
    op.create_table(
        "knowledge_edges",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("collection_id", sa.UUID(), nullable=False),
        sa.Column("source_node_id", sa.UUID(), nullable=False),
        sa.Column("target_node_id", sa.UUID(), nullable=False),
        sa.Column(
            "kind",
            postgresql.ENUM(
                "file_defines_symbol",
                "symbol_contains_symbol",
                "file_imports_file",
                "symbol_calls_symbol",
                "symbol_references_symbol",
                "table_has_column",
                "column_fk_to_column",
                "table_has_constraint",
                "system_exposes_endpoint",
                "endpoint_uses_schema",
                "job_defined_in_file",
                "rule_derived_from_candidate",
                "rule_evidenced_by",
                "documented_by",
                "belongs_to_context",
                name="knowledge_edge_kind",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column("meta", postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_node_id"], ["knowledge_nodes.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["target_node_id"], ["knowledge_nodes.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_knowledge_edge_collection",
        "knowledge_edges",
        ["collection_id", "kind"],
    )
    op.create_index(
        "ix_knowledge_edge_source",
        "knowledge_edges",
        ["source_node_id", "kind"],
    )
    op.create_index(
        "ix_knowledge_edge_target",
        "knowledge_edges",
        ["target_node_id", "kind"],
    )

    # Create knowledge_node_evidence link table
    op.create_table(
        "knowledge_node_evidence",
        sa.Column("node_id", sa.UUID(), nullable=False),
        sa.Column("evidence_id", sa.UUID(), nullable=False),
        sa.ForeignKeyConstraint(["evidence_id"], ["knowledge_evidence.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["node_id"], ["knowledge_nodes.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("node_id", "evidence_id"),
    )

    # Create knowledge_edge_evidence link table
    op.create_table(
        "knowledge_edge_evidence",
        sa.Column("edge_id", sa.UUID(), nullable=False),
        sa.Column("evidence_id", sa.UUID(), nullable=False),
        sa.ForeignKeyConstraint(["edge_id"], ["knowledge_edges.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["evidence_id"], ["knowledge_evidence.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("edge_id", "evidence_id"),
    )

    # Create knowledge_artifacts table
    op.create_table(
        "knowledge_artifacts",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("collection_id", sa.UUID(), nullable=False),
        sa.Column(
            "kind",
            postgresql.ENUM(
                "erd_mermaid",
                "arc42",
                "rule_catalog",
                "surface_catalog",
                name="knowledge_artifact_kind",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column("name", sa.String(length=512), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("meta", postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("collection_id", "kind", "name", name="uq_knowledge_artifact_name"),
    )
    op.create_index(
        "ix_knowledge_artifact_collection_kind",
        "knowledge_artifacts",
        ["collection_id", "kind"],
    )

    # Create knowledge_artifact_evidence link table
    op.create_table(
        "knowledge_artifact_evidence",
        sa.Column("artifact_id", sa.UUID(), nullable=False),
        sa.Column("evidence_id", sa.UUID(), nullable=False),
        sa.ForeignKeyConstraint(["artifact_id"], ["knowledge_artifacts.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["evidence_id"], ["knowledge_evidence.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("artifact_id", "evidence_id"),
    )


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table("knowledge_artifact_evidence")
    op.drop_table("knowledge_artifacts")
    op.drop_table("knowledge_edge_evidence")
    op.drop_table("knowledge_node_evidence")
    op.drop_table("knowledge_edges")
    op.drop_table("knowledge_nodes")
    op.drop_table("knowledge_evidence")

    # Drop enum types
    op.execute("DROP TYPE knowledge_artifact_kind")
    op.execute("DROP TYPE knowledge_edge_kind")
    op.execute("DROP TYPE knowledge_node_kind")
