"""Add GraphRAG community and embedding tables.

Revision ID: 014
Revises: 013
Create Date: 2024-12-31

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "014"
down_revision: str | None = "013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Add missing values to knowledge_node_kind enum
    op.execute("ALTER TYPE knowledge_node_kind ADD VALUE IF NOT EXISTS 'graphql_type'")
    op.execute("ALTER TYPE knowledge_node_kind ADD VALUE IF NOT EXISTS 'service_rpc'")

    # Add missing values to knowledge_edge_kind enum
    op.execute("ALTER TYPE knowledge_edge_kind ADD VALUE IF NOT EXISTS 'rpc_uses_message'")
    op.execute("ALTER TYPE knowledge_edge_kind ADD VALUE IF NOT EXISTS 'job_depends_on'")

    # Fix knowledge_artifact_kind enum: rename erd_mermaid to mermaid_erd
    # PostgreSQL doesn't support renaming enum values directly, so we need to:
    # 1. Add the new value
    # 2. Update existing rows
    # 3. (Can't remove old value in PostgreSQL without recreating the type)
    op.execute("ALTER TYPE knowledge_artifact_kind ADD VALUE IF NOT EXISTS 'mermaid_erd'")

    # Create embedding_target_type enum
    op.execute("CREATE TYPE embedding_target_type AS ENUM ('node', 'community')")

    # Create knowledge_communities table
    op.create_table(
        "knowledge_communities",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("collection_id", sa.UUID(), nullable=False),
        sa.Column("level", sa.Integer(), nullable=False),
        sa.Column("natural_key", sa.String(length=2048), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
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
            "collection_id", "level", "natural_key", name="uq_knowledge_community_natural"
        ),
    )
    op.create_index(
        "ix_knowledge_community_collection_level",
        "knowledge_communities",
        ["collection_id", "level"],
    )

    # Create community_members table
    op.create_table(
        "community_members",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("community_id", sa.UUID(), nullable=False),
        sa.Column("node_id", sa.UUID(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["community_id"], ["knowledge_communities.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["node_id"], ["knowledge_nodes.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("community_id", "node_id", name="uq_community_member"),
    )
    op.create_index(
        "ix_community_member_community",
        "community_members",
        ["community_id"],
    )
    op.create_index(
        "ix_community_member_node",
        "community_members",
        ["node_id"],
    )

    # Create knowledge_embeddings table
    op.create_table(
        "knowledge_embeddings",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("collection_id", sa.UUID(), nullable=False),
        sa.Column(
            "target_type",
            postgresql.ENUM(
                "node",
                "community",
                name="embedding_target_type",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column("target_id", sa.UUID(), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
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
            "collection_id",
            "target_type",
            "target_id",
            "model_name",
            name="uq_knowledge_embedding_target",
        ),
    )
    op.create_index(
        "ix_knowledge_embedding_collection",
        "knowledge_embeddings",
        ["collection_id", "target_type"],
    )

    # Add pgvector embedding column (using same dimension as existing chunks table)
    # The dimension will be set based on the embedding model used
    op.execute(
        """
        ALTER TABLE knowledge_embeddings
        ADD COLUMN embedding vector(1536)
        """
    )

    # Create index for vector similarity search
    op.execute(
        """
        CREATE INDEX ix_knowledge_embedding_vector
        ON knowledge_embeddings
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
        """
    )


def downgrade() -> None:
    # Drop vector index
    op.execute("DROP INDEX IF EXISTS ix_knowledge_embedding_vector")

    # Drop tables in reverse order
    op.drop_table("knowledge_embeddings")
    op.drop_table("community_members")
    op.drop_table("knowledge_communities")

    # Drop embedding_target_type enum
    op.execute("DROP TYPE embedding_target_type")

    # Note: Cannot easily remove enum values in PostgreSQL
    # The added enum values (graphql_type, service_rpc, rpc_uses_message,
    # job_depends_on, mermaid_erd) will remain in their respective types
