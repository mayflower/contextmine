"""Add embedding_models table and embedding columns to chunks.

Revision ID: 009
Revises: 008
Create Date: 2024-01-09 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "009"
down_revision: str | None = "008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Create embedding provider enum (use DO block for IF NOT EXISTS in older PG)
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE embedding_provider AS ENUM ('openai', 'gemini');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)

    # Create embedding_models table using raw SQL to avoid SQLAlchemy enum auto-creation
    op.execute("""
        CREATE TABLE embedding_models (
            id UUID PRIMARY KEY,
            provider embedding_provider NOT NULL,
            model_name VARCHAR(255) NOT NULL,
            dimension INTEGER NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (provider, model_name)
        )
    """)

    # Add embedding columns to chunks
    op.add_column(
        "chunks",
        sa.Column("embedding_model_id", sa.UUID(), nullable=True),
    )
    op.add_column(
        "chunks",
        sa.Column("embedded_at", sa.DateTime(timezone=True), nullable=True),
    )

    # Add FK constraint
    op.create_foreign_key(
        "fk_chunks_embedding_model",
        "chunks",
        "embedding_models",
        ["embedding_model_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # Create index on embedding_model_id
    op.create_index("ix_chunks_embedding_model_id", "chunks", ["embedding_model_id"])

    # Add vector column for embeddings
    # Using 1536 dimensions (OpenAI text-embedding-3-small) as default
    # Note: pgvector requires fixed dimensions for HNSW index
    op.execute("ALTER TABLE chunks ADD COLUMN embedding vector(1536)")

    # Create HNSW index for vector similarity search
    op.execute("""
        CREATE INDEX ix_chunks_embedding_hnsw
        ON chunks USING hnsw (embedding vector_cosine_ops)
    """)

    # Insert default embedding models
    op.execute("""
        INSERT INTO embedding_models (id, provider, model_name, dimension)
        VALUES
            (gen_random_uuid(), 'openai', 'text-embedding-3-small', 1536),
            (gen_random_uuid(), 'openai', 'text-embedding-3-large', 3072),
            (gen_random_uuid(), 'gemini', 'text-embedding-004', 768)
    """)

    # Add config column to collections for per-collection settings (e.g., embedding model)
    op.add_column(
        "collections",
        sa.Column("config", sa.JSON(), nullable=False, server_default="{}"),
    )


def downgrade() -> None:
    # Drop config column from collections
    op.drop_column("collections", "config")

    # Drop index
    op.execute("DROP INDEX IF EXISTS ix_chunks_embedding_hnsw")

    # Drop embedding column
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS embedding")

    # Drop FK and columns from chunks
    op.drop_constraint("fk_chunks_embedding_model", "chunks", type_="foreignkey")
    op.drop_index("ix_chunks_embedding_model_id", "chunks")
    op.drop_column("chunks", "embedded_at")
    op.drop_column("chunks", "embedding_model_id")

    # Drop embedding_models table
    op.drop_table("embedding_models")

    # Drop enum
    op.execute("DROP TYPE IF EXISTS embedding_provider")
