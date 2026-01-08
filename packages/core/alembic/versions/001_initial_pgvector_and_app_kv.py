"""Initial migration: enable pgvector extension and create app_kv table.

Revision ID: 001
Revises:
Create Date: 2024-01-01 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Check pgvector extension exists (created by postgres-operator or manually)
    conn = op.get_bind()
    result = conn.execute(sa.text("SELECT 1 FROM pg_extension WHERE extname = 'vector'"))
    if result.fetchone() is None:
        # Try to create - works in local dev, skipped if already exists
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create app_kv table
    op.create_table(
        "app_kv",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("key", sa.String(length=255), nullable=False),
        sa.Column("value", sa.Text(), nullable=True),
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
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("key"),
    )


def downgrade() -> None:
    op.drop_table("app_kv")
    op.execute("DROP EXTENSION IF EXISTS vector")
