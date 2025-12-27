"""Add deploy key fields to sources.

Revision ID: 010
Revises: 009
Create Date: 2025-01-01
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "010"
down_revision: str | None = "009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add deploy_key_encrypted and deploy_key_fingerprint columns to sources."""
    op.add_column(
        "sources",
        sa.Column("deploy_key_encrypted", sa.Text(), nullable=True),
    )
    op.add_column(
        "sources",
        sa.Column("deploy_key_fingerprint", sa.String(255), nullable=True),
    )


def downgrade() -> None:
    """Remove deploy key columns from sources."""
    op.drop_column("sources", "deploy_key_fingerprint")
    op.drop_column("sources", "deploy_key_encrypted")
