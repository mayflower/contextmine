"""Add collections, collection_members, and collection_invites tables.

Revision ID: 004
Revises: 003
Create Date: 2024-01-04 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "004"
down_revision: str | None = "003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Create collections table (enum is auto-created)
    op.create_table(
        "collections",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("slug", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column(
            "visibility",
            sa.Enum("global", "private", name="collection_visibility", create_type=True),
            nullable=False,
        ),
        sa.Column("owner_user_id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["owner_user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("slug"),
    )
    op.create_index("ix_collections_owner_user_id", "collections", ["owner_user_id"])

    # Create collection_members table
    op.create_table(
        "collection_members",
        sa.Column("collection_id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("collection_id", "user_id"),
    )

    # Create collection_invites table
    op.create_table(
        "collection_invites",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("collection_id", sa.UUID(), nullable=False),
        sa.Column("github_login", sa.String(length=255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("collection_id", "github_login"),
    )
    op.create_index("ix_collection_invites_github_login", "collection_invites", ["github_login"])


def downgrade() -> None:
    op.drop_table("collection_invites")
    op.drop_table("collection_members")
    op.drop_table("collections")

    # Drop the enum
    sa.Enum(name="collection_visibility").drop(op.get_bind(), checkfirst=True)
