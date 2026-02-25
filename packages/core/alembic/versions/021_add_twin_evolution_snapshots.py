"""Add evolution snapshot tables for ownership and temporal coupling.

Revision ID: 021
Revises: 020
Create Date: 2026-02-25
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "021"
down_revision: str | None = "020"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "twin_ownership_snapshots",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("scenario_id", sa.UUID(), nullable=False),
        sa.Column("node_natural_key", sa.String(length=2048), nullable=False),
        sa.Column("author_key", sa.String(length=320), nullable=False),
        sa.Column("author_label", sa.String(length=320), nullable=False),
        sa.Column("additions", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("deletions", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("touches", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("ownership_share", sa.Float(), nullable=False, server_default="0"),
        sa.Column("last_touched_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("window_days", sa.Integer(), nullable=False, server_default="365"),
        sa.Column(
            "captured_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["scenario_id"], ["twin_scenarios.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_twin_ownership_scenario_node",
        "twin_ownership_snapshots",
        ["scenario_id", "node_natural_key"],
    )
    op.create_index(
        "ix_twin_ownership_scenario_author",
        "twin_ownership_snapshots",
        ["scenario_id", "author_key"],
    )
    op.create_index(
        "ix_twin_ownership_scenario_captured",
        "twin_ownership_snapshots",
        ["scenario_id", "captured_at"],
    )

    op.create_table(
        "twin_temporal_coupling_snapshots",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("scenario_id", sa.UUID(), nullable=False),
        sa.Column("entity_level", sa.String(length=32), nullable=False),
        sa.Column("source_key", sa.String(length=2048), nullable=False),
        sa.Column("target_key", sa.String(length=2048), nullable=False),
        sa.Column("co_change_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("source_change_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("target_change_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("ratio_source_to_target", sa.Float(), nullable=False, server_default="0"),
        sa.Column("ratio_target_to_source", sa.Float(), nullable=False, server_default="0"),
        sa.Column("jaccard", sa.Float(), nullable=False, server_default="0"),
        sa.Column("cross_boundary", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("window_days", sa.Integer(), nullable=False, server_default="365"),
        sa.Column(
            "captured_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "entity_level IN ('file','container','component')",
            name="ck_twin_temporal_coupling_entity_level",
        ),
        sa.ForeignKeyConstraint(["scenario_id"], ["twin_scenarios.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "scenario_id",
            "entity_level",
            "source_key",
            "target_key",
            "window_days",
            name="uq_twin_temporal_coupling_key",
        ),
    )
    op.create_index(
        "ix_twin_temporal_coupling_scenario_level",
        "twin_temporal_coupling_snapshots",
        ["scenario_id", "entity_level", "captured_at"],
    )
    op.create_index(
        "ix_twin_temporal_coupling_scenario_level_jaccard",
        "twin_temporal_coupling_snapshots",
        ["scenario_id", "entity_level", "jaccard"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_twin_temporal_coupling_scenario_level_jaccard",
        table_name="twin_temporal_coupling_snapshots",
    )
    op.drop_index(
        "ix_twin_temporal_coupling_scenario_level",
        table_name="twin_temporal_coupling_snapshots",
    )
    op.drop_table("twin_temporal_coupling_snapshots")

    op.drop_index("ix_twin_ownership_scenario_captured", table_name="twin_ownership_snapshots")
    op.drop_index("ix_twin_ownership_scenario_author", table_name="twin_ownership_snapshots")
    op.drop_index("ix_twin_ownership_scenario_node", table_name="twin_ownership_snapshots")
    op.drop_table("twin_ownership_snapshots")
