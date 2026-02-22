"""Add first-class quality metric columns to metric_snapshots.

Revision ID: 019
Revises: 018
Create Date: 2026-02-22
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "019"
down_revision: str | None = "018"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("metric_snapshots", sa.Column("cohesion", sa.Float(), nullable=True))
    op.add_column("metric_snapshots", sa.Column("instability", sa.Float(), nullable=True))
    op.add_column("metric_snapshots", sa.Column("fan_in", sa.Integer(), nullable=True))
    op.add_column("metric_snapshots", sa.Column("fan_out", sa.Integer(), nullable=True))
    op.add_column("metric_snapshots", sa.Column("cycle_participation", sa.Boolean(), nullable=True))
    op.add_column("metric_snapshots", sa.Column("cycle_size", sa.Integer(), nullable=True))
    op.add_column("metric_snapshots", sa.Column("duplication_ratio", sa.Float(), nullable=True))
    op.add_column("metric_snapshots", sa.Column("crap_score", sa.Float(), nullable=True))

    # Backfill from snapshot meta when present.
    op.execute(
        sa.text(
            """
            UPDATE metric_snapshots
            SET cohesion = NULLIF(meta->>'cohesion', '')::double precision
            WHERE meta ? 'cohesion'
            """
        )
    )
    op.execute(
        sa.text(
            """
            UPDATE metric_snapshots
            SET instability = NULLIF(meta->>'instability', '')::double precision
            WHERE meta ? 'instability'
            """
        )
    )
    op.execute(
        sa.text(
            """
            UPDATE metric_snapshots
            SET fan_in = NULLIF(meta->>'fan_in', '')::integer
            WHERE meta ? 'fan_in'
            """
        )
    )
    op.execute(
        sa.text(
            """
            UPDATE metric_snapshots
            SET fan_out = NULLIF(meta->>'fan_out', '')::integer
            WHERE meta ? 'fan_out'
            """
        )
    )
    op.execute(
        sa.text(
            """
            UPDATE metric_snapshots
            SET cycle_participation = NULLIF(meta->>'cycle_participation', '')::boolean
            WHERE meta ? 'cycle_participation'
            """
        )
    )
    op.execute(
        sa.text(
            """
            UPDATE metric_snapshots
            SET cycle_size = NULLIF(meta->>'cycle_size', '')::integer
            WHERE meta ? 'cycle_size'
            """
        )
    )
    op.execute(
        sa.text(
            """
            UPDATE metric_snapshots
            SET duplication_ratio = NULLIF(meta->>'duplication_ratio', '')::double precision
            WHERE meta ? 'duplication_ratio'
            """
        )
    )
    op.execute(
        sa.text(
            """
            UPDATE metric_snapshots
            SET crap_score = NULLIF(meta->>'crap_score', '')::double precision
            WHERE meta ? 'crap_score'
            """
        )
    )
    op.execute(
        sa.text(
            """
            UPDATE metric_snapshots
            SET crap_score = (
                (complexity * complexity) *
                power((1.0 - (LEAST(GREATEST(COALESCE(coverage, 0.0), 0.0), 100.0) / 100.0)), 3)
            ) + complexity
            WHERE crap_score IS NULL
              AND complexity IS NOT NULL
              AND coverage IS NOT NULL
            """
        )
    )


def downgrade() -> None:
    op.drop_column("metric_snapshots", "crap_score")
    op.drop_column("metric_snapshots", "duplication_ratio")
    op.drop_column("metric_snapshots", "cycle_size")
    op.drop_column("metric_snapshots", "cycle_participation")
    op.drop_column("metric_snapshots", "fan_out")
    op.drop_column("metric_snapshots", "fan_in")
    op.drop_column("metric_snapshots", "instability")
    op.drop_column("metric_snapshots", "cohesion")
