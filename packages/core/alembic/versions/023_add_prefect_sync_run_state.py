"""Link ContextMine sync runs to Prefect flow runs.

Revision ID: 023
Revises: 022
Create Date: 2026-09-01
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "023"
down_revision: str | None = "022"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(sa.text("ALTER TYPE sync_run_status ADD VALUE IF NOT EXISTS 'scheduled'"))
    op.execute(sa.text("ALTER TYPE sync_run_status ADD VALUE IF NOT EXISTS 'cancelled'"))
    op.execute(sa.text("ALTER TYPE sync_run_status ADD VALUE IF NOT EXISTS 'timed_out'"))
    op.add_column("sync_runs", sa.Column("flow_run_id", sa.String(length=255), nullable=True))
    op.create_index(
        "uq_sync_runs_flow_run_id",
        "sync_runs",
        ["flow_run_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_sync_runs_flow_run_id", table_name="sync_runs")
    op.drop_column("sync_runs", "flow_run_id")
