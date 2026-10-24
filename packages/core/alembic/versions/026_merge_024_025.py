"""Merge the AGE graph (024) and semantic extraction (025) heads.

Revision ID: 026
Revises: 024, 025
Create Date: 2026-10-24
"""

from collections.abc import Sequence

revision: str = "026"
down_revision: tuple[str, str] = ("024", "025")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
