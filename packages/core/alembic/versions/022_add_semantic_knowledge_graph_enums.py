"""Add semantic GraphRAG enum values for knowledge graph.

Revision ID: 022
Revises: 021
Create Date: 2026-03-05
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "022"
down_revision: str | None = "021"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Semantic GraphRAG node kinds
    op.execute("ALTER TYPE knowledge_node_kind ADD VALUE IF NOT EXISTS 'semantic_entity'")

    # Semantic GraphRAG edge kinds
    op.execute("ALTER TYPE knowledge_edge_kind ADD VALUE IF NOT EXISTS 'semantic_relationship'")
    op.execute("ALTER TYPE knowledge_edge_kind ADD VALUE IF NOT EXISTS 'file_mentions_entity'")


def downgrade() -> None:
    # PostgreSQL enum value removal requires full type recreation.
    # Keep downgrade as no-op to avoid destructive schema rewrite.
    return
