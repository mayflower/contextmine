"""Add behavioral digital twin enum values.

Revision ID: 020
Revises: 019
Create Date: 2026-02-23
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "020"
down_revision: str | None = "019"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Knowledge node kinds
    op.execute("ALTER TYPE knowledge_node_kind ADD VALUE IF NOT EXISTS 'test_suite'")
    op.execute("ALTER TYPE knowledge_node_kind ADD VALUE IF NOT EXISTS 'test_case'")
    op.execute("ALTER TYPE knowledge_node_kind ADD VALUE IF NOT EXISTS 'test_fixture'")
    op.execute("ALTER TYPE knowledge_node_kind ADD VALUE IF NOT EXISTS 'ui_route'")
    op.execute("ALTER TYPE knowledge_node_kind ADD VALUE IF NOT EXISTS 'ui_view'")
    op.execute("ALTER TYPE knowledge_node_kind ADD VALUE IF NOT EXISTS 'ui_component'")
    op.execute("ALTER TYPE knowledge_node_kind ADD VALUE IF NOT EXISTS 'user_flow'")
    op.execute("ALTER TYPE knowledge_node_kind ADD VALUE IF NOT EXISTS 'flow_step'")
    op.execute("ALTER TYPE knowledge_node_kind ADD VALUE IF NOT EXISTS 'interface_contract'")

    # Knowledge edge kinds
    op.execute("ALTER TYPE knowledge_edge_kind ADD VALUE IF NOT EXISTS 'test_case_covers_symbol'")
    op.execute("ALTER TYPE knowledge_edge_kind ADD VALUE IF NOT EXISTS 'test_case_validates_rule'")
    op.execute("ALTER TYPE knowledge_edge_kind ADD VALUE IF NOT EXISTS 'test_uses_fixture'")
    op.execute("ALTER TYPE knowledge_edge_kind ADD VALUE IF NOT EXISTS 'ui_route_renders_view'")
    op.execute(
        "ALTER TYPE knowledge_edge_kind ADD VALUE IF NOT EXISTS 'ui_view_composes_component'"
    )
    op.execute("ALTER TYPE knowledge_edge_kind ADD VALUE IF NOT EXISTS 'flow_step_calls_endpoint'")
    op.execute("ALTER TYPE knowledge_edge_kind ADD VALUE IF NOT EXISTS 'user_flow_has_step'")
    op.execute("ALTER TYPE knowledge_edge_kind ADD VALUE IF NOT EXISTS 'test_case_verifies_flow'")
    op.execute("ALTER TYPE knowledge_edge_kind ADD VALUE IF NOT EXISTS 'contract_governs_endpoint'")

    # Artifact kind
    op.execute("ALTER TYPE knowledge_artifact_kind ADD VALUE IF NOT EXISTS 'twin_manifest'")


def downgrade() -> None:
    # PostgreSQL enum value removal requires full type recreation.
    # Keep downgrade as no-op to avoid destructive schema rewrite.
    return
