"""Add symbols and symbol_edges tables for code graph.

Revision ID: 011
Revises: 010
Create Date: 2024-12-29 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "011"
down_revision: str | None = "010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Note: Enums are created automatically by SQLAlchemy when creating the table
    # The sa.Enum(...) column definition handles enum creation

    # Create symbols table
    op.create_table(
        "symbols",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("document_id", sa.UUID(), nullable=False),
        sa.Column("qualified_name", sa.String(length=1024), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column(
            "kind",
            sa.Enum(
                "function",
                "class",
                "method",
                "variable",
                "constant",
                "module",
                "interface",
                "enum",
                "property",
                "type_alias",
                name="symbol_kind",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column("start_line", sa.Integer(), nullable=False),
        sa.Column("end_line", sa.Integer(), nullable=False),
        sa.Column("signature", sa.Text(), nullable=True),
        sa.Column("parent_name", sa.String(length=1024), nullable=True),
        sa.Column("meta", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    # Indexes for symbols
    op.create_index("ix_symbols_document_id", "symbols", ["document_id"])
    op.create_index("ix_symbols_name", "symbols", ["name"])
    op.create_index("ix_symbols_qualified_name", "symbols", ["qualified_name"])
    # Composite index for document + qualified_name uniqueness check
    op.create_index(
        "ix_symbols_document_qualified",
        "symbols",
        ["document_id", "qualified_name"],
        unique=True,
    )

    # Create symbol_edges table
    op.create_table(
        "symbol_edges",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("source_symbol_id", sa.UUID(), nullable=False),
        sa.Column("target_symbol_id", sa.UUID(), nullable=False),
        sa.Column(
            "edge_type",
            sa.Enum(
                "calls",
                "imports",
                "inherits",
                "implements",
                "contains",
                "references",
                name="symbol_edge_type",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column("source_line", sa.Integer(), nullable=True),
        sa.Column("meta", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["source_symbol_id"], ["symbols.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["target_symbol_id"], ["symbols.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    # Indexes for symbol_edges
    op.create_index("ix_symbol_edges_source", "symbol_edges", ["source_symbol_id"])
    op.create_index("ix_symbol_edges_target", "symbol_edges", ["target_symbol_id"])
    op.create_index("ix_symbol_edges_type", "symbol_edges", ["edge_type"])
    # Composite index for finding edges by source and type
    op.create_index(
        "ix_symbol_edges_source_type",
        "symbol_edges",
        ["source_symbol_id", "edge_type"],
    )


def downgrade() -> None:
    op.drop_table("symbol_edges")
    op.drop_table("symbols")
    op.execute("DROP TYPE symbol_edge_type")
    op.execute("DROP TYPE symbol_kind")
