"""Add digital twin and architecture intent engine tables.

Revision ID: 015
Revises: 014
Create Date: 2026-02-15

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "015"
down_revision: str | None = "014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Required for AGE-backed cypher queries.
    op.execute("CREATE EXTENSION IF NOT EXISTS age")

    # Extend artifact enum with new export kinds.
    op.execute("ALTER TYPE knowledge_artifact_kind ADD VALUE IF NOT EXISTS 'lpg_jsonl'")
    op.execute("ALTER TYPE knowledge_artifact_kind ADD VALUE IF NOT EXISTS 'cc_json'")
    op.execute("ALTER TYPE knowledge_artifact_kind ADD VALUE IF NOT EXISTS 'cx2'")
    op.execute("ALTER TYPE knowledge_artifact_kind ADD VALUE IF NOT EXISTS 'jgf'")
    op.execute("ALTER TYPE knowledge_artifact_kind ADD VALUE IF NOT EXISTS 'mermaid_c4_asis'")
    op.execute("ALTER TYPE knowledge_artifact_kind ADD VALUE IF NOT EXISTS 'mermaid_c4_tobe'")

    op.execute(
        "CREATE TYPE twin_layer AS ENUM ('portfolio_system', 'domain_container', "
        "'component_interface', 'code_controlflow')"
    )
    op.execute(
        "CREATE TYPE architecture_intent_action AS ENUM ("
        "'extract_domain', 'split_container', 'move_component', "
        "'define_interface', 'set_validator', 'apply_data_boundary')"
    )
    op.execute(
        "CREATE TYPE architecture_intent_status AS ENUM ("
        "'pending', 'blocked', 'approved', 'executed', 'failed')"
    )
    op.execute("CREATE TYPE intent_risk_level AS ENUM ('low', 'high')")
    op.execute("CREATE TYPE validation_source_kind AS ENUM ('tekton', 'argo', 'temporal')")

    op.create_table(
        "twin_scenarios",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("collection_id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("base_scenario_id", sa.UUID(), nullable=True),
        sa.Column("is_as_is", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("meta", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_by_user_id", sa.UUID(), nullable=True),
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
        sa.ForeignKeyConstraint(["base_scenario_id"], ["twin_scenarios.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["created_by_user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_twin_scenario_collection", "twin_scenarios", ["collection_id"])
    op.create_index("ix_twin_scenario_parent", "twin_scenarios", ["base_scenario_id"])

    op.create_table(
        "twin_nodes",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("scenario_id", sa.UUID(), nullable=False),
        sa.Column("natural_key", sa.String(length=2048), nullable=False),
        sa.Column("kind", sa.String(length=128), nullable=False),
        sa.Column("name", sa.String(length=512), nullable=False),
        sa.Column("meta", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("provenance_node_id", sa.UUID(), nullable=True),
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
        sa.ForeignKeyConstraint(
            ["provenance_node_id"], ["knowledge_nodes.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(["scenario_id"], ["twin_scenarios.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("scenario_id", "natural_key", name="uq_twin_node_natural"),
    )
    op.create_index("ix_twin_node_scenario_kind", "twin_nodes", ["scenario_id", "kind"])

    op.create_table(
        "twin_edges",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("scenario_id", sa.UUID(), nullable=False),
        sa.Column("source_node_id", sa.UUID(), nullable=False),
        sa.Column("target_node_id", sa.UUID(), nullable=False),
        sa.Column("kind", sa.String(length=128), nullable=False),
        sa.Column("meta", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["scenario_id"], ["twin_scenarios.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_node_id"], ["twin_nodes.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["target_node_id"], ["twin_nodes.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "scenario_id",
            "source_node_id",
            "target_node_id",
            "kind",
            name="uq_twin_edge_unique",
        ),
    )
    op.create_index("ix_twin_edge_scenario_kind", "twin_edges", ["scenario_id", "kind"])

    op.create_table(
        "twin_node_layers",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("node_id", sa.UUID(), nullable=False),
        sa.Column(
            "layer",
            postgresql.ENUM(
                "portfolio_system",
                "domain_container",
                "component_interface",
                "code_controlflow",
                name="twin_layer",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["node_id"], ["twin_nodes.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("node_id", "layer", name="uq_twin_node_layer"),
    )

    op.create_table(
        "twin_edge_layers",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("edge_id", sa.UUID(), nullable=False),
        sa.Column(
            "layer",
            postgresql.ENUM(
                "portfolio_system",
                "domain_container",
                "component_interface",
                "code_controlflow",
                name="twin_layer",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["edge_id"], ["twin_edges.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("edge_id", "layer", name="uq_twin_edge_layer"),
    )

    op.create_table(
        "architecture_intents",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("scenario_id", sa.UUID(), nullable=False),
        sa.Column("intent_version", sa.String(length=16), nullable=False),
        sa.Column(
            "action",
            postgresql.ENUM(
                "extract_domain",
                "split_container",
                "move_component",
                "define_interface",
                "set_validator",
                "apply_data_boundary",
                name="architecture_intent_action",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column("target_type", sa.String(length=64), nullable=False),
        sa.Column("target_id", sa.String(length=2048), nullable=False),
        sa.Column("params", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("expected_scenario_version", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            postgresql.ENUM(
                "pending",
                "blocked",
                "approved",
                "executed",
                "failed",
                name="architecture_intent_status",
                create_type=False,
            ),
            nullable=False,
            server_default="pending",
        ),
        sa.Column(
            "risk_level",
            postgresql.ENUM(
                "low",
                "high",
                name="intent_risk_level",
                create_type=False,
            ),
            nullable=False,
            server_default="low",
        ),
        sa.Column(
            "requires_approval", sa.Boolean(), nullable=False, server_default=sa.text("false")
        ),
        sa.Column("requested_by_user_id", sa.UUID(), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
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
        sa.ForeignKeyConstraint(["requested_by_user_id"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["scenario_id"], ["twin_scenarios.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_arch_intent_requested_by", "architecture_intents", ["requested_by_user_id"])
    op.create_index("ix_arch_intent_scenario", "architecture_intents", ["scenario_id", "status"])

    op.create_table(
        "architecture_intent_runs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("intent_id", sa.UUID(), nullable=False),
        sa.Column("scenario_version_before", sa.Integer(), nullable=False),
        sa.Column("scenario_version_after", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["intent_id"], ["architecture_intents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_arch_intent_run_intent", "architecture_intent_runs", ["intent_id"])

    op.create_table(
        "twin_patches",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("scenario_id", sa.UUID(), nullable=False),
        sa.Column("scenario_version", sa.Integer(), nullable=False),
        sa.Column("intent_id", sa.UUID(), nullable=True),
        sa.Column("patch_ops", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_by_user_id", sa.UUID(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["created_by_user_id"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["intent_id"], ["architecture_intents.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["scenario_id"], ["twin_scenarios.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("scenario_id", "scenario_version", name="uq_twin_patch_version"),
    )
    op.create_index("ix_twin_patch_scenario", "twin_patches", ["scenario_id", "scenario_version"])

    op.create_table(
        "validation_snapshots",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("collection_id", sa.UUID(), nullable=True),
        sa.Column(
            "source_kind",
            postgresql.ENUM(
                "tekton",
                "argo",
                "temporal",
                name="validation_source_kind",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column("metric_key", sa.String(length=128), nullable=False),
        sa.Column("metric_value", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=True),
        sa.Column("meta", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "captured_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_validation_snapshot_collection",
        "validation_snapshots",
        ["collection_id", "captured_at"],
    )
    op.create_index(
        "ix_validation_snapshot_source",
        "validation_snapshots",
        ["source_kind", "metric_key"],
    )

    op.create_table(
        "metric_snapshots",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("scenario_id", sa.UUID(), nullable=False),
        sa.Column("node_natural_key", sa.String(length=2048), nullable=False),
        sa.Column("loc", sa.Integer(), nullable=True),
        sa.Column("symbol_count", sa.Integer(), nullable=True),
        sa.Column("coupling", sa.Float(), nullable=True),
        sa.Column("coverage", sa.Float(), nullable=True),
        sa.Column("complexity", sa.Float(), nullable=True),
        sa.Column("change_frequency", sa.Float(), nullable=True),
        sa.Column("meta", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
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
        "ix_metric_snapshot_scenario", "metric_snapshots", ["scenario_id", "captured_at"]
    )
    op.create_index(
        "ix_metric_snapshot_node", "metric_snapshots", ["scenario_id", "node_natural_key"]
    )


def downgrade() -> None:
    op.drop_index("ix_metric_snapshot_node", table_name="metric_snapshots")
    op.drop_index("ix_metric_snapshot_scenario", table_name="metric_snapshots")
    op.drop_table("metric_snapshots")

    op.drop_index("ix_validation_snapshot_source", table_name="validation_snapshots")
    op.drop_index("ix_validation_snapshot_collection", table_name="validation_snapshots")
    op.drop_table("validation_snapshots")

    op.drop_index("ix_twin_patch_scenario", table_name="twin_patches")
    op.drop_table("twin_patches")

    op.drop_index("ix_arch_intent_run_intent", table_name="architecture_intent_runs")
    op.drop_table("architecture_intent_runs")

    op.drop_index("ix_arch_intent_scenario", table_name="architecture_intents")
    op.drop_index("ix_arch_intent_requested_by", table_name="architecture_intents")
    op.drop_table("architecture_intents")

    op.drop_table("twin_edge_layers")
    op.drop_table("twin_node_layers")

    op.drop_index("ix_twin_edge_scenario_kind", table_name="twin_edges")
    op.drop_table("twin_edges")

    op.drop_index("ix_twin_node_scenario_kind", table_name="twin_nodes")
    op.drop_table("twin_nodes")

    op.drop_index("ix_twin_scenario_parent", table_name="twin_scenarios")
    op.drop_index("ix_twin_scenario_collection", table_name="twin_scenarios")
    op.drop_table("twin_scenarios")

    op.execute("DROP TYPE validation_source_kind")
    op.execute("DROP TYPE intent_risk_level")
    op.execute("DROP TYPE architecture_intent_status")
    op.execute("DROP TYPE architecture_intent_action")
    op.execute("DROP TYPE twin_layer")
