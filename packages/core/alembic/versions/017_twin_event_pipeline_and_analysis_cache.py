"""Add twin event pipeline, analysis cache, and findings tables.

Revision ID: 017
Revises: 016
Create Date: 2026-02-22

"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "017"
down_revision: str | None = "016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "twin_source_versions",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("collection_id", sa.UUID(), nullable=False),
        sa.Column("source_id", sa.UUID(), nullable=False),
        sa.Column("revision_key", sa.String(length=255), nullable=False),
        sa.Column("extractor_version", sa.String(length=64), nullable=False),
        sa.Column("language_profile", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="queued"),
        sa.Column("stats", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("joern_status", sa.String(length=32), nullable=False, server_default="pending"),
        sa.Column("joern_project", sa.String(length=255), nullable=True),
        sa.Column("joern_cpg_path", sa.String(length=2048), nullable=True),
        sa.Column("joern_server_url", sa.String(length=512), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_id"], ["sources.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "source_id",
            "revision_key",
            "extractor_version",
            name="uq_twin_source_version_revision",
        ),
    )
    op.create_index(
        "ix_twin_source_version_collection",
        "twin_source_versions",
        ["collection_id", "finished_at"],
    )
    op.create_index(
        "ix_twin_source_version_source_status",
        "twin_source_versions",
        ["source_id", "status"],
    )

    op.create_table(
        "twin_events",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("collection_id", sa.UUID(), nullable=False),
        sa.Column("scenario_id", sa.UUID(), nullable=True),
        sa.Column("source_id", sa.UUID(), nullable=True),
        sa.Column("source_version_id", sa.UUID(), nullable=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="queued"),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "event_ts",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("idempotency_key", sa.String(length=255), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["scenario_id"], ["twin_scenarios.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["source_id"], ["sources.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(
            ["source_version_id"], ["twin_source_versions.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("idempotency_key", name="uq_twin_event_idempotency"),
    )
    op.create_index("ix_twin_event_collection_ts", "twin_events", ["collection_id", "event_ts"])
    op.create_index("ix_twin_event_source_ts", "twin_events", ["source_id", "event_ts"])
    op.create_index("ix_twin_event_scenario_ts", "twin_events", ["scenario_id", "event_ts"])

    op.create_table(
        "twin_analysis_cache",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("cache_key", sa.String(length=128), nullable=False),
        sa.Column("scenario_id", sa.UUID(), nullable=False),
        sa.Column("tool_name", sa.String(length=128), nullable=False),
        sa.Column("params_hash", sa.String(length=128), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["scenario_id"], ["twin_scenarios.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "scenario_id",
            "tool_name",
            "params_hash",
            "cache_key",
            name="uq_twin_analysis_cache_key",
        ),
    )
    op.create_index(
        "ix_twin_analysis_cache_lookup",
        "twin_analysis_cache",
        ["scenario_id", "tool_name", "expires_at"],
    )

    op.create_table(
        "twin_findings",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("scenario_id", sa.UUID(), nullable=False),
        sa.Column("source_version_id", sa.UUID(), nullable=True),
        sa.Column("finding_type", sa.String(length=128), nullable=False),
        sa.Column("severity", sa.String(length=32), nullable=False),
        sa.Column("confidence", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="open"),
        sa.Column("filename", sa.String(length=2048), nullable=False),
        sa.Column("line_number", sa.Integer(), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("flow_data", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("meta", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
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
        sa.ForeignKeyConstraint(["scenario_id"], ["twin_scenarios.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["source_version_id"], ["twin_source_versions.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_twin_findings_scenario_created",
        "twin_findings",
        ["scenario_id", "created_at"],
    )
    op.create_index("ix_twin_findings_status", "twin_findings", ["scenario_id", "status"])
    op.create_index("ix_twin_findings_type", "twin_findings", ["scenario_id", "finding_type"])

    op.add_column("twin_nodes", sa.Column("source_id", sa.UUID(), nullable=True))
    op.add_column("twin_nodes", sa.Column("source_version_id", sa.UUID(), nullable=True))
    op.add_column(
        "twin_nodes",
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
    )
    op.add_column(
        "twin_nodes",
        sa.Column(
            "first_seen_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.add_column(
        "twin_nodes",
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_foreign_key(
        "fk_twin_nodes_source_id",
        "twin_nodes",
        "sources",
        ["source_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_twin_nodes_source_version_id",
        "twin_nodes",
        "twin_source_versions",
        ["source_version_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_twin_node_source_id", "twin_nodes", ["source_id"])
    op.create_index("ix_twin_node_source_version_id", "twin_nodes", ["source_version_id"])
    op.create_index(
        "ix_twin_node_active_seen", "twin_nodes", ["scenario_id", "is_active", "last_seen_at"]
    )

    op.add_column("twin_edges", sa.Column("source_id", sa.UUID(), nullable=True))
    op.add_column("twin_edges", sa.Column("source_version_id", sa.UUID(), nullable=True))
    op.add_column(
        "twin_edges",
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
    )
    op.add_column(
        "twin_edges",
        sa.Column(
            "first_seen_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.add_column(
        "twin_edges",
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_foreign_key(
        "fk_twin_edges_source_id",
        "twin_edges",
        "sources",
        ["source_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_twin_edges_source_version_id",
        "twin_edges",
        "twin_source_versions",
        ["source_version_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_twin_edge_source_id", "twin_edges", ["source_id"])
    op.create_index("ix_twin_edge_source_version_id", "twin_edges", ["source_version_id"])
    op.create_index(
        "ix_twin_edge_active_seen", "twin_edges", ["scenario_id", "is_active", "last_seen_at"]
    )

    _bootstrap_source_versions_and_events()


def _bootstrap_source_versions_and_events() -> None:
    bind = op.get_bind()
    now = datetime.now(UTC)
    rows = bind.execute(
        sa.text(
            """
            SELECT s.id, s.collection_id, s.cursor, s.last_run_at, ts.id AS scenario_id
            FROM sources s
            LEFT JOIN twin_scenarios ts
              ON ts.collection_id = s.collection_id
             AND ts.is_as_is = true
            """
        )
    ).mappings()

    for row in rows:
        source_id = row["id"]
        collection_id = row["collection_id"]
        scenario_id = row["scenario_id"]
        revision_key = row["cursor"] or f"bootstrap:{source_id}"
        status = "ready" if row["last_run_at"] else "queued"
        source_version_id = uuid.uuid4()

        bind.execute(
            sa.text(
                """
                INSERT INTO twin_source_versions (
                    id, collection_id, source_id, revision_key, extractor_version,
                    language_profile, status, stats,
                    joern_status, joern_project, joern_cpg_path, joern_server_url,
                    started_at, finished_at, created_at, updated_at
                )
                VALUES (
                    :id, :collection_id, :source_id, :revision_key, :extractor_version,
                    :language_profile, :status, :stats,
                    :joern_status, :joern_project, :joern_cpg_path, :joern_server_url,
                    :started_at, :finished_at, :created_at, :updated_at
                )
                ON CONFLICT ON CONSTRAINT uq_twin_source_version_revision DO NOTHING
                """
            ),
            {
                "id": source_version_id,
                "collection_id": collection_id,
                "source_id": source_id,
                "revision_key": revision_key,
                "extractor_version": "legacy-v1",
                "language_profile": None,
                "status": status,
                "stats": "{}",
                "joern_status": "pending",
                "joern_project": None,
                "joern_cpg_path": None,
                "joern_server_url": None,
                "started_at": row["last_run_at"],
                "finished_at": row["last_run_at"],
                "created_at": now,
                "updated_at": now,
            },
        )

        idempotency_key = f"bootstrap:{source_id}:{revision_key}"
        bind.execute(
            sa.text(
                """
                INSERT INTO twin_events (
                    id, collection_id, scenario_id, source_id, source_version_id,
                    event_type, status, payload, event_ts, idempotency_key, error
                )
                VALUES (
                    :id, :collection_id, :scenario_id, :source_id, :source_version_id,
                    :event_type, :status, :payload, :event_ts, :idempotency_key, :error
                )
                ON CONFLICT ON CONSTRAINT uq_twin_event_idempotency DO NOTHING
                """
            ),
            {
                "id": uuid.uuid4(),
                "collection_id": collection_id,
                "scenario_id": scenario_id,
                "source_id": source_id,
                "source_version_id": source_version_id,
                "event_type": "bootstrap",
                "status": status,
                "payload": "{}",
                "event_ts": now,
                "idempotency_key": idempotency_key,
                "error": None,
            },
        )


def downgrade() -> None:
    op.drop_index("ix_twin_edge_active_seen", table_name="twin_edges")
    op.drop_index("ix_twin_edge_source_version_id", table_name="twin_edges")
    op.drop_index("ix_twin_edge_source_id", table_name="twin_edges")
    op.drop_constraint("fk_twin_edges_source_version_id", "twin_edges", type_="foreignkey")
    op.drop_constraint("fk_twin_edges_source_id", "twin_edges", type_="foreignkey")
    op.drop_column("twin_edges", "last_seen_at")
    op.drop_column("twin_edges", "first_seen_at")
    op.drop_column("twin_edges", "is_active")
    op.drop_column("twin_edges", "source_version_id")
    op.drop_column("twin_edges", "source_id")

    op.drop_index("ix_twin_node_active_seen", table_name="twin_nodes")
    op.drop_index("ix_twin_node_source_version_id", table_name="twin_nodes")
    op.drop_index("ix_twin_node_source_id", table_name="twin_nodes")
    op.drop_constraint("fk_twin_nodes_source_version_id", "twin_nodes", type_="foreignkey")
    op.drop_constraint("fk_twin_nodes_source_id", "twin_nodes", type_="foreignkey")
    op.drop_column("twin_nodes", "last_seen_at")
    op.drop_column("twin_nodes", "first_seen_at")
    op.drop_column("twin_nodes", "is_active")
    op.drop_column("twin_nodes", "source_version_id")
    op.drop_column("twin_nodes", "source_id")

    op.drop_index("ix_twin_findings_type", table_name="twin_findings")
    op.drop_index("ix_twin_findings_status", table_name="twin_findings")
    op.drop_index("ix_twin_findings_scenario_created", table_name="twin_findings")
    op.drop_table("twin_findings")

    op.drop_index("ix_twin_analysis_cache_lookup", table_name="twin_analysis_cache")
    op.drop_table("twin_analysis_cache")

    op.drop_index("ix_twin_event_scenario_ts", table_name="twin_events")
    op.drop_index("ix_twin_event_source_ts", table_name="twin_events")
    op.drop_index("ix_twin_event_collection_ts", table_name="twin_events")
    op.drop_table("twin_events")

    op.drop_index("ix_twin_source_version_source_status", table_name="twin_source_versions")
    op.drop_index("ix_twin_source_version_collection", table_name="twin_source_versions")
    op.drop_table("twin_source_versions")
