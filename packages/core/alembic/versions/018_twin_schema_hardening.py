"""Harden twin/graph schema with dedup constraints and cache engine dimension.

Revision ID: 018
Revises: 017
Create Date: 2026-02-22
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "018"
down_revision: str | None = "017"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _dedupe_rows() -> None:
    bind = op.get_bind()
    bind.execute(
        sa.text(
            """
            WITH ranked AS (
                SELECT
                    ctid,
                    row_number() OVER (
                        PARTITION BY collection_id, source_node_id, target_node_id, kind
                        ORDER BY created_at ASC, id ASC
                    ) AS rn
                FROM knowledge_edges
            )
            DELETE FROM knowledge_edges e
            USING ranked r
            WHERE e.ctid = r.ctid
              AND r.rn > 1
            """
        )
    )
    bind.execute(
        sa.text(
            """
            WITH ranked AS (
                SELECT
                    ctid,
                    row_number() OVER (
                        PARTITION BY document_id, qualified_name, start_line, end_line, kind
                        ORDER BY created_at ASC, id ASC
                    ) AS rn
                FROM symbols
            )
            DELETE FROM symbols s
            USING ranked r
            WHERE s.ctid = r.ctid
              AND r.rn > 1
            """
        )
    )
    bind.execute(
        sa.text(
            """
            WITH ranked AS (
                SELECT
                    ctid,
                    row_number() OVER (
                        PARTITION BY source_symbol_id, target_symbol_id, edge_type, COALESCE(source_line, -1)
                        ORDER BY created_at ASC, id ASC
                    ) AS rn
                FROM symbol_edges
            )
            DELETE FROM symbol_edges s
            USING ranked r
            WHERE s.ctid = r.ctid
              AND r.rn > 1
            """
        )
    )


def _normalize_status_values() -> None:
    bind = op.get_bind()
    bind.execute(
        sa.text(
            """
            UPDATE twin_source_versions
            SET status = 'failed'
            WHERE status IS NULL
               OR status NOT IN ('queued','materializing','ready','failed','stale','loading','generating')
            """
        )
    )
    bind.execute(
        sa.text(
            """
            UPDATE twin_source_versions
            SET joern_status = 'pending'
            WHERE joern_status IS NULL
               OR joern_status NOT IN ('pending','generating','loading','ready','failed')
            """
        )
    )
    bind.execute(
        sa.text(
            """
            UPDATE twin_events
            SET status = 'failed'
            WHERE status IS NULL
               OR status NOT IN ('queued','materializing','ready','failed','stale','loading','generating','degraded')
            """
        )
    )
    bind.execute(
        sa.text(
            """
            UPDATE twin_findings
            SET status = 'open'
            WHERE status IS NULL
               OR status NOT IN ('open','triaged','resolved','false_positive','suppressed')
            """
        )
    )


def _backfill_finding_fingerprints() -> None:
    bind = op.get_bind()
    bind.execute(
        sa.text(
            """
            UPDATE twin_findings
            SET fingerprint = md5(
                concat_ws(
                    '|',
                    COALESCE(source_version_id::text, ''),
                    lower(COALESCE(finding_type, '')),
                    lower(COALESCE(severity, '')),
                    lower(COALESCE(confidence, '')),
                    lower(COALESCE(status, '')),
                    COALESCE(filename, ''),
                    COALESCE(line_number::text, ''),
                    COALESCE(message, ''),
                    COALESCE(flow_data::text, ''),
                    COALESCE(meta::text, '')
                )
            )
            WHERE fingerprint IS NULL OR fingerprint = ''
            """
        )
    )
    bind.execute(
        sa.text(
            """
            WITH ranked AS (
                SELECT
                    ctid,
                    row_number() OVER (
                        PARTITION BY scenario_id, fingerprint
                        ORDER BY updated_at DESC, created_at DESC, id DESC
                    ) AS rn
                FROM twin_findings
            )
            DELETE FROM twin_findings f
            USING ranked r
            WHERE f.ctid = r.ctid
              AND r.rn > 1
            """
        )
    )


def upgrade() -> None:
    _dedupe_rows()

    op.create_unique_constraint(
        "uq_knowledge_edge_unique",
        "knowledge_edges",
        ["collection_id", "source_node_id", "target_node_id", "kind"],
    )
    op.create_unique_constraint(
        "uq_symbol_identity",
        "symbols",
        ["document_id", "qualified_name", "start_line", "end_line", "kind"],
    )
    op.create_index(
        "uq_symbol_edge_identity",
        "symbol_edges",
        [
            "source_symbol_id",
            "target_symbol_id",
            "edge_type",
            sa.text("coalesce(source_line, -1)"),
        ],
        unique=True,
    )

    op.add_column(
        "twin_edges",
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    op.add_column(
        "twin_analysis_cache",
        sa.Column(
            "engine",
            sa.String(length=32),
            nullable=False,
            server_default="graphrag",
        ),
    )
    op.drop_index("ix_twin_analysis_cache_lookup", table_name="twin_analysis_cache")
    op.drop_constraint("uq_twin_analysis_cache_key", "twin_analysis_cache", type_="unique")
    op.create_unique_constraint(
        "uq_twin_analysis_cache_key",
        "twin_analysis_cache",
        ["scenario_id", "engine", "tool_name", "params_hash", "cache_key"],
    )
    op.create_index(
        "ix_twin_analysis_cache_lookup",
        "twin_analysis_cache",
        ["scenario_id", "engine", "tool_name", "expires_at"],
    )

    op.add_column(
        "twin_findings",
        sa.Column("fingerprint", sa.String(length=64), nullable=True),
    )
    _backfill_finding_fingerprints()
    op.alter_column("twin_findings", "fingerprint", nullable=False)
    op.create_unique_constraint(
        "uq_twin_finding_fingerprint",
        "twin_findings",
        ["scenario_id", "fingerprint"],
    )

    _normalize_status_values()
    op.create_check_constraint(
        "ck_twin_source_version_status",
        "twin_source_versions",
        "status IN ('queued','materializing','ready','failed','stale','loading','generating')",
    )
    op.create_check_constraint(
        "ck_twin_source_version_joern_status",
        "twin_source_versions",
        "joern_status IN ('pending','generating','loading','ready','failed')",
    )
    op.create_check_constraint(
        "ck_twin_event_status",
        "twin_events",
        "status IN ('queued','materializing','ready','failed','stale','loading','generating','degraded')",
    )
    op.create_check_constraint(
        "ck_twin_findings_status",
        "twin_findings",
        "status IN ('open','triaged','resolved','false_positive','suppressed')",
    )


def downgrade() -> None:
    op.drop_constraint("ck_twin_findings_status", "twin_findings", type_="check")
    op.drop_constraint("ck_twin_event_status", "twin_events", type_="check")
    op.drop_constraint("ck_twin_source_version_joern_status", "twin_source_versions", type_="check")
    op.drop_constraint("ck_twin_source_version_status", "twin_source_versions", type_="check")

    op.drop_constraint("uq_twin_finding_fingerprint", "twin_findings", type_="unique")
    op.drop_column("twin_findings", "fingerprint")

    op.drop_index("ix_twin_analysis_cache_lookup", table_name="twin_analysis_cache")
    op.drop_constraint("uq_twin_analysis_cache_key", "twin_analysis_cache", type_="unique")
    op.create_unique_constraint(
        "uq_twin_analysis_cache_key",
        "twin_analysis_cache",
        ["scenario_id", "tool_name", "params_hash", "cache_key"],
    )
    op.create_index(
        "ix_twin_analysis_cache_lookup",
        "twin_analysis_cache",
        ["scenario_id", "tool_name", "expires_at"],
    )
    op.drop_column("twin_analysis_cache", "engine")

    op.drop_column("twin_edges", "updated_at")

    op.drop_index("uq_symbol_edge_identity", table_name="symbol_edges")
    op.drop_constraint("uq_symbol_identity", "symbols", type_="unique")
    op.drop_constraint("uq_knowledge_edge_unique", "knowledge_edges", type_="unique")
