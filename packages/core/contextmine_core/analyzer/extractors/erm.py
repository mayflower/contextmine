"""ERM (Entity-Relationship Model) extraction and Mermaid ERD generation.

This module:
1. Extracts database schema from Alembic migrations
2. Builds knowledge graph nodes for tables/columns
3. Generates Mermaid ER diagrams
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from contextmine_core.analyzer.extractors.alembic import (
    AlembicExtraction,
    ForeignKeyDef,
    TableDef,
    extract_from_alembic,
)
from contextmine_core.models import (
    KnowledgeArtifact,
    KnowledgeArtifactKind,
    KnowledgeEdge,
    KnowledgeEdgeKind,
    KnowledgeEvidence,
    KnowledgeNode,
    KnowledgeNodeEvidence,
    KnowledgeNodeKind,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass
class ERMSchema:
    """Consolidated ERM schema from multiple sources."""

    tables: dict[str, TableDef] = field(default_factory=dict)
    foreign_keys: list[ForeignKeyDef] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)  # File paths that contributed


class ERMExtractor:
    """Extracts ERM schema from migration files."""

    def __init__(self) -> None:
        self.schema = ERMSchema()

    def add_alembic_extraction(self, extraction: AlembicExtraction) -> None:
        """Add extracted Alembic data to the schema."""
        self.schema.sources.append(extraction.file_path)

        # Add tables
        for table in extraction.tables:
            if table.name in self.schema.tables:
                # Merge columns
                existing = self.schema.tables[table.name]
                existing_cols = {c.name for c in existing.columns}
                for col in table.columns:
                    if col.name not in existing_cols:
                        existing.columns.append(col)
            else:
                self.schema.tables[table.name] = table

        # Add columns from add_column
        for table_name, column in extraction.added_columns:
            if table_name in self.schema.tables:
                existing = self.schema.tables[table_name]
                if not any(c.name == column.name for c in existing.columns):
                    existing.columns.append(column)
            else:
                # Table doesn't exist yet, create it
                self.schema.tables[table_name] = TableDef(name=table_name, columns=[column])

        # Add foreign keys
        self.schema.foreign_keys.extend(extraction.foreign_keys)

        # Also extract FKs from column definitions
        for table in extraction.tables:
            for col in table.columns:
                if col.foreign_key:
                    # Parse "table.column" format
                    parts = col.foreign_key.split(".")
                    if len(parts) == 2:
                        self.schema.foreign_keys.append(
                            ForeignKeyDef(
                                name=None,
                                source_table=table.name,
                                source_columns=[col.name],
                                target_table=parts[0],
                                target_columns=[parts[1]],
                            )
                        )

    def extract_from_directory(self, alembic_dir: Path) -> None:
        """Extract schema from all migrations in a directory.

        Args:
            alembic_dir: Path to alembic/versions directory
        """
        if not alembic_dir.exists():
            logger.warning("Alembic directory not found: %s", alembic_dir)
            return

        for migration_file in sorted(alembic_dir.glob("*.py")):
            if migration_file.name.startswith("__"):
                continue

            content = migration_file.read_text(encoding="utf-8", errors="replace")
            extraction = extract_from_alembic(str(migration_file), content)
            self.add_alembic_extraction(extraction)


def generate_mermaid_erd(schema: ERMSchema) -> str:
    """Generate a Mermaid ER diagram from the schema.

    Args:
        schema: The ERM schema

    Returns:
        Mermaid ERD diagram as a string
    """
    lines = ["erDiagram"]

    # Add tables with columns
    for table_name, table in sorted(schema.tables.items()):
        lines.append(f"    {_mermaid_safe(table_name)} {{")
        for col in table.columns:
            pk = "PK" if col.primary_key else ""
            fk = "FK" if col.foreign_key else ""
            markers = ",".join(filter(None, [pk, fk]))
            type_str = _mermaid_type(col.type_name)
            if markers:
                lines.append(f"        {type_str} {_mermaid_safe(col.name)} {markers}")
            else:
                lines.append(f"        {type_str} {_mermaid_safe(col.name)}")
        lines.append("    }")

    # Add relationships from foreign keys
    seen_relationships: set[tuple[str, str]] = set()
    for fk in schema.foreign_keys:
        source = _mermaid_safe(fk.source_table)
        target = _mermaid_safe(fk.target_table)

        # Skip if either table doesn't exist in schema
        if fk.source_table not in schema.tables or fk.target_table not in schema.tables:
            continue

        rel_key = (source, target)
        if rel_key in seen_relationships:
            continue
        seen_relationships.add(rel_key)

        # Use ||--o{ for one-to-many relationship
        lines.append(f"    {target} ||--o{{ {source} : has")

    return "\n".join(lines)


def _mermaid_safe(name: str) -> str:
    """Make a name safe for Mermaid diagrams."""
    # Replace problematic characters
    return name.replace("-", "_").replace(" ", "_")


def _mermaid_type(type_name: str) -> str:
    """Convert SQL type to Mermaid-friendly type."""
    type_map = {
        "String": "string",
        "Text": "text",
        "Integer": "int",
        "BigInteger": "bigint",
        "SmallInteger": "smallint",
        "Float": "float",
        "Boolean": "bool",
        "DateTime": "datetime",
        "Date": "date",
        "Time": "time",
        "UUID": "uuid",
        "JSON": "json",
        "ARRAY": "array",
        "Enum": "enum",
        "LargeBinary": "blob",
        "Numeric": "decimal",
        "unknown": "unknown",
    }
    return type_map.get(type_name, type_name.lower())


async def build_erm_graph(
    session: AsyncSession,
    collection_id: UUID,
    schema: ERMSchema,
) -> dict:
    """Build knowledge graph nodes and edges from ERM schema.

    Creates:
    - DB_TABLE nodes for each table
    - DB_COLUMN nodes for each column
    - TABLE_HAS_COLUMN edges
    - COLUMN_FK_TO_COLUMN edges

    Args:
        session: Database session
        collection_id: Collection UUID
        schema: The ERM schema

    Returns:
        Stats dict
    """
    from sqlalchemy import select
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    stats = {
        "table_nodes_created": 0,
        "column_nodes_created": 0,
        "edges_created": 0,
        "evidence_created": 0,
    }

    table_node_ids: dict[str, UUID] = {}
    column_node_ids: dict[str, UUID] = {}  # "table.column" -> node_id

    # Create table nodes
    for table_name, table in schema.tables.items():
        natural_key = f"db:{table_name}"

        stmt = pg_insert(KnowledgeNode).values(
            collection_id=collection_id,
            kind=KnowledgeNodeKind.DB_TABLE,
            natural_key=natural_key,
            name=table_name,
            meta={
                "column_count": len(table.columns),
                "primary_keys": table.primary_keys,
            },
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_knowledge_node_natural",
            set_={
                "name": stmt.excluded.name,
                "meta": stmt.excluded.meta,
            },
        ).returning(KnowledgeNode.id)

        result = await session.execute(stmt)
        node_id = result.scalar_one()
        table_node_ids[table_name] = node_id
        stats["table_nodes_created"] += 1

        # Create column nodes
        for col in table.columns:
            col_natural_key = f"db:{table_name}.{col.name}"

            col_stmt = pg_insert(KnowledgeNode).values(
                collection_id=collection_id,
                kind=KnowledgeNodeKind.DB_COLUMN,
                natural_key=col_natural_key,
                name=col.name,
                meta={
                    "table": table_name,
                    "type": col.type_name,
                    "nullable": col.nullable,
                    "primary_key": col.primary_key,
                    "foreign_key": col.foreign_key,
                },
            )
            col_stmt = col_stmt.on_conflict_do_update(
                constraint="uq_knowledge_node_natural",
                set_={
                    "name": col_stmt.excluded.name,
                    "meta": col_stmt.excluded.meta,
                },
            ).returning(KnowledgeNode.id)

            col_result = await session.execute(col_stmt)
            col_node_id = col_result.scalar_one()
            column_node_ids[f"{table_name}.{col.name}"] = col_node_id
            stats["column_nodes_created"] += 1

            # Create TABLE_HAS_COLUMN edge
            edge_exists = await session.execute(
                select(KnowledgeEdge.id).where(
                    KnowledgeEdge.collection_id == collection_id,
                    KnowledgeEdge.source_node_id == node_id,
                    KnowledgeEdge.target_node_id == col_node_id,
                    KnowledgeEdge.kind == KnowledgeEdgeKind.TABLE_HAS_COLUMN,
                )
            )
            if not edge_exists.scalar_one_or_none():
                session.add(
                    KnowledgeEdge(
                        collection_id=collection_id,
                        source_node_id=node_id,
                        target_node_id=col_node_id,
                        kind=KnowledgeEdgeKind.TABLE_HAS_COLUMN,
                        meta={},
                    )
                )
                stats["edges_created"] += 1

    # Create FK edges
    for fk in schema.foreign_keys:
        for src_col, tgt_col in zip(fk.source_columns, fk.target_columns, strict=False):
            src_key = f"{fk.source_table}.{src_col}"
            tgt_key = f"{fk.target_table}.{tgt_col}"

            src_node_id = column_node_ids.get(src_key)
            tgt_node_id = column_node_ids.get(tgt_key)

            if src_node_id and tgt_node_id:
                edge_exists = await session.execute(
                    select(KnowledgeEdge.id).where(
                        KnowledgeEdge.collection_id == collection_id,
                        KnowledgeEdge.source_node_id == src_node_id,
                        KnowledgeEdge.target_node_id == tgt_node_id,
                        KnowledgeEdge.kind == KnowledgeEdgeKind.COLUMN_FK_TO_COLUMN,
                    )
                )
                if not edge_exists.scalar_one_or_none():
                    session.add(
                        KnowledgeEdge(
                            collection_id=collection_id,
                            source_node_id=src_node_id,
                            target_node_id=tgt_node_id,
                            kind=KnowledgeEdgeKind.COLUMN_FK_TO_COLUMN,
                            meta={"fk_name": fk.name},
                        )
                    )
                    stats["edges_created"] += 1

    # Create evidence for source files
    for source_file in schema.sources:
        for table_name in schema.tables:
            table_node_id = table_node_ids.get(table_name)
            if not table_node_id:
                continue

            # Check if evidence already exists
            existing = await session.execute(
                select(KnowledgeNodeEvidence.evidence_id).where(
                    KnowledgeNodeEvidence.node_id == table_node_id
                )
            )
            if existing.scalar_one_or_none():
                continue

            evidence = KnowledgeEvidence(
                file_path=source_file,
                start_line=1,
                end_line=1,
            )
            session.add(evidence)
            await session.flush()

            link = KnowledgeNodeEvidence(
                node_id=table_node_id,
                evidence_id=evidence.id,
            )
            session.add(link)
            stats["evidence_created"] += 1

    return stats


async def save_erd_artifact(
    session: AsyncSession,
    collection_id: UUID,
    schema: ERMSchema,
    name: str = "Database ERD",
) -> UUID:
    """Generate and save Mermaid ERD as an artifact.

    Args:
        session: Database session
        collection_id: Collection UUID
        schema: The ERM schema
        name: Artifact name

    Returns:
        Artifact ID
    """
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    mermaid_content = generate_mermaid_erd(schema)

    stmt = pg_insert(KnowledgeArtifact).values(
        collection_id=collection_id,
        kind=KnowledgeArtifactKind.MERMAID_ERD,
        name=name,
        content=mermaid_content,
        meta={
            "table_count": len(schema.tables),
            "fk_count": len(schema.foreign_keys),
            "sources": schema.sources,
        },
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_knowledge_artifact_name",
        set_={
            "content": stmt.excluded.content,
            "meta": stmt.excluded.meta,
        },
    ).returning(KnowledgeArtifact.id)

    result = await session.execute(stmt)
    return result.scalar_one()
