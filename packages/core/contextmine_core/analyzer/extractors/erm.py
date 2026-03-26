"""ERM (Entity-Relationship Model) extraction from Alembic migrations.

This module provides the ERMExtractor class for building schema from
Alembic migration files. Graph-building, Mermaid ERD generation, and
artifact persistence are delegated to the generic ``schema`` module.

Legacy callers that imported ``generate_mermaid_erd``, ``build_erm_graph``,
or ``save_erd_artifact`` from this module will still work -- they are
re-exported here for backwards compatibility.
"""

from __future__ import annotations

import logging
from pathlib import Path

from contextmine_core.analyzer.extractors.alembic import (
    AlembicExtraction,
    ForeignKeyDef,
    TableDef,
    extract_from_alembic,
)

# Re-export from schema.py -- these were formerly duplicated here.
from contextmine_core.analyzer.extractors.schema import (  # noqa: I001
    AggregatedSchema,
    generate_mermaid_erd,
    save_erd_artifact,
)
from contextmine_core.analyzer.extractors.schema import (
    build_schema_graph as build_erm_graph,
)

logger = logging.getLogger(__name__)

# Backwards-compatible alias
ERMSchema = AggregatedSchema


class ERMExtractor:
    """Extracts ERM schema from Alembic migration files."""

    def __init__(self) -> None:
        self.schema = ERMSchema()

    def _merge_tables(self, extraction: AlembicExtraction) -> None:
        """Merge table definitions from an extraction into the schema."""
        for table in extraction.tables:
            if table.name in self.schema.tables:
                existing = self.schema.tables[table.name]
                existing_cols = {c.name for c in existing.columns}
                for col in table.columns:
                    if col.name not in existing_cols:
                        existing.columns.append(col)
            else:
                self.schema.tables[table.name] = table

    def _merge_added_columns(self, extraction: AlembicExtraction) -> None:
        """Merge add_column operations into the schema."""
        for table_name, column in extraction.added_columns:
            if table_name in self.schema.tables:
                existing = self.schema.tables[table_name]
                if not any(c.name == column.name for c in existing.columns):
                    existing.columns.append(column)
            else:
                self.schema.tables[table_name] = TableDef(name=table_name, columns=[column])

    def _extract_column_foreign_keys(self, extraction: AlembicExtraction) -> None:
        """Extract foreign keys embedded in column definitions."""
        for table in extraction.tables:
            for col in table.columns:
                if not col.foreign_key:
                    continue
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

    def add_alembic_extraction(self, extraction: AlembicExtraction) -> None:
        """Add extracted Alembic data to the schema."""
        self.schema.sources.append(extraction.file_path)
        self._merge_tables(extraction)
        self._merge_added_columns(extraction)
        self.schema.foreign_keys.extend(extraction.foreign_keys)
        self._extract_column_foreign_keys(extraction)

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


__all__ = [
    "ERMExtractor",
    "ERMSchema",
    "build_erm_graph",
    "generate_mermaid_erd",
    "save_erd_artifact",
]
