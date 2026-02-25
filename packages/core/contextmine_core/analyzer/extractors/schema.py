"""Database schema extractor using LLM.

This module extracts database schema definitions from ANY format:
- Migrations: Alembic, Django, Rails, Knex, Flyway, Liquibase, Prisma Migrate
- ORMs: SQLAlchemy models, Django models, TypeORM entities, Sequelize, ActiveRecord
- Schema files: Prisma schema, GraphQL SDL, SQL DDL
- Any other database schema definition format

Uses LLM for semantic analysis - no framework-specific hardcoding.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from contextmine_core.research.llm.provider import LLMProvider
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# ============================================================================
# Data models for extraction output
# ============================================================================


@dataclass
class ColumnDef:
    """Extracted column definition."""

    name: str
    type_name: str
    nullable: bool = True
    primary_key: bool = False
    foreign_key: str | None = None  # "table.column" format
    description: str | None = None


@dataclass
class TableDef:
    """Extracted table definition."""

    name: str
    columns: list[ColumnDef] = field(default_factory=list)
    primary_keys: list[str] = field(default_factory=list)
    description: str | None = None


@dataclass
class ForeignKeyDef:
    """Extracted foreign key definition."""

    name: str | None
    source_table: str
    source_columns: list[str]
    target_table: str
    target_columns: list[str]


@dataclass
class SchemaExtraction:
    """Result of extracting schema from a file."""

    file_path: str
    framework: str | None = None  # Detected framework (alembic, django, prisma, etc.)
    tables: list[TableDef] = field(default_factory=list)
    foreign_keys: list[ForeignKeyDef] = field(default_factory=list)


# ============================================================================
# Pydantic models for LLM structured output
# ============================================================================


class ColumnOutput(BaseModel):
    """A database column definition."""

    name: str = Field(description="Column name")
    type_name: str = Field(
        description="Data type (e.g., 'String', 'Integer', 'UUID', 'DateTime', 'Boolean', 'Text', 'JSON')"
    )
    nullable: bool = Field(default=True, description="Whether the column allows NULL values")
    primary_key: bool = Field(default=False, description="Whether this is a primary key column")
    foreign_key: str | None = Field(
        default=None, description="Foreign key reference in 'table.column' format, if any"
    )
    description: str | None = Field(default=None, description="Column description or purpose")


class TableOutput(BaseModel):
    """A database table definition."""

    name: str = Field(description="Table name")
    columns: list[ColumnOutput] = Field(default_factory=list, description="Columns in the table")
    description: str | None = Field(default=None, description="Table description or purpose")


class ForeignKeyOutput(BaseModel):
    """An explicit foreign key constraint."""

    name: str | None = Field(default=None, description="Constraint name if specified")
    source_table: str = Field(description="Table containing the foreign key")
    source_columns: list[str] = Field(description="Column(s) in the source table")
    target_table: str = Field(description="Referenced table")
    target_columns: list[str] = Field(description="Referenced column(s)")


class SchemaExtractionOutput(BaseModel):
    """LLM output for schema extraction."""

    framework: str | None = Field(
        default=None,
        description="Detected framework (alembic, django, rails, prisma, typeorm, sql, etc.)",
    )
    tables: list[TableOutput] = Field(
        default_factory=list, description="Extracted table definitions"
    )
    foreign_keys: list[ForeignKeyOutput] = Field(
        default_factory=list, description="Explicit foreign key constraints"
    )


# ============================================================================
# LLM-based extraction
# ============================================================================


SCHEMA_EXTRACTION_PROMPT = """Analyze this file and extract any database schema definitions (tables, columns, relationships).

Look for schema definitions from ANY format including but not limited to:
- Migration files: Alembic (Python), Django migrations, Rails migrations, Knex (JS), Flyway (SQL), Liquibase (XML/YAML)
- ORM models: SQLAlchemy, Django models, TypeORM entities, Sequelize models, ActiveRecord, Hibernate
- Schema files: Prisma schema (.prisma), SQL DDL (CREATE TABLE), GraphQL SDL with database types
- Any other database schema definition format

For each table found, extract:
1. Table name
2. All columns with:
   - Name
   - Data type (normalize to common types: String, Integer, BigInteger, Float, Boolean, DateTime, Date, UUID, JSON, Text, etc.)
   - Whether nullable
   - Whether primary key
   - Foreign key reference if any (format: "table.column")
3. Description if available

Also extract any explicit foreign key constraints defined separately from columns.

File: {file_path}

Content:
```
{content}
```

Return empty lists if no database schema definitions are found."""

SCHEMA_EXTRACTION_SYSTEM_PROMPT = "You are a database schema extraction specialist. Return only structured output based on the file content."

SQL_CREATE_TABLE_RE = re.compile(
    r"create\s+table(?:\s+if\s+not\s+exists)?\s+([^\s(]+)\s*\((.*?)\)\s*;",
    re.IGNORECASE | re.DOTALL,
)
SQL_PRIMARY_KEY_RE = re.compile(r"primary\s+key\s*\(([^)]+)\)", re.IGNORECASE)
SQL_FOREIGN_KEY_RE = re.compile(
    r"foreign\s+key\s*\(([^)]+)\)\s*references\s+([^\s(]+)\s*\(([^)]+)\)",
    re.IGNORECASE,
)
SQL_INLINE_REFERENCE_RE = re.compile(
    r"references\s+([^\s(]+)\s*\(([^)]+)\)",
    re.IGNORECASE,
)


def _strip_sql_identifier(raw: str) -> str:
    token = raw.strip().strip(",")
    if not token:
        return token
    parts = [part.strip().strip('`"[]') for part in token.split(".")]
    return ".".join(part for part in parts if part)


def _split_sql_items(body: str) -> list[str]:
    items: list[str] = []
    current: list[str] = []
    depth = 0
    in_single = False
    in_double = False
    in_backtick = False

    for char in body:
        if char == "'" and not in_double and not in_backtick:
            in_single = not in_single
        elif char == '"' and not in_single and not in_backtick:
            in_double = not in_double
        elif char == "`" and not in_single and not in_double:
            in_backtick = not in_backtick
        elif not in_single and not in_double and not in_backtick:
            if char == "(":
                depth += 1
            elif char == ")" and depth > 0:
                depth -= 1

        if char == "," and depth == 0 and not in_single and not in_double and not in_backtick:
            item = "".join(current).strip()
            if item:
                items.append(item)
            current = []
            continue

        current.append(char)

    tail = "".join(current).strip()
    if tail:
        items.append(tail)
    return items


def _normalize_sql_type(raw_type: str) -> str:
    token = raw_type.strip().lower()
    if not token:
        return "unknown"
    if token.startswith(("varchar", "char", "character varying")):
        return "String"
    if token.startswith("text"):
        return "Text"
    if token.startswith(("bigint", "bigserial")):
        return "BigInteger"
    if token.startswith(("smallint", "smallserial")):
        return "SmallInteger"
    if token.startswith(("int", "integer", "serial")):
        return "Integer"
    if token.startswith(("bool", "boolean")):
        return "Boolean"
    if token.startswith(("timestamp", "datetime")):
        return "DateTime"
    if token.startswith("date"):
        return "Date"
    if token.startswith("time"):
        return "Time"
    if token.startswith("uuid"):
        return "UUID"
    if token.startswith("json"):
        return "JSON"
    if token.startswith(("numeric", "decimal", "money")):
        return "Numeric"
    if token.startswith(("float", "double", "real")):
        return "Float"
    if token.startswith(("blob", "bytea", "binary", "varbinary")):
        return "LargeBinary"
    return raw_type.strip()[:64]


def _extract_schema_from_sql_ddl(file_path: str, content: str) -> SchemaExtraction:
    extraction = SchemaExtraction(file_path=file_path, framework="sql")
    matches = list(SQL_CREATE_TABLE_RE.finditer(content))
    if not matches:
        return extraction

    for match in matches:
        table_name = _strip_sql_identifier(match.group(1))
        if not table_name:
            continue
        table = TableDef(name=table_name)
        body = match.group(2)
        items = _split_sql_items(body)

        for item in items:
            line = item.strip()
            if not line:
                continue
            line_upper = line.upper()

            pk_match = SQL_PRIMARY_KEY_RE.search(line)
            if pk_match:
                pk_columns = [
                    _strip_sql_identifier(col)
                    for col in pk_match.group(1).split(",")
                    if col.strip()
                ]
                for pk_col in pk_columns:
                    if pk_col and pk_col not in table.primary_keys:
                        table.primary_keys.append(pk_col)

            fk_match = SQL_FOREIGN_KEY_RE.search(line)
            if fk_match:
                source_columns = [
                    _strip_sql_identifier(col)
                    for col in fk_match.group(1).split(",")
                    if col.strip()
                ]
                target_table = _strip_sql_identifier(fk_match.group(2))
                target_columns = [
                    _strip_sql_identifier(col)
                    for col in fk_match.group(3).split(",")
                    if col.strip()
                ]
                if source_columns and target_table and target_columns:
                    extraction.foreign_keys.append(
                        ForeignKeyDef(
                            name=None,
                            source_table=table_name,
                            source_columns=source_columns,
                            target_table=target_table,
                            target_columns=target_columns,
                        )
                    )

            if line_upper.startswith(
                (
                    "CONSTRAINT ",
                    "PRIMARY KEY",
                    "FOREIGN KEY",
                    "UNIQUE ",
                    "KEY ",
                    "INDEX ",
                    "CHECK ",
                )
            ):
                continue

            parts = line.split(None, 2)
            if len(parts) < 2:
                continue
            column_name = _strip_sql_identifier(parts[0])
            raw_type = parts[1]
            inline_fk = SQL_INLINE_REFERENCE_RE.search(line)
            foreign_key = None
            if inline_fk:
                target_table = _strip_sql_identifier(inline_fk.group(1))
                target_column = _strip_sql_identifier(inline_fk.group(2))
                if target_table and target_column:
                    foreign_key = f"{target_table}.{target_column}"

            column = ColumnDef(
                name=column_name,
                type_name=_normalize_sql_type(raw_type),
                nullable="NOT NULL" not in line_upper,
                primary_key="PRIMARY KEY" in line_upper,
                foreign_key=foreign_key,
            )
            table.columns.append(column)
            if column.primary_key and column.name not in table.primary_keys:
                table.primary_keys.append(column.name)

        if table.primary_keys:
            for col in table.columns:
                if col.name in table.primary_keys:
                    col.primary_key = True

        if table.columns:
            extraction.tables.append(table)

    return extraction


def _is_code_or_schema_file(file_path: str) -> bool:
    """Check if a file could contain schema definitions.

    This is a basic filter to skip obvious non-code files (binaries, images, etc.)
    The actual relevance check is done by LLM triage.
    """
    path_lower = file_path.lower()

    # Skip binary/media files
    skip_extensions = (
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".ico",
        ".svg",
        ".woff",
        ".woff2",
        ".ttf",
        ".eot",
        ".zip",
        ".tar",
        ".gz",
        ".rar",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".pyc",
        ".pyo",
        ".class",
        ".lock",
        ".sum",
        ".md",
        ".txt",
        ".rst",  # Documentation
    )
    if path_lower.endswith(skip_extensions):
        return False

    # Accept code and config files that might define schemas
    code_extensions = (
        ".py",
        ".rb",
        ".js",
        ".ts",
        ".java",
        ".go",
        ".cs",
        ".php",
        ".sql",
        ".ddl",
        ".prisma",
        ".yml",
        ".yaml",
        ".json",
        ".xml",
    )
    return path_lower.endswith(code_extensions)


async def extract_schema_from_files(
    files: list[tuple[str, str]],
    provider: LLMProvider,
) -> list[SchemaExtraction]:
    """Extract schema definitions from multiple files using LLM triage.

    This function:
    1. Uses LLM to identify which files likely contain schemas (triage)
    2. Extracts schemas only from identified files

    Args:
        files: List of (file_path, content) tuples
        provider: LLM provider for triage and extraction

    Returns:
        List of SchemaExtraction results
    """
    from contextmine_core.analyzer.extractors.triage import triage_files_for_schema

    # Filter to code files only
    code_files = [(p, c) for p, c in files if _is_code_or_schema_file(p)]

    if not code_files:
        return []

    # Use LLM triage to identify relevant files
    relevant_paths = await triage_files_for_schema(code_files, provider)
    relevant_files = {p: c for p, c in code_files if p in relevant_paths}

    logger.info(
        "Schema triage: %d/%d files selected for extraction",
        len(relevant_files),
        len(code_files),
    )

    # Extract from relevant files
    results = []
    for file_path, content in relevant_files.items():
        result = await _extract_schema_from_single_file(file_path, content, provider)
        if result.tables:
            results.append(result)

    return results


async def extract_schema_from_file(
    file_path: str,
    content: str,
    provider: LLMProvider,
) -> SchemaExtraction:
    """Extract database schema from a single file using LLM.

    Use this for single-file extraction when you know the file contains schemas.
    For batch extraction with automatic file selection, use extract_schema_from_files().

    Args:
        file_path: Path to the file
        content: File content
        provider: LLM provider for analysis

    Returns:
        SchemaExtraction with extracted schema definitions
    """
    return await _extract_schema_from_single_file(file_path, content, provider)


async def _extract_schema_from_single_file(
    file_path: str,
    content: str,
    provider: LLMProvider,
) -> SchemaExtraction:
    """Internal: Extract schema from a single file.

    No filtering - assumes caller has already determined this file should be analyzed.
    """
    result = SchemaExtraction(file_path=file_path)

    # Skip very large files (likely not schema)
    if len(content) > 100000:
        logger.debug("Skipping large file for schema extraction: %s", file_path)
        return result

    # Skip binary or non-text content
    if "\x00" in content[:1000]:
        return result

    path_lower = file_path.lower()
    if path_lower.endswith((".sql", ".ddl")):
        deterministic = _extract_schema_from_sql_ddl(file_path, content)
        if deterministic.tables or deterministic.foreign_keys:
            return deterministic

    try:
        prompt = SCHEMA_EXTRACTION_PROMPT.format(
            file_path=file_path,
            content=content[:50000],  # Truncate very long files
        )

        llm_result = await provider.generate_structured(
            system=SCHEMA_EXTRACTION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            output_schema=SchemaExtractionOutput,
            temperature=0.0,
            max_tokens=3000,
        )

        result.framework = llm_result.framework

        # Convert tables
        for table_output in llm_result.tables:
            table = TableDef(
                name=table_output.name,
                description=table_output.description,
            )

            for col_output in table_output.columns:
                col = ColumnDef(
                    name=col_output.name,
                    type_name=col_output.type_name,
                    nullable=col_output.nullable,
                    primary_key=col_output.primary_key,
                    foreign_key=col_output.foreign_key,
                    description=col_output.description,
                )
                table.columns.append(col)
                if col.primary_key:
                    table.primary_keys.append(col.name)

            result.tables.append(table)

        # Convert foreign keys
        for fk_output in llm_result.foreign_keys:
            result.foreign_keys.append(
                ForeignKeyDef(
                    name=fk_output.name,
                    source_table=fk_output.source_table,
                    source_columns=fk_output.source_columns,
                    target_table=fk_output.target_table,
                    target_columns=fk_output.target_columns,
                )
            )

    except Exception as e:
        logger.warning("Failed to extract schema from %s: %s", file_path, e)

    return result


# ============================================================================
# Schema aggregation
# ============================================================================


@dataclass
class AggregatedSchema:
    """Consolidated schema from multiple source files."""

    tables: dict[str, TableDef] = field(default_factory=dict)
    foreign_keys: list[ForeignKeyDef] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


def aggregate_schema_extractions(extractions: list[SchemaExtraction]) -> AggregatedSchema:
    """Aggregate multiple schema extractions into a unified schema.

    Handles:
    - Merging tables with the same name from different files
    - Combining columns from multiple sources
    - Deduplicating foreign keys

    Args:
        extractions: List of schema extractions from individual files

    Returns:
        AggregatedSchema with consolidated schema
    """
    schema = AggregatedSchema()

    for extraction in extractions:
        if extraction.tables or extraction.foreign_keys:
            schema.sources.append(extraction.file_path)

        # Merge tables
        for table in extraction.tables:
            if table.name in schema.tables:
                # Merge columns into existing table
                existing = schema.tables[table.name]
                existing_col_names = {c.name for c in existing.columns}
                for col in table.columns:
                    if col.name not in existing_col_names:
                        existing.columns.append(col)
                        if col.primary_key and col.name not in existing.primary_keys:
                            existing.primary_keys.append(col.name)
            else:
                schema.tables[table.name] = table

        # Add foreign keys (dedupe by signature)
        existing_fks = {
            (fk.source_table, tuple(fk.source_columns), fk.target_table, tuple(fk.target_columns))
            for fk in schema.foreign_keys
        }

        for fk in extraction.foreign_keys:
            fk_sig = (
                fk.source_table,
                tuple(fk.source_columns),
                fk.target_table,
                tuple(fk.target_columns),
            )
            if fk_sig not in existing_fks:
                schema.foreign_keys.append(fk)
                existing_fks.add(fk_sig)

        # Also extract FKs from column definitions
        for table in extraction.tables:
            for col in table.columns:
                if col.foreign_key:
                    parts = col.foreign_key.split(".")
                    if len(parts) == 2:
                        fk = ForeignKeyDef(
                            name=None,
                            source_table=table.name,
                            source_columns=[col.name],
                            target_table=parts[0],
                            target_columns=[parts[1]],
                        )
                        fk_sig = (
                            fk.source_table,
                            tuple(fk.source_columns),
                            fk.target_table,
                            tuple(fk.target_columns),
                        )
                        if fk_sig not in existing_fks:
                            schema.foreign_keys.append(fk)
                            existing_fks.add(fk_sig)

    return schema


# ============================================================================
# Mermaid ERD generation
# ============================================================================


def generate_mermaid_erd(schema: AggregatedSchema) -> str:
    """Generate a Mermaid ER diagram from the schema.

    Args:
        schema: The aggregated schema

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


# ============================================================================
# Knowledge graph building
# ============================================================================


def get_table_natural_key(table_name: str) -> str:
    """Generate a natural key for a table."""
    return f"db:{table_name}"


def get_column_natural_key(table_name: str, column_name: str) -> str:
    """Generate a natural key for a column."""
    return f"db:{table_name}.{column_name}"


async def build_schema_graph(
    session: AsyncSession,
    collection_id: UUID,
    schema: AggregatedSchema,
) -> dict:
    """Build knowledge graph nodes and edges from schema.

    Creates:
    - DB_TABLE nodes for each table
    - DB_COLUMN nodes for each column
    - TABLE_HAS_COLUMN edges
    - COLUMN_FK_TO_COLUMN edges

    Args:
        session: Database session
        collection_id: Collection UUID
        schema: The aggregated schema

    Returns:
        Stats dict
    """
    from contextmine_core.models import (
        KnowledgeEdge,
        KnowledgeEdgeKind,
        KnowledgeEvidence,
        KnowledgeNode,
        KnowledgeNodeEvidence,
        KnowledgeNodeKind,
    )
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
        natural_key = get_table_natural_key(table_name)

        stmt = pg_insert(KnowledgeNode).values(
            collection_id=collection_id,
            kind=KnowledgeNodeKind.DB_TABLE,
            natural_key=natural_key,
            name=table_name,
            meta={
                "column_count": len(table.columns),
                "primary_keys": table.primary_keys,
                "description": table.description,
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
            col_natural_key = get_column_natural_key(table_name, col.name)

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
                    "description": col.description,
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
    schema: AggregatedSchema,
    name: str = "Database ERD",
) -> UUID:
    """Generate and save Mermaid ERD as an artifact.

    Args:
        session: Database session
        collection_id: Collection UUID
        schema: The aggregated schema
        name: Artifact name

    Returns:
        Artifact ID
    """
    from contextmine_core.models import KnowledgeArtifact, KnowledgeArtifactKind
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
