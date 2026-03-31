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
from typing import TYPE_CHECKING, Any
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
        in_single, in_double, in_backtick, depth = _update_sql_quote_state(
            char,
            in_single,
            in_double,
            in_backtick,
            depth,
        )
        in_quoted = in_single or in_double or in_backtick
        if char == "," and depth == 0 and not in_quoted:
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


def _update_sql_quote_state(
    char: str,
    in_single: bool,
    in_double: bool,
    in_backtick: bool,
    depth: int,
) -> tuple[bool, bool, bool, int]:
    """Update SQL quote/paren tracking state for a single character."""
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
    return in_single, in_double, in_backtick, depth


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


def _process_sql_ddl_item(
    item: str,
    table: TableDef,
    table_name: str,
    extraction: SchemaExtraction,
) -> None:
    """Process a single item from a SQL CREATE TABLE body."""
    line = item.strip()
    if not line:
        return
    line_upper = line.upper()

    pk_match = SQL_PRIMARY_KEY_RE.search(line)
    if pk_match:
        for pk_col in (
            _strip_sql_identifier(col) for col in pk_match.group(1).split(",") if col.strip()
        ):
            if pk_col and pk_col not in table.primary_keys:
                table.primary_keys.append(pk_col)

    fk_match = SQL_FOREIGN_KEY_RE.search(line)
    if fk_match:
        _collect_sql_fk(fk_match, table_name, extraction)

    if line_upper.startswith(
        ("CONSTRAINT ", "PRIMARY KEY", "FOREIGN KEY", "UNIQUE ", "KEY ", "INDEX ", "CHECK ")
    ):
        return

    parts = line.split(None, 2)
    if len(parts) < 2:
        return

    column = _parse_sql_column_def(parts, line, line_upper)
    table.columns.append(column)
    if column.primary_key and column.name not in table.primary_keys:
        table.primary_keys.append(column.name)


def _collect_sql_fk(
    fk_match: re.Match[str],
    table_name: str,
    extraction: SchemaExtraction,
) -> None:
    """Handle a FOREIGN KEY constraint match."""
    source_columns = [
        _strip_sql_identifier(col) for col in fk_match.group(1).split(",") if col.strip()
    ]
    target_table = _strip_sql_identifier(fk_match.group(2))
    target_columns = [
        _strip_sql_identifier(col) for col in fk_match.group(3).split(",") if col.strip()
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


def _parse_sql_column_def(parts: list[str], line: str, line_upper: str) -> ColumnDef:
    """Parse a SQL column definition from split parts."""
    column_name = _strip_sql_identifier(parts[0])
    raw_type = parts[1]
    inline_fk = SQL_INLINE_REFERENCE_RE.search(line)
    foreign_key = None
    if inline_fk:
        target_table = _strip_sql_identifier(inline_fk.group(1))
        target_column = _strip_sql_identifier(inline_fk.group(2))
        if target_table and target_column:
            foreign_key = f"{target_table}.{target_column}"
    return ColumnDef(
        name=column_name,
        type_name=_normalize_sql_type(raw_type),
        nullable="NOT NULL" not in line_upper,
        primary_key="PRIMARY KEY" in line_upper,
        foreign_key=foreign_key,
    )


def _finalize_table_pks(table: TableDef) -> None:
    """Mark columns as primary key based on the table's primary_keys list."""
    if not table.primary_keys:
        return
    pk_set = set(table.primary_keys)
    for col in table.columns:
        if col.name in pk_set:
            col.primary_key = True


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
        for item in _split_sql_items(match.group(2)):
            _process_sql_ddl_item(item, table, table_name, extraction)
        _finalize_table_pks(table)
        if table.columns:
            extraction.tables.append(table)

    return extraction


# ============================================================================
# Deterministic Python ORM model parsing (Django & SQLAlchemy)
# ============================================================================


_DJANGO_FIELD_TYPE_MAP = {
    "autofield": "Integer",
    "bigautofield": "BigInteger",
    "smallautofield": "SmallInteger",
    "charfield": "String",
    "textfield": "Text",
    "integerfield": "Integer",
    "bigintegerfield": "BigInteger",
    "smallintegerfield": "SmallInteger",
    "positiveintegerfield": "Integer",
    "floatfield": "Float",
    "decimalfield": "Numeric",
    "booleanfield": "Boolean",
    "nullbooleanfield": "Boolean",
    "datetimefield": "DateTime",
    "datefield": "Date",
    "timefield": "Time",
    "uuidfield": "UUID",
    "jsonfield": "JSON",
    "binaryfield": "LargeBinary",
    "filefield": "String",
    "imagefield": "String",
    "emailfield": "String",
    "urlfield": "String",
    "slugfield": "String",
    "ipaddressfield": "String",
    "genericipaddressfield": "String",
    "foreignkey": "Integer",
    "onetoonefield": "Integer",
    "manytomanyfield": "Integer",
}

_SQLALCHEMY_TYPE_MAP = {
    "string": "String",
    "text": "Text",
    "integer": "Integer",
    "biginteger": "BigInteger",
    "smallinteger": "SmallInteger",
    "float": "Float",
    "numeric": "Numeric",
    "boolean": "Boolean",
    "datetime": "DateTime",
    "date": "Date",
    "time": "Time",
    "uuid": "UUID",
    "json": "JSON",
    "largebinary": "LargeBinary",
    "enum": "Enum",
    "array": "ARRAY",
}


def _extract_schema_from_python_orm(file_path: str, content: str) -> SchemaExtraction:
    """Parse Django models and SQLAlchemy models using tree-sitter."""
    from contextmine_core.analyzer.extractors.ast_utils import (
        first_child,
        node_text,
        walk,
    )
    from contextmine_core.treesitter.languages import detect_language
    from contextmine_core.treesitter.manager import get_treesitter_manager

    extraction = SchemaExtraction(file_path=file_path)
    language = detect_language(file_path)
    if language is None:
        return extraction

    manager = get_treesitter_manager()
    try:
        tree = manager.parse(file_path, content)
    except Exception:
        return extraction

    root = tree.root_node
    is_django = "models.Model" in content or "from django" in content
    is_sqlalchemy = (
        "declarative_base" in content
        or "DeclarativeBase" in content
        or "mapped_column" in content
        or "Column(" in content
    )

    if not is_django and not is_sqlalchemy:
        return extraction

    for node in walk(root):
        if node.type != "class_definition":
            continue
        class_name = node_text(content, first_child(node, "identifier")).strip()
        if not class_name:
            continue

        # Check superclass
        superclass_node = node.child_by_field_name("superclasses")
        if superclass_node is None:
            # Try argument_list (Python 3 style)
            superclass_node = first_child(node, "argument_list")
        if superclass_node is None:
            continue
        superclass_text = node_text(content, superclass_node)

        if is_django and "Model" in superclass_text:
            table = _parse_django_model(content, node, class_name)
            if table and table.columns:
                extraction.tables.append(table)
                extraction.framework = "django"
        elif is_sqlalchemy and ("Base" in superclass_text or "DeclarativeBase" in superclass_text):
            table = _parse_sqlalchemy_model(content, node, class_name)
            if table and table.columns:
                extraction.tables.append(table)
                extraction.framework = "sqlalchemy"

    return extraction


def _parse_django_model(content: str, class_node: Any, class_name: str) -> TableDef | None:
    """Extract a Django model class into a TableDef."""
    from contextmine_core.analyzer.extractors.ast_utils import node_text, walk

    # Django convention: model name -> lowercase with underscores
    table_name = _django_table_name(class_name)
    table = TableDef(name=table_name)

    for node in walk(class_node):
        if node.type != "assignment" and node.type != "expression_statement":
            continue
        assign = node if node.type == "assignment" else None
        if assign is None:
            # Check for assignment inside expression_statement
            for child in node.children:
                if child.type == "assignment":
                    assign = child
                    break
        if assign is None:
            continue

        left = assign.child_by_field_name("left")
        right = assign.child_by_field_name("right")
        if left is None or right is None:
            continue

        field_name = node_text(content, left).strip()
        if not field_name or field_name.startswith("_") or field_name == "Meta":
            continue

        right_text = node_text(content, right).strip()
        if "models." not in right_text and "Field" not in right_text:
            continue

        col = _parse_django_field(field_name, right_text)
        if col:
            table.columns.append(col)

    # Django auto-adds an id PK unless explicitly overridden
    has_pk = any(c.primary_key for c in table.columns)
    if not has_pk and table.columns:
        table.columns.insert(
            0, ColumnDef(name="id", type_name="BigInteger", nullable=False, primary_key=True)
        )
        table.primary_keys.append("id")

    return table


def _django_table_name(class_name: str) -> str:
    """Convert PascalCase model name to Django-style table name (lowercase with underscores)."""
    import re

    name = re.sub(r"([A-Z])", r"_\1", class_name).lower().lstrip("_")
    return name


def _parse_django_field(field_name: str, right_text: str) -> ColumnDef | None:
    """Parse a Django model field assignment."""
    # Extract field type: models.CharField(...) or CharField(...)
    import re

    match = re.search(
        r"(?:models\.)?(\w+Field|ForeignKey|OneToOneField|ManyToManyField)\s*\(", right_text
    )
    if not match:
        return None
    field_type = match.group(1).lower()
    type_name = _DJANGO_FIELD_TYPE_MAP.get(field_type, "unknown")

    nullable = "null=True" in right_text.lower() or "null = True" in right_text.lower()
    primary_key = "primary_key=True" in right_text or "primary_key = True" in right_text

    # Foreign key reference
    foreign_key = None
    if field_type in {"foreignkey", "onetoonefield"}:
        fk_match = re.search(r"""(?:ForeignKey|OneToOneField)\s*\(\s*['"]?(\w+)['"]?""", right_text)
        if fk_match:
            ref_model = fk_match.group(1)
            foreign_key = f"{_django_table_name(ref_model)}.id"

    return ColumnDef(
        name=field_name,
        type_name=type_name,
        nullable=nullable,
        primary_key=primary_key,
        foreign_key=foreign_key,
    )


def _parse_sqlalchemy_model(content: str, class_node: Any, class_name: str) -> TableDef | None:
    """Extract a SQLAlchemy model class into a TableDef."""
    import re

    from contextmine_core.analyzer.extractors.ast_utils import node_text, walk

    table = TableDef(name=class_name)

    # Look for __tablename__ assignment
    for node in walk(class_node):
        if node.type != "assignment" and node.type != "expression_statement":
            continue
        text = node_text(content, node).strip()
        if "__tablename__" in text:
            match = re.search(r"""__tablename__\s*=\s*['"](\w+)['"]""", text)
            if match:
                table.name = match.group(1)
            break

    # Parse columns
    for node in walk(class_node):
        if node.type not in {"assignment", "expression_statement"}:
            continue
        assign_text = node_text(content, node).strip()

        # mapped_column() or Column() patterns
        col = _parse_sqlalchemy_column(assign_text)
        if col:
            table.columns.append(col)

    return table


def _parse_sqlalchemy_column(assign_text: str) -> ColumnDef | None:
    """Parse a SQLAlchemy Column() or mapped_column() assignment."""
    import re

    # Match: field_name = Column(...) or field_name: Mapped[...] = mapped_column(...)
    match = re.match(
        r"(\w+)\s*(?::\s*Mapped\[.*?\])?\s*=\s*(?:mapped_column|Column)\s*\(", assign_text
    )
    if not match:
        return None

    col_name = match.group(1)
    if col_name.startswith("_") or col_name == "__tablename__":
        return None

    # Extract type from first argument or Mapped[type] annotation
    type_name = "unknown"
    type_match = re.search(r"Mapped\[(\w+)", assign_text)
    if type_match:
        mapped_type = type_match.group(1).lower()
        py_to_sql = {
            "str": "String",
            "int": "Integer",
            "float": "Float",
            "bool": "Boolean",
            "datetime": "DateTime",
            "date": "Date",
            "uuid": "UUID",
            "bytes": "LargeBinary",
        }
        type_name = py_to_sql.get(mapped_type, mapped_type.title())
    else:
        # Check first positional arg: Column(String, ...) or Column(Integer, ...)
        arg_match = re.search(r"(?:mapped_column|Column)\s*\(\s*(\w+)", assign_text)
        if arg_match:
            sa_type = arg_match.group(1).lower()
            type_name = _SQLALCHEMY_TYPE_MAP.get(sa_type, arg_match.group(1))

    nullable = "nullable=False" not in assign_text
    primary_key = "primary_key=True" in assign_text or "primary_key = True" in assign_text

    foreign_key = None
    fk_match = re.search(r"""ForeignKey\s*\(\s*['"]([^'"]+)['"]""", assign_text)
    if fk_match:
        foreign_key = fk_match.group(1)

    return ColumnDef(
        name=col_name,
        type_name=type_name,
        nullable=nullable,
        primary_key=primary_key,
        foreign_key=foreign_key,
    )


# ============================================================================
# Deterministic Prisma schema parsing
# ============================================================================


def _extract_schema_from_prisma(file_path: str, content: str) -> SchemaExtraction:
    """Parse Prisma schema files deterministically using regex."""
    extraction = SchemaExtraction(file_path=file_path, framework="prisma")

    _PRISMA_TYPE_MAP = {
        "string": "String",
        "int": "Integer",
        "bigint": "BigInteger",
        "float": "Float",
        "decimal": "Numeric",
        "boolean": "Boolean",
        "datetime": "DateTime",
        "json": "JSON",
        "bytes": "LargeBinary",
    }

    # Match model blocks: model User { ... }
    model_pattern = re.compile(r"model\s+(\w+)\s*\{([^}]+)\}", re.DOTALL)
    for match in model_pattern.finditer(content):
        model_name = match.group(1)
        body = match.group(2)
        table = TableDef(name=model_name)

        for line in body.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("@@"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            field_name = parts[0]
            if field_name.startswith("@@") or field_name.startswith("//"):
                continue

            field_type_raw = parts[1]
            # Handle optional (?) and array ([]) markers
            is_optional = field_type_raw.endswith("?")
            field_type_clean = field_type_raw.rstrip("?").rstrip("[]")

            type_name = _PRISMA_TYPE_MAP.get(field_type_clean.lower(), field_type_clean)

            is_pk = "@id" in line
            is_relation = "@relation" in line
            foreign_key = None

            if is_relation:
                # Try to extract references
                ref_match = re.search(r"references:\s*\[(\w+)\]", line)
                fields_match = re.search(r"fields:\s*\[(\w+)\]", line)
                if ref_match:
                    foreign_key = f"{field_type_clean}.{ref_match.group(1)}"
                # Skip pure relation fields (no scalar column)
                if not fields_match:
                    continue

            col = ColumnDef(
                name=field_name,
                type_name=type_name,
                nullable=is_optional,
                primary_key=is_pk,
                foreign_key=foreign_key,
            )
            table.columns.append(col)
            if is_pk:
                table.primary_keys.append(field_name)

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

    if path_lower.endswith(".prisma"):
        deterministic = _extract_schema_from_prisma(file_path, content)
        if deterministic.tables:
            return deterministic

    if path_lower.endswith(".py"):
        deterministic = _extract_schema_from_python_orm(file_path, content)
        if deterministic.tables:
            return deterministic

    # LLM fallback for formats without deterministic parsers
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


def _fk_signature(fk: ForeignKeyDef) -> tuple[str, tuple[str, ...], str, tuple[str, ...]]:
    """Return a deduplication key for a foreign key definition."""
    return (fk.source_table, tuple(fk.source_columns), fk.target_table, tuple(fk.target_columns))


def _merge_table(schema: AggregatedSchema, table: TableDef) -> None:
    """Merge a table into the schema, combining columns if it already exists."""
    if table.name not in schema.tables:
        schema.tables[table.name] = table
        return
    existing = schema.tables[table.name]
    existing_col_names = {c.name for c in existing.columns}
    for col in table.columns:
        if col.name not in existing_col_names:
            existing.columns.append(col)
            if col.primary_key and col.name not in existing.primary_keys:
                existing.primary_keys.append(col.name)


def _collect_column_fks(
    tables: list[TableDef],
    existing_fks: set[tuple[str, tuple[str, ...], str, tuple[str, ...]]],
    schema: AggregatedSchema,
) -> None:
    """Extract FK definitions from column-level foreign_key references."""
    for table in tables:
        for col in table.columns:
            if not col.foreign_key:
                continue
            parts = col.foreign_key.split(".")
            if len(parts) != 2:
                continue
            fk = ForeignKeyDef(
                name=None,
                source_table=table.name,
                source_columns=[col.name],
                target_table=parts[0],
                target_columns=[parts[1]],
            )
            sig = _fk_signature(fk)
            if sig not in existing_fks:
                schema.foreign_keys.append(fk)
                existing_fks.add(sig)


def aggregate_schema_extractions(extractions: list[SchemaExtraction]) -> AggregatedSchema:
    """Aggregate multiple schema extractions into a unified schema."""
    schema = AggregatedSchema()

    for extraction in extractions:
        if extraction.tables or extraction.foreign_keys:
            schema.sources.append(extraction.file_path)

        for table in extraction.tables:
            _merge_table(schema, table)

        existing_fks = {_fk_signature(fk) for fk in schema.foreign_keys}

        for fk in extraction.foreign_keys:
            sig = _fk_signature(fk)
            if sig not in existing_fks:
                schema.foreign_keys.append(fk)
                existing_fks.add(sig)

        _collect_column_fks(extraction.tables, existing_fks, schema)

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

    for table_name, table in sorted(schema.tables.items()):
        _render_mermaid_table(lines, table_name, table)

    _render_mermaid_relationships(lines, schema)
    return "\n".join(lines)


def _render_mermaid_table(lines: list[str], table_name: str, table: TableDef) -> None:
    """Render a single table into mermaid ERD lines."""
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


def _render_mermaid_relationships(lines: list[str], schema: AggregatedSchema) -> None:
    """Render FK relationships into mermaid ERD lines."""
    seen: set[tuple[str, str]] = set()
    for fk in schema.foreign_keys:
        if fk.source_table not in schema.tables or fk.target_table not in schema.tables:
            continue
        source = _mermaid_safe(fk.source_table)
        target = _mermaid_safe(fk.target_table)
        rel_key = (source, target)
        if rel_key in seen:
            continue
        seen.add(rel_key)
        lines.append(f"    {target} ||--o{{ {source} : has")


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


async def _upsert_column_node(
    session: AsyncSession,
    collection_id: UUID,
    table_name: str,
    col: ColumnDef,
    table_node_id: UUID,
    pg_insert: Any,
    knowledge_node_cls: Any,
    node_kind_cls: Any,
    edge_cls: Any,
    edge_kind_cls: Any,
    select_fn: Any,
    stats: dict,
) -> UUID:
    """Upsert a column node and its TABLE_HAS_COLUMN edge. Returns the column node id."""
    col_natural_key = get_column_natural_key(table_name, col.name)
    col_stmt = pg_insert(knowledge_node_cls).values(
        collection_id=collection_id,
        kind=node_kind_cls.DB_COLUMN,
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
        set_={"name": col_stmt.excluded.name, "meta": col_stmt.excluded.meta},
    ).returning(knowledge_node_cls.id)
    col_result = await session.execute(col_stmt)
    col_node_id = col_result.scalar_one()
    stats["column_nodes_created"] += 1

    edge_exists = await session.execute(
        select_fn(edge_cls.id).where(
            edge_cls.collection_id == collection_id,
            edge_cls.source_node_id == table_node_id,
            edge_cls.target_node_id == col_node_id,
            edge_cls.kind == edge_kind_cls.TABLE_HAS_COLUMN,
        )
    )
    if not edge_exists.scalar_one_or_none():
        session.add(
            edge_cls(
                collection_id=collection_id,
                source_node_id=table_node_id,
                target_node_id=col_node_id,
                kind=edge_kind_cls.TABLE_HAS_COLUMN,
                meta={},
            )
        )
        stats["edges_created"] += 1
    return col_node_id


async def _create_fk_edges(
    session: AsyncSession,
    collection_id: UUID,
    foreign_keys: list[ForeignKeyDef],
    column_node_ids: dict[str, UUID],
    edge_cls: Any,
    edge_kind_cls: Any,
    select_fn: Any,
    stats: dict,
) -> None:
    """Create COLUMN_FK_TO_COLUMN edges from foreign key definitions."""
    for fk in foreign_keys:
        for src_col, tgt_col in zip(fk.source_columns, fk.target_columns, strict=False):
            src_node_id = column_node_ids.get(f"{fk.source_table}.{src_col}")
            tgt_node_id = column_node_ids.get(f"{fk.target_table}.{tgt_col}")
            if not src_node_id or not tgt_node_id:
                continue
            edge_exists = await session.execute(
                select_fn(edge_cls.id).where(
                    edge_cls.collection_id == collection_id,
                    edge_cls.source_node_id == src_node_id,
                    edge_cls.target_node_id == tgt_node_id,
                    edge_cls.kind == edge_kind_cls.COLUMN_FK_TO_COLUMN,
                )
            )
            if not edge_exists.scalar_one_or_none():
                session.add(
                    edge_cls(
                        collection_id=collection_id,
                        source_node_id=src_node_id,
                        target_node_id=tgt_node_id,
                        kind=edge_kind_cls.COLUMN_FK_TO_COLUMN,
                        meta={"fk_name": fk.name},
                    )
                )
                stats["edges_created"] += 1


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

        for col in table.columns:
            col_node_id = await _upsert_column_node(
                session,
                collection_id,
                table_name,
                col,
                node_id,
                pg_insert,
                KnowledgeNode,
                KnowledgeNodeKind,
                KnowledgeEdge,
                KnowledgeEdgeKind,
                select,
                stats,
            )
            column_node_ids[f"{table_name}.{col.name}"] = col_node_id

    await _create_fk_edges(
        session,
        collection_id,
        schema.foreign_keys,
        column_node_ids,
        KnowledgeEdge,
        KnowledgeEdgeKind,
        select,
        stats,
    )

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
