"""Alembic migration file extractor.

Parses Python AST to detect:
- op.create_table() calls
- op.add_column() calls
- op.create_foreign_key() calls
- op.create_index() calls

This avoids regex/string matching and uses proper AST parsing.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ColumnDef:
    """Extracted column definition."""

    name: str
    type_name: str
    nullable: bool = True
    primary_key: bool = False
    foreign_key: str | None = None  # "table.column" format


@dataclass
class TableDef:
    """Extracted table definition."""

    name: str
    columns: list[ColumnDef] = field(default_factory=list)
    primary_keys: list[str] = field(default_factory=list)


@dataclass
class ForeignKeyDef:
    """Extracted foreign key definition."""

    name: str | None
    source_table: str
    source_columns: list[str]
    target_table: str
    target_columns: list[str]


@dataclass
class AlembicExtraction:
    """Result of parsing an Alembic migration file."""

    file_path: str
    tables: list[TableDef] = field(default_factory=list)
    foreign_keys: list[ForeignKeyDef] = field(default_factory=list)
    added_columns: list[tuple[str, ColumnDef]] = field(default_factory=list)  # (table, column)


class AlembicVisitor(ast.NodeVisitor):
    """AST visitor for extracting Alembic operations."""

    def __init__(self) -> None:
        self.tables: list[TableDef] = []
        self.foreign_keys: list[ForeignKeyDef] = []
        self.added_columns: list[tuple[str, ColumnDef]] = []

    def visit_Call(self, node: ast.Call) -> None:
        """Visit function calls to detect Alembic operations."""
        # Check for op.xxx() pattern
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "op"
        ):
            method = node.func.attr

            if method == "create_table":
                self._handle_create_table(node)
            elif method == "add_column":
                self._handle_add_column(node)
            elif method == "create_foreign_key":
                self._handle_create_foreign_key(node)

        self.generic_visit(node)

    def _handle_create_table(self, node: ast.Call) -> None:
        """Handle op.create_table() call."""
        if not node.args:
            return

        table_name = self._get_string_value(node.args[0])
        if not table_name:
            return

        table = TableDef(name=table_name)

        # Process remaining args as columns
        for arg in node.args[1:]:
            if isinstance(arg, ast.Call):
                col = self._parse_column(arg)
                if col:
                    table.columns.append(col)
                    if col.primary_key:
                        table.primary_keys.append(col.name)

        self.tables.append(table)

    def _handle_add_column(self, node: ast.Call) -> None:
        """Handle op.add_column() call."""
        if len(node.args) < 2:
            return

        table_name = self._get_string_value(node.args[0])
        if not table_name:
            return

        # Second arg should be sa.Column()
        if isinstance(node.args[1], ast.Call):
            col = self._parse_column(node.args[1])
            if col:
                self.added_columns.append((table_name, col))

    def _handle_create_foreign_key(self, node: ast.Call) -> None:
        """Handle op.create_foreign_key() call."""
        # op.create_foreign_key(name, source_table, referent_table, local_cols, remote_cols)
        if len(node.args) < 5:
            return

        fk_name = self._get_string_value(node.args[0])
        source_table = self._get_string_value(node.args[1])
        target_table = self._get_string_value(node.args[2])
        source_cols = self._get_list_value(node.args[3])
        target_cols = self._get_list_value(node.args[4])

        if source_table and target_table and source_cols and target_cols:
            self.foreign_keys.append(
                ForeignKeyDef(
                    name=fk_name,
                    source_table=source_table,
                    source_columns=source_cols,
                    target_table=target_table,
                    target_columns=target_cols,
                )
            )

    def _parse_column(self, node: ast.Call) -> ColumnDef | None:
        """Parse a sa.Column() call."""
        # Check if it's Column() or sa.Column()
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name != "Column":
            return None

        if not node.args:
            return None

        col_name = self._get_string_value(node.args[0])
        if not col_name:
            return None

        # Get type from second arg
        type_name = "unknown"
        if len(node.args) > 1:
            type_name = self._get_type_name(node.args[1])

        # Check keywords for nullable, primary_key
        nullable = True
        primary_key = False
        foreign_key = None

        for kw in node.keywords:
            if kw.arg == "nullable":
                nullable = self._get_bool_value(kw.value, default=True)
            elif kw.arg == "primary_key":
                primary_key = self._get_bool_value(kw.value, default=False)

        # Check for ForeignKey in args
        for arg in node.args[1:]:
            if isinstance(arg, ast.Call):
                fk = self._get_foreign_key(arg)
                if fk:
                    foreign_key = fk

        return ColumnDef(
            name=col_name,
            type_name=type_name,
            nullable=nullable,
            primary_key=primary_key,
            foreign_key=foreign_key,
        )

    def _get_foreign_key(self, node: ast.Call) -> str | None:
        """Extract ForeignKey reference from a call node."""
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name != "ForeignKey":
            return None

        if node.args:
            return self._get_string_value(node.args[0])
        return None

    def _get_type_name(self, node: ast.expr) -> str:
        """Get the type name from a type expression."""
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return node.func.id
            elif isinstance(node.func, ast.Attribute):
                return node.func.attr
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return "unknown"

    def _get_string_value(self, node: ast.expr) -> str | None:
        """Extract string value from an AST node."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    def _get_bool_value(self, node: ast.expr, default: bool = False) -> bool:
        """Extract boolean value from an AST node."""
        if isinstance(node, ast.Constant):
            return bool(node.value)
        return default

    def _get_list_value(self, node: ast.expr) -> list[str]:
        """Extract list of strings from an AST node."""
        if isinstance(node, ast.List):
            result = []
            for elt in node.elts:
                val = self._get_string_value(elt)
                if val:
                    result.append(val)
            return result
        return []


def extract_from_alembic(file_path: str, content: str) -> AlembicExtraction:
    """Extract table/column/FK definitions from an Alembic migration file.

    Uses Python AST to parse the file and extract:
    - op.create_table() calls
    - op.add_column() calls
    - op.create_foreign_key() calls

    Args:
        file_path: Path to the migration file
        content: File content

    Returns:
        AlembicExtraction with extracted definitions
    """
    result = AlembicExtraction(file_path=file_path)

    try:
        tree = ast.parse(content)
        visitor = AlembicVisitor()
        visitor.visit(tree)

        result.tables = visitor.tables
        result.foreign_keys = visitor.foreign_keys
        result.added_columns = visitor.added_columns

    except SyntaxError as e:
        logger.warning("Failed to parse %s: %s", file_path, e)

    return result
