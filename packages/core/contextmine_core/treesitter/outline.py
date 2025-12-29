"""Symbol extraction from Tree-sitter parsed trees.

This module provides functions to extract symbol outlines (functions, classes,
methods, etc.) from source code using Tree-sitter syntax trees.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from contextmine_core.treesitter.languages import (
    TreeSitterLanguage,
    detect_language,
    get_symbol_query,
)
from contextmine_core.treesitter.manager import get_treesitter_manager

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SymbolKind(Enum):
    """Types of symbols that can be extracted."""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    STRUCT = "struct"
    ENUM = "enum"
    INTERFACE = "interface"
    TYPE = "type"
    TRAIT = "trait"
    IMPL = "impl"
    MODULE = "module"
    VARIABLE = "variable"
    UNKNOWN = "unknown"


# Map query capture names to SymbolKind
CAPTURE_TO_KIND: dict[str, SymbolKind] = {
    "definition.function": SymbolKind.FUNCTION,
    "definition.class": SymbolKind.CLASS,
    "definition.method": SymbolKind.METHOD,
    "definition.struct": SymbolKind.STRUCT,
    "definition.enum": SymbolKind.ENUM,
    "definition.interface": SymbolKind.INTERFACE,
    "definition.type": SymbolKind.TYPE,
    "definition.trait": SymbolKind.TRAIT,
    "definition.impl": SymbolKind.IMPL,
    "definition.module": SymbolKind.MODULE,
}


@dataclass
class Symbol:
    """A code symbol extracted from source.

    Represents a function, class, method, or other named code entity.
    """

    name: str
    """The symbol name."""

    kind: SymbolKind
    """The type of symbol (function, class, etc.)."""

    file_path: str
    """Path to the source file."""

    start_line: int
    """Starting line number (1-indexed)."""

    end_line: int
    """Ending line number (1-indexed)."""

    start_column: int = 0
    """Starting column (0-indexed)."""

    end_column: int = 0
    """Ending column (0-indexed)."""

    signature: str | None = None
    """Function/method signature if available."""

    parent: str | None = None
    """Name of the parent symbol (e.g., class for a method)."""

    docstring: str | None = None
    """Documentation string if available."""

    children: list[Symbol] = field(default_factory=list)
    """Child symbols (e.g., methods inside a class)."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "kind": self.kind.value,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "start_column": self.start_column,
            "end_column": self.end_column,
            "signature": self.signature,
            "parent": self.parent,
            "docstring": self.docstring,
            "children": [c.to_dict() for c in self.children],
        }

    def contains_line(self, line: int) -> bool:
        """Check if this symbol contains the given line.

        Args:
            line: Line number (1-indexed)

        Returns:
            True if line is within this symbol's range
        """
        return self.start_line <= line <= self.end_line


def extract_outline(
    file_path: str | Path,
    content: str | None = None,
    include_children: bool = True,
) -> list[Symbol]:
    """Extract symbol outline from a source file.

    Args:
        file_path: Path to the source file
        content: Optional file content (reads from file if None)
        include_children: Whether to extract nested symbols

    Returns:
        List of top-level symbols found in the file
    """
    file_path = str(Path(file_path).resolve())

    # Read content if not provided
    if content is None:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        content = path.read_text(encoding="utf-8", errors="replace")

    # Detect language
    language = detect_language(file_path)
    if language is None:
        logger.debug("Unsupported file type: %s", Path(file_path).suffix)
        return []

    # Try query-based extraction first
    query_str = get_symbol_query(language)
    if query_str:
        try:
            symbols = _extract_with_query(file_path, content, language, query_str)
            if symbols:
                if include_children:
                    _nest_symbols(symbols)
                return [s for s in symbols if s.parent is None]
        except Exception as e:
            logger.debug("Query-based extraction failed, falling back: %s", e)

    # Fallback to manual traversal
    return _extract_with_traversal(file_path, content, language, include_children)


def _extract_with_query(
    file_path: str,
    content: str,
    language: TreeSitterLanguage,
    query_str: str,
) -> list[Symbol]:
    """Extract symbols using tree-sitter queries.

    Args:
        file_path: Path to source file
        content: File content
        language: The programming language
        query_str: Tree-sitter query string

    Returns:
        List of extracted symbols
    """
    from tree_sitter import Query, QueryCursor
    from tree_sitter_language_pack import get_language

    manager = get_treesitter_manager()
    tree = manager.parse(file_path, content)

    # Get the language object and create query
    ts_language = get_language(language.value)
    query = Query(ts_language, query_str)
    cursor = QueryCursor(query)

    # Run the query using QueryCursor
    definition_nodes: dict[int, dict[str, Any]] = {}

    for _pattern_index, captures_dict in cursor.matches(tree.root_node):
        # Process each capture group
        for capture_name, nodes in captures_dict.items():
            for node in nodes:
                if capture_name.startswith("definition."):
                    node_id = id(node)
                    if node_id not in definition_nodes:
                        definition_nodes[node_id] = {
                            "node": node,
                            "kind": CAPTURE_TO_KIND.get(capture_name, SymbolKind.UNKNOWN),
                            "name": None,
                        }
                elif capture_name == "name":
                    # Find the parent definition for this name
                    parent = node.parent
                    while parent is not None:
                        parent_id = id(parent)
                        if parent_id in definition_nodes:
                            if node.text is not None:
                                definition_nodes[parent_id]["name"] = node.text.decode("utf-8")
                            break
                        parent = parent.parent

    # Build Symbol objects
    symbols: list[Symbol] = []
    lines = content.split("\n")

    for data in definition_nodes.values():
        node = data["node"]
        name = data["name"]
        if not name:
            continue

        # Extract signature (first line of the definition)
        start_line = node.start_point[0] + 1  # Convert to 1-indexed
        end_line = node.end_point[0] + 1

        signature = None
        if start_line <= len(lines):
            signature = lines[start_line - 1].strip()

        # Extract docstring (look for string after definition)
        docstring = _extract_docstring(node, language)

        symbol = Symbol(
            name=name,
            kind=data["kind"],
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            start_column=node.start_point[1],
            end_column=node.end_point[1],
            signature=signature,
            docstring=docstring,
        )
        symbols.append(symbol)

    return symbols


def _extract_with_traversal(
    file_path: str,
    content: str,
    language: TreeSitterLanguage,
    include_children: bool,
) -> list[Symbol]:
    """Extract symbols using manual tree traversal.

    Fallback when queries aren't available or fail.

    Args:
        file_path: Path to source file
        content: File content
        language: The programming language
        include_children: Whether to extract nested symbols

    Returns:
        List of extracted symbols
    """
    manager = get_treesitter_manager()
    tree = manager.parse(file_path, content)

    symbols: list[Symbol] = []
    lines = content.split("\n")

    # Node types that represent symbol definitions by language
    symbol_types = _get_symbol_node_types(language)

    def traverse(node: Any, parent_name: str | None = None) -> None:
        node_type = node.type

        if node_type in symbol_types:
            name = _extract_name_from_node(node, language)
            if name:
                kind = symbol_types[node_type]
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1

                signature = None
                if start_line <= len(lines):
                    signature = lines[start_line - 1].strip()

                docstring = _extract_docstring(node, language)

                symbol = Symbol(
                    name=name,
                    kind=kind,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    start_column=node.start_point[1],
                    end_column=node.end_point[1],
                    signature=signature,
                    parent=parent_name,
                    docstring=docstring,
                )
                symbols.append(symbol)

                # Traverse children with this symbol as parent
                if include_children:
                    for child in node.children:
                        traverse(child, name)
                return

        # Continue traversing
        for child in node.children:
            traverse(child, parent_name)

    traverse(tree.root_node)

    if include_children:
        _nest_symbols(symbols)

    return [s for s in symbols if s.parent is None]


def _get_symbol_node_types(language: TreeSitterLanguage) -> dict[str, SymbolKind]:
    """Get node types that represent symbols for a language.

    Args:
        language: The programming language

    Returns:
        Mapping of node type to SymbolKind
    """
    if language == TreeSitterLanguage.PYTHON:
        return {
            "function_definition": SymbolKind.FUNCTION,
            "class_definition": SymbolKind.CLASS,
            "decorated_definition": SymbolKind.FUNCTION,  # Will be refined
        }
    elif language in (
        TreeSitterLanguage.TYPESCRIPT,
        TreeSitterLanguage.TSX,
        TreeSitterLanguage.JAVASCRIPT,
    ):
        return {
            "function_declaration": SymbolKind.FUNCTION,
            "class_declaration": SymbolKind.CLASS,
            "method_definition": SymbolKind.METHOD,
            "interface_declaration": SymbolKind.INTERFACE,
            "type_alias_declaration": SymbolKind.TYPE,
        }
    elif language == TreeSitterLanguage.RUST:
        return {
            "function_item": SymbolKind.FUNCTION,
            "struct_item": SymbolKind.STRUCT,
            "enum_item": SymbolKind.ENUM,
            "impl_item": SymbolKind.IMPL,
            "trait_item": SymbolKind.TRAIT,
        }
    elif language == TreeSitterLanguage.GO:
        return {
            "function_declaration": SymbolKind.FUNCTION,
            "method_declaration": SymbolKind.METHOD,
            "type_declaration": SymbolKind.TYPE,
        }
    elif language == TreeSitterLanguage.JAVA:
        return {
            "method_declaration": SymbolKind.METHOD,
            "class_declaration": SymbolKind.CLASS,
            "interface_declaration": SymbolKind.INTERFACE,
            "enum_declaration": SymbolKind.ENUM,
        }
    else:
        # Generic fallback
        return {
            "function_definition": SymbolKind.FUNCTION,
            "function_declaration": SymbolKind.FUNCTION,
            "class_definition": SymbolKind.CLASS,
            "class_declaration": SymbolKind.CLASS,
            "method_definition": SymbolKind.METHOD,
            "method_declaration": SymbolKind.METHOD,
        }


def _extract_name_from_node(node: Any, language: TreeSitterLanguage) -> str | None:
    """Extract the symbol name from a node.

    Args:
        node: Tree-sitter node
        language: The programming language

    Returns:
        Symbol name if found, None otherwise
    """
    # Look for name child
    for child in node.children:
        if child.type in ("identifier", "name", "type_identifier", "property_identifier"):
            return child.text.decode("utf-8")

        # Handle decorated definitions in Python
        if child.type in ("function_definition", "class_definition"):
            return _extract_name_from_node(child, language)

        # Handle function declarator in C/C++
        if child.type == "function_declarator":
            return _extract_name_from_node(child, language)

    return None


def _extract_docstring(node: Any, language: TreeSitterLanguage) -> str | None:
    """Extract docstring from a definition node.

    Args:
        node: Tree-sitter node for the definition
        language: The programming language

    Returns:
        Docstring if found, None otherwise
    """
    if language == TreeSitterLanguage.PYTHON:
        # Look for expression_statement with string as first child in body
        for child in node.children:
            if child.type == "block":
                for block_child in child.children:
                    if block_child.type == "expression_statement":
                        for expr_child in block_child.children:
                            if expr_child.type == "string":
                                text = expr_child.text.decode("utf-8")
                                # Strip quotes
                                if text.startswith('"""') or text.startswith("'''"):
                                    return text[3:-3].strip()
                                elif text.startswith('"') or text.startswith("'"):
                                    return text[1:-1].strip()
                        break
                break
    return None


def _nest_symbols(symbols: list[Symbol]) -> None:
    """Nest child symbols under their parents.

    Modifies symbols in place, setting parent references and populating
    children lists.

    Args:
        symbols: List of symbols to nest
    """
    # Sort by start line for proper nesting
    symbols.sort(key=lambda s: (s.start_line, -s.end_line))

    # Build parent-child relationships
    for i, symbol in enumerate(symbols):
        if symbol.parent:
            # Already has parent from traversal
            continue

        # Find potential parent (symbol that contains this one)
        for j in range(i - 1, -1, -1):
            potential_parent = symbols[j]
            if (
                potential_parent.start_line <= symbol.start_line
                and potential_parent.end_line >= symbol.end_line
                and potential_parent is not symbol
            ):
                symbol.parent = potential_parent.name
                potential_parent.children.append(symbol)
                break


def find_symbol_by_name(
    file_path: str | Path,
    name: str,
    content: str | None = None,
) -> Symbol | None:
    """Find a symbol by name in a file.

    Args:
        file_path: Path to the source file
        name: Symbol name to find
        content: Optional file content

    Returns:
        Symbol if found, None otherwise
    """
    symbols = extract_outline(file_path, content, include_children=True)

    def search(syms: list[Symbol]) -> Symbol | None:
        for sym in syms:
            if sym.name == name:
                return sym
            found = search(sym.children)
            if found:
                return found
        return None

    return search(symbols)


def find_enclosing_symbol(
    file_path: str | Path,
    line: int,
    content: str | None = None,
) -> Symbol | None:
    """Find the innermost symbol containing a line.

    Args:
        file_path: Path to the source file
        line: Line number (1-indexed)
        content: Optional file content

    Returns:
        Innermost Symbol containing the line, None if not in any symbol
    """
    symbols = extract_outline(file_path, content, include_children=True)

    def find_innermost(syms: list[Symbol]) -> Symbol | None:
        best: Symbol | None = None
        for sym in syms:
            if sym.contains_line(line):
                # This symbol contains the line, but check children for more specific
                best = sym
                inner = find_innermost(sym.children)
                if inner:
                    best = inner
        return best

    return find_innermost(symbols)


def get_symbol_content(
    symbol: Symbol,
    content: str | None = None,
) -> str:
    """Get the source code content of a symbol.

    Args:
        symbol: The symbol to get content for
        content: Optional file content (reads from file if None)

    Returns:
        The source code of the symbol
    """
    if content is None:
        path = Path(symbol.file_path)
        if not path.exists():
            return ""
        content = path.read_text(encoding="utf-8", errors="replace")

    lines = content.split("\n")
    start_idx = symbol.start_line - 1
    end_idx = symbol.end_line

    return "\n".join(lines[start_idx:end_idx])
