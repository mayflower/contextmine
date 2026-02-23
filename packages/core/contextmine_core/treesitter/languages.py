"""Tree-sitter language detection and configuration.

This module provides language detection based on file extensions and
query patterns for symbol extraction.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path


class TreeSitterLanguage(Enum):
    """Languages supported by Tree-sitter in this module."""

    PYTHON = "python"
    TYPESCRIPT = "typescript"
    TSX = "tsx"
    JAVASCRIPT = "javascript"
    RUST = "rust"
    GO = "go"
    JAVA = "java"
    C = "c"
    CPP = "cpp"
    CSHARP = "c_sharp"
    RUBY = "ruby"
    PHP = "php"
    GRAPHQL = "graphql"
    PROTO = "proto"


# Map file extensions to tree-sitter language names
EXTENSION_TO_LANGUAGE: dict[str, TreeSitterLanguage] = {
    ".py": TreeSitterLanguage.PYTHON,
    ".pyi": TreeSitterLanguage.PYTHON,
    ".ts": TreeSitterLanguage.TYPESCRIPT,
    ".tsx": TreeSitterLanguage.TSX,
    ".js": TreeSitterLanguage.JAVASCRIPT,
    ".jsx": TreeSitterLanguage.JAVASCRIPT,
    ".mjs": TreeSitterLanguage.JAVASCRIPT,
    ".cjs": TreeSitterLanguage.JAVASCRIPT,
    ".rs": TreeSitterLanguage.RUST,
    ".go": TreeSitterLanguage.GO,
    ".java": TreeSitterLanguage.JAVA,
    ".c": TreeSitterLanguage.C,
    ".h": TreeSitterLanguage.C,
    ".cpp": TreeSitterLanguage.CPP,
    ".hpp": TreeSitterLanguage.CPP,
    ".cc": TreeSitterLanguage.CPP,
    ".cxx": TreeSitterLanguage.CPP,
    ".cs": TreeSitterLanguage.CSHARP,
    ".rb": TreeSitterLanguage.RUBY,
    ".php": TreeSitterLanguage.PHP,
    ".graphql": TreeSitterLanguage.GRAPHQL,
    ".gql": TreeSitterLanguage.GRAPHQL,
    ".proto": TreeSitterLanguage.PROTO,
}


def detect_language(file_path: str | Path) -> TreeSitterLanguage | None:
    """Detect the programming language from a file path.

    Args:
        file_path: Path to a source file (can be a URI with query params)

    Returns:
        TreeSitterLanguage if recognized, None otherwise
    """
    # Strip query parameters from URIs (e.g., git://...?ref=main)
    path_str = str(file_path).split("?")[0]
    path = Path(path_str)
    suffix = path.suffix.lower()
    return EXTENSION_TO_LANGUAGE.get(suffix)


# Query patterns for symbol extraction (tree-sitter query syntax)
# These are S-expression patterns that match symbol definitions
SYMBOL_QUERIES: dict[TreeSitterLanguage, str] = {
    TreeSitterLanguage.PYTHON: """
        (function_definition
            name: (identifier) @name) @definition.function

        (class_definition
            name: (identifier) @name) @definition.class

        (decorated_definition
            definition: (function_definition
                name: (identifier) @name)) @definition.function

        (decorated_definition
            definition: (class_definition
                name: (identifier) @name)) @definition.class
    """,
    TreeSitterLanguage.TYPESCRIPT: """
        (function_declaration
            name: (identifier) @name) @definition.function

        (class_declaration
            name: (type_identifier) @name) @definition.class

        (method_definition
            name: (property_identifier) @name) @definition.method

        (interface_declaration
            name: (type_identifier) @name) @definition.interface

        (type_alias_declaration
            name: (type_identifier) @name) @definition.type

        (lexical_declaration
            (variable_declarator
                name: (identifier) @name
                value: (arrow_function))) @definition.function
    """,
    TreeSitterLanguage.TSX: """
        (function_declaration
            name: (identifier) @name) @definition.function

        (class_declaration
            name: (type_identifier) @name) @definition.class

        (method_definition
            name: (property_identifier) @name) @definition.method

        (interface_declaration
            name: (type_identifier) @name) @definition.interface

        (type_alias_declaration
            name: (type_identifier) @name) @definition.type

        (lexical_declaration
            (variable_declarator
                name: (identifier) @name
                value: (arrow_function))) @definition.function
    """,
    TreeSitterLanguage.JAVASCRIPT: """
        (function_declaration
            name: (identifier) @name) @definition.function

        (class_declaration
            name: (identifier) @name) @definition.class

        (method_definition
            name: (property_identifier) @name) @definition.method

        (lexical_declaration
            (variable_declarator
                name: (identifier) @name
                value: (arrow_function))) @definition.function
    """,
    TreeSitterLanguage.RUST: """
        (function_item
            name: (identifier) @name) @definition.function

        (struct_item
            name: (type_identifier) @name) @definition.struct

        (enum_item
            name: (type_identifier) @name) @definition.enum

        (impl_item
            trait: (type_identifier)? @trait
            type: (type_identifier) @name) @definition.impl

        (trait_item
            name: (type_identifier) @name) @definition.trait
    """,
    TreeSitterLanguage.GO: """
        (function_declaration
            name: (identifier) @name) @definition.function

        (method_declaration
            name: (field_identifier) @name) @definition.method

        (type_declaration
            (type_spec
                name: (type_identifier) @name)) @definition.type
    """,
    TreeSitterLanguage.JAVA: """
        (method_declaration
            name: (identifier) @name) @definition.method

        (class_declaration
            name: (identifier) @name) @definition.class

        (interface_declaration
            name: (identifier) @name) @definition.interface

        (enum_declaration
            name: (identifier) @name) @definition.enum
    """,
    TreeSitterLanguage.C: """
        (function_definition
            declarator: (function_declarator
                declarator: (identifier) @name)) @definition.function

        (struct_specifier
            name: (type_identifier) @name) @definition.struct

        (enum_specifier
            name: (type_identifier) @name) @definition.enum
    """,
    TreeSitterLanguage.CPP: """
        (function_definition
            declarator: (function_declarator
                declarator: (identifier) @name)) @definition.function

        (class_specifier
            name: (type_identifier) @name) @definition.class

        (struct_specifier
            name: (type_identifier) @name) @definition.struct
    """,
    TreeSitterLanguage.RUBY: """
        (method
            name: (identifier) @name) @definition.method

        (class
            name: (constant) @name) @definition.class

        (module
            name: (constant) @name) @definition.module
    """,
    TreeSitterLanguage.PHP: """
        (function_definition
            name: (name) @name) @definition.function

        (class_declaration
            name: (name) @name) @definition.class

        (method_declaration
            name: (name) @name) @definition.method
    """,
}


def get_symbol_query(language: TreeSitterLanguage) -> str | None:
    """Get the symbol extraction query for a language.

    Args:
        language: The TreeSitterLanguage

    Returns:
        Query string if available, None otherwise
    """
    return SYMBOL_QUERIES.get(language)
