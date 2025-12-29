"""Tree-sitter integration for symbol extraction.

This module provides code analysis capabilities using Tree-sitter:
- Language detection from file extensions
- Parsing source files into syntax trees
- Extracting symbol outlines (functions, classes, methods)
- Finding symbols by name or line number
"""

from contextmine_core.treesitter.languages import (
    EXTENSION_TO_LANGUAGE,
    SYMBOL_QUERIES,
    TreeSitterLanguage,
    detect_language,
    get_symbol_query,
)
from contextmine_core.treesitter.manager import (
    TreeSitterManager,
    get_treesitter_manager,
)
from contextmine_core.treesitter.outline import (
    Symbol,
    SymbolKind,
    extract_outline,
    find_enclosing_symbol,
    find_symbol_by_name,
    get_symbol_content,
)

__all__ = [
    # Languages
    "TreeSitterLanguage",
    "EXTENSION_TO_LANGUAGE",
    "SYMBOL_QUERIES",
    "detect_language",
    "get_symbol_query",
    # Manager
    "TreeSitterManager",
    "get_treesitter_manager",
    # Outline
    "Symbol",
    "SymbolKind",
    "extract_outline",
    "find_symbol_by_name",
    "find_enclosing_symbol",
    "get_symbol_content",
]
