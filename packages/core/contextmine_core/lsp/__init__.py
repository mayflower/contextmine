"""LSP (Language Server Protocol) integration for code intelligence.

This module provides LSP-based code analysis capabilities for the research agent,
including go-to-definition, find-references, hover info, and diagnostics.
"""

from contextmine_core.lsp.client import (
    Diagnostic,
    Location,
    LspClient,
    MockLspClient,
    SymbolInfo,
)
from contextmine_core.lsp.exceptions import (
    LspError,
    LspNotAvailableError,
    LspServerError,
    LspTimeoutError,
)
from contextmine_core.lsp.languages import (
    EXTENSION_TO_LANGUAGE,
    PROJECT_ROOT_MARKERS,
    SupportedLanguage,
    detect_language,
    find_project_root,
)
from contextmine_core.lsp.manager import LspManager, get_lsp_manager

__all__ = [
    # Client
    "Diagnostic",
    "Location",
    "LspClient",
    "MockLspClient",
    "SymbolInfo",
    # Exceptions
    "LspError",
    "LspNotAvailableError",
    "LspServerError",
    "LspTimeoutError",
    # Languages
    "EXTENSION_TO_LANGUAGE",
    "PROJECT_ROOT_MARKERS",
    "SupportedLanguage",
    "detect_language",
    "find_project_root",
    # Manager
    "LspManager",
    "get_lsp_manager",
]
