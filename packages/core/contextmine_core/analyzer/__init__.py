"""Analyzer module for extracting derived knowledge from source files.

This module provides extractors for:
- ERM (Entity-Relationship Model) from Alembic migrations
- System surfaces from OpenAPI, GraphQL, Protobuf specs (Step 4)
- Business rule candidates from code AST (Step 5)
"""

from contextmine_core.analyzer.extractors import (
    ERMExtractor,
    build_erm_graph,
    extract_from_alembic,
    generate_mermaid_erd,
)

__all__ = [
    "extract_from_alembic",
    "ERMExtractor",
    "build_erm_graph",
    "generate_mermaid_erd",
]
