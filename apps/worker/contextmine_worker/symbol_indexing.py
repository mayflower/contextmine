"""Symbol extraction and indexing service.

This module provides functions to extract code symbols from documents and store
them in the database. It's designed to be called as part of the sync flow,
after chunking and before embedding.

Symbols are extracted using tree-sitter and stored incrementally:
- When a document is created or updated, symbols are re-extracted
- When a document is deleted, symbols cascade delete automatically
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

from contextmine_core.models import Document, Symbol, SymbolKind
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# File extensions that support symbol extraction
SYMBOL_EXTRACTABLE_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
}


@dataclass
class SymbolStats:
    """Statistics from symbol extraction."""

    symbols_created: int = 0
    symbols_deleted: int = 0
    files_processed: int = 0
    files_skipped: int = 0


def _is_code_file(uri: str) -> bool:
    """Check if a document URI represents a code file we can extract symbols from."""
    # Strip query parameters before checking extension
    path = uri.split("?")[0]
    suffix = Path(path).suffix.lower()
    return suffix in SYMBOL_EXTRACTABLE_EXTENSIONS


def _map_symbol_kind(ts_kind: str) -> SymbolKind:
    """Map tree-sitter symbol kind to database model SymbolKind."""
    kind_map = {
        "function": SymbolKind.FUNCTION,
        "class": SymbolKind.CLASS,
        "method": SymbolKind.METHOD,
        "struct": SymbolKind.CLASS,  # Map struct to class
        "enum": SymbolKind.ENUM,
        "interface": SymbolKind.INTERFACE,
        "type": SymbolKind.TYPE_ALIAS,
        "trait": SymbolKind.INTERFACE,  # Map trait to interface
        "impl": SymbolKind.MODULE,  # Map impl to module
        "module": SymbolKind.MODULE,
        "variable": SymbolKind.VARIABLE,
        "unknown": SymbolKind.FUNCTION,  # Fallback
    }
    return kind_map.get(ts_kind, SymbolKind.FUNCTION)


async def extract_symbols_for_document(
    db: AsyncSession,
    document: Document,
) -> int:
    """Extract and store symbols for a single document.

    This function:
    1. Deletes any existing symbols for the document
    2. Extracts new symbols using tree-sitter
    3. Inserts the new symbols

    Args:
        db: Database session
        document: The document to extract symbols from

    Returns:
        Number of symbols created
    """
    # Skip non-code files
    if not _is_code_file(document.uri):
        return 0

    # Delete existing symbols for this document
    await db.execute(delete(Symbol).where(Symbol.document_id == document.id))

    # Extract symbols using tree-sitter
    try:
        from contextmine_core.treesitter import extract_outline
    except ImportError:
        logger.warning("Tree-sitter not available, skipping symbol extraction")
        return 0

    try:
        # Use document content (markdown is actually source code for code files)
        symbols = extract_outline(
            document.uri,
            content=document.content_markdown,
            include_children=True,
        )
    except Exception as e:
        logger.debug("Symbol extraction failed for %s: %s", document.uri, e)
        return 0

    if not symbols:
        return 0

    # Flatten symbols into a list with proper qualified names
    def flatten_symbols(
        syms: list,
        parent_qualified: str | None = None,
    ) -> list[tuple[str, str, str, int, int, str | None, str | None, dict]]:
        """Flatten nested symbols into a list with qualified names."""
        result = []
        for sym in syms:
            # Build qualified name
            qualified = f"{parent_qualified}.{sym.name}" if parent_qualified else sym.name

            result.append(
                (
                    qualified,  # qualified_name
                    sym.name,  # name
                    sym.kind.value,  # kind
                    sym.start_line,  # start_line
                    sym.end_line,  # end_line
                    sym.signature,  # signature
                    parent_qualified,  # parent_name
                    {
                        "docstring": sym.docstring,
                        "start_column": sym.start_column,
                        "end_column": sym.end_column,
                    },  # meta
                )
            )

            # Recursively process children
            if sym.children:
                result.extend(flatten_symbols(sym.children, qualified))

        return result

    flattened = flatten_symbols(symbols)

    # Deduplicate by qualified_name (keep first occurrence)
    # This handles cases like multiple functions with the same name in a file
    seen_qualified: set[str] = set()
    unique_symbols = []
    for item in flattened:
        qualified_name = item[0]
        if qualified_name not in seen_qualified:
            seen_qualified.add(qualified_name)
            unique_symbols.append(item)
        else:
            # Make unique by appending line number
            start_line = item[3]
            unique_qualified = f"{qualified_name}@L{start_line}"
            if unique_qualified not in seen_qualified:
                seen_qualified.add(unique_qualified)
                # Create new tuple with modified qualified_name
                unique_symbols.append((unique_qualified, *item[1:]))

    # Create Symbol records
    created = 0
    for (
        qualified_name,
        name,
        kind_str,
        start_line,
        end_line,
        signature,
        parent_name,
        meta,
    ) in unique_symbols:
        symbol = Symbol(
            id=uuid.uuid4(),
            document_id=document.id,
            qualified_name=qualified_name,
            name=name,
            kind=_map_symbol_kind(kind_str),
            start_line=start_line,
            end_line=end_line,
            signature=signature,
            parent_name=parent_name,
            meta=meta,
        )
        db.add(symbol)
        created += 1

    return created


async def maintain_symbols_for_document(
    db: AsyncSession,
    document_id: uuid.UUID,
) -> tuple[int, int]:
    """Maintain symbols for a document by ID.

    Fetches the document and re-extracts symbols. Used when a document's
    content has changed.

    Args:
        db: Database session
        document_id: ID of the document

    Returns:
        Tuple of (symbols_created, symbols_deleted)
    """
    # Get the document
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if not document:
        logger.warning("Document not found: %s", document_id)
        return 0, 0

    # Count existing symbols before deletion
    result = await db.execute(select(Symbol.id).where(Symbol.document_id == document_id))
    existing_count = len(result.all())

    # Extract and store new symbols
    created = await extract_symbols_for_document(db, document)

    return created, existing_count


async def index_symbols_for_source(
    db: AsyncSession,
    source_id: uuid.UUID,
    document_ids: list[uuid.UUID] | None = None,
) -> SymbolStats:
    """Index symbols for all documents in a source.

    Args:
        db: Database session
        source_id: ID of the source
        document_ids: Optional list of specific document IDs to process.
                     If None, processes all documents in the source.

    Returns:
        SymbolStats with extraction statistics
    """
    stats = SymbolStats()

    # Get documents to process
    if document_ids:
        result = await db.execute(select(Document).where(Document.id.in_(document_ids)))
    else:
        result = await db.execute(select(Document).where(Document.source_id == source_id))

    documents = result.scalars().all()

    for doc in documents:
        if not _is_code_file(doc.uri):
            stats.files_skipped += 1
            continue

        try:
            # Count existing symbols
            result = await db.execute(select(Symbol.id).where(Symbol.document_id == doc.id))
            old_count = len(result.all())

            # Extract new symbols
            new_count = await extract_symbols_for_document(db, doc)

            stats.symbols_created += new_count
            stats.symbols_deleted += old_count
            stats.files_processed += 1

        except Exception as e:
            logger.error("Failed to extract symbols from %s: %s", doc.uri, e)
            stats.files_skipped += 1

    return stats
