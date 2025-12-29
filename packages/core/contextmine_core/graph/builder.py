"""Graph builder for constructing code graphs from LSP and Tree-sitter data.

This module provides the GraphBuilder class that combines:
- Tree-sitter for structure (symbols, containment)
- LSP for semantic relationships (definitions, references, calls)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from contextmine_core.graph.store import (
    CodeGraph,
    Edge,
    EdgeType,
    SymbolNode,
    make_symbol_id,
)

if TYPE_CHECKING:
    from contextmine_core.lsp.manager import LspManager
    from contextmine_core.treesitter.manager import TreeSitterManager
    from contextmine_core.treesitter.outline import Symbol

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Builds code graphs from LSP and Tree-sitter data.

    The builder can work in different modes:
    - Full mode: Both LSP and Tree-sitter available
    - Tree-sitter only: Structure without semantic relationships
    - Minimal: No LSP or Tree-sitter (returns empty graphs)

    Tree-sitter based methods are synchronous.
    LSP-based methods are async and require a running event loop.
    """

    def __init__(
        self,
        lsp_manager: LspManager | None = None,
        treesitter_manager: TreeSitterManager | None = None,
    ) -> None:
        """Initialize the builder.

        Args:
            lsp_manager: Optional LSP manager for semantic relationships
            treesitter_manager: Optional Tree-sitter manager for structure
        """
        self._lsp = lsp_manager
        self._ts = treesitter_manager

    @property
    def has_lsp(self) -> bool:
        """Check if LSP is available."""
        return self._lsp is not None

    @property
    def has_treesitter(self) -> bool:
        """Check if Tree-sitter is available."""
        return self._ts is not None and self._ts.is_available()

    def build_file_subgraph(
        self,
        file_path: str,
        content: str | None = None,
    ) -> CodeGraph:
        """Build a subgraph for a single file.

        Extracts all symbols and containment edges using Tree-sitter.

        Args:
            file_path: Path to the source file
            content: Optional file content (reads from file if None)

        Returns:
            CodeGraph with file's symbols and containment edges
        """
        graph = CodeGraph()

        if not self.has_treesitter:
            logger.debug("Tree-sitter unavailable, returning empty graph")
            return graph

        # Read content if not provided
        if content is None:
            path = Path(file_path)
            if not path.exists():
                return graph
            content = path.read_text(encoding="utf-8", errors="replace")

        try:
            from contextmine_core.treesitter import extract_outline

            symbols = extract_outline(file_path, content, include_children=True)
            self._add_symbols_to_graph(graph, symbols, file_path)
        except Exception as e:
            logger.debug("Failed to extract outline: %s", e)

        return graph

    def _add_symbols_to_graph(
        self,
        graph: CodeGraph,
        symbols: list[Symbol],
        file_path: str,
        parent_id: str | None = None,
    ) -> None:
        """Add symbols to graph with containment edges.

        Args:
            graph: The graph to add to
            symbols: List of symbols to add
            file_path: Path to the source file
            parent_id: ID of the parent symbol (for containment edges)
        """
        for symbol in symbols:
            # Create symbol node
            symbol_id = make_symbol_id(
                file_path,
                symbol.name,
                symbol.parent,
            )

            node = SymbolNode(
                id=symbol_id,
                name=symbol.name,
                kind=symbol.kind.value,
                file_path=file_path,
                start_line=symbol.start_line,
                end_line=symbol.end_line,
                signature=symbol.signature,
                parent_id=parent_id,
                metadata={
                    "docstring": symbol.docstring,
                },
            )
            graph.add_node(node)

            # Add containment edge from parent
            if parent_id:
                graph.add_edge(
                    Edge(
                        source_id=parent_id,
                        target_id=symbol_id,
                        edge_type=EdgeType.CONTAINS,
                    )
                )

            # Recursively add children
            if symbol.children:
                self._add_symbols_to_graph(
                    graph,
                    symbol.children,
                    file_path,
                    parent_id=symbol_id,
                )

    async def add_definition_edges(
        self,
        graph: CodeGraph,
        symbol_id: str,
        file_path: str,
        line: int,
        column: int = 0,
    ) -> None:
        """Add definition edges using LSP go-to-definition.

        Args:
            graph: The graph to add edges to
            symbol_id: ID of the symbol requesting definition
            file_path: Path to the source file
            line: Line number (1-indexed)
            column: Column number (0-indexed)
        """
        if not self.has_lsp or self._lsp is None:
            return

        try:
            client = await self._lsp.get_client(file_path)
            definitions = await client.get_definition(file_path, line, column)

            for definition in definitions:
                # Try to resolve the target symbol
                target_id = self.resolve_symbol_at(
                    definition.file_path,
                    definition.start_line,
                    definition.start_column,
                )
                if target_id and target_id != symbol_id:
                    graph.add_edge(
                        Edge(
                            source_id=symbol_id,
                            target_id=target_id,
                            edge_type=EdgeType.DEFINES,
                            metadata={
                                "source_line": line,
                                "target_file": definition.file_path,
                                "target_line": definition.start_line,
                            },
                        )
                    )
        except Exception as e:
            logger.debug("Failed to get definition: %s", e)

    async def add_reference_edges(
        self,
        graph: CodeGraph,
        symbol_id: str,
        file_path: str,
        line: int,
        column: int = 0,
    ) -> None:
        """Add reference edges using LSP find-references.

        Args:
            graph: The graph to add edges to
            symbol_id: ID of the symbol to find references for
            file_path: Path to the source file
            line: Line number (1-indexed)
            column: Column number (0-indexed)
        """
        if not self.has_lsp or self._lsp is None:
            return

        try:
            client = await self._lsp.get_client(file_path)
            references = await client.get_references(file_path, line, column)

            for ref in references:
                # Try to resolve what symbol contains this reference
                ref_symbol_id = self.resolve_symbol_at(
                    ref.file_path,
                    ref.start_line,
                    ref.start_column,
                )
                if ref_symbol_id and ref_symbol_id != symbol_id:
                    graph.add_edge(
                        Edge(
                            source_id=ref_symbol_id,
                            target_id=symbol_id,
                            edge_type=EdgeType.REFERENCES,
                            metadata={
                                "ref_file": ref.file_path,
                                "ref_line": ref.start_line,
                            },
                        )
                    )
        except Exception as e:
            logger.debug("Failed to get references: %s", e)

    def resolve_symbol_at(
        self,
        file_path: str,
        line: int,
        column: int = 0,
    ) -> str | None:
        """Resolve symbol ID at a location.

        Uses Tree-sitter to find the enclosing symbol.

        Args:
            file_path: Path to the source file
            line: Line number (1-indexed)
            column: Column number (0-indexed)

        Returns:
            Symbol ID if found, None otherwise
        """
        if not self.has_treesitter:
            return None

        try:
            from contextmine_core.treesitter import find_enclosing_symbol

            symbol = find_enclosing_symbol(file_path, line)
            if symbol:
                return make_symbol_id(file_path, symbol.name, symbol.parent)
        except Exception as e:
            logger.debug("Failed to resolve symbol: %s", e)

        return None

    def build_multi_file_graph(
        self,
        file_paths: list[str],
    ) -> CodeGraph:
        """Build a graph spanning multiple files (Tree-sitter only).

        For LSP-based edges, use async methods after building.

        Args:
            file_paths: List of file paths to include

        Returns:
            CodeGraph with all files' symbols
        """
        graph = CodeGraph()

        for file_path in file_paths:
            file_graph = self.build_file_subgraph(file_path)
            graph.merge(file_graph)

        return graph


def get_graph_builder() -> GraphBuilder:
    """Get a GraphBuilder with available backends.

    Tries to initialize with both LSP and Tree-sitter,
    falling back gracefully if unavailable.

    Returns:
        GraphBuilder instance
    """
    lsp_manager = None
    ts_manager = None

    # Try to get LSP manager
    try:
        from contextmine_core.lsp.manager import get_lsp_manager

        lsp_manager = get_lsp_manager()
    except ImportError:
        logger.debug("LSP manager not available")

    # Try to get Tree-sitter manager
    try:
        from contextmine_core.treesitter.manager import get_treesitter_manager

        ts_manager = get_treesitter_manager()
    except ImportError:
        logger.debug("Tree-sitter manager not available")

    return GraphBuilder(lsp_manager, ts_manager)
