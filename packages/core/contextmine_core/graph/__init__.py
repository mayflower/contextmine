"""Code graph module for symbol relationship analysis.

This module provides a typed graph for representing code symbols and their
relationships, enabling multi-hop evidence selection for research queries.

Main components:
- CodeGraph: In-memory graph with efficient lookup indices
- SymbolNode: Node representing a code symbol
- Edge: Directed relationship between symbols
- EdgeType: Types of relationships (contains, calls, etc.)
- GraphBuilder: Build graphs from LSP + Tree-sitter data
- Retrieval functions: expand_graph, pack_subgraph, trace_path
"""

from contextmine_core.graph.builder import GraphBuilder, get_graph_builder
from contextmine_core.graph.retrieval import (
    PackedNode,
    PathStep,
    expand_graph,
    pack_subgraph,
    trace_path,
)
from contextmine_core.graph.store import (
    CodeGraph,
    Edge,
    EdgeType,
    SymbolNode,
    make_symbol_id,
)

__all__ = [
    "CodeGraph",
    "Edge",
    "EdgeType",
    "GraphBuilder",
    "PackedNode",
    "PathStep",
    "SymbolNode",
    "expand_graph",
    "get_graph_builder",
    "make_symbol_id",
    "pack_subgraph",
    "trace_path",
]
