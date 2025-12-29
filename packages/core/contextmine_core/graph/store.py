"""Graph store for code symbol relationships.

This module provides the core data structures for representing a code graph:
- SymbolNode: A node representing a code symbol (function, class, etc.)
- Edge: A directed relationship between two symbols
- EdgeType: Types of relationships (contains, defines, references, etc.)
- CodeGraph: The graph container with indices for efficient lookup
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EdgeType(Enum):
    """Types of edges in the code graph."""

    CONTAINS = "contains"  # File contains class, class contains method
    DEFINES = "defines"  # Definition relationship
    REFERENCES = "references"  # Usage relationship
    CALLS = "calls"  # Function call
    IMPORTS = "imports"  # Import statement
    INHERITS = "inherits"  # Class inheritance


@dataclass
class SymbolNode:
    """A node representing a code symbol.

    Symbols are identified by qualified IDs like "src/auth.py::verify_token"
    or "src/models.py::User.save" for nested symbols.
    """

    id: str
    """Qualified symbol ID: file_path::Class.method"""

    name: str
    """Simple name: method"""

    kind: str
    """Symbol kind: function, class, method, etc."""

    file_path: str
    """Path to the source file."""

    start_line: int
    """Starting line number (1-indexed)."""

    end_line: int
    """Ending line number (1-indexed)."""

    signature: str | None = None
    """Function/method signature if available."""

    parent_id: str | None = None
    """ID of the containing symbol (e.g., class for a method)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "signature": self.signature,
            "parent_id": self.parent_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SymbolNode:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            kind=data["kind"],
            file_path=data["file_path"],
            start_line=data["start_line"],
            end_line=data["end_line"],
            signature=data.get("signature"),
            parent_id=data.get("parent_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Edge:
    """A directed edge between two symbols."""

    source_id: str
    """ID of the source node."""

    target_id: str
    """ID of the target node."""

    edge_type: EdgeType
    """Type of relationship."""

    weight: float = 1.0
    """Edge weight for ranking."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata (e.g., call site location)."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Edge:
        """Create from dictionary."""
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            edge_type=EdgeType(data["edge_type"]),
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata", {}),
        )


class CodeGraph:
    """In-memory graph of code symbols and their relationships.

    Provides efficient lookup by:
    - Symbol ID (direct)
    - File path (all symbols in a file)
    - Symbol kind (all functions, classes, etc.)

    The graph is directed: edges have a source and target.
    """

    def __init__(self) -> None:
        """Initialize an empty graph."""
        self._nodes: dict[str, SymbolNode] = {}
        self._edges: list[Edge] = []

        # Indices for efficient lookup
        self._outgoing: dict[str, list[Edge]] = {}  # source_id -> edges
        self._incoming: dict[str, list[Edge]] = {}  # target_id -> edges
        self._by_file: dict[str, set[str]] = {}  # file_path -> node_ids
        self._by_kind: dict[str, set[str]] = {}  # kind -> node_ids

    def add_node(self, node: SymbolNode) -> None:
        """Add a node to the graph.

        If a node with the same ID already exists, it is replaced.

        Args:
            node: The symbol node to add
        """
        # Remove from indices if replacing
        if node.id in self._nodes:
            old_node = self._nodes[node.id]
            self._by_file.get(old_node.file_path, set()).discard(node.id)
            self._by_kind.get(old_node.kind, set()).discard(node.id)

        # Add node
        self._nodes[node.id] = node

        # Update indices
        if node.file_path not in self._by_file:
            self._by_file[node.file_path] = set()
        self._by_file[node.file_path].add(node.id)

        if node.kind not in self._by_kind:
            self._by_kind[node.kind] = set()
        self._by_kind[node.kind].add(node.id)

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph.

        Duplicate edges (same source, target, type) are allowed.

        Args:
            edge: The edge to add
        """
        self._edges.append(edge)

        # Update indices
        if edge.source_id not in self._outgoing:
            self._outgoing[edge.source_id] = []
        self._outgoing[edge.source_id].append(edge)

        if edge.target_id not in self._incoming:
            self._incoming[edge.target_id] = []
        self._incoming[edge.target_id].append(edge)

    def get_node(self, node_id: str) -> SymbolNode | None:
        """Get a node by ID.

        Args:
            node_id: The qualified symbol ID

        Returns:
            The node if found, None otherwise
        """
        return self._nodes.get(node_id)

    def has_node(self, node_id: str) -> bool:
        """Check if a node exists.

        Args:
            node_id: The qualified symbol ID

        Returns:
            True if the node exists
        """
        return node_id in self._nodes

    def get_neighbors(
        self,
        node_id: str,
        edge_types: list[EdgeType] | None = None,
        direction: str = "both",
    ) -> list[tuple[SymbolNode, Edge]]:
        """Get neighboring nodes.

        Args:
            node_id: The source node ID
            edge_types: Filter by edge types (None for all)
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of (neighbor_node, edge) tuples
        """
        results: list[tuple[SymbolNode, Edge]] = []

        if direction in ("outgoing", "both"):
            for edge in self._outgoing.get(node_id, []):
                if edge_types is None or edge.edge_type in edge_types:
                    neighbor = self._nodes.get(edge.target_id)
                    if neighbor:
                        results.append((neighbor, edge))

        if direction in ("incoming", "both"):
            for edge in self._incoming.get(node_id, []):
                if edge_types is None or edge.edge_type in edge_types:
                    neighbor = self._nodes.get(edge.source_id)
                    if neighbor:
                        results.append((neighbor, edge))

        return results

    def get_nodes_in_file(self, file_path: str) -> list[SymbolNode]:
        """Get all nodes in a file.

        Args:
            file_path: Path to the source file

        Returns:
            List of symbol nodes in the file
        """
        node_ids = self._by_file.get(file_path, set())
        return [self._nodes[nid] for nid in node_ids if nid in self._nodes]

    def get_nodes_by_kind(self, kind: str) -> list[SymbolNode]:
        """Get all nodes of a specific kind.

        Args:
            kind: Symbol kind (function, class, method, etc.)

        Returns:
            List of symbol nodes of that kind
        """
        node_ids = self._by_kind.get(kind, set())
        return [self._nodes[nid] for nid in node_ids if nid in self._nodes]

    def get_all_nodes(self) -> list[SymbolNode]:
        """Get all nodes in the graph.

        Returns:
            List of all symbol nodes
        """
        return list(self._nodes.values())

    def get_all_edges(self) -> list[Edge]:
        """Get all edges in the graph.

        Returns:
            List of all edges
        """
        return list(self._edges)

    def node_count(self) -> int:
        """Get the number of nodes."""
        return len(self._nodes)

    def edge_count(self) -> int:
        """Get the number of edges."""
        return len(self._edges)

    def to_dict(self) -> dict[str, Any]:
        """Serialize graph to dictionary.

        Returns:
            Dictionary representation of the graph
        """
        return {
            "nodes": [node.to_dict() for node in self._nodes.values()],
            "edges": [edge.to_dict() for edge in self._edges],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CodeGraph:
        """Deserialize graph from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            CodeGraph instance
        """
        graph = cls()
        for node_data in data.get("nodes", []):
            graph.add_node(SymbolNode.from_dict(node_data))
        for edge_data in data.get("edges", []):
            graph.add_edge(Edge.from_dict(edge_data))
        return graph

    def subgraph(self, node_ids: set[str]) -> CodeGraph:
        """Create a subgraph containing only the specified nodes.

        Includes edges where both source and target are in the subgraph.

        Args:
            node_ids: Set of node IDs to include

        Returns:
            New CodeGraph with only the specified nodes and their edges
        """
        sub = CodeGraph()

        # Add nodes
        for node_id in node_ids:
            node = self._nodes.get(node_id)
            if node:
                sub.add_node(node)

        # Add edges where both endpoints are in the subgraph
        for edge in self._edges:
            if edge.source_id in node_ids and edge.target_id in node_ids:
                sub.add_edge(edge)

        return sub

    def merge(self, other: CodeGraph) -> None:
        """Merge another graph into this one.

        Nodes with the same ID are replaced.
        Edges are added (may create duplicates).

        Args:
            other: The graph to merge in
        """
        for node in other._nodes.values():
            self.add_node(node)
        for edge in other._edges:
            self.add_edge(edge)


def make_symbol_id(file_path: str, name: str, parent: str | None = None) -> str:
    """Create a qualified symbol ID.

    Args:
        file_path: Path to the source file
        name: Symbol name
        parent: Parent symbol name (for nested symbols)

    Returns:
        Qualified ID like "src/auth.py::verify_token" or "src/models.py::User.save"
    """
    if parent:
        return f"{file_path}::{parent}.{name}"
    return f"{file_path}::{name}"
