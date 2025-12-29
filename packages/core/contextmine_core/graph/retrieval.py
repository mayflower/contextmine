"""Graph traversal and retrieval algorithms.

This module provides algorithms for:
- expand_graph: BFS expansion from seed nodes
- pack_subgraph: Select minimal evidence set
- trace_path: Find paths between symbols
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from contextmine_core.graph.store import CodeGraph, EdgeType, SymbolNode

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class PackedNode:
    """A node selected for evidence with reason."""

    node: SymbolNode
    reason: str
    score: float


@dataclass
class PathStep:
    """A step in a path between symbols."""

    node: SymbolNode
    edge_type: EdgeType | None  # None for the starting node
    direction: str  # "forward" or "backward"


def expand_graph(
    graph: CodeGraph,
    seeds: list[str],
    edge_types: list[EdgeType] | None = None,
    depth: int = 2,
    limit: int = 100,
) -> CodeGraph:
    """Expand from seed nodes following edges.

    Performs BFS from seeds, collecting all reachable nodes within
    the specified depth and limit.

    Args:
        graph: The graph to traverse
        seeds: Starting node IDs
        edge_types: Edge types to follow (None for all)
        depth: Maximum traversal depth
        limit: Maximum nodes to collect

    Returns:
        Subgraph containing expanded nodes and their edges
    """
    visited: set[str] = set()
    frontier: deque[tuple[str, int]] = deque()  # (node_id, current_depth)

    # Initialize frontier with seeds
    for seed in seeds:
        if graph.has_node(seed):
            frontier.append((seed, 0))

    # BFS expansion
    while frontier and len(visited) < limit:
        node_id, current_depth = frontier.popleft()

        if node_id in visited:
            continue

        visited.add(node_id)

        # Don't expand beyond max depth
        if current_depth >= depth:
            continue

        # Get neighbors following specified edge types
        neighbors = graph.get_neighbors(node_id, edge_types, direction="both")
        for neighbor, _edge in neighbors:
            if neighbor.id not in visited:
                frontier.append((neighbor.id, current_depth + 1))

    # Return subgraph with collected nodes
    return graph.subgraph(visited)


def pack_subgraph(
    graph: CodeGraph,
    node_ids: set[str] | None = None,
    target_count: int = 10,
) -> list[PackedNode]:
    """Select minimal evidence set from graph.

    Scores nodes by connectivity and importance, selecting the most
    relevant ones up to target_count.

    Args:
        graph: The graph to select from
        node_ids: Specific nodes to consider (None for all)
        target_count: Maximum nodes to select

    Returns:
        List of PackedNode with reasons, ordered by relevance
    """
    candidates = node_ids if node_ids else set(n.id for n in graph.get_all_nodes())

    scored: list[PackedNode] = []

    for node_id in candidates:
        node = graph.get_node(node_id)
        if not node:
            continue

        # Calculate score based on connectivity and kind
        score = _compute_node_score(graph, node)

        # Generate reason based on what makes this node important
        reason = _generate_reason(graph, node)

        scored.append(PackedNode(node=node, reason=reason, score=score))

    # Sort by score descending
    scored.sort(key=lambda x: x.score, reverse=True)

    # Return top N
    return scored[:target_count]


def _compute_node_score(graph: CodeGraph, node: SymbolNode) -> float:
    """Compute importance score for a node.

    Factors:
    - Number of incoming edges (more references = more important)
    - Number of outgoing edges (more dependencies = core functionality)
    - Symbol kind (classes > functions > methods)
    - Containment (top-level symbols preferred)

    Args:
        graph: The graph containing the node
        node: The node to score

    Returns:
        Importance score (higher is more important)
    """
    score = 0.0

    # Connectivity score
    incoming = len(graph.get_neighbors(node.id, direction="incoming"))
    outgoing = len(graph.get_neighbors(node.id, direction="outgoing"))
    score += incoming * 2.0  # Incoming weighted more
    score += outgoing * 1.0

    # Kind-based score
    kind_weights = {
        "class": 5.0,
        "struct": 5.0,
        "interface": 4.0,
        "trait": 4.0,
        "function": 3.0,
        "method": 2.0,
        "enum": 3.0,
        "type": 2.0,
        "impl": 2.0,
        "module": 1.0,
        "variable": 0.5,
    }
    score += kind_weights.get(node.kind, 1.0)

    # Top-level bonus (no parent)
    if node.parent_id is None:
        score += 2.0

    # Size bonus (larger symbols often more important)
    lines = node.end_line - node.start_line + 1
    if lines > 20:
        score += 1.0
    if lines > 50:
        score += 1.0

    return score


def _generate_reason(graph: CodeGraph, node: SymbolNode) -> str:
    """Generate a human-readable reason for including a node.

    Args:
        graph: The graph containing the node
        node: The node to explain

    Returns:
        Explanation string
    """
    parts = []

    # Describe the symbol
    parts.append(f"{node.kind.capitalize()} '{node.name}'")

    # Add connectivity info
    incoming = graph.get_neighbors(node.id, direction="incoming")
    outgoing = graph.get_neighbors(node.id, direction="outgoing")

    if incoming:
        # Count references by type
        ref_count = sum(1 for _, e in incoming if e.edge_type == EdgeType.REFERENCES)
        if ref_count > 0:
            parts.append(f"referenced {ref_count} time(s)")

    if outgoing:
        # Count what it contains or calls
        contains_count = sum(1 for _, e in outgoing if e.edge_type == EdgeType.CONTAINS)
        if contains_count > 0:
            parts.append(f"contains {contains_count} member(s)")

    return ", ".join(parts)


def trace_path(
    graph: CodeGraph,
    from_id: str,
    to_id: str,
    edge_types: list[EdgeType] | None = None,
    max_depth: int = 5,
) -> list[list[PathStep]]:
    """Find paths between two symbols.

    Uses bidirectional BFS to find shortest paths.

    Args:
        graph: The graph to search
        from_id: Starting node ID
        to_id: Target node ID
        edge_types: Edge types to follow (None for all)
        max_depth: Maximum path length

    Returns:
        List of paths, each a list of PathStep objects
    """
    if not graph.has_node(from_id) or not graph.has_node(to_id):
        return []

    if from_id == to_id:
        node = graph.get_node(from_id)
        if node:
            return [[PathStep(node=node, edge_type=None, direction="forward")]]
        return []

    # Bidirectional BFS
    forward_visited: dict[str, list[PathStep]] = {}
    backward_visited: dict[str, list[PathStep]] = {}

    from_node = graph.get_node(from_id)
    to_node = graph.get_node(to_id)

    if not from_node or not to_node:
        return []

    forward_visited[from_id] = [PathStep(node=from_node, edge_type=None, direction="forward")]
    backward_visited[to_id] = [PathStep(node=to_node, edge_type=None, direction="backward")]

    forward_frontier: deque[str] = deque([from_id])
    backward_frontier: deque[str] = deque([to_id])

    paths: list[list[PathStep]] = []
    current_depth = 0

    while (forward_frontier or backward_frontier) and current_depth < max_depth:
        # Expand forward
        if forward_frontier:
            next_frontier: deque[str] = deque()
            while forward_frontier:
                node_id = forward_frontier.popleft()
                path_so_far = forward_visited[node_id]

                # Get outgoing neighbors
                neighbors = graph.get_neighbors(node_id, edge_types, direction="outgoing")
                for neighbor, edge in neighbors:
                    if neighbor.id in forward_visited:
                        continue

                    new_path = path_so_far + [
                        PathStep(node=neighbor, edge_type=edge.edge_type, direction="forward")
                    ]
                    forward_visited[neighbor.id] = new_path
                    next_frontier.append(neighbor.id)

                    # Check for meeting point
                    if neighbor.id in backward_visited:
                        combined = _combine_paths(new_path, backward_visited[neighbor.id])
                        paths.append(combined)

            forward_frontier = next_frontier

        # Expand backward
        if backward_frontier:
            next_frontier = deque()
            while backward_frontier:
                node_id = backward_frontier.popleft()
                path_so_far = backward_visited[node_id]

                # Get incoming neighbors (traverse edges in reverse)
                neighbors = graph.get_neighbors(node_id, edge_types, direction="incoming")
                for neighbor, edge in neighbors:
                    if neighbor.id in backward_visited:
                        continue

                    new_path = [
                        PathStep(node=neighbor, edge_type=edge.edge_type, direction="backward")
                    ] + path_so_far
                    backward_visited[neighbor.id] = new_path
                    next_frontier.append(neighbor.id)

                    # Check for meeting point
                    if neighbor.id in forward_visited:
                        combined = _combine_paths(forward_visited[neighbor.id], new_path)
                        paths.append(combined)

            backward_frontier = next_frontier

        current_depth += 1

    return paths


def _combine_paths(
    forward_path: list[PathStep],
    backward_path: list[PathStep],
) -> list[PathStep]:
    """Combine forward and backward paths at meeting point.

    Args:
        forward_path: Path from start to meeting point
        backward_path: Path from meeting point to end

    Returns:
        Combined path from start to end
    """
    # The meeting point is the last node in forward_path
    # and the first node in backward_path
    # We skip the duplicate

    if not forward_path or not backward_path:
        return forward_path + backward_path

    # Skip the first node of backward_path (it's the meeting point)
    return forward_path + backward_path[1:]
