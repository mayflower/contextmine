"""Graph-based actions for the research agent.

These actions use the code graph to provide multi-hop evidence selection:
- graph_expand: Expand from seed symbols following relationships
- graph_pack: Select minimal evidence set from expanded graph
- graph_trace: Find paths between symbols (impact analysis)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from contextmine_core.graph import (
    CodeGraph,
    EdgeType,
    GraphBuilder,
    expand_graph,
    get_graph_builder,
    pack_subgraph,
    trace_path,
)
from contextmine_core.research.actions.registry import Action, ActionResult
from contextmine_core.research.run import Evidence

if TYPE_CHECKING:
    from contextmine_core.graph import SymbolNode
    from contextmine_core.research.run import ResearchRun

logger = logging.getLogger(__name__)


def _node_to_dict(node: SymbolNode) -> dict[str, Any]:
    """Convert SymbolNode to dictionary for output."""
    return {
        "id": node.id,
        "name": node.name,
        "kind": node.kind,
        "file_path": node.file_path,
        "start_line": node.start_line,
        "end_line": node.end_line,
        "signature": node.signature,
        "parent_id": node.parent_id,
    }


def _read_file_content(file_path: str, start_line: int, end_line: int) -> str:
    """Read content from a file between lines."""
    try:
        path = Path(file_path)
        if not path.exists():
            return f"[File not found: {file_path}]"
        content = path.read_text(encoding="utf-8", errors="replace")
        lines = content.split("\n")
        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), end_line)
        return "\n".join(lines[start_idx:end_idx])
    except Exception as e:
        return f"[Error reading file: {e}]"


def _parse_edge_types(edge_type_strs: list[str] | None) -> list[EdgeType] | None:
    """Parse edge type strings to EdgeType enums."""
    if not edge_type_strs:
        return None

    result = []
    for s in edge_type_strs:
        try:
            result.append(EdgeType(s.lower()))
        except ValueError:
            logger.warning("Unknown edge type: %s", s)
            continue
    return result if result else None


class GraphExpandAction(Action):
    """Expand graph from seed symbols following relationships."""

    def __init__(self, builder: GraphBuilder | None = None):
        """Initialize with optional graph builder.

        Args:
            builder: Graph builder (creates default if None)
        """
        self._builder = builder

    def _get_builder(self) -> GraphBuilder:
        """Get or create graph builder."""
        if self._builder is None:
            self._builder = get_graph_builder()
        return self._builder

    @property
    def name(self) -> str:
        return "graph_expand"

    @property
    def description(self) -> str:
        return (
            "Expand from seed symbols following relationship types to find related code. "
            "Useful for understanding what code is connected to a symbol."
        )

    async def execute(
        self,
        run: ResearchRun,
        params: dict[str, Any],
    ) -> ActionResult:
        """Execute graph_expand action.

        Args:
            run: Current research run
            params: Must contain 'seeds', optionally 'edge_types', 'depth', 'limit'

        Returns:
            ActionResult with expanded subgraph
        """
        seeds = params.get("seeds", [])
        edge_type_strs = params.get("edge_types")
        depth = params.get("depth", 2)
        limit = params.get("limit", 50)

        if not seeds:
            return ActionResult(
                success=False,
                output_summary="No seed symbols provided",
                error="seeds is required",
            )

        try:
            builder = self._get_builder()

            if not builder.has_treesitter:
                return ActionResult(
                    success=False,
                    output_summary="Graph expansion requires Tree-sitter",
                    error="Tree-sitter not available",
                )

            # Extract file paths from seed IDs (format: file_path::symbol_name)
            file_paths = set()
            for seed in seeds:
                if "::" in seed:
                    file_path = seed.split("::")[0]
                    file_paths.add(file_path)

            if not file_paths:
                return ActionResult(
                    success=False,
                    output_summary="Could not extract file paths from seeds",
                    error="Invalid seed format. Expected: file_path::symbol_name",
                )

            # Build graph from files
            graph = builder.build_multi_file_graph(list(file_paths))

            if graph.node_count() == 0:
                return ActionResult(
                    success=True,
                    output_summary="No symbols found in seed files",
                    data={
                        "nodes": [],
                        "edges": [],
                        "seeds_found": 0,
                        "total_expanded": 0,
                    },
                )

            # Find which seeds actually exist in the graph
            valid_seeds = [s for s in seeds if graph.has_node(s)]

            if not valid_seeds:
                # Try partial matching
                all_node_ids = {n.id for n in graph.get_all_nodes()}
                valid_seeds = []
                for seed in seeds:
                    for node_id in all_node_ids:
                        if seed in node_id or node_id.endswith(f"::{seed}"):
                            valid_seeds.append(node_id)
                            break

            if not valid_seeds:
                return ActionResult(
                    success=True,
                    output_summary="No matching symbols found for seeds in graph",
                    data={
                        "nodes": [_node_to_dict(n) for n in graph.get_all_nodes()[:10]],
                        "edges": [],
                        "seeds_found": 0,
                        "total_expanded": graph.node_count(),
                    },
                )

            # Parse edge types
            edge_types = _parse_edge_types(edge_type_strs)

            # Expand from seeds
            expanded = expand_graph(graph, valid_seeds, edge_types, depth=depth, limit=limit)

            # Create evidence for key nodes
            evidence_items = []
            for node in list(expanded.get_all_nodes())[:10]:
                content = _read_file_content(node.file_path, node.start_line, node.end_line)
                evidence_id = (
                    f"ev-{run.run_id[:8]}-{len(run.evidence) + len(evidence_items) + 1:03d}"
                )
                evidence_items.append(
                    Evidence(
                        id=evidence_id,
                        file_path=node.file_path,
                        start_line=node.start_line,
                        end_line=node.end_line,
                        content=content[:2000],
                        reason=f"Expanded from graph: {node.kind} '{node.name}'",
                        provenance="graph",
                        symbol_id=node.id,
                        symbol_kind=node.kind,
                    )
                )

            summary = (
                f"Expanded from {len(valid_seeds)} seed(s) to {expanded.node_count()} "
                f"nodes with {expanded.edge_count()} edges"
            )

            return ActionResult(
                success=True,
                output_summary=summary,
                evidence=evidence_items,
                data={
                    "nodes": [_node_to_dict(n) for n in expanded.get_all_nodes()],
                    "edges": [
                        {
                            "source_id": e.source_id,
                            "target_id": e.target_id,
                            "edge_type": e.edge_type.value,
                            "weight": e.weight,
                        }
                        for e in expanded.get_all_edges()
                    ],
                    "seeds_found": len(valid_seeds),
                    "total_expanded": expanded.node_count(),
                },
            )

        except Exception as e:
            logger.warning("Graph expansion failed: %s", e)
            return ActionResult(
                success=False,
                output_summary=f"Graph expansion failed: {e}",
                error=str(e),
            )


class GraphPackAction(Action):
    """Select minimal evidence set from graph."""

    def __init__(self, builder: GraphBuilder | None = None):
        """Initialize with optional graph builder.

        Args:
            builder: Graph builder (creates default if None)
        """
        self._builder = builder

    def _get_builder(self) -> GraphBuilder:
        """Get or create graph builder."""
        if self._builder is None:
            self._builder = get_graph_builder()
        return self._builder

    @property
    def name(self) -> str:
        return "graph_pack"

    @property
    def description(self) -> str:
        return (
            "Select the most relevant nodes from an expanded graph. "
            "Useful for reducing evidence to the most important symbols."
        )

    async def execute(
        self,
        run: ResearchRun,
        params: dict[str, Any],
    ) -> ActionResult:
        """Execute graph_pack action.

        Args:
            run: Current research run
            params: Optionally 'node_ids', 'target_count'

        Returns:
            ActionResult with selected nodes
        """
        node_ids = params.get("node_ids")
        target_count = params.get("target_count", 10)

        try:
            builder = self._get_builder()

            if not builder.has_treesitter:
                return ActionResult(
                    success=False,
                    output_summary="Graph packing requires Tree-sitter",
                    error="Tree-sitter not available",
                )

            # If specific node_ids provided, build graph from those files
            if node_ids:
                file_paths = set()
                for node_id in node_ids:
                    if "::" in node_id:
                        file_path = node_id.split("::")[0]
                        file_paths.add(file_path)

                if file_paths:
                    graph = builder.build_multi_file_graph(list(file_paths))
                    node_id_set = set(node_ids)
                else:
                    graph = CodeGraph()
                    node_id_set = None
            else:
                # Use existing evidence to determine files
                file_paths = set()
                for ev in run.evidence:
                    file_paths.add(ev.file_path)

                if file_paths:
                    graph = builder.build_multi_file_graph(list(file_paths))
                else:
                    graph = CodeGraph()
                node_id_set = None

            if graph.node_count() == 0:
                return ActionResult(
                    success=True,
                    output_summary="No nodes to pack",
                    data={"selected": [], "total_considered": 0},
                )

            # Pack the graph
            packed = pack_subgraph(graph, node_id_set, target_count)

            # Create evidence for selected nodes
            evidence_items = []
            selected_data = []
            for packed_node in packed:
                node = packed_node.node
                content = _read_file_content(node.file_path, node.start_line, node.end_line)
                evidence_id = (
                    f"ev-{run.run_id[:8]}-{len(run.evidence) + len(evidence_items) + 1:03d}"
                )
                evidence_items.append(
                    Evidence(
                        id=evidence_id,
                        file_path=node.file_path,
                        start_line=node.start_line,
                        end_line=node.end_line,
                        content=content[:2000],
                        reason=packed_node.reason,
                        provenance="graph",
                        symbol_id=node.id,
                        symbol_kind=node.kind,
                    )
                )
                selected_data.append(
                    {
                        "node": _node_to_dict(node),
                        "reason": packed_node.reason,
                        "score": packed_node.score,
                    }
                )

            summary = (
                f"Selected {len(packed)} most relevant nodes from {graph.node_count()} candidates"
            )

            return ActionResult(
                success=True,
                output_summary=summary,
                evidence=evidence_items,
                data={
                    "selected": selected_data,
                    "total_considered": graph.node_count(),
                },
            )

        except Exception as e:
            logger.warning("Graph packing failed: %s", e)
            return ActionResult(
                success=False,
                output_summary=f"Graph packing failed: {e}",
                error=str(e),
            )


class GraphTraceAction(Action):
    """Find paths between two symbols."""

    def __init__(self, builder: GraphBuilder | None = None):
        """Initialize with optional graph builder.

        Args:
            builder: Graph builder (creates default if None)
        """
        self._builder = builder

    def _get_builder(self) -> GraphBuilder:
        """Get or create graph builder."""
        if self._builder is None:
            self._builder = get_graph_builder()
        return self._builder

    @property
    def name(self) -> str:
        return "graph_trace"

    @property
    def description(self) -> str:
        return (
            "Find paths between two symbols to understand how they are related. "
            "Useful for impact analysis and understanding code dependencies."
        )

    async def execute(
        self,
        run: ResearchRun,
        params: dict[str, Any],
    ) -> ActionResult:
        """Execute graph_trace action.

        Args:
            run: Current research run
            params: Must contain 'from_symbol', 'to_symbol', optionally 'edge_types'

        Returns:
            ActionResult with paths between symbols
        """
        from_symbol = params.get("from_symbol", "")
        to_symbol = params.get("to_symbol", "")
        edge_type_strs = params.get("edge_types")

        if not from_symbol:
            return ActionResult(
                success=False,
                output_summary="No from_symbol provided",
                error="from_symbol is required",
            )

        if not to_symbol:
            return ActionResult(
                success=False,
                output_summary="No to_symbol provided",
                error="to_symbol is required",
            )

        try:
            builder = self._get_builder()

            if not builder.has_treesitter:
                return ActionResult(
                    success=False,
                    output_summary="Graph tracing requires Tree-sitter",
                    error="Tree-sitter not available",
                )

            # Extract file paths from symbol IDs
            file_paths = set()
            for symbol_id in [from_symbol, to_symbol]:
                if "::" in symbol_id:
                    file_path = symbol_id.split("::")[0]
                    file_paths.add(file_path)

            if len(file_paths) < 1:
                return ActionResult(
                    success=False,
                    output_summary="Could not extract file paths from symbols",
                    error="Invalid symbol format. Expected: file_path::symbol_name",
                )

            # Build graph from files
            graph = builder.build_multi_file_graph(list(file_paths))

            if graph.node_count() == 0:
                return ActionResult(
                    success=True,
                    output_summary="No symbols found in source files",
                    data={"paths": [], "found": False, "shortest_length": None},
                )

            # Try to find matching node IDs
            from_id = from_symbol
            to_id = to_symbol

            # If not found directly, try partial matching
            if not graph.has_node(from_id):
                for node in graph.get_all_nodes():
                    if from_symbol in node.id or node.id.endswith(f"::{from_symbol}"):
                        from_id = node.id
                        break

            if not graph.has_node(to_id):
                for node in graph.get_all_nodes():
                    if to_symbol in node.id or node.id.endswith(f"::{to_symbol}"):
                        to_id = node.id
                        break

            if not graph.has_node(from_id):
                return ActionResult(
                    success=True,
                    output_summary=f"Symbol '{from_symbol}' not found in graph",
                    data={"paths": [], "found": False, "shortest_length": None},
                )

            if not graph.has_node(to_id):
                return ActionResult(
                    success=True,
                    output_summary=f"Symbol '{to_symbol}' not found in graph",
                    data={"paths": [], "found": False, "shortest_length": None},
                )

            # Parse edge types
            edge_types = _parse_edge_types(edge_type_strs)

            # Find paths
            paths = trace_path(graph, from_id, to_id, edge_types)

            if not paths:
                return ActionResult(
                    success=True,
                    output_summary=f"No path found between '{from_symbol}' and '{to_symbol}'",
                    data={"paths": [], "found": False, "shortest_length": None},
                )

            # Convert paths to output format and create evidence
            evidence_items = []
            paths_data = []

            for path in paths[:5]:  # Limit to 5 paths
                path_steps = []
                for step in path:
                    node = step.node
                    path_steps.append(
                        {
                            "node": _node_to_dict(node),
                            "edge_type": step.edge_type.value if step.edge_type else None,
                            "direction": step.direction,
                        }
                    )

                    # Create evidence for each node in the path
                    content = _read_file_content(node.file_path, node.start_line, node.end_line)
                    evidence_id = (
                        f"ev-{run.run_id[:8]}-{len(run.evidence) + len(evidence_items) + 1:03d}"
                    )
                    evidence_items.append(
                        Evidence(
                            id=evidence_id,
                            file_path=node.file_path,
                            start_line=node.start_line,
                            end_line=node.end_line,
                            content=content[:2000],
                            reason=f"Path step: {node.kind} '{node.name}'",
                            provenance="graph",
                            symbol_id=node.id,
                            symbol_kind=node.kind,
                        )
                    )

                paths_data.append(path_steps)

            shortest = min(len(p) for p in paths)
            summary = f"Found {len(paths)} path(s) between '{from_symbol}' and '{to_symbol}', shortest has {shortest} step(s)"

            return ActionResult(
                success=True,
                output_summary=summary,
                evidence=evidence_items,
                data={
                    "paths": paths_data,
                    "found": True,
                    "shortest_length": shortest,
                },
            )

        except Exception as e:
            logger.warning("Graph tracing failed: %s", e)
            return ActionResult(
                success=False,
                output_summary=f"Graph tracing failed: {e}",
                error=str(e),
            )


# =============================================================================
# MOCK ACTIONS FOR TESTING
# =============================================================================


class MockGraphExpandAction(Action):
    """Mock graph expand action for testing."""

    def __init__(
        self,
        mock_nodes: list[dict[str, Any]] | None = None,
        mock_edges: list[dict[str, Any]] | None = None,
    ):
        """Initialize with mock data.

        Args:
            mock_nodes: List of node dicts to return
            mock_edges: List of edge dicts to return
        """
        self._mock_nodes = mock_nodes or []
        self._mock_edges = mock_edges or []

    @property
    def name(self) -> str:
        return "graph_expand"

    @property
    def description(self) -> str:
        return "Mock graph expand for testing."

    def set_data(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]] | None = None,
    ) -> None:
        """Set mock data."""
        self._mock_nodes = nodes
        self._mock_edges = edges or []

    async def execute(
        self,
        run: ResearchRun,
        params: dict[str, Any],
    ) -> ActionResult:
        """Return mock expanded graph."""
        seeds = params.get("seeds", [])

        if not seeds:
            return ActionResult(
                success=False,
                output_summary="No seed symbols provided",
                error="seeds is required",
            )

        return ActionResult(
            success=True,
            output_summary=f"Expanded to {len(self._mock_nodes)} nodes (mock)",
            data={
                "nodes": self._mock_nodes,
                "edges": self._mock_edges,
                "seeds_found": len(seeds),
                "total_expanded": len(self._mock_nodes),
            },
        )


class MockGraphPackAction(Action):
    """Mock graph pack action for testing."""

    def __init__(
        self,
        mock_selected: list[dict[str, Any]] | None = None,
    ):
        """Initialize with mock data.

        Args:
            mock_selected: List of packed node dicts to return
        """
        self._mock_selected = mock_selected or []

    @property
    def name(self) -> str:
        return "graph_pack"

    @property
    def description(self) -> str:
        return "Mock graph pack for testing."

    def set_selected(self, selected: list[dict[str, Any]]) -> None:
        """Set mock selected nodes."""
        self._mock_selected = selected

    async def execute(
        self,
        run: ResearchRun,
        params: dict[str, Any],
    ) -> ActionResult:
        """Return mock packed nodes."""
        return ActionResult(
            success=True,
            output_summary=f"Selected {len(self._mock_selected)} nodes (mock)",
            data={
                "selected": self._mock_selected,
                "total_considered": len(self._mock_selected) * 2,
            },
        )


class MockGraphTraceAction(Action):
    """Mock graph trace action for testing."""

    def __init__(
        self,
        mock_paths: list[list[dict[str, Any]]] | None = None,
    ):
        """Initialize with mock data.

        Args:
            mock_paths: List of paths to return
        """
        self._mock_paths = mock_paths or []

    @property
    def name(self) -> str:
        return "graph_trace"

    @property
    def description(self) -> str:
        return "Mock graph trace for testing."

    def set_paths(self, paths: list[list[dict[str, Any]]]) -> None:
        """Set mock paths."""
        self._mock_paths = paths

    async def execute(
        self,
        run: ResearchRun,
        params: dict[str, Any],
    ) -> ActionResult:
        """Return mock paths."""
        from_symbol = params.get("from_symbol", "")
        to_symbol = params.get("to_symbol", "")

        if not from_symbol:
            return ActionResult(
                success=False,
                output_summary="No from_symbol provided",
                error="from_symbol is required",
            )

        if not to_symbol:
            return ActionResult(
                success=False,
                output_summary="No to_symbol provided",
                error="to_symbol is required",
            )

        if not self._mock_paths:
            return ActionResult(
                success=True,
                output_summary="No path found (mock)",
                data={"paths": [], "found": False, "shortest_length": None},
            )

        shortest = min(len(p) for p in self._mock_paths)
        return ActionResult(
            success=True,
            output_summary=f"Found {len(self._mock_paths)} path(s) (mock)",
            data={
                "paths": self._mock_paths,
                "found": True,
                "shortest_length": shortest,
            },
        )
