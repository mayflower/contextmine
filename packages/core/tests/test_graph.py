"""Tests for the code graph module."""

from pathlib import Path

import pytest
from contextmine_core.graph import (
    CodeGraph,
    Edge,
    EdgeType,
    GraphBuilder,
    PackedNode,
    SymbolNode,
    expand_graph,
    make_symbol_id,
    pack_subgraph,
    trace_path,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_graph() -> CodeGraph:
    """Create a sample graph for testing."""
    graph = CodeGraph()

    # Add nodes
    nodes = [
        SymbolNode(
            id="src/main.py::main",
            name="main",
            kind="function",
            file_path="src/main.py",
            start_line=1,
            end_line=10,
            signature="def main():",
        ),
        SymbolNode(
            id="src/main.py::helper",
            name="helper",
            kind="function",
            file_path="src/main.py",
            start_line=12,
            end_line=20,
            signature="def helper(x):",
        ),
        SymbolNode(
            id="src/utils.py::Utils",
            name="Utils",
            kind="class",
            file_path="src/utils.py",
            start_line=1,
            end_line=50,
            signature="class Utils:",
        ),
        SymbolNode(
            id="src/utils.py::Utils.process",
            name="process",
            kind="method",
            file_path="src/utils.py",
            start_line=10,
            end_line=30,
            signature="def process(self, data):",
            parent_id="src/utils.py::Utils",
        ),
    ]

    for node in nodes:
        graph.add_node(node)

    # Add edges
    edges = [
        Edge(
            source_id="src/utils.py::Utils",
            target_id="src/utils.py::Utils.process",
            edge_type=EdgeType.CONTAINS,
        ),
        Edge(
            source_id="src/main.py::main",
            target_id="src/main.py::helper",
            edge_type=EdgeType.CALLS,
        ),
        Edge(
            source_id="src/main.py::main",
            target_id="src/utils.py::Utils.process",
            edge_type=EdgeType.REFERENCES,
        ),
    ]

    for edge in edges:
        graph.add_edge(edge)

    return graph


# =============================================================================
# STORE TESTS
# =============================================================================


class TestSymbolNode:
    """Tests for SymbolNode dataclass."""

    def test_create_node(self) -> None:
        """Test creating a symbol node."""
        node = SymbolNode(
            id="test.py::foo",
            name="foo",
            kind="function",
            file_path="test.py",
            start_line=1,
            end_line=10,
        )
        assert node.id == "test.py::foo"
        assert node.name == "foo"
        assert node.kind == "function"

    def test_node_to_dict(self) -> None:
        """Test converting node to dictionary."""
        node = SymbolNode(
            id="test.py::foo",
            name="foo",
            kind="function",
            file_path="test.py",
            start_line=1,
            end_line=10,
            signature="def foo():",
        )
        d = node.to_dict()
        assert d["id"] == "test.py::foo"
        assert d["name"] == "foo"
        assert d["signature"] == "def foo():"

    def test_node_from_dict(self) -> None:
        """Test creating node from dictionary."""
        d = {
            "id": "test.py::bar",
            "name": "bar",
            "kind": "class",
            "file_path": "test.py",
            "start_line": 5,
            "end_line": 20,
        }
        node = SymbolNode.from_dict(d)
        assert node.id == "test.py::bar"
        assert node.kind == "class"


class TestEdge:
    """Tests for Edge dataclass."""

    def test_create_edge(self) -> None:
        """Test creating an edge."""
        edge = Edge(
            source_id="a::foo",
            target_id="b::bar",
            edge_type=EdgeType.CALLS,
        )
        assert edge.source_id == "a::foo"
        assert edge.target_id == "b::bar"
        assert edge.edge_type == EdgeType.CALLS

    def test_edge_to_dict(self) -> None:
        """Test converting edge to dictionary."""
        edge = Edge(
            source_id="a::foo",
            target_id="b::bar",
            edge_type=EdgeType.REFERENCES,
            weight=0.5,
        )
        d = edge.to_dict()
        assert d["source_id"] == "a::foo"
        assert d["edge_type"] == "references"
        assert d["weight"] == 0.5

    def test_edge_from_dict(self) -> None:
        """Test creating edge from dictionary."""
        d = {
            "source_id": "x::y",
            "target_id": "z::w",
            "edge_type": "contains",
        }
        edge = Edge.from_dict(d)
        assert edge.edge_type == EdgeType.CONTAINS


class TestCodeGraph:
    """Tests for CodeGraph class."""

    def test_empty_graph(self) -> None:
        """Test creating an empty graph."""
        graph = CodeGraph()
        assert graph.node_count() == 0
        assert graph.edge_count() == 0

    def test_add_node(self) -> None:
        """Test adding a node."""
        graph = CodeGraph()
        node = SymbolNode(
            id="test::foo",
            name="foo",
            kind="function",
            file_path="test.py",
            start_line=1,
            end_line=5,
        )
        graph.add_node(node)
        assert graph.node_count() == 1
        assert graph.has_node("test::foo")

    def test_add_edge(self) -> None:
        """Test adding an edge."""
        graph = CodeGraph()
        graph.add_node(
            SymbolNode(
                id="a::x",
                name="x",
                kind="function",
                file_path="a.py",
                start_line=1,
                end_line=5,
            )
        )
        graph.add_node(
            SymbolNode(
                id="b::y",
                name="y",
                kind="function",
                file_path="b.py",
                start_line=1,
                end_line=5,
            )
        )
        graph.add_edge(Edge(source_id="a::x", target_id="b::y", edge_type=EdgeType.CALLS))
        assert graph.edge_count() == 1

    def test_get_neighbors(self, sample_graph: CodeGraph) -> None:
        """Test getting neighbors of a node."""
        neighbors = sample_graph.get_neighbors("src/main.py::main")
        assert len(neighbors) == 2  # helper and Utils.process

    def test_get_neighbors_with_edge_filter(self, sample_graph: CodeGraph) -> None:
        """Test getting neighbors with edge type filter."""
        neighbors = sample_graph.get_neighbors("src/main.py::main", edge_types=[EdgeType.CALLS])
        assert len(neighbors) == 1
        assert neighbors[0][0].name == "helper"

    def test_get_neighbors_direction(self, sample_graph: CodeGraph) -> None:
        """Test getting neighbors by direction."""
        outgoing = sample_graph.get_neighbors("src/main.py::main", direction="outgoing")
        incoming = sample_graph.get_neighbors("src/main.py::helper", direction="incoming")
        assert len(outgoing) == 2
        assert len(incoming) == 1

    def test_get_nodes_in_file(self, sample_graph: CodeGraph) -> None:
        """Test getting all nodes in a file."""
        nodes = sample_graph.get_nodes_in_file("src/main.py")
        assert len(nodes) == 2
        names = {n.name for n in nodes}
        assert names == {"main", "helper"}

    def test_get_nodes_by_kind(self, sample_graph: CodeGraph) -> None:
        """Test getting nodes by kind."""
        functions = sample_graph.get_nodes_by_kind("function")
        assert len(functions) == 2

        classes = sample_graph.get_nodes_by_kind("class")
        assert len(classes) == 1
        assert classes[0].name == "Utils"

    def test_subgraph(self, sample_graph: CodeGraph) -> None:
        """Test creating a subgraph."""
        node_ids = {"src/main.py::main", "src/main.py::helper"}
        subgraph = sample_graph.subgraph(node_ids)
        assert subgraph.node_count() == 2
        # Only edges between included nodes
        assert subgraph.edge_count() == 1

    def test_merge(self) -> None:
        """Test merging graphs."""
        graph1 = CodeGraph()
        graph1.add_node(
            SymbolNode(
                id="a::x",
                name="x",
                kind="function",
                file_path="a.py",
                start_line=1,
                end_line=5,
            )
        )

        graph2 = CodeGraph()
        graph2.add_node(
            SymbolNode(
                id="b::y",
                name="y",
                kind="function",
                file_path="b.py",
                start_line=1,
                end_line=5,
            )
        )

        graph1.merge(graph2)
        assert graph1.node_count() == 2

    def test_serialize_deserialize(self, sample_graph: CodeGraph) -> None:
        """Test serialization and deserialization."""
        d = sample_graph.to_dict()
        restored = CodeGraph.from_dict(d)
        assert restored.node_count() == sample_graph.node_count()
        assert restored.edge_count() == sample_graph.edge_count()


class TestMakeSymbolId:
    """Tests for make_symbol_id function."""

    def test_simple_id(self) -> None:
        """Test creating a simple symbol ID."""
        symbol_id = make_symbol_id("src/main.py", "foo")
        assert symbol_id == "src/main.py::foo"

    def test_nested_id(self) -> None:
        """Test creating a nested symbol ID."""
        symbol_id = make_symbol_id("src/main.py", "method", parent="MyClass")
        assert symbol_id == "src/main.py::MyClass.method"


# =============================================================================
# RETRIEVAL TESTS
# =============================================================================


class TestExpandGraph:
    """Tests for expand_graph function."""

    def test_expand_single_seed(self, sample_graph: CodeGraph) -> None:
        """Test expanding from a single seed."""
        expanded = expand_graph(sample_graph, ["src/main.py::main"], depth=1, limit=10)
        # Should include main and its immediate neighbors
        assert expanded.node_count() >= 1

    def test_expand_with_depth(self, sample_graph: CodeGraph) -> None:
        """Test expanding with depth limit."""
        expanded_depth1 = expand_graph(sample_graph, ["src/main.py::main"], depth=1, limit=100)
        expanded_depth2 = expand_graph(sample_graph, ["src/main.py::main"], depth=2, limit=100)
        # Depth 2 should include at least as many nodes
        assert expanded_depth2.node_count() >= expanded_depth1.node_count()

    def test_expand_with_edge_filter(self, sample_graph: CodeGraph) -> None:
        """Test expanding with edge type filter."""
        expanded = expand_graph(
            sample_graph,
            ["src/main.py::main"],
            edge_types=[EdgeType.CALLS],
            depth=1,
            limit=10,
        )
        # Should only follow CALLS edges
        assert expanded.has_node("src/main.py::main")
        assert expanded.has_node("src/main.py::helper")

    def test_expand_with_limit(self, sample_graph: CodeGraph) -> None:
        """Test expanding with node limit."""
        expanded = expand_graph(sample_graph, ["src/main.py::main"], depth=10, limit=2)
        assert expanded.node_count() <= 2


class TestPackSubgraph:
    """Tests for pack_subgraph function."""

    def test_pack_basic(self, sample_graph: CodeGraph) -> None:
        """Test packing a subgraph."""
        packed = pack_subgraph(sample_graph, target_count=2)
        assert len(packed) <= 2
        # Each item should be a PackedNode
        for item in packed:
            assert isinstance(item, PackedNode)
            assert item.reason  # Has a reason

    def test_pack_with_node_filter(self, sample_graph: CodeGraph) -> None:
        """Test packing with specific nodes."""
        node_ids = {"src/main.py::main", "src/main.py::helper"}
        packed = pack_subgraph(sample_graph, node_ids, target_count=10)
        # Should only consider specified nodes
        packed_ids = {p.node.id for p in packed}
        assert packed_ids.issubset(node_ids)

    def test_pack_scoring(self, sample_graph: CodeGraph) -> None:
        """Test that packing respects scoring."""
        packed = pack_subgraph(sample_graph, target_count=10)
        # Scores should be in descending order
        scores = [p.score for p in packed]
        assert scores == sorted(scores, reverse=True)


class TestTracePath:
    """Tests for trace_path function."""

    def test_trace_direct_path(self, sample_graph: CodeGraph) -> None:
        """Test tracing a direct path."""
        paths = trace_path(sample_graph, "src/main.py::main", "src/main.py::helper")
        assert len(paths) >= 1
        # Path should have at least 2 nodes
        assert len(paths[0]) >= 2

    def test_trace_no_path(self, sample_graph: CodeGraph) -> None:
        """Test tracing when no path exists."""
        # Add an isolated node
        sample_graph.add_node(
            SymbolNode(
                id="isolated::node",
                name="node",
                kind="function",
                file_path="isolated.py",
                start_line=1,
                end_line=5,
            )
        )
        paths = trace_path(sample_graph, "src/main.py::main", "isolated::node")
        assert len(paths) == 0

    def test_trace_same_node(self, sample_graph: CodeGraph) -> None:
        """Test tracing to the same node."""
        paths = trace_path(sample_graph, "src/main.py::main", "src/main.py::main")
        assert len(paths) == 1
        assert len(paths[0]) == 1

    def test_trace_with_edge_filter(self, sample_graph: CodeGraph) -> None:
        """Test tracing with edge type filter."""
        paths_all = trace_path(sample_graph, "src/main.py::main", "src/utils.py::Utils.process")
        paths_calls = trace_path(
            sample_graph,
            "src/main.py::main",
            "src/utils.py::Utils.process",
            edge_types=[EdgeType.CALLS],
        )
        # Filtered path should be different or empty
        assert len(paths_calls) <= len(paths_all)


# =============================================================================
# BUILDER TESTS
# =============================================================================


class TestGraphBuilder:
    """Tests for GraphBuilder class."""

    def test_builder_without_backends(self) -> None:
        """Test builder without LSP or Tree-sitter."""
        builder = GraphBuilder()
        assert not builder.has_lsp
        assert not builder.has_treesitter

    def test_build_empty_graph(self) -> None:
        """Test building graph from non-existent file."""
        builder = GraphBuilder()
        graph = builder.build_file_subgraph("/nonexistent/path.py")
        assert graph.node_count() == 0

    def test_builder_with_treesitter(self, tmp_path: Path) -> None:
        """Test builder with Tree-sitter."""
        try:
            from contextmine_core.treesitter import get_treesitter_manager

            ts_manager = get_treesitter_manager()
            if not ts_manager.is_available():
                pytest.skip("Tree-sitter not available")

            # Create test file
            test_file = tmp_path / "test.py"
            test_file.write_text("""
def foo():
    pass

class Bar:
    def method(self):
        pass
""")

            builder = GraphBuilder(treesitter_manager=ts_manager)
            assert builder.has_treesitter

            graph = builder.build_file_subgraph(str(test_file))
            assert graph.node_count() >= 2  # At least foo and Bar

        except ImportError:
            pytest.skip("Tree-sitter not installed")

    def test_multi_file_graph(self, tmp_path: Path) -> None:
        """Test building graph from multiple files."""
        try:
            from contextmine_core.treesitter import get_treesitter_manager

            ts_manager = get_treesitter_manager()
            if not ts_manager.is_available():
                pytest.skip("Tree-sitter not available")

            # Create test files
            file1 = tmp_path / "a.py"
            file1.write_text("def func_a(): pass")

            file2 = tmp_path / "b.py"
            file2.write_text("def func_b(): pass")

            builder = GraphBuilder(treesitter_manager=ts_manager)
            graph = builder.build_multi_file_graph([str(file1), str(file2)])

            # Should have symbols from both files
            assert graph.node_count() >= 2

        except ImportError:
            pytest.skip("Tree-sitter not installed")


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestGraphIntegration:
    """Integration tests for the graph module."""

    def test_full_workflow(self, tmp_path: Path) -> None:
        """Test full graph workflow: build -> expand -> pack."""
        try:
            from contextmine_core.treesitter import get_treesitter_manager

            ts_manager = get_treesitter_manager()
            if not ts_manager.is_available():
                pytest.skip("Tree-sitter not available")

            # Create test file
            test_file = tmp_path / "app.py"
            test_file.write_text("""
def main():
    helper()

def helper():
    pass

class Service:
    def run(self):
        pass
""")

            # Build graph
            builder = GraphBuilder(treesitter_manager=ts_manager)
            graph = builder.build_file_subgraph(str(test_file))
            assert graph.node_count() >= 3

            # Expand from a seed
            main_id = None
            for node in graph.get_all_nodes():
                if node.name == "main":
                    main_id = node.id
                    break

            if main_id:
                expanded = expand_graph(graph, [main_id], depth=2, limit=50)
                assert expanded.node_count() >= 1

                # Pack
                packed = pack_subgraph(expanded, target_count=3)
                assert len(packed) >= 1

        except ImportError:
            pytest.skip("Tree-sitter not installed")
