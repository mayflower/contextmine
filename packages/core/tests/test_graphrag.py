"""Tests for GraphRAG retrieval service."""

from contextmine_core.graphrag import (
    Evidence,
    GraphEdge,
    GraphNode,
    GraphRAGResult,
    _build_markdown_summary,
    _build_neighborhood_markdown,
    _build_path_markdown,
)


class TestGraphRAGResult:
    """Tests for GraphRAGResult dataclass."""

    def test_empty_result(self) -> None:
        """Test creating an empty result."""
        result = GraphRAGResult(query="test query")
        assert result.query == "test query"
        assert result.nodes == []
        assert result.edges == []
        assert result.evidence == []

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        result = GraphRAGResult(
            query="test",
            nodes=[
                GraphNode(
                    id="node1",
                    kind="file",
                    name="test.py",
                    natural_key="file:test.py",
                    meta={"size": 100},
                )
            ],
            edges=[
                GraphEdge(
                    source_id="node1",
                    target_id="node2",
                    kind="defines",
                    meta={},
                )
            ],
            evidence=[
                Evidence(
                    file_path="test.py",
                    start_line=1,
                    end_line=10,
                    snippet="def test():",
                )
            ],
            summary_markdown="# Test",
        )

        data = result.to_dict()

        assert data["query"] == "test"
        assert len(data["nodes"]) == 1
        assert data["nodes"][0]["id"] == "node1"
        assert data["nodes"][0]["kind"] == "file"
        assert len(data["edges"]) == 1
        assert data["edges"][0]["kind"] == "defines"
        assert len(data["evidence"]) == 1
        assert data["evidence"][0]["file_path"] == "test.py"


class TestGraphNode:
    """Tests for GraphNode dataclass."""

    def test_create_node(self) -> None:
        """Test creating a graph node."""
        node = GraphNode(
            id="123",
            kind="symbol",
            name="my_function",
            natural_key="symbol:test.py:my_function",
            meta={"language": "python"},
        )
        assert node.id == "123"
        assert node.kind == "symbol"
        assert node.name == "my_function"
        assert node.meta["language"] == "python"

    def test_node_default_meta(self) -> None:
        """Test node with default meta."""
        node = GraphNode(
            id="123",
            kind="file",
            name="test.py",
            natural_key="file:test.py",
        )
        assert node.meta == {}


class TestGraphEdge:
    """Tests for GraphEdge dataclass."""

    def test_create_edge(self) -> None:
        """Test creating a graph edge."""
        edge = GraphEdge(
            source_id="node1",
            target_id="node2",
            kind="file_defines_symbol",
            meta={"weight": 1.0},
        )
        assert edge.source_id == "node1"
        assert edge.target_id == "node2"
        assert edge.kind == "file_defines_symbol"


class TestEvidence:
    """Tests for Evidence dataclass."""

    def test_create_evidence(self) -> None:
        """Test creating evidence."""
        ev = Evidence(
            file_path="src/main.py",
            start_line=10,
            end_line=20,
            snippet="class MyClass:",
        )
        assert ev.file_path == "src/main.py"
        assert ev.start_line == 10
        assert ev.end_line == 20
        assert ev.snippet == "class MyClass:"

    def test_evidence_without_snippet(self) -> None:
        """Test evidence without snippet."""
        ev = Evidence(
            file_path="test.py",
            start_line=1,
            end_line=5,
        )
        assert ev.snippet is None


class TestMarkdownBuilders:
    """Tests for markdown building functions."""

    def test_build_neighborhood_markdown(self) -> None:
        """Test neighborhood markdown generation."""
        nodes = [
            GraphNode(id="1", kind="file", name="main.py", natural_key="file:main.py"),
            GraphNode(id="2", kind="symbol", name="main", natural_key="symbol:main"),
        ]
        edges = [
            GraphEdge(source_id="1", target_id="2", kind="file_defines_symbol"),
        ]
        evidence: list[Evidence] = []

        md = _build_neighborhood_markdown(nodes, edges, evidence)

        assert "Graph Neighborhood" in md
        assert "2 nodes" in md
        assert "1 edges" in md
        assert "main.py" in md
        assert "file_defines_symbol" in md

    def test_build_path_markdown(self) -> None:
        """Test path markdown generation."""
        nodes = [
            GraphNode(id="1", kind="file", name="a.py", natural_key="file:a.py"),
            GraphNode(id="2", kind="symbol", name="func_a", natural_key="symbol:func_a"),
            GraphNode(id="3", kind="symbol", name="func_b", natural_key="symbol:func_b"),
        ]
        edges = [
            GraphEdge(source_id="1", target_id="2", kind="defines"),
            GraphEdge(source_id="2", target_id="3", kind="calls"),
        ]

        md = _build_path_markdown(nodes, edges)

        assert "Path Trace" in md
        assert "3 nodes" in md
        assert "2 edges" in md
        assert "a.py" in md
        assert "func_a" in md
        assert "func_b" in md

    def test_build_path_markdown_empty(self) -> None:
        """Test path markdown with no path."""
        md = _build_path_markdown([], [])
        assert "No path found" in md

    def test_build_markdown_summary(self) -> None:
        """Test full summary markdown generation."""
        from contextmine_core.search import SearchResponse, SearchResult

        nodes = [
            GraphNode(id="1", kind="file", name="test.py", natural_key="file:test.py"),
            GraphNode(id="2", kind="symbol", name="test_func", natural_key="symbol:test"),
            GraphNode(id="3", kind="rule_candidate", name="Rule: x < 0", natural_key="rule:1"),
        ]
        edges = [
            GraphEdge(source_id="1", target_id="2", kind="file_defines_symbol"),
        ]
        evidence = [
            Evidence(file_path="test.py", start_line=10, end_line=20),
        ]
        search_response = SearchResponse(
            results=[
                SearchResult(
                    chunk_id="c1",
                    document_id="d1",
                    source_id="s1",
                    collection_id="col1",
                    content="test content",
                    uri="file://test.py",
                    title="test.py",
                    score=0.9,
                    fts_rank=1,
                    vector_rank=1,
                    fts_score=0.8,
                    vector_score=0.85,
                )
            ],
            query="test query",
            total_fts_matches=1,
            total_vector_matches=1,
        )

        md = _build_markdown_summary("test query", search_response, nodes, edges, evidence)

        assert "GraphRAG Results" in md
        assert "3 knowledge graph nodes" in md
        assert "1 edges" in md
        assert "FILE" in md
        assert "SYMBOL" in md
        assert "RULE_CANDIDATE" in md
        assert "test.py:10-20" in md


class TestResultSerialization:
    """Tests for result serialization."""

    def test_full_result_serialization(self) -> None:
        """Test that a full result serializes correctly."""
        result = GraphRAGResult(
            query="find authentication logic",
            nodes=[
                GraphNode(
                    id="n1",
                    kind="file",
                    name="auth.py",
                    natural_key="file:auth.py",
                    meta={"path": "/src/auth.py"},
                ),
                GraphNode(
                    id="n2",
                    kind="symbol",
                    name="authenticate",
                    natural_key="symbol:auth.py:authenticate",
                    meta={"kind": "function"},
                ),
            ],
            edges=[
                GraphEdge(
                    source_id="n1",
                    target_id="n2",
                    kind="file_defines_symbol",
                    meta={},
                ),
            ],
            evidence=[
                Evidence(
                    file_path="/src/auth.py",
                    start_line=10,
                    end_line=50,
                    snippet="def authenticate(user, password):",
                ),
            ],
            summary_markdown="# Auth Results\n\nFound authentication module.",
        )

        data = result.to_dict()

        # Verify structure
        assert "query" in data
        assert "nodes" in data
        assert "edges" in data
        assert "evidence" in data
        assert "summary_markdown" in data

        # Verify content
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1
        assert len(data["evidence"]) == 1

        # Verify it's JSON-serializable
        import json

        json_str = json.dumps(data)
        assert len(json_str) > 0
