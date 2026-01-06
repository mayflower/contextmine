"""Tests for GraphRAG retrieval service."""

from uuid import UUID

from contextmine_core.graphrag import (
    Citation,
    CommunityContext,
    ContextPack,
    EdgeContext,
    EntityContext,
    PathContext,
)


class TestCitation:
    """Tests for Citation dataclass."""

    def test_create_citation(self) -> None:
        """Test creating a citation."""
        cit = Citation(
            file_path="src/main.py",
            start_line=10,
            end_line=20,
            snippet="def foo():",
        )
        assert cit.file_path == "src/main.py"
        assert cit.start_line == 10
        assert cit.end_line == 20

    def test_citation_format(self) -> None:
        """Test citation formatting."""
        cit = Citation(
            file_path="src/main.py",
            start_line=10,
            end_line=20,
        )
        assert cit.format() == "src/main.py:10-20"

    def test_citation_without_snippet(self) -> None:
        """Test citation without snippet."""
        cit = Citation(
            file_path="test.py",
            start_line=1,
            end_line=5,
        )
        assert cit.snippet is None


class TestCommunityContext:
    """Tests for CommunityContext dataclass."""

    def test_create_community_context(self) -> None:
        """Test creating community context."""
        ctx = CommunityContext(
            community_id=UUID("12345678-1234-5678-1234-567812345678"),
            level=1,
            title="Auth Module",
            summary="Handles authentication",
            relevance_score=0.85,
            member_count=10,
        )
        assert ctx.level == 1
        assert ctx.title == "Auth Module"
        assert ctx.relevance_score == 0.85


class TestEntityContext:
    """Tests for EntityContext dataclass."""

    def test_create_entity_context(self) -> None:
        """Test creating entity context."""
        ctx = EntityContext(
            node_id=UUID("12345678-1234-5678-1234-567812345678"),
            kind="symbol",
            natural_key="symbol:main.py:10:0",
            name="authenticate",
            relevance_score=0.9,
        )
        assert ctx.kind == "symbol"
        assert ctx.name == "authenticate"
        assert ctx.evidence == []

    def test_entity_with_evidence(self) -> None:
        """Test entity with citations."""
        ctx = EntityContext(
            node_id=UUID("12345678-1234-5678-1234-567812345678"),
            kind="symbol",
            natural_key="symbol:main.py:10:0",
            name="authenticate",
            evidence=[
                Citation(file_path="auth.py", start_line=10, end_line=50),
            ],
        )
        assert len(ctx.evidence) == 1
        assert ctx.evidence[0].file_path == "auth.py"


class TestEdgeContext:
    """Tests for EdgeContext dataclass."""

    def test_create_edge_context(self) -> None:
        """Test creating edge context."""
        ctx = EdgeContext(
            source_id="node1",
            target_id="node2",
            kind="calls",
            source_name="foo",
            target_name="bar",
        )
        assert ctx.kind == "calls"
        assert ctx.source_name == "foo"

    def test_edge_without_names(self) -> None:
        """Test edge context without names."""
        ctx = EdgeContext(
            source_id="node1",
            target_id="node2",
            kind="file_defines_symbol",
        )
        assert ctx.source_name is None
        assert ctx.target_name is None


class TestPathContext:
    """Tests for PathContext dataclass."""

    def test_create_path_context(self) -> None:
        """Test creating path context."""
        ctx = PathContext(
            nodes=["node1", "node2", "node3"],
            edges=["calls", "references"],
            description="foo → bar → baz",
        )
        assert len(ctx.nodes) == 3
        assert len(ctx.edges) == 2

    def test_path_context_defaults(self) -> None:
        """Test path context with defaults."""
        ctx = PathContext(
            nodes=["node1"],
            edges=[],
        )
        assert ctx.description == ""


class TestContextPack:
    """Tests for ContextPack dataclass."""

    def test_empty_context_pack(self) -> None:
        """Test creating empty context pack."""
        pack = ContextPack(query="test query")
        assert pack.query == "test query"
        assert pack.communities == []
        assert pack.entities == []
        assert pack.edges == []
        assert pack.citations == []

    def test_context_pack_to_markdown(self) -> None:
        """Test context pack markdown generation."""
        pack = ContextPack(
            query="find auth logic",
            communities=[
                CommunityContext(
                    community_id=UUID("12345678-1234-5678-1234-567812345678"),
                    level=1,
                    title="Auth Module",
                    summary="Handles user authentication and session management",
                    relevance_score=0.9,
                    member_count=15,
                )
            ],
            entities=[
                EntityContext(
                    node_id=UUID("22345678-1234-5678-1234-567812345678"),
                    kind="symbol",
                    natural_key="symbol:auth.py:10:0",
                    name="authenticate",
                    relevance_score=0.95,
                )
            ],
            citations=[Citation(file_path="auth.py", start_line=10, end_line=50)],
        )

        md = pack.to_markdown()

        assert "find auth logic" in md
        assert "Global Context" in md
        assert "Auth Module" in md
        assert "Local Context" in md
        assert "authenticate" in md
        assert "Source Citations" in md

    def test_context_pack_to_markdown_with_edges(self) -> None:
        """Test markdown includes relationships."""
        pack = ContextPack(
            query="test",
            edges=[
                EdgeContext(source_id="n1", target_id="n2", kind="calls"),
                EdgeContext(source_id="n2", target_id="n3", kind="calls"),
                EdgeContext(source_id="n1", target_id="n3", kind="imports"),
            ],
        )

        md = pack.to_markdown()

        assert "Relationships" in md
        assert "calls: 2" in md
        assert "imports: 1" in md

    def test_context_pack_to_markdown_with_paths(self) -> None:
        """Test markdown includes paths."""
        pack = ContextPack(
            query="test",
            paths=[
                PathContext(
                    nodes=["a", "b", "c"],
                    edges=["calls", "returns"],
                    description="a → b → c",
                )
            ],
        )

        md = pack.to_markdown()

        assert "Key Paths" in md
        assert "a → b → c" in md

    def test_context_pack_to_dict(self) -> None:
        """Test context pack dict serialization."""
        pack = ContextPack(
            query="test",
            communities=[
                CommunityContext(
                    community_id=UUID("12345678-1234-5678-1234-567812345678"),
                    level=1,
                    title="Test",
                    summary="Test summary",
                    relevance_score=0.5,
                    member_count=5,
                )
            ],
        )

        data = pack.to_dict()

        assert data["query"] == "test"
        assert len(data["communities"]) == 1
        assert data["communities"][0]["title"] == "Test"

        # Verify JSON-serializable
        import json

        json_str = json.dumps(data)
        assert len(json_str) > 0

    def test_context_pack_to_dict_full(self) -> None:
        """Test full context pack serialization."""
        pack = ContextPack(
            query="find auth",
            communities=[
                CommunityContext(
                    community_id=UUID("12345678-1234-5678-1234-567812345678"),
                    level=1,
                    title="Auth",
                    summary="Authentication",
                    relevance_score=0.9,
                    member_count=10,
                )
            ],
            entities=[
                EntityContext(
                    node_id=UUID("22345678-1234-5678-1234-567812345678"),
                    kind="symbol",
                    natural_key="symbol:auth.py:10:0",
                    name="authenticate",
                    evidence=[Citation(file_path="auth.py", start_line=10, end_line=20)],
                )
            ],
            edges=[EdgeContext(source_id="n1", target_id="n2", kind="calls")],
            paths=[PathContext(nodes=["n1", "n2"], edges=["calls"], description="n1 → n2")],
            citations=[Citation(file_path="auth.py", start_line=10, end_line=50)],
        )

        data = pack.to_dict()

        # Verify structure
        assert "query" in data
        assert "communities" in data
        assert "entities" in data
        assert "edges" in data
        assert "paths" in data
        assert "citations" in data

        # Verify content
        assert len(data["communities"]) == 1
        assert len(data["entities"]) == 1
        assert len(data["edges"]) == 1
        assert len(data["paths"]) == 1
        assert len(data["citations"]) == 1

        # Verify entity has evidence
        assert len(data["entities"][0]["evidence"]) == 1

        # Verify JSON-serializable
        import json

        json_str = json.dumps(data)
        assert len(json_str) > 0
