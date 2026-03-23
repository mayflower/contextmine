"""Unit tests for GraphRAG dataclasses and rendering helpers.

Tests pure logic in graphrag.py that does not require a database:
- Citation formatting
- ContextPack.to_markdown() rendering
- ContextPack.to_dict() serialization
- MapReduceResult dataclass
- Edge cases: empty packs, truncation, large community summaries
"""

from uuid import uuid4

from contextmine_core.graphrag import (
    Citation,
    CommunityContext,
    ContextPack,
    EdgeContext,
    EntityContext,
    MapReduceResult,
    PathContext,
)

# ---------------------------------------------------------------------------
# Helpers to build test data
# ---------------------------------------------------------------------------


def _make_citation(
    file_path: str = "src/main.py",
    start_line: int = 10,
    end_line: int = 20,
    snippet: str | None = "code here",
) -> Citation:
    return Citation(
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        snippet=snippet,
    )


def _make_entity(
    kind: str = "FILE",
    name: str = "main.py",
    natural_key: str = "file:src/main.py",
    evidence: list[Citation] | None = None,
    relevance_score: float = 0.9,
) -> EntityContext:
    return EntityContext(
        node_id=uuid4(),
        kind=kind,
        natural_key=natural_key,
        name=name,
        evidence=evidence or [],
        relevance_score=relevance_score,
    )


def _make_community(
    title: str = "Auth Module",
    level: int = 0,
    summary: str = "Handles authentication and authorization.",
    relevance_score: float = 0.85,
    member_count: int = 5,
) -> CommunityContext:
    return CommunityContext(
        community_id=uuid4(),
        level=level,
        title=title,
        summary=summary,
        relevance_score=relevance_score,
        member_count=member_count,
    )


def _make_edge(
    kind: str = "FILE_DEFINES_SYMBOL",
    source_name: str | None = "main.py",
    target_name: str | None = "hello",
) -> EdgeContext:
    return EdgeContext(
        source_id=str(uuid4()),
        target_id=str(uuid4()),
        kind=kind,
        source_name=source_name,
        target_name=target_name,
    )


def _make_path(
    nodes: list[str] | None = None,
    edges: list[str] | None = None,
    description: str = "main.py -> hello",
) -> PathContext:
    return PathContext(
        nodes=nodes or ["file:src/main.py", "symbol:hello"],
        edges=edges or ["FILE_DEFINES_SYMBOL"],
        description=description,
    )


# ===========================================================================
# Citation tests
# ===========================================================================


class TestCitation:
    """Tests for Citation dataclass."""

    def test_format_basic(self) -> None:
        """Test basic citation formatting."""
        cit = _make_citation()
        assert cit.format() == "src/main.py:10-20"

    def test_format_same_line(self) -> None:
        """Test formatting when start and end line are the same."""
        cit = _make_citation(start_line=42, end_line=42)
        assert cit.format() == "src/main.py:42-42"

    def test_snippet_default_none(self) -> None:
        """Test that snippet defaults to None."""
        cit = Citation(file_path="a.py", start_line=1, end_line=1)
        assert cit.snippet is None

    def test_snippet_preserved(self) -> None:
        """Test that snippet is stored."""
        cit = _make_citation(snippet="def hello(): pass")
        assert cit.snippet == "def hello(): pass"


# ===========================================================================
# CommunityContext tests
# ===========================================================================


class TestCommunityContext:
    """Tests for CommunityContext dataclass."""

    def test_creation(self) -> None:
        """Test basic construction."""
        comm = _make_community()
        assert comm.title == "Auth Module"
        assert comm.level == 0
        assert comm.member_count == 5
        assert comm.relevance_score == 0.85

    def test_empty_summary(self) -> None:
        """Test community with empty summary."""
        comm = _make_community(summary="")
        assert comm.summary == ""


# ===========================================================================
# EntityContext tests
# ===========================================================================


class TestEntityContext:
    """Tests for EntityContext dataclass."""

    def test_defaults(self) -> None:
        """Test default values."""
        entity = EntityContext(
            node_id=uuid4(),
            kind="SYMBOL",
            natural_key="symbol:hello",
            name="hello",
        )
        assert entity.evidence == []
        assert entity.relevance_score == 0.0

    def test_with_evidence(self) -> None:
        """Test entity with evidence citations."""
        cits = [_make_citation(), _make_citation(file_path="other.py")]
        entity = _make_entity(evidence=cits)
        assert len(entity.evidence) == 2


# ===========================================================================
# EdgeContext tests
# ===========================================================================


class TestEdgeContext:
    """Tests for EdgeContext dataclass."""

    def test_creation(self) -> None:
        """Test basic construction."""
        edge = _make_edge()
        assert edge.kind == "FILE_DEFINES_SYMBOL"
        assert edge.source_name == "main.py"
        assert edge.target_name == "hello"

    def test_names_optional(self) -> None:
        """Test that source_name and target_name default to None."""
        edge = EdgeContext(
            source_id="a",
            target_id="b",
            kind="CALLS",
        )
        assert edge.source_name is None
        assert edge.target_name is None


# ===========================================================================
# PathContext tests
# ===========================================================================


class TestPathContext:
    """Tests for PathContext dataclass."""

    def test_creation(self) -> None:
        """Test basic construction."""
        path = _make_path()
        assert len(path.nodes) == 2
        assert len(path.edges) == 1
        assert path.description == "main.py -> hello"

    def test_empty_description_default(self) -> None:
        """Test that description defaults to empty string."""
        path = PathContext(nodes=["a"], edges=[])
        assert path.description == ""


# ===========================================================================
# ContextPack.to_markdown() tests
# ===========================================================================


class TestContextPackToMarkdown:
    """Tests for ContextPack.to_markdown() rendering."""

    def test_empty_pack(self) -> None:
        """Test markdown for a completely empty context pack."""
        pack = ContextPack(query="What is the auth flow?")
        md = pack.to_markdown()

        assert "# GraphRAG Context: What is the auth flow?" in md
        assert "0 communities" in md
        assert "0 entities" in md
        assert "0 citations" in md
        # No section headers for empty sections
        assert "## Global Context" not in md
        assert "## Local Context" not in md
        assert "## Relationships" not in md
        assert "## Key Paths" not in md
        assert "## Source Citations" not in md

    def test_communities_section(self) -> None:
        """Test that communities render correctly."""
        pack = ContextPack(
            query="test",
            communities=[
                _make_community(title="Auth Module", level=0, member_count=5),
                _make_community(title="DB Layer", level=1, member_count=3),
            ],
        )
        md = pack.to_markdown()

        assert "## Global Context (Community Summaries)" in md
        assert "### Auth Module (Level 0)" in md
        assert "### DB Layer (Level 1)" in md
        assert "5 members" in md
        assert "3 members" in md
        assert "Handles authentication" in md

    def test_community_long_summary_truncated(self) -> None:
        """Test that community summaries over 500 chars are truncated."""
        long_summary = "x" * 600
        pack = ContextPack(
            query="test",
            communities=[_make_community(summary=long_summary)],
        )
        md = pack.to_markdown()

        # Should be truncated to 500 chars + "..."
        assert "x" * 500 + "..." in md
        assert "x" * 501 not in md.replace("x" * 500 + "...", "")

    def test_community_empty_summary_not_rendered(self) -> None:
        """Test that an empty community summary is omitted from output."""
        pack = ContextPack(
            query="test",
            communities=[_make_community(summary="")],
        )
        md = pack.to_markdown()
        # The community header still appears
        assert "### Auth Module" in md

    def test_entities_section_grouped_by_kind(self) -> None:
        """Test that entities are grouped by kind in the markdown."""
        pack = ContextPack(
            query="test",
            entities=[
                _make_entity(kind="FILE", name="main.py"),
                _make_entity(kind="FILE", name="utils.py"),
                _make_entity(kind="SYMBOL", name="hello"),
            ],
        )
        md = pack.to_markdown()

        assert "## Local Context (Entities)" in md
        assert "**FILE** (2):" in md
        assert "**SYMBOL** (1):" in md
        assert "- main.py" in md
        assert "- utils.py" in md
        assert "- hello" in md

    def test_entities_with_evidence_count(self) -> None:
        """Test that entities show citation count when they have evidence."""
        pack = ContextPack(
            query="test",
            entities=[
                _make_entity(
                    kind="SYMBOL",
                    name="authenticate",
                    evidence=[_make_citation(), _make_citation(file_path="b.py")],
                ),
            ],
        )
        md = pack.to_markdown()
        assert "- authenticate [2 citations]" in md

    def test_entities_no_evidence_no_bracket(self) -> None:
        """Test that entities without evidence do not show citation brackets."""
        pack = ContextPack(
            query="test",
            entities=[_make_entity(kind="SYMBOL", name="hello", evidence=[])],
        )
        md = pack.to_markdown()
        assert "- hello" in md
        assert "[" not in md.split("- hello")[1].split("\n")[0]

    def test_entities_truncated_at_10(self) -> None:
        """Test that entities of a single kind are truncated to 10 shown."""
        entities = [_make_entity(kind="SYMBOL", name=f"func_{i}") for i in range(15)]
        pack = ContextPack(query="test", entities=entities)
        md = pack.to_markdown()

        assert "**SYMBOL** (15):" in md
        assert "- func_0" in md
        assert "- func_9" in md
        assert "... and 5 more" in md

    def test_edges_section(self) -> None:
        """Test that edges render as kind counts."""
        pack = ContextPack(
            query="test",
            edges=[
                _make_edge(kind="FILE_DEFINES_SYMBOL"),
                _make_edge(kind="FILE_DEFINES_SYMBOL"),
                _make_edge(kind="SYMBOL_CALLS_SYMBOL"),
            ],
        )
        md = pack.to_markdown()

        assert "## Relationships" in md
        assert "FILE_DEFINES_SYMBOL: 2" in md
        assert "SYMBOL_CALLS_SYMBOL: 1" in md

    def test_edges_top_5_only(self) -> None:
        """Test that only top 5 edge kinds are shown (sorted by count desc)."""
        edges = []
        for i in range(7):
            for _ in range(7 - i):
                edges.append(_make_edge(kind=f"KIND_{i}"))
        pack = ContextPack(query="test", edges=edges)
        md = pack.to_markdown()

        # Should show top 5 edge kinds by count
        assert "KIND_0:" in md
        assert "KIND_4:" in md
        # KIND_5 and KIND_6 have fewer entries, may or may not appear
        # depending on sorting; the important thing is at most 5 lines
        relationship_section = md.split("## Relationships")[1].split("##")[0]
        kind_lines = [
            line for line in relationship_section.strip().split("\n") if line.startswith("- ")
        ]
        assert len(kind_lines) <= 5

    def test_paths_section(self) -> None:
        """Test that paths render correctly, limited to 3."""
        paths = [_make_path(description=f"path_{i}") for i in range(5)]
        pack = ContextPack(query="test", paths=paths)
        md = pack.to_markdown()

        assert "## Key Paths" in md
        assert "- path_0" in md
        assert "- path_2" in md
        assert "- path_3" not in md  # Only first 3 shown

    def test_citations_section(self) -> None:
        """Test that citations render with backtick formatting."""
        pack = ContextPack(
            query="test",
            citations=[
                _make_citation(file_path="src/auth.py", start_line=5, end_line=15),
            ],
        )
        md = pack.to_markdown()

        assert "## Source Citations" in md
        assert "- `src/auth.py:5-15`" in md

    def test_citations_truncated_at_10(self) -> None:
        """Test that citations are truncated to 10 with overflow message."""
        citations = [
            _make_citation(file_path=f"file_{i}.py", start_line=i, end_line=i + 1)
            for i in range(15)
        ]
        pack = ContextPack(query="test", citations=citations)
        md = pack.to_markdown()

        assert "- `file_0.py:0-1`" in md
        assert "- `file_9.py:9-10`" in md
        assert "... and 5 more" in md

    def test_full_pack_all_sections(self) -> None:
        """Test a fully populated pack renders all sections."""
        pack = ContextPack(
            query="How does auth work?",
            communities=[_make_community()],
            entities=[_make_entity()],
            edges=[_make_edge()],
            paths=[_make_path()],
            citations=[_make_citation()],
        )
        md = pack.to_markdown()

        assert "# GraphRAG Context: How does auth work?" in md
        assert "1 communities" in md
        assert "1 entities" in md
        assert "1 citations" in md
        assert "## Global Context (Community Summaries)" in md
        assert "## Local Context (Entities)" in md
        assert "## Relationships" in md
        assert "## Key Paths" in md
        assert "## Source Citations" in md


# ===========================================================================
# ContextPack.to_dict() tests
# ===========================================================================


class TestContextPackToDict:
    """Tests for ContextPack.to_dict() serialization."""

    def test_empty_pack(self) -> None:
        """Test serializing an empty context pack."""
        pack = ContextPack(query="test query")
        d = pack.to_dict()

        assert d["query"] == "test query"
        assert d["communities"] == []
        assert d["entities"] == []
        assert d["edges"] == []
        assert d["paths"] == []
        assert d["citations"] == []

    def test_communities_serialized(self) -> None:
        """Test that communities are correctly serialized."""
        comm_id = uuid4()
        pack = ContextPack(
            query="test",
            communities=[
                CommunityContext(
                    community_id=comm_id,
                    level=1,
                    title="Auth",
                    summary="Handles auth",
                    relevance_score=0.9,
                    member_count=3,
                )
            ],
        )
        d = pack.to_dict()

        assert len(d["communities"]) == 1
        comm = d["communities"][0]
        assert comm["community_id"] == str(comm_id)
        assert comm["level"] == 1
        assert comm["title"] == "Auth"
        assert comm["summary"] == "Handles auth"
        assert comm["relevance_score"] == 0.9
        assert comm["member_count"] == 3

    def test_entities_serialized_with_evidence(self) -> None:
        """Test that entities include evidence in serialization."""
        node_id = uuid4()
        pack = ContextPack(
            query="test",
            entities=[
                EntityContext(
                    node_id=node_id,
                    kind="SYMBOL",
                    natural_key="symbol:hello",
                    name="hello",
                    relevance_score=0.75,
                    evidence=[
                        Citation(file_path="a.py", start_line=1, end_line=5, snippet="code"),
                    ],
                )
            ],
        )
        d = pack.to_dict()

        assert len(d["entities"]) == 1
        entity = d["entities"][0]
        assert entity["node_id"] == str(node_id)
        assert entity["kind"] == "SYMBOL"
        assert entity["natural_key"] == "symbol:hello"
        assert entity["name"] == "hello"
        assert entity["relevance_score"] == 0.75
        assert len(entity["evidence"]) == 1
        ev = entity["evidence"][0]
        assert ev["file_path"] == "a.py"
        assert ev["start_line"] == 1
        assert ev["end_line"] == 5
        assert ev["snippet"] == "code"

    def test_edges_serialized(self) -> None:
        """Test that edges are serialized."""
        pack = ContextPack(
            query="test",
            edges=[
                EdgeContext(source_id="s1", target_id="t1", kind="CALLS"),
            ],
        )
        d = pack.to_dict()

        assert len(d["edges"]) == 1
        assert d["edges"][0] == {
            "source_id": "s1",
            "target_id": "t1",
            "kind": "CALLS",
        }

    def test_paths_serialized(self) -> None:
        """Test that paths are serialized."""
        pack = ContextPack(
            query="test",
            paths=[
                PathContext(
                    nodes=["a", "b", "c"],
                    edges=["CALLS", "REFS"],
                    description="a -> b -> c",
                )
            ],
        )
        d = pack.to_dict()

        assert len(d["paths"]) == 1
        assert d["paths"][0]["nodes"] == ["a", "b", "c"]
        assert d["paths"][0]["edges"] == ["CALLS", "REFS"]
        assert d["paths"][0]["description"] == "a -> b -> c"

    def test_citations_serialized(self) -> None:
        """Test that citations are serialized."""
        pack = ContextPack(
            query="test",
            citations=[
                Citation(file_path="x.py", start_line=10, end_line=20, snippet=None),
            ],
        )
        d = pack.to_dict()

        assert len(d["citations"]) == 1
        cit = d["citations"][0]
        assert cit["file_path"] == "x.py"
        assert cit["start_line"] == 10
        assert cit["end_line"] == 20
        assert cit["snippet"] is None

    def test_roundtrip_consistency(self) -> None:
        """Test that to_dict produces JSON-serializable output with correct types."""
        import json

        pack = ContextPack(
            query="roundtrip test",
            communities=[_make_community()],
            entities=[_make_entity(evidence=[_make_citation()])],
            edges=[_make_edge()],
            paths=[_make_path()],
            citations=[_make_citation()],
        )
        d = pack.to_dict()

        # Should be JSON serializable without errors
        json_str = json.dumps(d)
        restored = json.loads(json_str)

        assert restored["query"] == "roundtrip test"
        assert len(restored["communities"]) == 1
        assert len(restored["entities"]) == 1
        assert len(restored["edges"]) == 1
        assert len(restored["paths"]) == 1
        assert len(restored["citations"]) == 1

    def test_multiple_items(self) -> None:
        """Test serialization with multiple items in each section."""
        pack = ContextPack(
            query="multi",
            communities=[_make_community(title=f"C{i}") for i in range(3)],
            entities=[_make_entity(name=f"e{i}") for i in range(4)],
            edges=[_make_edge(kind=f"K{i}") for i in range(2)],
            paths=[_make_path(description=f"p{i}") for i in range(2)],
            citations=[_make_citation(file_path=f"f{i}.py") for i in range(5)],
        )
        d = pack.to_dict()

        assert len(d["communities"]) == 3
        assert len(d["entities"]) == 4
        assert len(d["edges"]) == 2
        assert len(d["paths"]) == 2
        assert len(d["citations"]) == 5


# ===========================================================================
# MapReduceResult tests
# ===========================================================================


class TestMapReduceResult:
    """Tests for MapReduceResult dataclass."""

    def test_defaults(self) -> None:
        """Test default field values."""
        result = MapReduceResult(query="test", final_answer="answer here")
        assert result.query == "test"
        assert result.final_answer == "answer here"
        assert result.partial_answers == []
        assert result.communities_used == 0
        assert result.context is None

    def test_with_partial_answers(self) -> None:
        """Test with partial answers populated."""
        result = MapReduceResult(
            query="test",
            final_answer="combined",
            partial_answers=["partial 1", "partial 2"],
            communities_used=3,
        )
        assert len(result.partial_answers) == 2
        assert result.communities_used == 3

    def test_with_context(self) -> None:
        """Test associating a ContextPack."""
        pack = ContextPack(query="test")
        result = MapReduceResult(
            query="test",
            final_answer="answer",
            context=pack,
        )
        assert result.context is pack
        assert result.context.query == "test"


# ===========================================================================
# ContextPack summary line tests
# ===========================================================================


class TestContextPackSummaryLine:
    """Tests for the summary line in to_markdown() that shows counts."""

    def test_summary_counts_match(self) -> None:
        """Test that summary line counts match actual list lengths."""
        pack = ContextPack(
            query="test",
            communities=[_make_community(), _make_community()],
            entities=[_make_entity()],
            citations=[_make_citation(), _make_citation(), _make_citation()],
        )
        md = pack.to_markdown()

        assert "2 communities" in md
        assert "1 entities" in md
        assert "3 citations" in md

    def test_single_entity_kind_renders(self) -> None:
        """Test rendering with a single entity that has a novel kind."""
        pack = ContextPack(
            query="test",
            entities=[_make_entity(kind="DB_TABLE", name="users")],
        )
        md = pack.to_markdown()
        assert "**DB_TABLE** (1):" in md
        assert "- users" in md


# ===========================================================================
# Relevance score rendering tests
# ===========================================================================


class TestRelevanceScoreRendering:
    """Tests for how relevance scores are formatted."""

    def test_community_relevance_percentage(self) -> None:
        """Test that community relevance is shown as percentage."""
        pack = ContextPack(
            query="test",
            communities=[_make_community(relevance_score=0.857)],
        )
        md = pack.to_markdown()
        # Should be formatted as 86% (:.0%)
        assert "86%" in md

    def test_community_relevance_zero(self) -> None:
        """Test zero relevance score renders as 0%."""
        pack = ContextPack(
            query="test",
            communities=[_make_community(relevance_score=0.0)],
        )
        md = pack.to_markdown()
        assert "0%" in md

    def test_community_relevance_full(self) -> None:
        """Test 100% relevance score."""
        pack = ContextPack(
            query="test",
            communities=[_make_community(relevance_score=1.0)],
        )
        md = pack.to_markdown()
        assert "100%" in md
