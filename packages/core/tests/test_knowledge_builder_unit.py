"""Unit tests for knowledge graph builder and related modules.

Tests pure logic that does not require a database:
- GraphBuildStats dataclass and to_dict()
- Community dataclass and HierarchicalCommunities methods
- Community title generation (_generate_title)
- Summary formatting (_format_llm_summary, _build_summary_prompt)
- CommunitySummaryOutput validation
- SummaryStats dataclass
- CommunityContext dataclass
"""

from uuid import uuid4

import pytest
from contextmine_core.knowledge.builder import GraphBuildStats
from contextmine_core.knowledge.communities import (
    Community,
    HierarchicalCommunities,
    _generate_title,
)
from contextmine_core.knowledge.summaries import (
    CommunityContext,
    CommunitySummaryOutput,
    SummaryStats,
    _build_summary_prompt,
    _format_llm_summary,
)
from pydantic import ValidationError

# ===========================================================================
# GraphBuildStats tests
# ===========================================================================


class TestGraphBuildStats:
    """Tests for GraphBuildStats dataclass."""

    def test_default_values(self) -> None:
        """Test that defaults are all zero / empty."""
        stats = GraphBuildStats()
        assert stats.file_nodes_created == 0
        assert stats.symbol_nodes_created == 0
        assert stats.edges_created == 0
        assert stats.evidence_created == 0
        assert stats.nodes_deleted == 0
        assert stats.evidence_ids == []

    def test_to_dict_excludes_evidence_ids(self) -> None:
        """Test that to_dict returns numeric counts without evidence_ids."""
        stats = GraphBuildStats(
            file_nodes_created=3,
            symbol_nodes_created=10,
            edges_created=15,
            evidence_created=10,
            nodes_deleted=2,
            evidence_ids=["ev1", "ev2"],
        )
        d = stats.to_dict()

        assert d == {
            "file_nodes_created": 3,
            "symbol_nodes_created": 10,
            "edges_created": 15,
            "evidence_created": 10,
            "nodes_deleted": 2,
        }
        # evidence_ids should NOT be in the dict
        assert "evidence_ids" not in d

    def test_to_dict_empty_stats(self) -> None:
        """Test to_dict with default (zero) stats."""
        d = GraphBuildStats().to_dict()
        assert all(v == 0 for v in d.values())
        assert len(d) == 5

    def test_evidence_ids_is_mutable_list(self) -> None:
        """Test that evidence_ids list can be appended to."""
        stats = GraphBuildStats()
        stats.evidence_ids.append("ev-001")
        stats.evidence_ids.append("ev-002")
        assert len(stats.evidence_ids) == 2

    def test_evidence_ids_not_shared_between_instances(self) -> None:
        """Test that separate instances get separate evidence_ids lists."""
        stats1 = GraphBuildStats()
        stats2 = GraphBuildStats()
        stats1.evidence_ids.append("only-in-1")
        assert stats2.evidence_ids == []


# ===========================================================================
# Community title generation tests
# ===========================================================================


class TestGenerateTitle:
    """Tests for _generate_title helper function."""

    def test_empty_keys(self) -> None:
        """Test with no node keys."""
        title = _generate_title([], 5)
        assert title == "Community (5 members)"

    def test_single_entity_key(self) -> None:
        """Test with a single entity key."""
        title = _generate_title(["entity:user_management"], 1)
        assert "User Management" in title

    def test_multiple_entity_keys(self) -> None:
        """Test with multiple entity keys, showing up to 3."""
        keys = [
            "entity:user_authentication",
            "entity:session_handling",
            "entity:token_validation",
        ]
        title = _generate_title(keys, 3)
        assert "User Authentication" in title
        assert "Session Handling" in title
        assert "Token Validation" in title

    def test_more_than_3_entity_keys(self) -> None:
        """Test that title shows '+N more' when size > 3."""
        keys = [
            "entity:alpha",
            "entity:beta",
            "entity:gamma",
            "entity:delta",
        ]
        title = _generate_title(keys, 10)
        assert "+7 more" in title

    def test_exactly_3_entities_no_more(self) -> None:
        """Test that size <= 3 does not show '+N more'."""
        keys = [
            "entity:alpha",
            "entity:beta",
            "entity:gamma",
        ]
        title = _generate_title(keys, 3)
        assert "+" not in title

    def test_non_entity_keys_fallback(self) -> None:
        """Test with non-entity keys that do not match the entity: prefix."""
        # Keys without 'entity:' prefix do get split but produce empty names
        # because the prefix is not "entity"
        keys = ["file:src/main.py", "symbol:hello"]
        title = _generate_title(keys, 2)
        # Since these are not "entity:" prefixed, names list stays empty
        assert title == "Community (2 members)"

    def test_snake_case_to_title_case(self) -> None:
        """Test that entity keys with snake_case are converted to Title Case."""
        title = _generate_title(["entity:data_access_layer"], 1)
        assert "Data Access Layer" in title

    def test_deduplicated_names(self) -> None:
        """Test that duplicate entity names are not repeated."""
        keys = [
            "entity:user_auth",
            "entity:user_auth",  # duplicate
            "entity:session",
        ]
        title = _generate_title(keys, 3)
        # "User Auth" should appear only once
        count = title.count("User Auth")
        assert count == 1

    def test_empty_entity_value(self) -> None:
        """Test with entity key that has empty value after prefix."""
        # "entity:" with empty value -> name becomes "" after replace/title
        # The code checks `if name and name not in names`
        title = _generate_title(["entity:"], 1)
        # Empty name is falsy, so falls through to fallback
        assert title == "Community (1 members)"


# ===========================================================================
# HierarchicalCommunities additional tests
# ===========================================================================


class TestHierarchicalCommunitiesExtended:
    """Extended tests for HierarchicalCommunities beyond test_communities.py."""

    def test_get_community_returns_correct_instance(self) -> None:
        """Test that get_community returns the exact community object."""
        result = HierarchicalCommunities()
        comm_a = Community(id=0, level=0, size=5)
        comm_b = Community(id=1, level=0, size=3)
        result.levels[0] = [comm_a, comm_b]

        found = result.get_community(0, 0)
        assert found is comm_a

        found = result.get_community(0, 1)
        assert found is comm_b

    def test_get_community_nonexistent_level(self) -> None:
        """Test get_community with a level that does not exist."""
        result = HierarchicalCommunities()
        result.levels[0] = [Community(id=0, level=0, size=5)]

        assert result.get_community(99, 0) is None

    def test_get_community_nonexistent_id(self) -> None:
        """Test get_community with a community id that does not exist."""
        result = HierarchicalCommunities()
        result.levels[0] = [Community(id=0, level=0, size=5)]

        assert result.get_community(0, 999) is None

    def test_total_communities_across_levels(self) -> None:
        """Test total count across multiple levels."""
        result = HierarchicalCommunities()
        result.levels[0] = [Community(id=i, level=0, size=1) for i in range(5)]
        result.levels[1] = [Community(id=i, level=1, size=1) for i in range(3)]
        result.levels[2] = [Community(id=0, level=2, size=1)]

        assert result.total_communities() == 9

    def test_node_membership_multi_level(self) -> None:
        """Test node membership across multiple levels."""
        result = HierarchicalCommunities()
        node_id = uuid4()

        result.node_membership[node_id] = {0: 2, 1: 1, 2: 0}

        assert result.node_membership[node_id][0] == 2
        assert result.node_membership[node_id][1] == 1
        assert result.node_membership[node_id][2] == 0

    def test_community_natural_key_format(self) -> None:
        """Test that natural_key follows the expected format."""
        comm = Community(id=7, level=3, size=15)
        assert comm.natural_key == "community:L3:C7"


# ===========================================================================
# CommunitySummaryOutput tests
# ===========================================================================


class TestCommunitySummaryOutput:
    """Tests for CommunitySummaryOutput Pydantic model."""

    def test_valid_creation(self) -> None:
        """Test creating a valid summary output."""
        summary = CommunitySummaryOutput(
            title="Auth Module",
            responsibilities=["Handle login", "Manage sessions"],
            key_concepts=["JWT", "OAuth"],
            key_dependencies=["redis"],
            key_paths=["src/auth/"],
            confidence=0.85,
        )
        assert summary.title == "Auth Module"
        assert len(summary.responsibilities) == 2
        assert summary.confidence == 0.85

    def test_defaults(self) -> None:
        """Test default values for optional fields."""
        summary = CommunitySummaryOutput(
            title="Test",
            responsibilities=["Do stuff"],
            key_concepts=["Thing"],
            confidence=0.5,
        )
        assert summary.key_dependencies == []
        assert summary.key_paths == []

    def test_confidence_bounds(self) -> None:
        """Test that confidence must be between 0 and 1."""
        # Valid boundaries
        CommunitySummaryOutput(
            title="A", responsibilities=["B"], key_concepts=["C"], confidence=0.0
        )
        CommunitySummaryOutput(
            title="A", responsibilities=["B"], key_concepts=["C"], confidence=1.0
        )

        # Out of bounds
        with pytest.raises(ValidationError):
            CommunitySummaryOutput(
                title="A", responsibilities=["B"], key_concepts=["C"], confidence=1.5
            )
        with pytest.raises(ValidationError):
            CommunitySummaryOutput(
                title="A", responsibilities=["B"], key_concepts=["C"], confidence=-0.1
            )


# ===========================================================================
# SummaryStats tests
# ===========================================================================


class TestSummaryStats:
    """Tests for SummaryStats dataclass."""

    def test_defaults(self) -> None:
        """Test default values."""
        stats = SummaryStats()
        assert stats.communities_summarized == 0
        assert stats.communities_skipped == 0
        assert stats.embeddings_created == 0
        assert stats.embeddings_skipped == 0

    def test_increment(self) -> None:
        """Test incrementing counters."""
        stats = SummaryStats()
        stats.communities_summarized += 1
        stats.embeddings_created += 1
        stats.embeddings_skipped += 2

        assert stats.communities_summarized == 1
        assert stats.embeddings_created == 1
        assert stats.embeddings_skipped == 2


# ===========================================================================
# _format_llm_summary tests
# ===========================================================================


class TestFormatLlmSummary:
    """Tests for _format_llm_summary helper."""

    def test_full_summary(self) -> None:
        """Test formatting a complete summary."""
        summary = CommunitySummaryOutput(
            title="Database Layer",
            responsibilities=["Manage connections", "Execute queries"],
            key_concepts=["Connection pool", "ORM"],
            key_dependencies=["SQLAlchemy", "asyncpg"],
            key_paths=["packages/core/models.py"],
            confidence=0.9,
        )
        text = _format_llm_summary(summary)

        assert text.startswith("# Database Layer")
        assert "## Responsibilities" in text
        assert "- Manage connections" in text
        assert "- Execute queries" in text
        assert "## Key Concepts" in text
        assert "- Connection pool" in text
        assert "## Dependencies" in text
        assert "- SQLAlchemy" in text
        assert "## Key Paths" in text
        assert "- packages/core/models.py" in text
        assert "Confidence: 90%" in text

    def test_minimal_summary(self) -> None:
        """Test formatting with only required fields."""
        summary = CommunitySummaryOutput(
            title="Minimal",
            responsibilities=["One thing"],
            key_concepts=["One concept"],
            confidence=0.5,
        )
        text = _format_llm_summary(summary)

        assert "# Minimal" in text
        assert "## Responsibilities" in text
        assert "## Key Concepts" in text
        assert "## Dependencies" not in text  # Empty list
        assert "## Key Paths" not in text  # Empty list
        assert "Confidence: 50%" in text

    def test_empty_optional_sections_omitted(self) -> None:
        """Test that empty dependencies and key_paths sections are omitted."""
        summary = CommunitySummaryOutput(
            title="No Deps",
            responsibilities=["Something"],
            key_concepts=["Something else"],
            key_dependencies=[],
            key_paths=[],
            confidence=0.7,
        )
        text = _format_llm_summary(summary)

        assert "## Dependencies" not in text
        assert "## Key Paths" not in text

    def test_confidence_formatting(self) -> None:
        """Test various confidence values are formatted as percentages."""
        for conf, expected in [(0.0, "0%"), (0.333, "33%"), (1.0, "100%")]:
            summary = CommunitySummaryOutput(
                title="T",
                responsibilities=["R"],
                key_concepts=["C"],
                confidence=conf,
            )
            text = _format_llm_summary(summary)
            assert f"Confidence: {expected}" in text


# ===========================================================================
# _build_summary_prompt tests
# ===========================================================================


class TestBuildSummaryPrompt:
    """Tests for _build_summary_prompt helper."""

    def test_empty_context(self) -> None:
        """Test prompt with empty context."""
        context = CommunityContext(
            community_id=uuid4(),
            level=0,
        )
        prompt = _build_summary_prompt(context)

        assert "Summarize this software component" in prompt
        assert "Level: 0" in prompt
        assert "Member count: 0" in prompt

    def test_with_member_nodes(self) -> None:
        """Test prompt includes member node information."""
        context = CommunityContext(
            community_id=uuid4(),
            level=1,
            member_nodes=[
                {
                    "name": "User Authentication",
                    "type": "concept",
                    "description": "Handles user login flows",
                },
                {
                    "name": "Session Manager",
                    "type": "component",
                    "description": "",
                },
            ],
        )
        prompt = _build_summary_prompt(context)

        assert "## Domain Concepts (Semantic Entities)" in prompt
        assert "**User Authentication** (concept)" in prompt
        assert "Handles user login flows" in prompt
        assert "**Session Manager** (component)" in prompt

    def test_member_count_uses_full_community_size(self) -> None:
        """Test prompt shows stored community size, not just sampled members."""
        context = CommunityContext(
            community_id=uuid4(),
            level=0,
            member_count=24,
            member_nodes=[
                {"name": "Authentication", "type": "concept", "description": ""},
                {"name": "Session", "type": "component", "description": ""},
            ],
        )
        prompt = _build_summary_prompt(context)

        assert "Member count: 24" in prompt

    def test_with_entity_types(self) -> None:
        """Test prompt includes entity type breakdown."""
        context = CommunityContext(
            community_id=uuid4(),
            level=0,
            entity_types={"concept": 3, "component": 1},
        )
        prompt = _build_summary_prompt(context)

        assert "## Entity Types" in prompt
        assert "3 concept" in prompt
        assert "1 component" in prompt

    def test_with_source_symbols(self) -> None:
        """Test prompt includes associated code symbols."""
        context = CommunityContext(
            community_id=uuid4(),
            level=0,
            source_symbols=["auth.login()", "session.create()"],
        )
        prompt = _build_summary_prompt(context)

        assert "## Associated Code Symbols" in prompt
        assert "- auth.login()" in prompt
        assert "- session.create()" in prompt

    def test_with_source_files(self) -> None:
        """Test prompt includes source files for key path generation."""
        context = CommunityContext(
            community_id=uuid4(),
            level=0,
            source_files=["src/auth.py", "src/session/store.py"],
        )
        prompt = _build_summary_prompt(context)

        assert "## Source Files" in prompt
        assert "- src/auth.py" in prompt
        assert "- src/session/store.py" in prompt

    def test_with_entity_descriptions(self) -> None:
        """Test prompt includes entity descriptions section."""
        context = CommunityContext(
            community_id=uuid4(),
            level=0,
            entity_descriptions=[
                "Manages user identity verification",
                "Stores session tokens in Redis",
            ],
        )
        prompt = _build_summary_prompt(context)

        assert "## Entity Descriptions" in prompt
        assert "- Manages user identity verification" in prompt
        assert "- Stores session tokens in Redis" in prompt

    def test_with_evidence_snippets(self) -> None:
        """Test prompt includes code snippets."""
        context = CommunityContext(
            community_id=uuid4(),
            level=0,
            evidence_snippets=["def login(user, password): ..."],
        )
        prompt = _build_summary_prompt(context)

        assert "## Code Snippets" in prompt
        assert "def login(user, password): ..." in prompt

    def test_member_nodes_limited_to_10(self) -> None:
        """Test that only first 10 member nodes are included."""
        nodes = [{"name": f"Entity_{i}", "type": "concept", "description": ""} for i in range(15)]
        context = CommunityContext(
            community_id=uuid4(),
            level=0,
            member_nodes=nodes,
        )
        prompt = _build_summary_prompt(context)

        assert "Entity_0" in prompt
        assert "Entity_9" in prompt
        assert "Entity_10" not in prompt

    def test_source_symbols_limited_to_10(self) -> None:
        """Test that only first 10 source symbols are included."""
        symbols = [f"symbol_{i}()" for i in range(15)]
        context = CommunityContext(
            community_id=uuid4(),
            level=0,
            source_symbols=symbols,
        )
        prompt = _build_summary_prompt(context)

        assert "symbol_0()" in prompt
        assert "symbol_9()" in prompt
        assert "symbol_10()" not in prompt

    def test_entity_descriptions_limited_to_5(self) -> None:
        """Test that only first 5 entity descriptions are included."""
        descriptions = [f"Description {i}" for i in range(8)]
        context = CommunityContext(
            community_id=uuid4(),
            level=0,
            entity_descriptions=descriptions,
        )
        prompt = _build_summary_prompt(context)

        assert "Description 0" in prompt
        assert "Description 4" in prompt
        assert "Description 5" not in prompt

    def test_evidence_snippets_limited_to_3(self) -> None:
        """Test that only first 3 evidence snippets are included."""
        snippets = [f"snippet_{i}" for i in range(5)]
        context = CommunityContext(
            community_id=uuid4(),
            level=0,
            evidence_snippets=snippets,
        )
        prompt = _build_summary_prompt(context)

        assert "snippet_0" in prompt
        assert "snippet_2" in prompt
        assert "snippet_3" not in prompt

    def test_evidence_snippet_truncated_at_200(self) -> None:
        """Test that individual evidence snippets are truncated at 200 chars."""
        long_snippet = "x" * 300
        context = CommunityContext(
            community_id=uuid4(),
            level=0,
            evidence_snippets=[long_snippet],
        )
        prompt = _build_summary_prompt(context)

        # The snippet in the prompt should be truncated to 200 chars
        # The format is ```\nsnippet[:200]\n```
        assert "x" * 200 in prompt
        # 201 consecutive x's should not appear
        assert "x" * 201 not in prompt


# ===========================================================================
# CommunityContext dataclass tests
# ===========================================================================


class TestCommunityContextDataclass:
    """Tests for CommunityContext dataclass in summaries module."""

    def test_defaults(self) -> None:
        """Test default field values."""
        ctx = CommunityContext(
            community_id=uuid4(),
            level=0,
        )
        assert ctx.member_nodes == []
        assert ctx.evidence_snippets == []
        assert ctx.member_count == 0
        assert ctx.entity_names == []
        assert ctx.entity_types == {}
        assert ctx.entity_descriptions == []
        assert ctx.source_symbols == []
        assert ctx.source_files == []

    def test_not_shared_between_instances(self) -> None:
        """Test that mutable defaults are not shared."""
        ctx1 = CommunityContext(community_id=uuid4(), level=0)
        ctx2 = CommunityContext(community_id=uuid4(), level=0)

        ctx1.entity_names.append("only_in_ctx1")
        assert ctx2.entity_names == []

        ctx1.member_nodes.append({"name": "test"})
        assert ctx2.member_nodes == []


# ===========================================================================
# generate_community_summaries validation tests
# ===========================================================================


class TestGenerateCommunitySummariesValidation:
    """Tests for input validation in generate_community_summaries."""

    @pytest.mark.anyio
    async def test_raises_without_llm_provider(self) -> None:
        """Test that None LLM provider raises ValueError."""
        from unittest.mock import AsyncMock

        from contextmine_core.knowledge.summaries import generate_community_summaries

        session = AsyncMock()
        embed_provider = AsyncMock()

        with pytest.raises(ValueError, match="LLM provider is required"):
            await generate_community_summaries(
                session=session,
                collection_id=uuid4(),
                provider=None,
                embed_provider=embed_provider,
            )

    @pytest.mark.anyio
    async def test_raises_without_embed_provider(self) -> None:
        """Test that None embedding provider raises ValueError."""
        from unittest.mock import AsyncMock

        from contextmine_core.knowledge.summaries import generate_community_summaries

        session = AsyncMock()
        llm_provider = AsyncMock()

        with pytest.raises(ValueError, match="Embedding provider is required"):
            await generate_community_summaries(
                session=session,
                collection_id=uuid4(),
                provider=llm_provider,
                embed_provider=None,
            )
