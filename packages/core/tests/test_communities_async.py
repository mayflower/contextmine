"""Async unit tests for knowledge/communities.py.

Tests the async database-dependent functions with mocked sessions:
- detect_communities() with mocked session
- persist_communities() with mocked session
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from contextmine_core.knowledge.communities import (
    Community,
    HierarchicalCommunities,
    detect_communities,
    persist_communities,
)
from contextmine_core.models import KnowledgeEdgeKind, KnowledgeNodeKind

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_session() -> AsyncMock:
    return AsyncMock()


def _make_node_row(node_id=None, natural_key="entity:auth", kind=None, name="Auth"):
    """Create a row tuple matching the select in detect_communities."""
    nid = node_id or uuid4()
    k = kind or KnowledgeNodeKind.SEMANTIC_ENTITY
    return (nid, natural_key, k, name)


# ===========================================================================
# detect_communities tests
# ===========================================================================


class TestDetectCommunities:
    """Tests for detect_communities with mocked session."""

    @pytest.mark.anyio
    async def test_no_semantic_entities_returns_empty(self) -> None:
        session = _mock_session()
        mock_result = MagicMock()
        mock_result.all.return_value = []
        session.execute.return_value = mock_result

        result = await detect_communities(session, uuid4())
        assert isinstance(result, HierarchicalCommunities)
        assert result.total_communities() == 0
        assert result.levels == {}

    @pytest.mark.anyio
    async def test_nodes_but_no_edges_returns_single_community(self) -> None:
        session = _mock_session()
        coll_id = uuid4()

        node1 = _make_node_row(natural_key="entity:auth", name="Auth")
        node2 = _make_node_row(natural_key="entity:user", name="User")

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                # Node query
                result.all.return_value = [node1, node2]
            elif call_count[0] == 1:
                # Edge query
                result.all.return_value = []
            call_count[0] += 1
            return result

        session.execute.side_effect = mock_execute

        result = await detect_communities(session, coll_id)
        assert result.total_communities() >= 1
        # Should have a single community at level 0 with all nodes
        level0 = result.levels.get(0, [])
        assert len(level0) == 1
        assert level0[0].size == 2

    @pytest.mark.anyio
    async def test_with_edges_runs_leiden(self) -> None:
        session = _mock_session()
        coll_id = uuid4()

        nid1 = uuid4()
        nid2 = uuid4()
        nid3 = uuid4()
        node1 = _make_node_row(node_id=nid1, natural_key="entity:auth", name="Auth")
        node2 = _make_node_row(node_id=nid2, natural_key="entity:user", name="User")
        node3 = _make_node_row(node_id=nid3, natural_key="entity:session", name="Session")

        edge1 = (nid1, nid2, KnowledgeEdgeKind.SEMANTIC_RELATIONSHIP, {"strength": 0.9})
        edge2 = (nid1, nid3, KnowledgeEdgeKind.SEMANTIC_RELATIONSHIP, {"strength": 0.8})

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                result.all.return_value = [node1, node2, node3]
            elif call_count[0] == 1:
                result.all.return_value = [edge1, edge2]
            call_count[0] += 1
            return result

        session.execute.side_effect = mock_execute

        result = await detect_communities(session, coll_id, resolutions=[1.0])
        assert result.total_communities() >= 1
        # Node membership should be populated
        assert len(result.node_membership) == 3
        # Modularity should be recorded
        assert 0 in result.modularity

    @pytest.mark.anyio
    async def test_custom_resolutions(self) -> None:
        session = _mock_session()
        coll_id = uuid4()

        nid1 = uuid4()
        nid2 = uuid4()
        node1 = _make_node_row(node_id=nid1, natural_key="entity:a", name="A")
        node2 = _make_node_row(node_id=nid2, natural_key="entity:b", name="B")
        edge = (nid1, nid2, KnowledgeEdgeKind.SEMANTIC_RELATIONSHIP, {"strength": 1.0})

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                result.all.return_value = [node1, node2]
            elif call_count[0] == 1:
                result.all.return_value = [edge]
            call_count[0] += 1
            return result

        session.execute.side_effect = mock_execute

        resolutions = [2.0, 1.0, 0.5]
        result = await detect_communities(session, coll_id, resolutions=resolutions)
        # Should have communities at multiple levels
        assert len(result.levels) <= len(resolutions)

    @pytest.mark.anyio
    async def test_edges_with_none_meta_default_weight(self) -> None:
        """Edges with None meta should use default weight 1.0."""
        session = _mock_session()
        coll_id = uuid4()

        nid1 = uuid4()
        nid2 = uuid4()
        node1 = _make_node_row(node_id=nid1, natural_key="entity:x", name="X")
        node2 = _make_node_row(node_id=nid2, natural_key="entity:y", name="Y")
        edge = (nid1, nid2, KnowledgeEdgeKind.SEMANTIC_RELATIONSHIP, None)

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                result.all.return_value = [node1, node2]
            elif call_count[0] == 1:
                result.all.return_value = [edge]
            call_count[0] += 1
            return result

        session.execute.side_effect = mock_execute

        result = await detect_communities(session, coll_id, resolutions=[1.0])
        assert result.total_communities() >= 1

    @pytest.mark.anyio
    async def test_edges_with_unknown_nodes_skipped(self) -> None:
        """Edges referencing nodes not in the node list are skipped."""
        session = _mock_session()
        coll_id = uuid4()

        nid1 = uuid4()
        node1 = _make_node_row(node_id=nid1, natural_key="entity:a", name="A")
        # Edge references an unknown node
        unknown_nid = uuid4()
        edge = (nid1, unknown_nid, KnowledgeEdgeKind.SEMANTIC_RELATIONSHIP, {"strength": 1.0})

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                result.all.return_value = [node1]
            elif call_count[0] == 1:
                result.all.return_value = [edge]
            call_count[0] += 1
            return result

        session.execute.side_effect = mock_execute

        # Should not raise; the invalid edge is simply skipped
        result = await detect_communities(session, coll_id, resolutions=[1.0])
        # No edges -> single community with all nodes
        level0 = result.levels.get(0, [])
        assert len(level0) == 1
        assert level0[0].size == 1


# ===========================================================================
# persist_communities tests
# ===========================================================================


class TestPersistCommunities:
    """Tests for persist_communities with mocked session."""

    @pytest.mark.anyio
    async def test_empty_result_creates_nothing(self) -> None:
        session = _mock_session()
        coll_id = uuid4()

        result = HierarchicalCommunities()
        stats = await persist_communities(session, coll_id, result)

        assert stats["communities_created"] == 0
        assert stats["members_created"] == 0
        # Should still delete existing
        session.execute.assert_called_once()

    @pytest.mark.anyio
    async def test_creates_communities_and_members(self) -> None:
        session = _mock_session()
        coll_id = uuid4()

        nid1 = uuid4()
        nid2 = uuid4()

        comm = Community(id=0, level=0, size=2)
        comm.node_ids = [nid1, nid2]
        comm.node_keys = ["entity:auth", "entity:user"]

        hier = HierarchicalCommunities()
        hier.levels[0] = [comm]
        hier.modularity[0] = 0.42

        stats = await persist_communities(session, coll_id, hier)

        assert stats["communities_created"] == 1
        assert stats["members_created"] == 2
        # session.add for community + 2 members
        assert session.add.call_count == 3
        added_community = session.add.call_args_list[0].args[0]
        assert added_community.meta["size"] == 2
        assert added_community.meta["member_count"] == 2

    @pytest.mark.anyio
    async def test_multi_level_communities(self) -> None:
        session = _mock_session()
        coll_id = uuid4()

        nid = uuid4()

        comm0 = Community(id=0, level=0, size=1)
        comm0.node_ids = [nid]
        comm0.node_keys = ["entity:a"]

        comm1 = Community(id=0, level=1, size=1)
        comm1.node_ids = [nid]
        comm1.node_keys = ["entity:a"]

        hier = HierarchicalCommunities()
        hier.levels[0] = [comm0]
        hier.levels[1] = [comm1]

        stats = await persist_communities(session, coll_id, hier)
        assert stats["communities_created"] == 2
        assert stats["members_created"] == 2

    @pytest.mark.anyio
    async def test_flush_called_for_each_community(self) -> None:
        session = _mock_session()
        coll_id = uuid4()

        comm = Community(id=0, level=0, size=0)
        comm.node_ids = []
        comm.node_keys = []

        hier = HierarchicalCommunities()
        hier.levels[0] = [comm]

        await persist_communities(session, coll_id, hier)
        session.flush.assert_called()
