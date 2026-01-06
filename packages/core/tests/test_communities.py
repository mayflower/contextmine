"""Tests for Leiden-based community detection.

The community detection uses the Leiden algorithm via igraph/leidenalg
to cluster SEMANTIC_ENTITY nodes by their SEMANTIC_RELATIONSHIP edges.
"""

from uuid import uuid4

from contextmine_core.knowledge.communities import (
    Community,
    HierarchicalCommunities,
)


class TestCommunity:
    """Tests for Community dataclass."""

    def test_community_creation(self) -> None:
        """Test creating a community."""
        comm = Community(id=0, level=0, size=5)
        assert comm.id == 0
        assert comm.level == 0
        assert comm.size == 5
        assert comm.node_ids == []
        assert comm.node_keys == []

    def test_community_natural_key(self) -> None:
        """Test community natural key generation."""
        comm = Community(id=3, level=2, size=10)
        assert comm.natural_key == "community:L2:C3"

    def test_community_with_members(self) -> None:
        """Test community with member nodes."""
        node_id = uuid4()
        comm = Community(id=1, level=0, size=1)
        comm.node_ids = [node_id]
        comm.node_keys = ["entity:user_management"]

        assert len(comm.node_ids) == 1
        assert comm.node_ids[0] == node_id
        assert comm.node_keys[0] == "entity:user_management"


class TestHierarchicalCommunities:
    """Tests for HierarchicalCommunities result class."""

    def test_empty_result(self) -> None:
        """Test empty community result."""
        result = HierarchicalCommunities()
        assert result.community_count(0) == 0
        assert result.community_count(1) == 0
        assert result.total_communities() == 0

    def test_single_level_result(self) -> None:
        """Test result with single level."""
        result = HierarchicalCommunities()
        result.levels[0] = [
            Community(id=0, level=0, size=3),
            Community(id=1, level=0, size=2),
        ]
        result.modularity[0] = 0.75

        assert result.community_count(0) == 2
        assert result.community_count(1) == 0
        assert result.total_communities() == 2

    def test_hierarchical_result(self) -> None:
        """Test result with multiple levels."""
        result = HierarchicalCommunities()
        result.levels[0] = [
            Community(id=0, level=0, size=5),
            Community(id=1, level=0, size=3),
            Community(id=2, level=0, size=2),
        ]
        result.levels[1] = [
            Community(id=0, level=1, size=10),
        ]
        result.levels[2] = [
            Community(id=0, level=2, size=10),
        ]
        result.modularity[0] = 0.8
        result.modularity[1] = 0.6
        result.modularity[2] = 0.3

        assert result.community_count(0) == 3
        assert result.community_count(1) == 1
        assert result.community_count(2) == 1
        assert result.total_communities() == 5

    def test_get_community(self) -> None:
        """Test retrieving a specific community."""
        result = HierarchicalCommunities()
        comm0 = Community(id=0, level=0, size=5)
        comm1 = Community(id=1, level=0, size=3)
        result.levels[0] = [comm0, comm1]

        assert result.get_community(0, 0) == comm0
        assert result.get_community(0, 1) == comm1
        assert result.get_community(0, 2) is None
        assert result.get_community(1, 0) is None

    def test_node_membership(self) -> None:
        """Test node membership tracking."""
        result = HierarchicalCommunities()
        node_id = uuid4()

        result.node_membership[node_id] = {0: 1, 1: 0, 2: 0}

        assert result.node_membership[node_id][0] == 1  # Community 1 at level 0
        assert result.node_membership[node_id][1] == 0  # Community 0 at level 1
        assert result.node_membership[node_id][2] == 0  # Community 0 at level 2
