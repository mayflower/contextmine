"""Tests for Leiden-based community detection."""

import igraph as ig
import leidenalg


class TestLeidenAlgorithm:
    """Tests that verify Leiden algorithm works correctly."""

    def test_leiden_basic(self) -> None:
        """Test basic Leiden community detection on a small graph."""
        # Create a simple graph with two clear communities
        # Community 1: nodes 0, 1, 2 (fully connected)
        # Community 2: nodes 3, 4, 5 (fully connected)
        # One edge between communities (2-3)
        edges = [
            (0, 1),
            (1, 2),
            (0, 2),  # Triangle 1
            (3, 4),
            (4, 5),
            (3, 5),  # Triangle 2
            (2, 3),  # Bridge between communities
        ]

        g = ig.Graph(n=6, edges=edges, directed=False)

        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            seed=42,
        )

        # Should find 2 communities
        assert len(set(partition.membership)) == 2

        # Nodes in same triangle should be in same community
        assert partition.membership[0] == partition.membership[1]
        assert partition.membership[1] == partition.membership[2]
        assert partition.membership[3] == partition.membership[4]
        assert partition.membership[4] == partition.membership[5]

        # Nodes in different triangles should be in different communities
        assert partition.membership[0] != partition.membership[3]

    def test_leiden_with_weights(self) -> None:
        """Test Leiden with edge weights."""
        # Same graph but with weighted edges
        edges = [
            (0, 1),
            (1, 2),
            (0, 2),
            (3, 4),
            (4, 5),
            (3, 5),
            (2, 3),
        ]
        weights = [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 0.1]  # Weak bridge

        g = ig.Graph(n=6, edges=edges, directed=False)
        g.es["weight"] = weights

        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            seed=42,
        )

        # Should still find 2 communities due to weak bridge
        assert len(set(partition.membership)) == 2

    def test_leiden_resolution_parameter(self) -> None:
        """Test that resolution parameter affects granularity."""
        # Create a larger graph with hierarchical structure
        # 4 cliques connected in a ring
        edges = []
        for clique in range(4):
            base = clique * 4
            # Fully connected clique
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append((base + i, base + j))
            # Connect to next clique
            next_clique = ((clique + 1) % 4) * 4
            edges.append((base + 3, next_clique))

        g = ig.Graph(n=16, edges=edges, directed=False)

        # High resolution = more communities
        high_res = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=2.0,
            seed=42,
        )

        # Low resolution = fewer communities
        low_res = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=0.1,
            seed=42,
        )

        # High resolution should find more communities than low
        high_communities = len(set(high_res.membership))
        low_communities = len(set(low_res.membership))

        assert high_communities >= low_communities

    def test_leiden_deterministic(self) -> None:
        """Test that Leiden with seed is deterministic."""
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
        g = ig.Graph(n=4, edges=edges, directed=False)

        results = []
        for _ in range(5):
            partition = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                seed=42,
            )
            results.append(tuple(partition.membership))

        # All results should be identical
        assert len(set(results)) == 1

    def test_leiden_modularity(self) -> None:
        """Test that Leiden optimizes modularity."""
        # Create graph with clear community structure
        edges = [
            (0, 1),
            (1, 2),
            (0, 2),  # Community 1
            (3, 4),
            (4, 5),
            (3, 5),  # Community 2
            (2, 3),  # Weak bridge
        ]
        g = ig.Graph(n=6, edges=edges, directed=False)

        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            seed=42,
        )

        # Modularity should be positive (good community structure)
        assert partition.modularity > 0

    def test_leiden_single_node(self) -> None:
        """Test Leiden with isolated nodes."""
        g = ig.Graph(n=3, edges=[(0, 1)], directed=False)

        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            seed=42,
        )

        # Should handle isolated node (node 2)
        assert len(partition.membership) == 3
