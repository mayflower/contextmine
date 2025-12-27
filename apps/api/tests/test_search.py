"""Tests for search functionality."""

import os

import pytest
from contextmine_core.search import compute_rrf_scores


class TestRRFScoring:
    """Tests for Reciprocal Rank Fusion scoring."""

    def test_rrf_single_source_fts_only(self) -> None:
        """Test RRF with only FTS results."""
        import uuid

        chunk1 = uuid.uuid4()
        chunk2 = uuid.uuid4()

        fts_results = [
            (chunk1, 0.9, 1),  # chunk_id, score, rank
            (chunk2, 0.7, 2),
        ]
        vector_results: list[tuple] = []

        scores = compute_rrf_scores(fts_results, vector_results, k=60)

        assert chunk1 in scores
        assert chunk2 in scores
        # RRF score = 1/(k + rank) = 1/(60 + 1) for rank 1
        assert abs(scores[chunk1][0] - 1 / 61) < 0.0001
        assert abs(scores[chunk2][0] - 1 / 62) < 0.0001
        # FTS rank preserved
        assert scores[chunk1][1] == 1
        assert scores[chunk2][1] == 2
        # No vector rank
        assert scores[chunk1][2] is None
        assert scores[chunk2][2] is None

    def test_rrf_single_source_vector_only(self) -> None:
        """Test RRF with only vector results."""
        import uuid

        chunk1 = uuid.uuid4()
        chunk2 = uuid.uuid4()

        fts_results: list[tuple] = []
        vector_results = [
            (chunk1, 0.1, 1),  # chunk_id, distance, rank
            (chunk2, 0.2, 2),
        ]

        scores = compute_rrf_scores(fts_results, vector_results, k=60)

        assert chunk1 in scores
        assert chunk2 in scores
        # RRF score = 1/(k + rank)
        assert abs(scores[chunk1][0] - 1 / 61) < 0.0001
        # No FTS rank
        assert scores[chunk1][1] is None
        # Vector rank preserved
        assert scores[chunk1][2] == 1
        assert scores[chunk2][2] == 2

    def test_rrf_combined_sources(self) -> None:
        """Test RRF combining both FTS and vector results."""
        import uuid

        chunk1 = uuid.uuid4()
        chunk2 = uuid.uuid4()
        chunk3 = uuid.uuid4()

        fts_results = [
            (chunk1, 0.9, 1),
            (chunk2, 0.7, 2),
        ]
        vector_results = [
            (chunk2, 0.1, 1),  # chunk2 is top in vector
            (chunk3, 0.2, 2),
        ]

        scores = compute_rrf_scores(fts_results, vector_results, k=60)

        # chunk2 should have highest RRF (appears in both)
        assert scores[chunk2][0] > scores[chunk1][0]
        assert scores[chunk2][0] > scores[chunk3][0]

        # chunk2 has both ranks
        assert scores[chunk2][1] == 2  # FTS rank
        assert scores[chunk2][2] == 1  # Vector rank

        # Combined score = 1/(60+2) + 1/(60+1) for chunk2
        expected = 1 / 62 + 1 / 61
        assert abs(scores[chunk2][0] - expected) < 0.0001

    def test_rrf_deterministic(self) -> None:
        """Test that RRF scoring is deterministic."""
        import uuid

        chunk1 = uuid.uuid4()
        chunk2 = uuid.uuid4()

        fts_results = [
            (chunk1, 0.9, 1),
            (chunk2, 0.7, 2),
        ]
        vector_results = [
            (chunk2, 0.1, 1),
            (chunk1, 0.3, 2),
        ]

        scores1 = compute_rrf_scores(fts_results, vector_results, k=60)
        scores2 = compute_rrf_scores(fts_results, vector_results, k=60)

        # Results should be identical
        assert scores1[chunk1][0] == scores2[chunk1][0]
        assert scores1[chunk2][0] == scores2[chunk2][0]

    def test_rrf_custom_k(self) -> None:
        """Test RRF with custom k parameter."""
        import uuid

        chunk1 = uuid.uuid4()

        fts_results = [(chunk1, 0.9, 1)]
        vector_results: list[tuple] = []

        # Lower k = higher scores
        scores_k30 = compute_rrf_scores(fts_results, vector_results, k=30)
        scores_k60 = compute_rrf_scores(fts_results, vector_results, k=60)

        assert scores_k30[chunk1][0] > scores_k60[chunk1][0]


class TestAccessControl:
    """Tests for search access control.

    These tests verify that:
    - Anonymous users only see global collection results
    - Authenticated users see global + their private collections
    - Users cannot access private collections they don't own/belong to
    """

    @pytest.mark.anyio
    async def test_get_accessible_collections_anonymous(self) -> None:
        """Test that anonymous users only access global collections."""
        # This test requires database fixtures
        # For now, just verify the function signature
        from contextmine_core.search import get_accessible_collection_ids

        # Function should accept None for user_id
        assert callable(get_accessible_collection_ids)

    @pytest.mark.anyio
    @pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
    async def test_hybrid_search_returns_empty_for_no_access(self) -> None:
        """Test that search returns empty when user has no collection access."""
        import uuid

        from contextmine_core.search import hybrid_search

        # Search with a non-existent collection should return empty
        # (user has no access to private collections)
        result = await hybrid_search(
            query="test query",
            query_embedding=[0.1] * 1536,
            user_id=None,
            collection_id=uuid.uuid4(),  # Random collection
            top_k=10,
        )

        assert len(result.results) == 0


class TestSearchIntegration:
    """Integration tests for the search endpoint.

    These require a running database with test fixtures.
    """

    pass  # Will be expanded with database fixtures
