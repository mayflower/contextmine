"""Tests for the search module - pure functions and dataclasses.

Covers:
- SearchResult / SearchResponse dataclasses
- compute_rrf_scores (pure function, no DB needed)
- search_fts / search_vector with empty collection_ids (early return)
- get_chunk_details with empty chunk_ids (early return)
"""

from __future__ import annotations

import uuid

import pytest
from contextmine_core.search import (
    SearchResponse,
    SearchResult,
    compute_rrf_scores,
    get_chunk_details,
    search_fts,
    search_vector,
)

# ---------------------------------------------------------------------------
# compute_rrf_scores — pure function, no DB required
# ---------------------------------------------------------------------------


class TestComputeRRFScores:
    def test_fts_only(self) -> None:
        """FTS results only, no vector results."""
        cid = uuid.uuid4()
        fts = [(cid, 0.9, 1)]
        vector: list[tuple[uuid.UUID, float, int]] = []

        scores = compute_rrf_scores(fts, vector)

        assert cid in scores
        rrf_score, fts_rank, vector_rank, fts_score, vector_score = scores[cid]
        assert fts_rank == 1
        assert vector_rank is None
        assert fts_score == 0.9
        assert vector_score is None
        assert rrf_score == pytest.approx(1.0 / (60 + 1))

    def test_vector_only(self) -> None:
        """Vector results only, no FTS results."""
        cid = uuid.uuid4()
        fts: list[tuple[uuid.UUID, float, int]] = []
        vector = [(cid, 0.2, 1)]

        scores = compute_rrf_scores(fts, vector)

        assert cid in scores
        rrf_score, fts_rank, vector_rank, fts_score, vector_score = scores[cid]
        assert fts_rank is None
        assert vector_rank == 1
        assert fts_score is None
        # distance 0.2 -> similarity 0.8
        assert vector_score == pytest.approx(0.8)
        assert rrf_score == pytest.approx(1.0 / (60 + 1))

    def test_overlapping_chunk(self) -> None:
        """Chunk appears in both FTS and vector results."""
        cid = uuid.uuid4()
        fts = [(cid, 0.8, 2)]
        vector = [(cid, 0.1, 3)]

        scores = compute_rrf_scores(fts, vector)

        rrf_score, fts_rank, vector_rank, fts_score, vector_score = scores[cid]
        assert fts_rank == 2
        assert vector_rank == 3
        assert fts_score == 0.8
        assert vector_score == pytest.approx(0.9)
        expected = 1.0 / (60 + 2) + 1.0 / (60 + 3)
        assert rrf_score == pytest.approx(expected)

    def test_multiple_chunks_separate(self) -> None:
        """Multiple distinct chunks from different sources."""
        cid_a = uuid.uuid4()
        cid_b = uuid.uuid4()
        fts = [(cid_a, 0.9, 1)]
        vector = [(cid_b, 0.15, 1)]

        scores = compute_rrf_scores(fts, vector)

        assert len(scores) == 2
        assert cid_a in scores
        assert cid_b in scores

    def test_empty_inputs(self) -> None:
        scores = compute_rrf_scores([], [])
        assert scores == {}

    def test_custom_k(self) -> None:
        """Custom k parameter affects scores."""
        cid = uuid.uuid4()
        fts = [(cid, 0.9, 1)]

        scores_k30 = compute_rrf_scores(fts, [], k=30)
        scores_k60 = compute_rrf_scores(fts, [], k=60)

        # Smaller k means higher RRF contribution
        assert scores_k30[cid][0] > scores_k60[cid][0]

    def test_ranking_order(self) -> None:
        """Chunks appearing in both sources rank higher than single-source."""
        overlap = uuid.uuid4()
        fts_only = uuid.uuid4()
        vec_only = uuid.uuid4()

        fts = [(overlap, 0.8, 1), (fts_only, 0.7, 2)]
        vector = [(overlap, 0.1, 1), (vec_only, 0.3, 2)]

        scores = compute_rrf_scores(fts, vector)

        # Overlap should have the highest combined score
        assert scores[overlap][0] > scores[fts_only][0]
        assert scores[overlap][0] > scores[vec_only][0]


# ---------------------------------------------------------------------------
# Early returns for empty inputs (no DB needed)
# ---------------------------------------------------------------------------


class TestSearchFTSEmptyInputs:
    @pytest.mark.anyio
    async def test_empty_collection_ids(self) -> None:
        """search_fts returns empty list for no collections."""
        from unittest.mock import AsyncMock

        session = AsyncMock()
        result = await search_fts(session, "query", [])
        assert result == []
        session.execute.assert_not_called()


class TestSearchVectorEmptyInputs:
    @pytest.mark.anyio
    async def test_empty_collection_ids(self) -> None:
        """search_vector returns empty list for no collections."""
        from unittest.mock import AsyncMock

        session = AsyncMock()
        result = await search_vector(session, [0.1, 0.2], [])
        assert result == []
        session.execute.assert_not_called()


class TestGetChunkDetailsEmptyInputs:
    @pytest.mark.anyio
    async def test_empty_chunk_ids(self) -> None:
        """get_chunk_details returns empty dict for no chunks."""
        from unittest.mock import AsyncMock

        session = AsyncMock()
        result = await get_chunk_details(session, [])
        assert result == {}
        session.execute.assert_not_called()


# ---------------------------------------------------------------------------
# Dataclass structure
# ---------------------------------------------------------------------------


class TestSearchResult:
    def test_fields(self) -> None:
        sr = SearchResult(
            chunk_id="abc",
            document_id="doc1",
            source_id="src1",
            collection_id="col1",
            content="hello",
            uri="http://example.com",
            title="Title",
            score=0.5,
            fts_rank=1,
            vector_rank=None,
            fts_score=0.8,
            vector_score=None,
        )
        assert sr.chunk_id == "abc"
        assert sr.fts_rank == 1
        assert sr.vector_rank is None


class TestSearchResponse:
    def test_fields(self) -> None:
        sr = SearchResponse(
            results=[],
            query="test",
            total_fts_matches=5,
            total_vector_matches=3,
        )
        assert sr.query == "test"
        assert sr.total_fts_matches == 5
        assert len(sr.results) == 0
