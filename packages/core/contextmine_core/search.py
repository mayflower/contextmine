"""Hybrid search service with FTS + vector + RRF ranking."""

import uuid
from dataclasses import dataclass

from contextmine_core.database import get_session
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class SearchResult:
    """A single search result."""

    chunk_id: str
    document_id: str
    source_id: str
    collection_id: str
    content: str
    uri: str
    title: str
    score: float
    fts_rank: int | None  # Position in FTS results (1-indexed), None if not in FTS
    vector_rank: int | None  # Position in vector results (1-indexed), None if not in vector
    fts_score: float | None
    vector_score: float | None


@dataclass
class SearchResponse:
    """Response from search service."""

    results: list[SearchResult]
    query: str
    total_fts_matches: int
    total_vector_matches: int


async def get_accessible_collection_ids(
    session: AsyncSession,
    user_id: uuid.UUID | None,
) -> list[uuid.UUID]:
    """Get IDs of collections accessible to the user.

    Returns:
        List of collection IDs that are:
        - Global (visible to all)
        - Owned by the user
        - User is a member of
    """
    if user_id is None:
        # Anonymous user: only global collections
        result = await session.execute(
            text("""
                SELECT id FROM collections
                WHERE visibility = 'global'
            """)
        )
    else:
        result = await session.execute(
            text("""
                SELECT id FROM collections
                WHERE visibility = 'global'
                   OR owner_user_id = :user_id
                   OR id IN (
                       SELECT collection_id FROM collection_members
                       WHERE user_id = :user_id
                   )
            """),
            {"user_id": user_id},
        )

    return [row[0] for row in result.fetchall()]


async def search_fts(
    session: AsyncSession,
    query: str,
    collection_ids: list[uuid.UUID],
    limit: int = 50,
) -> list[tuple[uuid.UUID, float, int]]:
    """Perform full-text search over chunks.

    Returns:
        List of (chunk_id, ts_rank score, rank position) tuples
    """
    if not collection_ids:
        return []

    # Convert query to tsquery format
    result = await session.execute(
        text("""
            WITH ranked AS (
                SELECT
                    c.id as chunk_id,
                    ts_rank_cd(c.tsv, plainto_tsquery('english', :query)) as score
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                JOIN sources s ON d.source_id = s.id
                WHERE s.collection_id = ANY(:collection_ids)
                  AND c.tsv @@ plainto_tsquery('english', :query)
                ORDER BY score DESC
                LIMIT :limit
            )
            SELECT chunk_id, score, ROW_NUMBER() OVER (ORDER BY score DESC) as rank
            FROM ranked
        """),
        {
            "query": query,
            "collection_ids": collection_ids,
            "limit": limit,
        },
    )

    return [(row[0], float(row[1]), int(row[2])) for row in result.fetchall()]


async def search_vector(
    session: AsyncSession,
    query_embedding: list[float],
    collection_ids: list[uuid.UUID],
    limit: int = 50,
) -> list[tuple[uuid.UUID, float, int]]:
    """Perform vector similarity search over chunks.

    Returns:
        List of (chunk_id, cosine_distance, rank position) tuples
    """
    if not collection_ids:
        return []

    # Convert embedding to pgvector format
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    result = await session.execute(
        text("""
            WITH ranked AS (
                SELECT
                    c.id as chunk_id,
                    c.embedding <=> CAST(:embedding AS vector) as distance
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                JOIN sources s ON d.source_id = s.id
                WHERE s.collection_id = ANY(:collection_ids)
                  AND c.embedding IS NOT NULL
                ORDER BY distance ASC
                LIMIT :limit
            )
            SELECT chunk_id, distance, ROW_NUMBER() OVER (ORDER BY distance ASC) as rank
            FROM ranked
        """),
        {
            "embedding": embedding_str,
            "collection_ids": collection_ids,
            "limit": limit,
        },
    )

    return [(row[0], float(row[1]), int(row[2])) for row in result.fetchall()]


def compute_rrf_scores(
    fts_results: list[tuple[uuid.UUID, float, int]],
    vector_results: list[tuple[uuid.UUID, float, int]],
    k: int = 60,
) -> dict[uuid.UUID, tuple[float, int | None, int | None, float | None, float | None]]:
    """Compute Reciprocal Rank Fusion scores.

    RRF formula: score = sum(1 / (k + rank_i)) for each ranking system

    Args:
        fts_results: List of (chunk_id, score, rank) from FTS
        vector_results: List of (chunk_id, distance, rank) from vector search
        k: RRF constant (default 60, commonly used value)

    Returns:
        Dict mapping chunk_id to (rrf_score, fts_rank, vector_rank, fts_score, vector_score)
    """
    scores: dict[uuid.UUID, tuple[float, int | None, int | None, float | None, float | None]] = {}

    # Add FTS contributions
    for chunk_id, fts_score, rank in fts_results:
        rrf_contribution = 1.0 / (k + rank)
        scores[chunk_id] = (rrf_contribution, rank, None, fts_score, None)

    # Add vector contributions
    for chunk_id, vector_distance, rank in vector_results:
        rrf_contribution = 1.0 / (k + rank)
        # Convert distance to similarity score (1 - distance for cosine)
        vector_score = 1.0 - vector_distance

        if chunk_id in scores:
            existing = scores[chunk_id]
            scores[chunk_id] = (
                existing[0] + rrf_contribution,  # Combined RRF score
                existing[1],  # FTS rank
                rank,  # Vector rank
                existing[3],  # FTS score
                vector_score,  # Vector score
            )
        else:
            scores[chunk_id] = (rrf_contribution, None, rank, None, vector_score)

    return scores


async def get_chunk_details(
    session: AsyncSession,
    chunk_ids: list[uuid.UUID],
) -> dict[uuid.UUID, dict]:
    """Get detailed information for chunks.

    Returns:
        Dict mapping chunk_id to chunk details
    """
    if not chunk_ids:
        return {}

    result = await session.execute(
        text("""
            SELECT
                c.id as chunk_id,
                c.document_id,
                c.content,
                d.uri,
                d.title,
                d.source_id,
                s.collection_id
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            JOIN sources s ON d.source_id = s.id
            WHERE c.id = ANY(:chunk_ids)
        """),
        {"chunk_ids": chunk_ids},
    )

    return {
        row[0]: {
            "document_id": row[1],
            "content": row[2],
            "uri": row[3],
            "title": row[4],
            "source_id": row[5],
            "collection_id": row[6],
        }
        for row in result.fetchall()
    }


async def hybrid_search(
    query: str,
    query_embedding: list[float],
    user_id: uuid.UUID | None = None,
    collection_id: uuid.UUID | None = None,
    top_k: int = 20,
    fts_limit: int = 50,
    vector_limit: int = 50,
) -> SearchResponse:
    """Perform hybrid search combining FTS and vector search with RRF ranking.

    Args:
        query: The search query text
        query_embedding: The query embedding vector
        user_id: Optional user ID for access control
        collection_id: Optional collection ID to filter by
        top_k: Number of results to return
        fts_limit: Max FTS results to consider
        vector_limit: Max vector results to consider

    Returns:
        SearchResponse with ranked results
    """
    async with get_session() as session:
        # Get accessible collections
        if collection_id:
            # Verify access to specific collection
            accessible = await get_accessible_collection_ids(session, user_id)
            if collection_id not in accessible:
                # Return empty results if no access
                return SearchResponse(
                    results=[],
                    query=query,
                    total_fts_matches=0,
                    total_vector_matches=0,
                )
            collection_ids = [collection_id]
        else:
            collection_ids = await get_accessible_collection_ids(session, user_id)

        if not collection_ids:
            return SearchResponse(
                results=[],
                query=query,
                total_fts_matches=0,
                total_vector_matches=0,
            )

        # Perform both searches
        fts_results = await search_fts(session, query, collection_ids, fts_limit)
        vector_results = await search_vector(session, query_embedding, collection_ids, vector_limit)

        # Compute RRF scores
        rrf_scores = compute_rrf_scores(fts_results, vector_results)

        # Sort by RRF score and take top_k
        sorted_ids = sorted(
            rrf_scores.keys(),
            key=lambda x: rrf_scores[x][0],
            reverse=True,
        )[:top_k]

        # Get chunk details
        chunk_details = await get_chunk_details(session, sorted_ids)

        # Build results
        results = []
        for chunk_id in sorted_ids:
            rrf_score, fts_rank, vector_rank, fts_score, vector_score = rrf_scores[chunk_id]
            details = chunk_details.get(chunk_id)

            if details:
                results.append(
                    SearchResult(
                        chunk_id=str(chunk_id),
                        document_id=str(details["document_id"]),
                        source_id=str(details["source_id"]),
                        collection_id=str(details["collection_id"]),
                        content=details["content"],
                        uri=details["uri"],
                        title=details["title"],
                        score=rrf_score,
                        fts_rank=fts_rank,
                        vector_rank=vector_rank,
                        fts_score=fts_score,
                        vector_score=vector_score,
                    )
                )

        return SearchResponse(
            results=results,
            query=query,
            total_fts_matches=len(fts_results),
            total_vector_matches=len(vector_results),
        )
