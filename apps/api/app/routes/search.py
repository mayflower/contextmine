"""Search routes for hybrid retrieval."""

import uuid

from contextmine_core import (
    get_settings,
    hybrid_search,
)
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.middleware import get_session

router = APIRouter(prefix="/search", tags=["search"])


class SearchRequest(BaseModel):
    """Request model for search."""

    query: str
    collection_id: str | None = None
    top_k: int = 20


class ChunkResult(BaseModel):
    """Response model for a single search result."""

    chunk_id: str
    document_id: str
    source_id: str
    collection_id: str
    content: str
    uri: str
    title: str
    score: float
    fts_rank: int | None
    vector_rank: int | None
    fts_score: float | None
    vector_score: float | None


class SearchResponse(BaseModel):
    """Response model for search results."""

    results: list[ChunkResult]
    query: str
    total_fts_matches: int
    total_vector_matches: int


def get_current_user_id(request: Request) -> uuid.UUID | None:
    """Get the current user ID from session, or None if not authenticated."""
    session = get_session(request)
    user_id = session.get("user_id")
    if not user_id:
        return None
    return uuid.UUID(user_id)


async def get_query_embedding(query: str, collection_id: str | None = None) -> list[float]:
    """Get embedding for query text.

    Uses FakeEmbedder for now; will be replaced with real embedder.
    """
    # Import here to avoid circular imports
    from contextmine_core import FakeEmbedder, get_embedder, parse_embedding_model_spec

    settings = get_settings()

    # If collection_id provided, could look up collection's embedding config
    # For now, use global default
    try:
        provider, model_name = parse_embedding_model_spec(settings.default_embedding_model)
        embedder = get_embedder(provider, model_name)
    except (ValueError, Exception):
        # Fall back to FakeEmbedder if no API key configured
        embedder = FakeEmbedder()

    result = await embedder.embed_batch([query])
    return result.embeddings[0]


@router.post("", response_model=SearchResponse)
async def search(request: Request, body: SearchRequest) -> SearchResponse:
    """Perform hybrid search over chunks.

    Combines FTS (full-text search) and vector similarity search
    using Reciprocal Rank Fusion (RRF).

    Access control: Returns results from global collections and
    private collections where the user is owner or member.
    """
    user_id = get_current_user_id(request)

    # Parse collection_id if provided
    collection_uuid: uuid.UUID | None = None
    if body.collection_id:
        try:
            collection_uuid = uuid.UUID(body.collection_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail="Invalid collection_id") from e

    # Get query embedding
    query_embedding = await get_query_embedding(body.query, body.collection_id)

    # Perform hybrid search
    response = await hybrid_search(
        query=body.query,
        query_embedding=query_embedding,
        user_id=user_id,
        collection_id=collection_uuid,
        top_k=body.top_k,
    )

    # Convert to response model
    results = [
        ChunkResult(
            chunk_id=r.chunk_id,
            document_id=r.document_id,
            source_id=r.source_id,
            collection_id=r.collection_id,
            content=r.content,
            uri=r.uri,
            title=r.title,
            score=r.score,
            fts_rank=r.fts_rank,
            vector_rank=r.vector_rank,
            fts_score=r.fts_score,
            vector_score=r.vector_score,
        )
        for r in response.results
    ]

    return SearchResponse(
        results=results,
        query=response.query,
        total_fts_matches=response.total_fts_matches,
        total_vector_matches=response.total_vector_matches,
    )
