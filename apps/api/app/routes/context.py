"""Context assembly routes for generating Markdown documents."""

import json
import logging
import uuid
from collections.abc import AsyncIterator

from contextmine_core import (
    LLMProvider,
    StreamingContextMetadata,
    assemble_context,
    assemble_context_stream,
)
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.middleware import get_session
from app.rate_limit import RATE_LIMIT_RESEARCH, RATE_LIMIT_SEARCH, limiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/context", tags=["context"])


class ContextRequest(BaseModel):
    """Request model for context assembly."""

    query: str
    collection_id: str | None = None
    max_chunks: int = 10
    max_tokens: int = 4000
    provider: str = "openai"  # openai, anthropic, gemini
    model: str | None = None


class SourceInfo(BaseModel):
    """Source information for a chunk."""

    uri: str
    title: str
    file_path: str | None = None


class ContextResponse(BaseModel):
    """Response model for context assembly."""

    markdown: str
    query: str
    chunks_used: int
    sources: list[SourceInfo]


def get_current_user_id(request: Request) -> uuid.UUID | None:
    """Get the current user ID from session, or None if not authenticated."""
    session = get_session(request)
    user_id = session.get("user_id")
    if not user_id:
        return None
    return uuid.UUID(user_id)


@router.post("", response_model=ContextResponse)
@limiter.limit(RATE_LIMIT_SEARCH)
async def create_context(request: Request, body: ContextRequest) -> ContextResponse:
    """Assemble a context document from retrieved chunks.

    This endpoint:
    1. Retrieves relevant chunks using hybrid search (FTS + vector)
    2. Assembles them into a prompt
    3. Uses an LLM to generate a coherent Markdown document
    4. Returns the document with source citations

    The generated document will:
    - Use ONLY information from the retrieved chunks
    - Preserve code fences exactly as they appear
    - Include a Sources section at the end
    """
    user_id = get_current_user_id(request)

    # Parse collection_id if provided
    collection_uuid: uuid.UUID | None = None
    if body.collection_id:
        try:
            collection_uuid = uuid.UUID(body.collection_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail="Invalid collection_id") from e

    # Parse provider
    try:
        provider = LLMProvider(body.provider)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail="Invalid provider. Must be one of: openai, anthropic, gemini",
        ) from e

    # Assemble context
    try:
        response = await assemble_context(
            query=body.query,
            user_id=user_id,
            collection_id=collection_uuid,
            max_chunks=body.max_chunks,
            max_tokens=body.max_tokens,
            provider=provider,
            model=body.model,
        )
    except Exception as e:
        logger.exception("Error assembling context: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Error assembling context. Please try again.",
        ) from e

    # Convert sources
    sources = [
        SourceInfo(
            uri=s["uri"],
            title=s["title"],
            file_path=s.get("file_path"),
        )
        for s in response.sources
    ]

    return ContextResponse(
        markdown=response.markdown,
        query=response.query,
        chunks_used=response.chunks_used,
        sources=sources,
    )


@router.post("/stream")
@limiter.limit(RATE_LIMIT_SEARCH)
async def create_context_stream(request: Request, body: ContextRequest) -> StreamingResponse:
    """Stream a context document using Server-Sent Events.

    This endpoint streams the LLM response as it's generated, providing
    a more responsive user experience for large responses.

    The stream sends:
    1. First event: metadata (query, chunks_used, sources)
    2. Subsequent events: text chunks as they're generated
    3. Final event: [DONE] marker

    Event format: SSE with JSON data
    - event: metadata | content | done
    - data: JSON payload
    """
    user_id = get_current_user_id(request)

    # Parse collection_id if provided
    collection_uuid: uuid.UUID | None = None
    if body.collection_id:
        try:
            collection_uuid = uuid.UUID(body.collection_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail="Invalid collection_id") from e

    # Parse provider
    try:
        provider = LLMProvider(body.provider)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail="Invalid provider. Must be one of: openai, anthropic, gemini",
        ) from e

    async def event_generator():
        """Generate SSE events."""
        try:
            async for item in assemble_context_stream(
                query=body.query,
                user_id=user_id,
                collection_id=collection_uuid,
                max_chunks=body.max_chunks,
                max_tokens=body.max_tokens,
                provider=provider,
                model=body.model,
            ):
                if isinstance(item, StreamingContextMetadata):
                    # Send metadata event
                    data = json.dumps(
                        {
                            "query": item.query,
                            "chunks_used": item.chunks_used,
                            "sources": item.sources,
                        }
                    )
                    yield f"event: metadata\ndata: {data}\n\n"
                else:
                    # Send content chunk
                    data = json.dumps({"text": item})
                    yield f"event: content\ndata: {data}\n\n"

            # Send done event
            yield "event: done\ndata: {}\n\n"

        except Exception as e:
            # Send error event - log details server-side, return generic message
            logger.exception("Error during context streaming: %s", e)
            data = json.dumps({"error": "An error occurred while generating context"})
            yield f"event: error\ndata: {data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


class ResearchRequest(BaseModel):
    """Request model for deep research."""

    question: str
    scope: str | None = None  # Path pattern like 'src/api/**'
    budget: int = 10  # Max investigation steps


class ResearchStepInfo(BaseModel):
    """Information about a research step."""

    action: str
    description: str


class ResearchResponse(BaseModel):
    """Response model for deep research."""

    answer: str
    citations: list[str]
    steps_used: int
    run_id: str | None = None


@router.post("/research/stream")
@limiter.limit(RATE_LIMIT_RESEARCH)
async def research_stream(request: Request, body: ResearchRequest) -> StreamingResponse:
    """Stream deep research results using Server-Sent Events.

    This endpoint runs a multi-step research agent that:
    1. Searches indexed documentation and code
    2. Reads relevant files
    3. Collects evidence
    4. Synthesizes an answer with citations

    The stream sends:
    - event: step - Current research action being performed
    - event: answer - Final answer
    - event: citations - List of citations
    - event: done - Completion marker
    - event: error - Error if something fails
    """
    # Validate budget
    budget = max(1, min(20, body.budget))

    async def event_generator() -> AsyncIterator[str]:
        """Generate SSE events for research progress."""
        try:
            from contextmine_core.research import run_research

            # Send start event
            yield f"event: step\ndata: {json.dumps({'action': 'starting', 'description': 'Initializing research agent...', 'step': 0})}\n\n"

            # Run research (this executes the full agent)
            result = await run_research(
                question=body.question,
                scope=body.scope,
                max_steps=budget,
            )

            # Send step summary
            yield f"event: step\ndata: {json.dumps({'action': 'completed', 'description': f'Research completed with {len(result.evidence)} evidence items', 'step': len(result.steps)})}\n\n"

            # Send answer
            if result.answer:
                yield f"event: answer\ndata: {json.dumps({'text': result.answer})}\n\n"

                # Extract citations from evidence
                citations = [
                    f"[{e.id}] {e.file_path}:{e.start_line}-{e.end_line}" for e in result.evidence
                ]
                yield f"event: citations\ndata: {json.dumps({'citations': citations, 'steps_used': len(result.steps), 'run_id': result.run_id})}\n\n"
            else:
                error_msg = (
                    result.error_message or "Research completed but no conclusive answer was found."
                )
                yield f"event: answer\ndata: {json.dumps({'text': error_msg})}\n\n"
                yield f"event: citations\ndata: {json.dumps({'citations': [], 'steps_used': len(result.steps), 'run_id': result.run_id})}\n\n"

            yield "event: done\ndata: {}\n\n"

        except ImportError as e:
            logger.error("Research agent not available: %s", e)
            yield f"event: error\ndata: {json.dumps({'error': 'Research agent is not available'})}\n\n"
        except Exception as e:
            logger.exception("Error during research: %s", e)
            yield f"event: error\ndata: {json.dumps({'error': 'An error occurred during research'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
