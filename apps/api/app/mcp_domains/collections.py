"""Collections and retrieval MCP tools."""

import uuid
from typing import Annotated, Literal

from contextmine_core import (
    Collection,
    Document,
    Source,
    accessible_collections_clause,
    assemble_context,
    get_settings,
    user_can_access_collection,
)
from contextmine_core import get_session as get_db_session
from contextmine_core.embeddings import FakeEmbedder, get_embedder, parse_embedding_model_spec
from contextmine_core.search import hybrid_search
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select

from app.mcp_auth import get_current_user_id


class CollectionItem(BaseModel):
    """Collection metadata returned to MCP clients."""

    id: str
    name: str
    slug: str
    visibility: str
    source_count: int


class CollectionListResult(BaseModel):
    """Structured collection listing."""

    collections: list[CollectionItem]
    total: int


class DocumentItem(BaseModel):
    """Document metadata returned to MCP clients."""

    id: str
    uri: str
    title: str
    source_url: str


class DocumentListResult(BaseModel):
    """Bounded structured document listing."""

    collection_id: str
    collection_name: str
    documents: list[DocumentItem]
    limit: int
    truncated: bool


class ContextSource(BaseModel):
    """Source cited by a retrieval result."""

    uri: str
    title: str
    file_path: str | None = None


class MarkdownResult(BaseModel):
    """Structured retrieval result with bounded Markdown content."""

    query: str
    collection_id: str | None
    mode: Literal["retrieval", "synthesis"]
    markdown: str
    chunks_used: int
    sources: list[ContextSource]
    offset: int


def escape_like_pattern(value: str) -> str:
    """Escape special characters in LIKE patterns."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


async def _require_collection(db, collection_id: str, user_id: uuid.UUID | None) -> Collection:
    try:
        collection_uuid = uuid.UUID(collection_id)
    except ValueError as exc:
        raise ToolError(f"Invalid collection_id: {collection_id}") from exc

    collection = (
        await db.execute(select(Collection).where(Collection.id == collection_uuid))
    ).scalar_one_or_none()
    if collection is None:
        raise ToolError("Collection not found.")
    if not await user_can_access_collection(db, collection, user_id):
        raise ToolError("Access denied to this collection.")
    return collection


async def _query_collections(*, user_id: uuid.UUID | None, search: str | None) -> list:
    async with get_db_session() as db:
        query = (
            select(
                Collection.id,
                Collection.name,
                Collection.slug,
                Collection.visibility,
                func.count(Source.id).label("source_count"),
            )
            .outerjoin(Source, Source.collection_id == Collection.id)
            .where(accessible_collections_clause(user_id))
        )
        if search:
            escaped_search = escape_like_pattern(search)
            query = query.where(
                or_(
                    Collection.name.ilike(f"%{escaped_search}%", escape="\\"),
                    Collection.slug.ilike(f"%{escaped_search}%", escape="\\"),
                )
            )
        result = await db.execute(query.group_by(Collection.id).order_by(Collection.name))
        return list(result.all())


async def list_collections(
    search: Annotated[
        str | None, Field(description="Optional name or slug filter", max_length=200)
    ] = None,
) -> CollectionListResult:
    """List documentation collections visible to the current identity."""
    rows = await _query_collections(user_id=get_current_user_id(), search=search)
    collections = []
    for collection_id, name, slug, visibility, source_count in rows:
        collections.append(
            CollectionItem(
                id=str(collection_id),
                name=name,
                slug=slug,
                visibility=visibility.value,
                source_count=source_count,
            )
        )
    return CollectionListResult(collections=collections, total=len(collections))


async def list_documents(
    collection_id: Annotated[str, Field(description="Collection UUID")],
    topic: Annotated[
        str | None, Field(description="Optional title or URI filter", max_length=200)
    ] = None,
    limit: Annotated[int, Field(description="Maximum documents to return", ge=1, le=100)] = 50,
) -> DocumentListResult:
    """List a bounded page of documents in an accessible collection."""
    user_id = get_current_user_id()
    async with get_db_session() as db:
        collection = await _require_collection(db, collection_id, user_id)
        query = (
            select(Document.id, Document.uri, Document.title, Source.url)
            .join(Source, Document.source_id == Source.id)
            .where(Source.collection_id == collection.id)
        )
        if topic:
            escaped_topic = escape_like_pattern(topic)
            query = query.where(
                or_(
                    Document.title.ilike(f"%{escaped_topic}%", escape="\\"),
                    Document.uri.ilike(f"%{escaped_topic}%", escape="\\"),
                )
            )
        rows = list((await db.execute(query.order_by(Document.title).limit(limit + 1))).all())

    return DocumentListResult(
        collection_id=str(collection.id),
        collection_name=collection.name,
        documents=[
            DocumentItem(
                id=str(document_id),
                uri=uri,
                title=title or "Untitled",
                source_url=source_url,
            )
            for document_id, uri, title, source_url in rows[:limit]
        ],
        limit=limit,
        truncated=len(rows) > limit,
    )


async def _get_raw_chunks(
    *,
    query: str,
    user_id: uuid.UUID | None,
    collection_id: uuid.UUID | None,
    topic: str | None,
    max_chunks: int,
    offset: int,
) -> MarkdownResult:
    settings = get_settings()
    query_embedding: list[float] | None = None
    if settings.model_calls_enabled:
        try:
            provider, model = parse_embedding_model_spec(settings.default_embedding_model)
            embedder = get_embedder(provider, model)
        except Exception:
            embedder = FakeEmbedder()
        query_embedding = (await embedder.embed_batch([query])).embeddings[0]

    search_limit = max_chunks + offset + (50 if topic else 0)
    response = await hybrid_search(
        query=query,
        query_embedding=query_embedding,
        user_id=user_id,
        collection_id=collection_id,
        top_k=search_limit,
    )
    results = response.results
    if topic:
        topic_lower = topic.lower()
        results = [
            result
            for result in results
            if topic_lower in result.title.lower() or topic_lower in result.uri.lower()
        ]
    results = results[offset : offset + max_chunks]

    lines: list[str] = []
    if not settings.model_calls_enabled:
        lines.append("*Retrieval mode: deterministic full-text search; model calls disabled.*")
    for index, result in enumerate(results, offset + 1):
        lines.extend(
            [
                f"## Result {index}: {result.title}",
                f"*Source: {result.uri}*",
                "",
                result.content,
                "",
            ]
        )
    sources = [ContextSource(uri=result.uri, title=result.title) for result in results]
    return MarkdownResult(
        query=query,
        collection_id=str(collection_id) if collection_id else None,
        mode="retrieval",
        markdown="\n".join(lines),
        chunks_used=len(results),
        sources=sources,
        offset=offset,
    )


async def get_context_markdown(
    query: Annotated[str, Field(description="Natural-language search query", min_length=1)],
    collection_id: Annotated[str | None, Field(description="Optional collection UUID")] = None,
    topic: Annotated[
        str | None, Field(description="Optional title or URI filter", max_length=200)
    ] = None,
    max_chunks: Annotated[int, Field(description="Maximum retrieved chunks", ge=1, le=50)] = 10,
    max_tokens: Annotated[
        int, Field(description="Maximum synthesis tokens", ge=1, le=16000)
    ] = 4000,
    offset: Annotated[int, Field(description="Retrieval offset", ge=0)] = 0,
    raw: Annotated[bool, Field(description="Skip synthesis")] = False,
) -> MarkdownResult:
    """Retrieve evidence or synthesize bounded Markdown from indexed context."""
    collection_uuid: uuid.UUID | None = None
    if collection_id:
        try:
            collection_uuid = uuid.UUID(collection_id)
        except ValueError as exc:
            raise ToolError(f"Invalid collection_id: {collection_id}") from exc

    user_id = get_current_user_id()
    settings = get_settings()
    try:
        if raw or topic or not settings.model_calls_enabled:
            return await _get_raw_chunks(
                query=query,
                user_id=user_id,
                collection_id=collection_uuid,
                topic=topic,
                max_chunks=max_chunks,
                offset=offset,
            )
        response = await assemble_context(
            query=query,
            user_id=user_id,
            collection_id=collection_uuid,
            max_chunks=max_chunks,
            max_tokens=max_tokens,
        )
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Failed to retrieve context: {exc}") from exc

    return MarkdownResult(
        query=response.query,
        collection_id=str(collection_uuid) if collection_uuid else None,
        mode="synthesis",
        markdown=response.markdown,
        chunks_used=response.chunks_used,
        sources=[ContextSource.model_validate(source) for source in response.sources],
        offset=offset,
    )


def register_collection_tools(mcp: FastMCP) -> None:
    """Register collection and retrieval tools on the shared MCP server."""
    mcp.tool(name="list_collections")(list_collections)
    mcp.tool(name="list_documents")(list_documents)
    mcp.tool(name="get_markdown")(get_context_markdown)
