"""MCP server implementation with FastMCP 2 and Streamable HTTP transport."""

import uuid
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Annotated

from contextmine_core import (
    Collection,
    CollectionMember,
    CollectionVisibility,
    Document,
    MCPApiToken,
    Source,
    get_settings,
    verify_api_token,
)
from contextmine_core import get_session as get_db_session
from contextmine_core.context import assemble_context
from contextmine_core.embeddings import FakeEmbedder, get_embedder, parse_embedding_model_spec
from contextmine_core.search import hybrid_search
from fastmcp import FastMCP
from sqlalchemy import func, or_, select, update
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount

# Create FastMCP server
mcp = FastMCP(
    name="contextmine",
    instructions="""ContextMine MCP server for documentation and code retrieval.

To use ContextMine, add "use contextmine" to your prompt.

Available tools:
- context.list_collections: Discover available documentation collections
- context.list_documents: Browse documents in a collection
- context.get_markdown: Search and retrieve assembled context

Typical workflow:
1. Call list_collections to find the right collection
2. Optionally call list_documents to explore what's available
3. Call get_markdown with your query to get relevant documentation
""",
)

# Store user_id for the current request using contextvars (async-safe)
_current_user_id: ContextVar[uuid.UUID | None] = ContextVar("current_user_id", default=None)


def set_current_user_id(user_id: uuid.UUID | None) -> None:
    """Set the current user ID for authorization."""
    _current_user_id.set(user_id)


def get_current_user_id() -> uuid.UUID | None:
    """Get the current user ID."""
    return _current_user_id.get()


class MCPAuthMiddleware(BaseHTTPMiddleware):
    """Middleware to authenticate MCP requests with Bearer tokens and validate Origin."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Validate Bearer token and Origin header."""
        settings = get_settings()

        # Check Origin allowlist
        origin = request.headers.get("origin")
        allowed_origins = [
            o.strip() for o in settings.mcp_allowed_origins.split(",") if o.strip()
        ]

        if allowed_origins and origin and origin not in allowed_origins:
            return JSONResponse(
                {"error": "Origin not allowed"},
                status_code=403,
            )

        # Check Bearer token
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                {"error": "Missing or invalid Authorization header"},
                status_code=401,
            )

        token = auth_header[7:]  # Remove "Bearer " prefix

        # Verify token against database
        user_id = await self._verify_token(token)
        if not user_id:
            return JSONResponse(
                {"error": "Invalid or revoked token"},
                status_code=401,
            )

        # Store user_id for tool handlers
        set_current_user_id(user_id)

        try:
            return await call_next(request)
        finally:
            # Clear user_id after request
            set_current_user_id(None)

    async def _verify_token(self, token: str) -> uuid.UUID | None:
        """Verify token against database and return user_id if valid."""
        async with get_db_session() as db:
            # Get all non-revoked tokens
            result = await db.execute(
                select(MCPApiToken).where(MCPApiToken.revoked_at.is_(None))
            )
            tokens = result.scalars().all()

            for db_token in tokens:
                if verify_api_token(token, db_token.token_hash):
                    # Update last_used_at
                    await db.execute(
                        update(MCPApiToken)
                        .where(MCPApiToken.id == db_token.id)
                        .values(last_used_at=datetime.now(UTC))
                    )
                    return db_token.user_id

        return None


@mcp.tool(name="context.list_collections")
async def list_collections(
    search: Annotated[str | None, "Optional search term to filter collections by name"] = None,
) -> str:
    """List all documentation collections available to search.

    Call this first to discover what collections are available.
    Returns collection IDs, names, and descriptions that you can use
    with other tools like get_markdown or list_documents.

    Example: To find React documentation, call list_collections with search="react"
    """
    user_id = get_current_user_id()

    async with get_db_session() as db:
        # Build query for accessible collections
        query = select(
            Collection.id,
            Collection.name,
            Collection.slug,
            Collection.visibility,
            func.count(Source.id).label("source_count"),
        ).outerjoin(Source, Source.collection_id == Collection.id)

        # Filter by visibility/ownership
        if user_id:
            query = query.where(
                or_(
                    Collection.visibility == CollectionVisibility.GLOBAL,
                    Collection.owner_user_id == user_id,
                    Collection.id.in_(
                        select(CollectionMember.collection_id).where(
                            CollectionMember.user_id == user_id
                        )
                    ),
                )
            )
        else:
            query = query.where(Collection.visibility == CollectionVisibility.GLOBAL)

        # Optional search filter
        if search:
            query = query.where(
                or_(
                    Collection.name.ilike(f"%{search}%"),
                    Collection.slug.ilike(f"%{search}%"),
                )
            )

        query = query.group_by(Collection.id).order_by(Collection.name)
        result = await db.execute(query)
        rows = result.all()

    if not rows:
        return "# No Collections Found\n\nNo collections are available."

    lines = ["# Available Collections\n"]
    for row in rows:
        coll_id, name, slug, visibility, source_count = row
        lines.append(f"## {name}")
        lines.append(f"- **ID**: `{coll_id}`")
        lines.append(f"- **Slug**: {slug}")
        lines.append(f"- **Sources**: {source_count}")
        lines.append(f"- **Visibility**: {visibility.value}")
        lines.append("")

    return "\n".join(lines)


@mcp.tool(name="context.list_documents")
async def list_documents(
    collection_id: Annotated[str, "The collection ID to list documents from"],
    topic: Annotated[str | None, "Optional topic/path filter (e.g., 'routing', 'api')"] = None,
    limit: Annotated[int, "Maximum number of documents to return"] = 50,
) -> str:
    """List documents in a collection to explore available content.

    Use this to browse what documentation is available before searching.
    Optionally filter by topic to focus on specific areas like 'authentication',
    'routing', or 'api'.

    Example: list_documents(collection_id="...", topic="hooks")
    """
    user_id = get_current_user_id()

    try:
        coll_uuid = uuid.UUID(collection_id)
    except ValueError:
        return f"# Error\n\nInvalid collection_id: {collection_id}"

    async with get_db_session() as db:
        # Verify access to collection
        coll_result = await db.execute(
            select(Collection).where(Collection.id == coll_uuid)
        )
        collection = coll_result.scalar_one_or_none()

        if not collection:
            return "# Error\n\nCollection not found."

        # Check access
        has_access = collection.visibility == CollectionVisibility.GLOBAL
        if user_id and not has_access:
            if collection.owner_user_id == user_id:
                has_access = True
            else:
                member_result = await db.execute(
                    select(CollectionMember).where(
                        CollectionMember.collection_id == coll_uuid,
                        CollectionMember.user_id == user_id,
                    )
                )
                if member_result.scalar_one_or_none():
                    has_access = True

        if not has_access:
            return "# Error\n\nAccess denied to this collection."

        # Query documents
        query = (
            select(Document.id, Document.uri, Document.title, Source.url)
            .join(Source, Document.source_id == Source.id)
            .where(Source.collection_id == coll_uuid)
        )

        if topic:
            query = query.where(
                or_(
                    Document.title.ilike(f"%{topic}%"),
                    Document.uri.ilike(f"%{topic}%"),
                )
            )

        query = query.order_by(Document.title).limit(limit)
        result = await db.execute(query)
        rows = result.all()

    if not rows:
        msg = "# No Documents Found\n\nNo documents in collection"
        if topic:
            msg += f" matching topic '{topic}'"
        return msg + "."

    lines = [f"# Documents in {collection.name}\n"]
    if topic:
        lines.append(f"*Filtered by topic: {topic}*\n")

    for _doc_id, uri, title, _source_url in rows:
        lines.append(f"- **{title or 'Untitled'}**")
        lines.append(f"  - URI: `{uri}`")
        lines.append("")

    if len(rows) == limit:
        lines.append(f"*Showing first {limit} documents. Use topic filter to narrow results.*")

    return "\n".join(lines)


@mcp.tool(name="context.get_markdown")
async def get_context_markdown(
    query: Annotated[str, "The search query to find relevant documentation"],
    collection_id: Annotated[
        str | None,
        "Collection ID to search within. Call list_collections first to find IDs.",
    ] = None,
    topic: Annotated[
        str | None,
        "Focus on specific topic like 'routing', 'authentication', or 'hooks'",
    ] = None,
    max_chunks: Annotated[int, "Maximum number of chunks to retrieve (1-50)"] = 10,
    max_tokens: Annotated[int, "Maximum tokens for LLM response"] = 4000,
    offset: Annotated[int, "Skip first N results for pagination"] = 0,
    raw: Annotated[
        bool,
        "Return raw chunks instead of LLM-assembled response (faster, cheaper)",
    ] = False,
) -> str:
    """Search and retrieve documentation context as Markdown.

    Uses hybrid retrieval (full-text + vector search) to find relevant
    documentation and code. By default, results are assembled into a
    coherent document using an LLM. Use raw=true for faster raw chunks.

    Tips:
    - Call list_collections first if you don't know the collection ID
    - Use topic parameter to focus on specific areas (e.g., topic="routing")
    - Use raw=true for quick lookups without LLM processing
    - Increase max_chunks for broader context

    Add "use contextmine" to your prompts to trigger this tool.
    """
    user_id = get_current_user_id()

    # Parse collection_id if provided
    collection_uuid: uuid.UUID | None = None
    if collection_id:
        try:
            collection_uuid = uuid.UUID(collection_id)
        except ValueError:
            return f"# Error\n\nInvalid collection_id: {collection_id}"

    try:
        # If raw mode or topic filter, we need to do the search ourselves
        if raw or topic:
            return await _get_raw_chunks(
                query=query,
                user_id=user_id,
                collection_id=collection_uuid,
                topic=topic,
                max_chunks=max_chunks,
                offset=offset,
            )

        # Standard LLM-assembled response
        response = await assemble_context(
            query=query,
            user_id=user_id,
            collection_id=collection_uuid,
            max_chunks=max_chunks,
            max_tokens=max_tokens,
        )
        return response.markdown
    except Exception as e:
        return f"# Error\n\nFailed to retrieve context: {e!s}"


async def _get_raw_chunks(
    query: str,
    user_id: uuid.UUID | None,
    collection_id: uuid.UUID | None,
    topic: str | None,
    max_chunks: int,
    offset: int,
) -> str:
    """Get raw search results without LLM assembly."""
    settings = get_settings()

    # Get query embedding
    try:
        emb_provider, emb_model = parse_embedding_model_spec(settings.default_embedding_model)
        embedder = get_embedder(emb_provider, emb_model)
    except (ValueError, Exception):
        embedder = FakeEmbedder()

    embed_result = await embedder.embed_batch([query])
    query_embedding = embed_result.embeddings[0]

    # Search with extra results to handle topic filtering and offset
    search_limit = max_chunks + offset + 50 if topic else max_chunks + offset
    search_response = await hybrid_search(
        query=query,
        query_embedding=query_embedding,
        user_id=user_id,
        collection_id=collection_id,
        top_k=search_limit,
    )

    results = search_response.results

    # Apply topic filter if specified
    if topic:
        topic_lower = topic.lower()
        results = [
            r for r in results
            if topic_lower in r.title.lower() or topic_lower in r.uri.lower()
        ]

    # Apply offset and limit
    results = results[offset : offset + max_chunks]

    if not results:
        msg = f"# No Results\n\nNo content found for query: {query}"
        if topic:
            msg += f" (topic: {topic})"
        return msg

    # Build markdown output
    lines = [f"# Search Results for: {query}\n"]
    if topic:
        lines.append(f"*Filtered by topic: {topic}*\n")

    seen_uris: set[str] = set()
    for i, result in enumerate(results, 1):
        lines.append(f"## Result {i + offset}: {result.title}")
        lines.append(f"*Source: {result.uri}*\n")
        lines.append(result.content)
        lines.append("")
        seen_uris.add(result.uri)

    # Add sources section
    lines.append("## Sources\n")
    for uri in seen_uris:
        lines.append(f"- {uri}")

    return "\n".join(lines)


def get_context_markdown_sync(query: str) -> str:
    """Synchronous wrapper for testing. Returns a simple placeholder."""
    return f"""# Context for: {query}

## Summary

This is a test placeholder. Use the async MCP endpoint for real results.

## Sources

- No sources (test mode)
"""


# Get the HTTP app from FastMCP with root path (mounted at /mcp in main.py)
# Use stateless_http=True for simpler API testing without session management
_mcp_http_app = mcp.http_app(path="/", stateless_http=True)

# Export the MCP lifespan for integration with FastAPI
mcp_lifespan = _mcp_http_app.lifespan


# Create Starlette app with auth middleware wrapping the MCP HTTP app
mcp_app = Starlette(
    routes=[
        Mount("/", app=_mcp_http_app),
    ],
    middleware=[
        Middleware(MCPAuthMiddleware),
    ],
)


# Export tool list for tests
def get_tools() -> list[dict]:
    """Get list of available tools for testing."""
    return [
        {
            "name": "context.list_collections",
            "description": "List all documentation collections available to search.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "search": {
                        "type": "string",
                        "description": "Optional search term to filter collections by name",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "context.list_documents",
            "description": "List documents in a collection to explore available content.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "collection_id": {
                        "type": "string",
                        "description": "The collection ID to list documents from",
                    },
                    "topic": {
                        "type": "string",
                        "description": "Optional topic/path filter",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of documents to return",
                        "default": 50,
                    },
                },
                "required": ["collection_id"],
            },
        },
        {
            "name": "context.get_markdown",
            "description": "Search and retrieve documentation context as Markdown.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant documentation",
                    },
                    "collection_id": {
                        "type": "string",
                        "description": "Collection ID to search within",
                    },
                    "topic": {
                        "type": "string",
                        "description": "Focus on specific topic like 'routing' or 'hooks'",
                    },
                    "max_chunks": {
                        "type": "integer",
                        "description": "Maximum number of chunks to retrieve",
                        "default": 10,
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens for LLM response",
                        "default": 4000,
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Skip first N results for pagination",
                        "default": 0,
                    },
                    "raw": {
                        "type": "boolean",
                        "description": "Return raw chunks instead of LLM-assembled response",
                        "default": False,
                    },
                },
                "required": ["query"],
            },
        },
    ]
