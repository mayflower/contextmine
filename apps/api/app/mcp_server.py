"""MCP server implementation with FastMCP 2 and Streamable HTTP transport."""

import uuid
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Annotated

from contextmine_core import MCPApiToken, get_settings, verify_api_token
from contextmine_core import get_session as get_db_session
from contextmine_core.context import assemble_context
from fastmcp import FastMCP
from sqlalchemy import select, update
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount

# Create FastMCP server
mcp = FastMCP(
    name="contextmine",
    instructions="ContextMine MCP server for documentation and code retrieval. "
    "Use the context.get_markdown tool to search and retrieve assembled context.",
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


@mcp.tool(name="context.get_markdown")
async def get_context_markdown(
    query: Annotated[str, "The search query to find relevant context"],
    collection_id: Annotated[
        str | None, "Optional collection ID to search within"
    ] = None,
    max_chunks: Annotated[int, "Maximum number of chunks to retrieve"] = 10,
    max_tokens: Annotated[int, "Maximum tokens for LLM response"] = 4000,
) -> str:
    """Retrieve assembled context as a Markdown document based on a search query.

    Uses hybrid retrieval (FTS + vector search) to find relevant documentation
    and code, then assembles results into a coherent Markdown document using an LLM.
    The response includes a Sources section listing all referenced documents.
    """
    # Get user_id from middleware
    user_id = get_current_user_id()

    # Parse collection_id if provided
    collection_uuid: uuid.UUID | None = None
    if collection_id:
        try:
            collection_uuid = uuid.UUID(collection_id)
        except ValueError:
            return f"# Error\n\nInvalid collection_id: {collection_id}"

    try:
        response = await assemble_context(
            query=query,
            user_id=user_id,
            collection_id=collection_uuid,
            max_chunks=max_chunks,
            max_tokens=max_tokens,
        )
        return response.markdown
    except Exception as e:
        return f"# Error\n\nFailed to assemble context: {e!s}"


def get_context_markdown_sync(query: str) -> str:
    """Synchronous wrapper for testing. Returns a simple placeholder."""
    return f"""# Context for: {query}

## Summary

This is a test placeholder. Use the async MCP endpoint for real results.

## Sources

- No sources (test mode)
"""


# Get the HTTP app from FastMCP with empty path (mounted at /mcp in main.py)
_mcp_http_app = mcp.http_app(path="")

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
            "name": "context.get_markdown",
            "description": "Retrieve assembled context as a Markdown document based on a search query.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant context",
                    },
                    "collection_id": {
                        "type": "string",
                        "description": "Optional collection ID to search within",
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
                },
                "required": ["query"],
            },
        }
    ]
