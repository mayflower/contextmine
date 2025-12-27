"""FastAPI application with MCP server mounted."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from contextmine_core import close_engine
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.mcp_server import mcp_app, mcp_lifespan
from app.middleware import SessionMiddleware
from app.routes import (
    auth,
    collections,
    context,
    db,
    documents,
    health,
    mcp_tokens,
    prefect,
    runs,
    search,
    sources,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan handler - integrates MCP lifespan."""
    # Run MCP lifespan (initializes StreamableHTTPSessionManager)
    async with mcp_lifespan(app):
        # Startup
        yield
    # Shutdown
    await close_engine()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="ContextMine API",
        description="Open-source Context7 alternative with MCP retrieval",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Session middleware (must be added before CORS)
    app.add_middleware(SessionMiddleware)

    # CORS middleware - permissive for dev
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:8000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(health.router, prefix="/api")
    app.include_router(db.router, prefix="/api")
    app.include_router(auth.router, prefix="/api")
    app.include_router(mcp_tokens.router, prefix="/api")
    app.include_router(collections.router, prefix="/api")
    app.include_router(sources.router, prefix="/api")
    app.include_router(runs.router, prefix="/api")
    app.include_router(documents.router, prefix="/api")
    app.include_router(search.router, prefix="/api")
    app.include_router(context.router, prefix="/api")
    app.include_router(prefect.router, prefix="/api")

    # Mount MCP server at /mcp
    app.mount("/mcp", mcp_app)

    return app


app = create_app()
