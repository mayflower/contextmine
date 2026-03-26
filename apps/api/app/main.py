"""FastAPI application with MCP server mounted."""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from contextmine_core import close_engine
from contextmine_core.telemetry import init_telemetry, shutdown_telemetry
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.mcp_server import mcp_app, mcp_lifespan
from app.middleware import SessionMiddleware
from app.rate_limit import limiter
from app.routes import (
    auth,
    collections,
    context,
    db,
    documents,
    health,
    metrics_ingest,
    prefect,
    runs,
    search,
    sources,
    twin,
    validation,
)

# Static files directory (built frontend)
STATIC_DIR = Path(os.getenv("STATIC_DIR", "/app/static"))
_INDEX_HTML = STATIC_DIR / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan handler - integrates MCP and telemetry."""
    # Initialize telemetry FIRST (before any other setup)
    telemetry_enabled = init_telemetry(service_suffix="-api")

    if telemetry_enabled:
        # Auto-instrument FastAPI (must be done before routes are accessed)
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        FastAPIInstrumentor.instrument_app(app)
        HTTPXClientInstrumentor().instrument()

    # Run MCP lifespan (initializes StreamableHTTPSessionManager)
    async with mcp_lifespan(app):
        # Startup
        yield
    # Shutdown
    await close_engine()
    await shutdown_telemetry()


def _register_api_routes(app: FastAPI) -> None:
    """Register all API route modules."""
    for router_module in (
        health,
        db,
        auth,
        collections,
        sources,
        runs,
        documents,
        search,
        context,
        prefect,
        twin,
        metrics_ingest,
        validation,
    ):
        app.include_router(router_module.router, prefix="/api")


def _register_spa_routes(app: FastAPI, instrumentator: Instrumentator) -> None:
    """Mount static assets and SPA catch-all if the static directory exists."""
    if not (STATIC_DIR.exists() and STATIC_DIR.is_dir()):
        instrumentator.expose(app, include_in_schema=False)
        return

    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")
    instrumentator.expose(app, include_in_schema=False)

    @app.get("/{path:path}", response_model=None)
    async def serve_spa(path: str) -> FileResponse | PlainTextResponse:
        """Serve the SPA frontend for all non-API routes."""
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        if path == "metrics":
            return PlainTextResponse(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
        file_path = (STATIC_DIR / path).resolve()
        if not file_path.is_relative_to(STATIC_DIR.resolve()):
            return FileResponse(_INDEX_HTML)
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(_INDEX_HTML)

    @app.get("/")
    async def serve_index() -> FileResponse:
        """Serve index.html at root."""
        return FileResponse(_INDEX_HTML)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="ContextMine API",
        description="Open-source Context7 alternative with MCP retrieval",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SessionMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _register_api_routes(app)
    app.mount("/mcp", mcp_app)

    @app.get("/.well-known/{well_known_type}")
    @app.get("/.well-known/{well_known_type}/{path:path}")
    async def forward_well_known(
        request: Request, well_known_type: str, path: str = ""
    ) -> JSONResponse:
        """Forward .well-known requests to MCP sub-app for OAuth discovery."""
        import httpx

        mcp_url = (
            f"http://localhost:{os.getenv('API_PORT', '8000')}/mcp/.well-known/{well_known_type}"
        )
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(mcp_url, timeout=5.0)
                if resp.status_code == 200:
                    return JSONResponse(content=resp.json())
                return JSONResponse(content={"error": "not_found"}, status_code=resp.status_code)
        except Exception:
            return JSONResponse(content={"error": "mcp_unavailable"}, status_code=503)

    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=False,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics", "/health", "/api/health"],
        inprogress_name="http_requests_inprogress",
        inprogress_labels=True,
    )
    instrumentator.instrument(app)
    _register_spa_routes(app, instrumentator)

    return app


app = create_app()
