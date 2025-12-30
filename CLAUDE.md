# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Install all Python dependencies (run from repo root)
uv sync --all-packages

# Quality checks (run from repo root)
uv run ruff check .        # Linting
uvx ty check               # Type checking
uv run pytest -v           # Run all tests

# Run a single test
uv run pytest apps/api/tests/test_health.py::test_health_check -v

# Start API server (run from repo root)
uv run uvicorn apps.api.app.main:app --reload --port 8000

# Start frontend dev server (from apps/web)
cd apps/web && npm install && npm run dev

# Docker (full stack)
docker compose up --build

# Database migrations (run from packages/core)
cd packages/core && DATABASE_URL=postgresql+asyncpg://contextmine:contextmine@localhost:5432/contextmine uv run alembic upgrade head

# Run migrations in Docker
docker compose exec api sh -c "cd /app/packages/core && alembic upgrade head"
```

## Architecture Overview

ContextMine is a documentation/code indexing system exposing context via MCP (Model Context Protocol).

### Monorepo Structure

- **apps/api**: FastAPI backend with MCP server mounted at `/mcp`
- **apps/web**: React frontend (Vite) admin console
- **apps/worker**: Prefect worker for background sync jobs
- **packages/core**: Shared Python library (settings, DB models, services)
- **rust/spider_md**: Rust-based web crawler binary (HTML→Markdown)

### Python Workspace

Uses uv workspaces. The root `pyproject.toml` defines:
- Workspace members: `apps/api`, `apps/worker`, `packages/*`
- Dev dependencies: ruff, ty, pytest, httpx
- Shared source: `contextmine-core` package

### API Structure

FastAPI app (`apps/api/app/main.py`) mounts:
- REST routes under `/api/*` (health, auth, collections, sources, etc.)
- MCP server at `/mcp` using Streamable HTTP transport

MCP exposes tools: `get_markdown` (semantic search), `list_collections`, `list_documents`, `outline`, `find_symbol`, `definition`, `references`, `expand`, `deep_research`.

### Key Conventions

- Backend routes: `/api/*`
- MCP endpoint: `/mcp` (Streamable HTTP)
- Environment config: `.env.example` documents all env vars
- Incremental builds: each step should pass `uv sync`, `ruff check`, `ty check`, `pytest`

## Tech Stack

- **Backend**: Python 3.12, FastAPI, SQLAlchemy 2.x async, Alembic, Postgres+pgvector
- **Frontend**: React, Vite, TypeScript
- **Orchestration**: Prefect (scheduled syncs)
- **Crawling**: spider-rs (Rust binary for HTML→Markdown)
- **Retrieval**: Hybrid FTS + vector search with RRF ranking
- **LLM providers**: OpenAI, Anthropic, Gemini (embeddings: OpenAI/Gemini only)
