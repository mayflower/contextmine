# Development Guide

This guide covers setting up a development environment and contributing to ContextMine.

## Prerequisites

- Python 3.12+
- Node.js 20+
- [uv](https://github.com/astral-sh/uv) for Python dependency management
- Docker (for PostgreSQL with pgvector)

## Project Structure

```
contextmine/
├── apps/
│   ├── api/              # FastAPI backend + MCP server
│   ├── web/              # React frontend (Vite)
│   └── worker/           # Prefect background jobs
├── packages/
│   └── core/             # Shared Python library
│       ├── contextmine_core/
│       │   ├── models.py           # SQLAlchemy models
│       │   ├── settings.py         # Configuration
│       │   ├── analyzer/           # Knowledge extraction
│       │   ├── knowledge/          # Graph builder
│       │   ├── research/           # Deep research agent
│       │   └── treesitter/         # Code parsing
│       └── alembic/                # Database migrations
├── rust/
│   └── spider_md/        # Rust web crawler
├── deploy/
│   └── helm/             # Kubernetes Helm chart
└── docs/                 # Documentation
```

## Local Development Setup

### 1. Start the Database

```bash
docker compose up -d postgres
```

### 2. Install Python Dependencies

ContextMine uses [uv](https://github.com/astral-sh/uv) for Python dependency management with workspaces.

```bash
# Install all packages
uv sync --all-packages
```

### 3. Run Database Migrations

```bash
cd packages/core
DATABASE_URL=postgresql+asyncpg://contextmine:contextmine@localhost:5432/contextmine \
  uv run alembic upgrade head
cd ../..
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 5. Start the API Server

```bash
# Build frontend first (one-time)
cd apps/web && npm install && npm run build && cd ../..

# Start API server (serves both API and frontend)
STATIC_DIR=apps/web/dist uv run uvicorn apps.api.app.main:app --reload --port 8000
```

### 6. Frontend Development (Optional)

For hot-reloading frontend development:

```bash
# Terminal 1: API server
uv run uvicorn apps.api.app.main:app --reload --port 8000

# Terminal 2: Frontend dev server (proxies to :8000)
cd apps/web && npm run dev
```

## Running Tests

```bash
# All tests
uv run pytest -v

# Specific test file
uv run pytest packages/core/tests/test_treesitter.py -v

# Single test
uv run pytest apps/api/tests/test_health.py::test_health_check -v

# With coverage
uv run pytest --cov=contextmine_core --cov-report=term-missing
```

## Code Quality

```bash
# Linting
uv run ruff check .

# Type checking
uvx ty check

# Auto-format
uv run ruff format .

# Pre-commit hooks
uv run pre-commit install
uv run pre-commit run --all-files
```

## Database Migrations

### Creating a New Migration

```bash
cd packages/core
uv run alembic revision --autogenerate -m "description of change"
```

### Applying Migrations

```bash
# Local
DATABASE_URL=postgresql+asyncpg://contextmine:contextmine@localhost:5432/contextmine \
  uv run alembic upgrade head

# Docker
docker compose exec api sh -c "cd /app/packages/core && alembic upgrade head"
```

## Adding New Features

### Adding a New MCP Tool

1. Define the tool in `apps/api/app/mcp_server.py`:

```python
@mcp.tool(name="my_tool")
async def my_tool(
    param: Annotated[str, "Parameter description"],
) -> str:
    """Tool description shown to the AI assistant."""
    # Implementation
    return "Result"
```

2. Add tests in `apps/api/tests/test_mcp_*.py`

### Adding a Knowledge Graph Extractor

1. Create extractor in `packages/core/contextmine_core/analyzer/extractors/`
2. Define node/edge kinds in `packages/core/contextmine_core/models.py`
3. Create migration if needed in `packages/core/alembic/versions/`
4. Add tests in `packages/core/tests/`

### Adding a New API Endpoint

1. Create route in `apps/api/app/routes/`
2. Register in `apps/api/app/main.py`
3. Add tests in `apps/api/tests/`

## Architecture Decisions

### Python Workspace

Uses uv workspaces with a root `pyproject.toml` defining:
- Workspace members: `apps/api`, `apps/worker`, `packages/*`
- Dev dependencies: ruff, ty, pytest, httpx
- Shared source: `contextmine-core` package

### Database

PostgreSQL with pgvector extension for:
- Document storage
- Vector embeddings
- Full-text search
- Knowledge graph nodes/edges

### Async Everywhere

All database operations use SQLAlchemy 2.x async with asyncpg.

### MCP Implementation

Uses FastMCP 2 with Streamable HTTP transport, mounted at `/mcp` on the FastAPI app.

## Key Modules

### `contextmine_core.models`

SQLAlchemy models for:
- `Collection`, `Source`, `Document` - Content management
- `Chunk`, `Symbol`, `SymbolEdge` - Code intelligence
- `KnowledgeNode`, `KnowledgeEdge` - Knowledge graph
- `KnowledgeEvidence`, `KnowledgeArtifact` - Evidence and artifacts

### `contextmine_core.search`

Hybrid search combining:
- PostgreSQL full-text search (BM25)
- pgvector similarity search
- Reciprocal Rank Fusion (RRF) for ranking

### `contextmine_core.research`

Deep research agent with:
- Tool-based investigation
- Evidence collection
- LLM-powered synthesis

### `contextmine_core.graphrag`

GraphRAG implementation with:
- Leiden community detection
- Global context (community summaries)
- Local context (entity expansion)
- Map-reduce answering

### `contextmine_core.analyzer`

Knowledge extraction:
- Business rule mining from code
- ERM extraction from Alembic migrations
- System surface catalog (OpenAPI, GraphQL, Protobuf)

## Docker Development

### Full Stack

```bash
docker compose up --build
```

### Rebuilding Specific Services

```bash
docker compose build api
docker compose up -d api
```

### Viewing Logs

```bash
docker compose logs -f api
docker compose logs -f worker
```

## Deployment

### Container Images

Pre-built images from GitHub Container Registry:

```bash
docker pull ghcr.io/mayflower/contextmine-api:latest
docker pull ghcr.io/mayflower/contextmine-worker:latest
docker pull ghcr.io/mayflower/contextmine-web:latest
```

### Helm Chart

```bash
# Install
helm install contextmine oci://ghcr.io/mayflower/contextmine -f values.yaml

# Upgrade
helm upgrade contextmine oci://ghcr.io/mayflower/contextmine -f values.yaml
```

See `deploy/helm/contextmine/README.md` for configuration options.

## Debugging

### API Debugging

```bash
# Enable debug mode
DEBUG=true uv run uvicorn apps.api.app.main:app --reload --port 8000
```

### Worker Debugging

```bash
# Run with verbose logging
DEBUG=true uv run python -m contextmine_worker
```

### Database Queries

```bash
# Connect to database
docker compose exec postgres psql -U contextmine -d contextmine
```

### Prefect UI

The Prefect UI is available at `http://localhost:4200` for monitoring sync jobs.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Ensure all checks pass:
   ```bash
   uv run ruff check .
   uvx ty check
   uv run pytest -v
   ```
5. Submit a pull request

### Commit Convention

Use conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `refactor:` Code refactoring
- `test:` Test changes
- `chore:` Maintenance
