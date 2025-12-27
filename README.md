# ContextMine

Open-source local Context7 alternative with MCP retrieval.

## Overview

ContextMine is a documentation and code indexing system that provides context to AI assistants via the Model Context Protocol (MCP). It supports:

- **Web crawling**: Index documentation sites with spider-rs
- **Git indexing**: Index GitHub repositories with incremental updates
- **Hybrid retrieval**: Full-text search + vector similarity search
- **MCP integration**: Expose context via Streamable HTTP (SSE)

## Quick Start

### Prerequisites

- Python 3.12+
- Node.js 20+
- Docker and Docker Compose
- [uv](https://github.com/astral-sh/uv) for Python dependency management

### Local Development

1. **Clone and set up environment**:

```bash
git clone <repo-url>
cd contextmine
cp .env.example .env
```

2. **Install Python dependencies**:

```bash
uv sync --all-packages
```

3. **Run the API server**:

```bash
cd apps/api
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

4. **Run the frontend** (in another terminal):

```bash
cd apps/web
npm install
npm run dev
```

5. **Access the application**:
   - Frontend: http://localhost:5173
   - API: http://localhost:8000
   - API Health: http://localhost:8000/api/health

### Docker Compose

```bash
docker compose up --build
```

Access:
- Frontend: http://localhost:5173
- API: http://localhost:8000
- Postgres: localhost:5432

### GitHub OAuth Setup

ContextMine uses GitHub OAuth for authentication. To set up:

1. **Create a GitHub OAuth App**:
   - Go to https://github.com/settings/developers
   - Click "New OAuth App"
   - Set **Application name**: ContextMine (or your preferred name)
   - Set **Homepage URL**: `http://localhost:5173`
   - Set **Authorization callback URL**: `http://localhost:8000/api/auth/callback`
   - Click "Register application"

2. **Configure environment variables**:
   - Copy the Client ID and generate a Client Secret
   - Update your `.env` file:
     ```
     GITHUB_CLIENT_ID=your_client_id_here
     GITHUB_CLIENT_SECRET=your_client_secret_here
     ```

3. **Generate secure keys for production**:
   ```bash
   # Generate session secret
   python -c "import secrets; print(secrets.token_urlsafe(32))"

   # Generate encryption key
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```
   Update `SESSION_SECRET` and `TOKEN_ENCRYPTION_KEY` in your `.env` file.

### Database Migrations

The project uses Alembic for database migrations. Migrations are located in `packages/core/alembic/`.

**Run migrations in Docker:**

```bash
# After starting docker compose
docker compose exec api sh -c "cd /app/packages/core && alembic upgrade head"
```

**Run migrations locally:**

```bash
# Ensure DATABASE_URL is set in your environment or .env file
cd packages/core
DATABASE_URL=postgresql+asyncpg://contextmine:contextmine@localhost:5432/contextmine uv run alembic upgrade head
```

**Create a new migration:**

```bash
cd packages/core
DATABASE_URL=... uv run alembic revision -m "description of changes"
```

## Smoke Tests

### Health Check

```bash
curl http://localhost:8000/api/health
# Expected: {"status":"ok"}
```

### Database Health Check

```bash
curl http://localhost:8000/api/db/health
# Expected: {"db":"ok"} (when database is configured and connected)
# Expected: {"db":"not_configured"} (when DATABASE_URL is not set)
```

### MCP Endpoint

The MCP server is mounted at `/mcp` using FastMCP 2 with Streamable HTTP transport:

- MCP endpoint: `http://localhost:8000/mcp`

**Authentication**: The MCP endpoint requires Bearer token authentication.

1. Log in to the admin UI at http://localhost:5173
2. Navigate to "MCP Tokens" in the sidebar
3. Create a new token and copy it (you'll only see it once!)
4. Use the token in your MCP client configuration

**Configuring MCP clients** (e.g., Claude Desktop):

```json
{
  "mcpServers": {
    "contextmine": {
      "url": "http://localhost:8000/mcp",
      "headers": {
        "Authorization": "Bearer your-token-here"
      }
    }
  }
}
```

**Origin allowlist**: In production, set `MCP_ALLOWED_ORIGINS` to restrict which origins can access the MCP endpoint (for DNS rebinding protection):
```
MCP_ALLOWED_ORIGINS=https://claude.ai,https://your-app.com
```

In development, leave this empty to allow all origins.

### MCP Tool: context.get_markdown

The MCP server exposes one tool for retrieval:

- **`context.get_markdown`**: Retrieve assembled context as a Markdown document
  - **Input**:
    - `query` (string, required): The search query
    - `collection_id` (string, optional): Filter to a specific collection
    - `max_chunks` (integer, optional, default: 10): Maximum chunks to retrieve
    - `max_tokens` (integer, optional, default: 4000): Maximum tokens for LLM response
  - **Output**: Markdown document with assembled context and sources

The tool uses hybrid retrieval (FTS + vector search) to find relevant chunks and assembles them into a coherent Markdown document using an LLM.

### Testing MCP with curl

You can test the MCP endpoint using JSON-RPC over HTTP:

```bash
# Call the context.get_markdown tool
curl -X POST http://localhost:8000/mcp \
  -H "Authorization: Bearer your-token-here" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "context.get_markdown",
      "arguments": {
        "query": "How do I authenticate with the API?"
      }
    }
  }'
```

### Example MCP Response

The `context.get_markdown` tool returns a Markdown document like:

```markdown
# Response to: How do I authenticate with the API?

## Summary

Based on the retrieved documentation...

## Authentication

To authenticate, use Bearer tokens...

## Sources

- https://docs.example.com/auth
- git://github.com/org/repo/src/auth.py?ref=main
```

The response includes a **Sources** section listing the document URIs used, with file paths for git:// URIs.

### MCP Smoke Test

A smoke test script is provided to verify the MCP endpoint is working:

```bash
# Run the smoke test (requires an MCP token)
python scripts/mcp_smoke_test.py --token YOUR_TOKEN

# With custom base URL
python scripts/mcp_smoke_test.py --token YOUR_TOKEN --base-url http://localhost:8000

# With custom query
python scripts/mcp_smoke_test.py --token YOUR_TOKEN --query "How do I authenticate?"
```

The smoke test:
1. Connects to the MCP SSE endpoint
2. Calls the `context.get_markdown` tool
3. Verifies the response contains a "Sources" section

## Development

### Run Quality Checks

```bash
# Linting
uv run ruff check .

# Type checking
uv run pyright

# Tests
uv run pytest -v
```

### Project Structure

```
contextmine/
├── apps/
│   ├── api/          # FastAPI backend + MCP server
│   ├── web/          # React frontend (Vite)
│   └── worker/       # Prefect worker (future)
├── packages/
│   └── core/         # Shared Python code (DB models, settings, migrations)
├── rust/
│   └── spider_md/    # Rust crawler (future)
├── docker-compose.yml
├── pyproject.toml    # Workspace configuration
└── README.md
```

## License

MIT
