# ContextMine

Self-hosted documentation and code indexing with MCP integration. An open-source alternative to Context7.

## What is ContextMine?

ContextMine indexes your documentation and code repositories, making them searchable via the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/). Connect it to Claude Desktop, Cursor, or any MCP-compatible AI assistant to give your AI accurate, up-to-date context from your own sources.

**Key features:**
- **Web crawling** - Index documentation sites automatically
- **Git indexing** - Index GitHub repositories with incremental updates
- **Hybrid search** - Full-text search + vector similarity for accurate retrieval
- **MCP integration** - Works with Claude Desktop, Cursor, and other MCP clients
- **Self-hosted** - Your data stays on your infrastructure

## Quick Start with Docker

```bash
# Clone the repository
git clone https://github.com/mayflower/contextmine.git
cd contextmine

# Copy environment template
cp .env.example .env

# Start all services
docker compose up -d

# Run database migrations
docker compose exec api sh -c "cd /app/packages/core && alembic upgrade head"
```

Access the admin UI at http://localhost:5173

## Connecting to MCP Clients

### 1. Create an MCP Token

1. Open the admin UI at http://localhost:5173
2. Log in with GitHub OAuth
3. Navigate to **MCP Tokens** in the sidebar
4. Create a new token and copy it

### 2. Configure Your MCP Client

**Claude Desktop** (`~/.config/claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "contextmine": {
      "url": "http://localhost:8000/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_TOKEN_HERE"
      }
    }
  }
}
```

**Cursor** (Settings → MCP):

Add a new server with URL `http://localhost:8000/mcp` and your Bearer token.

### 3. Use It

In your AI assistant, say **"use contextmine"** to activate context retrieval, then ask questions about your indexed documentation.

## Available MCP Tools

| Tool | Description |
|------|-------------|
| `context.list_collections` | Discover available documentation collections |
| `context.list_documents` | Browse documents in a collection |
| `context.get_markdown` | Search and retrieve context as Markdown |

## Adding Sources

### Web Documentation

1. Go to **Collections** → Create a new collection
2. Go to **Sources** → Add a new source
3. Select **Web** as the source type
4. Enter the base URL (e.g., `https://docs.example.com/`)
5. Click **Sync** to start crawling

### GitHub Repositories

1. Create a collection if you haven't already
2. Add a new source with type **GitHub**
3. Enter the repository (e.g., `owner/repo`)
4. Optionally specify a branch and path filter
5. Click **Sync** to start indexing

## Configuration

Copy `.env.example` to `.env` and configure:

| Variable | Description | Required |
|----------|-------------|----------|
| `DATABASE_URL` | PostgreSQL connection string | Yes |
| `GITHUB_CLIENT_ID` | GitHub OAuth app client ID | Yes |
| `GITHUB_CLIENT_SECRET` | GitHub OAuth app secret | Yes |
| `SESSION_SECRET` | Secret for session cookies | Yes |
| `TOKEN_ENCRYPTION_KEY` | Key for encrypting tokens | Yes |
| `OPENAI_API_KEY` | OpenAI API key for embeddings | Yes* |
| `GEMINI_API_KEY` | Alternative: Gemini API key | Yes* |
| `MCP_ALLOWED_ORIGINS` | Allowed origins for MCP (production) | No |

*At least one embedding provider is required.

### GitHub OAuth Setup

1. Go to https://github.com/settings/developers
2. Create a new OAuth App:
   - **Homepage URL**: `http://localhost:5173`
   - **Callback URL**: `http://localhost:8000/api/auth/callback`
3. Copy the Client ID and Secret to your `.env`

### Generate Secure Keys

```bash
# Generate session secret
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate encryption key
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Web UI    │────▶│   FastAPI   │────▶│  PostgreSQL │
│  (React)    │     │   + MCP     │     │  + pgvector │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                    ┌──────┴──────┐
                    ▼             ▼
              ┌─────────┐   ┌─────────┐
              │ Prefect │   │ spider  │
              │ Worker  │   │   _md   │
              └─────────┘   └─────────┘
```

- **API** (`apps/api`): FastAPI backend with MCP server at `/mcp`
- **Web** (`apps/web`): React admin console
- **Worker** (`apps/worker`): Prefect-based background sync jobs
- **Core** (`packages/core`): Shared models, settings, migrations

## Development

### Prerequisites

- Python 3.12+
- Node.js 20+
- [uv](https://github.com/astral-sh/uv) for Python dependency management

### Local Setup

```bash
# Install dependencies
uv sync --all-packages

# Run quality checks
uv run ruff check .          # Linting
uvx ty check                 # Type checking
uv run pytest -v             # Tests

# Start API server
cd apps/api && uv run uvicorn app.main:app --reload --port 8000

# Start frontend (separate terminal)
cd apps/web && npm install && npm run dev
```

### Pre-commit Hooks

```bash
# Install hooks
uv run pre-commit install

# Run manually
uv run pre-commit run --all-files
```

## Container Images

Pre-built images are available from GitHub Container Registry:

```bash
docker pull ghcr.io/mayflower/contextmine-api:latest
docker pull ghcr.io/mayflower/contextmine-worker:latest
docker pull ghcr.io/mayflower/contextmine-web:latest
```

## License

MIT License - see [LICENSE](LICENSE) for details.
