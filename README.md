<p align="center">
  <img src="logo.png" alt="ContextMine" width="120" />
</p>

<h1 align="center">ContextMine</h1>

<p align="center">
  Self-hosted documentation and code indexing with MCP integration.<br/>
  Give your AI assistant accurate, up-to-date context from your own sources.
</p>

## What is ContextMine?

ContextMine indexes your documentation and code repositories, making them searchable via the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/). Connect it to Claude Desktop, Cursor, or any MCP-compatible AI assistant to provide rich context for code understanding, documentation lookup, and codebase exploration.

**Key features:**

- **Hybrid search** - Full-text + vector similarity with RRF ranking for accurate retrieval
- **Deep research agent** - Multi-step AI agent with LSP and Tree-sitter for complex codebase questions
- **Code intelligence** - Symbol extraction, code outlines, and structural navigation via Tree-sitter
- **Web crawling** - Index documentation sites automatically
- **Git indexing** - Index GitHub repositories with incremental updates
- **Self-hosted** - Your data stays on your infrastructure

### Deep Research Agent

The deep research agent goes beyond simple search to answer complex questions about your codebase. It uses an iterative approach with multiple tools:

| Tool | Description |
|------|-------------|
| **Hybrid Search** | BM25 + vector similarity search with RRF ranking |
| **LSP Go to Definition** | Jump to symbol definitions across files |
| **LSP Find References** | Find all usages of a symbol |
| **LSP Hover** | Get type information and documentation |
| **Tree-sitter Outline** | Extract file structure (classes, functions, methods) |
| **Tree-sitter Find Symbol** | Locate symbols by name pattern |
| **Graph Traversal** | Navigate call graphs and dependencies |

The agent collects evidence from multiple sources, verifies findings, and synthesizes a comprehensive answer with citations.

## Quick Start

Choose your deployment method:
- [Docker Compose](#docker-compose) (recommended for local development)
- [Kubernetes (Helm)](#kubernetes-helm) (recommended for production)

### Docker Compose

```bash
# Clone the repository
git clone https://github.com/mayflower/contextmine.git
cd contextmine

# Copy environment template and configure
cp .env.example .env
# Edit .env with your API keys (see Configuration section)

# Start all services
docker compose up -d

# Run database migrations
docker compose exec api sh -c "cd /app/packages/core && alembic upgrade head"
```

### Kubernetes (Helm)

For production deployments, use the Helm chart from GHCR:

```bash
# Create a values file with your configuration
cat > my-values.yaml << EOF
api:
  image:
    repository: ghcr.io/mayflower/contextmine-api
    tag: latest
worker:
  image:
    repository: ghcr.io/mayflower/contextmine-worker
    tag: latest
config:
  publicBaseUrl: "https://contextmine.example.com"
secrets:
  github:
    clientId: "your-github-client-id"
    clientSecret: "your-github-client-secret"
  sessionSecret: "$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
  tokenEncryptionKey: "$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
  openaiApiKey: "sk-..."
EOF

# Install from OCI registry
helm install contextmine oci://ghcr.io/mayflower/contextmine -f my-values.yaml

# Access the application
kubectl port-forward svc/contextmine-api 8000:8000
```

See [deploy/helm/contextmine/README.md](deploy/helm/contextmine/README.md) for full configuration options.

### 2. Create Your First Collection

1. Open the admin UI at **http://localhost:8000**
2. Log in with GitHub OAuth
3. Create a new **Collection** (e.g., "My Docs")
4. Add a **Source**:
   - **Web**: Enter a documentation URL (e.g., `https://docs.python.org/3/`)
   - **GitHub**: Enter `owner/repo` (e.g., `fastapi/fastapi`)
5. Click **Sync** to start indexing

### 3. Connect Your AI Assistant

Configure your MCP client to connect to ContextMine. Authentication is handled via GitHub OAuth automatically.

**Claude Desktop** (`~/.config/claude/claude_desktop_config.json` on Linux, `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "contextmine": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

When you first connect, your MCP client will redirect to GitHub for authentication.

**Cursor**: Settings → MCP → Add server with URL `http://localhost:8000/mcp`

### 4. Start Using It

In your AI assistant, you can now:

```
Search the FastAPI docs for information about dependency injection
```

```
What authentication methods does this codebase support?
```

```
Show me the outline of src/auth/handlers.py
```

## Available MCP Tools

### Context Retrieval

| Tool | Description |
|------|-------------|
| `get_markdown` | **Primary search tool.** Searches indexed content and returns relevant context as Markdown. Supports filtering by collection. |
| `list_collections` | List available documentation collections |
| `list_documents` | Browse documents in a collection |

### Code Intelligence

| Tool | Description |
|------|-------------|
| `outline` | List all functions, classes, and methods in a file with line numbers |
| `find_symbol` | Get the source code of a specific function or class by name |
| `definition` | Jump to where a symbol is defined (requires LSP) |
| `references` | Find all usages of a symbol for impact analysis (requires LSP) |
| `expand` | Explore code relationships - what a function calls, what calls it, imports, etc. |

### Advanced Research

| Tool | Description |
|------|-------------|
| `deep_research` | Multi-step AI agent for complex questions. Autonomously searches, reads code, and builds answers with citations. |

## Configuration

Copy `.env.example` to `.env` and configure these variables:

### Required

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string (default works with docker compose) |
| `GITHUB_CLIENT_ID` | GitHub OAuth app client ID |
| `GITHUB_CLIENT_SECRET` | GitHub OAuth app secret |
| `SESSION_SECRET` | Secret for session cookies |
| `TOKEN_ENCRYPTION_KEY` | Key for encrypting stored tokens |
| `OPENAI_API_KEY` | OpenAI API key for embeddings |

### Optional

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Alternative to OpenAI for embeddings |
| `ANTHROPIC_API_KEY` | For deep_research agent (uses Claude) |
| `MCP_ALLOWED_ORIGINS` | CORS origins for MCP in production |

### Setting Up GitHub OAuth

1. Go to https://github.com/settings/developers
2. Click **New OAuth App**
3. Fill in:
   - **Application name**: ContextMine (or your preferred name)
   - **Homepage URL**: `http://localhost:8000`
   - **Authorization callback URL**: `http://localhost:8000/api/auth/callback`
4. Copy the **Client ID** and **Client Secret** to your `.env`

> **Note**: Both the admin UI and MCP clients use the same callback URL. The server automatically routes OAuth flows to the appropriate handler.

### Generating Secure Keys

```bash
# Generate session secret
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate encryption key
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Adding Sources

### Web Documentation

Best for: API docs, guides, reference documentation

1. Create a collection in the admin UI
2. Add a source with type **Web**
3. Enter the base URL (e.g., `https://docs.example.com/`)
4. The crawler follows links within the same domain

### GitHub Repositories

Best for: Source code, README files, inline documentation

1. Add a source with type **GitHub**
2. Enter the repository as `owner/repo`
3. Optionally specify:
   - **Branch**: defaults to the default branch
   - **Path filter**: limit to specific directories (e.g., `src/`, `docs/`)
4. Code files are parsed for symbols (functions, classes, methods)

**Supported languages for symbol extraction:**
Python, TypeScript, JavaScript, Go, Rust, Java, C, C++, Ruby, PHP

## Architecture

```
┌───────────────────────────────┐     ┌─────────────┐
│     FastAPI + React SPA       │────▶│  PostgreSQL │
│  /api/* /mcp/* /* (frontend)  │     │  + pgvector │
└───────────────────────────────┘     └─────────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
  ┌─────────┐   ┌─────────┐
  │ Prefect │   │ spider  │
  │ Worker  │   │   _md   │
  └─────────┘   └─────────┘
```

- **API** (`apps/api`): FastAPI serving REST API at `/api/*`, MCP at `/mcp/*`, and React frontend at `/*`
- **Web** (`apps/web`): React admin console (built and served by API)
- **Worker** (`apps/worker`): Background sync jobs using Prefect
- **Core** (`packages/core`): Shared models, database, and utilities

## Development

### Prerequisites

- Python 3.12+
- Node.js 20+
- [uv](https://github.com/astral-sh/uv) for Python dependency management
- Docker (for PostgreSQL with pgvector)

### Local Development Setup

```bash
# Start database
docker compose up -d postgres

# Install Python dependencies
uv sync --all-packages

# Run migrations
cd packages/core
DATABASE_URL=postgresql+asyncpg://contextmine:contextmine@localhost:5432/contextmine \
  uv run alembic upgrade head
cd ../..

# Build frontend (one-time, or after frontend changes)
cd apps/web && npm install && npm run build && cd ../..

# Start API server (serves both API and frontend)
STATIC_DIR=apps/web/dist uv run uvicorn apps.api.app.main:app --reload --port 8000
```

For frontend development with hot reload, run the Vite dev server separately:

```bash
# Terminal 1: API server
uv run uvicorn apps.api.app.main:app --reload --port 8000

# Terminal 2: Frontend dev server (proxies API requests to :8000)
cd apps/web && npm run dev
```

### Running Tests

```bash
# All tests
uv run pytest -v

# Specific test file
uv run pytest packages/core/tests/test_treesitter.py -v

# With coverage
uv run pytest --cov=contextmine_core --cov-report=term-missing
```

### Code Quality

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

## Container Images

Pre-built images are available from GitHub Container Registry:

```bash
docker pull ghcr.io/mayflower/contextmine-api:latest
docker pull ghcr.io/mayflower/contextmine-worker:latest
docker pull ghcr.io/mayflower/contextmine-web:latest
```

## Troubleshooting

### "No collections found" in MCP client

1. Ensure you've created at least one collection in the admin UI
2. Check that the collection visibility is set to **Global** (or you're authenticated)
3. Verify you've completed the GitHub OAuth flow when prompted by your MCP client

### Sync not finding documents

1. Check the Prefect UI at http://localhost:4200 for job status
2. For GitHub sources, ensure the repository is accessible
3. For web sources, verify the URL is reachable and returns HTML

### Symbols not being extracted

Symbol extraction works for supported languages only. Check that:
1. The file has a recognized extension (`.py`, `.ts`, `.js`, `.go`, etc.)
2. The sync has completed (symbols are extracted during sync)

## License

MIT License - see [LICENSE](LICENSE) for details.
