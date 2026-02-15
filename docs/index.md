# ContextMine

ContextMine is a self-hosted documentation and code indexing system that exposes context via the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/). Connect it to Claude Desktop, Cursor, or any MCP-compatible AI assistant to provide rich context for code understanding, documentation lookup, and codebase exploration.

## Key Features

- **Hybrid Search** - Full-text + vector similarity with RRF (Reciprocal Rank Fusion) ranking for accurate retrieval
- **Deep Research Agent** - Multi-step AI agent with LSP and Tree-sitter for complex codebase questions
- **Code Intelligence** - Symbol extraction, code outlines, and structural navigation via Tree-sitter
- **Knowledge Graph** - GraphRAG-powered retrieval with business rules, data models, and architecture
- **Web Crawling** - Index documentation sites automatically
- **Git Indexing** - Index GitHub repositories with incremental updates
- **Self-Hosted** - Your data stays on your infrastructure

## Architecture Overview

```
┌───────────────────────────────┐     ┌─────────────┐
│     FastAPI + React SPA       │────▶│  PostgreSQL │
│  /api/* /mcp/* /* (frontend)  │     │    pg4ai    │
└───────────────────────────────┘     └─────────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
  ┌─────────┐   ┌─────────┐
  │ Prefect │   │ spider  │
  │ Worker  │   │   _md   │
  └─────────┘   └─────────┘
```

### Components

| Component | Description |
|-----------|-------------|
| **API** (`apps/api`) | FastAPI serving REST API at `/api/*`, MCP at `/mcp/*`, and React frontend at `/*` |
| **Web** (`apps/web`) | React admin console for managing collections and sources |
| **Worker** (`apps/worker`) | Background sync jobs using Prefect for scheduled indexing |
| **Core** (`packages/core`) | Shared Python library with models, database, and services |
| **spider_md** (`rust/spider_md`) | Rust-based web crawler for HTML to Markdown conversion |

## Quick Start

### Docker Compose (Development)

```bash
# Clone the repository
git clone https://github.com/mayflower/contextmine.git
cd contextmine

# Copy environment template and configure
cp .env.example .env
# Edit .env with your API keys

# Start all services
docker compose up -d

# Run database migrations
docker compose exec api sh -c "cd /app/packages/core && alembic upgrade head"
```

### Kubernetes (Production)

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
```

## Configuration

### Required Environment Variables

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string |
| `GITHUB_CLIENT_ID` | GitHub OAuth app client ID |
| `GITHUB_CLIENT_SECRET` | GitHub OAuth app secret |
| `SESSION_SECRET` | Secret for session cookies |
| `TOKEN_ENCRYPTION_KEY` | Key for encrypting stored tokens |
| `OPENAI_API_KEY` | OpenAI API key for embeddings |

### Optional Environment Variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Alternative to OpenAI for embeddings |
| `ANTHROPIC_API_KEY` | For deep_research agent (uses Claude) |
| `DEFAULT_LLM_PROVIDER` | LLM provider: `openai`, `anthropic`, or `gemini` |
| `DEFAULT_LLM_MODEL` | Model name (e.g., `claude-haiku-4-5-20251001`) |
| `MCP_ALLOWED_ORIGINS` | CORS origins for MCP in production |

### Setting Up GitHub OAuth

1. Go to [GitHub Developer Settings](https://github.com/settings/developers)
2. Click **New OAuth App**
3. Fill in:
   - **Application name**: ContextMine
   - **Homepage URL**: `http://localhost:8000`
   - **Authorization callback URL**: `http://localhost:8000/api/auth/callback`
4. Copy the **Client ID** and **Client Secret** to your `.env`

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Python 3.12, FastAPI, SQLAlchemy 2.x async, Alembic |
| **Database** | pg4ai (PostgreSQL with pgvector + Apache AGE) |
| **Frontend** | React, Vite, TypeScript |
| **Orchestration** | Prefect for scheduled sync jobs |
| **Crawling** | spider-rs (Rust binary for HTML→Markdown) |
| **Retrieval** | Hybrid FTS + vector search with RRF ranking |
| **LLM Providers** | OpenAI, Anthropic, Gemini |

## Adding Sources

### Web Documentation

Best for API docs, guides, and reference documentation.

1. Create a collection in the admin UI
2. Add a source with type **Web**
3. Enter the base URL (e.g., `https://docs.example.com/`)
4. The crawler follows links within the same domain

### GitHub Repositories

Best for source code, README files, and inline documentation.

1. Add a source with type **GitHub**
2. Enter the repository as `owner/repo`
3. Optionally specify:
   - **Branch**: defaults to the default branch
   - **Path filter**: limit to specific directories

**Supported languages for symbol extraction:**
Python, TypeScript, JavaScript, Go, Rust, Java, C, C++, Ruby, PHP

## Knowledge Graph Features

ContextMine builds a knowledge graph from indexed content that powers advanced retrieval:

- **Business Rules** - Extracted validation logic with LLM labeling
- **Data Model** - Database schema from Alembic migrations
- **System Surface** - API endpoints, GraphQL types, Protobuf services
- **GraphRAG** - Microsoft GraphRAG-style retrieval with community detection
- **Code Intelligence** - Symbols, relationships, and call graphs

See [Knowledge Graph](KNOWLEDGE_GRAPH.md) for implementation details.

## Troubleshooting

### "No collections found" in MCP client

1. Ensure you've created at least one collection in the admin UI
2. Check that the collection visibility is set to **Global**
3. Verify you've completed the GitHub OAuth flow

### Sync not finding documents

1. Check the Prefect UI at `http://localhost:4200` for job status
2. For GitHub sources, ensure the repository is accessible
3. For web sources, verify the URL is reachable

### Symbols not being extracted

Symbol extraction works for supported languages only:

1. Check the file has a recognized extension (`.py`, `.ts`, `.js`, `.go`, etc.)
2. Verify the sync has completed (symbols are extracted during sync)

## License

MIT License - see [LICENSE](https://github.com/mayflower/contextmine/blob/main/LICENSE) for details.
