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

## Analyzer Implementation Progress

Tracking implementation of the Knowledge Graph / Derived Knowledge subsystem (see [Knowledge Graph docs](docs/KNOWLEDGE_GRAPH.md)).

| Step | Description | Status | Notes |
|------|-------------|--------|-------|
| 1 | Knowledge Graph storage layer | DONE | Models + migration 013 |
| 2 | Graph builder from indexing output | DONE | Builder + tests (skipped for SQLite) |
| 3 | ERM extraction + Mermaid ERD | DONE | AST parser + 10 tests |
| 4 | System Surface Catalog | DONE | OpenAPI, GraphQL, Proto, Jobs |
| 5 | Business Rule candidate mining | DONE | Tree-sitter AST + 15 tests |
| 6 | LLM labeling (RuleCandidate→BusinessRule) | DONE | Pydantic schemas + 12 tests |
| 7 | GraphRAG retrieval | DONE | Bundle + neighborhood + path + 12 tests |
| 8 | MCP tools for Claude Code | DONE | 7 tools + 14 tests |
| 9 | arc42 Architecture Twin | DONE | Generator + drift report + 17 tests |
| 10 | Final hardening + e2e tests | PENDING | |

### Step 1 Summary (Knowledge Graph Storage Layer)

**Schema decisions:**
- Generic node/edge tables with `kind` enum instead of one-table-per-concept
- Natural key constraint: `(collection_id, kind, natural_key)` for idempotent upserts
- Evidence stored separately with links to nodes/edges/artifacts via join tables
- Column name: `meta` (not `metadata` - reserved by SQLAlchemy)

**Files created/modified:**
- `packages/core/contextmine_core/models.py` - Added enums + 7 new tables
- `packages/core/alembic/versions/013_add_knowledge_graph.py` - Migration
- `packages/core/contextmine_core/knowledge/__init__.py` - Module init
- `packages/core/contextmine_core/knowledge/schemas.py` - Pydantic schemas

### Step 2 Summary (Graph Builder)

**What was built:**
- `build_knowledge_graph_for_source()` - Creates FILE/SYMBOL nodes and edges
- `cleanup_orphan_nodes()` - Removes nodes for deleted documents
- Edges: FILE_DEFINES_SYMBOL, SYMBOL_CONTAINS_SYMBOL, plus symbol edges from SymbolEdge table
- Evidence creation linking nodes to source locations

**Integration point:**
- Call `build_knowledge_graph_for_source(session, source_id)` after symbol indexing in sync pipeline

**Files created:**
- `packages/core/contextmine_core/knowledge/builder.py` - Builder functions
- `packages/core/tests/test_knowledge_graph.py` - Tests (require PostgreSQL)

### Step 3 Summary (ERM Extraction + Mermaid ERD)

**What was built:**
- `alembic.py` - AST-based parser for Alembic migration files (no regex)
  - Extracts `op.create_table()`, `op.add_column()`, `op.create_foreign_key()` calls
  - Parses column definitions including types, nullability, primary keys, foreign keys
- `erm.py` - ERM schema builder and Mermaid ERD generator
  - `ERMExtractor` - Consolidates schema from multiple migration files
  - `generate_mermaid_erd()` - Creates Mermaid ER diagram syntax
  - `build_erm_graph()` - Creates DB_TABLE/DB_COLUMN nodes and edges
  - `save_erd_artifact()` - Stores ERD as a KnowledgeArtifact

**Integration point:**
- Call `ERMExtractor.extract_from_directory(alembic_dir)` to parse migrations
- Call `build_erm_graph(session, collection_id, schema)` to populate knowledge graph
- Call `save_erd_artifact(session, collection_id, schema)` to store Mermaid ERD

**Files created:**
- `packages/core/contextmine_core/analyzer/__init__.py` - Module init
- `packages/core/contextmine_core/analyzer/extractors/__init__.py` - Extractors init
- `packages/core/contextmine_core/analyzer/extractors/alembic.py` - Alembic parser
- `packages/core/contextmine_core/analyzer/extractors/erm.py` - ERM builder
- `packages/core/tests/test_erm_extractor.py` - 10 tests (all passing)

### Step 4 Summary (System Surface Catalog)

**What was built:**
- `openapi.py` - OpenAPI 3.x specification parser (YAML/JSON)
  - Extracts endpoints, operations, request/response schemas
- `graphql.py` - GraphQL schema parser
  - Extracts types, operations (Query/Mutation/Subscription), fields
- `protobuf.py` - Protobuf (.proto) parser
  - Extracts messages, services, RPCs, enums
- `jobs.py` - Job definition parser for:
  - GitHub Actions workflows (handles YAML 1.1 `on` → True issue)
  - Kubernetes CronJobs
  - Prefect deployments
- `surface.py` - Unified surface catalog builder
  - `SurfaceCatalogExtractor` - Auto-detects and processes spec files
  - `build_surface_graph()` - Creates knowledge graph nodes/edges

**Node kinds added:**
- GRAPHQL_TYPE, SERVICE_RPC (added to models.py)

**Edge kinds added:**
- RPC_USES_MESSAGE (added to models.py)

**Integration point:**
- Call `SurfaceCatalogExtractor().add_file(path, content)` for each spec file
- Call `build_surface_graph(session, collection_id, catalog)` to populate knowledge graph

**Files created:**
- `packages/core/contextmine_core/analyzer/extractors/openapi.py`
- `packages/core/contextmine_core/analyzer/extractors/graphql.py`
- `packages/core/contextmine_core/analyzer/extractors/protobuf.py`
- `packages/core/contextmine_core/analyzer/extractors/jobs.py`
- `packages/core/contextmine_core/analyzer/extractors/surface.py`
- `packages/core/tests/test_surface_extractors.py` - 14 tests (all passing)

### Step 5 Summary (Business Rule Candidate Mining)

**What was built:**
- `rules.py` - Rule candidate extractor using Tree-sitter AST
  - Detects conditional branches leading to failure actions
  - Python: `if condition: raise Exception` patterns
  - Python: `assert` statements
  - TypeScript/JavaScript: `if (condition) throw Error` patterns
  - Captures predicate text, failure text, container function, evidence
  - Heuristic confidence scoring based on validation keywords

**Failure kinds detected:**
- RAISE_EXCEPTION (Python raise)
- THROW_ERROR (JS/TS throw)
- RETURN_ERROR (return null/None/error)
- ASSERT_FAIL (assert statements)

**Integration point:**
- Call `extract_rule_candidates(file_path, content)` to get candidates
- Call `build_rule_candidates_graph(session, collection_id, extractions)` to populate knowledge graph
- Natural key: `rule:{file_path}:{start_line}:{content_hash}` for idempotent upserts

**Files created:**
- `packages/core/contextmine_core/analyzer/extractors/rules.py` - Rule extractor
- `packages/core/tests/test_rule_extractor.py` - 15 tests (all passing)

### Step 6 Summary (LLM Labeling)

**What was built:**
- `labeling.py` - LLM-based rule candidate labeling service
  - `BusinessRuleOutput` - Pydantic schema for structured LLM output
  - `label_rule_candidates()` - Main labeling function
  - Content hash for idempotency (skips unchanged candidates)
  - Creates BUSINESS_RULE nodes with edges to source candidates
  - Links evidence and citations

**Key features:**
- Temperature 0 for deterministic output
- Strict JSON schema validation via Pydantic
- LLM only labels, never discovers new rules
- Idempotent: content hash prevents relabeling unchanged candidates
- Categories: validation, authorization, invariant, constraint, other
- Severity levels: error, warning, info

**Integration point:**
- Call `label_rule_candidates(session, collection_id, provider)` after rule candidate mining
- Pass a configured LLMProvider (from `contextmine_core.research.llm`)

**Files created:**
- `packages/core/contextmine_core/analyzer/labeling.py` - Labeling service
- `packages/core/tests/test_rule_labeling.py` - 12 tests (all passing)

### Step 7 Summary (GraphRAG Retrieval)

**What was built:**
- `graphrag.py` - Graph-augmented retrieval service with Microsoft GraphRAG approach:
  - `graph_rag_context()` - Context retrieval combining:
    1. Hybrid search to find relevant documents/chunks
    2. Mapping search hits to Knowledge Graph nodes
    3. Expanding graph neighborhood (configurable depth)
    4. Gathering evidence citations
    5. Building ContextPack with communities + entities + edges
  - `graph_rag_query()` - Full map-reduce answering using LLM
  - `graph_neighborhood()` - Local exploration from a single node
  - `trace_path()` - BFS shortest path between two nodes

**Key features:**
- Maps search results to FILE nodes via document_id or URI
- BFS-based neighborhood expansion with depth limit
- Community-aware retrieval (global + local context)
- Evidence gathering from KnowledgeEvidence table
- Markdown rendering with node categorization (FILE, SYMBOL, DB_TABLE, etc.)
- Returns ContextPack with communities, entities, edges, paths, citations

**Output formats:**
- `ContextPack.to_markdown()` - Human-readable summary with citations
- `ContextPack.to_dict()` - JSON-serializable structure
- Evidence citations with file_path:start_line-end_line format

**Integration point:**
- Call `graph_rag_context(session, query, collection_id, user_id)` for context retrieval
- Call `graph_rag_query(session, query, collection_id, user_id, provider)` for answered queries
- Call `graph_neighborhood(session, node_id)` for local exploration
- Call `trace_path(session, from_node_id, to_node_id)` for dependency analysis

**Files created:**
- `packages/core/contextmine_core/graphrag.py` - GraphRAG service
- `packages/core/tests/test_graphrag.py` - 12 tests (all passing)

### Step 8 Summary (MCP Tools for Claude Code)

**What was built:**
7 new MCP tools exposed via FastMCP server for Claude Code / Cursor:
1. `list_business_rules(collection_id?, query?)` - List extracted business rules
2. `get_business_rule(rule_id)` - Full rule details with evidence
3. `get_erd(collection_id?, format?)` - ERD as Mermaid or JSON
4. `list_system_surfaces(collection_id?, kind?, limit?)` - API endpoints, jobs, schemas
5. `graph_neighborhood(node_id, depth?, edge_kinds?, limit?)` - Local graph exploration
6. `trace_path(from_node_id, to_node_id, max_hops?)` - Shortest path between nodes
7. `graph_rag(query, collection_id?, max_depth?, max_results?)` - Graph-augmented retrieval

**Key features:**
- All tools return Markdown for assistant consumption
- Structured data available via parameters (format="json")
- Input validation with helpful error messages
- Access control respects collection visibility
- Depth/hop limits capped for safety
- Updated MCP server instructions to guide tool selection

**Integration:**
- Tools are auto-registered via @mcp.tool() decorator
- Access the MCP server at `/mcp` endpoint
- Tools call into graphrag.py, models.py, and search.py

**Files modified:**
- `apps/api/app/mcp_server.py` - Added 7 new tools + updated instructions

**Files created:**
- `apps/api/tests/test_mcp_knowledge.py` - 14 tests (all passing)

### Step 9 Summary (arc42 Architecture Twin)

**What was built:**
- `arc42.py` - Architecture documentation generator
  - `generate_arc42()` - Generates full arc42 document from extracted facts
  - `save_arc42_artifact()` - Stores document as KnowledgeArtifact
  - `compute_drift_report()` - Compares stored vs current state

**arc42 Sections generated:**
1. Context - System boundary, external interfaces (API counts)
2. Building Blocks - Components, database schema, symbol counts
3. Runtime View - Entry points, execution flows
4. Deployment View - Jobs, workflows from manifests
5. Crosscutting Concepts - Validation patterns, security hints
6. Risks & Technical Debt - Unreviewed candidates, TODOs
7. Glossary - Domain terms from database schema

**Key features:**
- Every statement is evidence-backed or explicitly marked "inferred"
- Drift detection compares stored artifact with current graph state
- Supports caching with `regenerate` parameter
- Section-specific retrieval via `section` parameter

**MCP tools added:**
- `get_arc42(collection_id?, section?, regenerate?)` - Get architecture doc
- `arc42_drift_report(collection_id?)` - Show what changed

**Files created:**
- `packages/core/contextmine_core/analyzer/arc42.py` - Generator
- `packages/core/tests/test_arc42.py` - 10 tests

**Files modified:**
- `apps/api/app/mcp_server.py` - Added 2 arc42 MCP tools
- `apps/api/tests/test_mcp_knowledge.py` - Added 7 arc42 schema tests
- `packages/core/contextmine_core/models.py` - Fixed MERMAID_ERD enum name
