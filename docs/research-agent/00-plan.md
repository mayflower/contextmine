# Research Agent Integration Plan

This document describes the architecture and integration plan for adding an internal "research agent" (R2R-like pattern) to the ContextMine MCP server, with LSP, Tree-sitter, and graph-based retrieval.

## Current Architecture

### Module Map

```
opencontext/
├── apps/
│   ├── api/                          # FastAPI backend
│   │   ├── app/
│   │   │   ├── main.py               # FastAPI app, mounts MCP at /mcp
│   │   │   ├── mcp_server.py         # FastMCP server, 3 tools
│   │   │   └── routes/               # REST API routes
│   │   │       └── search.py         # /api/search endpoint
│   │   └── tests/
│   ├── worker/                       # Prefect worker
│   │   └── contextmine_worker/
│   │       ├── flows.py              # Sync flows (GitHub, web)
│   │       ├── chunking.py           # Document chunking logic
│   │       └── spider.py             # Web crawler wrapper
│   └── web/                          # React frontend
├── packages/
│   └── core/                         # Shared Python library
│       └── contextmine_core/
│           ├── models.py             # SQLAlchemy models (Document, Chunk, etc.)
│           ├── search.py             # Hybrid search (FTS + vector + RRF)
│           ├── embeddings.py         # Embedding providers (OpenAI, Gemini)
│           ├── llm.py                # LLM providers for context assembly
│           └── settings.py           # Configuration via pydantic-settings
└── rust/
    └── spider_md/                    # Rust web crawler binary
```

### Call Graph: Search Flow

```
MCP Tool: context.get_markdown (mcp_server.py:284)
    │
    ├── get_accessible_collection_ids() → access control
    │
    ├── get_embedder() → OpenAI/Gemini/Fake embedder
    │   └── embed_batch([query]) → query embedding
    │
    ├── hybrid_search() (search.py:89)
    │   ├── search_fts() → PostgreSQL ts_rank_cd
    │   ├── search_vector() → pgvector cosine distance
    │   └── compute_rrf_scores() → Reciprocal Rank Fusion (k=60)
    │
    └── assemble_context() (llm.py) → LLM synthesizes response
```

### Current MCP Tools

| Tool | Purpose | Output Size |
|------|---------|-------------|
| `context.list_collections` | Discover collections | Small (list) |
| `context.list_documents` | Browse documents in collection | Medium (up to 50 docs) |
| `context.get_markdown` | Search + retrieve context | Large (up to 4000 tokens) |

### Current Retrieval Pipeline

1. **FTS**: PostgreSQL `plainto_tsquery` with `ts_rank_cd` scoring
2. **Vector**: pgvector HNSW index with cosine distance
3. **Fusion**: RRF with k=60, combining ranks from both systems
4. **Assembly**: Optional LLM synthesis via `assemble_context()`

### What's NOT Implemented

- **MCP Resources**: No `resources/list` or `resources/read` handlers
- **LSP Integration**: No language server support
- **Tree-sitter**: No AST-based parsing
- **Code Graph**: No symbol relationships or call graph
- **Internal Agent Loop**: No multi-step reasoning

---

## Integration Points

### A. New High-Level Tool: `code_deep_research`

**Location**: `apps/api/app/mcp_server.py`

**Hook Point**: Add new `@mcp.tool()` decorator alongside existing tools.

```python
@mcp.tool(name="code_deep_research")
async def code_deep_research(
    question: Annotated[str, "Research question about the codebase"],
    scope: Annotated[str | None, "Limit to path pattern"] = None,
    budget: Annotated[int, "Max agent steps"] = 10,
    debug: Annotated[bool, "Include run_id for trace inspection"] = False,
) -> str:
    """Deep research with multi-step reasoning, LSP grounding, and graph traversal."""
```

**Key Constraint**: Output must be small (<800 tokens). Heavy artifacts stored as resources.

### B. MCP Resources for Artifact Store

**Location**: `apps/api/app/mcp_server.py`

**Hook Point**: Add `@mcp.resource()` handlers using FastMCP's resource decorator.

```python
@mcp.resource("research://runs")
async def list_runs() -> list[Resource]:
    """List recent research runs."""

@mcp.resource("research://runs/{run_id}/trace.json")
async def get_trace(run_id: str) -> str:
    """Get detailed trace for a run."""

@mcp.resource("research://runs/{run_id}/evidence.json")
async def get_evidence(run_id: str) -> str:
    """Get collected evidence for a run."""

@mcp.resource("research://runs/{run_id}/report.md")
async def get_report(run_id: str) -> str:
    """Get markdown report for a run."""
```

**URI Scheme**:
- `research://runs` - list all runs
- `research://runs/<run_id>/trace.json` - agent trace
- `research://runs/<run_id>/evidence.json` - collected evidence
- `research://runs/<run_id>/report.md` - human-readable report

### C. LSP Manager

**New Location**: `packages/core/contextmine_core/lsp/`

**Integration**: Called by research agent actions when symbol-centric queries detected.

```
packages/core/contextmine_core/lsp/
├── __init__.py
├── manager.py          # LspManager: spawn/manage language servers
├── client.py           # JSON-RPC client for LSP protocol
└── languages.py        # Language detection, server configs
```

**Dependencies**: External language servers (user-installed):
- Python: `pylsp` or `pyright`
- TypeScript/JS: `typescript-language-server`
- Go: `gopls`
- Rust: `rust-analyzer`

### D. Tree-sitter Parser

**New Location**: `packages/core/contextmine_core/treesitter/`

```
packages/core/contextmine_core/treesitter/
├── __init__.py
├── manager.py          # TreeSitterManager: parse files, cache trees
├── outline.py          # Symbol extraction (functions, classes)
└── languages.py        # Language grammars (py-tree-sitter-languages)
```

**Integration with Chunking**: Improve `apps/worker/contextmine_worker/chunking.py` to use symbol boundaries instead of naive fixed-size splits.

### E. Code Graph Store

**New Location**: `packages/core/contextmine_core/graph/`

```
packages/core/contextmine_core/graph/
├── __init__.py
├── store.py            # CodeGraph: SQLite or PostgreSQL-based
├── builder.py          # Build graph from LSP + Tree-sitter
└── retrieval.py        # Graph queries (expand, pack, trace)
```

**Schema** (new tables or SQLite file):
- `symbols`: id, qualified_name, kind, file_path, start_line, end_line
- `symbol_edges`: from_symbol, to_symbol, edge_type (CALLS, REFERENCES, IMPORTS)

---

## Staged Implementation Plan

### Stage 1: Research Run Infrastructure + Artifact Store (Prompt 1)

**Goal**: Create `ResearchRun` concept and expose artifacts via MCP resources.

**Files to Create**:
- `packages/core/contextmine_core/research/run.py` - ResearchRun dataclass
- `packages/core/contextmine_core/research/artifacts.py` - ArtifactStore (memory + file)
- `apps/api/app/mcp_resources.py` - MCP resource handlers

**Files to Modify**:
- `apps/api/app/mcp_server.py` - import and register resource handlers

**Config**:
- `ARTIFACT_STORE`: `memory` or `file`
- `ARTIFACT_DIR`: Directory for file-backed store (default `.mcp_artifacts/`)
- `ARTIFACT_TTL_MINUTES`: Eviction TTL (default 60)

### Stage 2: Server-Side LLM Provider (Prompt 2)

**Goal**: Add LLM provider abstraction for internal agent calls.

**Files to Create**:
- `packages/core/contextmine_core/research/llm/__init__.py`
- `packages/core/contextmine_core/research/llm/provider.py` - LLMProvider protocol
- `packages/core/contextmine_core/research/llm/anthropic.py` - AnthropicProvider
- `packages/core/contextmine_core/research/llm/mock.py` - MockProvider for tests

**Integration**: Reuse existing `contextmine_core/llm.py` patterns but with strict JSON schema validation.

**Config**:
- `ANTHROPIC_API_KEY`: API key for Anthropic
- `RESEARCH_MODEL`: Model to use (default `claude-sonnet-4-20250514`)
- `RESEARCH_MAX_TOKENS`: Max tokens per call (default 4096)

### Stage 3: Research Agent Loop + `code_deep_research` Tool (Prompt 3)

**Goal**: Implement the internal agentic loop and expose as MCP tool.

**Files to Create**:
- `packages/core/contextmine_core/research/agent.py` - ResearchAgent class
- `packages/core/contextmine_core/research/actions/__init__.py`
- `packages/core/contextmine_core/research/actions/registry.py` - ActionRegistry
- `packages/core/contextmine_core/research/actions/search.py` - hybrid_search wrapper
- `packages/core/contextmine_core/research/actions/span.py` - open_span action
- `packages/core/contextmine_core/research/actions/summarize.py` - summarize_evidence
- `packages/core/contextmine_core/research/actions/finalize.py` - finalize action

**Files to Modify**:
- `apps/api/app/mcp_server.py` - add `code_deep_research` tool

### Stage 4: LSP Manager + Actions (Prompt 4)

**Goal**: Add LSP-based code intelligence actions.

**Files to Create**:
- `packages/core/contextmine_core/lsp/manager.py`
- `packages/core/contextmine_core/lsp/client.py`
- `packages/core/contextmine_core/research/actions/lsp.py`

**New Actions**:
- `lsp_definition(file, line, char)`
- `lsp_references(file, line, char)`
- `lsp_hover(file, line, char)`
- `lsp_diagnostics(paths)`

### Stage 5: Tree-sitter Parsing (Prompt 5)

**Goal**: Add robust symbol boundary extraction.

**Files to Create**:
- `packages/core/contextmine_core/treesitter/manager.py`
- `packages/core/contextmine_core/treesitter/outline.py`
- `packages/core/contextmine_core/research/actions/treesitter.py`

**Files to Modify**:
- `apps/worker/contextmine_worker/chunking.py` - use symbol-level chunks

**New Actions**:
- `ts_outline(file)`
- `ts_find_symbol(name)`
- `ts_enclosing_symbol(file, line)`

### Stage 6: Code Graph + GraphRAG (Prompt 6)

**Goal**: Add typed graph for multi-hop evidence selection.

**Files to Create**:
- `packages/core/contextmine_core/graph/store.py`
- `packages/core/contextmine_core/graph/builder.py`
- `packages/core/contextmine_core/graph/retrieval.py`
- `packages/core/contextmine_core/research/actions/graph.py`

**New Actions**:
- `graph_expand(seeds, edge_types, depth)`
- `graph_pack(subgraph)` - minimal evidence set
- `graph_trace(from_symbol, to_symbol)` - impact analysis

### Stage 7: Verification + Evaluation (Prompt 7)

**Goal**: Add verification hooks and evaluation harness.

**Files to Create**:
- `packages/core/contextmine_core/research/verification.py`
- `eval/questions/*.json` - test questions
- `eval/runner.py` - evaluation runner

**Files to Modify**:
- `packages/core/contextmine_core/research/agent.py` - add verification hooks

---

## Do Not List

The following should NOT be changed unless strictly necessary:

1. **RRF Ranking Parameters**: Keep k=60 and existing score combination logic
2. **Embedding Dimensions**: Keep 1536 default (OpenAI text-embedding-3-small)
3. **Chunk Size/Overlap**: Keep 1500/200 defaults unless Tree-sitter integration requires changes
4. **Index Formats**: Keep existing pgvector HNSW and GIN indexes
5. **Existing Tool Signatures**: `context.list_collections`, `context.list_documents`, `context.get_markdown` should remain unchanged
6. **Authentication Flow**: Keep existing MCPAuthMiddleware and bearer token validation

---

## Key Integration Points Summary

| Component | File Path | Key Function/Class |
|-----------|-----------|-------------------|
| MCP Tool Registration | `apps/api/app/mcp_server.py` | `@mcp.tool()` decorator |
| MCP Resources | `apps/api/app/mcp_server.py` | `@mcp.resource()` decorator (to add) |
| Hybrid Search | `packages/core/contextmine_core/search.py` | `hybrid_search()` |
| RRF Scoring | `packages/core/contextmine_core/search.py` | `compute_rrf_scores()` |
| Embeddings | `packages/core/contextmine_core/embeddings.py` | `OpenAIEmbedder`, `GeminiEmbedder` |
| LLM Assembly | `packages/core/contextmine_core/llm.py` | `assemble_context()` |
| Chunking | `apps/worker/contextmine_worker/chunking.py` | `chunk_text()`, `split_markdown_preserving_code_fences()` |
| Document Model | `packages/core/contextmine_core/models.py` | `Document`, `Chunk` |
| Settings | `packages/core/contextmine_core/settings.py` | `Settings` |
| FastMCP Server | `apps/api/app/mcp_server.py` | `mcp = FastMCP(...)` |
| Auth Middleware | `apps/api/app/mcp_server.py` | `MCPAuthMiddleware` |
