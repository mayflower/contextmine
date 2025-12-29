# Research Agent Artifacts

This document describes the artifact storage system for research agent runs and how to access them in Claude Code.

## Overview

When the research agent investigates a question about your codebase, it produces several artifacts:

- **Trace**: Step-by-step execution log showing actions taken, timing, and errors
- **Evidence**: Code snippets and documentation spans that support the answer
- **Report**: Human-readable markdown summary of the investigation

These artifacts are stored in an artifact store and exposed via MCP resources, allowing you to inspect them without polluting your main context.

## Configuration

Configure artifact storage via environment variables:

```bash
# Artifact storage type: 'memory' or 'file'
ARTIFACT_STORE=file

# Directory for file-backed store (default: .mcp_artifacts)
ARTIFACT_DIR=.mcp_artifacts

# How long to keep artifacts in minutes (default: 60)
ARTIFACT_TTL_MINUTES=60

# Maximum number of runs to keep (default: 100)
ARTIFACT_MAX_RUNS=100
```

### Memory vs File Store

| Feature | Memory Store | File Store |
|---------|--------------|------------|
| Persistence | Lost on restart | Survives restarts |
| Performance | Faster | Slightly slower |
| Use case | Development, testing | Production |

## MCP Resources

Research artifacts are exposed via MCP resources with the `research://` URI scheme.

### Listing Runs

```
research://runs
```

Returns a JSON list of recent research runs (newest first):

```json
[
  {
    "run_id": "550e8400-e29b-41d4-a716-446655440000",
    "question": "How does authentication work?",
    "status": "done",
    "created_at": "2024-01-15T10:30:00Z",
    "completed_at": "2024-01-15T10:30:05Z"
  }
]
```

### Getting Trace

```
research://runs/{run_id}/trace.json
```

Returns the execution trace showing each step:

```json
{
  "run_id": "550e8400-...",
  "question": "How does authentication work?",
  "status": "done",
  "steps": [
    {
      "step_number": 1,
      "action": "hybrid_search",
      "input": {"query": "authentication middleware"},
      "output_summary": "Found 5 results",
      "duration_ms": 150,
      "evidence_ids": ["ev-001", "ev-002"]
    }
  ]
}
```

### Getting Evidence

```
research://runs/{run_id}/evidence.json
```

Returns all collected evidence:

```json
{
  "run_id": "550e8400-...",
  "evidence_count": 3,
  "evidence": [
    {
      "id": "ev-001",
      "file_path": "src/auth/middleware.py",
      "start_line": 45,
      "end_line": 60,
      "content": "class AuthMiddleware:\n    ...",
      "reason": "Main authentication middleware",
      "provenance": "bm25",
      "score": 0.92
    }
  ]
}
```

### Getting Report

```
research://runs/{run_id}/report.md
```

Returns a human-readable markdown report:

```markdown
# Research Report: 550e8400

**Question:** How does authentication work?
**Status:** done
**Duration:** 350ms

## Answer

Authentication is handled by AuthMiddleware in src/auth/middleware.py...

## Evidence

### [ev-001] src/auth/middleware.py:45-60
**Reason:** Main authentication middleware
...
```

## Using in Claude Code

In Claude Code, you can reference research artifacts using `@` mentions:

1. After a research run completes, note the `run_id` in the response
2. Use `@research://runs/{run_id}/report.md` to view the full report
3. Use `@research://runs/{run_id}/evidence.json` to inspect raw evidence

Example workflow:

```
User: How does the search ranking work?

Claude: [Calls code_deep_research tool]

Response: The search uses hybrid retrieval combining BM25 and vector search,
with RRF (Reciprocal Rank Fusion) for score combination.

Key citations:
- packages/core/contextmine_core/search.py:52-75 (RRF implementation)
- packages/core/contextmine_core/search.py:89-120 (hybrid_search function)

Run ID: abc123-... (use @research://runs/abc123-.../report.md for details)
```

You can then reference `@research://runs/abc123-.../report.md` to see the full investigation trace and evidence.

## Data Structures

### ResearchRun

The core data structure for a research investigation:

```python
@dataclass
class ResearchRun:
    run_id: str               # Unique identifier
    question: str             # The research question
    status: RunStatus         # running | done | error
    created_at: datetime
    completed_at: datetime | None
    steps: list[ActionStep]   # Execution trace
    evidence: list[Evidence]  # Collected evidence
    answer: str | None        # Final answer
    error_message: str | None
    scope: str | None         # Path filter
    budget_steps: int         # Max allowed steps
    budget_used: int          # Steps actually used
    total_duration_ms: int
```

### Evidence

A piece of evidence collected during research:

```python
@dataclass
class Evidence:
    id: str               # Unique ID (e.g., "ev-001")
    file_path: str        # Path to source file
    start_line: int       # Starting line (1-indexed)
    end_line: int         # Ending line (inclusive)
    content: str          # The actual code/text
    reason: str           # Why this was selected
    provenance: str       # How found: bm25, vector, lsp, graph, manual
    score: float | None   # Relevance score
    symbol_id: str | None # Symbol name if LSP-derived
    symbol_kind: str | None # function, class, method, etc.
```

### ActionStep

A single step in the agent's execution:

```python
@dataclass
class ActionStep:
    step_number: int      # Sequential number (1-indexed)
    action: str           # Action name (e.g., "hybrid_search")
    input: dict           # Input parameters
    output_summary: str   # Brief output description
    duration_ms: int      # Execution time
    error: str | None     # Error message if failed
    evidence_ids: list[str]  # Evidence collected in this step
```

## Eviction Policy

Artifacts are automatically cleaned up based on:

1. **TTL (Time-to-Live)**: Runs older than `ARTIFACT_TTL_MINUTES` are evicted
2. **Max Runs**: When exceeding `ARTIFACT_MAX_RUNS`, oldest runs are removed

Eviction happens:
- On new run creation (enforces max_runs limit)
- Periodically via `evict_expired()` (if called explicitly)

## Security Considerations

- Artifacts may contain sensitive code snippets from your repository
- The file store directory (`.mcp_artifacts/`) is automatically added to `.gitignore`
- Access to MCP resources requires a valid API token (same auth as other MCP tools)
