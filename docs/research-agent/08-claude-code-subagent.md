# Claude Code Subagent: Repo Researcher

This document describes the Claude Code subagent configuration that acts as a "Repo Researcher", implemented as part of Prompt 8.

## Overview

The repo-researcher subagent provides a clean UX for codebase investigation by:

1. Delegating all research to the `code_deep_research` MCP tool
2. Running in a separate context window (context preservation)
3. Returning concise answers with citations and run_id links
4. Avoiding context pollution in the main Claude Code conversation

## Installation

The subagent is configured via a markdown file at:

```
.claude/agents/repo-researcher.md
```

No additional setup is required. Claude Code will automatically discover and use the subagent based on its description.

## When Claude Code Invokes This Subagent

Claude Code will proactively invoke the repo-researcher subagent when:

- The user asks questions about the codebase (e.g., "How does X work?")
- Investigation requires exploring multiple files
- Understanding code relationships or flows
- Early in tasks when context needs to be gathered

The subagent description emphasizes **early invocation** to front-load context gathering and avoid polluting the main conversation context.

## How It Works

```
User: "How does authentication work in this codebase?"
        │
        ▼
┌─────────────────────────────────────────────────────┐
│ Claude Code Main Agent                              │
│                                                     │
│ 1. Recognizes codebase question                     │
│ 2. Spawns repo-researcher subagent                  │
│ 3. Receives concise answer + citations              │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│ repo-researcher subagent (separate context)         │
│                                                     │
│ 1. Calls code_deep_research MCP tool                │
│ 2. Agent performs multi-step investigation:         │
│    - Hybrid search (BM25 + vector)                  │
│    - LSP symbol resolution                          │
│    - Code graph traversal                           │
│    - Tree-sitter analysis                           │
│ 3. Returns concise answer with citations            │
│ 4. Stores full trace as MCP resources               │
└─────────────────────────────────────────────────────┘
```

## Example Workflows

### Basic Question

```
User: "What is the entry point for the API server?"

Claude (via repo-researcher):
The API server entry point is in apps/api/app/main.py [ev-xyz-001].
The FastAPI app is created and configured with middleware, routes, and
the MCP server mounted at /mcp [ev-xyz-002].

Citations:
- [ev-xyz-001] apps/api/app/main.py:1-25 (bm25)
- [ev-xyz-002] apps/api/app/main.py:45-60 (vector)

Run ID: abc123... (access @research://runs/abc123/report.md for full trace)
```

### Scoped Investigation

```
User: "How do the database models handle relationships? Focus on packages/core"

Claude (via repo-researcher):
[Calls code_deep_research with scope="packages/core/**"]

The SQLAlchemy models use relationship() for foreign key associations...
```

### Accessing Detailed Artifacts

After receiving an answer, you can access detailed artifacts via MCP resources:

```
@research://runs/{run_id}/report.md    - Full markdown report
@research://runs/{run_id}/evidence.json - All collected evidence
@research://runs/{run_id}/trace.json    - Step-by-step execution trace
```

## Configuration

The subagent configuration (`.claude/agents/repo-researcher.md`) includes:

```yaml
---
name: repo-researcher
description: Use proactively for codebase investigation questions...
model: inherit
---
```

### Configuration Options

| Field | Value | Description |
|-------|-------|-------------|
| `name` | `repo-researcher` | Unique identifier for the subagent |
| `description` | (see file) | When Claude should invoke this subagent |
| `model` | `inherit` | Uses the same model as the parent conversation |

## Integration with Research Agent

The subagent leverages the full research agent stack:

| Component | Purpose |
|-----------|---------|
| `code_deep_research` tool | MCP tool that runs the agent (Prompt 3) |
| Hybrid search | BM25 + vector search for evidence retrieval |
| LSP actions | Go to definition, find references (Prompt 4) |
| Tree-sitter actions | AST parsing for symbols and structure (Prompt 5) |
| Code graph actions | Relationship traversal (Prompt 6) |
| Verification | Answer quality validation (Prompt 7) |
| MCP resources | Artifact storage and retrieval (Prompt 1) |

## When NOT to Use the Subagent

The subagent is optimized for investigation questions. For simple operations, use direct tools:

| Operation | Use Direct Tool |
|-----------|-----------------|
| Read a known file | `Read` tool |
| Find files by pattern | `Glob` tool |
| Search for text | `Grep` tool |
| Make edits | `Edit` tool |

The subagent's system prompt explicitly states to prefer direct tools when the user already knows which file they want.

## Context Preservation

A key benefit of the subagent architecture is **context preservation**:

1. **Separate Context Window**: The subagent runs in its own context window
2. **Lean Main Context**: Only the concise answer flows back to the main agent
3. **Heavy Traces Stored**: Full investigation details are stored as MCP resources
4. **On-Demand Access**: Users can access detailed artifacts when needed

This keeps the main Claude Code conversation focused and prevents context window exhaustion from verbose code exploration.

## Troubleshooting

### Subagent Not Being Invoked

- Ensure `.claude/agents/repo-researcher.md` exists
- Check that the file has valid YAML frontmatter
- Verify the description matches the type of questions you're asking

### Empty or Poor Results

- Increase the `budget` parameter (up to 20 for complex questions)
- Use `scope` to limit search to relevant directories
- Check that the codebase is indexed in the ContextMine collection

### Accessing Traces

- Set `debug=true` in `code_deep_research` to include run_id
- Use `@research://runs/{run_id}/...` to access artifacts
- Check `research://runs` for recent run history
