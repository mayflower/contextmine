# Research Agent

This document describes the internal research agent that performs multi-step code investigation, implemented as part of Prompt 3.

## Overview

The research agent is an internal "second agent" that runs inside the MCP server. When the user asks a codebase question via the `code_deep_research` tool, the agent:

1. Decides which action to take (via LLM)
2. Executes the action (search, read file, summarize, etc.)
3. Collects evidence
4. Repeats until sufficient evidence or budget exhausted
5. Produces a concise answer with citations

The agent's internal conversation and heavy traces are stored as MCP resources, keeping the main Claude Code context clean.

## Architecture

```
contextmine_core/research/
├── agent.py              # ResearchAgent + LangGraph workflow
├── actions/
│   ├── __init__.py
│   ├── schemas.py        # Action input/output Pydantic models
│   ├── registry.py       # ActionRegistry + Action base class
│   ├── search.py         # hybrid_search action
│   ├── open_span.py      # open_span action
│   ├── summarize.py      # summarize_evidence action
│   └── finalize.py       # finalize action
├── artifacts.py          # ArtifactStore (from Prompt 1)
├── run.py                # ResearchRun, Evidence, ActionStep
└── llm/                  # LLM provider (from Prompt 2)
```

## MCP Tool: code_deep_research

The agent is exposed via a single high-level MCP tool:

```python
@mcp.tool(name="code_deep_research")
async def code_deep_research(
    question: str,         # Research question about the codebase
    scope: str | None,     # Limit search to path pattern
    budget: int = 10,      # Maximum agent steps (1-20)
    debug: bool = False,   # Include run_id for trace inspection
) -> str:
    """Perform deep research on the codebase."""
```

### Example Usage

```
User: "How does authentication work in this codebase?"

Claude calls: code_deep_research(question="How does authentication work?")

Response:
Authentication is handled by the AuthMiddleware class in src/auth/middleware.py.
When a request arrives, the middleware:
1. Extracts the bearer token from the Authorization header
2. Validates the token using verify_api_token()
3. Sets the user context for downstream handlers

**Citations:**
- [ev-abc-001] src/auth/middleware.py:45-67 (bm25)
- [ev-abc-002] src/auth/tokens.py:12-28 (vector)

*Run ID: abc12345... (use debug=true for full trace)*
```

### Output Format

The tool returns a concise answer (<800 tokens) with:

- **Answer**: Direct response to the question
- **Citations**: List of evidence IDs with file:line references
- **Run ID**: For accessing full trace via resources

Use `@research://runs/{run_id}/report.md` to view the full investigation.

## Actions

The agent can take these actions during investigation:

### hybrid_search

Search the codebase using BM25 + vector retrieval.

```python
{
    "action": "hybrid_search",
    "query": "authentication middleware",
    "k": 10
}
```

Returns relevant code snippets with scores and provenance.

### open_span

Read a specific range of lines from a file.

```python
{
    "action": "open_span",
    "file_path": "src/auth/middleware.py",
    "start_line": 45,
    "end_line": 67
}
```

Registers the content as evidence for the final answer.

### summarize_evidence

Use LLM to compress collected evidence into a memo.

```python
{
    "action": "summarize_evidence",
    "goal": "Focus on the token validation flow"
}
```

Helps the agent organize complex evidence before finalizing.

### finalize

Produce the final answer with citations.

```python
{
    "action": "finalize",
    "answer": "Authentication is handled by...",
    "confidence": 0.9
}
```

Marks the run as complete and stops the agent loop.

## Agent Configuration

```python
from contextmine_core.research import AgentConfig

config = AgentConfig(
    max_steps=10,              # Maximum investigation steps
    max_wall_time_seconds=120, # Maximum wall clock time
    store_artifacts=True,      # Save traces to artifact store
    action_timeout_seconds=30, # Timeout per action
)
```

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `RESEARCH_MODEL` | `claude-sonnet-4-20250514` | LLM model for agent |
| `ANTHROPIC_API_KEY` | - | API key for Anthropic |

## Evidence Contract

Each evidence item contains:

```python
@dataclass
class Evidence:
    id: str              # Unique ID (e.g., "ev-abc-001")
    file_path: str       # Source file path
    start_line: int      # Starting line (1-indexed)
    end_line: int        # Ending line (inclusive)
    content: str         # The actual code/text
    reason: str          # Why this was selected
    provenance: str      # How found: bm25, vector, hybrid, lsp, manual
    score: float | None  # Relevance score
    symbol_id: str | None   # Symbol name if LSP-derived
    symbol_kind: str | None # function, class, method, etc.
```

## Action Selection Prompt

The agent decides actions via structured LLM output:

```python
class ActionSelection(BaseModel):
    action: Literal["hybrid_search", "open_span", "summarize_evidence", "finalize"]
    reasoning: str
    hybrid_search: HybridSearchInput | None
    open_span: OpenSpanInput | None
    summarize_evidence: SummarizeEvidenceInput | None
    finalize: FinalizeInput | None
```

The system prompt includes the prompt injection firewall (from Prompt 2) to protect against malicious repository content.

## LangGraph Integration

The agent provides two implementations:

### 1. Imperative Loop (Default)

```python
agent = ResearchAgent(llm_provider=provider)
run = await agent.research("How does search work?")
```

### 2. LangGraph Workflow (Alternative)

```python
from contextmine_core.research.agent import create_research_graph

graph = create_research_graph(llm_provider=provider)
state = await graph.ainvoke({"run": run, "should_stop": False})
```

Both produce identical results; choose based on debugging/visualization needs.

## Testing

The agent includes mock implementations for testing without API calls:

```python
from contextmine_core.research.actions import MockHybridSearchAction
from contextmine_core.research.llm import MockLLMProvider

# Create mock provider with canned responses
provider = MockLLMProvider()
provider.set_structured_response("ActionSelection", {
    "action": "finalize",
    "reasoning": "Test",
    "finalize": {"answer": "Test answer", "confidence": 0.9},
})

# Use mock search action
mock_search = MockHybridSearchAction(mock_results=[
    {"file_path": "test.py", "content": "# test", "score": 0.9}
])

registry = ActionRegistry()
registry.register(mock_search)
registry.register(FinalizeAction())

agent = ResearchAgent(
    llm_provider=provider,
    action_registry=registry,
    config=AgentConfig(store_artifacts=False),
)
```

## Artifacts

After each run, artifacts are stored (if `store_artifacts=True`):

- `research://runs/{run_id}/trace.json` - Step-by-step execution trace
- `research://runs/{run_id}/evidence.json` - All collected evidence
- `research://runs/{run_id}/report.md` - Human-readable report

Access via MCP resources or `@` mentions in Claude Code.

## Next Steps

The research agent is extended in subsequent prompts:

- **Prompt 4**: LSP actions (definition, references, hover)
- **Prompt 5**: Tree-sitter actions (outline, find_symbol)
- **Prompt 6**: Graph actions (expand, pack, trace)
- **Prompt 7**: Verification hooks + evaluation harness
