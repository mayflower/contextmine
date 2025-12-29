# LSP Integration

This document describes the Language Server Protocol (LSP) integration for the research agent, implemented as part of Prompt 4.

## Overview

The LSP module provides code intelligence capabilities to the research agent through language servers. When the agent needs precise code navigation (definition lookup, find references, hover info), it can use LSP-backed actions instead of relying solely on text search.

## Architecture

```
contextmine_core/lsp/
├── __init__.py         # Public exports
├── exceptions.py       # LSP-specific exceptions
├── languages.py        # Language detection, project root finding
├── client.py           # LspClient wrapper around multilspy
└── manager.py          # LspManager singleton with caching

contextmine_core/research/actions/
└── lsp.py              # LSP-backed actions for the agent
```

## LSP Actions

The research agent can use these LSP-backed actions:

### lsp_definition

Jump to the definition of a symbol.

```python
{
    "action": "lsp_definition",
    "file_path": "src/auth/middleware.py",
    "line": 45,       # 1-indexed
    "column": 15      # 0-indexed
}
```

Returns evidence with the definition location(s) and surrounding code.

### lsp_references

Find all usages of a symbol across the codebase.

```python
{
    "action": "lsp_references",
    "file_path": "src/auth/middleware.py",
    "line": 45,
    "column": 15
}
```

Returns up to 10 evidence items with reference locations.

### lsp_hover

Get type signature and documentation for a symbol.

```python
{
    "action": "lsp_hover",
    "file_path": "src/auth/middleware.py",
    "line": 45,
    "column": 15
}
```

Returns evidence with symbol info (name, kind, signature, docs).

### lsp_diagnostics

Get compiler errors and warnings for files.

```python
{
    "action": "lsp_diagnostics",
    "file_paths": ["src/auth/middleware.py", "src/auth/tokens.py"]
}
```

**Note**: Full diagnostics support is limited in the current implementation.

## Supported Languages

| Extension | Language | Typical Server |
|-----------|----------|----------------|
| .py, .pyi | Python | pylsp, pyright |
| .ts, .tsx | TypeScript | typescript-language-server |
| .js, .jsx | JavaScript | typescript-language-server |
| .rs | Rust | rust-analyzer |
| .go | Go | gopls |
| .java | Java | jdtls |
| .cs | C# | OmniSharp |

Language detection is automatic based on file extension.

## Configuration

### Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `LSP_IDLE_TIMEOUT_SECONDS` | `300` | Stop idle servers after 5 minutes |
| `LSP_REQUEST_TIMEOUT_SECONDS` | `30` | Timeout for individual requests |

### Dependencies

LSP support requires the `multilspy` library:

```bash
pip install multilspy
```

Without multilspy, LSP actions will gracefully degrade (return errors without crashing the agent).

## LspManager

The `LspManager` is a singleton that manages language server lifecycle:

```python
from contextmine_core.lsp import get_lsp_manager

manager = get_lsp_manager()

# Get a client for a file (starts server if needed)
client = await manager.get_client("src/main.py")

# Client is cached by (language, project_root)
# Reusing the same client for subsequent calls
```

### Features

- **Lazy initialization**: Servers start on first request
- **Caching**: Servers are reused for the same language/project
- **Auto-detection**: Project root found automatically via markers (.git, pyproject.toml, etc.)
- **Idle timeout**: Servers stop after 5 minutes of inactivity
- **Graceful shutdown**: All servers cleaned up on shutdown

## LspClient

The `LspClient` wraps multilspy with a simplified API:

```python
from contextmine_core.lsp import Location, SymbolInfo

# Get definition
locations: list[Location] = await client.get_definition(
    file_path="main.py",
    line=10,      # 1-indexed
    column=5,     # 0-indexed
)

# Get references
locations = await client.get_references(
    file_path="main.py",
    line=10,
    column=5,
)

# Get hover info
info: SymbolInfo | None = await client.get_hover(
    file_path="main.py",
    line=10,
    column=5,
)
```

### Data Types

```python
@dataclass
class Location:
    file_path: str
    start_line: int   # 1-indexed
    start_column: int # 0-indexed
    end_line: int     # 1-indexed
    end_column: int   # 0-indexed

@dataclass
class SymbolInfo:
    name: str
    kind: str         # function, class, method, variable
    signature: str | None
    documentation: str | None
```

## Testing

### Mock Client

For testing without language servers:

```python
from contextmine_core.lsp import MockLspClient, Location, SymbolInfo

mock_client = MockLspClient()

# Configure mock responses
mock_client.set_definition(
    "test.py", 10, 5,
    [Location("other.py", 20, 0, 25, 10)]
)

mock_client.set_hover(
    "test.py", 10, 5,
    SymbolInfo("my_func", "function", "def my_func(x: int)", "Docs here")
)

# Use in action
action = LspDefinitionAction(lsp_client=mock_client)
result = await action.execute(run, {"file_path": "test.py", "line": 10, "column": 5})
```

### Mock Action

For testing the agent without LSP:

```python
from contextmine_core.research.actions import MockLspDefinitionAction

action = MockLspDefinitionAction(mock_locations=[
    {"file_path": "def.py", "start_line": 100, "content": "def func(): ..."}
])

registry.register(action)
```

## Evidence

LSP actions produce evidence with special provenance:

```python
Evidence(
    id="ev-abc-001",
    file_path="src/auth.py",
    start_line=45,
    end_line=55,
    content="def verify_token(...):\n    ...",
    reason="Definition found via LSP go-to-definition",
    provenance="lsp",           # LSP-sourced
    symbol_id="verify_token",   # Symbol name
    symbol_kind="function",     # Symbol type
)
```

## Error Handling

LSP actions handle errors gracefully:

1. **Unsupported file type**: Returns error result, agent continues
2. **Server not available**: Returns error result with explanation
3. **Timeout**: Returns error result, agent can retry or use fallback
4. **No results**: Returns success with empty evidence

The agent will not crash if LSP is unavailable; it can fall back to text-based search actions.

## Next Steps

LSP integration is extended in subsequent prompts:

- **Prompt 5**: Tree-sitter actions for AST-based symbol extraction
- **Prompt 6**: Code graph combining LSP + Tree-sitter data
