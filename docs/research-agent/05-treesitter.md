# Tree-sitter Integration

This document describes the Tree-sitter integration for the research agent, implemented as part of Prompt 5.

## Overview

The Tree-sitter module provides fast, offline code structure analysis through incremental parsing. Unlike LSP which requires running language servers, Tree-sitter works purely locally with pre-compiled grammars for robust symbol extraction.

## Architecture

```
contextmine_core/treesitter/
├── __init__.py           # Public exports
├── languages.py          # Language detection, query patterns
├── manager.py            # TreeSitterManager with caching
└── outline.py            # Symbol extraction

contextmine_core/research/actions/
└── treesitter.py         # ts_outline, ts_find_symbol, ts_enclosing_symbol
```

## Tree-sitter Actions

The research agent can use these Tree-sitter backed actions:

### ts_outline

Get an outline of all symbols in a file.

```python
{
    "action": "ts_outline",
    "file_path": "src/auth/middleware.py"
}
```

Returns a list of top-level symbols with their children:
- Functions, classes, methods
- Structs, enums, interfaces (for Rust, Go, Java, etc.)
- Type definitions

### ts_find_symbol

Find a specific symbol by name in a file.

```python
{
    "action": "ts_find_symbol",
    "file_path": "src/auth/middleware.py",
    "name": "verify_token"
}
```

Returns the symbol location and its source code as evidence.

### ts_enclosing_symbol

Find what function/class/method contains a specific line.

```python
{
    "action": "ts_enclosing_symbol",
    "file_path": "src/auth/middleware.py",
    "line": 45      # 1-indexed
}
```

Returns the innermost symbol containing the line.

## Supported Languages

| Extension | Language |
|-----------|----------|
| .py, .pyi | Python |
| .ts, .tsx | TypeScript |
| .js, .jsx, .mjs, .cjs | JavaScript |
| .rs | Rust |
| .go | Go |
| .java | Java |
| .c, .h | C |
| .cpp, .hpp, .cc, .cxx | C++ |
| .cs | C# |
| .rb | Ruby |
| .php | PHP |

Language detection is automatic based on file extension.

## Configuration

### Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `TREESITTER_CACHE_SIZE` | `100` | Maximum parsed trees to cache |

### Dependencies

Tree-sitter support requires the `tree-sitter-language-pack` library:

```bash
pip install tree-sitter-language-pack
```

Without this dependency, Tree-sitter actions will gracefully degrade (return errors without crashing the agent).

## TreeSitterManager

The `TreeSitterManager` is a singleton that manages parser lifecycle and caching:

```python
from contextmine_core.treesitter import get_treesitter_manager

manager = get_treesitter_manager()

# Check availability
if manager.is_available():
    # Parse a file (uses cache if content unchanged)
    tree = manager.parse("src/main.py")

    # Parse with explicit content
    tree = manager.parse("src/main.py", content=code_string)

    # Invalidate cache for a file
    manager.invalidate("src/main.py")

    # Get cache statistics
    stats = manager.get_cache_stats()
```

### Features

- **Lazy initialization**: Parsers are loaded on first use
- **Content-hash caching**: Trees are cached by file path with content-hash invalidation
- **LRU eviction**: Oldest trees evicted when cache is full
- **Graceful degradation**: Works without tree-sitter installed

## Symbol Extraction

The `outline` module provides functions for extracting symbols:

```python
from contextmine_core.treesitter import (
    extract_outline,
    find_symbol_by_name,
    find_enclosing_symbol,
    get_symbol_content,
    Symbol,
    SymbolKind,
)

# Get all top-level symbols
symbols = extract_outline("main.py", include_children=True)

for sym in symbols:
    print(f"{sym.kind.value} {sym.name} (L{sym.start_line}-{sym.end_line})")
    for child in sym.children:
        print(f"  {child.kind.value} {child.name}")

# Find a specific symbol
symbol = find_symbol_by_name("main.py", "process_data")
if symbol:
    content = get_symbol_content(symbol)
    print(content)

# Find what contains a line
symbol = find_enclosing_symbol("main.py", 42)
if symbol:
    print(f"Line 42 is inside {symbol.name}")
```

### Data Types

```python
class SymbolKind(Enum):
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    STRUCT = "struct"
    ENUM = "enum"
    INTERFACE = "interface"
    TYPE = "type"
    TRAIT = "trait"
    IMPL = "impl"
    MODULE = "module"
    VARIABLE = "variable"
    UNKNOWN = "unknown"

@dataclass
class Symbol:
    name: str
    kind: SymbolKind
    file_path: str
    start_line: int       # 1-indexed
    end_line: int         # 1-indexed
    start_column: int     # 0-indexed
    end_column: int       # 0-indexed
    signature: str | None
    parent: str | None
    docstring: str | None
    children: list[Symbol]
```

## Testing

### Mock Actions

For testing without Tree-sitter:

```python
from contextmine_core.research.actions import MockTsOutlineAction, MockTsFindSymbolAction

# Mock outline
outline_action = MockTsOutlineAction(mock_symbols=[
    {"name": "foo", "kind": "function", "start_line": 1, "end_line": 10}
])

# Mock find symbol
find_action = MockTsFindSymbolAction(mock_symbol={
    "name": "target",
    "kind": "function",
    "start_line": 20,
    "end_line": 30,
    "content": "def target(): ..."
})

registry.register(outline_action)
registry.register(find_action)
```

## Evidence

Tree-sitter actions produce evidence with special provenance:

```python
Evidence(
    id="ev-abc-001",
    file_path="src/auth.py",
    start_line=45,
    end_line=55,
    content="def verify_token(...):\n    ...",
    reason="Found function 'verify_token' via Tree-sitter",
    provenance="treesitter",      # Tree-sitter sourced
    symbol_id="verify_token",     # Symbol name
    symbol_kind="function",       # Symbol type
)
```

## Error Handling

Tree-sitter actions handle errors gracefully:

1. **Unsupported file type**: Returns error result, agent continues
2. **Tree-sitter not installed**: Returns error result with explanation
3. **File not found**: Returns error result
4. **No symbols found**: Returns success with empty results

The agent will not crash if Tree-sitter is unavailable; it can fall back to text-based search actions.

## Comparison with LSP

| Feature | Tree-sitter | LSP |
|---------|-------------|-----|
| **Startup** | Instant | Requires server startup |
| **Dependencies** | Single package | Language-specific servers |
| **Offline** | Yes | Needs server process |
| **Type info** | Limited | Full type checking |
| **Cross-file** | No | Yes (references) |
| **Accuracy** | Syntax-based | Semantic analysis |

Use Tree-sitter for:
- Quick file structure overview
- Finding symbol boundaries
- Offline operation

Use LSP for:
- Type information
- Cross-file references
- Semantic analysis

## Next Steps

Tree-sitter integration is extended in subsequent prompts:

- **Prompt 6**: Code graph combining LSP + Tree-sitter data
- **Prompt 7**: Symbol-aware chunking using Tree-sitter boundaries
