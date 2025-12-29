# Code Graph + GraphRAG

This document describes the code graph integration for the research agent, implemented as part of Prompt 6.

## Overview

The code graph module provides a typed graph for multi-hop evidence selection, enabling the research agent to traverse symbol relationships for deeper code understanding.

## Architecture

```
contextmine_core/graph/
├── __init__.py           # Public exports
├── store.py              # SymbolNode, Edge, EdgeType, CodeGraph
├── builder.py            # GraphBuilder with LSP/Tree-sitter integration
└── retrieval.py          # expand_graph, pack_subgraph, trace_path

contextmine_core/research/actions/
└── graph.py              # graph_expand, graph_pack, graph_trace actions
```

## Data Structures

### SymbolNode

Represents a code symbol (function, class, method, etc.) in the graph:

```python
@dataclass
class SymbolNode:
    id: str                    # Qualified: "file_path::Class.method"
    name: str                  # Simple name: "method"
    kind: str                  # function, class, method, etc.
    file_path: str
    start_line: int
    end_line: int
    signature: str | None = None
    parent_id: str | None = None  # Containing symbol
    metadata: dict = field(default_factory=dict)
```

### Edge

Represents a directed relationship between two symbols:

```python
@dataclass
class Edge:
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    metadata: dict = field(default_factory=dict)
```

### EdgeType

Types of relationships in the graph:

```python
class EdgeType(Enum):
    CONTAINS = "contains"      # File contains class, class contains method
    DEFINES = "defines"        # Definition relationship
    REFERENCES = "references"  # Usage relationship
    CALLS = "calls"           # Function call
    IMPORTS = "imports"       # Import statement
    INHERITS = "inherits"     # Class inheritance
```

### CodeGraph

The graph container with efficient lookup indices:

```python
class CodeGraph:
    def add_node(node: SymbolNode) -> None
    def add_edge(edge: Edge) -> None
    def get_node(node_id: str) -> SymbolNode | None
    def has_node(node_id: str) -> bool
    def get_neighbors(node_id, edge_types, direction) -> list[tuple[SymbolNode, Edge]]
    def get_nodes_in_file(file_path: str) -> list[SymbolNode]
    def get_nodes_by_kind(kind: str) -> list[SymbolNode]
    def subgraph(node_ids: set[str]) -> CodeGraph
    def merge(other: CodeGraph) -> None
    def to_dict() -> dict
    @classmethod from_dict(data: dict) -> CodeGraph
```

## Graph Actions

The research agent can use these graph-backed actions:

### graph_expand

Expand from seed symbols following relationship types.

```python
{
    "action": "graph_expand",
    "seeds": ["src/auth.py::verify_token"],
    "edge_types": ["references", "calls"],  # Optional
    "depth": 2,  # 1-5, default 2
    "limit": 50  # Max nodes, default 50
}
```

Returns a subgraph with expanded nodes and edges, creating evidence for relevant symbols.

### graph_pack

Select minimal evidence set from an expanded graph.

```python
{
    "action": "graph_pack",
    "node_ids": ["...", "..."],  # Optional, None for all in context
    "target_count": 10  # Max nodes to select
}
```

Scores nodes by:
- Connectivity (incoming/outgoing edges)
- Symbol kind (classes > functions > methods)
- Size (larger symbols often more important)
- Position (top-level symbols preferred)

Returns ordered list of selected nodes with reasons.

### graph_trace

Find paths between two symbols (impact analysis).

```python
{
    "action": "graph_trace",
    "from_symbol": "src/auth.py::verify_token",
    "to_symbol": "src/api.py::handle_request",
    "edge_types": ["calls", "references"]  # Optional
}
```

Uses bidirectional BFS to find shortest paths. Returns all paths within reasonable bounds.

## GraphBuilder

The `GraphBuilder` combines Tree-sitter and LSP for graph construction:

```python
from contextmine_core.graph import GraphBuilder, get_graph_builder

# Get builder with available backends
builder = get_graph_builder()

# Build graph from a single file (synchronous, Tree-sitter only)
graph = builder.build_file_subgraph("src/main.py")

# Build from multiple files
graph = builder.build_multi_file_graph(["src/a.py", "src/b.py"])

# Add reference edges (async, requires LSP)
await builder.add_reference_edges(graph, symbol_id, file_path, line, column)
```

### Modes of Operation

1. **Full mode**: Both LSP and Tree-sitter available
   - Structure from Tree-sitter
   - Semantic relationships from LSP

2. **Tree-sitter only**: Structure without semantic relationships
   - Symbols and containment edges
   - No cross-file references

3. **Minimal**: Neither available
   - Returns empty graphs

## Retrieval Algorithms

### expand_graph

BFS expansion from seed nodes:

```python
from contextmine_core.graph import expand_graph, EdgeType

# Expand from seeds
subgraph = expand_graph(
    graph,
    seeds=["src/main.py::main"],
    edge_types=[EdgeType.CALLS, EdgeType.REFERENCES],
    depth=2,
    limit=100
)
```

### pack_subgraph

Select most relevant nodes:

```python
from contextmine_core.graph import pack_subgraph

# Pack to N most important nodes
packed = pack_subgraph(graph, node_ids=None, target_count=10)

for item in packed:
    print(f"{item.node.name}: {item.reason} (score: {item.score})")
```

### trace_path

Find paths between symbols:

```python
from contextmine_core.graph import trace_path

paths = trace_path(
    graph,
    from_id="src/main.py::main",
    to_id="src/utils.py::helper",
    edge_types=None,  # All edge types
    max_depth=5
)

for path in paths:
    print(" -> ".join(step.node.name for step in path))
```

## Configuration

### Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `GRAPH_MAX_DEPTH` | `3` | Maximum traversal depth |
| `GRAPH_MAX_NODES` | `100` | Maximum nodes to collect |

## Integration with Research Agent

The graph actions complement existing actions:

1. `hybrid_search` → find relevant code
2. `ts_find_symbol` → get symbol's qualified ID
3. `graph_expand` → explore relationships
4. `graph_pack` → select minimal evidence
5. `finalize` → answer with graph-derived evidence

### Example Flow

**Question**: "What functions call the `validate_token` function?"

1. Agent uses `ts_find_symbol` to get `src/auth.py::validate_token`
2. Agent uses `graph_expand` with `edge_types=["references"]`
3. Agent uses `graph_pack` to select most relevant callers
4. Agent uses `finalize` with evidence from graph

## Evidence Provenance

Graph actions produce evidence with `provenance="graph"`:

```python
Evidence(
    id="ev-abc-001",
    file_path="src/handlers.py",
    start_line=45,
    end_line=55,
    content="def handle_auth(...):\n    ...",
    reason="Expanded from graph: function 'handle_auth'",
    provenance="graph",
    symbol_id="src/handlers.py::handle_auth",
    symbol_kind="function",
)
```

## Testing

### Mock Actions

For testing without Tree-sitter:

```python
from contextmine_core.research.actions import (
    MockGraphExpandAction,
    MockGraphPackAction,
    MockGraphTraceAction,
)

# Mock expand
expand_action = MockGraphExpandAction(mock_nodes=[
    {"id": "a::foo", "name": "foo", "kind": "function", ...}
])

# Mock pack
pack_action = MockGraphPackAction(mock_selected=[
    {"node": {...}, "reason": "...", "score": 5.0}
])

# Mock trace
trace_action = MockGraphTraceAction(mock_paths=[
    [{"node": {...}, "edge_type": "calls", "direction": "forward"}, ...]
])
```

## Comparison with Tree-sitter Actions

| Feature | Tree-sitter Actions | Graph Actions |
|---------|---------------------|---------------|
| **Scope** | Single file | Multi-file |
| **Relationships** | Parent-child only | All edge types |
| **Use case** | Structure analysis | Dependency/impact analysis |
| **Performance** | Very fast | Depends on graph size |

Use Tree-sitter actions for:
- Quick file structure overview
- Finding a specific symbol

Use Graph actions for:
- Understanding cross-file dependencies
- Impact analysis
- Finding related code
