# MCP Interface

ContextMine exposes its functionality via the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) at the `/mcp` endpoint. This allows AI assistants like Claude Desktop and Cursor to query indexed documentation and code.

## Connecting to ContextMine

### Claude Desktop

Add to your Claude Desktop configuration (`~/.config/claude/claude_desktop_config.json` on Linux, `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "contextmine": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

### Cursor

Settings → MCP → Add server with URL `http://localhost:8000/mcp`

## Authentication

MCP authentication uses GitHub OAuth. When you first connect, your MCP client will redirect to GitHub for authentication. The same OAuth flow is used for both the admin UI and MCP clients.

## Available Tools

### Primary Research Tools

#### `get_markdown`

Semantic search across all indexed documentation and code. Returns synthesized context.

```
get_markdown(
    query: str,                    # Natural language question or search terms
    collection_id?: str,           # Limit to specific collection
    topic?: str,                   # Filter by topic path (e.g., 'api', 'hooks')
    max_chunks?: int = 10,         # Number of results (1-50)
    max_tokens?: int = 4000,       # Response length limit
    raw?: bool = False             # Return raw chunks without LLM synthesis
)
```

**Example:**
```
get_markdown(query="how does authentication work?")
```

#### `deep_research`

Multi-step research agent for complex codebase questions. Uses an iterative approach with multiple tools to search, read code, and build answers with citations.

```
deep_research(
    question: str,                 # Complex question requiring investigation
    scope?: str,                   # Limit to path pattern (e.g., 'src/api/**')
    budget?: int = 10,             # Max investigation steps (1-20)
    debug?: bool = False           # Return run_id for trace inspection
)
```

**Example:**
```
deep_research(question="explain the payment flow", budget=15)
```

#### `graph_rag`

Graph-augmented retrieval implementing Microsoft GraphRAG approach with global (community) and local (entity) context.

```
graph_rag(
    query: str,                    # Natural language query
    collection_id?: str,           # Filter to specific collection
    max_communities?: int = 5,     # Maximum community summaries
    max_entities?: int = 20,       # Maximum entities
    max_depth?: int = 2,           # Graph expansion depth (1-3)
    format?: str = "markdown",     # Output: 'markdown' or 'json'
    answer?: bool = False          # Use map-reduce for synthesized answer
)
```

**Example:**
```
graph_rag(query="how do X and Y relate?", answer=true)
```

### Discovery Tools

#### `list_collections`

List available documentation collections.

```
list_collections(search?: str)     # Optional search term to filter
```

#### `list_documents`

Browse documents in a collection.

```
list_documents(
    collection_id: str,            # The collection ID
    topic?: str,                   # Topic/path filter
    limit?: int = 50               # Maximum documents to return
)
```

### Code Navigation Tools

#### `outline`

List all functions, classes, and methods in a file with line numbers.

```
outline(
    file_path: str,                # Document URI from search results
    include_children?: bool = True # Include methods inside classes
)
```

**Example:**
```
outline(file_path="src/auth/handlers.py")
```

#### `find_symbol`

Get the source code of a specific function or class by name.

```
find_symbol(
    file_path: str,                # Document URI
    name: str                      # Function, class, or method name
)
```

**Example:**
```
find_symbol(file_path="src/auth.py", name="authenticate")
```

#### `definition`

Jump to where a symbol is defined.

```
definition(
    file_path: str,                # Document URI
    line: int,                     # Line number (1-indexed)
    column: int                    # Column position (0-indexed)
)
```

#### `references`

Find all usages of a symbol for impact analysis.

```
references(
    file_path: str,                # Document URI
    line: int,                     # Line number (1-indexed)
    column: int,                   # Column position (0-indexed)
    limit?: int = 20               # Max results
)
```

#### `expand`

Explore code relationships from a starting symbol. Shows what it calls, what calls it, imports, etc.

```
expand(
    seeds: list[str],              # Starting points as 'file_uri::function_name'
    depth?: int = 2,               # Hops to follow (1-3)
    edge_types?: list[str],        # Filter: 'calls', 'called_by', 'imports', 'inherits'
    limit?: int = 30               # Max nodes
)
```

**Example:**
```
expand(seeds=["src/auth.py::login"], depth=2)
```

### Knowledge Graph Tools

#### `research_validation`

Find validation rules, business logic, and constraints for specific code.

```
research_validation(
    code_path: str,                # File path or function name
    collection_id?: str            # Filter to specific collection
)
```

**Example:**
```
research_validation(code_path="auth.py")
research_validation(code_path="payment")
```

#### `research_data_model`

Research the data model for a specific entity or concept.

```
research_data_model(
    entity: str,                   # Table name, entity, or data concept
    collection_id?: str            # Filter to specific collection
)
```

**Example:**
```
research_data_model(entity="users")
research_data_model(entity="order")
```

#### `research_architecture`

Research system architecture for a specific topic.

```
research_architecture(
    topic: str,                    # 'deployment', 'security', 'api', 'database'
    collection_id?: str            # Filter to specific collection
)
```

**Example:**
```
research_architecture(topic="deployment")
research_architecture(topic="security")
```

#### `graph_neighborhood`

Explore the knowledge graph neighborhood around a node.

```
graph_neighborhood(
    node_id: str,                  # Starting node ID
    depth?: int = 1,               # Expansion depth (1-3)
    edge_kinds?: list[str],        # Filter by edge kinds
    limit?: int = 30               # Maximum nodes to return
)
```

#### `trace_path`

Find the shortest path between two nodes in the knowledge graph.

```
trace_path(
    from_node_id: str,             # Starting node ID
    to_node_id: str,               # Target node ID
    max_hops?: int = 6             # Maximum path length (1-10)
)
```

### Twin Architecture Tools

#### `get_arc42`

Generate or read arc42 documentation from the architecture facts layer.

```
get_arc42(
    collection_id?: str,           # Optional collection UUID (auto-picks accessible default if omitted)
    scenario_id?: str,             # Optional scenario UUID (defaults to AS-IS)
    section?: str,                 # Optional section filter (e.g. '5', 'quality', 'deployment')
    regenerate?: bool = false      # Force regeneration instead of cached artifact
)
```

#### `arc42_drift_report`

Compute advisory architecture drift between current and baseline scenario.

```
arc42_drift_report(
    collection_id?: str,           # Optional collection UUID
    scenario_id?: str,             # Optional scenario UUID
    baseline_scenario_id?: str     # Optional explicit baseline UUID
)
```

#### `list_ports_adapters`

List inferred ports/adapters mappings with confidence and evidence references.

```
list_ports_adapters(
    collection_id?: str,           # Optional collection UUID
    scenario_id?: str,             # Optional scenario UUID
    direction?: str,               # inbound|outbound
    container?: str                # Optional container filter
)
```

## MCP Resources

Research run artifacts are exposed as MCP resources for inspection:

| Resource URI | Description |
|-------------|-------------|
| `research://runs` | List recent research runs |
| `research://runs/{run_id}/trace.json` | Agent execution trace |
| `research://runs/{run_id}/evidence.json` | Collected evidence |
| `research://runs/{run_id}/report.md` | Human-readable report |

## Tool Selection Guide

| Use Case | Recommended Tool |
|----------|------------------|
| Simple question about docs/code | `get_markdown` |
| Complex investigation | `deep_research` |
| Understanding relationships | `graph_rag` |
| Find validation rules | `research_validation` |
| Understand data structures | `research_data_model` |
| Architecture overview | `research_architecture` |
| File structure | `outline` |
| Specific function code | `find_symbol` |
| Symbol definition | `definition` |
| Find usages | `references` |
| Explore call graph | `expand` |

## Example Workflows

### Understanding a Feature

1. Start with `get_markdown` to find relevant files
2. Use `outline` to see the structure of key files
3. Use `find_symbol` to read specific functions
4. Use `expand` to trace the call graph

### Investigating Business Logic

1. Use `research_validation` to find validation rules
2. Use `research_data_model` to understand the data structure
3. Use `graph_rag` to understand relationships

### Complex Investigation

1. Use `deep_research` with a specific question
2. Review the citations in the answer
3. Use `graph_neighborhood` to explore related nodes
4. Use `trace_path` to understand connections

## Transport Protocol

ContextMine uses Streamable HTTP transport for MCP, implemented with FastMCP 2. The server is mounted at `/mcp` on the FastAPI application.

Authentication is handled via GitHub OAuth using the same credentials as the admin UI.
