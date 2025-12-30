# Knowledge Graph Subsystem

This document describes the Knowledge Graph / Derived Knowledge subsystem that powers advanced code understanding and architecture documentation in ContextMine.

## Overview

The Knowledge Graph captures semantic relationships between code elements:
- **Files** and **Symbols** (functions, classes, methods)
- **Database Tables** and **Columns** from migrations
- **API Endpoints**, **GraphQL Types**, **RPC Services** from specs
- **Jobs** and **Workflows** from CI/CD configs
- **Business Rules** extracted from conditional logic

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MCP Tools Layer                         │
│  list_business_rules, get_erd, graph_rag, get_arc42, ...   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    GraphRAG Retrieval                       │
│  graph_rag_bundle, graph_neighborhood, trace_path          │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Knowledge Graph Store                     │
│  KnowledgeNode, KnowledgeEdge, KnowledgeEvidence           │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      Extractors                             │
│  ERM, OpenAPI, GraphQL, Protobuf, Jobs, Rules, arc42       │
└─────────────────────────────────────────────────────────────┘
```

## Data Model

### Node Kinds

| Kind | Description | Natural Key Pattern |
|------|-------------|---------------------|
| `FILE` | Source file | `file:{source_id}:{path}` |
| `SYMBOL` | Function, class, method | `symbol:{doc_id}:{qualified_name}` |
| `DB_TABLE` | Database table | `table:{collection}:{table_name}` |
| `DB_COLUMN` | Database column | `column:{table}:{column_name}` |
| `API_ENDPOINT` | REST API endpoint | `endpoint:{spec}:{method}:{path}` |
| `API_SCHEMA` | OpenAPI schema | `schema:{spec}:{name}` |
| `GRAPHQL_TYPE` | GraphQL type | `graphql:{schema}:{type_name}` |
| `SERVICE_RPC` | Protobuf RPC | `rpc:{proto}:{service}.{method}` |
| `JOB` | CI/CD job or workflow | `job:{file}:{name}` |
| `RULE_CANDIDATE` | Potential business rule | `rule:{file}:{line}:{hash}` |
| `BUSINESS_RULE` | Labeled business rule | `rule:labeled:{candidate_id}:{hash}` |

### Edge Kinds

| Kind | Source → Target | Description |
|------|-----------------|-------------|
| `FILE_DEFINES_SYMBOL` | FILE → SYMBOL | File contains symbol definition |
| `SYMBOL_CONTAINS_SYMBOL` | SYMBOL → SYMBOL | Parent-child relationship |
| `SYMBOL_CALLS_SYMBOL` | SYMBOL → SYMBOL | Function call relationship |
| `SYMBOL_IMPORTS_SYMBOL` | SYMBOL → SYMBOL | Import dependency |
| `TABLE_HAS_COLUMN` | DB_TABLE → DB_COLUMN | Schema structure |
| `TABLE_REFERENCES_TABLE` | DB_TABLE → DB_TABLE | Foreign key |
| `ENDPOINT_USES_SCHEMA` | API_ENDPOINT → API_SCHEMA | Request/response schema |
| `RPC_USES_MESSAGE` | SERVICE_RPC → SYMBOL | Protobuf message usage |
| `RULE_DERIVED_FROM_CANDIDATE` | RULE_CANDIDATE → BUSINESS_RULE | LLM labeling |

### Evidence

Evidence links nodes to source locations:
```python
KnowledgeEvidence(
    file_path="src/auth.py",
    start_line=42,
    end_line=55,
    document_id=...,  # Optional link to indexed document
)
```

## Extractors

### ERM Extractor (Database Schema)

Parses Alembic migrations to extract database schema:

```python
from contextmine_core.analyzer.extractors.erm import ERMExtractor, build_erm_graph

# Extract schema from migrations
extractor = ERMExtractor()
schema = extractor.extract_from_directory("packages/core/alembic/versions/")

# Build knowledge graph nodes
await build_erm_graph(session, collection_id, schema)

# Generate Mermaid ERD
from contextmine_core.analyzer.extractors.erm import generate_mermaid_erd
erd = generate_mermaid_erd(schema)
```

### Surface Catalog Extractor (APIs, Jobs)

Extracts external interfaces from specs:

```python
from contextmine_core.analyzer.extractors.surface import SurfaceCatalogExtractor

extractor = SurfaceCatalogExtractor()

# Add spec files
extractor.add_file("openapi.yaml", openapi_content)
extractor.add_file("schema.graphql", graphql_content)
extractor.add_file(".github/workflows/ci.yml", workflow_content)

# Build knowledge graph
await build_surface_graph(session, collection_id, extractor.catalog)
```

### Rule Candidate Extractor

Detects conditional branches leading to failures:

```python
from contextmine_core.analyzer.extractors.rules import extract_rule_candidates

# Extract from source file
result = extract_rule_candidates("auth.py", source_content)

for candidate in result.candidates:
    print(f"Rule at {candidate.start_line}: {candidate.predicate_text}")
    print(f"Failure: {candidate.failure_kind} - {candidate.failure_text}")
```

Detected patterns:
- Python: `if condition: raise Exception`
- Python: `assert condition, "message"`
- TypeScript/JS: `if (condition) throw new Error`
- Return error patterns: `if condition: return None/null`

### LLM Labeling

Converts rule candidates to labeled business rules:

```python
from contextmine_core.analyzer.labeling import label_rule_candidates
from contextmine_core.research.llm import get_research_llm_provider

# Get LLM provider
provider = get_research_llm_provider()

# Label candidates (requires OPENAI_API_KEY or similar)
stats = await label_rule_candidates(
    session=session,
    collection_id=collection_id,
    provider=provider,
    force_relabel=False,  # Skip unchanged candidates
)

print(f"Processed: {stats.candidates_processed}")
print(f"Rules created: {stats.rules_created}")
print(f"Skipped (invalid): {stats.skipped_invalid}")
```

**Note:** LLM labeling is optional. Rule candidates are useful even without labeling.

To disable LLM labeling, simply don't call `label_rule_candidates()`.

## GraphRAG Retrieval

Graph-augmented retrieval combines semantic search with knowledge graph:

```python
from contextmine_core.graphrag import graph_rag_bundle, graph_neighborhood, trace_path

# Full GraphRAG query
result = await graph_rag_bundle(
    session=session,
    query="how does authentication work?",
    collection_id=collection_id,
    user_id=user_id,
    max_depth=2,
)

print(result.summary_markdown)  # Human-readable
print(result.to_dict())         # JSON-serializable

# Local exploration
neighborhood = await graph_neighborhood(
    session=session,
    node_id=some_node_id,
    depth=1,
)

# Path finding
path = await trace_path(
    session=session,
    from_node_id=start_id,
    to_node_id=end_id,
    max_hops=6,
)
```

## MCP Tools

The Knowledge Graph is exposed via MCP tools for Claude Code / Cursor:

### Business Rules

```
# List all rules
list_business_rules(collection_id?, query?)

# Get rule details
get_business_rule(rule_id)
```

### Database Schema

```
# Get ERD diagram
get_erd(collection_id?, format="mermaid"|"json")
```

### System Surfaces

```
# List endpoints, jobs, schemas
list_system_surfaces(collection_id?, kind=["endpoint", "job", "graphql", "rpc", "schema"])
```

### Graph Navigation

```
# Explore node neighborhood
graph_neighborhood(node_id, depth=1, edge_kinds?, limit=30)

# Find path between nodes
trace_path(from_node_id, to_node_id, max_hops=6)

# Full GraphRAG query
graph_rag(query, collection_id?, max_depth=2, max_results=10)
```

### Architecture Documentation

```
# Get arc42 documentation
get_arc42(collection_id?, section?, regenerate=false)

# Check for architecture drift
arc42_drift_report(collection_id?)
```

## Integration

### Sync Pipeline

The Knowledge Graph is built during the sync pipeline:

```python
# After indexing documents and symbols
from contextmine_core.knowledge.builder import build_knowledge_graph_for_source

await build_knowledge_graph_for_source(session, source_id)
```

### Incremental Updates

- **Changed files**: Re-extract nodes/edges, upsert via natural keys
- **Deleted files**: `cleanup_orphan_nodes()` removes stale entries
- **Idempotent**: Safe to re-run; natural keys prevent duplicates

### Configuration

Environment variables:
- `OPENAI_API_KEY` - Required for LLM labeling (optional feature)
- `DEFAULT_LLM_MODEL` - Model for labeling (default: gpt-4o-mini)

## Database Migration

Apply the knowledge graph migration:

```bash
cd packages/core
DATABASE_URL=postgresql+asyncpg://... uv run alembic upgrade head
```

The migration creates:
- `knowledge_node` - Graph nodes
- `knowledge_edge` - Graph edges
- `knowledge_evidence` - Source citations
- `knowledge_artifact` - Generated documents (ERD, arc42)
- Junction tables for evidence linking

## Example Queries

### Find all business rules related to authentication

```
list_business_rules(query="auth")
```

### Explore what a function depends on

```
graph_neighborhood(node_id="<function_node_id>", depth=2)
```

### Understand how two components are connected

```
trace_path(from_node_id="<api_endpoint>", to_node_id="<db_table>")
```

### Get architecture overview

```
get_arc42(regenerate=true)
```

### Check for schema changes

```
arc42_drift_report()
```
