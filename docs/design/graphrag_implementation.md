# GraphRAG Implementation Plan (Adapted)

This plan adapts the requirements from `docs/prompts4.md` to the current ContextMine state.

**Non-negotiables (unchanged):**
1. Implement REAL GraphRAG: graph → communities (hierarchy) → offline summaries → query-time global+local retrieval
2. Language/framework agnostic: SCIP/LSIF preferred, Tree-sitter fallback (structural only)
3. No redundancy: replace old retrieval, don't keep parallel paths
4. Every step: `uv run ruff check . && uv run ruff format . && uvx ty check && uv run pytest -v`

---

## Current State Summary

### Already Implemented
- **Knowledge Graph schema** (`packages/core/contextmine_core/models.py`):
  - `KnowledgeNode`, `KnowledgeEdge`, `KnowledgeEvidence`
  - Node kinds: FILE, SYMBOL, DB_TABLE, DB_COLUMN, RULE_CANDIDATE, BUSINESS_RULE, etc.
  - Edge kinds: FILE_DEFINES_SYMBOL, SYMBOL_CONTAINS_SYMBOL, CALLS, REFERENCES, etc.
  - Migration 013 in place

- **Graph builder** (`packages/core/contextmine_core/knowledge/builder.py`):
  - Creates FILE/SYMBOL nodes from documents
  - Uses SymbolEdge table for cross-references
  - **Problem**: Uses document URIs directly, not SCIP/LSIF semantic IDs

- **Tree-sitter extraction** (`packages/core/contextmine_core/analyzer/extractors/rules.py`):
  - AST-based rule candidate mining (Python, TypeScript)
  - Control-flow patterns → RULE_CANDIDATE nodes

- **MCP tools** (`apps/api/app/mcp_server.py`):
  - `get_markdown`, `graph_neighborhood`, `trace_path`, `graph_rag`
  - **Problem**: `graph_rag` is just BFS expansion + word matching

- **Fake GraphRAG** (`packages/core/contextmine_core/graphrag.py`):
  - `graph_rag_bundle()` - hybrid search + node mapping + BFS
  - **Missing**: Leiden, communities, summaries, global retrieval

### Missing Components
1. **Semantic Snapshot** - SCIP/LSIF ingestion layer
2. **Community detection** - deterministic label propagation
3. **Community summaries** - offline LLM synthesis + embeddings
4. **Real GraphRAG retrieval** - global (community) + local (entity) context

---

## Step 0 — Schema Extensions for Communities

Add community tables to existing Knowledge Graph schema.

```text
Extend the existing Knowledge Graph schema in packages/core/contextmine_core/models.py.

Current state:
- KnowledgeNode, KnowledgeEdge, KnowledgeEvidence exist (migration 013)
- Missing: community hierarchy, community embeddings

Add models:
1) KnowledgeCommunity:
   - id, collection_id, level (1 or 2), natural_key (UNIQUE per collection)
   - title, summary (text), meta (JSONB)
   - created_at, updated_at

2) CommunityMember:
   - community_id FK, node_id FK
   - score (float) - membership strength
   - UNIQUE(community_id, node_id)

3) KnowledgeEmbedding:
   - id, collection_id
   - target_type ENUM('node', 'community')
   - target_id UUID
   - model_name, provider
   - vector (pgvector)
   - content_hash (for idempotency)
   - UNIQUE(collection_id, target_type, target_id, model_name)

Create Alembic migration 014_add_communities.py.

Validation:
- uv run ruff check . && uv run ruff format .
- uvx ty check
- Run migration: cd packages/core && DATABASE_URL=... uv run alembic upgrade head
- uv run pytest -v

End with git status showing only intended changes.
```

---

## Step 1 — Semantic Snapshot Layer

Create language-agnostic semantic indexing substrate.

```text
Create packages/core/contextmine_core/semantic_snapshot/ module.

This replaces the ad-hoc document-based symbol extraction with a proper semantic layer.

Data model (Pydantic, serializable):
- FileInfo(path: str, language: str | None)
- Symbol(def_id: str, kind: str, file_path: str, range: Range, name: str | None, container_def_id: str | None)
- Range(start_line: int, start_col: int, end_line: int, end_col: int)
- Occurrence(file_path: str, range: Range, role: Literal["definition", "reference"], def_id: str)
- Relation(src_def_id: str, kind: Literal["contains", "calls", "references", "extends", "implements"], dst_def_id: str, resolved: bool, meta: dict)
- Snapshot(files: list[FileInfo], symbols: list[Symbol], occurrences: list[Occurrence], relations: list[Relation])

Providers (implement in order):
1) TreeSitterProvider (immediate priority):
   - Use existing Tree-sitter setup from rules.py
   - Extract declarations as symbols
   - Extract containment edges (function in class, etc.)
   - Extract call sites as UNRESOLVED relations (no string tricks)
   - Languages: Python, TypeScript, JavaScript (match existing support)

2) SCIPProvider (next priority):
   - Parse .scip protobuf files
   - Map SCIP symbol IDs to def_id
   - Map occurrences to Occurrence model
   - Fully resolved relations

3) LSIFProvider (future):
   - Parse .lsif JSON lines
   - Similar mapping to SCIP

Entry function:
```python
async def build_snapshot(
    collection_id: UUID,
    files: list[tuple[str, str]],  # (path, content) pairs
    scip_path: Path | None = None,
    lsif_path: Path | None = None,
) -> Snapshot:
    """Build semantic snapshot. Prefers SCIP > LSIF > Tree-sitter."""
```

Integration:
- This will be called by sync pipeline instead of direct symbol table writes
- Graph builder (Step 2) will consume Snapshot, not raw documents

Tests:
- Snapshot model serialization round-trip
- TreeSitterProvider on Python snippet with class + method + call
- TreeSitterProvider on TypeScript snippet
- Verify unresolved calls are marked as such

Files to create:
- packages/core/contextmine_core/semantic_snapshot/__init__.py
- packages/core/contextmine_core/semantic_snapshot/models.py
- packages/core/contextmine_core/semantic_snapshot/treesitter.py
- packages/core/contextmine_core/semantic_snapshot/scip.py (stub with NotImplementedError)
- packages/core/tests/test_semantic_snapshot.py

Validation:
- uv run ruff check . && uv run ruff format .
- uvx ty check
- uv run pytest -v

End with git status.
```

---

## Step 2 — Graph Builder from Snapshot

Replace document-based graph builder with Snapshot-based builder.

```text
Refactor packages/core/contextmine_core/knowledge/builder.py to consume Snapshot.

Current state:
- build_knowledge_graph_for_source() queries Document + SymbolEdge tables
- Creates FILE/SYMBOL nodes with natural_key = URI or symbol name
- Problem: natural keys don't match SCIP/LSIF IDs

New implementation:
```python
async def build_graph_from_snapshot(
    session: AsyncSession,
    collection_id: UUID,
    snapshot: Snapshot,
    changed_files: set[str],
    deleted_files: set[str],
) -> GraphBuildStats:
    """
    Build knowledge graph from semantic snapshot.

    - Creates FILE nodes for each FileInfo
    - Creates SYMBOL nodes for each Symbol
    - Creates edges from Relations
    - Attaches evidence from Occurrences
    - Handles incremental updates (changed/deleted files)
    """
```

Natural key conventions (must match across all code):
- FILE: `file:{path}`
- SYMBOL: `symbol:{def_id}` (SCIP/LSIF) or `symbol:{path}:{range}:{kind}` (Tree-sitter)
- EDGE: `edge:{kind}:{src_natural_key}->{dst_natural_key}`

Idempotency:
- Upsert nodes/edges by natural_key
- Delete nodes/edges for deleted_files
- Update nodes/edges for changed_files

Evidence:
- Create KnowledgeEvidence for each Occurrence
- Link to nodes via node_evidence_link

Migration path:
- Keep old build_knowledge_graph_for_source() temporarily
- Add build_graph_from_snapshot() as new entry point
- Sync pipeline will switch to new function

Tests:
- Create Snapshot fixture programmatically
- Verify nodes/edges created correctly
- Verify idempotency (run twice, same counts)
- Verify deletion removes stale nodes

Validation:
- uv run ruff check . && uv run ruff format .
- uvx ty check
- uv run pytest -v

End with git status.
```

---

## Step 3 — Deterministic Community Detection

Implement hierarchical community detection without heavy dependencies.

```text
Create packages/core/contextmine_core/knowledge/communities.py.

Algorithm (deterministic weighted label propagation):
1. Extract SYMBOL nodes from knowledge graph
2. Build adjacency from RESOLVED edges only (calls, references, extends, implements, contains)
3. Initial label = node natural_key (sorted)
4. Iterate in sorted natural_key order:
   - For each node, compute neighbor label weights
   - New label = highest total weight, tie-break by smallest string
5. Converge when no label changes or max_iters reached
6. Level-1 communities = label groups
7. Build community graph (aggregate edge weights between communities)
8. Repeat label propagation on community graph → Level-2 communities

```python
@dataclass
class CommunityResult:
    level1: dict[str, list[str]]  # community_key -> [node_natural_keys]
    level2: dict[str, list[str]]  # community_key -> [level1_community_keys]
    membership_scores: dict[str, float]  # node_natural_key -> score

async def detect_communities(
    session: AsyncSession,
    collection_id: UUID,
    max_iters: int = 100,
) -> CommunityResult:
    """Deterministic community detection on knowledge graph."""

async def persist_communities(
    session: AsyncSession,
    collection_id: UUID,
    result: CommunityResult,
) -> None:
    """
    Replace all communities for collection.
    Creates KnowledgeCommunity + CommunityMember records.
    """
```

Determinism requirements:
- Same graph → identical community assignments
- No randomness (no random seeds, no shuffling)
- Sorted iteration order everywhere

Tests:
- Determinism: run twice, compare natural_keys
- Disconnected components → separate communities
- Single node → single community
- Star graph → predictable structure

Validation:
- uv run ruff check . && uv run ruff format .
- uvx ty check
- uv run pytest -v

End with git status.
```

---

## Step 4 — Community Summaries and Embeddings

Generate offline summaries for communities (the "global" layer of GraphRAG).

```text
Create packages/core/contextmine_core/knowledge/summaries.py.

For each community, gather:
- Top-K symbols by membership score
- Their definition evidence (file + line range + excerpt)
- File list
- Relation statistics (calls, references, etc.)

Output modes:

1) Extractive mode (no LLM):
```python
@dataclass
class ExtractedSummary:
    title: str  # derived from top symbol names
    top_files: list[str]
    top_symbols: list[str]
    relation_counts: dict[str, int]

def extractive_summary(community_id: UUID, ...) -> ExtractedSummary
```

2) LLM mode (when provider available):
```python
class CommunitySummaryOutput(BaseModel):
    title: str = Field(description="Short descriptive title")
    responsibilities: list[str] = Field(description="What this component does")
    key_concepts: list[str] = Field(description="Main abstractions")
    key_dependencies: list[str] = Field(description="External dependencies")
    key_paths: list[str] = Field(description="Important code paths")
    confidence: float = Field(ge=0, le=1)

async def llm_summarize_community(
    community_id: UUID,
    provider: LLMProvider,
    ...
) -> CommunitySummaryOutput
```

Embeddings:
- Embed community summary text
- Store in KnowledgeEmbedding(target_type='community', target_id=community_id)
- Content hash for idempotency (skip if unchanged)

```python
async def generate_community_summaries(
    session: AsyncSession,
    collection_id: UUID,
    provider: LLMProvider | None = None,
    embed_provider: EmbeddingProvider | None = None,
) -> None:
    """Generate summaries and embeddings for all communities."""
```

Tests:
- Mock LLM returns valid schema
- Extractive mode produces non-empty summary
- Embedding stored with correct hash
- Idempotency: unchanged community not re-summarized

Validation:
- uv run ruff check . && uv run ruff format .
- uvx ty check
- uv run pytest -v

End with git status.
```

---

## Step 5 — GraphRAG Retrieval (Global + Local)

Replace fake graphrag.py with real implementation.

```text
Rewrite packages/core/contextmine_core/graphrag.py.

Current state:
- graph_rag_bundle() does hybrid search + BFS expansion
- No community awareness, no global context
- Natural key format mismatch (bug)

New implementation:
```python
@dataclass
class ContextPack:
    # Global context (from communities)
    communities: list[CommunityContext]

    # Local context (from entities)
    entities: list[EntityContext]

    # Graph structure
    edges: list[EdgeContext]
    paths: list[PathContext]

    # All citations
    citations: list[Citation]

    def to_markdown(self) -> str: ...
    def to_dict(self) -> dict: ...

@dataclass
class CommunityContext:
    community_id: UUID
    level: int
    title: str
    summary: str
    relevance_score: float
    member_count: int

@dataclass
class EntityContext:
    node_id: UUID
    kind: str
    natural_key: str
    title: str
    evidence: list[Citation]
    relevance_score: float

async def graph_rag_context(
    session: AsyncSession,
    query: str,
    collection_id: UUID | None = None,
    user_id: UUID | None = None,
    max_communities: int = 5,
    max_entities: int = 20,
    max_depth: int = 2,
) -> ContextPack:
    """
    Real GraphRAG retrieval.

    1. Seed: hybrid search (existing) + community embedding similarity
    2. Global: select top communities by combined score
    3. Local: map seed hits to graph nodes, expand neighborhood
    4. Paths: find strong paths between key entities
    5. Citations: attach evidence to all claims
    """
```

Retrieval procedure:
1. **Hybrid seed**: Use existing `hybrid_search()` to get candidate docs/chunks
2. **Embed query**: Get query embedding
3. **Community ranking**: Vector similarity over community embeddings + overlap with seed files
4. **Node mapping**: Map seed doc IDs to FILE nodes (fix natural_key format!)
5. **Symbol expansion**: Find SYMBOL nodes in those files
6. **Neighborhood**: BFS expand with depth limit, resolved edges only
7. **Path finding**: BFS shortest paths between top symbols
8. **Citation gathering**: Attach KnowledgeEvidence to all nodes

Delete or deprecate:
- Old graph_rag_bundle() function
- graph_neighborhood() if redundant
- trace_path() if redundant (or keep as utility)

Tests:
- Fixture with communities + nodes + embeddings
- Query returns both community summaries AND entity evidence
- Citations present for all claims
- Deterministic output

Validation:
- uv run ruff check . && uv run ruff format .
- uvx ty check
- uv run pytest -v

End with git status.
```

---

## Step 6 — MCP Tools Update

Expose new GraphRAG via MCP, remove redundant tools.

```text
Update apps/api/app/mcp_server.py.

New/updated tools:
1) `graphrag_context(query, collection_id?, max_communities?, max_entities?)`:
   - Calls graph_rag_context()
   - Returns ContextPack as markdown + optional JSON
   - This is the PRIMARY retrieval tool

2) `graph_node(natural_key_or_id)`:
   - Returns single node with full evidence
   - Keep existing implementation if adequate

3) `graph_neighbors(node, depth?, edge_kinds?)`:
   - Keep but ensure it uses resolved edges only by default

4) `graph_path(from_node, to_node, max_hops?)`:
   - Keep but ensure citations included

Remove or redirect:
- `graph_rag()` old tool → replace with `graphrag_context()`
- `get_markdown()` → consider redirecting to graphrag_context() for code queries
- Any other redundant retrieval tools

Update MCP server instructions to guide tool selection.

Tests:
- graphrag_context returns valid structure
- Empty graph returns empty but valid response
- All tools have consistent citation format

Validation:
- uv run ruff check . && uv run ruff format .
- uvx ty check
- uv run pytest -v

Documentation:
- Update any MCP docs to reflect new tool surface

End with git status.
```

---

## Step 7 — Pipeline Integration

Wire everything into sync pipeline.

```text
Update apps/worker/ sync jobs to use new substrate.

Current flow:
1. Fetch repo/crawl pages
2. Create/update Document records
3. Chunk and embed
4. Build symbol table (Tree-sitter)
5. Build knowledge graph (old builder)

New flow:
1. Fetch repo/crawl pages
2. Create/update Document records
3. Chunk and embed (unchanged)
4. Build Semantic Snapshot (new)
5. Build knowledge graph from Snapshot (new)
6. Detect communities (new)
7. Generate community summaries (new)

Integration points:
- After document sync: call build_snapshot()
- After snapshot: call build_graph_from_snapshot()
- After graph: call detect_communities()
- After communities: call generate_community_summaries()

Incremental behavior:
- Track changed/deleted files
- Only rebuild affected portions
- Community detection runs on full graph (fast enough)
- Summary regeneration uses content hash to skip unchanged

Tests:
- Integration test with mock data
- Verify pipeline produces communities and summaries

Validation:
- uv run ruff check . && uv run ruff format .
- uvx ty check
- uv run pytest -v

End with git status.
```

---

## Execution Order

1. **Step 0**: Schema extensions (communities, embeddings)
2. **Step 1**: Semantic Snapshot layer (Tree-sitter first)
3. **Step 2**: Graph builder from Snapshot
4. **Step 3**: Community detection
5. **Step 4**: Community summaries
6. **Step 5**: GraphRAG retrieval
7. **Step 6**: MCP tools
8. **Step 7**: Pipeline integration

Each step must pass all validation before proceeding.

---

## Success Criteria

After completing all steps:
- [ ] Query returns community summaries (global) + entity evidence (local)
- [ ] All claims have citations (file + line range)
- [ ] Deterministic: same query → same results
- [ ] No parallel retrieval paths (old code removed)
- [ ] Works with Tree-sitter fallback (no SCIP required)
- [ ] All tests pass, no linter errors
