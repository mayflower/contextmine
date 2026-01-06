
	Key properties this plan enforces:

	* **GraphRAG (real)**: entity graph → community hierarchy → offline community summaries (+ embeddings) → query-time global+local retrieval.
	* **Language/framework agnostic by construction**: everything is driven by **SCIP/LSIF semantic indexes** when available; otherwise a **Tree-sitter structural fallback**. No framework scraping; no naming conventions; no regex “rule extraction.”
	* **No redundancy / no backward compatibility**: you will **replace** existing retrieval pathways and tools (or re-point them) rather than keeping parallel behavior.
	* **Validation + cleanup every step**: each prompt ends with lint/type/test + removal of temporary artifacts + `git status`.


	---

	## Prompt 0 — Non-negotiables + baseline inventory

	```text
	You are Claude Code working in the ContextMine repository.

	Non-negotiables:
	1) Implement REAL GraphRAG: graph → communities (hierarchy) → offline summaries → query-time global+local retrieval.
	2) Must be language/framework agnostic:
	   - Prefer SCIP/LSIF ingestion for definitions/references.
	   - Fallback to Tree-sitter ONLY for structural facts (decls/containment/call-sites as unresolved).
	   - No framework routing scrapers. No naming conventions. No regex-based semantics.
	3) No redundancy and no backward compatibility:
	   - If a new retrieval path/tool replaces old behavior, delete the old code and update all call sites/docs.
	4) Every step must end with:
	   - `uv run ruff check .`
	   - `uv run ruff format .`
	   - `uvx ty check`
	   - `uv run pytest -v`
	   - If DB schema changes: alembic migration + `alembic upgrade head` using repo conventions.
	5) Cleanup: remove temporary files/scripts/log dumps; ensure `git status` is clean except intended changes.

	First task:
	- Inspect repo structure and identify:
	  - where embeddings are stored/used today,
	  - where MCP tools are defined (especially get_markdown),
	  - where Prefect sync/index pipeline runs,
	  - existing Tree-sitter symbol extraction code,
	  - existing LSP integration points.

	Output:
	- A short “baseline inventory” note in `docs/design/graphrag_inventory.md` listing file paths and current behavior.
	- Do not change runtime behavior yet.

	End by running tests/linters (as above) and show `git status`.
	```

	---

	## Step 1 — Write the GraphRAG + agnostic substrate spec (prevents random special cases)

	```text
	Create `docs/design/graphrag.md` defining the new subsystem end-to-end.

	The spec MUST include:
	A) What we mean by GraphRAG (required components):
	   - entity graph (typed nodes/edges + evidence)
	   - community hierarchy (at least two levels)
	   - offline community summaries stored and embedded
	   - query-time retriever using global (community) + local (entity neighborhood) context

	B) Language/framework agnosticism contract:
	   - Semantic Snapshot model that can be populated from SCIP or LSIF
	   - Tree-sitter fallback that only provides structural facts (no semantic resolution)
	   - Degradation rules when only structural facts exist

	C) Data model (conceptual):
	   - KnowledgeNode/Edge/Evidence
	   - Community + membership
	   - Embeddings (node/community)
	   - Natural keys and idempotent rebuild rules

	D) Tool surface:
	   - MCP tools: `graphrag_context`, `graph_node`, `graph_neighbors`, `graph_path`
	   - Declare that legacy retrieval tools will be removed or re-pointed; no compatibility shims

	E) Acceptance tests:
	   - Determinism: same inputs → identical communities and retrieval packs
	   - Non-trivial query: requires graph expansion + community summary to answer (not just lexical search)

	Implementation:
	- Only documentation in this step.

	Validation:
	- Run ruff/format/ty/pytest and show git status.
	Cleanup:
	- No temp artifacts.

	End with: link to spec file and list touched files.
	```

	---

	## Step 2 — Implement the Semantic Snapshot (SCIP/LSIF ingestion + Tree-sitter fallback)

	This is what makes the system genuinely language/framework agnostic.

	```text
	Implement a language-agnostic “Semantic Snapshot” layer used by ALL later steps.

	Create in packages/core (follow repo conventions) a module, e.g. `contextmine_core/semantic_snapshot/`:

	Data model (must be stable, serializable, testable):
	- FileInfo(path, language?)
	- Symbol(def_id, kind, file_path, range(start_line,start_col,end_line,end_col), name?, container_def_id?)
	- Occurrence(file_path, range, role=DEFINITION|REFERENCE, def_id)
	- Relation(src_def_id, kind=CONTAINS|CALLS|REFERENCES|EXTENDS|IMPLEMENTS, dst_def_id, weight?, meta)
	- Snapshot(files, symbols, occurrences, relations, meta)

	Providers:
	1) SCIPProvider:
	   - Ingest .scip files (do not run external indexers yet)
	   - Parse to Snapshot using SCIP symbol ids/occurrences as def_id sources
	2) LSIFProvider:
	   - Ingest .lsif (JSON lines) to Snapshot (definitions/references edges)
	3) TreeSitterProvider (fallback):
	   - Extract declarations + containment edges
	   - Extract syntactic call sites as relations but mark meta.resolution="unresolved"
	   - Do NOT attempt to resolve calls/references with string tricks

	Provider selection:
	- If SCIP exists for a repo/collection, use it.
	- Else if LSIF exists, use it.
	- Else use Tree-sitter fallback.

	Integration point:
	- Add a single entry function:
	  `build_snapshot(collection_id, repo_checkout_path, config) -> Snapshot`
	  This is the only interface later steps use.

	Testing:
	- Unit tests for Snapshot model
	- Provider tests using minimal in-test fixtures (avoid opaque binaries if possible)
	- Ensure Tree-sitter fallback test runs on a tiny OO snippet (any supported language)

	Validation:
	- ruff/format/ty/pytest

	Cleanup:
	- No debug dumps committed, no temp files.

	End with summary + list touched files + git status.
	```

	---

	## Step 3 — Add the property graph + evidence + communities + embeddings schema (DB foundation)

	```text
	Implement the DB schema and core repository methods for:
	- Knowledge Nodes/Edges
	- Evidence and evidence links
	- Community hierarchy + memberships
	- Embeddings for nodes and communities

	Requirements:
	- Must be generic, typed by `kind` and `meta` jsonb
	- Must support idempotent upserts via stable natural keys
	- Must support deletions when files disappear
	- Must not duplicate raw file contents in graph storage

	Tables (or equivalent) required:
	- knowledge_node(collection_id, kind, natural_key UNIQUE, title, meta, created_at, updated_at)
	- knowledge_edge(collection_id, kind, natural_key UNIQUE, src_node_id, dst_node_id, weight, meta)
	- knowledge_evidence(collection_id, source_type, source_ref, start/end line/col, excerpt, hash, UNIQUE by (collection_id, hash))
	- node_evidence_link(node_id, evidence_id)
	- edge_evidence_link(edge_id, evidence_id)
	- knowledge_community(collection_id, level, natural_key UNIQUE, title, summary, meta)
	- community_member(community_id, node_id, score)
	- knowledge_embedding(target_type node|community, target_id, model/provider, vector, content_hash UNIQUE)

	Implementation:
	- SQLAlchemy models + migrations (Alembic)
	- Repository methods:
	  - upsert_node/upsert_edge
	  - attach_evidence
	  - replace_collection_graph_for_files(changed_paths, deleted_paths)
	  - replace_collection_communities(level, …)
	  - upsert_embedding

	Validation:
	- ruff/format/ty/pytest
	- alembic migration + upgrade head

	Cleanup:
	- Remove scratch scripts; ensure no redundant schema overlaps remain.

	End with summary + touched files + git status.
	```

	---

	## Step 4 — Build the graph from the Snapshot (no special cases, no heuristics)

	```text
	Implement `GraphBuilder` that converts Snapshot -> Knowledge graph nodes/edges/evidence.

	Mapping:
	- FileInfo -> FILE nodes
	- Symbol -> SYMBOL nodes (meta includes file path and range)
	- Occurrence -> evidence links for SYMBOL nodes (definitions) and REFERENCES edges (if resolvable)
	- Relation -> edges (CONTAINS/CALLS/REFERENCES/EXTENDS/IMPLEMENTS)

	Rules:
	- No naming heuristics.
	- If relation meta.resolution="unresolved" (Tree-sitter fallback), keep the edge but tag it, and ensure later steps can exclude unresolved edges when needed.

	Idempotency:
	- Natural keys must be stable:
	  - FILE: `file:{path}`
	  - SYMBOL: `symbol:{def_id}` (SCIP/LSIF) OR fallback `symbol:{path}:{range}:{kind}`
	  - EDGE: `edge:{kind}:{src_nk}->{dst_nk}`
	  - EVIDENCE: hashed (location + excerpt)

	Incremental behavior:
	- On sync, only rebuild for changed files; remove nodes/edges for deleted files.
	- Do not “append forever.”

	Tests:
	- Synthetic Snapshot unit test (no external tools):
	  - verifies edges and nodes created
	  - verifies idempotency (run twice, identical counts)
	  - verifies deletion removes graph entries

	Validation:
	- ruff/format/ty/pytest

	Cleanup:
	- no graph dump files.

	End with summary + touched files + git status.
	```

	---

	## Step 5 — Community hierarchy (deterministic) and membership scoring

	This makes it GraphRAG (hierarchical communities).

	```text
	Implement deterministic community detection and store a 2-level hierarchy.

	Constraints:
	- Deterministic output (same graph -> same communities)
	- No heavy compiled deps
	- No language/framework assumptions

	Algorithm (implement exactly; no substitutions unless justified in spec):
	1) Consider SYMBOL nodes only.
	2) Use only RESOLVED edges: CALLS, REFERENCES, EXTENDS, IMPLEMENTS, CONTAINS
	3) Build undirected weighted adjacency.
	4) Deterministic weighted label propagation:
	   - initial label = node natural_key
	   - iterate nodes in sorted natural_key order
	   - new label = highest total weight among neighbor labels
	   - tie-break by smallest label string
	   - stop on convergence or max_iters
	5) Level-1 communities = label groups
	6) Build community graph (communities as nodes; weights aggregated)
	7) Repeat label propagation on community graph to get level-2 communities

	Persistence:
	- knowledge_community(level=1 and level=2)
	- community_member for level-1 (node memberships) with deterministic score (e.g., normalized internal degree)

	Replacement:
	- Replace all communities for the collection each run (simple + deterministic). No duplication.

	Tests:
	- determinism test (2 runs -> identical natural keys for communities and memberships)
	- disconnected graphs -> separate communities

	Validation:
	- ruff/format/ty/pytest

	Cleanup:
	- remove debug adjacency dumps.

	End with summary + touched files + git status.
	```

	---

	## Step 6 — Offline community summaries + embeddings (global layer)

	This is the “global” part of GraphRAG.

	```text
	Implement offline community summarization and embeddings.

	Inputs per community:
	- Top-K representative SYMBOL nodes by deterministic score
	- Their definition evidence excerpts (bounded)
	- List of top FILE nodes in the community
	- High-level relation counts (resolved only)

	Outputs:
	- knowledge_community.summary
	- knowledge_embedding for community and representative nodes
	- Both must be idempotent via content_hash

	Modes:
	- No-LLM mode (default if no API key): produce an extractive structured summary:
	  - Top files
	  - Top symbols
	  - Top relation types
	- LLM mode (optional):
	  - Strict JSON schema output: {title, responsibilities, key_concepts, key_dependencies, key_paths, confidence}
	  - Must be derived only from provided evidence; no hallucination
	  - Temperature 0; bounded tokens; retry on schema failure

	Tests:
	- mock LLM to verify schema validation and idempotency
	- no-LLM mode produces stable non-empty summary

	Validation:
	- ruff/format/ty/pytest

	Cleanup:
	- remove prompt dumps and temporary files.

	End with summary + touched files + git status.
	```

	---

	## Step 7 — Query-time GraphRAG retrieval (global + local) producing a context pack with citations

	This is the “retrieval” part of GraphRAG, not optional.

	```text
	Implement GraphRAG retrieval as a single core function (not tied to MCP):

	`graph_rag_context(collection_id, query, options) -> ContextPack`

	ContextPack must include:
	- global: top-N communities with summaries + why selected
	- local: top-M entity nodes (symbols/files) with evidence excerpts
	- neighborhood: edges among selected nodes + 1-hop expansion
	- paths: a small number of strongest paths between key seed entities (resolved edges only)
	- citations: every excerpt must point to knowledge_evidence (file+range), no uncited text

	Retrieval procedure:
	1) Seed:
	   - hybrid search over existing index (docs/code) to get candidate files/chunks
	   - vector similarity over knowledge_embedding (communities + nodes)
	   - combine using RRF (single unified ranking)
	2) Global selection:
	   - select communities by embedding similarity + overlap with seed nodes
	3) Local selection:
	   - map seed hits to graph nodes (FILE and SYMBOL)
	   - expand neighborhood (depth <= 2) using resolved edges
	4) Build pack with strict size limits (token budget estimate)
	5) Deterministic ordering of outputs.

	Tests:
	- Build a tiny test graph fixture:
	  - Ensure query returns both:
	    - community summaries (global)
	    - entity evidence (local)
	  - Ensure determinism
	  - Ensure citations present

	Validation:
	- ruff/format/ty/pytest

	Cleanup:
	- remove temporary test scripts.

	End with summary + touched files + git status.
	```

	---

	## Step 8 — MCP: replace existing retrieval with GraphRAG (no redundancy, no compat)

	```text
	Expose GraphRAG via MCP and remove redundant/legacy retrieval implementations.

	Implement MCP tools (exact names):
	1) graphrag_context(query, collection_id?, limits?) -> returns ContextPack Markdown + structured fields
	2) graph_node(node_natural_key or id) -> node + evidence
	3) graph_neighbors(node, depth=1, edge_kinds?) -> subgraph + evidence
	4) graph_path(from, to, max_hops=6) -> path explanation with citations

	Replace/remove:
	- Find existing MCP tool(s) providing retrieval (e.g., get_markdown).
	- Either:
	  A) remove them and update docs + client examples, OR
	  B) re-implement them to call graphrag_context internally and delete old logic.
	Do not keep two parallel retrieval behaviors.

	Docs:
	- Update README / MCP docs to reflect new tools and behavior.

	Tests:
	- MCP routing tests ensuring tools respond with expected structure.
	- Ensure empty graph returns sane empty responses.

	Validation:
	- ruff/format/ty/pytest

	Cleanup:
	- delete old retrieval code paths and unused utilities.
	- ensure docs match code.

	End with summary + touched files + git status.
	```

	---

	# Optional Phase 2 (after GraphRAG substrate is real): derived knowledge without special cases

	These are “on top of GraphRAG” and remain agnostic because they’re graph transforms or protocol-parsers, not framework scraping.

	## Step 9 — Business rules (candidate mining) as AST control-flow facts + optional labeler

	```text
	Implement RULE_CANDIDATE mining as AST-derived control-flow facts.

	RuleCandidate definition (language-agnostic concept):
	- a guard predicate that leads to a failure effect (throw/raise/return error)
	- captured as spans + container symbol link + evidence

	Implementation:
	- Plugin interface per language using Tree-sitter queries (no naming heuristics).
	- Start with 2 languages already supported by your Tree-sitter setup.
	- Store as KnowledgeNode(kind=RULE_CANDIDATE) + edges to SYMBOL and FILE evidence.

	Optional labeler (LLM):
	- Converts RULE_CANDIDATE -> BUSINESS_RULE with strict schema + citations
	- Must be idempotent and temperature 0

	Validation:
	- tests for candidate detection, idempotency, schema validation
	- ruff/ty/pytest

	Cleanup:
	- no debug dumps.

	End with summary + touched files + git status.
	```

	## Step 10 — ERM (DDL-first, structural parsing only)

	```text
	Implement ERM extraction as DDL-first (framework-agnostic).

	Inputs:
	- SQL DDL files parsed structurally using a real SQL parser (no regex).
	Outputs:
	- Nodes: DB_TABLE, DB_COLUMN, DB_CONSTRAINT
	- Edges: HAS_COLUMN, FK_TO, HAS_CONSTRAINT
	- Artifact: Mermaid ER diagram stored as KnowledgeArtifact + evidence

	Validation:
	- fixtures + tests + ruff/ty/pytest
	Cleanup:
	- no temp files

	End with summary + touched files + git status.
	```

	## Step 11 — arc42 as derived summaries from communities + graph paths (no guessing)

	```text
	Generate arc42 sections from existing communities and graph facts.

	Mapping:
	- Level-2 communities -> building blocks
	- Cross-community edges -> dependencies/interfaces
	- Strong paths -> runtime scenarios
	- DB graph -> data view

	Store as artifact; expose via MCP.
	Every claim must cite evidence or be explicitly marked as inference with confidence.

	Validation:
	- deterministic output tests
	- ruff/ty/pytest
	Cleanup:
	- no temp docs

	End with summary + touched files + git status.
	```

	---

	## What you should expect after Step 8

	You will have:

	* A **language-agnostic semantic substrate** (SCIP/LSIF when present, Tree-sitter fallback when not).
	* A **real GraphRAG implementation**:

	  * graph,
	  * hierarchical communities,
	  * offline summaries + embeddings,
	  * query-time global+local context pack with citations.
	* **One** retrieval path exposed via MCP (no redundant old search stack).


