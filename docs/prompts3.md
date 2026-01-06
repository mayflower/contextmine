# Analyzer Implementation Guide

This document provides step-by-step prompts for Claude Code to implement a "Derived Knowledge" subsystem in ContextMine: Knowledge Graph, Business Rules extraction, ERM analysis, GraphRAG, and arc42 architecture documentation.

Each step is designed to be:
- **Incremental** (small, testable changes)
- **Deterministic-first** (facts/graphs before LLM labeling)
- **No naming-convention dependence** (use AST / symbol resolution / schemas)
- **Validated at the end** (commands + acceptance checks)
- **Clean** (delete temporary artifacts, remove redundancy)
- **No backward-compat shims** (if something is replaced, remove the old path)

---

## Operating Rules (paste once at the start of a session)

```text
You are Claude Code working inside the ContextMine repository.

Operating rules (must follow):
1) Do not rely on naming conventions (e.g., *Service, *Repository, etc.) or string regex extraction to infer semantics. Prefer AST (Tree-sitter), symbol resolution (LSP/SCIP/LSIF), schema parsers, and graph analysis.
2) Prefer deterministic extraction and stored evidence. LLM usage is allowed ONLY for labeling/summarization of already-extracted facts, and outputs must be schema-validated.
3) No backward-compatibility shims. If you replace something, remove the old implementation and update all call sites. Avoid redundant parallel pipelines.
4) Never leave temporary artifacts behind. Before finishing any step: remove scratch scripts, temporary files, debug logs, and unused modules. Ensure `git status` is clean except intended changes.
5) Every step must end with validation:
   - `uv run ruff check .`
   - `uv run ruff format .`
   - `uvx ty check`
   - `uv run pytest -v`
   - If DB schema changed: `cd packages/core && DATABASE_URL=postgresql+asyncpg://contextmine:contextmine@localhost:5432/contextmine uv run alembic upgrade head`
6) Keep dependencies minimal; if adding a dependency, justify it in code comments and ensure it is used. Avoid "helper" deps that could be replaced with stdlib.
7) When done with a step: summarize changes, list touched files, and show the exact commands run and their results (pass/fail).

Existing infrastructure to build upon:
- packages/core/contextmine_core/models.py: Symbol, SymbolEdge, SymbolKind, SymbolEdgeType tables
- packages/core/contextmine_core/graph/: CodeGraph in-memory structure, builder, retrieval
- packages/core/contextmine_core/treesitter/: Tree-sitter parsing for symbols
- packages/core/contextmine_core/lsp/: LSP client for definitions/references
- apps/api/app/mcp/: MCP server with existing tools (get_markdown, list_collections, outline, find_symbol, etc.)

Start by briefly inspecting the repo structure, then proceed with Step 1 when asked.
```

---

## Step 1 - Add a language-agnostic Knowledge Graph storage layer

```text
Implement the foundation for a language-agnostic Knowledge Graph in ContextMine.

Goal:
Extend the existing Symbol/SymbolEdge tables to support a generic knowledge graph with:
- Nodes (typed entities: File, Symbol, Table, Endpoint, RuleCandidate, BusinessRule, BoundedContext, Arc42Section, etc.)
- Edges (typed relations: DEFINES, CALLS, REFERENCES, HAS_COLUMN, FK_TO, EMITS, CONSUMES, DOCUMENTED_BY, etc.)
- Evidence (file path + line spans + document/chunk references) attached to nodes/edges
- Artifacts (Mermaid ERD, arc42 docs, rule catalogs) with evidence links

Requirements:
- Put the models in packages/core/contextmine_core/models.py (follow existing patterns).
- Use Alembic migrations (files go in packages/core/alembic/versions/).
- Idempotency & incremental updates:
  - Each node and edge must have a stable "natural key" (e.g., collection_id + kind + external_id/path + span + lang + fqname) so rebuilds upsert without duplication.
- Reuse existing identifiers where possible (reference Document.id, Source.id rather than copying).
- Keep schema generic: use typed nodes + JSON metadata with indexes, NOT one table per concept.

Deliverables:
1) SQLAlchemy models + Pydantic schemas for:
   - KnowledgeNode
   - KnowledgeEdge
   - KnowledgeEvidence (link table for node<->evidence and edge<->evidence)
   - KnowledgeArtifact (+ artifact_evidence links)
2) Indexes for common queries:
   - (collection_id, kind)
   - (kind, natural_key) unique
   - (src_node_id, edge_kind), (dst_node_id, edge_kind)
   - JSON metadata GIN index
3) Alembic migration file: packages/core/alembic/versions/013_add_knowledge_graph.py

Validation (must run and pass):
- `uv run ruff check .`
- `uv run ruff format .`
- `uvx ty check`
- `uv run pytest -v`
- Run alembic upgrade head

At the end:
- Summarize the schema decisions (natural keys, evidence model).
- List touched files.
- Show `git status`.
```

---

## Step 2 - Build graph nodes/edges from existing indexing output

```text
Add a graph builder that populates the Knowledge Graph from existing ContextMine indexing outputs.

Scope (MVP graph):
- File nodes for indexed code files and docs (from Document table)
- Symbol nodes for extracted symbols (from existing Symbol table - reuse, don't duplicate)
- Edges:
  - FILE_DEFINES_SYMBOL
  - SYMBOL_CONTAINS_SYMBOL (nesting via parent_name)
  - FILE_IMPORTS_FILE (from SymbolEdge with type=imports)
- Evidence:
  - Each symbol node must link to evidence (document_id + start_line/end_line)

Integration:
- Add to packages/core/contextmine_core/graph/ or new analyzer/ module
- Hook into the worker sync pipeline (apps/worker) so graph updates run after indexing completes.
- Make it incremental: only rebuild graph entries for changed files/documents since last sync.
- Ensure upsert behavior using natural keys; no duplicates on re-run.

Testing:
- Add fixtures under packages/core/tests/fixtures/ with at least two Python files and nested symbols.
- Add unit tests in packages/core/tests/test_knowledge_graph.py verifying:
  - Correct node counts
  - Correct edge kinds
  - Idempotency (running builder twice does not duplicate)

Validation:
- `uv run ruff check .`
- `uv run ruff format .`
- `uvx ty check`
- `uv run pytest -v`

End with:
- Short summary of graph builder flow and incremental strategy.
- List touched files and `git status`.
```

---

## Step 3 - ERM extraction (migrations -> tables/columns/constraints) + Mermaid ERD

```text
Implement database ERM extraction and ERD artifact generation.

Key rule: avoid string-regex heuristics. Use parsers/AST when possible.

MVP extractors (plugin-based):
1) SQL migration files:
   - Parse DDL (CREATE TABLE, ALTER TABLE ADD COLUMN/CONSTRAINT, FK constraints).
   - Use sqlglot or a minimal SQL parser (justify dependency).
2) Alembic migrations (Python):
   - Parse Python AST to detect op.create_table, op.add_column, op.create_foreign_key, etc.
   - Use ast module, NOT regex on strings.

Graph output:
- Nodes:
  - DB_TABLE (qualified name)
  - DB_COLUMN (table + name)
  - DB_CONSTRAINT (type + details)
- Edges:
  - TABLE_HAS_COLUMN
  - COLUMN_FK_TO_COLUMN
  - TABLE_HAS_CONSTRAINT

Artifact:
- Generate Mermaid ER diagram per collection, store as KnowledgeArtifact(kind="ERD_MERMAID").
- Link artifact to evidence (which migrations contributed).

Testing:
- Add fixtures: a tiny SQL migration + an Alembic migration.
- Tests must validate:
  - Table/column nodes created
  - FK edges created
  - Mermaid output is stable (snapshot test or structured assertions)
  - Idempotent rebuild

Validation:
- `uv run ruff check .`
- `uv run ruff format .`
- `uvx ty check`
- `uv run pytest -v`

End with summary + touched files + git status.
```

---

## Step 4 - System Surface Catalog (spec-driven): OpenAPI + GraphQL + Protobuf + Jobs

```text
Implement a System Surface Catalog that is spec-driven and cross-language.

Avoid framework-name heuristics in source code for now. Prefer "hard" sources:
- OpenAPI specs (yaml/json)
- GraphQL schema files (.graphql/.gql)
- Protobuf (.proto)
- Job definitions (GitHub Actions yaml, Kubernetes CronJob manifests, Prefect deployment specs)

Extraction outputs:
- Nodes:
  - API_ENDPOINT (method + path, schema refs if available)
  - GRAPHQL_OPERATION (query/mutation/subscription) and types
  - MESSAGE_SCHEMA (proto message/service)
  - JOB (cron/scheduled workflow)
- Edges:
  - SYSTEM_EXPOSES_ENDPOINT
  - ENDPOINT_USES_SCHEMA (if resolvable)
  - JOB_DEFINED_IN_FILE

Implementation:
- Add extractor plugins in packages/core/contextmine_core/analyzer/extractors/
- Interface: input = file inventory + contents, output = nodes/edges + evidence
- Integrate into sync pipeline; incremental updates only.

Testing:
- Fixtures: minimal OpenAPI, GraphQL schema, proto, CronJob yaml.
- Tests must assert node/edge creation and idempotency.

Validation:
- `uv run ruff check .`
- `uv run ruff format .`
- `uvx ty check`
- `uv run pytest -v`

End with summary + touched files + git status.
```

---

## Step 5 - Business Rule candidate mining (deterministic)

```text
Implement deterministic Business Rule Candidate mining without naming conventions.

Definition (MVP):
A RuleCandidate is detected when:
- There is a conditional branch that leads to a failure action (throw/raise/return error)
- Capture predicate AST and failure action AST
- Attach evidence (file+lines) and container symbol (function/method) if available

Notes:
- Do not attempt semantic "this is a rule" via strings.
- Use Tree-sitter AST patterns (packages/core/contextmine_core/treesitter/).
- Implement minimal adapter per supported language, fall back gracefully.

Graph output:
- Node kind: RULE_CANDIDATE
- Metadata: predicate AST snippet, failure kind, container symbol id, confidence score (heuristic), language.

Integration:
- Run after symbol graph build. Store candidates with stable natural keys (file + span + predicate hash).

Testing:
- Fixtures for Python + TypeScript (both supported by tree-sitter in this repo).
- Tests must validate:
  - Detection occurs
  - Metadata exists
  - Evidence spans correct
  - Idempotent rebuild

Validation:
- `uv run ruff check .`
- `uv run ruff format .`
- `uvx ty check`
- `uv run pytest -v`

End with summary + touched files + git status.
```

---

## Step 6 - LLM labeling: RuleCandidate -> BusinessRule

```text
Add an optional LLM labeling step that turns deterministic RuleCandidates into normalized BusinessRules.

Rules:
- LLM is NOT allowed to discover rules from scratch; only to label candidates.
- Output MUST be strict JSON schema (Pydantic) and validated.
- Must include citations: references back to evidence spans and candidate ids.
- Temperature must be 0. Enforce max tokens and safe retries.
- Idempotent: do not relabel unchanged candidates; use content hash of inputs.

Implementation:
- Reuse existing LLM provider patterns from packages/core/contextmine_core/research/llm/
- Add a service function `label_rule_candidates(collection_id)`:
  - Fetch unlabeled RULE_CANDIDATE nodes
  - Build evidence bundle (predicate + failure + surrounding lines + symbol context)
  - Call provider with structured output
  - Persist BUSINESS_RULE node + edges:
    - BUSINESS_RULE_DERIVED_FROM_CANDIDATE
    - BUSINESS_RULE_EVIDENCED_BY
- Store raw model response for audit (truncate if needed).

Testing:
- Mock the LLM client. Ensure:
  - JSON schema validation works
  - Bad outputs are rejected and recorded
  - Idempotency works (second run does nothing)
  - Evidence citations preserved

Validation:
- `uv run ruff check .`
- `uv run ruff format .`
- `uvx ty check`
- `uv run pytest -v`

End with summary + touched files + git status.
```

---

## Step 7 - GraphRAG retrieval (graph expansion + evidence bundle)

```text
Implement GraphRAG-style retrieval on top of the Knowledge Graph.

MVP behavior:
- Input: natural language query + optional collection filter.
- Process:
  1) Use existing hybrid search (packages/core/contextmine_core/search.py) to find relevant documents/symbols.
  2) Map hits to Knowledge Nodes (file/symbol/rule/table/endpoint).
  3) Expand graph neighborhood (depth 1-2, edge-type aware).
  4) Build compact evidence bundle: top nodes + edges + citations + key source snippets.
  5) Return bundle as Markdown (with citations) AND as structured JSON.

Do NOT implement heavy community detection yet. Keep it simple and predictable.

Integration:
- Add to packages/core/contextmine_core/graphrag.py
- Core service functions:
  - `graph_rag_context(session, query, collection_id?, max_depth=2)` - context retrieval
  - `graph_rag_query(session, query, collection_id?, provider)` - answered queries with LLM
- Does not depend on MCP directly.

Testing:
- Unit tests for expansion (given a small graph fixture).
- Ensure output bundle is deterministic and limited in size.

Validation:
- `uv run ruff check .`
- `uv run ruff format .`
- `uvx ty check`
- `uv run pytest -v`

End with summary + touched files + git status.
```

---

## Step 8 - MCP tools for Claude Code: rules, ERD, surfaces, graph, trace

```text
Expose the new capabilities via MCP tools, designed for Claude Code / Cursor usage.

Implement these MCP tools in apps/api/app/mcp/ (following existing patterns):
1) list_business_rules(collection_id?, query?) - list rules with evidence
2) get_business_rule(rule_id) - full rule details
3) get_erd(collection_id?, format="mermaid") - return ERD artifact
4) list_system_surfaces(collection_id?, kind=["endpoint","job","schema"]) - catalog view
5) graph_neighborhood(node_id, depth=1, edge_kinds?) - local graph exploration
6) trace_path(from_node_id, to_node_id, max_hops=6) - shortest path with citations
7) graph_rag(query, collection_id?) - returns the bundle from Step 7

Requirements:
- Each tool must return:
  - Concise Markdown for the assistant
  - Structured fields for programmatic use
- Citations must be evidence-backed (file+line, document chunk).
- If renaming/removing old MCP tools that overlap, remove them completely.

Testing:
- Add MCP-level tests for tool routing and output shape.
- Ensure tools behave correctly when no data exists (empty collections).

Validation:
- `uv run ruff check .`
- `uv run ruff format .`
- `uvx ty check`
- `uv run pytest -v`

End with summary + touched files + git status.
```

---

## Step 9 - arc42 "Architecture Twin": generate + store + drift report

```text
Add an arc42 architecture artifact generator driven by extracted facts (graph + surfaces + ERD + rules).

Deliverables:
1) Artifact generator producing arc42 sections (Markdown), stored as KnowledgeArtifact(kind="ARC42"):
   - Context
   - Building blocks
   - Runtime view (based on traceable paths)
   - Deployment view (from manifests)
   - Crosscutting concepts (observability/security hints from surfaces; keep conservative)
   - Risks & technical debt (from TODO/FIXME + dependency hotspots; evidence-backed)
   - Glossary (from extracted domain terms if available; otherwise minimal)

2) Drift report tool:
   - Compare existing stored ARC42 artifact with current extracted facts
   - Simple diff: missing endpoints/tables/jobs, removed components
   - Output Markdown drift report with evidence

Expose via MCP:
- get_arc42(section?) - retrieve generated architecture doc
- arc42_drift_report() - show what's changed

Rules:
- Do not hallucinate architecture. Every statement must be supported by extracted evidence OR explicitly labeled "inferred" with confidence and rationale.
- Default to DB-stored artifact, not file-based.

Testing:
- Fixtures: minimal graph with endpoints + tables + jobs.
- Tests assert generator produces stable section headers and references key facts.

Validation:
- `uv run ruff check .`
- `uv run ruff format .`
- `uvx ty check`
- `uv run pytest -v`

End with summary + touched files + git status.
```

---

## Step 10 - Final hardening: remove redundancy, document, end-to-end validation

```text
Perform a hardening pass to ensure the new "Derived Knowledge" subsystem is production-grade.

Tasks:
1) Remove redundancy:
   - Identify any duplicated extraction logic and consolidate.
   - Remove dead code paths and unused utilities.
2) Ensure operational correctness:
   - Add timing metrics and structured logs for each extraction stage (no noisy logs).
   - Make every stage idempotent and safe to re-run.
   - Ensure incremental updates are correct (changed files trigger updates; deleted files remove nodes/edges).
3) Documentation:
   - Update README or add docs/KNOWLEDGE_GRAPH.md explaining:
     - Knowledge Graph model
     - How extractors work
     - How to use MCP tools (example queries for Claude Code)
     - How to enable/disable LLM labeling
4) End-to-end test:
   - Add a single e2e test (packages/core/tests/test_analyzer_e2e.py) that:
     - Indexes a small fixture repo
     - Builds graph
     - Extracts schema/surfaces
     - Mines rule candidates
     - Runs graph_rag
     - Verifies MCP tool outputs are non-empty and well-formed

Validation:
- `uv run ruff check .`
- `uv run ruff format .`
- `uvx ty check`
- `uv run pytest -v`
- Confirm migrations apply cleanly on a fresh DB.

End with:
- "Definition of Done" checklist with all commands run and passing.
- List touched files and a concise change summary.
```

---

## Implementation Notes

### Why this approach works without naming conventions

- **Steps 1-5** create a **deterministic, evidence-backed semantic substrate** (graph + extracted facts).
- **Step 6** uses an LLM only as a **labeler**, never as the discoverer.
- **Steps 7-9** turn the substrate into **agent-friendly functionality** (GraphRAG + MCP tools + arc42).
- **Step 10** ensures you don't end up with two competing pipelines or leftover artifacts.

### Existing code to leverage

| Component | Location | Notes |
|-----------|----------|-------|
| Symbol extraction | `packages/core/contextmine_core/treesitter/` | Tree-sitter parsing |
| LSP operations | `packages/core/contextmine_core/lsp/` | Definitions, references |
| In-memory graph | `packages/core/contextmine_core/graph/store.py` | CodeGraph, SymbolNode, Edge |
| Hybrid search | `packages/core/contextmine_core/search.py` | FTS + vector search |
| LLM providers | `packages/core/contextmine_core/research/llm/` | OpenAI, Anthropic, Gemini |
| MCP server | `apps/api/app/mcp/` | Existing tools |
| Worker flows | `apps/worker/` | Prefect sync jobs |

### Running steps

For best results, paste each step individually and validate before proceeding to the next. This allows catching issues early and maintaining a clean commit history.
