# Known Issues

Found while dogfooding ContextMine on its own repository (local mode, analyzer
worker, ~550 files / ~3000 symbols), October 2026. Each entry says what was
observed directly and what is only inferred from the code.

## Open

### 1. Architecture recovery is quadratic and hangs syncs of larger repositories

- **Status:** Confirmed by the maintainer of the architecture recovery. The
  cubic evidence merge is fixed on `fix/sync-time-sinks` (3d73670): one file
  with 800 symbols 51 s → 0.84 s, output byte-identical for 60 random graphs.
  The event-loop blocking is fixed in eb73244 (recovery runs in a worker
  thread; observed before: the web UI's arc42 drift view made the API
  unresponsive for minutes, health 503, MCP timeouts). Repeated recovery per
  build fixed in 9ba23af (cache keyed by inputs): a facts build 107 s →
  55 s, a repeated build 28 s.
- **Intended quadratic part (investigated through ContextMine's own MCP
  tools, consistent with the maintainer's note):** constrained LLM
  adjudication works on self-contained packets. `build_adjudication_packet`
  (`architecture/recovery_llm.py:191-236`, "only packet-local evidence and
  candidate IDs") gets its evidence from `build_snippet_bundle`
  (`recovery_llm.py:120-188`): the hypothesis evidence, the evidence of every
  membership of the symbol and the full evidence of every candidate
  component. Only those evidence ids are allowed; answers citing others are
  rejected (`test_apply_adjudication_rejects_outputs_referencing_unknown_evidence_ids`).
  A component built from a file with k symbols carries that file's evidence,
  and so does every membership, so the packets of up to k ambiguous symbols
  hold ~k² evidence items in total. That is deliberate (rules 1 and 2 of
  `docs/design/grounded-llm-detection.md`, phase B2 makes this the primary
  clustering); `llm_hypothesis_limit` (`ARCH_DOCS_LLM_MAX_HYPOTHESES`,
  default 12) bounds how many packets reach the LLM. The pairwise call check
  per file (`recovery.py:924-928`) produces the candidate seeds for it. Not
  intended was re-deduplicating the growing evidence list per member (k³),
  which 3d73670 removed with byte-identical output.
- **Observed:** A sync stayed at "Building digital twin..." (96 %) for over
  50 minutes with ~91 % CPU and no log output. Three py-spy dumps all showed
  `_dedupe_evidence` (`architecture/recovery.py:190`) called from
  `cluster_component_candidates` (`recovery.py:1075`), reached via
  `_generate_arch_docs_on_sync` → `build_architecture_facts` →
  `export_mermaid_c4_result` → `recover_architecture_model`.
- **Cause:** In the loop at `recovery.py:1047-1076`, every symbol assigned to a
  component extends `component_evidence_by_id[component_id]` (never
  deduplicated), then deduplicates the whole accumulated list, and
  `_merge_entity_candidate` deduplicates existing plus new evidence again
  (`recovery.py:280`). For a file cluster of k symbols that is ~k³ work
  (measured: doubling k multiplies the time by 8). Introduced in 7dc2b0d.
- **Amplifiers (from code, not measured):**
  - One arch-docs build runs the full recovery at least three times without
    caching: C4 context and deployment views (`facts.py:575`,
    `mermaid_c4.py:737`, `mermaid_c4.py:932`) and once directly (`facts.py:717`).
    The container, component and code views (`mermaid_c4.py:288/383/619`)
    each run it again when requested.
  - `recovery.py:~985` scans all nodes per symbol and target
    (`next(entity for entity in nodes if ...)`).
  - The recovery is synchronous CPU-bound code inside async functions, so it
    blocks the event loop: the twin step's `asyncio.wait_for` timeout
    (3600 s, `flows.py:~3790`) cannot fire, and the advisory `try/except`
    around arch docs (`flows.py:1934-1941`) only catches exceptions, not hangs.
    In the API, requesting a C4 view of such a collection would block every
    other request (inferred, not tested).
- **Open question:** A sync of Apache Superset (224k twin nodes, see b9211eb)
  apparently completed. Unknown whether it ran this path
  (`ARCH_DOCS_GENERATE_ON_SYNC`), or whether the component distribution
  differs (component ids derive from file names, `component:{file_slug}`).
- **Workaround:** `ARCH_DOCS_GENERATE_ON_SYNC=false`.
- **Fix idea:** Collect evidence per component inside the loop and merge once
  per component afterwards (same result, linear); compute the recovery once
  per build; run it in a worker thread. Add a regression test with thousands
  of symbols in one component.

### 2. LLM provider ignores keys from settings files

- **Status:** Fixed on `fix/llm-keys-from-settings` (branched from `main`).
  Not specific to local mode: on `main` it also breaks the README's host setup
  (keys in `.env`, `uv run uvicorn`), because pydantic-settings reads `.env`
  without exporting it to `os.environ`; Docker Compose hides it by turning
  `.env` into environment variables.
- **Observed:** With keys supplied through `CONTEXTMINE_SECRETS_DIR` (local
  mode), embeddings worked but the knowledge-graph LLM step failed:
  `ValueError: LLM provider configured but failed to initialize:
  OPENAI_API_KEY environment variable not set` (`flows.py:1502-1504`).
- **Cause:** `get_llm_provider` reads `os.environ["OPENAI_API_KEY"]` /
  `["ANTHROPIC_API_KEY"]` directly (`research/llm/provider.py:381`, `:404`)
  instead of `settings.openai_api_key` / `settings.anthropic_api_key`, which
  `get_research_llm_provider` (`provider.py:444-471`) and the embedders use.
- **Fix idea:** Resolve keys via settings (`secret_value(...)`).

### 3. Failed embedding batches are stored as fake embeddings

- **Cause:** `_embed_batch_with_fallback` (`flows.py:625-649`) returns
  `FakeEmbedder` vectors on timeout or any exception. `embed_chunks_for_document`
  stores them under the real embedding model, so the chunks count as embedded,
  are never retried, and vector search silently returns noise for them.
- **Observed:** Not triggered in the dogfooding runs (no "deterministic
  fallback" log lines); the behavior is from the code.
- **Fix idea:** Leave the batch unembedded on failure (or mark it) so a later
  run retries it.

### 4. Missing embeddings are never backfilled

- **Status:** Backfilled once by hand for this collection (4,613 chunks,
  0 errors, with a guard that refuses fake embeddings). The missing
  mechanism itself is still open.
- **Observed:** After a model-free first sync and a second sync with model
  calls enabled, the UI showed "Embeddings 1605 / 6567": all 1603 chunks of
  changed documents were embedded, 4962 of the 4964 chunks from the first run
  were not.
- **Cause:** Only documents whose `content_hash` changed are processed
  (`flows.py:3667`); nothing embeds existing chunks with `embedding IS NULL`.
  There is no full re-index option in API or UI, and resetting the source
  cursor would not help because the hashes are unchanged.
- **Fix idea:** After the document phase, embed chunks of the source that
  lack an embedding for the current model when model calls are enabled.

### 5. The twin is rewritten in full, row by row, on every sync

- **Status:** Row-by-row writes fixed on `fix/sync-time-sinks` (65e3a37):
  SCIP ingest and knowledge-graph seeding use multi-row upserts. Same real
  data, rolled-back transactions, full twin + knowledge-graph state
  identical (sha256 b5639a79d3dde546); ingest 399 s → 53 s, seed
  537 s → 78 s. 563e8b2 (executemany instead of huge VALUES statements,
  unchanged rows skipped; state identical) brought it to ingest 20 s and seed
  24 s. Still open: SCIP output is still ingested in full on every sync, and
  seeding runs twice per sync.
- **Note:** Much of the per-row time was Python building SQLAlchemy
  statements (py-spy: `stmt.excluded` in `_upsert_twin_edge`), not
  database round trips.
- **Observed:** An incremental sync (155 changed documents) spent more than
  20 minutes in "Building digital twin..." inside a single transaction issuing
  individual `INSERT INTO twin_edge_layers ... ON CONFLICT DO NOTHING`
  statements. Progress is invisible from outside (counters and rows only
  appear on commit).
- **Cause:** `build_twin_graph` (`flows.py:1822ff`) ingests every SCIP
  snapshot completely (`ingest_snapshot_into_as_is`) and then seeds from the
  knowledge graph; `twin/service.py:395-423` upserts each edge and each edge
  layer (and each node and node layer) with its own statement. On this repo
  that is roughly 36k nodes, 48k node layers, 138k edges and 138k edge
  layers, i.e. ~360k round trips per sync (counts from the first run).
- **Fix idea:** Bulk inserts (multi-row `INSERT` or `COPY`), restrict the
  ingest to changed files, report progress inside the step.

### 6. Prefect cancellation leaves the sync run `running`

- **Observed:** After cancelling a hung flow run in Prefect, the `sync_runs`
  row stayed `running` (UI showed the sync as running) until the next
  "Sync now", which reconciled it via `_release_finished_sync_run`
  (`routes/sources.py:448`): "RECONCILED_WITH_PREFECT: ... ended as CANCELLED".
- **Fix idea:** Reconcile on read or in the periodic due-check as well.

### 7. Branch of a GitHub source cannot be configured

- `validate_github_url` always stores `branch: None`
  (`routes/sources.py:119-135`); neither API nor UI accept a branch, while
  README says "Branch: defaults to the default branch" as if it were optional.
- Without a branch the worker uses `origin/HEAD` from the first clone and does
  not refresh it on later fetches (`github_sync.py:240-255`). For a local
  repository (`LOCAL_REPOSITORIES_DIR`) that is whichever branch was checked
  out at the first sync.
- **Fix idea:** Branch field in create/update (API and UI).

### 8. Every due-check spawns a Python process

- `sync_due_sources` runs every `PREFECT_DUE_INTERVAL_SECONDS` (60 s). The
  Prefect process worker starts a new `python -m prefect.engine` per run,
  which costs ~10 s of a full core for ~0.4 s of work, i.e. roughly a sixth of
  a core while idle. After downtime, Prefect replays all pre-scheduled runs
  (observed: ~130 runs, two at a time).
- After an unclean shutdown (host lost power), a manual "Sync now" waited
  8 minutes as `Late`: the worker picks runs by expected start time, so it
  first worked through the overdue due-checks scheduled before the outage.
  Due-check runs left `PENDING` by the dead worker stay `PENDING`.
- **Fix idea:** Longer interval for local setups, or run the due-check inside
  the long-lived worker process instead of as a deployment.

### 9. AGE graph sync rebuilds everything and matches edges without an index

- **Status:** Confirmed by the maintainer. Edge matching fixed on
  `fix/sync-time-sinks` (049e37c): matching in `WHERE` instead of a property
  map turns the plan into a hash join, 51 s → 0.66 s per batch of 600 edges on
  36k nodes (PostgreSQL 16.12, AGE 1.5.0). 14b807a (TRUNCATE instead of
  DETACH DELETE, batches of 2000) brought a full rebuild of the real
  scenario from ~590 s to 292 s. Still open: any twin change still rebuilds
  the whole graph (no incremental sync).
- **Observed:** After the twin rows were written, Postgres ran at ~100 % CPU
  while the worker sat idle; the active statement was a single
  `SELECT * FROM cypher('twin_<scenario>', $$ UNWIND [...] AS row MATCH ...`
  running for 7+ seconds.
- **Cause:** `sync_scenario_to_age` (`graph/age.py`, called at
  `flows.py:1946`) deletes the whole AGE graph (`MATCH (n) DETACH DELETE n`)
  and recreates it on every sync. Nodes are created in batches of 800, which
  is fast. Edges are created in batches of 600 with
  `MATCH (s:Node {id: row.source_id}), (t:Node {id: row.target_id})`, but no
  index exists on the `id` property (the code creates none; AGE only indexes
  its internal graph ids), so every row scans all nodes: O(edges × nodes).
  On this repo, ~36k nodes and ~138k edges (counts from the first run) give
  ~230 batches. A synthetic benchmark of the same shape measured 51 s per
  batch, i.e. roughly 3 hours for the step (the 7+ s seen during the sync was
  the age of a still-running statement).
- **Fix idea:** Create an index on the `id` property of the `Node` label
  table, or map twin node ids to AGE graph ids while creating nodes and create
  edges from those ids; sync only changed nodes and edges instead of rebuilding.

### 10. Architecture recovery output depends on the Python hash seed

- **Observed:** The same input produced a different evidence order in
  recovered hypotheses under different `PYTHONHASHSEED` values (found while
  verifying the fix for issue 1; independent of it).
- **Cause:** Candidate components are iterated from sets
  (e.g. `selected_seed_components_by_subject` in
  `cluster_component_candidates`), and string hashing is randomized per process.
- **Impact:** Recovered models, and anything rendered or stored from them,
  can differ between runs without any change in the input. This contradicts
  rule 5 of `docs/design/grounded-llm-detection.md` ("Same input -> same
  output").
- **Fix idea:** Iterate sorted, or use insertion-ordered dicts instead of sets.

### 11. Behavioral layers reload all symbols for every test case

- **Status:** Twin graph reload fixed on `fix/sync-time-sinks` (7b5ccb8): one
  `TwinScipGraph` per tests/UI graph build. Compared on real data from this
  repository (84 test cases from 13 files, each variant in a rolled-back
  transaction): identical `resolved_symbol_refs` for every test case,
  2278 symbol links in both, twin queries 168 → 2, 441.8 s → 122.7 s
  (5.3 s → 1.5 s per test case). The knowledge-symbol reload is fixed in
  a15bbe3 (shared `KnowledgeSymbolIndex`): same 84 test cases, identical
  `resolved_symbol_refs`, 439.4 s → 8.8 s with both caches shared (symbol +
  twin loads 254 → 5; run without LSP).
- **Origin:** Both caches came with the resolver in de8a981 (2026-02-23), but
  `resolve_symbol_refs_for_calls` creates a resolver per call, so they only
  lived for one test case or view. The repository then had ~520 test
  functions; the cost grows with test cases × graph size.
- **Observed:** The sync phase "Materializing behavioral layers" ran for
  45+ minutes with the worker at ~100 % of one core in `_jsonb_decoder`,
  issuing about one `SELECT ... FROM knowledge_nodes WHERE collection_id = ...
  AND kind = ...` per second.
- **Cause:** `build_tests_graph` calls `_resolve_and_link_symbols`
  (`analyzer/extractors/tests.py:1707`) per test case, which calls
  `resolve_symbol_refs_for_calls`; that creates a new `SymbolTraceResolver`
  every time (`analyzer/extractors/traceability.py:693`). Its caches are per
  instance, so each test case reloads every SYMBOL knowledge node with its
  JSONB metadata (`_ensure_knowledge_symbols`, `traceability.py:399ff`).
  Test cases that need SCIP resolution also reload the whole twin graph
  (`_ensure_scip_graph`: all twin nodes and edges, here ~36k and ~138k):
  sampled queries cycled knowledge_nodes → twin_nodes → twin_edges about every
  5 s, with the worker at ~185 % CPU. Cost grows with test cases × graph size;
  at the synced commit this repository has ~4,050 Python test functions and
  ~490 JS/TS `it()`/`test()` calls (grep count, not the extractor's count).
- **Fix idea:** One resolver per build, shared by all test cases (and
  possibly the UI extractor, not checked). Not yet verified to give an
  identical result: the resolver also creates SYMBOL nodes from SCIP and
  appends them to its cache, while a fresh resolver reloads them without
  `ORDER BY`, so candidate order for same-named symbols can differ. Compare
  the test-to-symbol edges of both variants on real data before changing it.
- **Note:** Long runtime is expected here to some degree (test extraction and
  SCIP/LSP/Joern call resolution); the reload per test case is the avoidable
  part.

### 12. AGE is rebuilt twice per sync, and every Cypher query rebuilds it again

- **Status:** Fixed on `fix/age-graph-freshness` (f57c3f0): one committed
  build per sync, a fingerprint per scenario (`twin_age_graphs`, migration
  024), queries rebuild only when the twin changed. Real data: unchanged
  twin 0.27 s instead of a full rebuild per query; a full rebuild still takes
  11-13 minutes (batch size and `TRUNCATE` are the next levers).
- **Observed:** Each sync rebuilds the whole AGE graph twice: at the end of
  `build_twin_graph` (`flows.py:1946`, "Keep AGE in sync as a mandatory M1
  requirement", b97f81f) and at the end of the behavioral layers
  (`flows.py:2164`, de8a981), which re-seed the twin and rebuild again so the
  new test/UI/flow nodes are included. The first build is superseded minutes
  later; it only matters if the behavioral layers fail (their errors are
  caught). Web sources (`sync_web_source`) follow the same order.
- **Readers:** The only readers of AGE are the MCP tool `query_twin_cypher`
  (`apps/api/app/mcp_domains/twin.py:211`) and `POST .../cypher`
  (`apps/api/app/routes/twin.py:5616`). Both call `sync_scenario_to_age`
  before every query and never commit (`get_session` does not commit,
  `database.py:80`), so each query rebuilds the graph and rolls it back.
  Before the fix for issue 9 that was roughly 3 hours per query on this
  repository; it is still minutes. The two builds in the sync are never read.
  A plausible reason for the per-query rebuild: queries can target TO-BE
  scenarios, which no sync builds into AGE.
- **Fix idea:** One build per sync, committed, plus a record of which
  scenario version is in AGE; queries rebuild (and commit) only when the
  scenario changed since. Quick win without changing query behavior: drop
  both builds from the sync (~20 minutes per sync). Clarify the intended role
  of AGE (persistent query store vs. per-query working graph) first.

### 13. The MCP endpoint only answers with a trailing slash

- **Observed:** `POST /mcp` returns 405 Method Not Allowed; only `POST /mcp/`
  works, so `claude mcp add --transport http contextmine http://localhost:8111/mcp`
  fails to connect. README and `docs/mcp.md` show the URL without the slash.
- **Fix idea:** Accept `/mcp` as well (route or redirect that keeps POST), or
  document `/mcp/`.

### 14. Business rule extraction: one LLM call per code unit, unbounded

- **Observed:** The first sync with working LLM keys spent its knowledge-graph
  phase waiting on the LLM: `extract_rules_from_file`
  (`analyzer/extractors/rules.py`) splits each changed code file into
  functions/classes with Tree-sitter and sends every unit to the research
  model (`RESEARCH_MODEL`, default `claude-sonnet-4-5`) in its own call;
  files run one after another, units of a file concurrently. 15 changed code
  files meant roughly 900 calls, test files included.
- **Unbounded full run:** `_kg_extract_business_rules` (`flows.py`) runs a full
  extraction over all code files when a sync has no changed documents and no
  business rules exist yet ("running initial extraction"); for this
  repository (~470 code files) that is several thousand calls. There is no
  cap, no triage step before the LLM, and tests are not excluded.
- **Parse failures:** Some answers fail validation, e.g. `1 validation error
  for ExtractionOutput rules: Input should be a valid list` with the list
  returned as a JSON string, sometimes with a trailing comma; that unit is
  skipped with a warning (4 in the first minutes).
- **Fix idea:** Triage before the LLM (the deterministic rule candidates from
  `rules.py` / the design doc's cost model), a per-sync cap, exclude tests by
  default, and tolerate stringified JSON in structured output.
- **Semantic entities, same pattern:** `_kg_step_semantic_entities` →
  `extract_from_documents` (`knowledge/extraction.py`) calls the default LLM
  (`gpt-5-mini`) once per chunk, sequentially (about one call every 2 s
  observed); 21 changed documents meant 1,029 chunks. Results are persisted
  only after all chunks, so an interrupted sync loses them, and those
  documents never get entities later (they no longer count as changed). With
  no changed documents and no entities yet, it processes all chunks (6,594
  here, over 3.5 hours). `SEMANTIC_EXTRACTION_MAX_CHUNKS` caps it but
  defaults to 0 (unlimited).

### 15. Semantic entities: duplicate natural keys abort the whole step

- **Observed:** An incremental sync's semantic-entity step failed with
  `duplicate key value violates unique constraint "uq_knowledge_node_natural"`
  for `entity:architecture_recovery`; no entities were stored. On the run
  before, the knowledge-graph pipeline hit its 3600 s timeout during the
  same step (`kg_errors: pipeline: timeout_after_3600s`), see issue 14.
- **Cause:** `persist_semantic_entities` inserted a new node per extracted
  entity although the natural key is unique per collection: entities from
  unchanged documents survive incremental runs, and names differing only in
  case or spacing share a key.
- **Status:** Fixed on `fix/semantic-entity-duplicates` (24bd984): reuse and
  merge by natural key, conflict-safe edges, PostgreSQL regression test.

### 16. A chunk budget fills up alphabetically, not by importance

- **Observed:** With `SEMANTIC_EXTRACTION_MAX_CHUNKS=150`, all extracted
  entities came from `apps/web`, `apps/api`, `apps/api/tests` and
  `apps/worker`; none from `packages/core`, where knowledge graph, twin,
  architecture recovery and GraphRAG live.
- **Cause:** `_chunk_priority_tuple` (`knowledge/extraction.py`) sorts by
  chunk kind, path penalty, chunk index, path. Nearly all chunks share kind
  and penalty, so the budget takes the first chunk of each file in
  alphabetical path order.
- **Status:** Fixed on `fix/semantic-budget-and-summaries`: under a budget,
  chunks are taken one per directory per round (78 directories here).

### 17. Community summaries are wiped and never regenerated

- **Observed:** After a knowledge-graph pass that extracted 176 entities into
  191 communities, none of the communities had a summary.
- **Cause:** `persist_communities` deletes and recreates every community on
  each pass, without summary. `_kg_step_summaries` skipped generation when
  no documents changed and semantic entities existed, using entities as a
  proxy for "summaries exist". Entities created earlier in the same pass,
  or summaries lost by an aborted pass, therefore left communities without
  summaries. The membership-hash cache in `generate_community_summaries`
  never hit, because the summaries it compares were always gone, so passes
  with changes re-summarized every community with the LLM. Its cache branch
  also referenced `EmbeddingTargetType.COMMUNITY_SUMMARY`, which does not
  exist. Community embeddings have no foreign key and were left orphaned.
- **Status:** Fixed on `fix/semantic-budget-and-summaries`: communities with
  the same members keep summary, title, hash and embedding; the summaries
  step always runs and skips unchanged communities; enum fixed; embeddings
  of dropped communities are deleted.

### 18. A sync without file changes skips the knowledge graph entirely

- **Observed:** A sync after an aborted extraction finished in 3 minutes
  without any LLM call; semantic entities stayed empty.
- **Cause:** `sync_github_source` exits early when no files changed
  (`flows.py`, "Early exit: skip expensive LLM/embedding phases"), before
  the knowledge-graph phase. That phase has its own catch-up logic (initial
  extraction when no entities exist), which this exit bypasses.
- **Fix idea:** Let the early exit still run knowledge-graph steps that are
  missing their output, or offer a way to trigger the knowledge-graph build
  for a source on its own.

### 19. Worker application logs are hard to follow

- **Observed:** `docker compose logs worker` is empty with the journald log
  driver (`journalctl CONTAINER_NAME=contextmine-worker-1` works), and the
  flow subprocess only emits warnings; step progress (`logger.info`) is not
  visible anywhere, and Prefect only stores its own messages.
- **Fix idea:** Configure INFO logging for the flow process and forward
  `contextmine_*` loggers to Prefect.

## Fixed on `feat/local-dev-mode`

- **Postgres data lost on `docker compose down`:** the named volume was
  mounted at `/var/lib/postgresql`, while the image's `VOLUME`/`PGDATA` is
  `/var/lib/postgresql/data`, so data lived in an anonymous volume
  (since the initial commit). Fixed in cf0fcd1 with a new volume name.
- **Compose worker image cannot index:** the default `worker` target has no
  SCIP/LSP tools, so syncs "succeed" without symbols or metrics.
  `docker-compose.local.yml` uses the `analyzer` target; the base
  `docker-compose.yml` is unchanged.
- **Docs:** Compose port 8111 (not 8000), GitHub OAuth callback must be the
  bare origin because MCP uses `/mcp/auth/callback`, worker entry point is
  `contextmine_worker.main` (a4fb961).
