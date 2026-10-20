# Known Issues

Found while dogfooding ContextMine on its own repository (local mode, analyzer
worker, ~550 files / ~3000 symbols), October 2026. Each entry says what was
observed directly and what is only inferred from the code.

## Open

### 1. Architecture recovery is quadratic and hangs syncs of larger repositories

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
  (`recovery.py:280`). O(n²) per component. Introduced in 7dc2b0d.
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
- **Fix idea:** Longer interval for local setups, or run the due-check inside
  the long-lived worker process instead of as a deployment.

### 9. AGE graph sync rebuilds everything and matches edges without an index

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
  ~230 batches of roughly 600 × 2 × 36k comparisons each. The duration of the
  whole step was not measured; at the observed 7+ s per batch it would be
  roughly 30-40 minutes.
- **Fix idea:** Create an index on the `id` property of the `Node` label
  table, or map twin node ids to AGE graph ids while creating nodes and create
  edges from those ids; sync only changed nodes and edges instead of rebuilding.

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
