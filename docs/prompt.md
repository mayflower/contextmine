Below is a **walking-skeleton, step-by-step prompt set** you can paste into **Claude Code**. Each step is designed to leave the repo in a **working state** (builds, starts via docker compose, tests pass), while adding one cohesive slice of functionality at a time.

I’m assuming you start from an **empty git repo**.

---

## Prompt 0 — Project guardrails (paste once at the start)

```text
You are working in a fresh git repository. Build an open-source local “Context7 alternative” with:

- Backend: Python 3.12, FastAPI, LangChain, Prefect, Postgres+pgvector, hybrid retrieval (FTS + vector), MCP Streamable HTTP (SSE streaming).
- Frontend: React (Vite), polished admin-style UI for collections/sources/tokens/runs.
- Web crawling: spider-rs used for crawling AND HTML→Markdown conversion (invoked from Python worker).
- Git indexing: clone/pull incrementally; support GitHub OAuth (required) and later SSH deploy keys (also required).
- Orchestration: Prefect schedules periodic pull + recrawl; all indexing incremental (hash-based; embed only new chunks; hard delete removed content).
- MCP tool surface: retrieval only (no collection management via MCP). MCP returns an assembled Markdown “context document” produced by an LLM.
- LLM providers allowed: OpenAI, Anthropic, Gemini only. Embeddings: OpenAI and/or Gemini (Anthropic has no embedding API).
- “Web url and pages in the same folder”: enforce same domain + same path prefix; respect robots.txt.

Engineering standards:
- Use uv for Python dependency management (no pip/poetry).
- Use ruff (format+lint), pyright (type checking), pytest.
- Add minimal but real tests for each step.
- Keep each step working: docker compose up --build must start, and core smoke checks must pass.
- Prefer small incremental commits; do NOT rewrite everything at once.

Repository structure (create if missing):
- apps/api (FastAPI + MCP mount)
- apps/worker (Prefect flows/tasks)
- apps/web (React)
- packages/core (shared python code: DB models, settings, services)
- rust/spider_md (spider-rs crawler+html2md binary)

Conventions:
- Backend routes under /api/*
- MCP mounted at /mcp
- Provide .env.example and README with run instructions.
- If you introduce any env vars, document them in .env.example and README.

After each step:
- Run: uv sync, uv run ruff check ., uv run pyright, uv run pytest
- Run: docker compose up --build and verify the documented smoke checks.
Stop after completing the step and summarize what changed + how to run it.
```

---

## Prompt 1 — Walking skeleton: FastAPI + MCP (static tool) + React shell + docker compose

```text
Implement the initial working skeleton.

Backend (apps/api):
- FastAPI app with:
  - GET /api/health -> { "status": "ok" }
  - MCP Streamable HTTP mounted at /mcp using the official mcp python SDK.
- MCP server should expose exactly one tool:
  - name: context.get_markdown
  - input: { "query": string }
  - output: a static Markdown document that includes the query and a placeholder “Not indexed yet” section.
- No auth yet (dev open).

Frontend (apps/web):
- Vite React app with a clean “Admin console” layout:
  - shows API health status fetched from /api/health
  - a page to display MCP connection info (base URL + note “auth comes later”)
- Configure dev proxy in Vite so frontend calls /api/* and /mcp/* via the same origin when running locally.

Infra:
- docker compose with services:
  - api (port 8000)
  - web (port 5173)
- Use uv in the api container build.
- Create root-level README.md with run instructions and curl smoke tests.
- Create .env.example.

Quality:
- Add pytest tests for /api/health (and a lightweight unit test that the MCP tool function exists/returns a markdown string).
- Add ruff + pyright configs at repo root, and ensure backend passes them.

Acceptance criteria:
- docker compose up --build starts api+web.
- curl http://localhost:8000/api/health returns status ok.
- The React UI loads and shows health as ok.
- Tests pass.

Stop after completion and provide the exact commands for running and smoke testing.
```

---

## Prompt 2 — Add Postgres + Alembic + pgvector extension + DB health endpoint

```text
Add Postgres and database plumbing while keeping the app working.

Infra:
- Add postgres service to docker compose (persisted volume).
- Add env vars for DATABASE_URL in .env.example and compose.
- Ensure api container can connect to postgres.

Backend:
- Add SQLAlchemy 2.x async setup in packages/core:
  - async engine + async session factory
  - settings via pydantic-settings
- Add Alembic migrations:
  - initial migration must enable pgvector extension (CREATE EXTENSION IF NOT EXISTS vector).
  - create a minimal table (e.g., app_kv or users) so migration is real.
- Add GET /api/db/health:
  - runs a trivial SELECT 1 via async session and returns { "db": "ok" } or error.

Quality:
- Add integration-ish test for /api/db/health using environment-configured DB (skip if DATABASE_URL not set).
- Update README with how to run migrations (e.g., docker compose exec api ...).

Acceptance criteria:
- docker compose up --build works and postgres is included.
- /api/db/health returns ok when DB is up.
- Alembic migration runs successfully in the container.

Stop after completion and include the migration/run commands.
```

---

## Prompt 3 — GitHub OAuth login (API + UI) and persisted users

```text
Implement GitHub OAuth login for the web UI, storing users in Postgres.

Backend:
- Add tables via Alembic:
  - users: id (uuid), github_user_id (unique), github_login, name, avatar_url, created_at
  - oauth_tokens: user_id FK, provider, access_token_encrypted, created_at
- Implement OAuth with GitHub:
  - /api/auth/login -> redirects to GitHub OAuth
  - /api/auth/callback -> exchanges code, fetches user profile, upserts user, stores encrypted token, starts a session
  - /api/auth/logout -> clears session
  - /api/auth/me -> returns current user or 401
- Add session middleware (secure cookie settings; allow local dev).

Frontend:
- Add a login page with “Sign in with GitHub”.
- After login, show user info in header and a logout button.

Security/Config:
- Add env vars to .env.example: GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET, PUBLIC_BASE_URL, SESSION_SECRET, TOKEN_ENCRYPTION_KEY.
- Add CORS settings appropriate for dev (ports 5173/8000).

Quality:
- Add unit tests for auth logic with mocked GitHub responses (do not require real GitHub).
- Keep existing endpoints working.

Acceptance:
- Login flow works end-to-end with real GitHub creds (document setup).
- /api/auth/me returns user after login.

Stop after completion with setup steps for GitHub OAuth app.
```

---

## Prompt 4 — MCP Bearer tokens + protect /mcp + origin allowlist

```text
Add MCP access tokens and secure the MCP endpoint.

Backend:
- Add tables:
  - mcp_api_tokens: id uuid, user_id, name, token_hash, created_at, last_used_at, revoked_at nullable
- Add API endpoints (session-authenticated):
  - POST /api/mcp-tokens (create): returns plaintext token once
  - GET /api/mcp-tokens (list): returns metadata (no plaintext)
  - DELETE /api/mcp-tokens/{id} (revoke)
- Implement Bearer token auth for /mcp requests:
  - Require Authorization: Bearer <token>
  - Verify token against stored hash (argon2 or bcrypt).
  - Attach user identity to request context for later authorization.
- Implement Origin allowlist for /mcp:
  - Config env: MCP_ALLOWED_ORIGINS (comma-separated). In dev allow empty = allow all.
  - Reject requests with mismatched Origin (per MCP DNS rebinding guidance).

Frontend:
- Add “MCP Tokens” page to create/revoke tokens.

Acceptance:
- /mcp returns 401 without token.
- Works with token from UI.
- Origin check can be toggled via env var.

Stop after completion and document how to configure MCP clients (base URL + token).
```

---

## Prompt 5 — Collections + sharing (global/private) + UI

```text
Add collections and sharing; keep system fully working.

Backend:
- Alembic tables:
  - collections: id uuid, slug unique, name, visibility enum('global','private'), owner_user_id, created_at
  - collection_members: collection_id, user_id, created_at (unique pair)
  - collection_invites: collection_id, github_login, created_at (optional but recommended for “share to colleagues” before they log in)
- API endpoints (session-auth):
  - POST /api/collections
  - GET /api/collections (returns collections visible to user: all global + private where owner/member)
  - POST /api/collections/{id}/share (by github_login; if user exists create member else invite)
  - DELETE /api/collections/{id}/share/{github_login_or_userid}
- On login, if a user has pending invites by github_login, auto-accept into members.

Frontend:
- Collections page:
  - Create collection (name, slug, global/private)
  - List collections
  - Share dialog: enter GitHub username, list members/invites

Acceptance:
- Access control correct: private collections only visible to authorized users.
- Invites are auto-accepted on first login of invited user.

Stop after completion.
```

---

## Prompt 6 — Sources (Git repo + Web base URL) + schedules + UI

```text
Implement source management per collection (still no ingestion yet).

Backend:
- Alembic table:
  - sources: id uuid, collection_id, type enum('github','web'), url, config jsonb, enabled bool,
            schedule_interval_minutes int, next_run_at timestamptz, last_run_at timestamptz,
            cursor text nullable, created_at
- API endpoints:
  - POST /api/collections/{id}/sources
  - GET /api/collections/{id}/sources
  - DELETE /api/sources/{id}
  - POST /api/sources/{id}/sync-now (sets next_run_at = now)
- Validation:
  - For web sources: store base_url; later enforce domain/path rules (add TODO).
  - For github sources: accept https GitHub repo URL; store branch default.

Frontend:
- Sources page per collection:
  - Add source form (GitHub repo URL OR Web base URL)
  - Configure schedule interval and enabled toggle
  - “Sync now” button (no-op for now)
  - Show last/next run

Acceptance:
- Full CRUD works.
- UI is polished and navigable.

Stop after completion.
```

---

## Prompt 7 — Prefect stack + worker + no-op incremental sync runs

```text
Add Prefect orchestration and a worker, but keep the actual sync as a no-op that records runs.

Infra:
- Extend docker compose with Prefect server components (server + UI + worker + redis + prefect-db if needed).
- Ensure worker can reach the same Postgres used by the app (or document clearly if separate).

Backend DB:
- Add table sync_runs:
  - id uuid, source_id, started_at, finished_at, status enum('running','success','failed'),
    stats jsonb, error text nullable

Worker (apps/worker):
- Create a Prefect flow `sync_due_sources` scheduled every N minutes:
  - queries sources where enabled=true and next_run_at <= now
  - for each source, creates a sync_run row (running), then marks success with stats { "noop": true }
  - sets source.last_run_at and next_run_at = now + interval
- Add basic concurrency control: don’t run two syncs for same source concurrently (DB advisory lock or “update … where” guard).

API:
- Add GET /api/runs?source_id=... returning recent runs.

Frontend:
- Runs view per source showing recent runs and status.

Acceptance:
- Prefect UI accessible.
- Runs appear in the app when schedules trigger.
- “Sync now” causes a run within the next worker polling cycle (or immediate if flow supports it).

Stop after completion.
```

---

## Prompt 8 — Incremental GitHub repo ingestion: Documents stored (no chunks/embeddings yet)

```text
Implement real incremental sync for GitHub repo sources, producing Document rows.

Backend DB:
- Add documents table:
  - id uuid, source_id, uri unique, title, content_markdown text, content_hash text,
    meta jsonb, last_seen_at, created_at, updated_at
- Add FK cascade rules so deleting a source deletes its docs.

Worker:
- For sources.type='github':
  - Use stored GitHub OAuth token of the *source creator* (or collection owner) to clone/pull.
  - Clone into a persistent volume path: /data/repos/{source_id}
  - Determine previous cursor (commit sha) from sources.cursor; fetch latest sha on configured branch.
  - If no cursor: full index (all eligible files).
  - Else: incremental:
    - compute changed + deleted files between old and new sha.
  - Eligible files: code + docs; use a conservative allowlist by extension and a max size limit.
  - Store each file as a Document with uri like:
    git://github.com/{owner}/{repo}/{path}?ref={branch}
  - For deleted files: hard delete documents with matching uri prefix.
  - Update sources.cursor to new sha.
  - Record counts in sync_runs.stats (files_changed, files_deleted, docs_upserted, etc).

API/UI:
- Add Documents count on sources list.
- Add a simple “Documents” debug endpoint/page:
  - GET /api/sources/{id}/documents (paginated) showing uri + updated_at.

Tests:
- Add tests for incremental logic using a local temporary git repo (no GitHub dependency).
  - Create repo, commit A, index, commit B with changes/deletes, index again, assert correct docs.

Acceptance:
- Running sync now for a github source populates documents and subsequent syncs are incremental.
- No chunks/embeddings yet.

Stop after completion.
```

---

## Prompt 9 — Incremental Web ingestion using spider-rs (HTML→Markdown) with strict scope + robots.txt

```text
Implement web crawling ingestion via a Rust spider-rs binary, producing Document rows incrementally.

Rust (rust/spider_md):
- Create a CLI binary that:
  - accepts: base_url, max_pages, output=jsonl, user_agent
  - respects robots.txt
  - stays strictly within:
    - same hostname
    - URL path prefix = path of base_url
    - no subdomains
  - fetches pages and converts HTML -> Markdown using spider-rs/html2md
  - emits JSON lines: { "url": "...", "title": "...", "markdown": "...", "content_hash": "..." }
- Provide a README for building/running the binary locally.

Worker:
- For sources.type='web':
  - invoke the spider_md binary (installed in worker image)
  - for each emitted page:
    - upsert Document by uri=url; store markdown and content_hash; set last_seen_at=now
  - incremental behavior:
    - if content_hash unchanged, do not update content
  - hard deletes:
    - after a successful run, delete docs for that source where last_seen_at < run_started_at (meaning they disappeared)

API/UI:
- Web sources should now populate documents similarly to github sources.

Tests:
- Unit test the URL scoping filter (domain + path prefix) in Python.
- Keep rust compilation in docker build (document how).

Acceptance:
- A web source sync indexes pages under the base path only.
- robots.txt is respected.
- Subsequent syncs are incremental on content_hash.

Stop after completion.
```

---

## Prompt 10 — Chunking with LangChain splitters + “intact code fences” + incremental chunk maintenance

```text
Add chunking and maintain chunks incrementally (still no embeddings).

DB:
- Add chunks table:
  - id uuid, document_id FK, chunk_index int, chunk_hash text, content text,
    tsv tsvector (generated or maintained), meta jsonb, created_at
  - unique(document_id, chunk_hash)
- Ensure deleting a document deletes chunks.

Core logic:
- Implement chunking service:
  - For Markdown documents:
    - Must NOT split inside fenced code blocks.
    - Use LangChain MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter for large sections.
  - For code files:
    - Use language-aware splitters where possible (LangChain).
- Incremental:
  - Compute chunk_hash for each chunk; upsert only new chunk_hashes.
  - Hard delete chunks that no longer exist for a document.

Worker:
- After document upsert for both github/web sources, run chunk maintenance for changed docs only.

Tests:
- Add tests proving:
  - fenced code blocks remain intact (a fence appears wholly within one chunk)
  - unchanged docs do not produce new chunks
  - changed docs replace chunks correctly (hard delete removed ones)

Acceptance:
- Sync produces chunks for documents.
- No embeddings yet; just chunk + FTS tsvector ready.

Stop after completion.
```

---

## Prompt 11 — Incremental embeddings (OpenAI first, optional Gemini) stored in pgvector

```text
Add embeddings and store vectors in Postgres+pgvector incrementally.

DB:
- Add embedding_models table:
  - id uuid, provider enum('openai','gemini'), model_name, dimension int, created_at
- Add columns to chunks:
  - embedding_model_id FK nullable
  - embedding vector (pgvector)
  - embedded_at timestamptz nullable
- Add appropriate pgvector indexes:
  - For each embedding model dimension, create a partial HNSW (or IVFFlat) index for chunks where embedding_model_id matches.

Config:
- Add env vars for OpenAI and Gemini keys (optional), and default embedding model selection per collection (store in collections.config jsonb).

Worker:
- After chunking, embed only chunks where embedding is NULL for the chosen embedding_model_id.
- Batch requests; implement basic retry/backoff; store stats in sync_runs.
- Provide a “FakeEmbedder” for tests that produces deterministic vectors.

Tests:
- Use FakeEmbedder in tests to avoid external calls.
- Verify only new chunks are embedded on incremental updates.

Acceptance:
- Sync embeds new chunks only.
- Database has vectors and indexes.

Stop after completion.
```

---

## Prompt 12 — Hybrid retrieval (FTS + vector) + RRF ranking + access control

```text
Implement hybrid retrieval within Postgres and expose a REST endpoint for debugging.

Backend:
- Implement search service that:
  - Takes query, optional collection_id, and user context.
  - Searches across:
    - all global collections
    - private collections where user is owner/member
  - Performs:
    - FTS top N over chunks.tsv
    - Vector top N using cosine distance against chunk.embedding
  - Combine results using Reciprocal Rank Fusion (RRF) into a final top K.
- Add POST /api/search:
  - input: { query, collection_id?: string, top_k?: number }
  - output: chunks with metadata (uri, title, snippet, score, method contributions)

Tests:
- With FakeEmbedder and seeded fixtures, assert:
  - access control: private chunks not returned for unauthorized user
  - RRF output deterministic for known data

Acceptance:
- UI (optional) can call /api/search to show results.
- Retrieval works even before context assembly.

Stop after completion.
```

---

## Prompt 13 — Context assembly endpoint (OpenAI/Anthropic/Gemini) producing Markdown context doc

```text
Implement “assembled context document” generation and expose it as a REST endpoint.

Backend:
- Add POST /api/context:
  - input: { query, collection_id?: string, max_chunks?: int, max_tokens?: int, provider?: 'openai'|'anthropic'|'gemini', model?: string }
  - behavior:
    1) hybrid retrieve chunks
    2) assemble a single Markdown document using the chosen LLM
- Prompt requirements:
  - Use ONLY the provided chunks (no hallucinated APIs).
  - Preserve code fences exactly.
  - Include a “Sources” section listing doc URIs and (for git) file paths.
- Add a stub “FakeLLM” for tests that returns deterministic Markdown.

Frontend:
- Add a “Query” page:
  - textbox + collection selector
  - renders returned Markdown nicely
  - shows citations/sources list

Tests:
- Ensure assembly calls retrieval and includes sources.
- Ensure code fences remain intact in output.

Acceptance:
- /api/context returns a good Markdown doc grounded in chunks.
- Works with FakeLLM tests and real provider via env vars.

Stop after completion.
```

---

## Prompt 14 — Wire MCP tool to real retrieval+assembly + Streamable HTTP streaming

```text
Upgrade the MCP tool so it returns real assembled context.

Backend:
- Update MCP tool context.get_markdown to:
  - require MCP Bearer token auth (already implemented)
  - accept input:
    { query: string, collection_id?: string, max_chunks?: number, max_tokens?: number }
  - call the same internal service as /api/context and return the Markdown context document.
- Implement streaming behavior:
  - If the MCP SDK supports streaming output chunks over SSE, stream progressively.
  - If not feasible, return the final document but keep the Streamable HTTP transport working.
- Ensure the MCP endpoint stays “retrieval only” (no create/list collections).

Docs:
- Update README with:
  - how to generate an MCP token in UI
  - MCP base URL (https://your-host/mcp)
  - example curl/JSON-RPC request
  - origin allowlist notes

Tests:
- Add an automated smoke test script (python) that:
  - calls the MCP endpoint with a token
  - requests context.get_markdown
  - asserts Markdown output contains “Sources” section

Acceptance:
- MCP endpoint returns real context grounded in indexed sources.
- No MCP management tools are exposed.

Stop after completion.
```

---

## Prompt 15 — SSH deploy key support for repo pulls + hardening pass

```text
Add SSH deploy key support and a hardening pass.

Backend:
- Add ability to attach an SSH deploy key to a github source:
  - store encrypted private key material in DB (or a secure file store, but document it)
  - store key fingerprint metadata
- UI: add “Deploy key” section for a repo source (upload/paste key).

Worker:
- When a source has deploy key, clone/pull using SSH:
  - write key to a temp file with 0600 perms
  - set GIT_SSH_COMMAND to use that key and strict known_hosts handling
  - ensure the key is never logged

Hardening:
- Enforce web crawling scope strictly (already) and add rate limits and max pages defaults.
- Add file size limits and binary detection for repo indexing.
- Add better run stats and error reporting in UI.
- Ensure ruff/pyright/pytest are clean.

Acceptance:
- Private repo can be indexed using deploy key.
- No secrets printed in logs.
- Everything still works end-to-end.

Stop after completion.
```

---

If you want, I can also add a **“release-ready” follow-on prompt set** (CI workflows, database backup/restore, migration safety checks, production reverse proxy/TLS, and an MCP client configuration example for common MCP hosts).

