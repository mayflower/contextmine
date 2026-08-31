# Dependency and Upgrade Inventory

This document is the baseline for dependency maintenance. It records the
dependency surfaces that must move together, the checks that protect them, and
the commands used to refresh the snapshot. It is not a request to upgrade every
package in one change.

Snapshot date: **2026-08-31**  
Repository baseline: **95caf9654eae39318a00d8c30eac0e6a3b2db242**

## Inventory Summary

| Surface | Manifest and lock | Current size | Runtime role |
| --- | --- | ---: | --- |
| Python workspace | `pyproject.toml`, three workspace `pyproject.toml` files, `uv.lock` | 230 locked packages; 29 core, 8 API, 11 worker, and 9 development declarations | API, MCP, sync worker, analysis, search, graph, telemetry |
| Web application | `apps/web/package.json`, `apps/web/package-lock.json` | 9 runtime and 19 development declarations; 456 lock entries | React cockpit, diagrams, observability |
| Rust crawler | `rust/spider_md/Cargo.toml`, `rust/spider_md/Cargo.lock` | 8 direct declarations; 357 locked packages | Deterministic website crawling binary embedded in the worker image |
| Runtime images | Dockerfiles, Compose files, Helm values | API, worker, web, PostgreSQL/pg4ai, Prefect, CodeCharta, OpenTelemetry | Build and deployment compatibility |
| CI actions and tools | `.github/workflows/*.yml`, `.pre-commit-config.yaml` | SHA-pinned GitHub Actions plus scanner/tool versions | Verification and delivery |

The Python requirements use lower bounds while `uv.lock` supplies the exact
resolved environment. This makes the lockfile essential: a fresh unlocked
resolution may cross major-version boundaries even when a manifest was not
edited.

## Component Ownership

| Component | Main dependencies | Compatibility boundary |
| --- | --- | --- |
| Shared persistence and configuration | Pydantic, Pydantic Settings, SQLAlchemy, asyncpg, Alembic, pgvector, cryptography | API and worker must use the same core package and database schema. Database-driver and migration changes need a fresh-database migration plus real PostgreSQL tests. |
| API and MCP | FastAPI, FastMCP, Uvicorn, SlowAPI, Prometheus instrumentation | Upgrade FastAPI/Starlette/Pydantic together only after checking route, lifespan, middleware, MCP discovery, and container startup. |
| Sync orchestration | Prefect, GitPython, Tenacity | The Prefect Python client and Prefect server image are one upgrade unit. Test a real submitted flow, not imports alone. |
| Model integrations | OpenAI, Anthropic, Google Gen AI, LangChain Core, LangChain provider packages, LangGraph | Provider SDK majors and the LangChain/LangGraph family are high-risk. Model-free indexing must remain green before any provider-specific tests are considered. |
| Code intelligence | protobuf, tree-sitter-language-pack, Lizard, SCIP CLIs installed by the worker image | Validate Python and TypeScript/JavaScript project detection, index creation, relation coverage, parsing, and persisted symbols. |
| Knowledge graph and Twin | igraph, leidenalg, SQLAlchemy, pg4ai with Apache AGE and pgvector | Exercise deterministic knowledge-graph and Twin materialization against the production database image. |
| Web cockpit | React, React Flow, Cytoscape, ELK, Mermaid, Grafana Faro | Run lint, component tests, TypeScript build, and API image build because the API image embeds the web bundle. |
| Crawler | Rust `spider`, Tokio, serde, URL and hashing crates | Build the worker image and run Rust formatting, Clippy, and tests when the crawler lock or toolchain changes. |
| Telemetry | OpenTelemetry API/SDK/exporter/instrumentation family and Grafana Faro | Upgrade each telemetry family as a synchronized set; verify disabled and enabled startup modes. |
| Deployment | Python/Node/Rust base images, uv image, pg4ai, Prefect, CodeCharta, OTel collector, Helm | Floating tags are not reproducible. Move them to reviewed version or digest pins in dedicated changes, with container and Helm checks. |

### DeepAgents

`deepagents` is neither declared nor imported. ContextMine currently implements
its research agent with its own LangChain/LangGraph-based code. Adding
DeepAgents would therefore be an architecture and behavior change, not a
dependency update, and should be evaluated separately.

## Upgrade Snapshot

A read-only refresh on the snapshot date found a broad pending set:

- A full Python resolution would change roughly one hundred transitive entries
  and increase the lock from 230 to 232 packages. Direct major boundaries
  include Anthropic `0.112.0 -> 1.2.0`, OpenAI `2.44.0 -> 3.6.0`,
  cryptography `49.0.0 -> 50.0.1`, protobuf `6.33.6 -> 7.36.0`, and pgvector
  `0.4.2 -> 0.5.0`. LangChain provider packages, LangChain Core, LangGraph,
  Prefect, FastAPI, FastMCP, Uvicorn, tree-sitter-language-pack, and the
  OpenTelemetry family also have newer compatible resolutions.
- The web runtime has compatible updates for the Grafana Faro family,
  Cytoscape, React, and React DOM. ELK moves from `0.11.1` to `0.12.0`; because
  it is pre-1.0, treat that as a potentially breaking change. Separate tooling
  majors are available for ESLint 10, TypeScript 7, jsdom 30, and Testing
  Library jest-dom 7.
- The completed non-major web security refresh moved Mermaid above its fixed
  boundary and refreshed affected transitive packages. `npm audit` now reports
  zero vulnerabilities in the current lock.
- `cargo update --dry-run` proposes 121 compatible lock changes, including
  `spider 2.52.4 -> 2.53.6`. This is a lock refresh, not evidence that a future
  direct major is safe.
- Compose and the default Helm values still contain floating `latest` tags for
  operational images. Docker base images and the copied uv binary are also tag
  based. The worker additionally fetches the Coursier launcher from `master`,
  the Composer installer dynamically, and scip-php from `dev-main`; the other
  SCIP versions are build arguments with defaults. These are part of the
  dependency inventory even though language package bots do not cover them
  reliably.

Do not combine these sets into one upgrade. The counts are a triage signal, not
a target change list.

## Required Upgrade Gate

The normal CI checks remain required:

1. Ruff lint and formatting plus `ty` type checking.
2. The complete PostgreSQL-backed Python test suite.
3. Web ESLint, Vitest, and production build.
4. Rust 1.88 formatting, strict Clippy, and all crawler tests.
5. Security workflows and the relevant container/Helm build checks. Helm
   publication waits for the complete image-build dependency chain.

Dependency changes additionally run:

```bash
./scripts/smoke/model-free-system.sh
```

The system gate deliberately takes longer than a unit smoke test. It:

- creates a fresh pg4ai database and applies every Alembic migration;
- starts a real Prefect server and builds the shipped API, web, and worker
  images;
- uses pinned Chromium/Playwright to verify the live login surface, static
  assets, and the web-to-API health proxy;
- fetches `fastapi/full-stack-fastapi-template` at the immutable commit
  `486f054cc8d1aead59ec96cc0a16933d06c10e0d`;
- refuses to install the fixture's dependencies;
- runs the real GitHub-source sync flow with `MODEL_CALLS_ENABLED=false` and
  fails if an embedding or LLM provider is initialized;
- requires strict Python and TypeScript/JavaScript SCIP project and relation
  coverage plus strict structural metrics;
- verifies persisted documents, chunks, symbols, knowledge nodes, Twin nodes,
  known fixture paths, and full-text-only retrieval;
- compares deterministic extraction and persistence counts to an explicit
  baseline (210 documents, 691 chunks, 553 symbols, 4 SCIP projects, 5,998
  SCIP relations, 4,665 knowledge nodes, and 4,621 active Twin nodes); a
  deliberate parser behavior change must update these expectations visibly;
- performs a second sync and requires it to be a no-op with unchanged persisted
  counts; and
- uses no host ports and removes its temporary repository, containers, network,
  and volumes after the run.

Joern is advisory in this gate, so this test does not qualify Joern/CPG changes.
Authenticated browser workflows, web crawling, and model-provider behavior also
need their own targeted checks when those surfaces change.

## Refresh Commands

These commands inspect available updates without modifying a lockfile:

```bash
uv lock --upgrade --dry-run

cd apps/web
npm outdated --long
cd ../..

cd rust/spider_md
cargo update --dry-run
cd ../..
```

Also review image tags and action references:

```bash
rg -n 'uses:|image:|FROM ' .github apps scripts docker-compose.yml deploy/helm
```

Record the date and baseline commit whenever the snapshot is refreshed.

## Upgrade Slicing

Use small, independently revertible changes in this order:

1. Establish and keep this baseline gate green.
2. Apply patch/minor lock refreshes within one ecosystem, starting with
   low-coupling tooling and libraries.
3. Upgrade synchronized families in separate changes: OpenTelemetry;
   LangChain/LangGraph provider stack; Prefect client/server; database stack;
   FastAPI/FastMCP; and web runtime/tooling.
4. Handle each major version in its own change with release-note review and a
   targeted regression test for the affected boundary.
5. Pin and update runtime images deliberately, including database capability
   checks and Helm/container startup checks.

Dependabot or Renovate can create the routine proposals after these groups and
required checks are defined. Automation is an input to this process; it does
not replace compatibility grouping, release-note review, or the system gate.
