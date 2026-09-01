# Dependency and Upgrade Inventory

This document records the dependency surfaces that must move together, the
checks that protect them, and the commands used to refresh the snapshot.

Snapshot date: **2026-09-01**  
Repository baseline: **5c08653044523ba321efbc0abc9cb2d1854d2912**

## Inventory Summary

| Surface | Manifest and lock | Current size | Runtime role |
| --- | --- | ---: | --- |
| Python workspace | `pyproject.toml`, three workspace `pyproject.toml` files, `uv.lock` | 246 locked packages | API, MCP, sync worker, analysis, search, graph, telemetry |
| Web application | `apps/web/package.json`, `apps/web/package-lock.json` | 12 runtime and 20 development declarations; 481 lock entries | React cockpit, diagrams, observability |
| Rust crawler | `rust/spider_md/Cargo.toml`, `rust/spider_md/Cargo.lock` | 9 direct declarations; 339 locked packages | Deterministic website crawling binary embedded in the worker image |
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
| Code intelligence | protobuf, tree-sitter-language-pack, Lizard, optional multilspy, and SCIP/LSP CLIs installed by the runtime images | Validate Python and TypeScript/JavaScript project detection, index creation, relation coverage, parsing, persisted symbols, and a semantic request through the ContextMine LSP adapter. |
| Knowledge graph and Twin | igraph, leidenalg, SQLAlchemy, pg4ai with Apache AGE and pgvector | Exercise deterministic knowledge-graph and Twin materialization against the production database image. |
| Web cockpit | React, React Flow, Cytoscape, ELK, Mermaid, Grafana Faro | Run lint, component tests, TypeScript build, and API image build because the API image embeds the web bundle. |
| Crawler | Rust `spider`, Tokio, serde, URL and hashing crates | Build the worker image and run Rust formatting, Clippy, and tests when the crawler lock or toolchain changes. |
| Telemetry | OpenTelemetry API/SDK/exporter/instrumentation family and Grafana Faro | Upgrade each telemetry family as a synchronized set; verify disabled and enabled startup modes. |
| Deployment | Python/Node/Rust base images, uv image, pg4ai, Prefect, CodeCharta, OTel collector, Helm | Floating tags are not reproducible. Move them to reviewed version or digest pins in dedicated changes, with container and Helm checks. |

### DeepAgents and agent sandbox

`deepagents` is not currently declared or imported by ContextMine; its research
graph uses LangGraph directly. The existing DeepAgents integration for the
Mayflower Kubernetes sandbox lives in
`mayflower/langchain-google` as `langchain-google-agent-sandbox`. Reuse that
backend when a ContextMine DeepAgents execution path needs sandboxed tools; do
not implement a second DeepAgents sandbox adapter here.

The repository analyzer is a deterministic Prefect job rather than a
DeepAgents tool backend. It therefore uses the sandbox control-plane Job/Result
API directly; `langchain-google-agent-sandbox` explicitly does not own that
control plane or durable product lifecycle.

## Upgrade Snapshot

The snapshot-date modernization is applied:

- Python direct dependencies and the `uv` lock are current. This includes
  Anthropic 1.2, OpenAI 3.6, Claude Agent SDK 0.2.149, LangGraph 1.2.11,
  Prefect 3.8.4, FastAPI 0.141.1, FastMCP 4, protobuf 7.36, pgvector 0.5,
  SQLAlchemy 2.0.52, and the OpenTelemetry 1.44/0.65 family.
- Web runtime and tooling dependencies are current except where the active
  runtime or peer contract requires a lower major. TypeScript stays on 5.9
  because openapi-typescript 7 requires TypeScript 5 and typescript-eslint 8
  requires a TypeScript version below 6.1; `@types/node` stays on 24 to match the
  Node 24 production runtime. Dependabot preserves these compatibility
  boundaries while continuing to propose updates within the supported majors.
  The current lock reports zero `npm audit` vulnerabilities.
- Shipped code intelligence uses the native TypeScript 7 LSP because TypeScript
  7 removed the JavaScript `tsserver` wrapped by `typescript-language-server`.
  The model-free fixture exercises this native server end to end.
- The Rust lock is refreshed to the latest Rust 1.98-compatible resolution,
  including `spider 2.53.6`. A direct feature activation works around
  `http-global-cache 0.2` not forwarding the middleware feature required by
  `http-cache-reqwest 1.0.0-alpha.8`. The crawler uses Spider's in-memory cache
  backend to avoid the vulnerable `memmap2 0.5` pulled by its disk-cache backend.
- Runtime inputs are pinned to versioned image tags and immutable digests.
  Python uses the supported 3.14 line, Node and `@types/node` use Node 24 LTS,
  and Rust uses 1.98 with Edition 2024. Coursier has an explicit SHA-256 check,
  Composer comes from a pinned image, and scip-php is checked out at an exact
  commit. These inputs still require manual refreshes because language package
  bots do not cover all of them reliably.

All direct Rust dependencies resolve to their current Rust-1.98-compatible
release line.

## Required Upgrade Gate

The normal CI checks remain required:

1. Ruff lint and formatting plus `ty` type checking.
2. The complete PostgreSQL-backed Python test suite.
3. Web ESLint, Vitest, Playwright end-to-end tests, and production build.
4. Rust 1.98 formatting, strict Clippy, and all crawler tests.
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
- starts the pinned TypeScript language server through ContextMine's real
  `LspManager` and requires hover, client caching, and cross-file definition
  resolution without downloading a server at runtime;
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

## Automated Update Policy

Dependabot checks the uv workspace, web application, Rust crawler and
toolchain, GitHub Actions, Dockerfiles, and Compose manifests every Monday.
Each ecosystem is limited to three open version-update pull requests. A
seven-day cooldown reduces exposure to freshly published releases; Dependabot
security updates are not delayed by that cooldown. Automated security pull
requests additionally require the repository-level Dependabot security update
setting; the configuration file does not enable that setting.

Patch and minor releases are grouped only where the repository already has a
shared compatibility boundary. Major releases remain individual proposals.
For uv and Cargo, direct dependencies remain eligible for every update type,
while indirect dependencies are limited to patch and minor proposals. This
keeps transitive lockfiles maintained without opening unbounded transitive
major-upgrade pull requests.
No Dependabot pull request is merged automatically: the normal branch
protection and change-classified CI gates still decide whether a proposal is
mergeable.

Docker Compose version proposals for `prefecthq/prefect` are disabled. Docker
ignore rules cannot express a stable-only tag policy, and Prefect pre-releases
are not production upgrade candidates. A stable uv Prefect proposal remains
the signal for a manually synchronized client, Compose, and Helm update.

The Prefect Python client and Prefect server images are grouped within their
respective ecosystems, but Dependabot cannot combine a selective
multi-ecosystem Prefect group with normal updates for the same uv and Compose
directories without overlapping update entries. Any Prefect proposal must
therefore be completed in its branch by synchronizing the client, Compose, and
Helm references before merge. The model-free system gate verifies the real
client/server version match. Image references in Helm values are synchronized
manually because Dependabot does not update arbitrary Helm values.

The Dependabot cooldown controls when proposals are opened; it does not alter
uv's resolver cutoff. A matching `[tool.uv] exclude-newer` setting remains a
separate controlled workspace refresh, so introducing this policy does not
re-resolve or downgrade the current lockfile.
