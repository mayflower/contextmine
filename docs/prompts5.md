
## Prompt 1 — Repo audit + implementation plan (no functional changes)

```text
You are working in THIS repository (a Docker/Kubernetes-based Python Prefect “indexer” service). 

Goal: produce a concrete implementation plan for adding “polyglot SCIP indexing” (Python, PHP, Java, TypeScript/JavaScript) with a unified Python interface, integrated into the existing Prefect flows and container/K8s deployment.

Tasks:
1) Inspect the repository structure. Identify:
   - Prefect flows/deployments (entrypoints, flow files, task modules)
   - How repositories are checked out (git clone, volume mount, etc.)
   - Where output artifacts are written and uploaded (S3/GCS/DB/etc.)
   - Dockerfile(s) and K8s manifests/Helm chart (if any)
   - Current runtime user (root vs non-root), resource limits, caching strategy

2) Write a design doc at docs/scip_polyglot_indexing.md covering:
   - Target interface in Python (e.g., `index_repo(repo_path, config) -> List[IndexArtifact]`)
   - Language/project detection strategy (monorepo-safe)
   - Execution model: in-container subprocess vs per-language Kubernetes pod jobs (choose what best matches the repo; document tradeoffs)
   - Tooling to use (open-source CLIs): 
       - scip-python (npm)
       - scip-typescript (npm)
       - scip-java (coursier or docker image)
       - scip-php (php/composer or docker image)
   - Artifact naming scheme and upload paths (include language + project root hashing)
   - Config knobs (enabled languages, timeouts, dependency install mode, caches)
   - Security considerations (untrusted code execution) and mitigations consistent with existing system

3) Add a short “Implementation checklist” section with phases (Phase 1: core API + TS/JS, Phase 2: Python, Phase 3: Java, Phase 4: PHP, Phase 5: container+k8s hardening, Phase 6: tests+docs).

Constraints:
- Do not implement functionality yet except adding the design doc.
- Do not break any existing flows or images.
- Keep proposed changes aligned with existing patterns in this repo (logging, config, testing).

Output:
- Commit the design doc only.
- Summarize the key decisions and list the files/modules you expect to add/change in later prompts.
```

---

## Prompt 2 — Add unified Python API + subprocess runner + language detection (still no Prefect wiring)

```text
Goal: implement the core “polyglot SCIP indexing” Python module with:
- A unified interface
- Project detection (Python/TS/JS/Java/PHP)
- A safe subprocess runner (timeouts, logging, exit code handling)

Tasks:
1) Create a new internal Python package/module (follow repo conventions) e.g. `src/<your_pkg>/scip_polyglot/` (or similar).
2) Implement data models (dataclasses or pydantic—match repo style):
   - `Language` enum: python, typescript, javascript, java, php
   - `ProjectTarget`: {language, root_path, metadata}
   - `IndexArtifact`: {language, project_root, scip_path, logs_path, tool_name, tool_version, duration_s}
   - `IndexConfig`: {enabled_languages, timeout_s_by_language, install_deps_mode, max_concurrency, output_dir, project_name, project_version, env_overrides}

3) Implement `detect_projects(repo_root) -> List[ProjectTarget]`:
   - Python: detect pyproject.toml / setup.cfg / requirements.txt (at least one)
   - TS/JS: detect package.json; classify TS if tsconfig.json exists
   - Java: detect pom.xml / build.gradle / build.gradle.kts / build.sbt (at least one)
   - PHP: detect composer.json + composer.lock
   - Ignore common vendored dirs: node_modules, vendor, .git, dist, build, target, .venv, etc.
   - Ensure monorepo behavior: allow multiple project roots; de-duplicate nested matches (document rule, e.g., prefer nearest root).

4) Implement `run_cmd(...)` utility:
   - Uses subprocess with:
     - cwd
     - env override
     - stdout/stderr streamed into structured logs and also captured into a logfile
     - timeout enforcement
   - Returns `{exit_code, stdout_tail, stderr_tail, elapsed}` and raises a typed exception on failure if requested.

5) Implement a top-level orchestration function:
   - `index_repo(repo_root: Path, cfg: IndexConfig) -> List[IndexArtifact]`
   - For now, it should ONLY:
     - detect projects
     - create per-project output directories under cfg.output_dir
     - for each project, call a stub backend that raises NotImplementedError
   - Include a clear backend plugin interface: `BaseIndexerBackend.can_handle(ProjectTarget)` and `index(ProjectTarget, cfg) -> IndexArtifact`

6) Add unit tests:
   - detection tests using small fixture directory structures under tests/fixtures/ (no real dependencies)
   - run_cmd tests using trivial commands (python -c "print('hi')") in CI-safe manner

Constraints:
- No Prefect flow changes yet.
- No Docker/K8s changes yet.
- Keep new module isolated and importable.

Deliverables:
- New module + tests + minimal docs in README or docs/scip_polyglot_indexing.md updated with API skeleton.
```

---

## Prompt 3 — Wire into Prefect (feature-flagged) + artifact upload integration

```text
Goal: integrate the new polyglot SCIP indexing module into the existing Prefect indexer flows without breaking current behavior.

Tasks:
1) Identify the “main indexing flow” and how it is triggered (deployment/CLI/etc.).
2) Add a feature-flagged path:
   - If ENV `SCIP_POLYGLOT_ENABLED=true` (or repo-appropriate config), run the polyglot indexing path.
   - Otherwise, preserve existing indexing behavior exactly.

3) Add Prefect tasks:
   - `task_detect_projects(repo_root) -> List[ProjectTarget]`
   - `task_index_project(project_target, cfg) -> IndexArtifact` (will call backend later)
   - `task_upload_artifacts(artifacts)` reusing existing upload mechanism for other artifacts; if none exists, create a minimal one consistent with current storage approach.

4) Ensure outputs:
   - Write `.scip` files and logs under the run’s working directory
   - Publish Prefect artifacts/metadata (if repo already does)
   - Return a structured result object for downstream steps

5) Add tests:
   - If there is a flow test framework already, add a minimal flow test that runs detection + stub indexing for fixtures and validates artifact records.

Constraints:
- Keep default behavior unchanged unless SCIP_POLYGLOT_ENABLED is on.
- Do not require installing new OS packages yet.
- The index backends can remain stubs (NotImplementedError) but the flow wiring should be ready.

Deliverables:
- Prefect flow/task integration + tests + documentation update describing the new flag and outputs.
```

---

## Prompt 4 — Implement TypeScript/JavaScript backend (scip-typescript)

```text
Goal: implement TS/JS indexing via the open-source `scip-typescript` CLI and integrate it as a backend.

Background assumptions:
- `scip-typescript` is installed in the runtime image (later prompt will add it). For now, implement backend assuming it is on PATH and provide a clear error if missing.

Backend behavior:
- For TS projects: in project root containing tsconfig.json:
    - Install deps (configurable):
        - if lockfile present: prefer `npm ci` when package-lock.json exists
        - if yarn.lock: use `yarn install --frozen-lockfile` if yarn available
        - if pnpm-lock.yaml: use `pnpm install --frozen-lockfile` if pnpm available
      Support cfg.install_deps_mode: "auto" | "always" | "never"
    - Run: `scip-typescript index`
- For JS projects: run `scip-typescript index --infer-tsconfig`

Implementation tasks:
1) Add `TypescriptIndexerBackend`:
   - Detect TS vs JS from ProjectTarget metadata
   - Build command and env
   - Capture logs to per-project logs file
   - Ensure the produced `index.scip` is moved/copied into the per-project output dir and recorded in IndexArtifact

2) Add memory guardrails:
   - If cfg provides node_memory_mb, run with `node --max-old-space-size=... "$(which scip-typescript)" index ...`
   - Optionally support flag `--no-global-caches` when cfg sets `disable_ts_global_cache=true`

3) Add robust error messages:
   - If scip-typescript not found: explain how to install; do not crash the whole run if cfg says “best effort”, else fail.

4) Add unit/integration tests:
   - Add a small TS fixture (package.json + tsconfig.json + one .ts file)
   - Add a small JS fixture (package.json + one .js file)
   - Tests should not require actually running scip-typescript yet; instead:
       - mock run_cmd and assert the command, cwd, env, and output handling
   - Add an optional “manual integration test” marker if your repo has a pattern for that.

Deliverables:
- TS/JS backend implemented and registered in index_repo()
- Tests for command construction and output handling
- Update docs with TS/JS backend behavior and config knobs
```

---

## Prompt 5 — Implement Python backend (scip-python) aligned with existing Python indexing logic

```text
Goal: implement Python indexing via `scip-python` CLI, and integrate it into the unified interface and Prefect flow.

Important: This repository already indexes Python somehow. Reuse existing environment setup logic if present (venv creation, requirements install, poetry, etc.) rather than duplicating it.

Backend behavior:
- Run in the Python project root.
- Ensure dependencies are installed in the active environment OR create/activate a venv inside the job workspace (based on existing repo approach).
- Run: `scip-python index . --project-name <name>`
- If cfg.project_version is available, pass it if supported.
- Support `--target-only` when indexing a subdir (if project root differs from repo root).
- Provide optional support for `--environment=env.json` if the repo already can compute installed packages; otherwise skip.

Implementation tasks:
1) Add `PythonIndexerBackend`:
   - Detect a Python project root
   - Prepare environment using existing code paths (VERY IMPORTANT)
   - Run scip-python
   - Collect produced index.scip into output dir

2) Add OOM handling:
   - If cfg.node_memory_mb provided, set env `NODE_OPTIONS=--max-old-space-size=...` (or run via node wrapper if consistent)

3) Tests:
   - Mock run_cmd and existing env setup; verify command line and env
   - Fixture-based detection tests (already exist) should cover Python roots.

Deliverables:
- Python backend implemented and registered
- Docs updated for Python backend requirements and dependency behavior
- Tests added
```

---

## Prompt 6 — Implement Java backend (scip-java) with build-tool detection and safe defaults

```text
Goal: implement Java indexing using `scip-java index`, including build-tool inference and JVM flags needed for Java 17+.

Backend behavior:
- For each Java project root:
  - Run `scip-java index` (in that directory)
  - Prefer writing output to a deterministic location using `--output <path>` if available
  - Allow extra build args via cfg (appended after `--`)

Implementation tasks:
1) Add `JavaIndexerBackend`:
   - Identify build tool: Maven vs Gradle vs sbt based on files present
   - Build command:
       - default: `scip-java index --output <out>/index.scip`
       - allow `cfg.java_build_args` appended like: `scip-java index --output ... -- <args...>`
   - Ensure JVM exports for Java 17+ are set (use JAVA_TOOL_OPTIONS or tool-specific env) with:
       --add-exports jdk.compiler/com.sun.tools.javac.model=ALL-UNNAMED
       --add-exports jdk.compiler/com.sun.tools.javac.api=ALL-UNNAMED
       --add-exports jdk.compiler/com.sun.tools.javac.tree=ALL-UNNAMED
       --add-exports jdk.compiler/com.sun.tools.javac.util=ALL-UNNAMED
       --add-exports jdk.compiler/com.sun.tools.javac.code=ALL-UNNAMED
     Apply these by default unless cfg disables them.

2) Add caching hooks:
   - If repo has a job workspace cache mechanism, ensure ~/.m2 and ~/.gradle are persisted/mounted when running in K8s (actual mounts come later; add documentation and paths now).

3) Tests:
   - Command-construction tests with mocked run_cmd
   - Detection tests for Maven/Gradle/sbt roots

Deliverables:
- Java backend implemented + registered
- Tests + docs updated
```

---

## Prompt 7 — Implement PHP backend (scip-php) without mutating the target repository

```text
Goal: implement PHP indexing via `scip-php` while avoiding repository mutation like `composer require` that changes composer.json/lock.

Backend behavior:
- Preconditions:
  - project root contains composer.json and composer.lock
  - vendor/ exists OR deps can be installed from lockfile
- Dependency install:
  - If cfg.install_deps_mode allows:
      - run `composer install --no-interaction --prefer-dist --no-progress`
    (do NOT run composer require; do NOT modify composer.lock)
- Index:
  - run the scip-php CLI
  - Collect index.scip into output dir

Implementation tasks:
1) Add `PhpIndexerBackend`:
   - Check for composer.json + composer.lock
   - If vendor missing and install allowed, run composer install
   - Run scip-php (assume `scip-php` exists on PATH; later Docker prompt will ensure)
   - Move/copy index.scip into output dir

2) Add better error handling:
   - If composer missing or install fails, emit clear error
   - Allow “best effort” mode to skip PHP indexing rather than failing whole run

3) Tests:
   - Mock run_cmd; verify composer install is called only when vendor missing and mode permits
   - Verify scip-php invocation and artifact collection

Deliverables:
- PHP backend implemented + registered
- Tests + docs updated
```

---

## Prompt 8 — Update Docker image to include all required toolchains + pin versions

```text
Goal: update the repository’s Dockerfile(s) to support running all indexers in-container:
- Node (for scip-python, scip-typescript)
- Java (for scip-java)
- PHP + Composer (for scip-php)
- Install CLI tools (pinned versions) and make them available on PATH

Tasks:
1) Locate Dockerfile(s) used by the Prefect worker/deployment. Modify the correct one(s).
2) Install:
   - Node 20 (or Node 18/20 per repo constraints). Ensure npm works.
   - scip-typescript via npm global install, pinned (e.g., `@sourcegraph/scip-typescript@<version>`).
   - scip-python via npm global install, pinned.
   - Java runtime (prefer 17). Ensure `JAVA_HOME` and `java` exist.
   - scip-java:
       - Prefer using coursier bootstrap to build a standalone `scip-java` binary pinned to a known version.
       - Put binary at /usr/local/bin/scip-java
   - PHP (compatible with scip-php requirements) + composer
   - scip-php:
       - Prefer installing scip-php as a global composer package (so we can run `scip-php` without modifying the indexed repo)
       - Ensure composer global bin dir is on PATH

3) Security and runtime hygiene:
   - Run as non-root if current image supports it
   - Avoid leaving build caches in final layers where possible
   - Add ENV defaults for cache dirs and JVM/Node options (but keep them overridable)

4) Add a container “smoke test” script (CI-friendly):
   - prints versions:
       python --version
       node --version
       scip-typescript --version (or help)
       scip-python --version (or help)
       scip-java --help
       php --version
       composer --version
       scip-php --help
   - Add to CI if this repo has container build CI.

Constraints:
- Keep image size reasonable; use multi-stage builds if appropriate.
- Pin versions in a central place (env vars, build args, or a versions file).
- Do not break existing entrypoints.

Deliverables:
- Updated Dockerfile(s)
- Optional: CI step or make target to build + run smoke test
- Docs updated with tool versions and how to bump them
```

---

## Prompt 9 — Kubernetes/Helm updates: resources, cache volumes, and execution safety

```text
Goal: update Kubernetes manifests/Helm chart (or Prefect Kubernetes infrastructure blocks) so polyglot indexing is reliable and reasonably fast.

Tasks:
1) Identify how this repo deploys to Kubernetes:
   - raw manifests, Helm chart, Kustomize, Prefect Kubernetes worker config, etc.
2) Add configuration options for polyglot indexing:
   - env: SCIP_POLYGLOT_ENABLED
   - env: SCIP_LANGUAGES (comma-separated) or equivalent
   - env: SCIP_INSTALL_DEPS_MODE (auto/always/never)
   - env: SCIP_TIMEOUT_* per language
   - env: SCIP_NODE_MEMORY_MB for TS/JS/Python indexers
3) Add resource requests/limits guidance (and defaults if repo allows):
   - TS/JS indexing can need high memory; allow tuning
   - Java indexing can need CPU and memory + disk for build caches
4) Add cache volumes (PVC or emptyDir depending on existing conventions):
   - ~/.npm, ~/.cache/pip, ~/.m2, ~/.gradle, ~/.composer, ~/.cache/composer
   - Mount them into the worker pods so repeated indexing is faster
5) Security posture:
   - Ensure pod security context is consistent with current practices (non-root, drop capabilities, readOnlyRootFilesystem if feasible)
   - If indexing untrusted repos, document that build tools may execute code; recommend running in dedicated namespace/nodepool and limiting service account permissions.

Deliverables:
- Updated K8s/Helm/Prefect infra configs
- docs/runbook section: “Tuning polyglot SCIP indexing in Kubernetes”
```

---

## Prompt 10 — End-to-end validation + operator docs

```text
Goal: add an end-to-end validation path and operator documentation so this is maintainable.

Tasks:
1) Add an e2e “smoke flow” that:
   - checks out a small public fixture repo(s) OR uses local fixtures (prefer local to avoid network)
   - runs polyglot indexing across at least 2 languages
   - verifies output artifacts exist and are uploaded/stored in the expected place

2) Add a troubleshooting guide:
   - TS/JS OOM: how to set node memory / disable global caches
   - Python env issues: venv activation / dependency resolution
   - Java build failures: adding build args after `--`, JVM exports, dependency cache
   - PHP composer failures: vendor missing, composer install mode

3) Add “How to add a new language backend” developer guide:
   - implement backend interface
   - add detection rules
   - register backend
   - update Docker + k8s docs

Deliverables:
- e2e test or smoke command (Makefile target or CI job)
- Updated docs with operational runbook + troubleshooting + extension guide
```

---

### Notes on how these prompts map to the actual tools you will be orchestrating

These prompts intentionally standardize around invoking **existing open-source SCIP indexer CLIs** from your Prefect worker runtime:

* `scip-typescript index` for TS/JS
* `scip-python index` for Python
* `scip-java index` for Java/Gradle/Maven/sbt
* `scip-php` for PHP (with `composer install` as needed)

That keeps your Python code focused on orchestration, detection, reproducibility, and artifact handling—exactly what a Prefect/Kubernetes indexer service should do.


