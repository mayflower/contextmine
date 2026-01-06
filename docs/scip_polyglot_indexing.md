# SCIP Polyglot Indexing

This document describes the polyglot SCIP indexing system for ContextMine, which provides semantic code intelligence for Python, TypeScript/JavaScript, Java, and PHP projects.

## Overview

SCIP (Sourcegraph Code Intelligence Protocol) is a language-agnostic format for code intelligence data. This system runs language-specific SCIP indexers and parses their output into a unified `Snapshot` model for integration with ContextMine's knowledge graph.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Prefect Worker                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ detect_      │───▶│ index_repo   │───▶│ build_       │      │
│  │ projects()   │    │ ()           │    │ snapshot()   │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ ProjectTarget│    │ IndexArtifact│    │ Snapshot     │      │
│  │ (lang, path) │    │ (.scip file) │    │ (symbols,    │      │
│  └──────────────┘    └──────────────┘    │  relations)  │      │
│                                          └──────────────┘      │
├─────────────────────────────────────────────────────────────────┤
│                    Language Backends                            │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐   │
│  │ scip-python│ │scip-       │ │ scip-java  │ │ scip-php   │   │
│  │            │ │typescript  │ │            │ │            │   │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Python API

### Core Functions

```python
from contextmine_core.semantic_snapshot import (
    build_snapshot,      # Parse .scip file → Snapshot
    detect_projects,     # Detect language projects in a repo
    index_repo,          # Run SCIP indexers on detected projects
)

# Detect projects in a repository
projects = detect_projects("/path/to/repo")
# Returns: [ProjectTarget(language=Language.PYTHON, root_path=...), ...]

# Run indexers and get artifacts
artifacts = index_repo("/path/to/repo", IndexConfig())
# Returns: [IndexArtifact(scip_path=..., success=True), ...]

# Parse SCIP file into Snapshot
snapshot = build_snapshot("/path/to/index.scip")
# Returns: Snapshot(symbols=[...], relations=[...], ...)
```

### Data Models

```python
class Language(Enum):
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    PHP = "php"

@dataclass
class ProjectTarget:
    language: Language
    root_path: Path
    metadata: dict[str, Any]  # e.g., {"build_tool": "maven"}

@dataclass
class IndexArtifact:
    language: Language
    project_root: Path
    scip_path: Path
    logs_path: Path | None
    tool_name: str
    tool_version: str
    duration_s: float
    success: bool = True
    error_message: str | None = None

@dataclass
class IndexConfig:
    enabled_languages: set[Language]
    timeout_s_by_language: dict[Language, int]
    install_deps_mode: InstallDepsMode  # auto | always | never
    max_concurrency: int = 2
    output_dir: Path | None = None
    project_name: str = "project"
    project_version: str = "0.0.0"
    env_overrides: dict[str, str] = field(default_factory=dict)
    node_memory_mb: int | None = None
    java_build_args: list[str] = field(default_factory=list)
    best_effort: bool = True  # Continue on individual failures
```

## Project Detection

Projects are detected by marker files:

| Language | Marker Files | Metadata |
|----------|--------------|----------|
| Python | `pyproject.toml`, `setup.cfg`, `requirements.txt` | `has_pyproject`, `has_requirements` |
| TypeScript | `package.json` + `tsconfig.json` | `has_tsconfig`, `package_manager` |
| JavaScript | `package.json` (no tsconfig) | `package_manager` |
| Java | `pom.xml`, `build.gradle`, `build.gradle.kts` | `build_tool` (maven/gradle) |
| PHP | `composer.json` + `composer.lock` | `has_composer_lock` |

### Ignored Directories

The following directories are ignored during detection:
- `node_modules`, `vendor`, `.git`, `dist`, `build`, `target`, `.venv`, `__pycache__`, `.tox`, `.nox`, `.eggs`

### Monorepo Support

Multiple projects can be detected in a single repository. Nested projects are supported (e.g., `backend/` with Python and `frontend/` with TypeScript).

## Language Backends

### TypeScript/JavaScript (scip-typescript)

**Tool**: `@sourcegraph/scip-typescript` (npm)

**Commands**:
- TypeScript: `scip-typescript index`
- JavaScript: `scip-typescript index --infer-tsconfig`

**Dependency Installation**:
- npm: `npm ci` (if package-lock.json) or `npm install`
- yarn: `yarn install --frozen-lockfile`
- pnpm: `pnpm install --frozen-lockfile`
- bun: `bun install --frozen-lockfile`

**Memory Management**:
- Set `NODE_OPTIONS=--max-old-space-size=<mb>` via `cfg.node_memory_mb`

### Python (scip-python)

**Tool**: `@anthropic/scip-python` (npm)

**Command**: `scip-python index . --project-name <name> --project-version <version>`

**Dependency Installation**:
- If `pyproject.toml`: `uv sync` or `pip install -e .`
- If `requirements.txt`: `pip install -r requirements.txt`

**Memory Management**:
- Set `NODE_OPTIONS=--max-old-space-size=<mb>` via `cfg.node_memory_mb`

### Java (scip-java)

**Tool**: `scip-java` (Coursier)

**Command**: `scip-java index --output <path>`

**Build Tool Detection**:
- Maven: `pom.xml`
- Gradle: `build.gradle` or `build.gradle.kts`

**JVM Exports (Java 17+)**:
```
--add-exports jdk.compiler/com.sun.tools.javac.model=ALL-UNNAMED
--add-exports jdk.compiler/com.sun.tools.javac.api=ALL-UNNAMED
--add-exports jdk.compiler/com.sun.tools.javac.tree=ALL-UNNAMED
--add-exports jdk.compiler/com.sun.tools.javac.util=ALL-UNNAMED
--add-exports jdk.compiler/com.sun.tools.javac.code=ALL-UNNAMED
```

**Custom Build Args**: Pass via `cfg.java_build_args`

### PHP (scip-php)

**Tool**: `scip-php` (Composer global)

**Command**: `scip-php`

**Requirements**: Both `composer.json` and `composer.lock` must exist.

**Dependency Installation**:
```bash
composer install --no-interaction --prefer-dist --no-progress
```

**Important**: Never runs `composer require` - only `composer install` to avoid modifying the repository.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SCIP_LANGUAGES` | `python,typescript,javascript,java,php` | Comma-separated enabled languages |
| `SCIP_INSTALL_DEPS_MODE` | `auto` | Dependency install mode: auto/always/never |
| `SCIP_TIMEOUT_PYTHON` | `300` | Python indexing timeout (seconds) |
| `SCIP_TIMEOUT_TYPESCRIPT` | `600` | TS/JS indexing timeout (seconds) |
| `SCIP_TIMEOUT_JAVA` | `900` | Java indexing timeout (seconds) |
| `SCIP_TIMEOUT_PHP` | `300` | PHP indexing timeout (seconds) |
| `SCIP_NODE_MEMORY_MB` | `4096` | Node.js memory limit for TS/JS/Python |

### Best Effort Mode

When `cfg.best_effort=True` (default), individual project failures don't abort the entire run. Failed projects are logged and skipped.

## Docker Image Requirements

The worker image must include:

```dockerfile
# Node.js 20 (for scip-python, scip-typescript)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs

# SCIP indexers (npm)
RUN npm install -g @sourcegraph/scip-typescript@0.3.0 \
                   @anthropic/scip-python@0.5.0

# Java 21 (for scip-java)
RUN apt-get install -y openjdk-21-jdk-headless

# scip-java (Coursier)
RUN curl -fL https://github.com/coursier/launchers/raw/master/cs-x86_64-pc-linux.gz | \
    gzip -d > /usr/local/bin/cs && chmod +x /usr/local/bin/cs && \
    cs install scip-java --install-dir /usr/local/bin

# PHP + Composer (for scip-php)
RUN apt-get install -y php-cli composer
RUN composer global require nicosantangelo/scip-php
ENV PATH="${PATH}:/root/.composer/vendor/bin"
```

## Kubernetes Configuration

### Resource Recommendations

| Language | CPU Request | Memory Request | Memory Limit |
|----------|-------------|----------------|--------------|
| TypeScript/JS | 500m | 2Gi | 8Gi |
| Python | 500m | 2Gi | 4Gi |
| Java | 1000m | 4Gi | 8Gi |
| PHP | 250m | 1Gi | 2Gi |

### Cache Volumes

Mount these paths for faster repeated indexing:

```yaml
volumes:
  - name: npm-cache
    emptyDir: {}
  - name: maven-cache
    emptyDir: {}
  - name: gradle-cache
    emptyDir: {}
  - name: composer-cache
    emptyDir: {}

volumeMounts:
  - name: npm-cache
    mountPath: /root/.npm
  - name: maven-cache
    mountPath: /root/.m2
  - name: gradle-cache
    mountPath: /root/.gradle
  - name: composer-cache
    mountPath: /root/.composer/cache
```

## Troubleshooting

### TypeScript/JavaScript OOM

**Symptoms**: Process killed, "JavaScript heap out of memory"

**Solutions**:
1. Increase `SCIP_NODE_MEMORY_MB` (default 4096)
2. Set `--max-old-space-size` higher in NODE_OPTIONS
3. For very large projects, consider indexing subprojects separately

### Python Environment Issues

**Symptoms**: Import errors, missing dependencies

**Solutions**:
1. Ensure virtual environment is activated
2. Check `SCIP_INSTALL_DEPS_MODE` is `auto` or `always`
3. Verify `pyproject.toml` or `requirements.txt` is complete

### Java Build Failures

**Symptoms**: Compilation errors, missing dependencies

**Solutions**:
1. Add custom build args via `SCIP_JAVA_BUILD_ARGS`
2. Ensure JVM exports are set (automatic for Java 17+)
3. Check Maven/Gradle cache is mounted

### PHP Composer Failures

**Symptoms**: "vendor/ not found", dependency errors

**Solutions**:
1. Ensure both `composer.json` and `composer.lock` exist
2. Set `SCIP_INSTALL_DEPS_MODE=always` to force install
3. Check composer is available and cache is mounted

## Adding a New Language Backend

1. Create `indexers/<language>.py` implementing `BaseIndexerBackend`:

```python
class NewLanguageBackend(BaseIndexerBackend):
    TOOL_NAME = "scip-newlang"

    def can_handle(self, target: ProjectTarget) -> bool:
        return target.language == Language.NEWLANG

    def index(self, target: ProjectTarget, cfg: IndexConfig) -> IndexArtifact:
        # Implementation
        pass
```

2. Add detection rules in `indexers/detection.py`

3. Register backend in `indexers/__init__.py`:
```python
BACKENDS = [..., NewLanguageBackend()]
```

4. Add `Language.NEWLANG` to `models.py`

5. Update Docker image to install the indexer tool

6. Add tests and documentation

## File Structure

```
packages/core/contextmine_core/semantic_snapshot/
├── __init__.py              # Public exports
├── models.py                # Data models (Language, ProjectTarget, etc.)
├── scip.py                  # SCIP protobuf parser
├── indexers/
│   ├── __init__.py          # index_repo() orchestrator
│   ├── base.py              # BaseIndexerBackend ABC
│   ├── detection.py         # detect_projects()
│   ├── runner.py            # run_cmd() subprocess utility
│   ├── typescript.py        # TypeScript/JS backend
│   ├── python.py            # Python backend
│   ├── java.py              # Java backend
│   └── php.py               # PHP backend
└── proto/
    ├── __init__.py
    ├── scip.proto           # SCIP protocol definition
    ├── scip_pb2.py          # Generated protobuf code
    └── scip_pb2.pyi         # Type stubs
```
