# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-05

### Added

- Enable agent-fleet for changelog-generator

### Changed

- Replace code review agents with sonarqube-issue-fixer
- Add comprehensive agent fleet configuration
- Move agent fleet docs to backstage repo
- Add comprehensive agent fleet setup guide
- Add agent fleet integration specification
- Add agent fleet configuration
- Fix TechDocs configuration for Backstage
- Fix TechDocs annotation to use explicit URL
- Fix failing tests in CI
- Update SonarQube action to v6 for security patch
- Add SonarQube configuration and GitHub Actions workflow
- Add OTEL tracing to Prefect flows and tasks
- Disable OTEL log export (Tempo only supports traces)
- Add SonarQube configuration and GitHub Actions workflow
- Bump Helm chart version to 0.1.7
- Add Grafana Faro frontend observability
- Add Prometheus and Grafana monitoring annotations to Backstage catalog
- Fix ruff formatting
- Fix ruff lint errors: Callable import and unused MagicMock
- Add optional OpenTelemetry support for distributed tracing
- Enable Prometheus instrumentation unconditionally (remove env var check)
- Add response_model=None for SPA catch-all route with Union response types
- Fix return type annotation for serve_spa to include PlainTextResponse
- Handle /metrics in SPA catch-all using prometheus_client directly
- Fix /metrics route priority over SPA catch-all
- Add Prometheus metrics endpoint to API
- Add Backstage catalog-info.yaml
- Add TechDocs support with mkdocs
- Remove prompt files from docs folder
- Add monitoring support to Helm chart (v0.1.6)
- Optimize business rule extraction: one LLM call per file
- Bump Helm chart version to 0.1.5
- Add podLabels support to Helm chart deployments
- Fix LLM provider API key resolution from environment
- Fix worker volume permissions with fsGroup
- Refactor Collections UI to inline expandable rows
- Bump chart to 0.1.3 to force pod restart with new secret
- Bump chart to 0.1.2, add chart-version annotation for restart
- Fix ruff format
- Fix SQL injection warning in graphrag
- Skip pgvector creation if extension already exists
- Fix Prefect deployment to use secret envFrom
- Add existingSecret support to Helm chart
- Remove arc42 architecture documentation feature
- Remove low-value tests, consolidate with parametrize
- Remove KG fallbacks - require LLM and embedder
- Integrate GraphRAG into research agent
- Update documentation to reflect current GraphRAG API
- Remove dead code and unused functions
- Add SCIP polyglot indexing for Python, TypeScript, Java, PHP
- Implement LLM-based extractors and arc42 generator
- Add Knowledge Graph subsystem with research-focused MCP tools
- UX improvements: larger logo, query in left column
- Enhance dashboard welcome section with feature highlights
- Move Query functionality to Dashboard, remove Query tab
- Fix CI and Semgrep issues
- Add security workflow with Semgrep and Bandit
- Add vulture and bandit to pre-commit hooks
- Security hardening: auth, rate limiting, and input validation
- Use larger white-background logo on dashboard
- Use pre-indexed data for MCP definition/references tools
- Replace live LSP tools with pre-indexed equivalents
- Add missing research agent tools from documentation
- Remove deprecated action registry pattern
- Update research agent to use pre-indexed symbol data
- Add LSP and Tree-sitter tools to research agent
- Remove deprecated MCP token authentication system
- Remove deprecated token-related CSS and update README for OAuth
- Remove unused token-related code (OAuth is used instead)
- Improve dashboard UX and update README
- Use dark logo variant in header for better contrast
- Improve frontend UX with mobile nav, footer, and gradient
- Add logo images to frontend login and header
- Increase max page size to 5MB for modern SPAs
- Use trafilatura for clean HTML content extraction
- Fix fast_html2md import - uses html2md module name
- Improve HTML to Markdown conversion with Readability extraction
- Filter out sync_due_sources polling runs from UI
- Separate start_url from base_url for web crawling
- Fix duplicate sync runs with row-level locking
- Fix CI: Make GitHub OAuth optional for testing
- Add deep research mode to Query tab
- Add Kubernetes/Helm deployment and improve infrastructure
- Fix duplicate symbol names causing sync failures
- Update README with user-focused setup and usage guide
- Add research agent, symbol indexing, and tree-sitter support
- Fix spider-rs API changes in spider_md
- Fix flaky search test with proper mocking
- Add MIT license and rewrite README for end users
- Add pre-commit hooks and GitHub Actions CI/CD
- Replace OpenContext references with ContextMine
- Fix MCP endpoint routing and enable stateless HTTP mode
- Add Context7-inspired MCP tools and features
- Upgrade deprecated packages and fix tests
- Improve Query page layout to use full width
- Fix sync run stats display on dashboard

### Fixed

- remove unused import to pass lint check
- prevent worker OOM from Prefect subprocess zombies

## [0.1.0] - 2026-02-05

### Added

- Enable agent-fleet for changelog-generator

### Changed

- Replace code review agents with sonarqube-issue-fixer
- Add comprehensive agent fleet configuration
- Move agent fleet docs to backstage repo
- Add comprehensive agent fleet setup guide
- Add agent fleet integration specification
- Add agent fleet configuration
- Fix TechDocs configuration for Backstage
- Fix TechDocs annotation to use explicit URL
- Fix failing tests in CI
- Update SonarQube action to v6 for security patch
- Add SonarQube configuration and GitHub Actions workflow
- Add OTEL tracing to Prefect flows and tasks
- Disable OTEL log export (Tempo only supports traces)
- Add SonarQube configuration and GitHub Actions workflow
- Bump Helm chart version to 0.1.7
- Add Grafana Faro frontend observability
- Add Prometheus and Grafana monitoring annotations to Backstage catalog
- Fix ruff formatting
- Fix ruff lint errors: Callable import and unused MagicMock
- Add optional OpenTelemetry support for distributed tracing
- Enable Prometheus instrumentation unconditionally (remove env var check)
- Add response_model=None for SPA catch-all route with Union response types
- Fix return type annotation for serve_spa to include PlainTextResponse
- Handle /metrics in SPA catch-all using prometheus_client directly
- Fix /metrics route priority over SPA catch-all
- Add Prometheus metrics endpoint to API
- Add Backstage catalog-info.yaml
- Add TechDocs support with mkdocs
- Remove prompt files from docs folder
- Add monitoring support to Helm chart (v0.1.6)
- Optimize business rule extraction: one LLM call per file
- Bump Helm chart version to 0.1.5
- Add podLabels support to Helm chart deployments
- Fix LLM provider API key resolution from environment
- Fix worker volume permissions with fsGroup
- Refactor Collections UI to inline expandable rows
- Bump chart to 0.1.3 to force pod restart with new secret
- Bump chart to 0.1.2, add chart-version annotation for restart
- Fix ruff format
- Fix SQL injection warning in graphrag
- Skip pgvector creation if extension already exists
- Fix Prefect deployment to use secret envFrom
- Add existingSecret support to Helm chart
- Remove arc42 architecture documentation feature
- Remove low-value tests, consolidate with parametrize
- Remove KG fallbacks - require LLM and embedder
- Integrate GraphRAG into research agent
- Update documentation to reflect current GraphRAG API
- Remove dead code and unused functions
- Add SCIP polyglot indexing for Python, TypeScript, Java, PHP
- Implement LLM-based extractors and arc42 generator
- Add Knowledge Graph subsystem with research-focused MCP tools
- UX improvements: larger logo, query in left column
- Enhance dashboard welcome section with feature highlights
- Move Query functionality to Dashboard, remove Query tab
- Fix CI and Semgrep issues
- Add security workflow with Semgrep and Bandit
- Add vulture and bandit to pre-commit hooks
- Security hardening: auth, rate limiting, and input validation
- Use larger white-background logo on dashboard
- Use pre-indexed data for MCP definition/references tools
- Replace live LSP tools with pre-indexed equivalents
- Add missing research agent tools from documentation
- Remove deprecated action registry pattern
- Update research agent to use pre-indexed symbol data
- Add LSP and Tree-sitter tools to research agent
- Remove deprecated MCP token authentication system
- Remove deprecated token-related CSS and update README for OAuth
- Remove unused token-related code (OAuth is used instead)
- Improve dashboard UX and update README
- Use dark logo variant in header for better contrast
- Improve frontend UX with mobile nav, footer, and gradient
- Add logo images to frontend login and header
- Increase max page size to 5MB for modern SPAs
- Use trafilatura for clean HTML content extraction
- Fix fast_html2md import - uses html2md module name
- Improve HTML to Markdown conversion with Readability extraction
- Filter out sync_due_sources polling runs from UI
- Separate start_url from base_url for web crawling
- Fix duplicate sync runs with row-level locking
- Fix CI: Make GitHub OAuth optional for testing
- Add deep research mode to Query tab
- Add Kubernetes/Helm deployment and improve infrastructure
- Fix duplicate symbol names causing sync failures
- Update README with user-focused setup and usage guide
- Add research agent, symbol indexing, and tree-sitter support
- Fix spider-rs API changes in spider_md
- Fix flaky search test with proper mocking
- Add MIT license and rewrite README for end users
- Add pre-commit hooks and GitHub Actions CI/CD
- Replace OpenContext references with ContextMine
- Fix MCP endpoint routing and enable stateless HTTP mode
- Add Context7-inspired MCP tools and features
- Upgrade deprecated packages and fix tests
- Improve Query page layout to use full width
- Fix sync run stats display on dashboard
- Add Prefect progress artifacts for detailed sync progress

### Fixed

- remove unused import to pass lint check
- prevent worker OOM from Prefect subprocess zombies
