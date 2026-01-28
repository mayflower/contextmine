# Agent Fleet Infrastructure Setup Guide

Complete guide to deploying Agent Fleet in your infrastructure. This covers Backstage configuration, agent manifest, GitHub Actions workflows, and per-repository setup.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Architecture Overview](#architecture-overview)
3. [Backstage Configuration](#backstage-configuration)
4. [Agents Manifest](#agents-manifest)
5. [GitHub Actions Workflow](#github-actions-workflow)
6. [Repository Configuration](#repository-configuration)
7. [Kubernetes Deployment](#kubernetes-deployment)
8. [Secrets Management](#secrets-management)
9. [Verification](#verification)
10. [Maintenance](#maintenance)

---

## Prerequisites

- Backstage instance (v1.20+)
- PostgreSQL database
- GitHub App or PAT with repo/workflow permissions
- Kubernetes cluster (for production)
- CI/CD pipeline (GitHub Actions, GitLab CI)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GitHub / GitLab                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │   Repo A    │  │   Repo B    │  │   Repo C    │                 │
│  │ .agents.yaml│  │ .agents.yaml│  │ .agents.yaml│                 │
│  │  workflow   │  │  workflow   │  │  workflow   │                 │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                 │
└─────────┼────────────────┼────────────────┼────────────────────────┘
          │                │                │
          │   Run Reports (POST /agent-runs)
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Backstage                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Agent Fleet Plugin                         │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │  │
│  │  │   Backend   │  │   Frontend  │  │  Database   │          │  │
│  │  │   Router    │  │   UI Pages  │  │  (Postgres) │          │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘          │  │
│  │         │                                   ▲                 │  │
│  │         └───────────────────────────────────┘                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Agents Manifest (YAML)                           │  │
│  │   - Available agents                                          │  │
│  │   - Risk tiers                                                │  │
│  │   - Default parameters                                        │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Backstage Configuration

### app-config.yaml (Base Configuration)

```yaml
# =============================================================================
# Agent Fleet Configuration
# =============================================================================

agentFleet:
  # Bearer token for agent run ingestion (generate with: openssl rand -hex 32)
  ingestionToken: ${AGENT_FLEET_INGESTION_TOKEN}

  # Backstage group allowed to access admin endpoints
  adminGroup: 'platform-team'

  # Database configuration (uses Backstage's database by default)
  # Uncomment to use separate database
  # database:
  #   client: pg
  #   connection:
  #     host: ${AGENT_FLEET_DB_HOST}
  #     port: 5432
  #     user: ${AGENT_FLEET_DB_USER}
  #     password: ${AGENT_FLEET_DB_PASSWORD}
  #     database: agent_fleet

  # Agents manifest source
  manifest:
    source:
      # Option 1: Local file (development)
      type: file
      path: './agents-manifest.yaml'

      # Option 2: Git repository (production)
      # type: git
      # url: https://github.com/your-org/agent-fleet-config/blob/main/agents-manifest.yaml

  # Repositories allowed to use direct_commit mode (bypass PR)
  directCommitAllowlist:
    - 'your-org/infrastructure-config'
    - 'your-org/auto-generated-docs'

  # GitHub Actions workflow to trigger for manual runs
  trigger:
    workflowId: 'agent-fleet.yml'

  # How long to retain run history (days)
  retentionDays: 90

# =============================================================================
# Required Integrations
# =============================================================================

integrations:
  github:
    - host: github.com
      # Use GitHub App for production, PAT for development
      # apps:
      #   - appId: ${GITHUB_APP_ID}
      #     privateKey: ${GITHUB_APP_PRIVATE_KEY}
      #     webhookSecret: ${GITHUB_WEBHOOK_SECRET}
      #     clientId: ${GITHUB_APP_CLIENT_ID}
      #     clientSecret: ${GITHUB_APP_CLIENT_SECRET}
      token: ${GITHUB_TOKEN}

  # GitLab integration (if using GitLab)
  gitlab:
    - host: gitlab.com
      token: ${GITLAB_TOKEN}
    # Self-hosted GitLab
    - host: git.example.com
      apiBaseUrl: https://git.example.com/api/v4
      token: ${GITLAB_SELF_HOSTED_TOKEN}
```

### app-config.production.yaml (Production Overrides)

```yaml
app:
  baseUrl: ${BACKSTAGE_BASE_URL}

backend:
  baseUrl: ${BACKSTAGE_BASE_URL}
  listen:
    port: 7007
  database:
    client: pg
    connection:
      host: ${POSTGRES_HOST}
      port: ${POSTGRES_PORT}
      user: ${POSTGRES_USER}
      password: ${POSTGRES_PASSWORD}
      database: ${POSTGRES_DB}
      ssl:
        require: true
        rejectUnauthorized: false

agentFleet:
  manifest:
    source:
      type: git
      url: ${AGENT_FLEET_MANIFEST_URL}

auth:
  environment: production
  providers:
    google:
      production:
        clientId: ${AUTH_GOOGLE_CLIENT_ID}
        clientSecret: ${AUTH_GOOGLE_CLIENT_SECRET}
    # Or use your preferred auth provider
```

---

## Agents Manifest

Create `agents-manifest.yaml` with your available agents:

```yaml
# =============================================================================
# Agent Fleet Manifest
# =============================================================================
# This file defines all available agents, their risk classifications,
# and default configurations. Store in a central config repository.
# =============================================================================

version: "1"

agents:
  # ---------------------------------------------------------------------------
  # Tier 0: Read-Only Agents (comment_only mode required)
  # ---------------------------------------------------------------------------

  code-reviewer:
    name: code-reviewer
    description: |
      Analyzes pull requests for code quality, security issues, and best practices.
      Adds review comments but does not modify code.
    risk_tier: 0
    default_params:
      review_types:
        - security
        - performance
        - style
        - bugs
      max_comments: 20
      severity_threshold: warning

  security-scanner:
    name: security-scanner
    description: |
      Scans code for security vulnerabilities, secrets, and compliance issues.
      Reports findings as PR comments.
    risk_tier: 0
    default_params:
      scan_types:
        - secrets
        - vulnerabilities
        - compliance
      fail_on_critical: true

  architecture-analyzer:
    name: architecture-analyzer
    description: |
      Analyzes architectural changes and provides feedback on design decisions.
      Checks for coupling, cohesion, and dependency issues.
    risk_tier: 0
    default_params:
      check_dependencies: true
      check_layering: true

  # ---------------------------------------------------------------------------
  # Tier 1: Low Risk Agents (direct_commit or pr mode, no auto_merge)
  # ---------------------------------------------------------------------------

  formatter:
    name: formatter
    description: |
      Applies code formatting according to project style guides.
      Runs prettier, black, gofmt, etc.
    risk_tier: 1
    default_params:
      tools:
        - prettier
        - black
        - gofmt
        - rustfmt

  linter-fixer:
    name: linter-fixer
    description: |
      Automatically fixes linting issues that have safe auto-fix options.
      Does not fix issues requiring human judgment.
    risk_tier: 1
    default_params:
      auto_fix_only: true
      tools:
        - eslint
        - ruff
        - golangci-lint

  license-header:
    name: license-header
    description: |
      Ensures all source files have required license headers.
      Adds missing headers automatically.
    risk_tier: 1
    default_params:
      license_type: Apache-2.0
      file_patterns:
        - "**/*.py"
        - "**/*.js"
        - "**/*.ts"
        - "**/*.go"

  # ---------------------------------------------------------------------------
  # Tier 2: Medium Risk Agents (pr mode required, no auto_merge)
  # ---------------------------------------------------------------------------

  documentation:
    name: documentation
    description: |
      Generates and updates documentation based on code changes.
      Updates README, API docs, and inline documentation.
    risk_tier: 2
    default_params:
      formats:
        - markdown
        - openapi
      update_readme: true
      generate_api_docs: true

  test-generator:
    name: test-generator
    description: |
      Generates unit tests for new or modified code.
      Creates test skeletons that require human review.
    risk_tier: 2
    default_params:
      frameworks:
        python: pytest
        javascript: jest
        go: testing
      coverage_target: 80

  changelog-generator:
    name: changelog-generator
    description: |
      Generates changelog entries based on commit messages and PR descriptions.
      Follows Keep a Changelog format.
    risk_tier: 2
    default_params:
      format: keepachangelog
      include_breaking_changes: true

  migration-generator:
    name: migration-generator
    description: |
      Generates database migration files when schema changes are detected.
      Requires human review before applying.
    risk_tier: 2
    default_params:
      frameworks:
        - alembic
        - prisma
        - knex

  # ---------------------------------------------------------------------------
  # Tier 3: High Risk Agents (pr mode, auto_merge requires checks)
  # ---------------------------------------------------------------------------

  dependency-upgrader:
    name: dependency-upgrader
    description: |
      Upgrades project dependencies to latest compatible versions.
      Creates PRs with detailed changelogs and breaking change warnings.
    risk_tier: 3
    default_params:
      ecosystems:
        - npm
        - pip
        - go
        - cargo
      major_updates: false
      security_only: false
      group_updates: true

  refactoring-agent:
    name: refactoring-agent
    description: |
      Performs automated refactoring based on detected code smells.
      Extracts methods, renames variables, simplifies conditionals.
    risk_tier: 3
    default_params:
      refactoring_types:
        - extract_method
        - rename_variable
        - simplify_conditional
        - remove_dead_code
      min_confidence: 0.9

  code-modernizer:
    name: code-modernizer
    description: |
      Updates code to use modern language features and patterns.
      Python 2->3, ES5->ES6, etc.
    risk_tier: 3
    default_params:
      languages:
        python:
          target_version: "3.11"
        javascript:
          target_version: "ES2022"
        typescript:
          target_version: "5.0"

  security-patcher:
    name: security-patcher
    description: |
      Applies security patches for known vulnerabilities.
      Updates dependencies with CVE fixes.
    risk_tier: 3
    default_params:
      severity_threshold: high
      auto_merge_critical: true
      require_checks:
        - ci/test
        - ci/security-scan

  infrastructure-updater:
    name: infrastructure-updater
    description: |
      Updates infrastructure-as-code configurations.
      Terraform, Kubernetes manifests, Helm charts.
    risk_tier: 3
    default_params:
      tools:
        - terraform
        - kubernetes
        - helm
      plan_only: false

# =============================================================================
# Global Settings
# =============================================================================

settings:
  # Default branch for operations
  default_branch: main

  # Commit message prefix for agent commits
  commit_prefix: "chore(agent)"

  # PR label for agent-created PRs
  pr_label: "agent-fleet"

  # Maximum concurrent agent runs per repository
  max_concurrent_runs: 1

  # Timeout for agent execution (minutes)
  execution_timeout: 30
```

---

## GitHub Actions Workflow

Create `.github/workflows/agent-fleet.yml` in each repository:

```yaml
# =============================================================================
# Agent Fleet Orchestration Workflow
# =============================================================================
# This workflow orchestrates AI agents based on .agents.yaml configuration.
# It handles triggers, execution, and reporting to Backstage.
# =============================================================================

name: Agent Fleet

on:
  # Manual trigger from Backstage UI
  workflow_dispatch:
    inputs:
      agent:
        description: 'Agent to run ("all" for all enabled agents)'
        required: true
        default: 'all'
      correlation_id:
        description: 'Correlation ID for tracking'
        required: true
      trigger_type:
        description: 'Trigger type'
        required: false
        default: 'manual'

  # Automatic triggers
  push:
    branches: [main, master]

  pull_request:
    types: [opened, synchronize, reopened]

  schedule:
    # Nightly at 3am UTC
    - cron: '0 3 * * *'

# Prevent concurrent runs on the same repository
concurrency:
  group: agent-fleet-${{ github.repository }}-${{ github.event.pull_request.number || github.ref }}
  cancel-in-progress: false

env:
  BACKSTAGE_URL: ${{ secrets.BACKSTAGE_URL }}
  BACKSTAGE_TOKEN: ${{ secrets.BACKSTAGE_TOKEN }}
  AGENT_GIT_TOKEN: ${{ secrets.AGENT_GIT_TOKEN }}

jobs:
  # ===========================================================================
  # Determine which agents to run
  # ===========================================================================
  orchestrate:
    name: Orchestrate
    runs-on: ubuntu-latest
    # Skip if commit message contains [skip agents] or branch starts with agent/
    if: |
      !contains(github.event.head_commit.message, '[skip agents]') &&
      !startsWith(github.head_ref || github.ref_name, 'agent/') &&
      github.actor != 'agent-fleet[bot]'
    outputs:
      agents: ${{ steps.determine.outputs.agents }}
      correlation_id: ${{ steps.determine.outputs.correlation_id }}
      trigger_type: ${{ steps.determine.outputs.trigger_type }}
      should_run: ${{ steps.determine.outputs.should_run }}
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Install yq
        run: |
          sudo wget -qO /usr/local/bin/yq https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64
          sudo chmod +x /usr/local/bin/yq

      - name: Determine agents and trigger
        id: determine
        run: |
          # Check if .agents.yaml exists
          if [ ! -f .agents.yaml ]; then
            echo "No .agents.yaml found, skipping"
            echo "should_run=false" >> $GITHUB_OUTPUT
            echo "agents=[]" >> $GITHUB_OUTPUT
            exit 0
          fi

          # Determine trigger type
          if [ "${{ github.event_name }}" = "workflow_dispatch" ]; then
            TRIGGER_TYPE="${{ github.event.inputs.trigger_type }}"
            CORRELATION_ID="${{ github.event.inputs.correlation_id }}"
            REQUESTED_AGENT="${{ github.event.inputs.agent }}"
          elif [ "${{ github.event_name }}" = "pull_request" ]; then
            TRIGGER_TYPE="on-pr"
            CORRELATION_ID="pr-${{ github.event.pull_request.number }}-$(date +%s)"
          elif [ "${{ github.event_name }}" = "push" ]; then
            TRIGGER_TYPE="on-merge"
            CORRELATION_ID="push-${{ github.sha }}-$(date +%s)"
          elif [ "${{ github.event_name }}" = "schedule" ]; then
            TRIGGER_TYPE="nightly"
            CORRELATION_ID="nightly-$(date +%Y%m%d)"
          else
            TRIGGER_TYPE="manual"
            CORRELATION_ID="manual-$(uuidgen || date +%s)"
          fi

          echo "Trigger type: $TRIGGER_TYPE"
          echo "Correlation ID: $CORRELATION_ID"

          # Get list of agents to run
          if [ "${{ github.event_name }}" = "workflow_dispatch" ] && [ "$REQUESTED_AGENT" != "all" ]; then
            # Specific agent requested
            AGENTS="[\"$REQUESTED_AGENT\"]"
          else
            # Filter agents by trigger type and enabled status
            AGENTS=$(yq -r --arg trigger "$TRIGGER_TYPE" '
              .agents // {} | to_entries |
              map(select(
                .value.enabled != false and
                (
                  (.value.triggers // [{"type": "manual"}]) |
                  map(.type) |
                  contains([$trigger])
                )
              )) |
              map(.key) | @json
            ' .agents.yaml)
          fi

          echo "Agents to run: $AGENTS"

          # Check if we have any agents to run
          if [ "$AGENTS" = "[]" ] || [ "$AGENTS" = "null" ] || [ -z "$AGENTS" ]; then
            echo "No agents to run for trigger: $TRIGGER_TYPE"
            echo "should_run=false" >> $GITHUB_OUTPUT
            echo "agents=[]" >> $GITHUB_OUTPUT
          else
            echo "should_run=true" >> $GITHUB_OUTPUT
            echo "agents=$AGENTS" >> $GITHUB_OUTPUT
          fi

          echo "correlation_id=$CORRELATION_ID" >> $GITHUB_OUTPUT
          echo "trigger_type=$TRIGGER_TYPE" >> $GITHUB_OUTPUT

  # ===========================================================================
  # Run each agent
  # ===========================================================================
  run-agent:
    name: Run ${{ matrix.agent }}
    needs: orchestrate
    if: needs.orchestrate.outputs.should_run == 'true'
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      max-parallel: 2
      matrix:
        agent: ${{ fromJson(needs.orchestrate.outputs.agents) }}
    env:
      AGENT_NAME: ${{ matrix.agent }}
      CORRELATION_ID: ${{ needs.orchestrate.outputs.correlation_id }}
      TRIGGER_TYPE: ${{ needs.orchestrate.outputs.trigger_type }}
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
          token: ${{ secrets.AGENT_GIT_TOKEN }}

      - name: Install dependencies
        run: |
          sudo wget -qO /usr/local/bin/yq https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64
          sudo chmod +x /usr/local/bin/yq

      - name: Extract agent config
        id: config
        run: |
          # Extract agent-specific configuration
          CONFIG=$(yq -r --arg agent "$AGENT_NAME" '.agents[$agent] // {}' .agents.yaml)
          echo "config<<EOF" >> $GITHUB_OUTPUT
          echo "$CONFIG" >> $GITHUB_OUTPUT
          echo "EOF" >> $GITHUB_OUTPUT

          # Extract specific fields
          POLICY_MODE=$(yq -r --arg agent "$AGENT_NAME" '.agents[$agent].policy.mode // "pr"' .agents.yaml)
          AUTO_MERGE=$(yq -r --arg agent "$AGENT_NAME" '.agents[$agent].policy.auto_merge // false' .agents.yaml)
          PRIORITY=$(yq -r --arg agent "$AGENT_NAME" '.agents[$agent].priority // 5' .agents.yaml)

          echo "policy_mode=$POLICY_MODE" >> $GITHUB_OUTPUT
          echo "auto_merge=$AUTO_MERGE" >> $GITHUB_OUTPUT
          echo "priority=$PRIORITY" >> $GITHUB_OUTPUT

      - name: Report running status
        run: |
          curl -sf -X POST "$BACKSTAGE_URL/api/agent-fleet/agent-runs" \
            -H "Authorization: Bearer $BACKSTAGE_TOKEN" \
            -H "Content-Type: application/json" \
            -H "X-Idempotency-Key: $CORRELATION_ID-$AGENT_NAME-running" \
            -d '{
              "correlation_id": "'"$CORRELATION_ID"'",
              "repository": "${{ github.repository }}",
              "agent": "'"$AGENT_NAME"'",
              "trigger": "'"$TRIGGER_TYPE"'",
              "status": "running",
              "started_at": "'"$(date -u +%Y-%m-%dT%H:%M:%SZ)"'",
              "commit_sha": "${{ github.sha }}",
              "base_ref": "${{ github.base_ref || github.ref_name }}",
              "head_ref": "${{ github.head_ref || github.ref_name }}",
              "actor": "${{ github.actor }}"
            }' || echo "Warning: Failed to report running status"

      # =========================================================================
      # Agent Execution
      # =========================================================================
      # Replace this section with your actual agent implementation
      # =========================================================================

      - name: Setup Python (for Python-based agents)
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Setup Node.js (for JS-based agents)
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Run agent
        id: agent
        env:
          GITHUB_TOKEN: ${{ secrets.AGENT_GIT_TOKEN }}
          AGENT_CONFIG: ${{ steps.config.outputs.config }}
          POLICY_MODE: ${{ steps.config.outputs.policy_mode }}
        run: |
          echo "Running agent: $AGENT_NAME"
          echo "Policy mode: $POLICY_MODE"
          echo "Config: $AGENT_CONFIG"

          # =====================================================================
          # YOUR AGENT IMPLEMENTATION HERE
          # =====================================================================
          # Example structure:
          #
          # 1. Install agent-specific dependencies
          # pip install your-agent-package
          #
          # 2. Run the agent
          # python -m your_agent \
          #   --name "$AGENT_NAME" \
          #   --config "$AGENT_CONFIG" \
          #   --policy "$POLICY_MODE" \
          #   --output-file /tmp/agent-result.json
          #
          # 3. Parse results
          # STATUS=$(jq -r '.status' /tmp/agent-result.json)
          # SUMMARY=$(jq -r '.summary' /tmp/agent-result.json)
          # ACTIONS=$(jq -c '.actions' /tmp/agent-result.json)
          # LINKS=$(jq -c '.links' /tmp/agent-result.json)
          # =====================================================================

          # Placeholder implementation
          echo "Agent $AGENT_NAME executed successfully"
          STATUS="success"
          SUMMARY="Agent $AGENT_NAME completed (placeholder implementation)"
          ACTIONS='[]'
          LINKS='[]'

          # Set outputs
          echo "status=$STATUS" >> $GITHUB_OUTPUT
          echo "summary=$SUMMARY" >> $GITHUB_OUTPUT
          echo "actions=$ACTIONS" >> $GITHUB_OUTPUT
          echo "links=$LINKS" >> $GITHUB_OUTPUT

      # =========================================================================
      # Report Results
      # =========================================================================

      - name: Report final status
        if: always()
        env:
          AGENT_STATUS: ${{ steps.agent.outputs.status || 'failure' }}
          AGENT_SUMMARY: ${{ steps.agent.outputs.summary || 'Agent execution failed' }}
          AGENT_ACTIONS: ${{ steps.agent.outputs.actions || '[]' }}
          AGENT_LINKS: ${{ steps.agent.outputs.links || '[]' }}
        run: |
          # Escape special characters in summary
          SAFE_SUMMARY=$(echo "$AGENT_SUMMARY" | jq -Rs '.')

          curl -sf -X POST "$BACKSTAGE_URL/api/agent-fleet/agent-runs" \
            -H "Authorization: Bearer $BACKSTAGE_TOKEN" \
            -H "Content-Type: application/json" \
            -H "X-Idempotency-Key: $CORRELATION_ID-$AGENT_NAME-final" \
            -d '{
              "correlation_id": "'"$CORRELATION_ID"'",
              "repository": "${{ github.repository }}",
              "agent": "'"$AGENT_NAME"'",
              "trigger": "'"$TRIGGER_TYPE"'",
              "status": "'"$AGENT_STATUS"'",
              "started_at": "'"$(date -u +%Y-%m-%dT%H:%M:%SZ)"'",
              "finished_at": "'"$(date -u +%Y-%m-%dT%H:%M:%SZ)"'",
              "summary": '"$SAFE_SUMMARY"',
              "commit_sha": "${{ github.sha }}",
              "base_ref": "${{ github.base_ref || github.ref_name }}",
              "head_ref": "${{ github.head_ref || github.ref_name }}",
              "actor": "${{ github.actor }}",
              "actions": '"$AGENT_ACTIONS"',
              "links": '"$AGENT_LINKS"'
            }' || echo "Warning: Failed to report final status"

  # ===========================================================================
  # Summary
  # ===========================================================================
  summary:
    name: Summary
    needs: [orchestrate, run-agent]
    if: always() && needs.orchestrate.outputs.should_run == 'true'
    runs-on: ubuntu-latest
    steps:
      - name: Generate summary
        run: |
          echo "## Agent Fleet Run Summary" >> $GITHUB_STEP_SUMMARY
          echo "" >> $GITHUB_STEP_SUMMARY
          echo "**Correlation ID:** ${{ needs.orchestrate.outputs.correlation_id }}" >> $GITHUB_STEP_SUMMARY
          echo "**Trigger:** ${{ needs.orchestrate.outputs.trigger_type }}" >> $GITHUB_STEP_SUMMARY
          echo "**Agents:** ${{ needs.orchestrate.outputs.agents }}" >> $GITHUB_STEP_SUMMARY
          echo "" >> $GITHUB_STEP_SUMMARY
          echo "View results in [Backstage Agent Fleet](${{ secrets.BACKSTAGE_URL }}/catalog)" >> $GITHUB_STEP_SUMMARY
```

---

## Repository Configuration

### Example .agents.yaml for a Python Project

```yaml
version: v2

agents:
  # Code review on every PR
  code-reviewer:
    enabled: true
    triggers:
      - type: on-pr
    priority: 1
    policy:
      mode: comment_only
    params:
      review_types:
        - security
        - performance
        - bugs
      max_comments: 15

  # Format code on PR
  formatter:
    enabled: true
    triggers:
      - type: on-pr
    priority: 2
    policy:
      mode: pr
      auto_merge: false
    conditions:
      paths:
        include:
          - "**/*.py"
          - "**/*.js"
          - "**/*.ts"
    params:
      tools:
        - black
        - prettier
        - isort

  # Update documentation when code changes
  documentation:
    enabled: true
    triggers:
      - type: on-merge
    priority: 3
    policy:
      mode: pr
      auto_merge: false
    conditions:
      paths:
        include:
          - "src/**"
          - "lib/**"
        exclude:
          - "**/*_test.py"
          - "**/test_*.py"
    params:
      formats:
        - markdown
      update_readme: true

  # Weekly dependency updates
  dependency-upgrader:
    enabled: true
    triggers:
      - type: nightly
        cron: "0 3 * * 1"  # Monday 3am
    priority: 5
    policy:
      mode: pr
      auto_merge: true
      require_checks:
        - ci / test
        - ci / lint
    params:
      ecosystems:
        - pip
        - npm
      major_updates: false
      security_only: false

  # Security scanning
  security-scanner:
    enabled: true
    triggers:
      - type: on-pr
      - type: nightly
    priority: 1
    policy:
      mode: comment_only
    params:
      scan_types:
        - secrets
        - vulnerabilities
      fail_on_critical: true
```

### Example .agents.yaml for a Node.js Project

```yaml
version: v2

agents:
  code-reviewer:
    enabled: true
    triggers:
      - type: on-pr
    policy:
      mode: comment_only
    params:
      review_types:
        - security
        - performance
        - react-best-practices

  linter-fixer:
    enabled: true
    triggers:
      - type: on-pr
    policy:
      mode: pr
    conditions:
      paths:
        include:
          - "src/**/*.ts"
          - "src/**/*.tsx"
    params:
      tools:
        - eslint

  test-generator:
    enabled: true
    triggers:
      - type: on-pr
    policy:
      mode: pr
    conditions:
      paths:
        include:
          - "src/components/**"
          - "src/hooks/**"
    params:
      framework: jest
      coverage_target: 80

  dependency-upgrader:
    enabled: true
    triggers:
      - type: nightly
        cron: "0 4 * * 2"  # Tuesday 4am
    policy:
      mode: pr
      auto_merge: true
      require_checks:
        - build
        - test
        - lint
    params:
      ecosystems:
        - npm
      major_updates: false
```

### Example .agents.yaml for Infrastructure Repository

```yaml
version: v2

agents:
  security-scanner:
    enabled: true
    triggers:
      - type: on-pr
    policy:
      mode: comment_only
    params:
      scan_types:
        - secrets
        - terraform-security
        - kubernetes-security

  documentation:
    enabled: true
    triggers:
      - type: on-merge
    policy:
      mode: direct_commit  # Allowed for infra repos
    conditions:
      paths:
        include:
          - "terraform/**"
          - "kubernetes/**"
    params:
      formats:
        - markdown
      generate_diagrams: true

  infrastructure-updater:
    enabled: true
    triggers:
      - type: nightly
    policy:
      mode: pr
      auto_merge: false  # Always require review for infra
    params:
      tools:
        - terraform
        - helm
      plan_only: true
```

---

## Kubernetes Deployment

### Helm Values (values-agent-fleet.yaml)

```yaml
# Add to your Backstage Helm values

backstage:
  extraEnvVars:
    - name: AGENT_FLEET_INGESTION_TOKEN
      valueFrom:
        secretKeyRef:
          name: backstage-secrets
          key: agent-fleet-ingestion-token

  extraVolumes:
    - name: agents-manifest
      configMap:
        name: agents-manifest

  extraVolumeMounts:
    - name: agents-manifest
      mountPath: /app/agents-manifest.yaml
      subPath: agents-manifest.yaml

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: agents-manifest
data:
  agents-manifest.yaml: |
    version: "1"
    agents:
      code-reviewer:
        name: code-reviewer
        risk_tier: 0
      formatter:
        name: formatter
        risk_tier: 1
      documentation:
        name: documentation
        risk_tier: 2
      dependency-upgrader:
        name: dependency-upgrader
        risk_tier: 3
```

### External Secrets (if using External Secrets Operator)

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: backstage-agent-fleet-secrets
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-backend
    kind: ClusterSecretStore
  target:
    name: backstage-secrets
  data:
    - secretKey: agent-fleet-ingestion-token
      remoteRef:
        key: secret/data/backstage
        property: agent_fleet_ingestion_token
    - secretKey: github-token
      remoteRef:
        key: secret/data/backstage
        property: github_token
```

---

## Secrets Management

### Required Secrets

| Secret | Description | Where Used |
|--------|-------------|------------|
| `BACKSTAGE_URL` | Backstage instance URL | GitHub Actions |
| `BACKSTAGE_TOKEN` | Ingestion token (matches `agentFleet.ingestionToken`) | GitHub Actions |
| `AGENT_GIT_TOKEN` | GitHub PAT for agent Git operations | GitHub Actions |
| `GITHUB_TOKEN` | GitHub token for Backstage integrations | Backstage |
| `AGENT_FLEET_INGESTION_TOKEN` | Same as BACKSTAGE_TOKEN | Backstage |

### Generating Tokens

```bash
# Generate ingestion token (64 hex characters)
openssl rand -hex 32

# Example output: a1b2c3d4e5f6...
```

### GitHub Repository Secrets Setup

```bash
# Using GitHub CLI
gh secret set BACKSTAGE_URL --body "https://backstage.example.com"
gh secret set BACKSTAGE_TOKEN --body "your-ingestion-token"
gh secret set AGENT_GIT_TOKEN --body "ghp_your_personal_access_token"
```

### Organization-Level Secrets

For organization-wide deployment, set secrets at the org level:

```bash
gh secret set BACKSTAGE_URL --org your-org --visibility all
gh secret set BACKSTAGE_TOKEN --org your-org --visibility all
```

---

## Verification

### 1. Test Backstage Health

```bash
curl -s https://backstage.example.com/api/agent-fleet/health
# Expected: {"status":"ok"}
```

### 2. Test Manifest Endpoint

```bash
curl -s https://backstage.example.com/api/agent-fleet/agents-manifest
# Expected: Your agents manifest JSON
```

### 3. Test Run Ingestion

```bash
curl -X POST https://backstage.example.com/api/agent-fleet/agent-runs \
  -H "Authorization: Bearer YOUR_INGESTION_TOKEN" \
  -H "Content-Type: application/json" \
  -H "X-Idempotency-Key: test-run-001" \
  -d '{
    "correlation_id": "test-001",
    "repository": "your-org/test-repo",
    "agent": "code-reviewer",
    "trigger": "manual",
    "status": "success",
    "started_at": "2026-01-28T12:00:00Z",
    "finished_at": "2026-01-28T12:01:00Z",
    "summary": "Test run completed"
  }'
# Expected: 201 Created with run data
```

### 4. Verify UI

1. Open Backstage: `https://backstage.example.com`
2. Navigate to a catalog component
3. Click the "Agents" tab
4. Verify manifest and config load correctly

### 5. Test Workflow Trigger

```bash
# Trigger workflow manually
gh workflow run agent-fleet.yml \
  --repo your-org/your-repo \
  -f agent=code-reviewer \
  -f correlation_id=manual-test-001
```

---

## Maintenance

### Log Retention

Configure retention in Backstage config:

```yaml
agentFleet:
  retentionDays: 90  # Keep 90 days of run history
```

### Cleanup Job (Optional)

Add a scheduled cleanup if not using Backstage's built-in retention:

```sql
-- Run periodically to clean old records
DELETE FROM agent_runs
WHERE started_at < NOW() - INTERVAL '90 days';
```

### Monitoring

Key metrics to monitor:

- Agent run success/failure rates
- Run duration by agent type
- API response times
- Database size growth

### Updating Agents Manifest

1. Update `agents-manifest.yaml` in your config repository
2. Manifest is cached for 60 seconds
3. No restart required - changes apply automatically

### Adding New Agents

1. Add agent definition to manifest
2. Assign appropriate risk tier
3. Update repository `.agents.yaml` files to enable
4. Implement agent logic in workflow

---

## Quick Start Checklist

- [ ] Deploy Backstage with agent-fleet plugin
- [ ] Configure `agentFleet` section in app-config.yaml
- [ ] Create and deploy agents-manifest.yaml
- [ ] Generate and configure ingestion token
- [ ] Set up GitHub/GitLab integration
- [ ] Create organization secrets (BACKSTAGE_URL, BACKSTAGE_TOKEN, AGENT_GIT_TOKEN)
- [ ] Add `.github/workflows/agent-fleet.yml` to repositories
- [ ] Add `.agents.yaml` to repositories
- [ ] Verify health endpoint
- [ ] Test manual workflow trigger
- [ ] Monitor first automated runs

---

## Support

- **Backstage Docs**: https://backstage.io/docs
- **Agent Fleet Plugin**: See `plugins/agent-fleet/README.md`
- **API Reference**: See `docs/agent_specs.md`
