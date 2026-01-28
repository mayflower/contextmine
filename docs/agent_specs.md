# Agent Fleet Integration Specification

This document provides all information needed to implement agents that integrate with the Backstage Agent Fleet system.

## Overview

Agent Fleet is a Backstage plugin that orchestrates AI agents across repositories. It provides:

- **Manifest**: Central registry of available agents with risk classifications
- **Configuration**: Per-repo `.agents.yaml` files controlling which agents run
- **Runs API**: Ingestion endpoint for agents to report execution status
- **Triggers**: Manual and automated agent invocation via GitHub Actions
- **UI**: Backstage pages for viewing agent status and managing configuration

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  GitHub Actions │────▶│  Backstage API   │◀────│  Backstage UI   │
│  (Agent Runner) │     │  (agent-fleet)   │     │  (agent-fleet)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                        │
        │                        ▼
        │               ┌──────────────────┐
        └──────────────▶│    PostgreSQL    │
          (run reports) │  (agent_runs)    │
                        └──────────────────┘
```

---

## API Endpoints

Base URL: `https://<backstage-url>/api/agent-fleet`

### Health Check

```
GET /health
```

Response:
```json
{"status": "ok"}
```

### Ingest Agent Run

Reports agent execution status to Backstage.

```
POST /agent-runs
Authorization: Bearer <INGESTION_TOKEN>
X-Idempotency-Key: <unique-key>
Content-Type: application/json
```

**Request Body:**

```json
{
  "correlation_id": "run-abc123",
  "repository": "owner/repo",
  "agent": "documentation",
  "trigger": "on-pr",
  "status": "success",
  "started_at": "2026-01-28T10:00:00Z",
  "finished_at": "2026-01-28T10:05:00Z",
  "summary": "Updated 3 documentation files",
  "commit_sha": "abc123def456",
  "base_ref": "main",
  "head_ref": "feature/new-api",
  "actor": "github-actions[bot]",
  "actions": [
    {"type": "file_modified", "path": "docs/api.md"},
    {"type": "pr_created", "url": "https://github.com/owner/repo/pull/42"}
  ],
  "links": [
    {"title": "PR #42", "url": "https://github.com/owner/repo/pull/42"},
    {"title": "Workflow Run", "url": "https://github.com/owner/repo/actions/runs/123"}
  ]
}
```

**Field Reference:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `correlation_id` | string | Yes | Unique identifier for this run (UUID recommended) |
| `repository` | string | Yes | Repository slug in `owner/repo` format |
| `agent` | string | Yes | Agent identifier (must match manifest) |
| `trigger` | string | Yes | What triggered this run (see Trigger Types) |
| `status` | string | Yes | Execution status (see Status Values) |
| `started_at` | string | Yes | ISO 8601 timestamp when run started |
| `finished_at` | string | No | ISO 8601 timestamp when run completed |
| `summary` | string | No | Human-readable summary of what the agent did |
| `commit_sha` | string | No | Git commit SHA that triggered the run |
| `base_ref` | string | No | Base branch (for PRs) |
| `head_ref` | string | No | Head branch (for PRs) |
| `actor` | string | No | User/bot that triggered the run |
| `actions` | array | No | Structured list of actions taken |
| `links` | array | No | Related URLs (PRs, workflow runs, etc.) |

**Response:**

- `201 Created` - New run recorded
- `200 OK` - Existing run (idempotent replay)

```json
{
  "id": 42,
  "correlation_id": "run-abc123",
  "repository": "owner/repo",
  "agent": "documentation",
  "trigger": "on-pr",
  "status": "success",
  "started_at": "2026-01-28T10:00:00Z",
  "finished_at": "2026-01-28T10:05:00Z",
  "summary": "Updated 3 documentation files",
  "created_at": "2026-01-28T10:05:01Z"
}
```

### List Agent Runs

```
GET /agent-runs?repository=owner/repo&agent=documentation&limit=50
```

Requires authenticated Backstage user.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `repository` | string | Yes | Repository slug |
| `agent` | string | No | Filter by agent |
| `limit` | number | No | Max results to return |

### Get Agents Manifest

```
GET /agents-manifest
```

Returns the central agents manifest with available agents and risk tiers.

### Get Repository Config

```
GET /repositories/:owner/:repo/agents-config
```

Requires authenticated Backstage user. Returns the `.agents.yaml` for a repository.

### Update Repository Config

```
PUT /repositories/:owner/:repo/agents-config
Content-Type: application/json
```

Creates a PR (or direct commit if allowlisted) to update `.agents.yaml`.

### Trigger Agent Run

```
POST /repositories/:owner/:repo/trigger
Content-Type: application/json

{"agent": "documentation"}
```

Dispatches a GitHub Actions workflow to run the specified agent (or `"all"`).

---

## Status Values

Agents should report one of these status values:

| Status | Description |
|--------|-------------|
| `pending` | Run is queued or starting |
| `running` | Run is in progress |
| `success` | Run completed successfully |
| `failure` | Run failed with an error |
| `skipped` | Run was skipped (conditions not met) |
| `cancelled` | Run was cancelled |

---

## Trigger Types

| Trigger | Description |
|---------|-------------|
| `on-commit` | Triggered on every commit to default branch |
| `on-merge` | Triggered when a PR is merged |
| `on-pr` | Triggered on PR open/update |
| `nightly` | Triggered on a cron schedule |
| `manual` | Triggered manually via UI or API |

---

## Risk Tiers and Policy Modes

Agents are classified into risk tiers that determine allowed policies:

### Tier 0 - Read-Only
- **Policy mode**: `comment_only`
- **Auto-merge**: Not allowed
- **Use case**: Analysis agents that only comment on PRs

### Tier 1 - Low Risk
- **Policy modes**: `direct_commit` or `pr`
- **Auto-merge**: Not allowed
- **Use case**: Safe modifications like formatting, linting fixes

### Tier 2 - Medium Risk
- **Policy mode**: `pr` only
- **Auto-merge**: Not allowed
- **Use case**: Code changes requiring human review

### Tier 3 - High Risk
- **Policy mode**: `pr` only
- **Auto-merge**: Allowed only with `require_checks` specified
- **Use case**: Significant code changes with CI verification

### Policy Modes

| Mode | Description |
|------|-------------|
| `comment_only` | Agent can only comment, no code changes |
| `direct_commit` | Agent commits directly to branch (use carefully) |
| `pr` | Agent creates a pull request for changes |

---

## Repository Configuration (.agents.yaml)

Each repository has an `.agents.yaml` file in its root:

```yaml
version: v2
agents:
  documentation:
    enabled: true
    triggers:
      - type: on-pr
      - type: on-merge
    priority: 5
    policy:
      mode: pr
      auto_merge: false
      max_prs_per_run: 3
      require_checks:
        - ci/build
        - ci/test
    conditions:
      paths:
        include:
          - "docs/**"
          - "README.md"
          - "*.md"
        exclude:
          - "docs/internal/**"
    params:
      format: markdown
      max_files: 10

  dependency-upgrader:
    enabled: true
    triggers:
      - type: nightly
        cron: "0 3 * * 1"  # Monday at 3am
    priority: 3
    policy:
      mode: pr
      auto_merge: true
      require_checks:
        - ci/test
    params:
      ecosystems:
        - python
        - npm
```

### Schema Reference

```typescript
type AgentsConfig = {
  version: string;  // "v2" or "1"
  agents: Record<string, AgentConfig>;
};

type AgentConfig = {
  enabled?: boolean;           // Default: true
  triggers?: Trigger[];        // When to run
  priority?: number;           // Execution order (lower = first)
  policy?: Policy;             // How changes are applied
  conditions?: Conditions;     // Path filtering
  params?: Record<string, unknown>;  // Agent-specific parameters
};

type Trigger = {
  type: "on-commit" | "on-merge" | "on-pr" | "nightly" | "manual";
  cron?: string;  // Only valid for type: nightly
};

type Policy = {
  mode: "pr" | "direct_commit" | "comment_only";
  auto_merge?: boolean;
  max_prs_per_run?: number;
  require_checks?: string[];  // Required for tier 3 auto_merge
};

type Conditions = {
  paths?: {
    include?: string[];  // Glob patterns to include
    exclude?: string[];  // Glob patterns to exclude
  };
};
```

---

## Agents Manifest

The manifest is a central registry of available agents:

```yaml
version: "1"
agents:
  documentation:
    name: documentation
    description: Generates and updates documentation
    risk_tier: 2
    default_params:
      format: markdown

  code-review:
    name: code-review
    description: Reviews code for issues and improvements
    risk_tier: 0

  dependency-upgrader:
    name: dependency-upgrader
    description: Upgrades project dependencies
    risk_tier: 3
    default_params:
      ecosystems: ["python", "npm"]
```

---

## Implementing an Agent

### 1. GitHub Actions Workflow

Create `.github/workflows/agent-fleet.yml`:

```yaml
name: Agent Fleet

on:
  workflow_dispatch:
    inputs:
      agent:
        description: 'Agent to run (or "all")'
        required: true
        default: 'all'
      correlation_id:
        description: 'Correlation ID for tracking'
        required: true

  push:
    branches: [main]

  pull_request:
    types: [opened, synchronize]

concurrency:
  group: agent-fleet-${{ github.repository }}
  cancel-in-progress: false

env:
  BACKSTAGE_URL: ${{ secrets.BACKSTAGE_URL }}
  BACKSTAGE_TOKEN: ${{ secrets.BACKSTAGE_TOKEN }}

jobs:
  orchestrate:
    runs-on: ubuntu-latest
    if: |
      !contains(github.event.head_commit.message, '[skip agents]') &&
      !startsWith(github.head_ref || github.ref_name, 'agent/')
    outputs:
      agents: ${{ steps.determine.outputs.agents }}
      correlation_id: ${{ steps.determine.outputs.correlation_id }}
    steps:
      - uses: actions/checkout@v4

      - name: Determine agents to run
        id: determine
        run: |
          if [ "${{ github.event_name }}" = "workflow_dispatch" ]; then
            CORRELATION_ID="${{ github.event.inputs.correlation_id }}"
            if [ "${{ github.event.inputs.agent }}" = "all" ]; then
              AGENTS=$(yq -r '.agents | keys | @json' .agents.yaml)
            else
              AGENTS='["${{ github.event.inputs.agent }}"]'
            fi
          else
            CORRELATION_ID=$(uuidgen)
            # Determine trigger type and filter enabled agents
            if [ "${{ github.event_name }}" = "pull_request" ]; then
              TRIGGER="on-pr"
            elif [ "${{ github.event_name }}" = "push" ]; then
              TRIGGER="on-merge"
            else
              TRIGGER="manual"
            fi
            AGENTS=$(yq -r --arg trigger "$TRIGGER" '
              .agents | to_entries |
              map(select(.value.enabled != false and
                  (.value.triggers // [] | map(.type) | contains([$trigger])))) |
              map(.key) | @json' .agents.yaml)
          fi
          echo "agents=$AGENTS" >> $GITHUB_OUTPUT
          echo "correlation_id=$CORRELATION_ID" >> $GITHUB_OUTPUT

  run-agent:
    needs: orchestrate
    runs-on: ubuntu-latest
    if: needs.orchestrate.outputs.agents != '[]'
    strategy:
      fail-fast: false
      matrix:
        agent: ${{ fromJson(needs.orchestrate.outputs.agents) }}
    steps:
      - uses: actions/checkout@v4

      - name: Report pending
        run: |
          curl -X POST "$BACKSTAGE_URL/api/agent-fleet/agent-runs" \
            -H "Authorization: Bearer $BACKSTAGE_TOKEN" \
            -H "Content-Type: application/json" \
            -H "X-Idempotency-Key: ${{ needs.orchestrate.outputs.correlation_id }}-${{ matrix.agent }}-pending" \
            -d '{
              "correlation_id": "${{ needs.orchestrate.outputs.correlation_id }}",
              "repository": "${{ github.repository }}",
              "agent": "${{ matrix.agent }}",
              "trigger": "${{ github.event_name }}",
              "status": "running",
              "started_at": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'",
              "commit_sha": "${{ github.sha }}",
              "actor": "${{ github.actor }}"
            }'

      - name: Run agent
        id: run
        run: |
          # Your agent implementation here
          # Set outputs: status, summary, actions, links
          echo "status=success" >> $GITHUB_OUTPUT
          echo "summary=Agent completed successfully" >> $GITHUB_OUTPUT

      - name: Report result
        if: always()
        run: |
          STATUS="${{ steps.run.outputs.status || 'failure' }}"
          SUMMARY="${{ steps.run.outputs.summary || 'Agent execution failed' }}"

          curl -X POST "$BACKSTAGE_URL/api/agent-fleet/agent-runs" \
            -H "Authorization: Bearer $BACKSTAGE_TOKEN" \
            -H "Content-Type: application/json" \
            -H "X-Idempotency-Key: ${{ needs.orchestrate.outputs.correlation_id }}-${{ matrix.agent }}-final" \
            -d '{
              "correlation_id": "${{ needs.orchestrate.outputs.correlation_id }}",
              "repository": "${{ github.repository }}",
              "agent": "${{ matrix.agent }}",
              "trigger": "${{ github.event_name }}",
              "status": "'"$STATUS"'",
              "started_at": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'",
              "finished_at": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'",
              "summary": "'"$SUMMARY"'",
              "commit_sha": "${{ github.sha }}",
              "actor": "${{ github.actor }}"
            }'
```

### 2. Agent Implementation Pattern

```python
#!/usr/bin/env python3
"""
Example agent implementation pattern.
"""
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any

@dataclass
class AgentResult:
    status: str = "success"  # success, failure, skipped
    summary: str = ""
    actions: list[dict[str, Any]] = field(default_factory=list)
    links: list[dict[str, str]] = field(default_factory=list)

class BaseAgent:
    """Base class for agent implementations."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.params = config.get("params", {})
        self.policy = config.get("policy", {})
        self.conditions = config.get("conditions", {})

    def should_run(self, changed_files: list[str]) -> bool:
        """Check if agent should run based on path conditions."""
        paths = self.conditions.get("paths", {})
        include = paths.get("include", ["**"])
        exclude = paths.get("exclude", [])

        # Implement glob matching logic
        # Return True if any changed file matches include and not exclude
        return True

    def run(self) -> AgentResult:
        """Execute the agent. Override in subclasses."""
        raise NotImplementedError

    def create_pr(self, branch: str, title: str, body: str) -> str:
        """Create a PR if policy allows."""
        if self.policy.get("mode") != "pr":
            raise ValueError("Policy does not allow PR creation")
        # Implement PR creation via GitHub API
        return "https://github.com/owner/repo/pull/123"

    def commit_direct(self, message: str) -> str:
        """Commit directly if policy allows."""
        if self.policy.get("mode") != "direct_commit":
            raise ValueError("Policy does not allow direct commits")
        # Implement direct commit
        return "abc123"

    def add_comment(self, body: str) -> None:
        """Add a comment (allowed for all policies)."""
        # Implement comment via GitHub API
        pass


class DocumentationAgent(BaseAgent):
    """Example documentation agent."""

    def run(self) -> AgentResult:
        result = AgentResult()

        try:
            # 1. Analyze codebase
            # 2. Generate/update documentation
            # 3. Apply changes based on policy

            if self.policy.get("mode") == "pr":
                pr_url = self.create_pr(
                    branch="agent/docs-update",
                    title="docs: Update documentation",
                    body="Automated documentation update"
                )
                result.actions.append({"type": "pr_created", "url": pr_url})
                result.links.append({"title": "Documentation PR", "url": pr_url})

            result.status = "success"
            result.summary = "Updated documentation for 5 files"

        except Exception as e:
            result.status = "failure"
            result.summary = f"Failed: {str(e)}"

        return result


def main():
    # Load config from .agents.yaml
    agent_name = os.environ.get("AGENT_NAME", "documentation")

    # Parse .agents.yaml and get agent config
    with open(".agents.yaml") as f:
        import yaml
        config = yaml.safe_load(f)

    agent_config = config.get("agents", {}).get(agent_name, {})

    # Run agent
    agent = DocumentationAgent(agent_config)
    result = agent.run()

    # Output for GitHub Actions
    print(f"::set-output name=status::{result.status}")
    print(f"::set-output name=summary::{result.summary}")
    if result.actions:
        print(f"::set-output name=actions::{json.dumps(result.actions)}")
    if result.links:
        print(f"::set-output name=links::{json.dumps(result.links)}")

    sys.exit(0 if result.status == "success" else 1)


if __name__ == "__main__":
    main()
```

---

## Required Secrets

Configure these in your repository or organization:

| Secret | Description |
|--------|-------------|
| `BACKSTAGE_URL` | Backstage instance URL (e.g., `https://backstage.example.com`) |
| `BACKSTAGE_TOKEN` | Ingestion token matching `agentFleet.ingestionToken` in Backstage config |
| `AGENT_GIT_TOKEN` | GitHub PAT for agent Git operations (do not use `GITHUB_TOKEN`) |

---

## Loop Prevention

To prevent infinite loops, agents should skip execution when:

1. Commit message contains `[skip agents]`
2. Branch name starts with `agent/`
3. Actor is a known bot account

Example check:
```yaml
if: |
  !contains(github.event.head_commit.message, '[skip agents]') &&
  !startsWith(github.head_ref || github.ref_name, 'agent/')
```

---

## Best Practices

### Idempotency
- Use unique idempotency keys for each run report
- Pattern: `{correlation_id}-{agent}-{phase}`

### Error Handling
- Always report final status, even on failure
- Include meaningful summaries for debugging
- Set appropriate exit codes

### Resource Management
- Use concurrency groups to prevent parallel runs
- Respect `max_prs_per_run` limits
- Clean up temporary branches

### Security
- Never log or expose `BACKSTAGE_TOKEN`
- Use dedicated tokens, not user PATs
- Validate inputs and sanitize outputs

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| 401 on run ingestion | Invalid token | Verify `BACKSTAGE_TOKEN` matches config |
| 400 validation error | Schema mismatch | Check required fields and types |
| Agent not in manifest | Unknown agent ID | Add agent to manifest or fix ID |
| Policy validation failed | Risk tier violation | Adjust policy to match agent's tier |
| Config fetch failed | No `.agents.yaml` | Create config file in repo root |

### Debugging

1. Check Backstage backend logs for API errors
2. Verify GitHub Actions workflow runs
3. Use `/admin/agents` page for run history
4. Test ingestion with curl before automation

---

## Database Schema

Reference for advanced integrations:

```sql
CREATE TABLE agent_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  correlation_id TEXT NOT NULL,
  idempotency_key TEXT NOT NULL UNIQUE,
  repository TEXT NOT NULL,
  agent TEXT NOT NULL,
  trigger TEXT NOT NULL,
  status TEXT NOT NULL,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  commit_sha TEXT,
  base_ref TEXT,
  head_ref TEXT,
  actor TEXT,
  summary TEXT,
  actions TEXT,  -- JSON
  links TEXT,    -- JSON
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_agent_runs_repository ON agent_runs(repository);
CREATE INDEX idx_agent_runs_started_at ON agent_runs(started_at);
```

---

## Version History

| Version | Changes |
|---------|---------|
| v2 | Current version, supports all features |
| v1 | Legacy format, array-based agent list |
