# Reference: Cockpit Contracts

This page documents the current backend contracts used by the read-only Architecture Cockpit.

## Core View Endpoints

1. `GET /api/twin/collections/{collection_id}/views/city`
2. `GET /api/twin/collections/{collection_id}/views/topology`
3. `GET /api/twin/collections/{collection_id}/views/deep-dive`
4. `GET /api/twin/collections/{collection_id}/views/mermaid`

## Scenario and Export Endpoints

1. `GET /api/twin/scenarios?collection_id=<uuid>`
2. `POST /api/twin/scenarios/{scenario_id}/exports`
3. `GET /api/twin/scenarios/{scenario_id}/exports/{export_id}`
4. `GET /api/twin/scenarios/{scenario_id}/graph/neighborhood?node_id=<id>&projection=<architecture|code_file|code_symbol>&hops=1&limit=200`

## City View Response (important)

`GET /api/twin/collections/{collection_id}/views/city`

```json
{
  "collection_id": "uuid",
  "scenario": {
    "id": "uuid",
    "collection_id": "uuid",
    "name": "AS-IS",
    "version": 12,
    "is_as_is": true,
    "base_scenario_id": null
  },
  "summary": {
    "metric_nodes": 120,
    "coverage_avg": 71.4,
    "complexity_avg": 9.8,
    "coupling_avg": 3.2,
    "change_frequency_avg": 4.6,
    "churn_avg": 21.3
  },
  "metrics_status": {
    "status": "ready",
    "reason": "ok",
    "strict_mode": true
  },
  "hotspots": [
    {
      "node_natural_key": "file:src/main.py",
      "loc": 210,
      "symbol_count": 12,
      "coverage": 73.2,
      "complexity": 16.1,
      "coupling": 5.0,
      "change_frequency": 8.0,
      "churn": 46.0
    }
  ],
  "cc_json": {}
}
```

If metrics are unavailable:
1. `metrics_status.status = "unavailable"`
2. `metrics_status.reason = "no_real_metrics"`
3. `summary.coverage_avg|complexity_avg|coupling_avg|change_frequency_avg|churn_avg = null`

## Mermaid C4 View Response

`GET /api/twin/collections/{collection_id}/views/mermaid`

Query parameters:
1. `scenario_id` (optional)
2. `compare_with_base` (`true|false`, default `true`)
3. `c4_view` (`context|container|component|code|deployment`, default `container`)
4. `c4_scope` (optional focus selector)
5. `max_nodes` (optional, default `120`, min `10`, max `5000`)

Single-mode response shape:
```json
{
  "mode": "single",
  "c4_view": "component",
  "c4_scope": "billing",
  "max_nodes": 120,
  "warnings": [],
  "content": "C4Component\\n..."
}
```

Compare-mode response shape:
```json
{
  "mode": "compare",
  "c4_view": "code",
  "c4_scope": "billing",
  "max_nodes": 120,
  "as_is": "C4Component\\n...",
  "to_be": "C4Component\\n...",
  "warnings": ["..."],
  "as_is_warnings": ["..."],
  "to_be_warnings": ["..."]
}
```

Best-effort views:
1. `context` and `deployment` may include warnings when source signals are sparse.
2. `code` may include fallback warnings when call edges are unavailable.

## Source Config Contract for Coverage Reports

`POST /api/collections/{collection_id}/sources`
`PATCH /api/sources/{source_id}`

Optional field:

```json
{
  "coverage_report_patterns": ["**/coverage/lcov.info", "**/coverage.xml"]
}
```

Rules:
1. Allowed only for `type=github`.
2. Patterns must be repository-relative globs.
3. Empty arrays are invalid.
4. Stored under `Source.config.metrics.coverage_report_patterns`.

## Strict Metrics Gate Semantics

For GitHub sources, metrics extraction enforces real file metrics on relevant production files:
1. `loc`
2. `complexity` (sum CCN per file)
3. `coupling` (`coupling_in + coupling_out`)
4. `coverage` (report-ingested)

Gate failures surface as sync errors with `METRICS_GATE_FAILED:*` codes.

## Graph View Query Controls

`GET /api/twin/collections/{collection_id}/views/topology`
`GET /api/twin/collections/{collection_id}/views/deep-dive`

Supported filtering/paging query params:
1. `page`
2. `limit`
3. `include_kinds` (comma-separated)
4. `exclude_kinds` (comma-separated)

## Relevant File Filter (production scope)

Included:
1. Files present in semantic snapshot and detected project scope.

Excluded (examples):
1. `node_modules`, `vendor`, `dist`, `build`, `target`, `.venv`, `venv`, `__pycache__`
2. `**/test/**`, `**/tests/**`, `**/__tests__/**`, `*.spec.*`, `*.test.*`, `*_test.py`, `*Test.java`
3. `**/generated/**`, `**/gen/**`, `*.generated.*`

## Language Scope

Real metrics scope currently matches SCIP support:
1. Python
2. TypeScript
3. JavaScript
4. Java
5. PHP
