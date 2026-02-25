# Reference: Cockpit Contracts

This page documents the current backend contracts used by the read-only Architecture Cockpit.

## Core View Endpoints

1. `GET /api/twin/collections/{collection_id}/views/city`
2. `GET /api/twin/collections/{collection_id}/views/topology`
3. `GET /api/twin/collections/{collection_id}/views/deep-dive`
4. `GET /api/twin/collections/{collection_id}/views/mermaid`
5. `GET /api/twin/collections/{collection_id}/views/arc42`
6. `GET /api/twin/collections/{collection_id}/views/arc42/drift`
7. `GET /api/twin/collections/{collection_id}/views/ports-adapters`
8. `GET /api/twin/collections/{collection_id}/views/erm`
9. `GET /api/twin/collections/{collection_id}/views/evolution/investment-utilization`
10. `GET /api/twin/collections/{collection_id}/views/evolution/knowledge-islands`
11. `GET /api/twin/collections/{collection_id}/views/evolution/temporal-coupling`
12. `GET /api/twin/collections/{collection_id}/views/evolution/fitness-functions`

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

## arc42 View Response

`GET /api/twin/collections/{collection_id}/views/arc42`

Query parameters:
1. `scenario_id` (optional)
2. `section` (optional, e.g. `5`, `quality`, `deployment`)
3. `regenerate` (optional, default `false`)

Response shape:

```json
{
  "collection_id": "uuid",
  "scenario": { "id": "uuid", "name": "AS-IS", "version": 5 },
  "artifact": { "id": "uuid", "name": "scenario-id.arc42.md", "kind": "arc42", "cached": false },
  "section": "10_quality_requirements",
  "arc42": {
    "title": "arc42 - AS-IS",
    "generated_at": "2026-02-22T19:00:00Z",
    "sections": { "10_quality_requirements": "..." },
    "markdown": "# arc42 - AS-IS\\n...",
    "warnings": [],
    "confidence_summary": { "total": 42, "avg": 0.84 },
    "section_coverage": { "10_quality_requirements": true }
  },
  "facts_hash": "sha256...",
  "warnings": []
}
```

## arc42 Drift Response

`GET /api/twin/collections/{collection_id}/views/arc42/drift`

Query parameters:
1. `scenario_id` (optional)
2. `baseline_scenario_id` (optional)

Response highlights:
1. `summary.total`
2. `summary.by_type` (`added|removed|changed_confidence|moved_component|new_port|removed_adapter`)
3. `summary.severity` (`low|medium`)
4. `deltas[]` with `before`/`after`

## Ports/Adapters Response

`GET /api/twin/collections/{collection_id}/views/ports-adapters`

Query parameters:
1. `scenario_id` (optional)
2. `direction` (`inbound|outbound`, optional)
3. `container` (optional exact match)

Response highlights:
1. `summary.total|inbound|outbound`
2. `items[]` with `direction`, `port_name`, `adapter_name`, `container`, `component`, `protocol`, `confidence`, `evidence`

## ERM View Response

`GET /api/twin/collections/{collection_id}/views/erm`

Query parameters:
1. `scenario_id` (optional)
2. `include_mermaid` (`true|false`, default `true`)

Response highlights:
1. `summary.tables|columns|foreign_keys|has_mermaid`
2. `tables[]` with `name`, `column_count`, `primary_keys`, and `columns[]`
3. `foreign_keys[]` with source/target table+column pairs
4. `mermaid.content` with ERD Mermaid source when `MERMAID_ERD` artifact exists

## Evolution View Responses

### Investment/Utilization

`GET /api/twin/collections/{collection_id}/views/evolution/investment-utilization`

Query parameters:
1. `scenario_id` (optional)
2. `entity_level` (`container|component`, default `container`)
3. `window_days` (default `365`)

Response highlights:
1. `status|reason` (`ready|unavailable`)
2. `summary.total_entities|coverage_entity_ratio|utilization_available`
3. `summary.quadrants`
4. `items[]` with `investment_score`, `utilization_score`, `quadrant`, `size`

### Knowledge Islands

`GET /api/twin/collections/{collection_id}/views/evolution/knowledge-islands`

Query parameters:
1. `scenario_id` (optional)
2. `entity_level` (`container|component`, default `container`)
3. `window_days` (default `365`)
4. `ownership_threshold` (default `0.7`)

Response highlights:
1. `summary.bus_factor_global|single_owner_files|churn_p75`
2. `entities[]` with `bus_factor`, `dominant_owner`, `single_owner_ratio`
3. `at_risk_files[]` with `dominant_share`, `churn`, `coverage`, `last_touched_at`

### Temporal Coupling

`GET /api/twin/collections/{collection_id}/views/evolution/temporal-coupling`

Query parameters:
1. `scenario_id` (optional)
2. `entity_level` (`file|container|component`, default `component`)
3. `window_days` (default `365`)
4. `min_jaccard` (default `0.2`)
5. `max_edges` (default `300`)

Response highlights:
1. `summary.nodes|edges|cross_boundary_edges|avg_jaccard`
2. `graph.nodes[]` and `graph.edges[]`
3. `edges[]` include `co_change_count`, directional ratios, and `cross_boundary`

### Fitness Functions

`GET /api/twin/collections/{collection_id}/views/evolution/fitness-functions`

Query parameters:
1. `scenario_id` (optional)
2. `window_days` (default `365`)
3. `include_resolved` (`true|false`, default `false`)

Response highlights:
1. `summary.rules|violations|open|resolved|highest_severity`
2. `rules[]` grouped by `rule_id`
3. `violations[]` sourced from persisted `twin_findings` (`finding_type` prefixed `fitness.`)

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
