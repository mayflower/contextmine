# QA Checklist: Extracted Cockpit Views (Per Project/Collection)

## Purpose

Validate that the read-only Cockpit views are loaded from the correct Twin scenario per collection, and that metrics semantics are explicit (`ready` vs `unavailable`).

## Preconditions

1. At least two collections exist.
2. At least one collection has an AS-IS scenario.
3. Optional: one TO-BE scenario with `base_scenario_id` for C4 compare.

## Test 1: Collection Isolation

1. Open `Architecture Cockpit`.
2. Select Collection A and note scenarios + visible data.
3. Switch to Collection B.

Expected:
1. Scenario list and content change to Collection B data.
2. No stale data from Collection A remains.

## Test 2: Layered Graph Filtering

1. Open `Topology`.
2. Switch all layers:
   - `portfolio_system`
   - `domain_container`
   - `component_interface`
   - `code_controlflow`

Expected:
1. Node/edge set changes per layer.
2. No request errors.

## Test 3: Overview Metrics Semantics

1. Open `Overview`.
2. Check KPI cards and hotspot table.
3. Inspect `GET /api/twin/collections/{collection_id}/views/city` in DevTools.

Expected:
1. If `metrics_status.status=ready`: KPIs are numeric.
2. If `metrics_status.status=unavailable`: KPIs show `N/A` (not `0.00`).
3. Hotspots render only when real metric rows exist.

## Test 4: C4 Diff AS-IS vs TO-BE

1. Select AS-IS scenario and open `C4 Diff`.
2. Select TO-BE scenario (with base) and open `C4 Diff`.

Expected:
1. AS-IS scenario: single representation.
2. TO-BE scenario: compare mode with explicit `AS-IS` and `TO-BE` sections.

## Test 5: Export Smoke Test

1. Open `Exports`.
2. Generate each format:
   - `cc_json`
   - `cx2`
   - `jgf`
   - `lpg_jsonl`
   - `mermaid_c4`

Expected:
1. `cc_json`: contains `projectName`, `nodes`, `edges`.
2. `cx2`: contains `CXVersion`.
3. `jgf`: contains `graph`.
4. `lpg_jsonl`: contains JSONL entries with `type=node|edge`.
5. `mermaid_c4`: contains C4 Mermaid content.

## API Endpoints (Debug)

1. `GET /api/twin/collections/{collection_id}/views/topology`
2. `GET /api/twin/collections/{collection_id}/views/deep-dive`
3. `GET /api/twin/collections/{collection_id}/views/city`
4. `GET /api/twin/collections/{collection_id}/views/mermaid`
5. `POST /api/twin/scenarios/{scenario_id}/exports`
6. `GET /api/twin/scenarios/{scenario_id}/exports/{export_id}`
