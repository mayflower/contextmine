# How-To: Validate Cockpit in the Browser

Use these five browser checks after syncing a collection.

## 1. Discoverability and Navigation

1. Open the app.
2. Verify sidebar entry `Architecture Cockpit`.
3. Open Cockpit from Dashboard CTA (`Open Cockpit`) if available.

Expected:
1. Both paths navigate to Cockpit.
2. URL query state is preserved (`page`, `collection`, `scenario`, `view`, `layer`).

## 2. Collection/Scenario Isolation

1. Select Collection A and note scenario list + graph data.
2. Switch to Collection B.
3. Switch back.

Expected:
1. No cross-collection leakage.
2. Scenario options and view payloads update per collection.

## 3. Metrics Availability Semantics

1. Open `Overview`.
2. Check KPI cards and hotspot table.
3. Inspect the network response of `/api/twin/collections/{collection_id}/views/city`.

Expected:
1. If `metrics_status.status=ready`, KPI values are numeric and hotspots can render.
2. If `metrics_status.status=unavailable`, KPI values show `N/A`, not `0.00`.
3. Unavailable state includes reason `no_real_metrics`.

## 4. Layered Graph Views

1. Open `Topology` and `Deep Dive`.
2. Switch layer values:
   - `portfolio_system`
   - `domain_container`
   - `component_interface`
   - `code_controlflow`

Expected:
1. Node/edge set changes with layer.
2. Pagination metadata remains stable.
3. No layer-related request errors.

## 5. Exports and C4 Diff

1. Open `C4 Diff` for a TO-BE scenario with a base scenario.
2. Open `Exports` and generate all formats:
   - `cc_json`
   - `cx2`
   - `jgf`
   - `lpg_jsonl`
   - `mermaid_c4`

Expected:
1. C4 Diff shows explicit AS-IS and TO-BE panes.
2. Export generation returns content and supports copy/download actions.
3. Artifact retrieval works via `GET /api/twin/scenarios/{scenario_id}/exports/{export_id}`.
