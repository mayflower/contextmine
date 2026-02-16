# Tutorial: First Architecture Cockpit Walkthrough

This tutorial walks through the first successful run of the read-only Architecture Cockpit with real Twin metrics.

## Goal

By the end, you can:
1. sync a GitHub source,
2. open the Cockpit,
3. switch through all extracted views,
4. verify that Overview metrics are real (not placeholder zero defaults).

## Prerequisites

1. ContextMine is running locally.
2. You can log in to the web UI.
3. You have a GitHub repository source in a collection.
4. The repository has a coverage report (`lcov.info`, `coverage.xml`, `jacoco.xml`, `clover.xml`, etc.).

## Step 1: Add a GitHub Source with Coverage Patterns (optional but recommended)

If your report path is non-standard, set explicit patterns via API:

```bash
curl -X PATCH "http://localhost:8000/api/sources/<source_id>" \
  -H "Content-Type: application/json" \
  -b "<session-cookie>" \
  -d '{
    "coverage_report_patterns": [
      "**/coverage/lcov.info",
      "**/reports/coverage.xml"
    ]
  }'
```

Notes:
1. `coverage_report_patterns` is GitHub-only.
2. Patterns are repository-relative globs.
3. Empty arrays are rejected.

## Step 2: Run Sync

1. Open the collection in the web UI.
2. Trigger `Sync now` for the GitHub source.
3. Wait until the run finishes with `SUCCESS`.

Strict behavior:
1. If coverage cannot be found or mapped, sync fails (`METRICS_GATE_FAILED:*`).
2. On success, file-level LOC/complexity/coupling/coverage are written to Twin file nodes.

## Step 3: Open Architecture Cockpit

1. Go to `Architecture Cockpit` in the sidebar.
2. Select collection + scenario.
3. Start on `Overview`.

## Step 4: Check Overview

Expected outcome:
1. KPI values render as real numbers when metrics are ready.
2. If metrics are unavailable, UI shows `N/A` and an explicit unavailable message.
3. Hotspots table is shown only when valid metric rows exist.

API contract behind this view:
1. `GET /api/twin/collections/{collection_id}/views/city`
2. Includes `metrics_status` with:
   - `status`: `ready` or `unavailable`
   - `reason`: `ok` or `no_real_metrics`
   - `strict_mode`: `true|false`

## Step 5: Check Remaining Views

1. `Topology`: graph by selected architecture layer.
2. `Deep Dive`: larger code/controlflow graph slices.
3. `C4 Diff`: AS-IS / TO-BE Mermaid comparison.
4. `Exports`: generate `cc_json`, `cx2`, `jgf`, `lpg_jsonl`, `mermaid_c4`.

You now have a complete first Cockpit walkthrough with strict real metrics.
