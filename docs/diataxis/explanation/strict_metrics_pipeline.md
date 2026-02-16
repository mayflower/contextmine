# Explanation: Strict Real Metrics for Twin/City

The Cockpit `Overview` should represent measurable code health, not placeholders. This is why ContextMine now enforces a strict real-metrics pipeline for GitHub sources.

## Problem That Was Solved

Before this change, Twin file nodes could exist without real metric payloads, and City snapshots were still created using default `0` values. This blurred the difference between:
1. actual low metrics,
2. missing metrics,
3. ingestion/configuration failures.

## Design Choice

ContextMine now separates these states explicitly:
1. **Ready**: real metrics were extracted and persisted.
2. **Unavailable**: no valid real metrics for the scenario.

The City API encodes this with `metrics_status`, and the UI renders `N/A` for unavailable KPI values.

## Why Strict Gate (GitHub)

GitHub sync is deterministic and repository-based. That makes it suitable for hard validation:
1. coverage report must exist (config patterns first, autodiscovery fallback),
2. report paths must map to relevant production files,
3. complexity and coupling extraction must succeed,
4. every relevant file must have all required metrics.

If any step fails, sync fails with `METRICS_GATE_FAILED:*`.

## Why Web Sources Are Different

Web sources do not have repository code + coverage report semantics. Applying the same hard metric gate would create false failures, so strict gate is limited to GitHub sources.

## Coupling and Complexity Semantics

1. **Complexity** is computed per file as sum of cyclomatic complexity numbers (CCN) from `lizard`.
2. **Coupling** is file-level, bidirectional:
   - `coupling_in` (afferent)
   - `coupling_out` (efferent)
   - `coupling = coupling_in + coupling_out`

## Operational Result

The Cockpit now communicates data quality directly:
1. no implicit `0.00 means unknown`,
2. explicit unavailable mode for missing real metrics,
3. deterministic failure when metric prerequisites are not met.
