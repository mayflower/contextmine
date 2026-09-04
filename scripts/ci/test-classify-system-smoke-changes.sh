#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
classifier="${script_dir}/classify-system-smoke-changes.sh"

assert_classification() {
  local name="$1"
  local expected_model_free="$2"
  local expected_otel="$3"
  local expected_live_web="$4"
  shift 4

  local expected
  local actual
  expected="$(printf 'model_free=%s\notel=%s\nlive_web=%s' \
    "${expected_model_free}" "${expected_otel}" "${expected_live_web}")"
  actual="$(printf '%s\n' "$@" | "${classifier}")"

  if [[ "${actual}" != "${expected}" ]]; then
    printf 'Classification failed for %s\nExpected:\n%s\nActual:\n%s\n' \
      "${name}" "${expected}" "${actual}" >&2
    return 1
  fi
}

assert_classification "documentation only" false false false \
  README.md docs/operations.md

# A frontend change needs the browser check, not the indexing run.
assert_classification "web application" false false true \
  apps/web/src/App.tsx

# The API sits behind the UI, so it needs both.
assert_classification "api application" true true true \
  apps/api/app/main.py

# Indexing is backend only - no reason to build the web image.
assert_classification "worker application" true true false \
  apps/worker/contextmine_worker/flows.py

assert_classification "shared Python package" true true true \
  packages/core/contextmine_core/config.py
assert_classification "OpenTelemetry smoke" true true true \
  scripts/smoke/otel_enabled.py
assert_classification "CI policy" true true true \
  .github/workflows/ci.yml
assert_classification "deployment only" false false false \
  deploy/helm/contextmine/values.yaml
assert_classification "rust crawler" true true false \
  rust/spider_md/src/main.rs
assert_classification "combined changes" true true true \
  README.md apps/web/src/App.tsx apps/api/app/main.py

echo "System smoke change classifier tests passed"
