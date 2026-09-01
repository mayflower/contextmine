#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
classifier="${script_dir}/classify-system-smoke-changes.sh"

assert_classification() {
  local name="$1"
  local expected_model_free="$2"
  local expected_otel="$3"
  shift 3

  local expected
  local actual
  expected="$(printf 'model_free=%s\notel=%s' "${expected_model_free}" "${expected_otel}")"
  actual="$(printf '%s\n' "$@" | "${classifier}")"

  if [[ "${actual}" != "${expected}" ]]; then
    printf 'Classification failed for %s\nExpected:\n%s\nActual:\n%s\n' \
      "${name}" "${expected}" "${actual}" >&2
    return 1
  fi
}

assert_classification "documentation only" false false \
  README.md docs/operations.md
assert_classification "web application" true false \
  apps/web/src/App.tsx
assert_classification "shared Python package" true true \
  packages/core/contextmine_core/config.py
assert_classification "OpenTelemetry smoke" true true \
  scripts/smoke/otel_enabled.py
assert_classification "CI policy" true true \
  .github/workflows/ci.yml
assert_classification "deployment only" false false \
  deploy/helm/contextmine/values.yaml
assert_classification "combined changes" true true \
  README.md apps/web/src/App.tsx apps/api/contextmine_api/main.py

echo "System smoke change classifier tests passed"
