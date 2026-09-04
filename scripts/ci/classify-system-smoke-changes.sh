#!/usr/bin/env bash

set -euo pipefail

model_free="false"
otel="false"
live_web="false"

while IFS= read -r path || [[ -n "${path}" ]]; do
  [[ -z "${path}" ]] && continue

  case "${path}" in
    .dockerignore | \
      .github/workflows/ci.yml | \
      docker-compose.yml | \
      pyproject.toml | \
      uv.lock | \
      apps/api/* | \
      apps/worker/* | \
      packages/* | \
      rust/* | \
      scripts/ci/* | \
      scripts/docker/* | \
      scripts/smoke/*)
      model_free="true"
      ;;
  esac

  case "${path}" in
    .dockerignore | \
      .github/workflows/ci.yml | \
      docker-compose.yml | \
      pyproject.toml | \
      uv.lock | \
      apps/api/* | \
      apps/worker/* | \
      packages/* | \
      rust/* | \
      scripts/ci/* | \
      scripts/docker/* | \
      scripts/smoke/docker-compose.yml | \
      scripts/smoke/otel-enabled.sh | \
      scripts/smoke/otel_enabled.py | \
      scripts/smoke/verify_otel_output.py)
      otel="true"
      ;;
  esac

  case "${path}" in
    .dockerignore | \
      .github/workflows/ci.yml | \
      docker-compose.yml | \
      pyproject.toml | \
      uv.lock | \
      apps/web/* | \
      apps/api/* | \
      packages/* | \
      scripts/ci/* | \
      scripts/docker/* | \
      scripts/smoke/*)
      live_web="true"
      ;;
  esac
done

printf 'model_free=%s\n' "${model_free}"
printf 'otel=%s\n' "${otel}"
printf 'live_web=%s\n' "${live_web}"
