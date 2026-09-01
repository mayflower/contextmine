#!/usr/bin/env bash

set -euo pipefail

repository_root="$(git rev-parse --show-toplevel)"
smoke_root="$(mktemp -d -t contextmine-otel-smoke.XXXXXX)"
fixture_placeholder="${smoke_root}/fixture-placeholder"
compose_file="${repository_root}/scripts/smoke/docker-compose.yml"
compose_project="contextmine-otel-smoke-${BASHPID}"
compose_up_build_args=(--build)
compose_run_build_args=(--build)

if [[ "${CONTEXTMINE_SMOKE_USE_PREBUILT_IMAGES:-false}" == "true" ]]; then
  compose_up_build_args=(--no-build)
  compose_run_build_args=()

  for image in \
    "${CONTEXTMINE_SMOKE_API_IMAGE:?prebuilt API image is required}" \
    "${CONTEXTMINE_SMOKE_WORKER_IMAGE:?prebuilt worker image is required}"; do
    if ! docker image inspect "${image}" >/dev/null; then
      echo "Required prebuilt smoke image is unavailable: ${image}" >&2
      exit 1
    fi
  done
fi

cleanup() {
  docker compose \
    --project-name "${compose_project}" \
    --file "${compose_file}" \
    down --volumes --remove-orphans >/dev/null 2>&1 || true
  rm -rf "${smoke_root}"
}
trap cleanup EXIT

mkdir -p "${fixture_placeholder}"
export CONTEXTMINE_SMOKE_FIXTURE_DIR="${fixture_placeholder}"
export CONTEXTMINE_SMOKE_FIXTURE_REVISION="otel-enabled"
export CONTEXTMINE_SMOKE_OTEL_ENABLED="true"
export CONTEXTMINE_SMOKE_WORKER_TARGET="worker"

docker compose \
  --project-name "${compose_project}" \
  --file "${compose_file}" \
  up "${compose_up_build_args[@]}" --detach --wait --wait-timeout 300 \
  postgres prefect-server otel-collector api

# Exercise the production FastAPI instrumentation with application routes.
docker compose \
  --project-name "${compose_project}" \
  --file "${compose_file}" \
  exec --no-TTY api /app/.venv/bin/python -c \
  "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health').read(); urllib.request.urlopen('http://localhost:8000/api/config').read()"

# Exercise the production worker flow and SQLAlchemy instrumentation, then
# force-flush the trace and metric providers during graceful shutdown.
docker compose \
  --project-name "${compose_project}" \
  --file "${compose_file}" \
  run "${compose_run_build_args[@]}" --rm --no-deps smoke /bin/bash -c \
  "cd /app/packages/core && /app/.venv/bin/alembic upgrade head && cd /app && /app/.venv/bin/python /smoke/otel_enabled.py"

verified="false"
collector_output="${smoke_root}/otel-collector.log"
for _ in $(seq 1 30); do
  docker compose \
    --project-name "${compose_project}" \
    --file "${compose_file}" \
    logs --no-color otel-collector >"${collector_output}" 2>/dev/null
  if python3 "${repository_root}/scripts/smoke/verify_otel_output.py" \
    <"${collector_output}" >/dev/null 2>&1; then
    verified="true"
    break
  fi
  sleep 1
done

if [[ "${verified}" != "true" ]]; then
  python3 "${repository_root}/scripts/smoke/verify_otel_output.py" <"${collector_output}" || true
  grep -E -n \
    "service.name|Name:|prefect\\.|contextmine\\.|http\\.|url\\." \
    "${collector_output}" >&2 || true
  exit 1
fi

echo "OpenTelemetry enabled smoke passed"
