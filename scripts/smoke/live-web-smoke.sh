#!/usr/bin/env bash

# Drive the web UI in a real browser against a live API.
#
# Split out of model-free-system.sh: that gate additionally indexes a pinned
# repository, which needs the analyzer image and takes minutes. A frontend
# change needs the browser check, not the indexing run.

set -euo pipefail

repository_root="$(git rev-parse --show-toplevel)"
compose_file="${repository_root}/scripts/smoke/docker-compose.yml"
compose_project="contextmine-live-web-${BASHPID}"
compose_up_build_args=(--build)
compose_run_build_args=(--build)

if [[ "${CONTEXTMINE_SMOKE_USE_PREBUILT_IMAGES:-false}" == "true" ]]; then
  compose_up_build_args=(--no-build)
  compose_run_build_args=()

  for image in \
    "${CONTEXTMINE_SMOKE_API_IMAGE:?prebuilt API image is required}" \
    "${CONTEXTMINE_SMOKE_WEB_IMAGE:?prebuilt web image is required}"; do
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
}
trap cleanup EXIT

# The compose file demands these for the analyzer service, which this smoke
# never starts. Compose still interpolates them, so give them inert values.
export CONTEXTMINE_SMOKE_FIXTURE_DIR="${CONTEXTMINE_SMOKE_FIXTURE_DIR:-${repository_root}}"
export CONTEXTMINE_SMOKE_FIXTURE_REVISION="${CONTEXTMINE_SMOKE_FIXTURE_REVISION:-unused}"
export CONTEXTMINE_SMOKE_WORKER_TARGET="${CONTEXTMINE_SMOKE_WORKER_TARGET:-analyzer}"

docker compose \
  --project-name "${compose_project}" \
  --file "${compose_file}" \
  up "${compose_up_build_args[@]}" --detach --wait --wait-timeout 300 \
  postgres prefect-server api web browser

docker compose \
  --project-name "${compose_project}" \
  --file "${compose_file}" \
  run "${compose_run_build_args[@]}" --rm --no-deps \
  --env CONTEXTMINE_SMOKE_WEB_URL=http://web:5173 \
  --env CONTEXTMINE_SMOKE_BROWSER_URL=http://browser:9222 \
  --volume "${repository_root}/scripts/smoke/live_web.mjs:/app/live_web.mjs:ro" \
  web node /app/live_web.mjs
