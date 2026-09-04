#!/usr/bin/env bash

# Index a pinned repository end to end and assert the resulting counts.
#
# The browser check against the web UI lives in live-web-smoke.sh, so that a
# frontend change need not wait for an indexing run and an indexing change
# need not build the web image.

set -euo pipefail

readonly FIXTURE_REPOSITORY="https://github.com/fastapi/full-stack-fastapi-template.git"
readonly FIXTURE_REVISION="486f054cc8d1aead59ec96cc0a16933d06c10e0d"
readonly FIXTURE_BRANCH="contextmine-smoke"

repository_root="$(git rev-parse --show-toplevel)"
fixture_root="$(mktemp -d -t contextmine-model-free-smoke.XXXXXX)"
fixture_repository="${fixture_root}/repository"
compose_file="${repository_root}/scripts/smoke/docker-compose.yml"
compose_project="contextmine-model-free-smoke-${BASHPID}"
compose_up_build_args=(--build)
compose_run_build_args=(--build)

if [[ "${CONTEXTMINE_SMOKE_USE_PREBUILT_IMAGES:-false}" == "true" ]]; then
  compose_up_build_args=(--no-build)
  compose_run_build_args=()

  for image in \
    "${CONTEXTMINE_SMOKE_API_IMAGE:?prebuilt API image is required}" \
    "${CONTEXTMINE_SMOKE_WORKER_IMAGE:?prebuilt analyzer image is required}"; do
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
  rm -rf "${fixture_root}"
}
trap cleanup EXIT

git init --quiet "${fixture_repository}"
git -C "${fixture_repository}" remote add origin "${FIXTURE_REPOSITORY}"
git -C "${fixture_repository}" fetch --quiet --depth 1 origin "${FIXTURE_REVISION}"
git -C "${fixture_repository}" switch --quiet --create "${FIXTURE_BRANCH}" FETCH_HEAD

resolved_revision="$(git -C "${fixture_repository}" rev-parse HEAD)"
if [[ "${resolved_revision}" != "${FIXTURE_REVISION}" ]]; then
  echo "Fixture revision mismatch: expected ${FIXTURE_REVISION}, got ${resolved_revision}" >&2
  exit 1
fi

export CONTEXTMINE_SMOKE_FIXTURE_DIR="${fixture_repository}"
export CONTEXTMINE_SMOKE_FIXTURE_REVISION="${FIXTURE_REVISION}"
export CONTEXTMINE_SMOKE_WORKER_TARGET="analyzer"

docker compose \
  --project-name "${compose_project}" \
  --file "${compose_file}" \
  up "${compose_up_build_args[@]}" --detach --wait --wait-timeout 300 \
  postgres prefect-server api

prefect_server_version="$(
  docker compose \
    --project-name "${compose_project}" \
    --file "${compose_file}" \
    exec --no-TTY prefect-server python -c 'import prefect; print(prefect.__version__)'
)"
if [[ -z "${prefect_server_version}" ]]; then
  echo "Prefect server version could not be determined" >&2
  exit 1
fi

docker compose \
  --project-name "${compose_project}" \
  --file "${compose_file}" \
  exec --no-TTY api /app/.venv/bin/python - \
  < "${repository_root}/scripts/smoke/typescript_lsp.py"

docker compose \
  --project-name "${compose_project}" \
  --file "${compose_file}" \
  run "${compose_run_build_args[@]}" --rm --no-deps \
  --env CONTEXTMINE_SMOKE_PREFECT_SERVER_VERSION="${prefect_server_version}" \
  smoke
