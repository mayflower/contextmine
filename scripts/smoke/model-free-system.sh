#!/usr/bin/env bash

set -euo pipefail

readonly FIXTURE_REPOSITORY="https://github.com/fastapi/full-stack-fastapi-template.git"
readonly FIXTURE_REVISION="486f054cc8d1aead59ec96cc0a16933d06c10e0d"
readonly FIXTURE_BRANCH="contextmine-smoke"

repository_root="$(git rev-parse --show-toplevel)"
fixture_root="$(mktemp -d -t contextmine-model-free-smoke.XXXXXX)"
fixture_repository="${fixture_root}/repository"
compose_file="${repository_root}/scripts/smoke/docker-compose.yml"
compose_project="contextmine-model-free-smoke-${BASHPID}"

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

docker compose \
  --project-name "${compose_project}" \
  --file "${compose_file}" \
  up --build --detach --wait --wait-timeout 300 \
  postgres prefect-server api web browser

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
  run --rm --no-deps \
  --env CONTEXTMINE_SMOKE_WEB_URL=http://web:5173 \
  --env CONTEXTMINE_SMOKE_BROWSER_URL=http://browser:9222 \
  --volume "${repository_root}/scripts/smoke/live_web.mjs:/app/live_web.mjs:ro" \
  web node /app/live_web.mjs

docker compose \
  --project-name "${compose_project}" \
  --file "${compose_file}" \
  exec --no-TTY api /app/.venv/bin/python - \
  < "${repository_root}/scripts/smoke/typescript_lsp.py"

docker compose \
  --project-name "${compose_project}" \
  --file "${compose_file}" \
  run --build --rm --no-deps \
  --env CONTEXTMINE_SMOKE_PREFECT_SERVER_VERSION="${prefect_server_version}" \
  smoke
