#!/usr/bin/env bash

set -euo pipefail

# The fixture and its Git directory are read-only mounts whose host UID differs on CI.
git config --global --add safe.directory /fixtures/repository
git config --global --add safe.directory /fixtures/repository/.git

cd /app/packages/core
/app/.venv/bin/alembic upgrade head

cd /app
/app/.venv/bin/python /smoke/model_free_system.py
