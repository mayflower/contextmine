#!/usr/bin/env bash

set -euo pipefail

# The fixture is a read-only bind mount whose host UID differs on CI runners.
git config --global --add safe.directory /fixtures/repository

cd /app/packages/core
/app/.venv/bin/alembic upgrade head

cd /app
/app/.venv/bin/python /smoke/model_free_system.py
