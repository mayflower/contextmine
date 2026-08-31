#!/usr/bin/env bash

set -euo pipefail

cd /app/packages/core
/app/.venv/bin/alembic upgrade head

cd /app
/app/.venv/bin/python /smoke/model_free_system.py
