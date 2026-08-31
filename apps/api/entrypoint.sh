#!/bin/sh
set -e

# Run database migrations
echo "Running database migrations..."
cd /app/packages/core
/app/.venv/bin/alembic upgrade head

# Start the API server
echo "Starting API server..."
cd /app/apps/api
exec /app/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
