#!/bin/bash
set -e

# Create the prefect database for Prefect server
# and enable required extensions for local environments.
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE EXTENSION IF NOT EXISTS age;
    SELECT 'CREATE DATABASE prefect'
    WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'prefect')
    \gexec
    GRANT ALL PRIVILEGES ON DATABASE prefect TO $POSTGRES_USER;
EOSQL

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "prefect" <<-EOSQL
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE EXTENSION IF NOT EXISTS age;
EOSQL
