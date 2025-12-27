#!/bin/bash
set -e

# Create the prefect database for Prefect server
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE prefect;
    GRANT ALL PRIVILEGES ON DATABASE prefect TO $POSTGRES_USER;
EOSQL
