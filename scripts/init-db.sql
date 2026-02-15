-- Generic bootstrap for local PostgreSQL/pg4ai instances.
-- The contextmine database is created automatically by POSTGRES_DB env var.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS age;

SELECT 'CREATE DATABASE prefect'
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'prefect')
\gexec

GRANT ALL PRIVILEGES ON DATABASE prefect TO contextmine;

\connect prefect
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS age;
