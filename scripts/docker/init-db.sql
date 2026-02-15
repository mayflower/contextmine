-- ContextMine local DB bootstrap for pg4ai (PostgreSQL + pgvector + AGE).
-- NOTE: Docker entrypoint init scripts run only on first initialization
-- (i.e. with an empty PGDATA volume).

-- Enable required extensions in main application DB.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS age;

-- Create Prefect DB if missing.
SELECT 'CREATE DATABASE prefect'
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'prefect')
\gexec

GRANT ALL PRIVILEGES ON DATABASE prefect TO contextmine;

-- Enable required extensions in Prefect DB as well.
\connect prefect
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS age;
