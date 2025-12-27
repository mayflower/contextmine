-- Create prefect database for Prefect server
-- The contextmine database is created automatically by POSTGRES_DB env var
CREATE DATABASE prefect;
GRANT ALL PRIVILEGES ON DATABASE prefect TO contextmine;
