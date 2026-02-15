#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="${1:-postgres}"
POSTGRES_USER="${POSTGRES_USER:-contextmine}"
POSTGRES_DB="${POSTGRES_DB:-contextmine}"

docker compose exec -T "${SERVICE_NAME}" \
  psql -v ON_ERROR_STOP=1 -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" <<'SQL'
\echo 'Checking pg4ai extensions...'
SELECT extname
FROM pg_extension
WHERE extname IN ('vector', 'age')
ORDER BY extname;

\echo 'Checking pgvector operator...'
SELECT '[1,2,3]'::vector <-> '[1,2,4]'::vector AS distance;

\echo 'Checking Apache AGE cypher execution...'
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

SELECT create_graph('contextmine_smoke')
WHERE NOT EXISTS (
  SELECT 1 FROM ag_catalog.ag_graph WHERE name = 'contextmine_smoke'
);

SELECT * FROM cypher(
  'contextmine_smoke',
  $$ CREATE (n:Smoke {name: 'ok'}) RETURN n $$
) AS (result agtype);

SELECT * FROM cypher(
  'contextmine_smoke',
  $$ MATCH (n:Smoke) RETURN count(n) $$
) AS (result agtype);

SELECT drop_graph('contextmine_smoke', true)
WHERE EXISTS (
  SELECT 1 FROM ag_catalog.ag_graph WHERE name = 'contextmine_smoke'
);
SQL

echo "pg4ai smoke check passed."
