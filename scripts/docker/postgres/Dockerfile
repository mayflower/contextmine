FROM pgvector/pgvector:pg16

# Create init script inline to avoid permission issues
RUN echo '#!/bin/sh\n\
set -e\n\
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL\n\
    CREATE DATABASE prefect;\n\
    GRANT ALL PRIVILEGES ON DATABASE prefect TO contextmine;\n\
EOSQL' > /docker-entrypoint-initdb.d/init-db.sh \
    && chmod +x /docker-entrypoint-initdb.d/init-db.sh
