FROM pgvector/pgvector:pg16@sha256:ccc6e83d6e35e931dc7c5def2022729d5a6c370318d099181995567ff1fb4d6b

# Create init script inline to avoid permission issues
RUN echo '#!/bin/sh\n\
set -e\n\
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL\n\
    CREATE DATABASE prefect;\n\
    GRANT ALL PRIVILEGES ON DATABASE prefect TO contextmine;\n\
EOSQL' > /docker-entrypoint-initdb.d/init-db.sh \
    && chmod +x /docker-entrypoint-initdb.d/init-db.sh
