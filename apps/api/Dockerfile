# Stage 1: Build frontend
FROM node:24-alpine@sha256:e67514e5d0f6c46656005e1b693b2ec9d52e80b641307de684d4a015ba7a4eaf AS frontend-builder

WORKDIR /app

# Copy frontend source
COPY apps/web/package*.json ./
RUN npm ci

COPY apps/web/ ./
RUN npm run build

# Supply the runtime Node.js installation from an immutable official image
# instead of executing the mutable NodeSource setup script during the build.
FROM node:24-slim@sha256:ba849c60be29959425b8734d57b8b4b7d56f98edd9504c9af091d5281095a71e AS node-runtime

# Stage 2: Python API with frontend
FROM python:3.12-slim@sha256:e5c9fa26ffb76e11e0f054f30dc2523a2f9693f0c36c0cf1e39b27e152d899fc

WORKDIR /app

ARG NODE_VERSION=24.20.0
ARG TYPESCRIPT_VERSION=6.0.3
ARG TYPESCRIPT_LANGUAGE_SERVER_VERSION=6.0.0

COPY --from=node-runtime /usr/local/ /usr/local/

# Install system dependencies for LSP servers and TypeScript language server
# - Node.js: Required for TypeScript/JavaScript language server
# - git: Required by some language servers for project detection
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g \
       typescript@${TYPESCRIPT_VERSION} \
       typescript-language-server@${TYPESCRIPT_LANGUAGE_SERVER_VERSION} \
    && test "$(node --version)" = "v${NODE_VERSION}" \
    && test "$(tsc --version)" = "Version ${TYPESCRIPT_VERSION}" \
    && test "$(typescript-language-server --version)" = "${TYPESCRIPT_LANGUAGE_SERVER_VERSION}"
COPY --from=ghcr.io/astral-sh/uv:latest@sha256:d1cbaeadc234fe19c0d93daabcf5e98738cd93c6d1dd4918ef6aa30735feb23a /uv /usr/local/bin/uv

# Copy workspace files
COPY pyproject.toml uv.lock ./
COPY packages/ packages/
COPY apps/api/ apps/api/
COPY apps/worker/pyproject.toml apps/worker/

# Sync the API package and its workspace dependencies during the image build.
# The container must not resolve or install packages when it starts.
RUN uv sync --frozen --no-dev --package contextmine-api

# Copy built frontend from stage 1
COPY --from=frontend-builder /app/dist /app/static

# Copy entrypoint, make executable, and create non-root user
COPY apps/api/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && \
    useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Run migrations then start server
CMD ["/entrypoint.sh"]
