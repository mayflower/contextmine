# Stage 1: Build frontend
FROM node:24-alpine AS frontend-builder

WORKDIR /app

# Copy frontend source
COPY apps/web/package*.json ./
RUN npm ci

COPY apps/web/ ./
RUN npm run build

# Stage 2: Python API with frontend
FROM python:3.12-slim

WORKDIR /app

ARG NODE_VERSION=24
ARG TYPESCRIPT_VERSION=6.0.3
ARG TYPESCRIPT_LANGUAGE_SERVER_VERSION=6.0.0

# Install system dependencies for LSP servers and TypeScript language server
# - Node.js: Required for TypeScript/JavaScript language server
# - git: Required by some language servers for project detection
# - curl and gnupg: Required to install a supported Node.js release
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION}.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g \
       typescript@${TYPESCRIPT_VERSION} \
       typescript-language-server@${TYPESCRIPT_LANGUAGE_SERVER_VERSION} \
    && node --version | grep -Eq "^v${NODE_VERSION}\\." \
    && test "$(tsc --version)" = "Version ${TYPESCRIPT_VERSION}" \
    && test "$(typescript-language-server --version)" = "${TYPESCRIPT_LANGUAGE_SERVER_VERSION}"
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

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
