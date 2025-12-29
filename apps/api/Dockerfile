# Stage 1: Build frontend
FROM node:20-alpine AS frontend-builder

WORKDIR /app

# Copy frontend source
COPY apps/web/package*.json ./
RUN npm ci

COPY apps/web/ ./
RUN npm run build

# Stage 2: Python API with frontend
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for LSP servers
# - Node.js: Required for TypeScript/JavaScript language server
# - git: Required by some language servers for project detection
# - curl: For downloading language server binaries
RUN apt-get update && apt-get install -y --no-install-recommends \
    nodejs \
    npm \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install TypeScript language server globally
RUN npm install -g typescript typescript-language-server

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy workspace files
COPY pyproject.toml uv.lock ./
COPY packages/ packages/
COPY apps/api/ apps/api/
COPY apps/worker/pyproject.toml apps/worker/

# Sync dependencies (includes multilspy + jedi-language-server for LSP support)
RUN uv sync --frozen --no-dev

# Copy built frontend from stage 1
COPY --from=frontend-builder /app/dist /app/static

# Copy and make entrypoint executable
COPY apps/api/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose port
EXPOSE 8000

# Run migrations then start server
CMD ["/entrypoint.sh"]
