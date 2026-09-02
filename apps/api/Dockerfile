# Stage 1: Build frontend
FROM node:24.20.0-alpine@sha256:e67514e5d0f6c46656005e1b693b2ec9d52e80b641307de684d4a015ba7a4eaf AS frontend-builder

WORKDIR /app

COPY apps/web/package*.json ./
RUN npm ci

COPY apps/web/ ./
RUN npm run build

# Keep the existing optional TypeScript LSP capability until the analyzer
# sandbox cutover can move it across the proven execution boundary.
FROM node:24.20.0-slim@sha256:ba849c60be29959425b8734d57b8b4b7d56f98edd9504c9af091d5281095a71e AS lsp-tools

ARG TYPESCRIPT_VERSION=7.0.2
RUN npm install -g typescript@${TYPESCRIPT_VERSION} \
    && test "$(tsc --version)" = "Version ${TYPESCRIPT_VERSION}"

# Stage 2: Python API with frontend
FROM python:3.14.7-slim@sha256:656d12e70054d5fda18a045e2494c96701e9792dd1445f95b3d038df954f57e9

WORKDIR /app

# Package installation happens only while building the image. Node and the
# language server are copied from the pinned official Node image above.
# The pinned Python image records the Debian snapshot it was built from in
# debian.sources. Use that immutable snapshot for the few runtime packages too.
# hadolint ignore=DL3008
RUN sed -i \
      -e 's|^URIs: http://deb.debian.org/debian$|URIs: https://snapshot.debian.org/archive/debian/20260824T000000Z|' \
      -e 's|^URIs: http://deb.debian.org/debian-security$|URIs: https://snapshot.debian.org/archive/debian-security/20260824T000000Z|' \
      /etc/apt/sources.list.d/debian.sources \
    && printf 'Acquire::Check-Valid-Until "false";\n' > /etc/apt/apt.conf.d/99snapshot \
    && apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=lsp-tools /usr/local/ /usr/local/
COPY --from=ghcr.io/astral-sh/uv:0.12.5@sha256:e85be844203885286c60ffad8a858d48afb6c5a5c237ca0e67f12e74b8f174b1 /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
COPY packages/ packages/
COPY apps/api/ apps/api/
COPY apps/worker/pyproject.toml apps/worker/

# The runtime environment is fully materialized at build time.
RUN uv sync --frozen --no-dev --package contextmine-api

COPY --from=frontend-builder /app/dist /app/static

COPY apps/api/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
    && useradd --create-home --shell /bin/bash appuser
USER appuser

EXPOSE 8000

CMD ["/entrypoint.sh"]
