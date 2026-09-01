# Build stage for Rust spider_md binary
# spider_md dependencies currently require rustc >= 1.88.
FROM rust:1.88-slim@sha256:38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89 AS rust-builder

WORKDIR /build

# Install build dependencies, copy source, and build release binary
COPY rust/spider_md/ ./
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && cargo build --release

# Supply the runtime Node.js installation from an immutable official image
# instead of executing the mutable NodeSource setup script during the build.
FROM node:24-slim@sha256:ba849c60be29959425b8734d57b8b4b7d56f98edd9504c9af091d5281095a71e AS node-runtime

# Python worker stage
FROM python:3.12-slim@sha256:e5c9fa26ffb76e11e0f054f30dc2523a2f9693f0c36c0cf1e39b27e152d899fc

# SCIP tool versions (pinned for reproducibility)
ARG SCIP_TYPESCRIPT_VERSION=0.3.14
ARG SCIP_PYTHON_VERSION=0.6.6
ARG SCIP_JAVA_VERSION=0.10.3
ARG SCIP_PHP_COMMIT=efaf87cd05cf174db8ac25ad4d07eb646d4883d1
ARG SCIP_PHP_ARCHIVE_SHA256=28c6c05da118d8d052d2b6f99d63956f72dd7d656cf6e90166396ee62dbcc939
ARG COURSIER_LAUNCHER_COMMIT=15f36c167c30be237105f923151adaf177e7ee61
ARG COURSIER_LAUNCHER_AMD64_SHA256=62b141b186e4dfdef03af64c36bdcdfeaab2df30de14950ca220e6e43e72c26b
ARG COURSIER_LAUNCHER_ARM64_SHA256=11d980e5714abdebe93688f22cedfa8ac227a95f4de9f96c53c4779d2547cff7
ARG COMPOSER_VERSION=2.10.3
ARG COMPOSER_SHA256=7a2d379d5b8ffdaa028580ef26494c36d2feef4b178d3dd1473a4dbc5e17c8d6
ARG NODE_VERSION=24.20.0
ARG TYPESCRIPT_VERSION=6.0.3
ARG TYPESCRIPT_LANGUAGE_SERVER_VERSION=6.0.0

COPY --from=node-runtime /usr/local/ /usr/local/
COPY --from=node-runtime /opt/ /opt/

# Install system dependencies
# - git: for cloning repos
# - cloc: language census for multi-language detection
# - ca-certificates, libssl3: for spider_md and HTTPS
# - curl: for installing pinned Coursier, Composer, and scip-php artifacts
# - openjdk-21-jre-headless: for scip-java
# - php-cli, composer: for scip-php
RUN apt-get update && apt-get install -y \
    git \
    cloc \
    ca-certificates \
    libssl3 \
    curl \
    unzip \
    openjdk-21-jre-headless \
    php-cli \
    && rm -rf /var/lib/apt/lists/* \
    && test "$(node --version)" = "v${NODE_VERSION}" \
    && corepack enable \
    && npm install -g \
       @sourcegraph/scip-typescript@${SCIP_TYPESCRIPT_VERSION} \
       @sourcegraph/scip-python@${SCIP_PYTHON_VERSION} \
       typescript@${TYPESCRIPT_VERSION} \
       typescript-language-server@${TYPESCRIPT_LANGUAGE_SERVER_VERSION} \
    && test "$(tsc --version)" = "Version ${TYPESCRIPT_VERSION}" \
    && test "$(typescript-language-server --version)" = "${TYPESCRIPT_LANGUAGE_SERVER_VERSION}"

# Install Coursier/scip-java, Composer/scip-php, and verify SCIP tools.
# Every downloaded artifact is selected by an immutable revision and checksum.
ENV SCIP_PHP_HOME=/opt/scip-php
ENV PATH="${PATH}:${SCIP_PHP_HOME}/bin"
RUN ARCH=$(dpkg --print-architecture) && \
    case "$ARCH" in \
      arm64) CS_ARTIFACT="cs-aarch64-pc-linux.gz"; CS_SHA256="${COURSIER_LAUNCHER_ARM64_SHA256}" ;; \
      amd64) CS_ARTIFACT="cs-x86_64-pc-linux.gz"; CS_SHA256="${COURSIER_LAUNCHER_AMD64_SHA256}" ;; \
      *) echo "Unsupported Coursier architecture: $ARCH" >&2; exit 1 ;; \
    esac && \
    CS_URL="https://raw.githubusercontent.com/coursier/launchers/${COURSIER_LAUNCHER_COMMIT}/${CS_ARTIFACT}" && \
    curl -fsSL "$CS_URL" -o /tmp/cs.gz && \
    echo "${CS_SHA256}  /tmp/cs.gz" | sha256sum -c - && \
    gzip -d /tmp/cs.gz && \
    mv /tmp/cs /usr/local/bin/cs && \
    chmod +x /usr/local/bin/cs && \
    /usr/local/bin/cs install --contrib \
      scip-java:${SCIP_JAVA_VERSION} \
      --install-dir /usr/local/bin && \
    curl -fsSL "https://getcomposer.org/download/${COMPOSER_VERSION}/composer.phar" \
      -o /usr/local/bin/composer && \
    echo "${COMPOSER_SHA256}  /usr/local/bin/composer" | sha256sum -c - && \
    chmod +x /usr/local/bin/composer && \
    test "$(composer --no-ansi --version | cut -d' ' -f3)" = "${COMPOSER_VERSION}" && \
    curl -fsSL "https://github.com/mayflower/scip-php/archive/${SCIP_PHP_COMMIT}.tar.gz" \
      -o /tmp/scip-php.tar.gz && \
    echo "${SCIP_PHP_ARCHIVE_SHA256}  /tmp/scip-php.tar.gz" | sha256sum -c - && \
    mkdir -p ${SCIP_PHP_HOME} && \
    tar -xzf /tmp/scip-php.tar.gz --strip-components=1 -C ${SCIP_PHP_HOME} && \
    rm /tmp/scip-php.tar.gz && \
    composer install \
      --working-dir=${SCIP_PHP_HOME} \
      --no-interaction --prefer-dist --no-progress --no-dev && \
    test -f ${SCIP_PHP_HOME}/vendor/autoload.php && \
    chmod -R a+rX ${SCIP_PHP_HOME} && \
    scip-typescript --version && \
    scip-python --version && \
    scip-java --help | head -1 && \
    test -x ${SCIP_PHP_HOME}/bin/scip-php && \
    php --version

WORKDIR /app

# Copy spider_md binary and uv from build stages
COPY --from=rust-builder /build/target/release/spider_md /usr/local/bin/spider_md
COPY --from=ghcr.io/astral-sh/uv:latest@sha256:d1cbaeadc234fe19c0d93daabcf5e98738cd93c6d1dd4918ef6aa30735feb23a /uv /usr/local/bin/uv
RUN chmod +x /usr/local/bin/spider_md

# Copy workspace files
COPY pyproject.toml uv.lock ./
COPY packages/ packages/
COPY apps/worker/ apps/worker/
COPY apps/api/pyproject.toml apps/api/

# Sync the worker package and its workspace dependencies during the image build.
# The container must not resolve or install packages when it starts.
RUN uv sync --frozen --no-dev --package contextmine-worker && \
    useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /data/repos \
      /home/appuser/.npm \
      /home/appuser/.m2 \
      /home/appuser/.gradle \
      /home/appuser/.cache/pip \
      /home/appuser/.composer/cache && \
    chown -R appuser:appuser /app /data /home/appuser && \
    ln -s /usr/lib/jvm/java-21-openjdk-* /usr/lib/jvm/java-21-openjdk
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk
ENV PATH="${PATH}:${JAVA_HOME}/bin"
ENV NPM_CONFIG_CACHE=/home/appuser/.npm
ENV GRADLE_USER_HOME=/home/appuser/.gradle
ENV MAVEN_OPTS="-Dmaven.repo.local=/home/appuser/.m2/repository"
ENV COMPOSER_CACHE_DIR=/home/appuser/.composer/cache

USER appuser

# Set working directory to worker
WORKDIR /app/apps/worker

# Run from the environment materialized during the image build.
CMD ["/app/.venv/bin/python", "-m", "contextmine_worker.main"]
