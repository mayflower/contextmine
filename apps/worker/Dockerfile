# Build stage for Rust spider_md binary
FROM rust:latest AS rust-builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Rust source
COPY rust/spider_md/ ./

# Build release binary
RUN cargo build --release

# Python worker stage
FROM python:3.12-slim

# SCIP tool versions (pinned for reproducibility)
ARG SCIP_TYPESCRIPT_VERSION=0.3.14
ARG SCIP_PYTHON_VERSION=0.6.6
ARG SCIP_JAVA_VERSION=0.10.3
ARG SCIP_PHP_VERSION=dev-lint-php8.4
ARG NODE_VERSION=20

# Install system dependencies
# - git: for cloning repos
# - cloc: language census for multi-language detection
# - ca-certificates, libssl3: for spider_md and HTTPS
# - curl: for installing Node.js and Coursier
# - openjdk-21-jre-headless: for scip-java
# - php-cli, composer: for scip-php
RUN apt-get update && apt-get install -y \
    git \
    cloc \
    ca-certificates \
    libssl3 \
    curl \
    gnupg \
    openjdk-21-jre-headless \
    php-cli \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js (for scip-python, scip-typescript)
RUN curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION}.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/* && \
    corepack enable

# Install npm-based SCIP indexers globally
RUN npm install -g \
    @sourcegraph/scip-typescript@${SCIP_TYPESCRIPT_VERSION} \
    @sourcegraph/scip-python@${SCIP_PYTHON_VERSION}

# Install Coursier and scip-java (--contrib required for scip-java channel)
RUN curl -fL https://github.com/coursier/launchers/raw/master/cs-x86_64-pc-linux.gz | \
    gzip -d > /usr/local/bin/cs && \
    chmod +x /usr/local/bin/cs && \
    /usr/local/bin/cs install --contrib scip-java:${SCIP_JAVA_VERSION} --install-dir /usr/local/bin

# Install Composer
RUN curl -sS https://getcomposer.org/installer | php -- --install-dir=/usr/local/bin --filename=composer

# Install scip-php via Composer global
ENV COMPOSER_HOME=/opt/composer
ENV PATH="${PATH}:${COMPOSER_HOME}/vendor/bin"
RUN composer global require davidrjenni/scip-php:${SCIP_PHP_VERSION} && \
    composer install --working-dir=${COMPOSER_HOME}/vendor/davidrjenni/scip-php --no-interaction --prefer-dist --no-progress --no-dev && \
    test -f ${COMPOSER_HOME}/vendor/davidrjenni/scip-php/vendor/autoload.php && \
    chmod -R 755 ${COMPOSER_HOME}

# Verify SCIP tools are installed (smoke test)
RUN scip-typescript --version && \
    scip-python --version && \
    scip-java --help | head -1 && \
    test -x ${COMPOSER_HOME}/vendor/bin/scip-php && \
    php --version

WORKDIR /app

# Copy spider_md binary from Rust build stage
COPY --from=rust-builder /build/target/release/spider_md /usr/local/bin/spider_md
RUN chmod +x /usr/local/bin/spider_md

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy workspace files
COPY pyproject.toml uv.lock ./
COPY packages/ packages/
COPY apps/worker/ apps/worker/
COPY apps/api/pyproject.toml apps/api/

# Sync dependencies
RUN uv sync --frozen --no-dev

# Create non-root user and data directory
# Also create cache directories for build tools
RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /data/repos && \
    mkdir -p /home/appuser/.npm && \
    mkdir -p /home/appuser/.m2 && \
    mkdir -p /home/appuser/.gradle && \
    mkdir -p /home/appuser/.cache/pip && \
    mkdir -p /home/appuser/.composer/cache && \
    chown -R appuser:appuser /app /data /home/appuser

# Set environment for SCIP tools
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH="${PATH}:${JAVA_HOME}/bin"
ENV NPM_CONFIG_CACHE=/home/appuser/.npm
ENV GRADLE_USER_HOME=/home/appuser/.gradle
ENV MAVEN_OPTS="-Dmaven.repo.local=/home/appuser/.m2/repository"
ENV COMPOSER_CACHE_DIR=/home/appuser/.composer/cache

USER appuser

# Set working directory to worker
WORKDIR /app/apps/worker

# Run the worker
CMD ["uv", "run", "python", "-m", "contextmine_worker.main"]
