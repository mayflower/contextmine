# Build stage for Rust spider_md binary
# spider_md dependencies currently require rustc >= 1.88.
FROM rust:1.88-slim AS rust-builder

WORKDIR /build

# Install build dependencies, copy source, and build release binary
COPY rust/spider_md/ ./
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && cargo build --release

# Python worker stage
FROM python:3.12-slim

# SCIP tool versions (pinned for reproducibility)
ARG SCIP_TYPESCRIPT_VERSION=0.3.14
ARG SCIP_PYTHON_VERSION=0.6.6
ARG SCIP_JAVA_VERSION=0.10.3
ARG SCIP_PHP_VERSION=dev-main
ARG SCIP_PHP_REPO_URL=https://github.com/mayflower/scip-php.git
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
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION}.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && corepack enable \
    && npm install -g \
       @sourcegraph/scip-typescript@${SCIP_TYPESCRIPT_VERSION} \
       @sourcegraph/scip-python@${SCIP_PYTHON_VERSION}

# Install Coursier/scip-java, Composer/scip-php, and verify SCIP tools
ENV COMPOSER_HOME=/opt/composer
ENV PATH="${PATH}:${COMPOSER_HOME}/vendor/bin"
RUN ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then \
      CS_URL="https://github.com/coursier/launchers/raw/master/cs-aarch64-pc-linux.gz"; \
    else \
      CS_URL="https://github.com/coursier/launchers/raw/master/cs-x86_64-pc-linux.gz"; \
    fi && \
    curl -fL "$CS_URL" | gzip -d > /usr/local/bin/cs && \
    chmod +x /usr/local/bin/cs && \
    /usr/local/bin/cs install --contrib \
      scip-java:${SCIP_JAVA_VERSION} \
      --install-dir /usr/local/bin && \
    curl -sS https://getcomposer.org/installer \
      | php -- --install-dir=/usr/local/bin --filename=composer && \
    composer global config repositories.scip-php vcs ${SCIP_PHP_REPO_URL} && \
    composer global require davidrjenni/scip-php:${SCIP_PHP_VERSION} && \
    composer install \
      --working-dir=${COMPOSER_HOME}/vendor/davidrjenni/scip-php \
      --no-interaction --prefer-dist --no-progress --no-dev && \
    test -f ${COMPOSER_HOME}/vendor/davidrjenni/scip-php/vendor/autoload.php && \
    chmod -R 755 ${COMPOSER_HOME} && \
    scip-typescript --version && \
    scip-python --version && \
    scip-java --help | head -1 && \
    test -x ${COMPOSER_HOME}/vendor/bin/scip-php && \
    php --version

WORKDIR /app

# Copy spider_md binary and uv from build stages
COPY --from=rust-builder /build/target/release/spider_md /usr/local/bin/spider_md
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN chmod +x /usr/local/bin/spider_md

# Copy workspace files
COPY pyproject.toml uv.lock ./
COPY packages/ packages/
COPY apps/worker/ apps/worker/
COPY apps/api/pyproject.toml apps/api/

# Sync dependencies, create non-root user and data/cache directories
RUN uv sync --frozen --no-dev && \
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

# Run the worker
CMD ["uv", "run", "python", "-m", "contextmine_worker.main"]
