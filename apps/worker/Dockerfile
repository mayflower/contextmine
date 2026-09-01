# Build the spider_md binary with the declared stable Rust toolchain.
FROM rust:1.98.0-slim@sha256:17d1ba895198f9934c6314ec5346a0d5115372f3243390c3d731e242f35c2f27 AS rust-builder

WORKDIR /build
COPY rust/spider_md/ ./
# The pinned Rust image records this Debian snapshot in debian.sources.
# hadolint ignore=DL3008
RUN sed -i \
      -e 's|^URIs: http://deb.debian.org/debian$|URIs: https://snapshot.debian.org/archive/debian/20260824T000000Z|' \
      -e 's|^URIs: http://deb.debian.org/debian-security$|URIs: https://snapshot.debian.org/archive/debian-security/20260824T000000Z|' \
      /etc/apt/sources.list.d/debian.sources \
    && printf 'Acquire::Check-Valid-Until "false";\n' > /etc/apt/apt.conf.d/99snapshot \
    && apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && cargo build --release

# Materialize the current Node 24 LTS toolchain for the analyzer stage.
FROM node:24.20.0-slim@sha256:ba849c60be29959425b8734d57b8b4b7d56f98edd9504c9af091d5281095a71e AS node-tools

ARG SCIP_TYPESCRIPT_VERSION=0.4.0
ARG SCIP_PYTHON_VERSION=0.6.6
ARG TYPESCRIPT_VERSION=7.0.2
RUN npm install -g \
      @sourcegraph/scip-typescript@${SCIP_TYPESCRIPT_VERSION} \
      @sourcegraph/scip-python@${SCIP_PYTHON_VERSION} \
      typescript@${TYPESCRIPT_VERSION} \
    && test "$(tsc --version)" = "Version ${TYPESCRIPT_VERSION}" \
    && scip-typescript --version \
    && scip-python --version

FROM composer:2.10.2@sha256:d020706319701a44468968321dccd0fce6620190159a7a9ec195d78e6e971c71 AS composer-tools

FROM python:3.14.7-slim@sha256:656d12e70054d5fda18a045e2494c96701e9792dd1445f95b3d038df954f57e9 AS analyzer

ARG SCIP_JAVA_VERSION=0.12.3
ARG SCIP_PHP_COMMIT=efaf87cd05cf174db8ac25ad4d07eb646d4883d1
ARG SCIP_PHP_REPO_URL=https://github.com/mayflower/scip-php.git
ARG COURSIER_LAUNCHERS_COMMIT=15f36c167c30be237105f923151adaf177e7ee61
ARG COURSIER_X86_64_SHA256=62b141b186e4dfdef03af64c36bdcdfeaab2df30de14950ca220e6e43e72c26b
ARG COURSIER_AARCH64_SHA256=11d980e5714abdebe93688f22cedfa8ac227a95f4de9f96c53c4779d2547cff7

# Keep installed analyzer packages on the same immutable Debian snapshot as
# the pinned Python base. This target is published as the sandbox snapshot input.
# hadolint ignore=DL3008
RUN sed -i \
      -e 's|^URIs: http://deb.debian.org/debian$|URIs: https://snapshot.debian.org/archive/debian/20260824T000000Z|' \
      -e 's|^URIs: http://deb.debian.org/debian-security$|URIs: https://snapshot.debian.org/archive/debian-security/20260824T000000Z|' \
      /etc/apt/sources.list.d/debian.sources \
    && printf 'Acquire::Check-Valid-Until "false";\n' > /etc/apt/apt.conf.d/99snapshot \
    && apt-get update && apt-get install -y --no-install-recommends \
    git \
    cloc \
    ca-certificates \
    libssl3 \
    libstdc++6 \
    curl \
    unzip \
    openjdk-21-jre-headless \
    php-cli \
    && rm -rf /var/lib/apt/lists/*
COPY --from=node-tools /usr/local/ /usr/local/
COPY --from=composer-tools /usr/bin/composer /usr/local/bin/composer

# Both downloaded tool inputs are immutable: Coursier is verified before use,
# and scip-php is checked out at an exact commit with its lockfile installed.
RUN ARCH=$(dpkg --print-architecture) \
    && if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then \
         CS_ARCH="aarch64"; CS_SHA256="${COURSIER_AARCH64_SHA256}"; \
       else \
         CS_ARCH="x86_64"; CS_SHA256="${COURSIER_X86_64_SHA256}"; \
       fi \
    && curl -fL \
         "https://raw.githubusercontent.com/coursier/launchers/${COURSIER_LAUNCHERS_COMMIT}/cs-${CS_ARCH}-pc-linux.gz" \
         -o /tmp/cs.gz \
    && printf '%s  %s\n' "${CS_SHA256}" /tmp/cs.gz > /tmp/cs.sha256 \
    && sha256sum -c /tmp/cs.sha256 \
    && gzip -d /tmp/cs.gz \
    && mv /tmp/cs /usr/local/bin/cs \
    && chmod +x /usr/local/bin/cs \
    && cs install --contrib scip-java:${SCIP_JAVA_VERSION} --install-dir /usr/local/bin \
    && git init /opt/scip-php \
    && git -C /opt/scip-php remote add origin "${SCIP_PHP_REPO_URL}" \
    && git -C /opt/scip-php fetch --depth 1 origin "${SCIP_PHP_COMMIT}" \
    && git -C /opt/scip-php checkout --detach "${SCIP_PHP_COMMIT}" \
    && composer install --working-dir=/opt/scip-php \
         --no-interaction --prefer-dist --no-progress --no-dev \
    && ln -s /opt/scip-php/bin/scip-php /usr/local/bin/scip-php \
    && test -f /opt/scip-php/vendor/autoload.php \
    && chmod -R 755 /opt/scip-php \
    && scip-typescript --version \
    && scip-python --version \
    && scip-java --help >/dev/null \
    && test -x /usr/local/bin/scip-php \
    && php --version

WORKDIR /app

COPY --from=rust-builder /build/target/release/spider_md /usr/local/bin/spider_md
COPY --from=ghcr.io/astral-sh/uv:0.12.5@sha256:e85be844203885286c60ffad8a858d48afb6c5a5c237ca0e67f12e74b8f174b1 /uv /usr/local/bin/uv
RUN chmod +x /usr/local/bin/spider_md

COPY pyproject.toml uv.lock ./
COPY packages/ packages/
COPY apps/worker/ apps/worker/
COPY apps/api/pyproject.toml apps/api/

RUN uv sync --frozen --no-dev --package contextmine-worker --extra analyzer \
    && useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /data/repos \
      /home/appuser/.npm \
      /home/appuser/.m2 \
      /home/appuser/.gradle \
      /home/appuser/.cache/pip \
      /home/appuser/.composer/cache \
    && chown -R appuser:appuser /app /data /home/appuser \
    && ln -s /usr/lib/jvm/java-21-openjdk-* /usr/lib/jvm/java-21-openjdk

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk
ENV PATH="${PATH}:${JAVA_HOME}/bin"
ENV NPM_CONFIG_CACHE=/home/appuser/.npm
ENV GRADLE_USER_HOME=/home/appuser/.gradle
ENV MAVEN_OPTS="-Dmaven.repo.local=/home/appuser/.m2/repository"
ENV COMPOSER_CACHE_DIR=/home/appuser/.composer/cache

USER appuser
WORKDIR /app/apps/worker

CMD ["/app/.venv/bin/python", "-m", "contextmine_worker.sandbox_analyzer"]

# The long-lived worker contains orchestration, crawling, and result-import
# code only. Repository clone/index/build tools stay in the analyzer target.
FROM python:3.14.7-slim@sha256:656d12e70054d5fda18a045e2494c96701e9792dd1445f95b3d038df954f57e9 AS worker

WORKDIR /app

# The development fallback performs repository clone/pull operations in this
# process. Production delegates analysis to the sandbox, but GitPython still
# imports as part of the shared orchestration flow.
# hadolint ignore=DL3008
RUN sed -i \
      -e 's|^URIs: http://deb.debian.org/debian$|URIs: https://snapshot.debian.org/archive/debian/20260824T000000Z|' \
      -e 's|^URIs: http://deb.debian.org/debian-security$|URIs: https://snapshot.debian.org/archive/debian-security/20260824T000000Z|' \
      /etc/apt/sources.list.d/debian.sources \
    && printf 'Acquire::Check-Valid-Until "false";\n' > /etc/apt/apt.conf.d/99snapshot \
    && apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && git --version

COPY --from=rust-builder /build/target/release/spider_md /usr/local/bin/spider_md
COPY --from=ghcr.io/astral-sh/uv:0.12.5@sha256:e85be844203885286c60ffad8a858d48afb6c5a5c237ca0e67f12e74b8f174b1 /uv /usr/local/bin/uv
RUN chmod +x /usr/local/bin/spider_md

COPY pyproject.toml uv.lock ./
COPY packages/ packages/
COPY apps/worker/ apps/worker/
COPY apps/api/pyproject.toml apps/api/

RUN uv sync --frozen --no-dev --package contextmine-worker \
    && useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app /home/appuser

USER appuser
WORKDIR /app/apps/worker

CMD ["/app/.venv/bin/python", "-m", "contextmine_worker.main"]
