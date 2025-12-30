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

# Install git for cloning repos and OpenSSL for spider_md
RUN apt-get update && apt-get install -y \
    git \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

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
RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /data/repos && \
    chown -R appuser:appuser /app /data
USER appuser

# Set working directory to worker
WORKDIR /app/apps/worker

# Run the worker
CMD ["uv", "run", "python", "-m", "contextmine_worker.main"]
