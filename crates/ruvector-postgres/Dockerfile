# Multi-stage Dockerfile for ruvector-postgres extension
# Builds the extension and creates a PostgreSQL image with it installed
# v0.3.1: Fixes — Cypher self-reference, graph/RDF persistence, SONA dimension panic

# Build stage
# Using nightly Rust to support edition2024 crates in the registry
FROM rustlang/rust:nightly-bookworm-slim AS builder

# Install build dependencies including PostgreSQL 17 from PGDG
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    pkg-config \
    clang \
    libclang-dev \
    flex \
    bison \
    curl \
    ca-certificates \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Add PostgreSQL official apt repository
RUN curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc | gpg --dearmor -o /usr/share/keyrings/postgresql-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/postgresql-keyring.gpg] http://apt.postgresql.org/pub/repos/apt bookworm-pgdg main" > /etc/apt/sources.list.d/pgdg.list

# Install PostgreSQL 17 development packages
RUN apt-get update && apt-get install -y \
    postgresql-server-dev-17 \
    postgresql-17 \
    && rm -rf /var/lib/apt/lists/*

# Install cargo-pgrx
RUN cargo install cargo-pgrx --version 0.12.9 --locked

# Set up workspace root — dependency crates use workspace inheritance
WORKDIR /workspace

# Create a minimal workspace Cargo.toml so dependency crates can resolve
# workspace inheritance (edition.workspace, version.workspace, etc.)
RUN cat > /workspace/Cargo.toml << 'WORKSPACE_EOF'
[workspace]
members = [
    "crates/ruvector-postgres",
    "crates/ruvector-solver",
    "crates/ruvector-math",
    "crates/ruvector-attention",
    "crates/sona",
    "crates/ruvector-domain-expansion",
    "crates/ruvector-mincut-gated-transformer",
]
resolver = "2"

[workspace.package]
version = "2.0.4"
edition = "2021"
rust-version = "1.77"
license = "MIT"
authors = ["Ruvector Team"]
repository = "https://github.com/ruvnet/ruvector"

[workspace.dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
thiserror = "2.0"
rand = "0.8"
rand_distr = "0.4"
tracing = "0.1"
rayon = "1.10"
crossbeam = "0.8"
dashmap = "6.1"
parking_lot = "0.12"
once_cell = "1.20"
criterion = { version = "0.5", features = ["html_reports"] }
proptest = "1.5"
nalgebra = { version = "0.33", default-features = false, features = ["std"] }
ndarray = "0.16"
chrono = "0.4"
anyhow = "1.0"

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
strip = true
panic = "unwind"
WORKSPACE_EOF

# Copy ruvector-postgres source
COPY crates/ruvector-postgres/ /workspace/crates/ruvector-postgres/

# Copy dependency crates
COPY crates/ruvector-mincut-gated-transformer /workspace/crates/ruvector-mincut-gated-transformer/
COPY crates/ruvector-solver /workspace/crates/ruvector-solver/
COPY crates/ruvector-math /workspace/crates/ruvector-math/
COPY crates/ruvector-attention /workspace/crates/ruvector-attention/
COPY crates/sona /workspace/crates/sona/
COPY crates/ruvector-domain-expansion /workspace/crates/ruvector-domain-expansion/

# Copy rvf crates (path deps of ruvector-domain-expansion)
COPY crates/rvf/rvf-types /workspace/crates/rvf/rvf-types/
COPY crates/rvf/rvf-wire /workspace/crates/rvf/rvf-wire/
COPY crates/rvf/rvf-crypto /workspace/crates/rvf/rvf-crypto/

# Copy the workspace Cargo.lock to pin dependency versions
COPY Cargo.lock /workspace/Cargo.lock

WORKDIR /workspace/crates/ruvector-postgres

# Initialize pgrx with system PostgreSQL
RUN cargo pgrx init --pg17=/usr/lib/postgresql/17/bin/pg_config

# Pre-fetch dependencies to lock versions and warm the registry cache
RUN cargo fetch

# Configure cargo to avoid sparse registry issues with edition2024 crates
# This uses the git protocol instead of sparse which skips problematic index entries
ENV CARGO_REGISTRIES_CRATES_IO_PROTOCOL=git

# Build the extension with all features including v0.3 modules
# graph-complete includes: graph, hyperbolic, sparse
RUN cargo pgrx package --features "pg17 index-all quant-all graph-complete embeddings gated-transformer analytics-complete attention-extended sona-learning domain-expansion"

# Build the model downloader binary
RUN cargo build --release --bin download-models --features "embeddings"

# Set cache path and download embedding models
# FASTEMBED_CACHE_DIR is the correct env var for fastembed-rs
ENV FASTEMBED_CACHE_DIR=/opt/ruvector/models
RUN mkdir -p /opt/ruvector/models && \
    ./target/release/download-models && \
    echo "Model cache size: $(du -sh /opt/ruvector/models)" && \
    ls -la /opt/ruvector/models/

# Copy all SQL schema files (control file default_version=0.3.0 selects the right one)
RUN for f in /workspace/crates/ruvector-postgres/sql/ruvector--*.sql; do \
        cp "$f" /workspace/target/release/ruvector-pg17/usr/share/postgresql/17/extension/ ; \
    done && \
    echo "SQL schemas copied:" && \
    ls -1 /workspace/target/release/ruvector-pg17/usr/share/postgresql/17/extension/ruvector--*.sql && \
    echo "v0.3.0 function count: $(grep -c 'CREATE FUNCTION\|CREATE OR REPLACE FUNCTION' /workspace/target/release/ruvector-pg17/usr/share/postgresql/17/extension/ruvector--0.3.0.sql)"

# Verify the extension files are complete
RUN ls -la /workspace/target/release/ruvector-pg17/usr/share/postgresql/17/extension/ && \
    echo "=== Extension control ===" && \
    cat /workspace/target/release/ruvector-pg17/usr/share/postgresql/17/extension/ruvector.control

# Runtime stage
FROM postgres:17-bookworm

# Labels
LABEL maintainer="ruvector team <info@ruv.io>"
LABEL description="PostgreSQL with ruvector extension - high-performance vector database with 270+ SQL functions, Graph/Cypher/SPARQL, GNN, hybrid search, multi-tenancy, self-healing, SONA self-learning, and local embeddings"
LABEL version="0.3.1"
LABEL org.opencontainers.image.title="ruvector-postgres"
LABEL org.opencontainers.image.version="0.3.1"
LABEL org.opencontainers.image.vendor="ruv.io"
LABEL org.opencontainers.image.source="https://github.com/ruvnet/ruvector"
LABEL org.opencontainers.image.description="Drop-in pgvector replacement with SIMD, Flash Attention, GNN, Cypher, SPARQL, hybrid search, multi-tenancy, self-healing, and SONA"

# Set embedding model cache path - models are pre-downloaded during build
# FASTEMBED_CACHE_DIR is the correct env var for fastembed-rs
ENV FASTEMBED_CACHE_DIR=/opt/ruvector/models

# Copy pre-downloaded embedding models from builder
COPY --from=builder /opt/ruvector/models /opt/ruvector/models

# Copy the built extension from builder
# Note: In a workspace, target/ is at the workspace root /workspace/target/
COPY --from=builder /workspace/target/release/ruvector-pg17/usr/share/postgresql/17/extension/* \
    /usr/share/postgresql/17/extension/
COPY --from=builder /workspace/target/release/ruvector-pg17/usr/lib/postgresql/17/lib/* \
    /usr/lib/postgresql/17/lib/

# Add initialization scripts
RUN mkdir -p /docker-entrypoint-initdb.d

# Copy the full initialization script with extension creation, role setup, and tests
COPY --from=builder /workspace/crates/ruvector-postgres/docker/init.sql /docker-entrypoint-initdb.d/01-init.sql

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD pg_isready -U postgres || exit 1

# Expose PostgreSQL port
EXPOSE 5432

# Use the default PostgreSQL entrypoint
CMD ["postgres"]
