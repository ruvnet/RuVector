# RuVector-Postgres Development & Testing Dockerfile
# Multi-stage build with PostgreSQL version support (14-17)
# Default: PostgreSQL 17 (latest with pgrx 0.12 support)
# Note: PostgreSQL 18 requires pgrx 0.15.0+ (planned for future release)

ARG PG_VERSION=17

# ============================================================================
# Builder Stage
# ============================================================================
FROM rust:1.83-bookworm AS builder

ARG PG_VERSION

# Add PostgreSQL APT repository
RUN sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt bookworm-pgdg main" > /etc/apt/sources.list.d/pgdg.list' && \
    wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add -

# Install PostgreSQL development dependencies for specified version
RUN apt-get update && apt-get install -y \
    postgresql-${PG_VERSION} \
    postgresql-server-dev-${PG_VERSION} \
    libclang-dev \
    clang \
    pkg-config \
    libssl-dev \
    cmake \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install pgrx (compatible with pgrx = "0.12" in Cargo.toml)
RUN cargo install cargo-pgrx --version 0.12.6 --locked

# Initialize pgrx for the specified PostgreSQL version
RUN cargo pgrx init --pg${PG_VERSION} /usr/lib/postgresql/${PG_VERSION}/bin/pg_config

# Set PGRX environment for consistent builds
ENV PGRX_PG_CONFIG_PATH=/usr/lib/postgresql/${PG_VERSION}/bin/pg_config
ENV PGRX_HOME=/root/.pgrx

# Set working directory - use /build to avoid any parent Cargo.toml issues
WORKDIR /build/ruvector-postgres

# Copy only the postgres crate - this is a standalone crate with no workspace dependencies
COPY crates/ruvector-postgres/Cargo.toml ./
COPY crates/ruvector-postgres/build.rs ./
COPY crates/ruvector-postgres/ruvector.control ./
COPY crates/ruvector-postgres/src ./src/
COPY crates/ruvector-postgres/sql ./sql/
COPY crates/ruvector-postgres/benches ./benches/

# Build the extension with the specified PG version feature
RUN cargo pgrx package \
    --pg-config /usr/lib/postgresql/${PG_VERSION}/bin/pg_config \
    --features pg${PG_VERSION}

# pgrx only generates .control and .so - copy pre-written SQL file
RUN if [ ! -f target/release/ruvector-pg${PG_VERSION}/usr/share/postgresql/${PG_VERSION}/extension/ruvector--0.1.0.sql ]; then \
        echo "Copying pre-written SQL file..." && \
        cp sql/ruvector--0.1.0.sql target/release/ruvector-pg${PG_VERSION}/usr/share/postgresql/${PG_VERSION}/extension/; \
    fi

# ============================================================================
# Runtime Stage
# ============================================================================
FROM postgres:${PG_VERSION}-bookworm

ARG PG_VERSION

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy built extension from builder
COPY --from=builder /build/ruvector-postgres/target/release/ruvector-pg${PG_VERSION}/usr/share/postgresql/${PG_VERSION}/extension/* /usr/share/postgresql/${PG_VERSION}/extension/
COPY --from=builder /build/ruvector-postgres/target/release/ruvector-pg${PG_VERSION}/usr/lib/postgresql/${PG_VERSION}/lib/* /usr/lib/postgresql/${PG_VERSION}/lib/

# Copy initialization script with proper permissions
COPY --chmod=644 crates/ruvector-postgres/docker/init.sql /docker-entrypoint-initdb.d/

# Set environment variables
ENV POSTGRES_USER=ruvector
ENV POSTGRES_PASSWORD=ruvector
ENV POSTGRES_DB=ruvector_test

# Labels for version tracking
LABEL org.opencontainers.image.title="RuVector PostgreSQL Extension"
LABEL org.opencontainers.image.description="High-performance vector database extension for PostgreSQL"
LABEL org.opencontainers.image.version="0.2.5"
LABEL org.opencontainers.image.vendor="ruv.io"
LABEL ruvector.pg.version="${PG_VERSION}"

# Health check
HEALTHCHECK --interval=5s --timeout=5s --start-period=10s --retries=5 \
    CMD pg_isready -U $POSTGRES_USER -d $POSTGRES_DB || exit 1

EXPOSE 5432
