# Deployment Guide

How to ship `ruvllm` to a server, into a Docker container, and onto an
ESP32 microcontroller.

## Targets at a Glance

| Target | Binary | Required features | Notes |
|---|---|---|---|
| Workstation REPL | `ruvllm-demo` | none | Mock inference, fastest to start |
| HTTP server (host) | `ruvllm-server` | `server` | Add `real-inference` for a real model |
| Bench harness | `ruvllm-benchmark-suite` | none | Reproducible Criterion run |
| Pretrain pipeline | `ruvllm-pretrain` | none | Offline; not deployed |
| HF export | `ruvllm-export` | `hf-export` | Tooling, not a service |
| ESP32 firmware | `esp32-flash` | (see below) | Separate sub-crate |

## Server Deployment

### Build

```sh
# Minimum: server + storage + metrics
cargo build --release --features server

# Recommended for production: real inference, parallel kernels, all opt-ins
cargo build --release --features "server,real-inference,parallel,metrics,storage"

# Everything (slower compile, useful for staging)
cargo build --release --features full
```

The release binary lands at `target/release/ruvllm-server`.

### Configuration

Copy and edit the example TOML:

```sh
cp config/example.toml /etc/ruvllm/config.toml
$EDITOR /etc/ruvllm/config.toml
```

The eight sections (`[system]`, `[embedding]`, `[memory]`, `[router]`,
`[inference]`, `[learning]`, plus runtime-specifics) are documented in
[Configuration Guide](configuration-guide.md). Pay particular attention to:

- `[system].data_dir` — needs to be writable by the service user.
- `[system].max_memory_mb` — set to ~80 % of available RAM.
- `[system].max_concurrent_requests` — start at 10, raise after profiling.
- `[memory].db_path` — separate disk from logs if possible.

### Run

```sh
./target/release/ruvllm-server --config /etc/ruvllm/config.toml
```

The server exposes the endpoints documented in [API Reference](api-reference.md).
Health check: `curl localhost:PORT/health`.

### systemd Unit (example)

Save as `/etc/systemd/system/ruvllm.service`:

```ini
[Unit]
Description=RuvLLM orchestrator
After=network.target

[Service]
Type=simple
User=ruvllm
Group=ruvllm
ExecStart=/usr/local/bin/ruvllm-server --config /etc/ruvllm/config.toml
Restart=on-failure
RestartSec=5
LimitNOFILE=65536

# Sandboxing
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/ruvllm /var/log/ruvllm
NoNewPrivileges=true
PrivateTmp=true

# Resource limits — match [system].max_memory_mb
MemoryMax=10G
TasksMax=4096

[Install]
WantedBy=multi-user.target
```

Enable and start:

```sh
sudo systemctl daemon-reload
sudo systemctl enable --now ruvllm.service
journalctl -u ruvllm -f
```

### Reverse Proxy

The server speaks plain HTTP. For TLS, terminate at nginx/Caddy/Traefik in
front of it. The endpoints under `/query` and `/feedback` are POST with JSON
bodies — no special proxy configuration is needed beyond a generous request
size limit if you send large contexts.

### Observability

With the `metrics` feature on (default), the server emits Prometheus metrics.
Scrape them from your monitoring stack and graph at minimum:

- p50 / p95 / p99 of `/query` latency.
- HNSW search count and median search-time.
- Router confidence histogram.
- Replay buffer fill rate.
- EWC consolidation runs (should fire about every `training_interval_ms`).

## Docker

The reference Dockerfile lives in `esp32-flash/Dockerfile` for the firmware
build, but a host-side image follows the standard Rust pattern. A minimal
Dockerfile for the server:

```dockerfile
FROM rust:1.81 AS build
WORKDIR /src
COPY . .
RUN cargo build --release --features "server,real-inference,parallel,metrics,storage" \
    --bin ruvllm-server

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libssl3 && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/ruvllm-server /usr/local/bin/
COPY --from=build /src/config/example.toml /etc/ruvllm/config.toml
EXPOSE 3000
ENTRYPOINT ["ruvllm-server"]
CMD ["--config", "/etc/ruvllm/config.toml"]
```

Build and run:

```sh
docker build -t ruvllm-server .
docker run --rm -p 3000:3000 -v /var/lib/ruvllm:/var/lib/ruvllm ruvllm-server
```

Mount `[memory].db_path`'s parent directory as a volume so the HNSW store
survives restarts.

## Edge Deployment — ESP32

The `esp32/` and `esp32-flash/` sub-crates ship the same orchestrator concepts
in a `no_std` profile sized for ESP32-class microcontrollers (320–512 KB
SRAM). Quantization is mandatory: pick INT8, INT4, or binary based on the
target and accuracy budget.

### Toolchain

The ESP32 build uses the Xtensa toolchain via `espup`:

```sh
cargo install espup espflash
espup install
. $HOME/export-esp.sh   # adds the toolchain to PATH
```

Verify:

```sh
rustc +esp --version
espflash --version
```

### Build the Firmware

The `esp32-flash/` directory has a `Makefile` with the canonical commands.
Common targets:

```sh
cd esp32-flash/

# Build with INT8 quantization (default for ESP32-S3 + PSRAM)
make build FEATURES=q8

# Smaller variant for plain ESP32 (520 KB SRAM)
make build FEATURES=q4

# Tightest fit, accuracy permitting
make build FEATURES=binary

# Federated cluster member
make build FEATURES="q8,federation"

# ESP32-S3 with vector instructions
make build FEATURES="q8,esp32s3-simd"
```

The build target is `xtensa-esp32-espidf`. The firmware artifact lands in
`target/xtensa-esp32-espidf/release/`.

### Flash a Single Chip

```sh
# From esp32-flash/
make flash PORT=/dev/cu.usbserial-XXXX

# Or directly:
espflash flash --monitor target/xtensa-esp32-espidf/release/esp32-flash
```

The `install.sh` helper in `esp32-flash/` wraps the toolchain check, build,
and flash into a single step for first-time setup.

### Cluster Flashing

`esp32-flash/cluster-flash.sh` flashes a fleet of chips in parallel. It
discovers attached devices, builds once, and dispatches `espflash` against
each port. Useful for federated deployments where many ESP32s join a
training mesh:

```sh
cd esp32-flash/
./cluster-flash.sh
```

The script honors environment variables for the feature set and the build
profile; read the script's header for the full list.

### Dockerized ESP32 Build

Cross-compiling the Xtensa toolchain on macOS or Linux can be brittle.
`esp32-flash/Dockerfile` provides a reproducible build environment with
the toolchain pre-installed:

```sh
cd esp32-flash/
docker build -t ruvllm-esp32-build .
docker run --rm -v "$PWD":/work -w /work ruvllm-esp32-build \
    make build FEATURES=q8
```

Flashing still happens on the host (the container does not have access to
USB serial devices unless you pass `--device`).

### Memory Budget on ESP32

| Quantization | Approx. weight size | Fits |
|---|---|---|
| `q8` (INT8) | ~M parameters in 100s of KB | ESP32-S3 with PSRAM |
| `q4` (INT4) | ~halves `q8` | Plain ESP32 |
| `binary` (1-bit XNOR) | ~8× smaller than `q8` | Tight RAM, accuracy-tolerant tasks |

The `esp32-std` feature lets you build the same library against the host
target for unit testing without flashing.

### Federation

When the `federation` feature is on, ESP32 nodes can share weight deltas
peer-to-peer without a central coordinator. Pair this with `q8` for the
practical case. See `esp32/` source for the wire format (`postcard`-encoded).

## Pre-Flight Checklist

Before promoting a build to production:

- [ ] `cargo test` passes (unit + integration).
- [ ] `cargo bench` shows no regression on `pipeline.rs`, `router.rs`,
      `memory.rs`, `attention.rs`, `sona_bench.rs`. See
      [Testing Guide](testing-guide.md).
- [ ] `cargo build --release --features "server,real-inference,parallel"` is
      green.
- [ ] `config.toml` is reviewed against
      [Configuration Guide](configuration-guide.md).
- [ ] systemd unit (or container orchestrator manifest) sets memory limits
      consistent with `[system].max_memory_mb`.
- [ ] Prometheus scrape target is configured.
- [ ] Backup plan for `[memory].db_path` (the HNSW store).

## Rollback

The server is stateless apart from the HNSW store at `[memory].db_path`.
Rollback is a binary swap plus a systemd restart. The store format is
backwards-compatible across patch releases; a major version bump will
document any migration step explicitly.

## See also

- [Configuration Guide](configuration-guide.md)
- [API Reference](api-reference.md)
- [Testing Guide](testing-guide.md)
- [Codebase Summary](codebase-summary.md)
