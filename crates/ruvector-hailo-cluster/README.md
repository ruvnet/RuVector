# ruvector-hailo-cluster

Multi-Pi cluster coordinator for ruvector's Hailo-8 NPU embedding workers.
Implements P2C+EWMA load balancing, fingerprint enforcement, optional
in-process caching, and Tailscale-tag-based discovery.

> **Status:** library + 8 CLI binaries production-shaped; **204 tests**
> passing across lib unit / cluster integration / 7 CLI integration /
> 7 doctest suites (iter 253).
>
> **NPU acceleration is the production default** as of iter 163
> (ADR-176). `cargo build --release --features hailo,cpu-fallback
> --bin ruvector-hailo-worker` produces a worker that auto-detects
> `model.hef` in the model dir and runs encoder forward pass on the
> Hailo-8 NPU. Measured **70.6 embeds/sec/worker** on cognitum-v0
> (Pi 5 + AI HAT+, iter-227 baseline reverified post-iter-237 deploy);
> ~10× over cpu-fallback baseline. iter-237's pool=2 default also
> cuts p50 latency by 23% under multi-bridge concurrent load.
>
> **cpu-fallback remains the automatic failover.** When `model.hef`
> isn't present, the worker uses host-CPU candle BERT-6 (~7
> embeds/sec/worker on Pi 5, ~3-4 on Pi 4). Hardware-agnostic across
> aarch64; no AI HAT+ required (see [ADR-177][adr177]).
> `deploy/download-cpu-fallback-model.sh` fetches the safetensors
> trio with sha256 pinning.

## Operator QUICKSTART (iter 171)

Three deploy paths, pick whichever matches your hardware:

### A — Pi 5 + AI HAT+ (NPU, fastest)

```bash
# On the Pi:
bash deploy/download-cpu-fallback-model.sh /var/lib/ruvector-hailo/models/all-minilm-l6-v2
bash deploy/download-encoder-hef.sh        /var/lib/ruvector-hailo/models/all-minilm-l6-v2
cargo build --release --features hailo,cpu-fallback \
    --bin ruvector-hailo-worker \
    --manifest-path crates/ruvector-hailo-cluster/Cargo.toml
sudo bash deploy/install.sh \
    crates/ruvector-hailo-cluster/target/release/ruvector-hailo-worker \
    /var/lib/ruvector-hailo/models/all-minilm-l6-v2
sudo systemctl start ruvector-hailo-worker
journalctl -u ruvector-hailo-worker -f          # watch self-test fire
```

Expected: 67 embeds/sec/worker, 57 ms p50.

### B — Pi 4 / Pi 5 without AI HAT+ (cpu-fallback)

Same workflow but skip the HEF download and build with `cpu-fallback`
only:

```bash
# On x86 dev box, cross-compile:
bash deploy/cross-build-bridges.sh --with-worker
scp target/aarch64-unknown-linux-gnu/release/ruvector-hailo-worker pi:/tmp/

# On the Pi:
bash deploy/download-cpu-fallback-model.sh /var/lib/ruvector-hailo/models/all-minilm-l6-v2
sudo bash deploy/install.sh /tmp/ruvector-hailo-worker /var/lib/ruvector-hailo/models/all-minilm-l6-v2
sudo systemctl start ruvector-hailo-worker
```

Expected: 7 embeds/sec/worker on Pi 5, ~3-4 on Pi 4. See
[ADR-177][adr177] for the full Pi 4 deploy story.

### C — Local dev / x86 (cpu-fallback)

```bash
bash deploy/download-cpu-fallback-model.sh /tmp/cpu-fallback-test
RUVECTOR_WORKER_BIND=127.0.0.1:50051 \
RUVECTOR_MODEL_DIR=/tmp/cpu-fallback-test \
RUVECTOR_CPU_FALLBACK_POOL_SIZE=4 \
cargo run --release --features cpu-fallback --bin ruvector-hailo-worker \
    --manifest-path crates/ruvector-hailo-cluster/Cargo.toml
```

Expected: 45 embeds/sec/worker on AVX2 release.

### Verifying the deploy

After `systemctl start`, journalctl should show:

```text
ruvector-hailo-worker starting bind=0.0.0.0:50051
model fingerprint computed fingerprint=...
startup self-test embed ok dim=384 sim_close=... sim_far=...
ruvector-hailo-worker serving addr=0.0.0.0:50051
```

The `sim_close > sim_far` line is the iter-167 ranking gate — if it
fails the worker exits non-zero.

Test from another host:

```bash
echo "hello world" | ./target/release/ruvector-hailo-embed \
    --workers <pi-host>:50051 --allow-empty-fingerprint
```

[adr177]: ../../docs/adr/ADR-177-pi4-no-hat-deploy.md

[adr167]: ../../docs/adr/ADR-167-ruvector-hailo-npu-embedding-backend.md
[adr168]: ../../docs/adr/ADR-168-ruvector-hailo-cluster-cli-surface.md
[adr169]: ../../docs/adr/ADR-169-ruvector-hailo-cluster-cache-architecture.md
[adr170]: ../../docs/adr/ADR-170-ruvector-hailo-cluster-tracing-correlation.md

## Security & DoS hardening

The worker ships with eight layered gates on the gRPC surface, all
env-tunable from `/etc/ruvector-hailo.env`. Defaults are production-
safe; floors are baked in so a misconfig can't lock out legitimate
traffic. Each gate logs its active value at worker startup:

| Iter | Env var | Default | Floor | What it bounds |
|------|---------|---------|-------|----------------|
| 180 | `RUVECTOR_MAX_REQUEST_BYTES` | 65536 | 4 KB | Per-RPC decode budget. ~63× tighter than tonic's ~4 MB transport default. |
| 190 | `RUVECTOR_MAX_RESPONSE_BYTES` | 16384 | 4 KB | Per-RPC encode budget. Defense-in-depth on the response side. |
| 181 | `RUVECTOR_MAX_CONCURRENT_STREAMS` | 256 | 8 | HTTP/2 SETTINGS_MAX_CONCURRENT_STREAMS — caps single-connection stream flooding. |
| 182 | `RUVECTOR_REQUEST_TIMEOUT_SECS` | 30 | 2 | Per-RPC handler timeout — bounds slow-loris. |
| 183 | `RUVECTOR_MAX_PENDING_RESETS` | 32 | 8 | CVE-2023-44487 rapid-reset cap. |
| 184 | `RUVECTOR_HTTP2_KEEPALIVE_SECS` | 60 | 10 (0=off) | HTTP/2 ping interval — reclaims half-closed TCP state. |
| 199 | `RUVECTOR_MAX_BATCH_SIZE` | 256 | 1 | `embed_stream` batch length — bounds per-RPC NPU work. |
| 191 | `RUVECTOR_NPU_VSTREAM_TIMEOUT_MS` | 2000 | 100 | HailoRT FFI timeout — wedged-NPU recovery. |

Plus three orthogonal hardening tracks:

* **HEF integrity** (iter 174): `RUVECTOR_HEF_SHA256=<hex>` pins the
  HEF at boot. Worker streams sha256 over `model.hef` and refuses to
  start on mismatch. ~16 ms cost on Pi 5 NEON.
* **Per-peer rate limit** (iter 104/200): `RUVECTOR_RATE_LIMIT_RPS`
  + `RUVECTOR_RATE_LIMIT_BURST`. iter 200 made the limiter debit per
  item on `embed_stream` so a 1 RPS peer can't extract
  `max_batch_size` embeds/sec via batched RPCs.
* **TLS + mTLS** (iter 99/100): `RUVECTOR_TLS_CERT` + `RUVECTOR_TLS_KEY`
  for HTTPS, optional `RUVECTOR_TLS_CLIENT_CA` for mTLS. Client tools
  (`embed`, `bench`, `stats`) carry symmetric `--tls-ca` /
  `--tls-domain` / `--tls-client-cert` / `--tls-client-key` flags
  (iter 187/188/189) under `--features tls`.

Shutdown hardening (iter 185): the worker exits via
`mem::forget` + `process::exit(0)` because every clean HailoRT
teardown SEGV'd in the upstream library; the OS reaps fds + mmaps
either way. Operators see `status=0/SUCCESS` instead of
`status=11/SEGV`. `RUVECTOR_SHUTDOWN_FORCE_CLEAN=1` reserved for a
future HailoRT release that fixes the upstream bug.

systemd units (iter 205) cap `Restart=on-failure` at
`StartLimitBurst=5` per `StartLimitIntervalSec=60` so a unit that
fails every startup parks in `failed` state instead of cycling
forever.

**Client-side tunables (iter 208).** The cluster client tools
(`bench`, `embed`, `stats`, the three bridges) read two extra env
vars at `GrpcTransport::new()` construction time. These are
client-side, not worker-side, so they don't appear in the worker
`.env`:

| Env var | Default | Floor | What it bounds |
|---|---|---|---|
| `RUVECTOR_CLIENT_CONNECT_TIMEOUT_MS` | 5000 | 100 | TCP connect deadline |
| `RUVECTOR_CLIENT_RPC_TIMEOUT_MS` | 10000 | 100 | Per-RPC client deadline |

The 10s RPC default exists because iter-199's batch cap of 256 means
a single legit `embed_stream` RPC can use ~3.6s of NPU time at b=256;
the prior 2s default would have guaranteed `deadline_exceeded` on
every large-batch RPC. The 10s cap stays well under iter-182's 30s
server-side `request_timeout`, so a real wedged worker still
surfaces to the client within the worker's own bound.

See `deploy/ruvector-hailo.env.example` for the full operator-tunable
set with rationale per knob.

## What it ships

### Library

`HailoClusterEmbedder` — the coordinator. Distributes embed requests
across N Pi 5 + Hailo-8 workers via a transport-agnostic dispatch loop.

```rust
use std::sync::Arc;
use ruvector_hailo_cluster::{
    GrpcTransport, HailoClusterEmbedder, WorkerEndpoint,
    transport::EmbeddingTransport,
};

let workers = vec![
    WorkerEndpoint::new("pi-a", "100.77.59.83:50051"),
    WorkerEndpoint::new("pi-b", "100.77.59.84:50051"),
];
let transport: Arc<dyn EmbeddingTransport + Send + Sync> =
    Arc::new(GrpcTransport::new()?);

let cluster = HailoClusterEmbedder::new(workers, transport, 384, "fp:abc")?
    .with_cache(4096);

cluster.validate_fleet()?;                         // boot-time integrity check
let v = cluster.embed_one_blocking("hello world")?;
```

8 entry-point methods cover the full sync × async × single × batch ×
random-id × caller-id matrix:

```rust
embed_one_blocking(text)                            -> Vec<f32>
embed_one(self, text).await                         -> Vec<f32>
embed_batch_blocking(texts)                         -> Vec<Vec<f32>>
embed_batch(self, texts).await                      -> Vec<Vec<f32>>
embed_one_blocking_with_request_id(text, id)        -> Vec<f32>
embed_one_with_request_id(self, text, id).await     -> Vec<f32>
embed_batch_blocking_with_request_id(texts, id)     -> Vec<Vec<f32>>
embed_batch_with_request_id(self, texts, id).await  -> Vec<Vec<f32>>
```

### CLI binaries

| Binary | Purpose |
|---|---|
| `ruvector-hailo-worker` | Hailo NPU server (runs on each Pi) |
| `ruvector-hailo-fakeworker` | Deterministic mock for demos / tests / dev |
| `ruvector-hailo-embed` | stdin / `--text` → JSONL embeddings |
| `ruvector-hailo-stats` | Fleet observability (TSV / JSON / Prom) |
| `ruvector-hailo-cluster-bench` | Sustained-load harness |

The 3 user-facing binaries (`embed`, `stats`, `cluster-bench`) share a
common flag vocabulary documented in [ADR-168][adr168].

## Quick start

### Local demo (no Pi required)

```bash
# Terminal 1 — fakeworker
RUVECTOR_FAKE_BIND=127.0.0.1:50051 ruvector-hailo-fakeworker

# Terminal 2 — embed via stdin
echo "hello world" | ruvector-hailo-embed --workers 127.0.0.1:50051 --dim 384

# Terminal 2 — bench
ruvector-hailo-cluster-bench --workers 127.0.0.1:50051 --concurrency 4 --duration-secs 10

# Terminal 2 — fleet stats
ruvector-hailo-stats --workers 127.0.0.1:50051
```

The cluster-bench against a single fakeworker on loopback sustains
**~94k req/s** (p99 153µs). With `--cache 2000 --cache-keyspace 100`
the same bench hits **~1.09M req/s** (p99 8µs, 99.98% hit rate).

### Real Pi fleet

```bash
# 1) Install worker binary + systemd unit on each Pi
deploy/install.sh

# 2) Tag each Pi in tailscale (one-time)
sudo tailscale up --advertise-tags=tag:ruvector-hailo-worker

# 3) From any tailnet member, embed via tag-based discovery
ruvector-hailo-embed --tailscale-tag tag:ruvector-hailo-worker --port 50051 \
    --auto-fingerprint --validate-fleet --health-check 30 \
    --batch 32 --cache 4096 --cache-ttl 600 \
    --output full --quiet \
    < corpus.jsonl > embeddings.jsonl
```

## Discovery

Three discovery sources, mutually exclusive:

```bash
# Inline CSV (auto-named static-N)
--workers pi-a-host:50051,pi-b-host:50051,pi-c-host:50051

# File manifest (named workers, comments + blank lines OK)
--workers-file deploy/production.manifest

# Tailscale tag query (resolves at boot)
--tailscale-tag tag:ruvector-hailo-worker --port 50051
```

Manifest format:
```
# Production fleet — fingerprint fp:abc, dim 384.
pi-a = 100.77.59.83:50051
pi-b = 100.77.59.84:50051
pi-c = 100.77.59.85:50051   # spare unit
```

Use `ruvector-hailo-stats --workers-file <path> --list-workers` to
verify a manifest expands as expected (no health probe — works even if
the workers are down).

## Safety surface

Three layers of fingerprint integrity, end-to-end:

1. **Boot** (`validate_fleet`): rejects mismatched workers from the
   coordinator's pool, fails the boot if zero healthy workers remain.
2. **Runtime** (background health-checker, `--health-check N`): ejects
   workers that drift mid-flight, **auto-clears the cache** so stale
   vectors don't outlive the offending worker (see [ADR-169][adr169]).
3. **Ops monitoring** (`stats --strict-homogeneous`): detects drift
   purely from fleet-wide observation; alerts via exit code 3 even if
   no coordinator has fingerprint enforcement enabled.

```bash
# CI-friendly fleet health gate, no console noise:
ruvector-hailo-stats --tailscale-tag tag:ruvector-hailo-worker \
                     --strict-homogeneous --quiet \
    || alert "fleet drift detected"
```

## Caching

Optional in-process LRU. Capacity 0 ≡ disabled (default). Key includes
the model fingerprint so swapping models invalidates everything for free.

```rust
let cluster = HailoClusterEmbedder::new(...)?
    .with_cache(4096);
// Or with a TTL ceiling:
let cluster = HailoClusterEmbedder::new(...)?
    .with_cache_ttl(4096, Duration::from_secs(600));
```

CLI (all four cluster CLIs + all three sensor bridges as of iter 245):
```bash
ruvector-hailo-embed --cache 4096 --cache-ttl 600 --health-check 30 ...
ruvector-mmwave-bridge --cache 4096 --cache-ttl 300 --health-check 30 ...
ruview-csi-bridge      --cache 4096 --cache-ttl 300 --health-check 30 ...
ruvllm-bridge          --cache 4096 --cache-ttl 300 ...
```

Three eviction triggers:
- LRU overflow (capacity-bounded)
- TTL expiry (time-bounded)
- Manual `cluster.invalidate_cache()` or auto-fired by health-checker
  on detected fingerprint mismatch

Cache-hit speedup measured via cluster-bench (iter-238 baseline,
cognitum-v0):

| configuration                                | throughput     | p50    | hit_rate |
|----------------------------------------------|----------------|--------|----------|
| no cache (NPU-bound, iter-227 base)          | 70.7 RPS       | 43.5ms | n/a      |
| `--cache 4096 --cache-keyspace 64`           | 2,305,282 RPS  | 0µs    | 1.000    |

See [ADR-169][adr169] for the full design.

## NPU pipeline pool (iter 234-237)

The NPU backend supports a multi-pipeline pool — N independent
HailoRT network groups + vstream pairs on the shared vdevice —
configured via `RUVECTOR_NPU_POOL_SIZE` on the worker. Empirical
result on cognitum-v0:

| pool size | concurrency | throughput | p50    | RSS   |
|-----------|-------------|------------|--------|-------|
| 1         | 1           | 70.6 RPS   | 14.1ms | 87 MB |
| 1         | 4           | 70.7 RPS   | 56.7ms | 87 MB |
| 2         | 4           | 70.7 RPS   | 43.3ms | 142 MB|
| 4         | 4           | 70.7 RPS   | 43.5ms | 251 MB|

Throughput is identical at every pool size — HailoRT serializes
inferences across network groups at the vdevice level so the
70 RPS = 1000ms / 14ms-per-inference ceiling is hard. p50 latency
under multi-concurrent load drops 23% at pool=2 because each
request gets its own host-side queue slot. Pool=2 is the iter-237
deploy default (captures the latency win at minimum RAM cost).

## Tracing correlation

Every embed RPC propagates a `request_id` via gRPC metadata header
(`x-request-id`) — workers' tracing spans log it verbatim, so
loki/datadog queries can grep one ID across web → coordinator → worker.

```rust
// Caller-supplied (typical web-handler use case):
let trace_id = req.headers().get("x-request-id")?.to_str()?;
let v = cluster.embed_one_blocking_with_request_id(&query, trace_id)?;

// Auto-generated (default — sortable timestamp prefix):
let v = cluster.embed_one_blocking(&query)?;
// → request_id like "0000019de68b5707983b8745" (24 hex chars,
//   first 16 = epoch ms, last 8 = random)
```

CLI:
```bash
ruvector-hailo-embed --request-id "ci-build-${BUILD_NUM}" ...
ruvector-hailo-cluster-bench --request-id "${BUILD_NUM}" ...
# (bench suffixes per-thread / per-call: <id>.t<tid>.c<counter>)
```

See [ADR-170][adr170] for the full design.

## Output formats

`ruvector-hailo-embed --output {head|full|none}`:
- **head** (default) — first 8 components in `vec_head`, demo-friendly
- **full** — entire vector in `vector`, suitable for ingestion pipelines
- **none** — metadata only, useful for I/O-free benchmarking

`ruvector-hailo-stats {default|--json|--prom|--prom-file <path>}`:
- TSV with header (default)
- NDJSON (one JSON object per worker per tick)
- Prometheus textfile-collector format on stdout
- Atomic textfile write to `<path>` (paired with `--watch N` for
  drop-in node_exporter monitoring)

`ruvector-hailo-cluster-bench --prom <path>`:
- Atomic Prometheus textfile after the bench, including cache hit rate
  metrics when `--cache N` is set

## Test suite

```
                ┌──────────────────────────────┐
                │ Doctests             (7)     │  module + 6 method examples
                ├──────────────────────────────┤
                │ Lib unit            (114)    │  pure Rust, no IO
                ├──────────────────────────────┤
                │ Cluster integration  (~30)   │  GrpcTransport + tonic mocks +
                │                              │   load distribution + DoS gates
                ├──────────────────────────────┤
                │ CLI integration      (~53)   │  real binaries, real subprocess —
                │                              │   bench/embed/stats/fakeworker +
                │                              │   3 sensor bridges
                └──────────────────────────────┘
                 204 tests in this crate (iter 253)
```

Run all of them:
```bash
cargo test                                    # all suites
cargo test --doc                              # just doctests
cargo test --test cluster_load_distribution   # tonic integration only
cargo test --test embed_cli                   # binary CLI tests
```

## Deployment

`deploy/`:
- `ruvector-hailo-worker.service` — hardened systemd unit (`DeviceAllow=/dev/hailo0`,
  `ProtectSystem=strict`, `NoNewPrivileges`, etc.)
- `ruvector-hailo.env.example` — env template (model path, bind addr)
- `install.sh` — copies binary + unit + env, enables/starts the service
- `cross-build.sh` — `aarch64-unknown-linux-gnu` cross-compile via
  `gcc-aarch64-linux-gnu`

## ADRs

| ADR | Topic |
|---|---|
| [ADR-167][adr167] | NPU embedding backend (overall design) |
| [ADR-168][adr168] | Cluster CLI surface (3-binary split + flag conventions) |
| [ADR-169][adr169] | Cache architecture (LRU + TTL + fingerprint isolation + auto-invalidate) |
| [ADR-170][adr170] | Tracing correlation (gRPC metadata + sortable IDs + caller propagation) |
