# ruLake

**A cache layer for vector search — sits in front of whatever database, lakehouse, or file store already holds your vectors, and makes every query fast.**

---

## What is ruLake?

You already have vectors somewhere: Parquet files on S3, rows in BigQuery, an Iceberg table, a Snowflake column, or `.bin` files on a disk. You want semantic search that's fast, consistent, and cheap.

ruLake is the piece in the middle.

- Your app asks ruLake for the nearest K vectors.
- ruLake serves hits from a compressed in-memory cache at **~1 % over raw library speed** (measured 1.00–1.02× intermediary tax).
- On a cache miss, ruLake pulls from your backend, compresses with [RaBitQ](../ruvector-rabitq/) 1-bit quantization, and serves the result.
- Every cache entry is anchored by a cryptographic **witness** so two processes pointing at the same bytes share one compressed copy automatically.

It's built on the **RuVector** stack:

```
your data  ──▶  RuVector RVF (durable)
                    │
                    ▼
              RuVector rabitq  ◀──  1-bit quantization + rerank
                    │
                    ▼
               ruLake           ◀──  this crate: cache + coherence + governance
                    │
                    ▼
              your agent / app
```

---

## Why ruLake exists

Today the tradeoff is ugly:

- A **managed vector DB** (Pinecone, Weaviate) is fast but it's a whole new system to operate, and your data has to move into it.
- A **lakehouse** (BigQuery, Snowflake, Iceberg) keeps your data where it is but vector queries are expensive, slow, or not supported natively.
- A **local library** (RaBitQ, HNSW) is fastest per-process but doesn't help with sharing, coherence, governance, or multi-source queries.

ruLake is the middle option: keep your data where it lives, get cache-speed reads, and pay governance once instead of per-backend.

---

## Features

Each of these is shipped, tested, and measured in the M1 release.

### 🚀 Cache-first performance
- **1.00–1.02× direct RaBitQ cost** on cache hit — the abstraction is effectively free.
- **23,681 QPS** single-shard, **33,094 QPS** 4-shard under 8 concurrent clients (n=100k, D=128).
- **37.6 ms prime** on n=100k — 11× faster than serial thanks to parallel rotation + bit-packing.

### 🔐 Witness-authenticated bundles
- Each cache entry is anchored on a SHAKE-256 digest of `(data_ref, dim, rotation_seed, rerank_factor, generation)`.
- Serializes to a tiny `table.rulake.json` sidecar alongside your data.
- Two backends (or two clouds, or two processes) pointing at the same bytes produce the same witness and **share one compressed cache entry** with zero coordination.

### 🔁 Federated search across any number of backends
- Register N backends (Parquet, BigQuery, custom), search them all in one call.
- Parallel fan-out via rayon; adaptive per-shard rerank keeps K-shard federation from paying K× the rerank cost.
- Global top-K merged by score — correct ranking across heterogeneous sources.

### 🎛 Three-mode consistency knob
| Mode | Use case | Cost per query |
|---|---|---|
| `Fresh` | compliance, finance, policy | 1 backend RTT |
| `Eventual { ttl_ms }` | search, RAG, recommendation | 1 RTT per TTL |
| `Frozen` | audit snapshots, content-addressed data | 0 RTTs after prime |

One deployment can serve all three depending on the collection. It's a product knob, not a build flag.

### 📦 Cross-process cache sync via sidecar
- `publish_bundle(key, dir)` — writer side, atomic write.
- `refresh_from_bundle_dir(key, dir)` — reader side, three-state response (`UpToDate` / `Invalidated` / `BundleMissing`).
- A cache-sidecar daemon is ~10 lines on top of these primitives; see `examples/sidecar_daemon.rs`.

### 📊 Built-in observability
- `hit_rate()`, `avg_prime_ms()`, `last_prime_ms` out of the box.
- Per-backend and per-collection attribution — find the hot collection, pin it.
- No external tracing layer needed for the headline metric.

### 🧱 Pluggable `VectorKernel` trait
- CPU kernel ships by default, always available.
- GPU / SIMD / WASM kernels plug in as separate crates — no dep bloat on laptop / edge builds.
- Dispatch policy enforces determinism on `Fresh` / `Frozen` paths; non-deterministic kernels are only used on `Eventual`.

### 🛡 Security by default
- Zero `unsafe` in the crate.
- Path-traversal validation on filesystem backends (12-form attack coverage).
- JSON size caps on bundle deserialization (prevents DoS on compromised sidecars).
- Witness verification on every bundle read — tampered files fail loudly.
- Atomic writes so concurrent readers never see torn sidecars.

---

## Benefits

**For the application developer**
- Drop-in acceleration for any vector store you already have.
- One API (`search_one`, `search_federated`, `search_batch`) regardless of where the data lives.
- Operational metrics that matter — hit rate, prime time, per-backend — without extra tracing infrastructure.

**For the platform team**
- One governance choke point instead of N per-backend stories.
- Cross-process, cross-cloud cache sharing with zero coordination (witness-addressed).
- Three consistency modes so compliance, AI, and audit workloads share one deployment.

**For the performance engineer**
- 1 % intermediary tax — the cache is effectively free.
- 11–12× concurrent throughput win from the Arc-drop-lock refactor.
- 11× prime-time speedup from parallel rotation.
- Determinism preserved end-to-end so witness chains stay valid across CPU / SIMD / GPU kernels.

**For the security engineer**
- Zero `unsafe`.
- Tampered bundles fail fast with a typed error.
- No path traversal, no unbounded allocations.
- Witness chain is domain-separated + length-prefixed SHAKE-256.

---

## How ruLake compares

ruLake is explicitly **not** a vector database — it doesn't own storage. It's the substrate that lets you query whichever vector DB or lakehouse you already have, with a coherent compression + governance story across all of them. If you want a standalone managed vector DB, use Pinecone or Weaviate. If you want to use the vectors that already live in your lake, use ruLake — part of the [RuVector](https://github.com/ruvnet/RuVector) ecosystem alongside RVF (durable segments), `ruvector-rabitq` (1-bit compression), and `ruvector-rulake` (this crate).

| System           | Abstraction cost | Cross-backend federation | Witness-authenticated | Cross-process cache sharing | CPU-first / GPU-optional | `unsafe` count |
|------------------|-----------------:|-------------------------:|----------------------:|----------------------------:|-------------------------:|---------------:|
| **ruLake**       | **1.02×**        | ✅ (rayon fan-out)       | ✅ (SHAKE-256)        | ✅ (content-addressed)      | ✅                       | **0**          |
| Pinecone         | n/a (hosted)     | ❌                        | ❌                    | ❌                          | n/a                      | n/a            |
| Weaviate         | n/a (hosted)     | ❌                        | ❌                    | ❌                          | ✅                       | n/a            |
| Milvus           | ~1.5–2×          | partial                   | ❌                    | ❌                          | ✅                       | many           |
| LanceDB          | ~1.1–1.3×        | ❌                        | ❌                    | ❌                          | ✅                       | some           |
| BQ Vector Search | n/a (hosted)     | ❌ (BQ-only)              | ❌                    | ❌                          | n/a                      | n/a            |

And within the RuVector stack:

| Crate                  | Role                                               |
|------------------------|----------------------------------------------------|
| `ruvector-rvf`         | durable segment format — appendable, witness-signed vector storage |
| `ruvector-rabitq`      | rotation-based 1-bit quantization kernel — the math that makes the cache fast |
| `ruvector-rulake`      | **this crate** — cache, coherence, federation, governance, adapters |

RVF is your durable truth. rabitq is your compressor. ruLake is the execution layer.

---

## Quick start

```toml
[dependencies]
ruvector-rulake = "2.2"
```

```rust
use std::sync::Arc;
use ruvector_rulake::{cache::Consistency, LocalBackend, RuLake};

// 1. Point ruLake at a backend.
let backend = Arc::new(LocalBackend::new("my-backend"));
backend.put_collection(
    "memories",
    /* dim    */ 128,
    /* ids    */ vec![1, 2, 3],
    /* vecs   */ vec![vec![0.0; 128]; 3],
)?;

// 2. Configure the cache.
let lake = RuLake::new(20, 42)
    .with_consistency(Consistency::Eventual { ttl_ms: 60_000 });
lake.register_backend(backend)?;

// 3. Query. First hit primes the cache; the rest serve from RaBitQ
//    at ~1 % over raw-library speed.
let hits = lake.search_one("my-backend", "memories", &vec![0.0; 128], 10)?;

// 4. Observe.
println!("hit rate: {:.3}", lake.cache_stats().hit_rate().unwrap_or(0.0));
```

For a full cross-process example with publish/refresh, see
[`examples/sidecar_daemon.rs`](examples/sidecar_daemon.rs).

---

## Usage recipes

### RAG / retrieval at 95 % hit rate

```rust
let lake = RuLake::new(20, 42)
    .with_consistency(Consistency::Eventual { ttl_ms: 60_000 })
    .with_max_cache_entries(100);   // LRU bound
lake.register_backend(parquet_backend)?;

// Batch API amortizes freshness check + lock acquisition across N queries.
let hits = lake.search_batch("parquet", "corpus", &query_batch, 10)?;

// Target metric: cache_stats().hit_rate() ≥ 0.95
```

### Federated search across clouds

```rust
let hits = lake.search_federated(
    &[
        ("bigquery",  "events"),
        ("snowflake", "profiles"),
        ("iceberg",   "archive"),
    ],
    &query,
    10,
)?;
// Adaptive per-shard rerank = max(5, 20 / 3) = 6 per shard.
// Global top-10 merged across all three, each returned with its
// backend + collection for audit.
```

### Audit-tier witness-sealed snapshot

```rust
let audit = RuLake::new(20, 42).with_consistency(Consistency::Frozen);
audit.register_backend(content_addressed_backend)?;

// First query primes from the backend; after that ruLake never
// re-checks, no matter what the backend reports. Operators can
// still force-refresh via refresh_from_bundle_dir.
let hits = audit.search_one("ca", "snapshot-2026-q2", &q, 10)?;
```

### Cross-process cache sidecar

```rust
// Reader process runs this loop next to the serving process:
loop {
    match lake.refresh_from_bundle_dir(&key, publish_dir)? {
        RefreshResult::Invalidated => metrics.bundle_rotations.inc(),
        _ => {}
    }
    std::thread::sleep(Duration::from_secs(5));
}
```

See [`examples/sidecar_daemon.rs`](examples/sidecar_daemon.rs) for the
runnable publisher + reader demo.

### Memory substrate for agent brain systems

ruLake tags bundles with an opaque `memory_class` (ADR-156): the
substrate stores it but never interprets it. Brain systems own the
semantics.

```rust
let bundle = RuLakeBundle::new("mem://episodic/2026-04-23", 768, 42, 20, gen.into())
    .with_memory_class("episodic")
    .with_pii_policy("pii://policies/redact-pii")
    .with_lineage_id("ol://jobs/episodic-consolidation");
bundle.write_to_dir("/mnt/brain/bundles/")?;
```

The six substrate guarantees (recall, verify, forget, rehydrate,
compact, location-transparency) are validated end-to-end by the
`brain_substrate_acceptance_*` test.

---

## Benchmarks

All numbers from a single reproducible run of:

```bash
cargo run --release -p ruvector-rulake --bin rulake-demo
```

on a commodity Ryzen-class laptop, deterministic seeds, warm cache
unless a row is labeled `prime`.

### Intermediary tax (cache-hit path)

Clustered Gaussian, D=128, 100 clusters, rerank×20, 300 queries.

| n       | direct RaBitQ+ | ruLake Fresh | ruLake Eventual | tax     |
|--------:|---------------:|-------------:|----------------:|--------:|
|   5 000 |        17,567  |      17,431  |         17,567  | 1.01×   |
|  50 000 |         4,985  |       4,932  |          4,959  | 1.01×   |
| 100 000 |         2,975  |       3,020  |          2,963  | 1.00×   |

**Takeaway:** the abstraction is free. The cache-hit path is as fast
as calling `ruvector-rabitq` directly.

### Concurrent clients × shard count

n=100k, 8 clients × 300 queries each, `Eventual` mode.

| shards | wall (ms) |       QPS | vs pre-Arc-refactor |
|-------:|----------:|----------:|--------------------:|
|      1 |     101.3 |    23,681 |                8.3× |
|      2 |      82.8 |    28,971 |               10.1× |
|      4 |      72.5 |    33,094 |               11.6× |

**Takeaway:** the `Arc<RabitqPlusIndex>` cache refactor lifted
concurrent QPS by 8-12×. The cache mutex no longer holds the scan.

### Cold-start prime time

Parallel rotation + bit-packing via rayon.

| n       | serial prime | parallel prime |   speedup |
|--------:|-------------:|---------------:|----------:|
|   5 000 |      22.3 ms |        4.5 ms  |    4.9×   |
|  50 000 |     213.3 ms |       19.6 ms  |   10.9×   |
| 100 000 |     424.8 ms |       37.6 ms  |   11.2×   |

**Takeaway:** real backend deployments where prime cost is the
critical-path on cache miss see a full order-of-magnitude drop.

### Recall

- `rulake_recall_at_10_above_90pct_vs_brute_force` — **≥ 90 %** on
  clustered D=128 n=5k rerank×20 vs exact L2² brute force.
- `adaptive_per_shard_rerank_preserves_recall` — **≥ 85 %** on K=2
  and K=4 with adaptive rerank = max(5, 20 / K).

See [`BENCHMARK.md`](BENCHMARK.md) for full methodology.

---

## How it works

### Data flow

```
search(backend, collection, query, k)
  │
  ▼
ensure_fresh(key) ─── Consistency mode?
  │                          │
  ├── Frozen  (skip after prime)
  ├── Eventual (skip within TTL)
  └── Fresh   (always check)
         │
         ▼
      ask backend for current witness
         │
    ┌────┴──────────────┐
  match                mismatch
  (hit)               │
                   witness cached elsewhere?
                   │              │
                  yes             no
                   │              │
              move pointer   pull + prime
              (0 work)       (compress into
                              RaBitQ codes)
         │              │
         ▼              ▼
  Arc<RabitqPlusIndex>::search (mutex dropped before scan)
         │
         ▼
     top-K results, sorted by L2²
```

### The bundle is the portable unit

Every cache entry is anchored by a `table.rulake.json` sidecar:

```json
{
  "format_version": 1,
  "data_ref": "gs://bucket/corpus.parquet",
  "dim": 768,
  "rotation_seed": 42,
  "rerank_factor": 20,
  "generation": 1729843200,
  "rvf_witness": "b3ac…0f7c",
  "pii_policy": "pii://policies/default",
  "lineage_id": "ol://jobs/ingest-42",
  "memory_class": "episodic"
}
```

The witness is a domain-separated, length-prefixed SHAKE-256 over the
load-bearing fields. Two processes that observe the same bundle share
one compressed cache entry — no coordination required.

### Adaptive per-shard rerank

Under federation, RaBitQ would run its `rerank_factor × k` rerank once
per shard, costing K× more work as shard count grows. ruLake divides
the budget:

```
per_shard_rerank = max(MIN_PER_SHARD_RERANK, global_rerank / K)
```

K=4 at rerank×20 gives 5 per shard. Measured recall@10 stays above
85% (gate test). Callers that need byte-exact single-shard parity use
`search_federated_with_rerank(.., Some(global_rerank))`.

### Arc-based concurrency

`CacheEntry::index` is `Arc<RabitqPlusIndex>`. Readers:

1. Lock the cache mutex
2. Clone the Arc (refcount bump, a few cycles)
3. **Drop the lock**
4. Scan without holding anything shared

The index is immutable after build, so concurrent scans race against
nothing. This is the single biggest performance win on the branch —
**8-12× concurrent QPS**.

### Parallel prime

On cache miss, `RabitqPlusIndex::from_vectors_parallel` rotates and
bit-packs every vector in parallel via rayon, then commits them into
the SoA storage serially. Output is bit-identical to the serial
`add()` loop because rotation is deterministic. Above 1024 vectors
this is faster than serial; below, the rayon task-queue overhead
dominates and we fall back.

---

## User guide

### Choose a consistency mode

| Symptom / requirement | Mode |
|---|---|
| Legal / compliance: can't serve stale data, ever | `Fresh` |
| Search, RAG, recommendation, agent retrieval | `Eventual { ttl_ms: 60_000 }` |
| Audit snapshot; data is cryptographically pinned | `Frozen` |

### Size the cache

```rust
// Unbounded — fine for small collections or low-cardinality serving
let lake = RuLake::new(20, 42);

// LRU-capped for memory-bounded serving processes
let lake = RuLake::new(20, 42).with_max_cache_entries(100);
```

Only unpinned entries (refcount == 0, no live pointer) are evicted;
active `(backend, collection)` pointers keep their entry alive.

### Operational metrics

| Metric | Signal | Action |
|---|---|---|
| `hit_rate` | < 0.95 | Grow cache or warm aggressively |
| `last_prime_ms` | spiking | Backend RTT changed or collection grew |
| `primes` | monotonic growth | Check for witness churn |
| `shared_hits` | > 0 | Cross-backend sharing is working |
| `invalidations` | climbing | Coherence protocol firing — inspect |

Per-backend (`cache_stats_by_backend()`) and per-collection
(`cache_stats_by_collection()`) views drill down.

### Write a custom backend

Implement the four-method `BackendAdapter` trait:

```rust
use ruvector_rulake::backend::{BackendAdapter, CollectionId, PulledBatch};

struct ParquetBackend { /* ... */ }

impl BackendAdapter for ParquetBackend {
    fn id(&self) -> &str { "parquet" }
    fn list_collections(&self) -> Result<Vec<CollectionId>> { /* ... */ }
    fn pull_vectors(&self, collection: &str) -> Result<PulledBatch> { /* ... */ }
    fn generation(&self, collection: &str) -> Result<u64> { /* ... */ }
}
```

See [`src/fs_backend.rs`](src/fs_backend.rs) for a 250-line reference
(atomic file writes, mtime-as-generation, header-only `current_bundle`).

### Run the examples

```bash
# Full publish/refresh/coherence demo
cargo run --release -p ruvector-rulake --example sidecar_daemon

# Benchmark harness (~2 minutes)
cargo run --release -p ruvector-rulake --bin rulake-demo

# Fast mode (~5 seconds, just n=5k)
cargo run --release -p ruvector-rulake --bin rulake-demo -- --fast
```

---

## Status

**M1 — shipped and measured** (2026-04-23)

- Core abstraction (`BackendAdapter` trait, `VectorCache`, bundle protocol, 3 consistency modes, LRU)
- Two reference backends (`LocalBackend`, `FsBackend`)
- Optimizations: adaptive per-shard rerank, Arc-concurrency (12× concurrent win), parallel prime (11× miss-path win)
- Observability: hit rate, prime times, per-backend, per-collection attribution
- Substrate acceptance test (recall → verify → forget → rehydrate)
- Security hardening (path traversal, JSON caps, witness verification)
- `VectorKernel` trait scaffolding
- 60 tests across the two crates, clippy `-D warnings` clean, zero `unsafe`

**M2+ on the roadmap**

- `ParquetBackend`, `BigQueryBackend`, `IcebergBackend`, `DeltaBackend`
- HTTP / gRPC wire layer
- Governance MVP (RBAC via OIDC, PII passthrough, OpenLineage)
- GPU kernels in separate crates (`ruvector-rabitq-cuda`, etc.)

Full design record:

- [`ADR-155`](../../docs/adr/ADR-155-rulake-datalake-layer.md) — cache-first fabric
- [`ADR-156`](../../docs/adr/ADR-156-rulake-as-memory-substrate.md) — memory substrate for agent brains
- [`ADR-157`](../../docs/adr/ADR-157-optional-accelerator-plane.md) — optional accelerator plane

---

## License

Apache-2.0 OR MIT (RuVector workspace default)
