# ruvector-rulake — Benchmarks

All numbers produced by a **single reproducible run** of

```bash
cargo run --release -p ruvector-rulake --bin rulake-demo
```

on a commodity Ryzen-class laptop, release build, single thread. Seeds
deterministic; reruns bit-identical.

## Headline (LocalBackend, same dataset as `ruvector-rabitq`)

Clustered Gaussian, D = 128, 100 clusters, rerank×20, 300 queries per
row (warm-cache; prime time reported separately).

### Intermediary tax is ~0× on a local backend

| n       | direct RaBitQ+ (QPS) | ruLake Fresh (QPS) | ruLake Eventual (QPS) | tax (Fresh/Eventual) |
|--------:|---------------------:|-------------------:|----------------------:|---------------------:|
|   5 000 |              17,635  |            17,166  |             17,682    | 1.03× / 1.00×        |
|  50 000 |               5,130  |             4,995  |              5,097    | 1.03× / 1.01×        |
| 100 000 |               3,050  |             2,991  |              2,990    | 1.02× / 1.02×        |

Interpretation:
- **Cache-hit path in `RuLake::search_one` costs effectively nothing** vs
  calling `RabitqPlusIndex::search` directly. The pos→id lookup + the
  HashMap get are in the noise.
- `Fresh` mode calls `LocalBackend::generation()` on every search (one
  hash-map read here). On a real backend this becomes a network RPC —
  **expect materially higher tax on BigQuery / Snowflake / S3-Parquet**.
  `Eventual { ttl_ms }` amortises it.
- Measured "prime" time is ≈ the `RabitqPlusIndex` build time on the
  pulled batch (210 ms / 50 k rows, 420 ms / 100 k rows, scales linearly).

### Federation — rayon parallel fan-out (M1)

Parallel fan-out is what the `prime` column measures now: a single
federated query that misses every shard primes them concurrently.

| n       | single-shard prime (ms) | 2-shard prime (ms) | 4-shard prime (ms) | speedup (2 / 4) |
|--------:|------------------------:|-------------------:|-------------------:|----------------:|
|   5 000 |                   22.3  |             12.7   |              6.6   |   1.76× / 3.38× |
|  50 000 |                  213.3  |            109.5   |             55.7   |   1.95× / 3.83× |
| 100 000 |                  424.8  |            215.3   |            110.1   |   1.97× / 3.86× |

Larger `n` hits closer to the theoretical K× ceiling because per-shard
work dominates over rayon + cache-insert serialization.

Warm-cache federated QPS (sequentially-issued queries):

| n       | single-shard QPS | 2 shards QPS | 4 shards QPS |
|--------:|-----------------:|-------------:|-------------:|
|   5 000 |          17,166  |      10,032  |       6,047  |
|  50 000 |           4,995  |       3,679  |       2,455  |
| 100 000 |           2,991  |       2,361  |       1,673  |

The QPS drop with shard count under this single-thread benchmark is
expected: the bench issues queries serially, so each query pays rayon's
`par_iter` startup for the shard fan-out. On a multi-client workload
where K+ queries are concurrent — which is the actual production path —
the wall-clock improvement shows up instead. The *tail latency* win
(prime times above) is the design target regardless.

## Acceptance checks (M1)

The smoke tests under `tests/federation_smoke.rs` gate M1 from
`docs/research/ruLake/07-implementation-plan.md`, plus bundle tests
in `src/bundle.rs` (including FS persistence):

| # | Test | What it proves |
|---|---|---|
| 1 | `rulake_matches_direct_rabitq_on_local_backend` | Federation path is byte-exact vs direct RaBitQ at the same seed + rerank factor |
| 2 | `rulake_recomputes_on_backend_generation_bump` | Cache coherence protocol works — backend mutation is observed on next search |
| 3 | `rulake_federates_across_two_backends` | Multi-backend fan-out + score merge produces the globally-correct top-k |
| 4 | `cache_hit_is_faster_than_miss` | Cache prime-then-serve path beats uncached (measurement-level sanity) |
| 5 | `rulake_recall_at_10_above_90pct_vs_brute_force` | End-to-end recall on clustered data stays above 90% |
| 6 | `two_backends_share_cache_when_witness_matches` | Witness-addressed cache lets two backends serving identical bytes share one compressed entry |
| 7 | `lru_eviction_caps_entry_count_when_pointers_dropped` | Bounded-memory mode: LRU evicts unpinned entries |
| 8-10 | `*_returns_error` | Error types surface on bad inputs / misconfig / unknown collections |
| 11-19 | bundle tests | Witness determinism, length-prefixing, tamper detection, FS roundtrip + atomic write, tamper-on-disk rejection |

```
cargo test -p ruvector-rulake --release
  → 19 passed / 0 failed
```

## What's NOT benchmarked (v1 scope)

- **Real-backend network latency.** `LocalBackend::pull_vectors` is an in-process
  HashMap read; the Fresh-mode tax reported above is the floor, not the ceiling.
  Real backends (Parquet on S3, BigQuery via Storage Read API) add 10-100 ms
  per prime. Measured numbers land in M2.
- **Recall regressions vs direct RaBitQ.** The test suite confirms byte-exact
  ordering + scores at the same seed. Formal recall sweeps across n / D /
  rerank_factor reuse `ruvector-rabitq::BENCHMARK.md` — ruLake doesn't change
  recall, only the distribution layer.
- **Push-down paths.** ADR-155 §Decision 4 defers backend-native vector ops
  to Tier-2 per-adapter. Not measured in v1.
- **Concurrent multi-client throughput.** Bench is single-thread. `RuLake` is
  `Send + Sync`; multi-threaded scaling is an M3 measurement.
- **Cache memory footprint vs backend size.** LRU eviction over unpinned
  entries is implemented (`RuLake::with_max_cache_entries`). Not yet
  tuned under memory pressure — that's an M3 measurement.

## Reproduce

```bash
cargo test  -p ruvector-rulake --release                   # 7 passed
cargo run   -p ruvector-rulake --release --bin rulake-demo # ~30 s on n=100k
cargo run   -p ruvector-rulake --release --bin rulake-demo -- --fast  # ~5 s
```

Dataset generator + seeds in `src/bin/rulake-demo.rs::clustered`.
