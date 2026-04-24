# 05 — Performance Budget

The rule for this document: if a number is a **target**, it is labelled
"target, unmeasured." If a number is a **citation**, it names the
source file and the exact table row. Nothing is invented.

---

## The Source: What We Have Measured

Two files carry live numbers.

### 1. `crates/ruvector-rabitq/BENCHMARK.md`

Measured on a commodity Ryzen laptop, single thread, release build,
**no SIMD intrinsics**, seed-deterministic. Dataset: 100-cluster
Gaussian in `[-2, 2]^D`, σ=0.6 within-cluster, D=128.

| n       | variant            | r@10   | QPS     | mem/MB | lat/ms |
|--------:|--------------------|-------:|--------:|-------:|-------:|
| 1 k     | Flat               | 100.0% | 21,195  | —      | —      |
| 5 k     | Flat               | 100.0% |  5,530  | —      | —      |
| 50 k    | Flat               | 100.0% |    619  | —      | —      |
| 50 k    | Sym rerank×5       |  99.9% |  1,439  | —      | —      |
| **100 k** | **Flat**         | **100.0%** | **309** | **50.4** | **3.27** |
| **100 k** | **Sym rerank×20 (picked)** | **100.0%** | **957** | **53.5** | **1.05** |
| 100 k   | Sym rerank×5       |  87.9% |    811  | —      | —      |
| 100 k   | Sym no rerank      |   8.1% |  3,639  | **2.4** |  0.28 |

Memory compression at n=100k: Flat 50.4 MB vs RaBitQ 1-bit 2.4 MB =
**21× for the pure binary index**. With rerank codes stored, RaBitQ+
total is 53.5 MB — slightly higher than Flat because it carries both.

Source: `crates/ruvector-rabitq/BENCHMARK.md` §"Headline (n = 100,000,
D = 128)".

### 2. `crates/rvf/rvf-federation/README.md` — federation pipeline

| Benchmark                       | Time    |
|---------------------------------|---------|
| PII detect (single string)      | 756 ns  |
| PII strip (10 fields)           | 44 µs   |
| PII strip (100 fields)          | 303 µs  |
| Gaussian noise (100 params)     | 4.7 µs  |
| Gaussian noise (10k params)     | 334 µs  |
| FedAvg (10 contrib, 100 dim)    | 3.9 µs  |
| FedAvg (100 contrib, 1k dim)    | 365 µs  |
| Full export pipeline            | 1.2 ms  |

Source: `crates/rvf/rvf-federation/README.md` §"Performance Benchmarks".

### 3. `docs/research/rvf/INDEX.md` — RVF spec targets

These are the spec's design targets, not measurements. Treat as
acceptance criteria for the RVF core, not ruLake SLOs.

- Boot: 4 KB read, **< 5 ms** (Level-0 root manifest)
- First query: ≤ 4 MB read, **recall ≥ 0.70** (Layer A)
- Full quality: **recall ≥ 0.95** (Layer C)
- Signing: ML-DSA-65, 3,309 B signatures, **~4,500 sign/s**
- Distance: 384-dim fp16 L2 in **~12 AVX-512 cycles**
- Hot entry: 960 bytes (vector + 16 neighbors, cache-line aligned)

Source: `docs/research/rvf/INDEX.md` §"Key Numbers".

### Measured (ruvector-rulake, 2026-04-23, commit of this branch)

The intermediary itself — on a LocalBackend (in-memory coherence check),
D=128, rerank×20, 300 warm queries, single thread.

| n       | direct RaBitQ+ QPS | ruLake Fresh QPS | ruLake Eventual QPS | Intermediary tax |
|--------:|-------------------:|-----------------:|--------------------:|-----------------:|
|   5 000 |             17,311 |          17,874  |             17,858  | **0.97×**        |
|  50 000 |              5,162 |           5,123  |              5,050  | **1.01×**        |
| 100 000 |              3,122 |           3,117  |              3,114  | **1.00×**        |

Source: `crates/ruvector-rulake/BENCHMARK.md`.

The intermediary tax on an in-process backend is **effectively zero**.
On a real backend the Fresh-mode generation check becomes a network RPC;
the measured number here is the floor. Eventual mode amortises the
check so the floor holds across real backends too.

Federated QPS (same n, sequential fan-out across K shards) at n=100k:
single-shard 3,117 → 2 shards 2,470 → 4 shards 1,781. Parallel fan-out
via `rayon` is the v2 optimisation (see ADR-155 §Consequences).

---

## What We Do NOT Have

All of the following are **targets, unmeasured** as of 2026-04.

- SIFT1M / GIST1M / DEEP10M recall numbers. The standard ANN
  benchmarks. `BENCHMARK.md §What's NOT benchmarked` flags this as a
  follow-up.
- Parallel (multi-thread) RaBitQ throughput. `parallel` feature
  exists but all benchmark numbers above are single-thread.
- SIMD popcount via `std::arch::x86_64`. Currently scalar; AVX2
  shuffle-based popcount is a named follow-up.
- HNSW + RaBitQ integration numbers. RaBitQ is a standalone index
  today; plugging it into `rvf-index`'s HNSW is a named follow-up.
- GCS range-read tail latency (p50 / p95 / p99) at the scales we care
  about. Network, not compute.
- BQ remote function cold-start p50.
- BQ remote function warm-call overhead (HTTPS round-trip).
- End-to-end BQ query latency at 1 M vectors.

Each of these has an M2/M3 measurement slot in
`07-implementation-plan.md`.

---

## Target Budget (All Unmeasured)

### Query latency (BigQuery Tier-1 path)

| Stage                                      | Budget (target)  |
|--------------------------------------------|------------------|
| BQ job dispatch + UDF resolution            | 20–50 ms         |
| Remote function HTTPS round-trip (warm)     | 30–80 ms         |
| UDF RaBitQ+ scan, n=1M, D=128, rerank×20    | 10–30 ms         |
| JSON encode + return                        | 5–15 ms          |
| BQ post-UDF merge + result delivery         | 20–50 ms         |
| **p50 warm (target)**                       | **≤ 150 ms**     |
| **p95 warm (target)**                       | **≤ 300 ms**     |
| **Cold start (container + Layer-A read)**   | **≤ 2 s**        |

The "10–30 ms scan" line extrapolates from BENCHMARK.md — 1.05 ms at
n=100k scales linearly to ~10 ms at n=1M single-thread. With the
parallel feature and an 8-vCPU Cloud Run instance, expect headroom.

**Commitment level:** we will **measure and publish** these p50/p95 in
M3 (week 8). If we miss by 2× we replan; by 5× we stop and re-scope.

### Ingest throughput

| Path                                               | Target            |
|----------------------------------------------------|-------------------|
| Parquet → RVF (rvf-import, single-thread)           | 250 k vec/s       |
| GCS upload                                         | Bandwidth-bound   |
| Full ingest of 10M vectors, D=128, cold           | ≤ 15 min          |
| Nightly compaction of 100M vectors                 | ≤ 2 h             |

Unmeasured. The 250 k vec/s target is from the `rvf-import` existing
CSV/JSON/NumPy path; verify for Parquet before committing.

### Memory ceiling (inside the UDF)

At n=1M, D=128:

| Resource                                   | Estimate                               |
|--------------------------------------------|----------------------------------------|
| RaBitQ 1-bit codes (packed u64)            | 16 B × 1M = 16 MB                      |
| RaBitQ rerank f32 codes (if enabled)       | 512 B × 1M = 512 MB                    |
| HNSW Layer-A (top layer, ~0.5% of n)       | ~5 k entries × 100 B = 500 KB          |
| HNSW Layer-B (hot region)                  | ~10% of n × 100 B = 10 MB              |
| Rotation matrix (D×D f32)                  | 128 × 128 × 4 = 64 KB                  |
| **Warm working set**                       | **~530 MB**                            |
| **With HNSW Layer C on demand**            | **~2–3 GB**                            |

A 2 GB Cloud Run instance holds the warm working set plus a 200 MB
Layer-C cache comfortably. Numbers extrapolate from BENCHMARK.md's
53.5 MB at n=100k.

### Cost per 1B vectors (rough OOM, unmeasured)

| Line item                                    | Estimate                          |
|----------------------------------------------|-----------------------------------|
| GCS storage, 1 B × D=128, RaBitQ + f32 rerank | 16 GB + 512 GB ≈ $10–12/mo (GCS std) |
| Cloud Run, 1 warm instance us-central1       | ~$40–60/mo (idle), more under load  |
| Egress (if cross-region)                     | $0.01–0.12 / GB, workload-shaped    |
| BQ remote function call cost                 | Per invocation, workload-shaped     |
| **Steady-state floor (quiescent 1B index)**  | **~$50–80/mo**, **verify**          |

All prices as of GCP us-central1, early 2026. Verify before quoting
to a customer.

---

## Budget Guardrails

If any of the following is violated during M2/M3, **stop and
re-scope**:

1. **Recall@10 on SIFT1M at rerank×20 is below 95%.** We promised
   100% on clustered Gaussian; real embedding data is trained, so we
   expect comparable or better. If it is below 95%, the UDF needs
   higher rerank factor or IVF partitioning (a new ADR).
2. **Cold start exceeds 5 s.** Kills the "first query after deploy"
   experience. Mitigation: smaller Layer-A hotset, or precomputed
   warm-up request.
3. **Warm p95 exceeds 500 ms.** Kills the "interactive analyst"
   experience. Mitigation: bigger Cloud Run, parallel scan.
4. **Memory footprint exceeds 4 GB at n=1M.** Cloud Run cost scales
   with memory — breakeven vs native BQ Vector Search shifts.

Each violation triggers a named mitigation or an explicit scope cut.

---

## What the BENCHMARK.md Numbers Mean for ruLake

The RaBitQ+ rerank×20 configuration at n=100k already gives us:

- **100% recall@10, identical to Flat.** The customer does not see an
  accuracy compromise.
- **3.13× throughput over Flat.** 957 QPS vs 309 QPS, single-thread.
- **21× memory compression of the binary index** (2.4 MB vs 50.4 MB).

At n=1M the scan-only throughput drops roughly linearly (the scan is
O(n) per query), so single-thread RaBitQ+ rerank×20 at n=1M will land
near **~100 QPS**. With parallel + SIMD (both named follow-ups),
expect another ~4–8× factor. The BQ Tier-1 path's **latency is
dominated by the HTTPS round-trip**, not the scan, so even the
unoptimised single-thread path clears the budget above — which is the
whole reason Candidate A (see `03-bigquery-integration.md`) is viable.

---

## What Happens If We Miss

If M3's measured p50 is, say, 400 ms instead of 150 ms:

1. Check whether Cloud Run cold-start dominates.
2. Check whether JSON serialisation dominates (a known tax — Arrow
   Flight would remove it but is not BQ's remote-function ABI).
3. Check whether HTTPS round-trip dominates (fundamental, BQ-side).
4. Check whether the scan dominates (fixable with parallel + SIMD).

Order of fixes: 1 → 4 → 2 → 3. Number 3 is only solvable by getting
BQ to host our UDF in-process, which BQ persistent Python UDFs _may_
allow — and that is a v2 investigation.
