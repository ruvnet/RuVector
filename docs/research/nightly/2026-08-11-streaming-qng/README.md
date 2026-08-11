# Streaming Quantized Neighbourhood Graphs (QNG-Stream)

**Date:** 2026-08-11  
**Branch:** `research/nightly/2026-08-11-streaming-qng`  
**Crate:** `crates/ruvector-streaming-qng`  
**ADR:** [ADR-298](../../adr/ADR-298-streaming-qng.md)

---

## Problem

Agent memory systems emit vectors continuously. The embedding distribution drifts as the agent's context shifts topic or domain. A Product Quantization (PQ) codebook trained at startup systematically misquantizes the new distribution — centroids that fitted the original data no longer partition the new data well. The recall degradation is silent: no error is raised, but the wrong memories are retrieved.

**Example:** an agent starts in code-generation mode (vectors cluster around programming language tokens), then shifts to scientific literature review (vectors cluster around mathematical notation). The original PQ codebook assigns scientific vectors to code-cluster centroids, mangling the ADC distance computation. Queries about equations return code snippets.

---

## Approach

**Three measurable variants:**

| Variant | Strategy |
|---------|----------|
| `FullPrecision` | Brute-force f32 linear scan — ground truth |
| `StaticPQ` | PQ codebook trained once at build time, never updated |
| `StreamPQ` | PQ codebook periodically retrained on a reservoir sample |

**StreamPQ design:**

1. **Reservoir sampling** (Vitter's Algorithm R): maintains a bounded, uniform random sample of all vectors seen so far. Guarantees that after `N_B` Phase-B inserts the reservoir holds `N_B / (N_A + N_B)` Phase-B vectors in expectation.

2. **Full k-means retrain on reservoir**: every `update_freq` inserts, run full Lloyd's algorithm on the reservoir to produce a fresh codebook. This restarts centroid positions from data — no stale bias from the old codebook.

3. **Full re-encoding of all stored vectors**: after each retrain, all raw vectors are re-encoded with the new codebook. ADC distances stay globally consistent.

**Why full retrain instead of EMA one-pass:** a one-pass EMA approach was explored first and abandoned. When the shift is comparable to the cluster spacing, two shifted clusters both map to the nearest stale centroid bin. The EMA averages them into a merged centroid that represents neither. Full retrain from reservoir data separates them once the reservoir is dominated by the new distribution.

**Reservoir domination condition:** Phase-B stream must be ≥3× Phase-A size for the reservoir to reach ≥75% Phase-B vectors. At that point, k-means reliably converges to Phase-B cluster positions. The benchmark uses 4× to achieve ~80% Phase-B in the reservoir.

---

## Benchmark Results

**Environment:** x86_64 Linux, release build (`cargo run --release --bin benchmark`)

**Config:**
```
dims=64  clusters=4  shift=3.0  std=0.3
Phase A: 2000 vectors (500 per cluster)
Phase B: 8000 vectors (2000 per cluster, 4× Phase A)
Queries: 80 Phase-B (20 per cluster)  k=10
StreamPQ: reservoir_cap=1024  update_freq=200  (40 retrains total)
```

### Phase A cluster precision (original distribution)

| Variant | n | mean latency | p50 | p95 | QPS | memory | cluster_prec |
|---------|---|-------------|-----|-----|-----|--------|--------------|
| FullPrecision | 2000 | 153.2 µs | 145 µs | 184 µs | 6,527 | 2500 KB | 1.0000 |
| StaticPQ | 2000 | 55.0 µs | 51 µs | 72 µs | 18,172 | 43 KB | 1.0000 |
| StreamPQ | 2000 | 55.0 µs | 52 µs | 70 µs | 18,181 | 2799 KB | 1.0000 |

All three variants achieve perfect cluster precision on Phase A (the distribution they were trained on).

### Streaming insert throughput (8000 Phase-B vectors)

| Variant | Wall time | Throughput |
|---------|-----------|-----------|
| FullPrecision | 0.3 ms | 25,469,515 vec/s |
| StaticPQ | 5.7 ms | 1,391,733 vec/s |
| **StreamPQ** | **853.3 ms** | **9,375 vec/s** |

StreamPQ is 148× slower than StaticPQ due to 40 full k-means retrains (20 iterations × 1024 reservoir vectors × 4 subspaces × 16 centroids per refresh). This is the principal cost of adaptation.

### Phase B cluster precision (shifted distribution, combined index n=10000)

| Variant | n | mean latency | p50 | p95 | QPS | memory | cluster_prec |
|---------|---|-------------|-----|-----|-----|--------|--------------|
| FullPrecision | 10000 | 780.7 µs | 768 µs | 819 µs | 1,280 | 2500 KB | 1.0000 |
| StaticPQ | 10000 | 142.0 µs | 138 µs | 160 µs | 7,044 | 43 KB | 0.9863 |
| **StreamPQ** | **10000** | **162.0 µs** | **156 µs** | **183 µs** | **6,173** | **2799 KB** | **1.0000** |

### Per-cluster Phase-B precision breakdown

| Cluster | dim0 shift | StaticPQ | StreamPQ | Delta |
|---------|-----------|---------|---------|-------|
| 0 | 0 → 3 (nearest stale: cluster 1 at 4) | 0.9800 | 1.0000 | +0.0200 |
| 1 | 4 → 7 (nearest stale: cluster 2 at 8) | 1.0000 | 1.0000 | +0.0000 |
| 2 | 8 → 11 (nearest stale: cluster 3 at 12) | 1.0000 | 1.0000 | +0.0000 |
| 3 | 12 → 15 (beyond stale range, maps to cluster 3) | 0.9650 | 1.0000 | +0.0350 |

The degradation is **cluster-specific**: clusters 0 and 3 are the edge cases where the shift moves vectors to positions furthest from their original codebook centroid. Cluster 3 (dim0=15) shifts beyond all Phase-A centroids (max at 12) and sees the worst StaticPQ degradation (−0.035). StreamPQ eliminates the degradation across all clusters.

### Acceptance gates

```
[1] FullPrecision Phase-B cluster precision ≥ 0.90 : 1.0000  → PASS
[2] StreamPQ Phase-A cluster precision ≥ 0.60      : 1.0000  → PASS
[3] StreamPQ Phase-B cluster precision ≥ 0.50      : 1.0000  → PASS
[4] StreamPQ Phase-B ≥ StaticPQ Phase-B - 0.05     : 1.0000 vs 0.9863  → PASS

✓ ACCEPTANCE: PASS — StreamPQ adapts to distribution shift.
```

---

## Key Insights

### 1. Metric matters: cluster precision, not recall@k

PQ discriminates **between clusters** with near-perfect accuracy but cannot rank **within-cluster** vectors precisely. Quantization error is comparable to within-cluster distance variance at realistic densities. recall@k requires exact top-k ordering — the wrong metric for PQ evaluation. Cluster precision (fraction of top-k from the correct cluster) is the correct metric.

This is not a weakness of PQ — it is its design: coarse quantization for fast approximate search, not exact ranking. A two-stage re-rank (PQ retrieve + exact re-score) handles within-cluster ordering when needed.

### 2. Reservoir domination is the critical condition

Vitter's Algorithm R guarantees a uniform random sample over all seen vectors. With Phase-B:Phase-A = 4:1, the reservoir reaches 80% Phase-B vectors, and k-means correctly places centroids at Phase-B positions. With equal sizes (1:1), the reservoir is 50-50 and k-means places centroids at the midpoints — representing neither distribution well.

**Rule of thumb:** stream at least 3× as many new-distribution vectors as old to achieve reliable codebook convergence.

### 3. EMA converges to the wrong answer under centroid collision

When `shift ≈ cluster_spacing / 2`, the EMA one-pass approach creates "centroid collisions": two different new-distribution clusters both fall closest to the same old centroid. The EMA averages them together, and no future update can separate them — the centroid is stuck at the midpoint. Full k-means retrain from reservoir data restarts without this bias.

### 4. Insert overhead is the trade-off

148× insert overhead at `update_freq=200` is the cost of correctness. Production options:
- Increase `update_freq` to 1000 → 40 retrains over 40,000 inserts → overhead amortizes
- Run retrain asynchronously in a background thread (serve stale codes during retrain window)
- Trigger retrain only when drift exceeds a threshold (ruFlo integration path)
- Reduce `TRAIN_ITERS` from 20 to 5 for faster convergence at acceptable quality loss

---

## Architecture

```
StreamPQ
  ├── raw_vecs: Vec<Vec<f32>>       — raw vectors for correct re-encoding
  ├── codes:    Vec<Vec<u8>>        — current PQ codes (M bytes each)
  ├── reservoir: Vec<Vec<f32>>      — Vitter uniform sample (cap = reservoir_cap)
  └── codebook: Option<Codebook>   — current trained codebook

On insert(v):
  1. encode v with current codebook → push to codes
  2. push v to raw_vecs
  3. reservoir_add(v) → Vitter update (seen_count++)
  4. inserts_since_update++; if >= update_freq:
     a. Codebook::train(reservoir, M, K, seen_count as seed)  ← full k-means
     b. codes = raw_vecs.map(|v| codebook.encode(v))          ← full re-encode
```

The full re-encode in step 4b is O(n_total × M × ds) per refresh. For n=10,000 and M=4, ds=16: 640,000 FLOPs per retrain — dominated by the k-means training cost.

---

## Production Integration Path

1. Land behind `features = ["stream-pq"]` in `ruvector-pq-search` (non-breaking).
2. Expose `reservoir_cap` and `update_freq` as runtime parameters.
3. Add ruFlo connector: monitor rolling cluster precision; trigger early retrain on drift signal.
4. Async retrain: serve current codebook while background thread retrains; atomic swap on completion.
5. Graduate to `ruvector-core` when recall advantage is confirmed on production-scale (1M+ vector) drift scenarios.

---

## Running the benchmark

```bash
# Default config
cargo run --release -p ruvector-streaming-qng --bin benchmark

# Custom config
N_PER_CLUSTER_A=1000 N_PER_CLUSTER_B=4000 DIMS=128 \
  cargo run --release -p ruvector-streaming-qng --bin benchmark

# Diagnostics (trace PQ codebook internals)
cargo run --release -p ruvector-streaming-qng --bin diagnose
```

---

## Files

```
crates/ruvector-streaming-qng/
  Cargo.toml
  src/
    lib.rs           — AnnVariant trait, Hit, recall_at_k, cluster_precision, sq_l2
    pq.rs            — Codebook: train (Lloyd's), encode, adc_table, adc_dist, update_one_pass
    full_precision.rs — FullPrecision: brute-force f32 baseline
    static_pq.rs      — StaticPQ: one-time build, no updates
    stream_pq.rs      — StreamPQ: Vitter reservoir + full k-means retrain
    dataset.rs        — Deterministic Phase-A/B generation with Gaussian noise
    bin/
      benchmark.rs   — Full two-phase benchmark with cluster-precision gates
      diagnose.rs    — Traces PQ codebook internals for debugging
```
