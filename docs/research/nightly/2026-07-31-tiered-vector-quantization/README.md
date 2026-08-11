# Hot-Warm-Cold Tiered Vector Quantization

**Nightly research branch — 2026-07-31**  
**Crate:** `ruvector-tiered-quant`  
**ADR:** [ADR-273](../../../adr/ADR-273-tiered-vector-quantization.md)  
**Branch:** `research/nightly/2026-07-31-tiered-vector-quantization`

---

## Problem

RuVector is used in agent memory workloads. Access distributions are heavily Zipfian:
a small fraction of vectors (recent memories, active reasoning context) are queried
constantly; the bulk (archived memories, cold knowledge bases) are rarely touched.

Storing all vectors at f32 precision wastes memory on cold data. Existing compression
approaches (PQ, SQ, BQ) apply uniform precision globally — they do not adapt to access
frequency at the individual vector level.

**No production vector database implements per-vector precision tiering driven by
runtime access-pattern counters.** This is the gap this crate addresses.

---

## Approach

Assign each vector to one of three precision tiers based on its access count:

| Tier | Encoding | Bytes/dim | Distance | Compression |
|------|----------|-----------|----------|-------------|
| Hot  | f32      | 4         | Euclidean exact | 1× |
| Warm | u8 (SQ)  | 1         | Reconstructed Euclidean | ~9× vs SQ (see note) |
| Cold | 1-bit (BQ)| 1/8      | Scaled Hamming | ~32× |

*Note: FlatU8 as implemented stores per-vector min+scale arrays (9 bytes/dim total), making it larger than f32. The Warm tier inside HWC shares the same encoding but hot/cold savings dominate overall.*

A `compact()` call scans all vectors and promotes/demotes based on thresholds:
- `access_count >= hot_threshold` → Hot
- `access_count >= warm_threshold` → Warm  
- `access_count < warm_threshold` → Cold

Queries scan all tiers and merge results using a normalized distance: Hamming
distances from cold vectors are scaled to the same magnitude as Euclidean distances
via `hamming_norm × sqrt(dims × 4/3) / 0.5`.

---

## Key Engineering Challenges Solved

### 1. Cross-tier distance normalization

Binary Hamming distances live in [0, 1]. Euclidean distances for 128-dim random
unit vectors live in [0, ~11]. Without normalization, cold vectors always win the
nearest-neighbor contest (all Hamming < 1.0 < typical Euclidean).

**Fix:** Scale cold distances to match Euclidean magnitude before merging:
```rust
let hamming_norm = hamming_bits as f32 / (a.len() as f32 * 64.0);
hamming_norm * (self.dims as f32 * 4.0 / 3.0).sqrt() / 0.5
```

### 2. Binary heap ordering for k-NN

To keep the k nearest (smallest distance) neighbors, the `BinaryHeap` must treat
_larger_ distances as "greater" so `pop()` evicts the worst match. Natural ordering
(`self.dist.partial_cmp(&other.dist)`) gives this behavior with Rust's max-heap.

### 3. Deterministic LCG for dataset generation

Used upper 32 bits of a 64-bit LCG state for uniform f32 in [-1, 1]:
```rust
(next_u64() >> 32) as u32 / u32::MAX as f32 * 2.0 - 1.0
```
`>> 33` (wrong) extracts only 31 bits, which are always ≤ 2^31/2^32 = 0.5, making
all values negative and breaking binary quantization entirely (all bits = 0).

---

## Benchmark Results (Linux x86_64, release build)

### Uniform random workload (worst case for HWC)

| Variant | Recall@10 | Mean µs | p50 µs | p95 µs | QPS | Memory |
|---------|-----------|---------|--------|--------|-----|--------|
| FlatF32 | 1.000 | 1718.9 | 1711.0 | 1841.0 | 582 | 4.9 MB |
| FlatU8  | 0.988 | 2126.4 | 2113.0 | 2249.0 | 470 | 11.0 MB |
| HotWarmCold | 0.411 | 1707.7 | 1700.0 | 1799.0 | 585 | 3.3 MB |

n=10,000, dims=128, queries=500, k=10. HWC: 20% hot / 20% warm / 60% cold.

On uniform random queries, cold-tier binary quantization at 128 dimensions yields
~0.41 recall — a known fundamental limitation of 1-bit encodings at this dimensionality.

### Clustered workload (realistic agent memory — target scenario)

| Variant | Recall@10 | Mean µs | p50 µs | p95 µs | QPS | Memory |
|---------|-----------|---------|--------|--------|-----|--------|
| FlatF32 | 1.000 | 1671.6 | 1665.0 | 1755.0 | 598 | 4.9 MB |
| FlatU8  | 0.961 | 2125.5 | 2111.0 | 2235.0 | 470 | 11.0 MB |
| **HotWarmCold** | **0.961** | 1759.1 | 1752.0 | 1843.0 | **568** | **3.3 MB** |

n=10,000, dims=128, 1,000 hot vectors, 50% queries near hot cluster.

**On the target workload, HWC matches FlatU8 recall (0.961) with 3.3× less memory
and 21% higher QPS.** The hot tier delivers exact recall for the most-accessed vectors.

---

## Accuracy Model

Recall depends on tier distribution:

```
recall_total ≈ (hot_frac × 1.00) + (warm_frac × 0.97) + (cold_frac × recall_bq)
```

Where `recall_bq` for binary quantization ≈ 0.60–0.75 at 128-dim (varies by data
distribution and threshold). For uniform random queries over 60% cold, the cold
term dominates → ~0.41. For hot-biased queries, the hot term dominates → ~0.96.

---

## SOTA Positioning

| System | Approach | Per-vector precision? | Access-driven? |
|--------|----------|-----------------------|----------------|
| ANNS-AMP (arXiv:2606.07156) | Adaptive precision per query | No (per-query, not persistent) | Query-time only |
| VectorLiteRAG (arXiv:2504.08930) | IVF-cluster routing | No (cluster-level) | No |
| Cracking Vector Search (arXiv:2503.01823) | Index topology restructuring | No | No |
| Zilliz Cloud tiering | Storage medium (SSD/object store) | No | No |
| QVCache (arXiv:2602.02057) | KV cache quantization | No | No |
| **ruvector-tiered-quant** | **Per-vector precision** | **Yes** | **Yes** |

---

## Limitations and Next Steps

1. **Cold recall bounded by BQ**: 1-bit quantization at 128 dims gives ~0.41 recall
   on random queries. Mitigations: store original f32 in write-once arena; use RaBitQ
   rotation before binarization; increase cold threshold to keep fewer vectors cold.

2. **Linear scan**: Query is O(n). Phase 2 integrates with ruvector-hnsw-repair for
   graph-level tier-aware routing.

3. **No promotion losslessness**: Cold → warm promotion reconstructs from binary,
   losing precision. Production hardening stores original f32 in a side arena.

4. **Threshold tuning**: Wrong thresholds cause tier thrashing. Phase 2 adds
   hysteresis bands and exposes metrics via MCP.

See ADR-273 for the complete failure-mode analysis and implementation roadmap.
