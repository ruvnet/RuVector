# SQ-HNSW: Scalar-Quantized HNSW with Calibrated Search in Rust

A minimal, dependency-free Rust implementation of three HNSW index variants that demonstrate **scalar quantization (SQ8) inside the graph traversal loop** — not just as a storage format.

## The Core Insight

Most vector databases store vectors as int8 to save memory but convert back to f32 before computing distances.  SQ-HNSW keeps the entire search in the int8 domain:

```
HNSW beam search → i8 distances → i8 neighbor ranking → rerank top-k with f32
```

This is faster (integer arithmetic), smaller (4× less DRAM), and — with calibrated quantization — barely changes recall.

## Three Variants, One Trait

```rust
pub trait AnnIndex {
    fn add(&mut self, id: usize, vector: Vec<f32>);
    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult>;
    fn len(&self) -> usize;
    fn bytes_per_vector(&self) -> usize;
}
```

| Variant | Storage | Traversal | Rerank | Mem/vec (128d) |
|---------|---------|-----------|--------|----------------|
| `F32Index` | f32 | f32 | — | 512 B |
| `Sq8Index` | i8 | i8 | — | 128 B |
| `Sq8RerankIndex` | i8 + f32 | i8 | f32 | 640 B |

## Calibrated Quantizer

```rust
pub struct ScalarQuantizer {
    pub dim_min: Vec<f32>,   // per-dimension lower bound
    pub dim_scale: Vec<f32>, // per-dimension (max - min)
    pub dims: usize,
}

impl ScalarQuantizer {
    pub fn calibrate(calibration_set: &[Vec<f32>]) -> Self { ... }

    pub fn encode(&self, v: &[f32]) -> Vec<i8> {
        // q[d] = round((x[d] - min[d]) / scale[d] * 254 - 127)
        // maps [min, max] → [-127, 127]
    }

    pub fn sq8_l2_sq(a: &[i8], b: &[i8]) -> i64 {
        // i64 accumulator: max term = 255², max sum = 255² × 1024 dims ≈ 66M → fits in i64
        a.iter().zip(b.iter()).map(|(&ai, &bi)| {
            let d = (ai as i64) - (bi as i64);
            d * d
        }).sum()
    }
}
```

**Online calibration strategy:** Collect the first `N` vectors, compute per-dimension min/max, freeze the quantizer, then flush the buffer into the graph.  No rebuild required.

## The Critical HNSW Bug (and Fix)

A common implementation mistake is giving `insert_node` a single-argument closure `dist_to_new(j)`.  This works for the greedy search phase but **silently destroys neighbor pruning**:

```rust
// BUG: when nb's neighbor list is full, we can't prune by distance from nb
fn insert_node(&mut self, new_id: usize, dist_fn: impl Fn(usize) -> f32)

// FIX: two arguments let pruning use the correct reference point
fn insert_node(&mut self, new_id: usize, dist_fn: impl Fn(usize, usize) -> f32)
```

With the single-arg closure, neighbor lists are truncated by insertion order rather than distance.  Observed effect: recall@10 drops from 0.77 to 0.03 on the same dataset.

The two-arg fix:

```rust
pub fn insert_node(&mut self, new_id: usize, dist_fn: impl Fn(usize, usize) -> f32) {
    let dist_to_new = |j: usize| dist_fn(new_id, j);
    // ... beam search using &dist_to_new ...
    // During reverse-link pruning:
    let mut cands: Vec<_> = self.nodes[nb].neighbors[0]
        .iter()
        .map(|&x| (OrderedFloat(dist_fn(nb, x)), x))  // ← dist from nb, not from new_id
        .collect();
    cands.sort_unstable();
    // truncate to m0 nearest
}
```

## Benchmarks (real numbers, n=10k, dims=128, k=10)

```
┌─────────────────┬──────────┬────────────┬────────────┬─────────────┬──────────────┐
│ Variant         │ Recall@10│ Mean(μs)   │ p95(μs)    │ QPS         │ Mem/vec(B)   │
├─────────────────┼──────────┼────────────┼────────────┼─────────────┼──────────────┤
│ F32 (baseline)  │ 0.7704   │      396.7 │      464.0 │        2521 │          512 │
│ SQ8 (no-rerank) │ 0.7682   │      256.6 │      302.3 │        3897 │          128 │
│ SQ8 + Rerank    │ 0.7690   │      270.6 │      315.9 │        3696 │          640 │
└─────────────────┴──────────┴────────────┴────────────┴─────────────┴──────────────┘
```

- SQ8 vs F32: **−35% latency, 4× less memory, −0.3% recall**
- SQ8+Rerank vs F32: **−32% latency, recall nearly identical**
- Tested on: linux x86_64, rustc 1.94.1, `--release`

## Rerank Pattern

```rust
// Phase 1: graph traversal with int8 distances (fast)
let overquery_k = k * overquery_factor;
let candidates = self.graph.search(ef_search.max(overquery_k), overquery_k, |j| {
    ScalarQuantizer::sq8_l2_sq(&q_code, &codes[j]) as f32
});

// Phase 2: exact f32 rerank on candidates (precise)
let mut reranked: Vec<(f32, usize)> = candidates
    .into_iter()
    .map(|(_, id)| (l2_sq(query, &originals[id]), id))
    .collect();
reranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Equal));
reranked.into_iter().take(k).map(|(d, id)| SearchResult { id, distance: d }).collect()
```

`overquery_factor = 3` is a reasonable default: fetch 3k candidates from the i8 graph, rerank with exact f32, return k.

## Running It

```bash
git clone https://github.com/ruvnet/ruvector
cd ruvector
cargo run --release -p ruvector-sq-hnsw

# Larger run
N=100000 DIMS=256 cargo run --release -p ruvector-sq-hnsw

# Tests
cargo test -p ruvector-sq-hnsw
```

## Source

- Crate: `crates/ruvector-sq-hnsw/`
- Research: `docs/research/nightly/2026-07-07-sq-hnsw-calibrated-search/README.md`
- ADR: `docs/adr/ADR-272-sq-hnsw-calibrated-search.md`
- Part of the [RuVector](https://github.com/ruvnet/ruvector) ANN ecosystem
