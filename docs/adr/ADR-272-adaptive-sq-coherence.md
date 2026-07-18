# ADR-272: Adaptive Scalar Quantization with Coherence-Precision Routing

**Status:** Accepted — Research PoC  
**Date:** 2026-07-17  
**Crate:** `crates/ruvector-adaptive-sq`  
**Branch:** `research/nightly/2026-07-17-adaptive-sq-coherence`

---

## Context

RuVector stores agent memories as floating-point vectors and retrieves them
via approximate nearest-neighbour (ANN) search.  Storage cost is a binding
constraint on edge deployments (Cognitum Seed, RVM, WASM runtimes) and on
long-running agent processes with growing memory stores.

Two scalar quantization options are widely deployed:

- **8-bit SQ (Uniform8):** halves memory vs f32 but introduces quantization
  noise that degrades recall by ~18% on clustered datasets (observed on our
  benchmark: 82.4% vs 100%).
- **16-bit SQ (Uniform16):** near-lossless recall but doubles memory vs 8-bit.

Neither option differentiates by the structural position of each vector.
Vectors in dense, contested embedding regions need high precision; vectors in
sparse regions can tolerate coarse quantization.  Uniform allocation is a
poor fit for the heterogeneous distributions that characterise agent memory.

---

## Decision

Introduce `ruvector-adaptive-sq`, a crate that implements **coherence-guided
precision routing**: each vector is assigned to either 8-bit (LP) or 16-bit
(HP) scalar quantization at insert time, based on its **density score** —
the mean L2 distance to its K nearest neighbours.

**Routing rule:**
```
density_score(v) ≤ mean(all scores) × threshold_factor  →  16-bit (HP)
density_score(v) >  mean(all scores) × threshold_factor  →   8-bit (LP)
```

The density score is a specific coherence signal: vectors with low mean kNN
distance are in tight, contested regions where quantization noise causes the
most recall disruption.

### API Shape (to Survive Into Production)

```rust
pub trait SqIndex {
    fn name(&self) -> &str;
    fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)>;
    fn memory_bytes(&self) -> usize;
    fn hp_ratio(&self) -> f32 { 0.0 }
}

pub struct AdaptiveSqIndex { ... }
impl AdaptiveSqIndex {
    pub fn build(
        vectors: &[Vec<f32>],
        dim: usize,
        knn_k: usize,
        threshold_factor: f32,
    ) -> Self;
}
```

The `SqIndex` trait and `AdaptiveSqIndex::build` signature are stable
candidates for the production API.

### Feature Flag

The density scoring path (O(N²) brute force) should remain behind a
`feature = "adaptive-build"` flag in a production integration.  The streaming
approximate variant (future work) should be the default.

---

## Consequences

### Positive

1. **Recall lift:** AdaptiveSQ achieves 95.2% recall vs 82.4% for Uniform8 on
   a structured benchmark, a +12.8 percentage point improvement.

2. **Memory efficiency:** AdaptiveSQ uses 62.5% of Uniform16 memory and 125%
   of Uniform8 memory — a good Pareto point between the two extremes.

3. **Search latency parity:** per-query latency is 421µs vs 410µs for Uniform8
   (+2.7%).  The routing table lookup and mixed-decode path add negligible
   overhead over a uniform scan.

4. **Correctness:** on the synthetic dataset, density scoring achieves 100%
   routing accuracy — all tight-cluster vectors land in HP, all loose-cluster
   vectors in LP.

5. **Zero external dependencies:** the crate core has no dependencies beyond
   `std`.  It compiles to WASM unmodified.

### Negative / Constraints

1. **Build time:** O(N²) density scoring takes 2.69 seconds at N=5000.
   At N=100,000 this is ~1,000 seconds.  Production requires approximate kNN.

2. **Static routing:** the tier assignment is fixed at build time.  If the
   memory distribution shifts significantly (common in long-running agents),
   routing decisions become stale without a periodic re-routing step.

3. **Routing table overhead:** 9 bytes per vector for `(Tier, usize)` routing
   entries.  At N=1 billion, this is 9 GB — itself requiring compression.

4. **Cross-tier distance asymmetry:** HP↔LP distance estimates use different
   precisions.  Formal error bounds are not yet derived.

---

## Alternatives Considered

### A: Uniform 8-bit with Residual Correction for Top-K

Apply 8-bit SQ globally, then correct the top-K candidates using f32 residuals
(as in `ResidualPqIndex` from `ruvector-pq-search`).  This avoids per-vector
routing but requires storing residuals for the top-K candidates, adding per-
query overhead.  **Rejected** because residual correction is query-time cost
rather than build-time amortisation.

### B: Learned Quantization (OPQ / LSQ)

Optimise a linear transformation to minimise global quantization error, as in
Optimized PQ[^1] or Learned Scalar Quantization[^2].  This requires a
calibration dataset and offline training.  **Rejected** because it conflicts
with RuVector's goal of working without an external ML pipeline.

### C: Per-Dimension Bit Allocation (VQ)

Allocate more bits to high-variance dimensions (PCA-informed).  This is
orthogonal to our approach and could be combined.  **Deferred** because it
requires a calibration matrix and makes decoding more complex.

### D: Matryoshka Coarse-Fine (benchmarked 2026-06-21)

Use different embedding dimensions for coarse and fine retrieval.  This is a
dimension reduction approach, not a precision routing approach.  Different
mechanism, complementary applicability.  **Not competitive** — addresses a
different tradeoff.

---

## Implementation Plan

### Phase 1 (Done — this ADR)

- [x] Brute-force density scoring (`coherence::density_scores`)
- [x] Mixed-precision index (`AdaptiveSqIndex`)
- [x] 17 unit tests, all passing
- [x] Benchmark binary with acceptance tests (both pass)
- [x] WASM-compatible (no external deps in library code)

### Phase 2 (Production Hardening)

- [ ] HNSW-based approximate density scoring — O(N log N) build time
- [ ] Streaming density score updates via reservoir sampling
- [ ] Minimum HP floor (never route fewer than 5% to HP)
- [ ] Percentile clipping for global bounds (remove outlier sensitivity)
- [ ] `AdaptiveSqIndex::rebalance()` method for periodic re-routing

### Phase 3 (Ecosystem Integration)

- [ ] MCP tool `vector_insert` with `precision: "auto"` hint
- [ ] ruFlo `memory_rebalance` step template
- [ ] `ruvector-proof-gate` integration for routing witness log
- [ ] WASM build target with streaming support

---

## Benchmark Evidence

All numbers from `cargo run --release -p ruvector-adaptive-sq --bin benchmark`
on x86_64 Linux, seed=42, N=5000, dim=32, 200 queries, k=10.

| Variant    | Recall@10 | Mean (µs) | Memory  | HP%  |
|------------|-----------|-----------|---------|------|
| Uniform8   | 0.8235    | 410.3     | 156 KB  | 0%   |
| Uniform16  | 1.0000    | 405.5     | 312 KB  | 0%   |
| AdaptiveSQ | 0.9520    | 421.1     | 195 KB  | 25%  |

Routing analysis: 1250/1250 tight-cluster vectors → HP (100%), 3750/3750
loose-cluster vectors → LP (100%).

Both acceptance tests pass:
- Recall: 0.9520 ≥ 0.93 × 1.0000 = 0.9300 ✓
- Memory: 195 KB ≤ 75% × 312 KB = 234 KB ✓

---

## Failure Modes

1. **Zero HP routing:** if `density_score` variance is low (uniform
   distribution), threshold may route nothing to HP.  Fix: enforce a minimum
   HP fraction.

2. **Build time explosion:** O(N²) density scoring becomes impractical at
   N>10,000.  Fix: Phase 2 HNSW-based approximate kNN.

3. **Routing staleness:** streaming agent memory changes cluster membership
   over time.  Fix: Phase 2 streaming density score updates.

4. **Outlier-blown global bounds:** a single extreme outlier can stretch the
   entire quantization range.  Fix: percentile clipping.

---

## Security Considerations

- Density scores reveal structural information about the dataset (which regions
  are dense).  They must be protected with the same access controls as the
  vector payloads.

- The routing table (tier bit per vector) can be included in the proof-gate
  witness log, making routing decisions verifiable and tamper-evident.

- Adversarial injection of dense-distribution queries to force HP routing on
  malicious content is a potential attack vector.  Query-based density score
  updates must be rate-limited or require authentication.

---

## Migration Path

1. The `SqIndex` trait is additive — existing flat-scan code is not touched.
2. `AdaptiveSqIndex` can coexist with `ruvector-pq-search` and
   `ruvector-coherence-hnsw` under different feature flags.
3. An existing Uniform8 index can be migrated offline: re-compute density
   scores, re-route vectors, rebuild.  No in-place mutation needed.

---

## Open Questions

1. What is the optimal `threshold_factor` for production agent memory
   distributions (non-Gaussian, multi-domain)?

2. How do density scores and routing decisions change during streaming
   inserts?  Is exponential moving average sufficient?

3. Can routing decisions be updated in O(1) amortised per insert, or do
   they require periodic full rebuilds?

4. What are the formal error bounds for cross-tier (HP↔LP) distance
   comparisons?

5. Would a 3-tier scheme (8-bit / 16-bit / f32) improve the Pareto frontier
   for extreme-precision requirements?

---

## Why This Belongs in RuVector

1. **Coherence scoring is native to RuVector.** The density score is a
   specific coherence signal, consistent with RuVector's coherence-gated
   search work (ADR-228, nightly 2026-06-16).

2. **Agent memory is the primary target.** Agent memory stores are
   structurally heterogeneous — exactly the case where uniform quantization
   wastes precision.

3. **Not just an experiment.** The acceptance tests pass on real measurements,
   the routing logic is exact on synthetic data, and the latency overhead is
   negligible.  This is a valid Pareto improvement over uniform quantization
   with a clear production path.

---

## Footnotes

[^1]: Ge, T., et al. (2013). Optimized product quantization. *CVPR 2013*.

[^2]: Martinez, J., et al. (2018). LSQ: Learned step size quantization. *arXiv:1902.08153*.
