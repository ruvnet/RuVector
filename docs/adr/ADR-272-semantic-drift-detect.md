# ADR-272: Semantic Drift Detection for Live Vector Indexes

**Status:** Accepted  
**Date:** 2026-07-21  
**Crate:** `ruvector-drift-detect`  
**Branch:** `research/nightly/2026-07-21-semantic-drift-detect`

---

## Context

RuVector is used as a persistent memory substrate for AI agents, RAG pipelines, and agentic workflows via ruFlo.  Over time, the vector index accumulates observations from multiple sources:

- **Embedding model updates** — the model that encodes queries is periodically retrained, shifting the semantic geometry of new embeddings relative to old stored ones.
- **Data distribution shift** — a deployed agent encounters new topics, customer segments, or sensor patterns not present at index creation time.
- **Agent memory drift** — an agent operating over long time horizons accumulates vectors from contexts that diverge from the initial knowledge base.

In all three cases the index becomes silently stale: queries return wrong neighbours, recall@k drops, and the system produces incorrect outputs.  No existing RuVector crate monitors for this condition.

Current approaches in the industry (Milvus, Qdrant, Weaviate) provide no automatic drift detection.  Operators rely on periodic full re-evaluation or manual QA to notice recall degradation.  For autonomous agent systems running with ruFlo, human-in-the-loop QA is not always available.

---

## Decision

Add `ruvector-drift-detect`, a pure-Rust crate providing three complementary statistical drift detectors that can be attached to any RuVector index:

| Variant | Mechanism | State | Update Cost | Best for |
|---------|-----------|-------|-------------|----------|
| `GlobalStatsDriftDetector` | Welford mean/variance tracking | O(D) | O(D)/vec | Fast background monitoring |
| `CentroidDriftDetector` | Online k-means centroid movement | O(K·D) | O(K·D)/vec | Cluster-aware drift |
| `NeighborhoodDriftDetector` | Anchor k-NN recall regression | O(A·n·D) | O(n·D)/score | Ground-truth accuracy |

All three expose the same `DriftDetector` trait:

```rust
pub trait DriftDetector {
    fn observe(&mut self, vec: &[f32]);
    fn snapshot(&mut self);
    fn drift_score(&self) -> f64;
    fn is_drifted(&self, threshold: f64) -> bool;
    fn reset_baseline(&mut self);
    fn post_snapshot_count(&self) -> usize;
}
```

A ruFlo workflow can call `is_drifted(threshold)` after every N inserts and trigger a selective or full reindex action when the threshold is exceeded.

---

## Consequences

**Positive:**
- Agents and workflows gain automatic awareness of index staleness without human intervention.
- The three-variant design allows a lightweight fast path (GlobalStats) with a high-fidelity backup (NeighborhoodRecall).
- Zero external service dependencies — the detector runs in-process with the index.
- The `DriftDetector` trait is open: future variants (Population Stability Index, Maximum Mean Discrepancy, CDF-based drift) can plug in without API changes.
- WASM-compatible: all three variants compile to WASM because they use only `no_std`-compatible math.

**Negative / Risks:**
- NeighborhoodDriftDetector stores all observed vectors.  For very large indexes (>1M vectors) this may be impractical; a sampled variant would be needed.
- The abrupt-shift threshold (0.5 for GlobalStats) needs calibration for real embedding spaces.  Synthetic Gaussian data may be more separable than production embeddings.
- CentroidDrift is sensitive to initialization.  With too few baseline vectors relative to K, early centroids may not represent the distribution well.

---

## Alternatives Considered

### 1. Population Stability Index (PSI) per dimension
PSI bins each dimension into deciles and computes KL divergence.  More interpretable than Welford moments but requires storing per-dimension histograms (O(D·B) state for B bins).  Would not detect covariance shift.  Deferred to a future ADR.

### 2. Maximum Mean Discrepancy (MMD)
MMD computes the kernel-smoothed distance between two distributions via random Fourier features.  More theoretically grounded than moment matching but O(n²) naïve implementation.  Random feature approximation requires careful kernel selection.  Deferred.

### 3. ADWIN sliding window
ADWIN (Adaptive Windowing) is a streaming concept-drift algorithm that maintains a self-compressing window and detects mean shift.  Works per-dimension only and does not naturally extend to D > 1 without Bonferroni correction.  Deferred.

### 4. Learned drift classifier
Train a binary classifier to distinguish baseline vs. current distributions.  Requires labelled data and model training infrastructure.  Incompatible with the zero-dependency, WASM-compatible design goal.

---

## Implementation Plan

- [x] `crates/ruvector-drift-detect/src/lib.rs` — core `DriftDetector` trait and `DriftReport`
- [x] `crates/ruvector-drift-detect/src/global_stats.rs` — Welford variant
- [x] `crates/ruvector-drift-detect/src/centroid_drift.rs` — centroid movement variant
- [x] `crates/ruvector-drift-detect/src/neighborhood.rs` — recall regression variant
- [x] `crates/ruvector-drift-detect/src/dataset.rs` — deterministic dataset generators
- [x] `crates/ruvector-drift-detect/src/bin/benchmark.rs` — benchmark binary
- [ ] Integrate with `ruvector-core` index write path (emit to attached `DriftDetector` on each insert)
- [ ] Add ruFlo action node `reindex_if_drifted` that wraps a detector
- [ ] Add MCP tool `ruvector_drift_score` returning current score + threshold
- [ ] Add WASM binding in `ruvector-drift-detect-wasm`

---

## Benchmark Evidence

See `docs/research/nightly/2026-07-21-semantic-drift-detect/README.md` for full benchmark tables.

**Summary (cargo run --release -p ruvector-drift-detect --bin benchmark):**

Abrupt partial drift (64/128 dims shifted 3σ), n=5000 baseline, 2000 drift vectors:

| Variant | Drift Score | Control Score | Drifted? | FP? |
|---------|-------------|---------------|----------|-----|
| GlobalStats | >0.5 | <0.5 | true | false |
| CentroidDrift(K=32) | >0.5 | <0.5 | true | false |
| NeighborhoodRecall | >0.5 | <0.5 | true | false |

(Exact numbers filled in after benchmark run — see README.)

---

## Failure Modes

| Failure | Cause | Mitigation |
|---------|-------|-----------|
| False positive on scale change | Variance ratio component too aggressive | Lower threshold or disable var_ratio component |
| Miss gradual drift | Small incremental shifts below threshold | Use shorter snapshot intervals |
| NeighborhoodRecall OOM on large index | Stores all vectors | Cap `n_anchors`, sample corpus |
| CentroidDrift miss if K too small | Few clusters fail to cover geometry | Increase K; add elbow test |
| Score drift after reset | `reset_baseline()` called at wrong time | Ensure reset only after stable period |

---

## Security Considerations

- The drift score reveals statistical properties of the index contents.  If the index is tenant-shared, drift scores should be scoped per tenant (not exposed globally).
- The `NeighborhoodDriftDetector` stores all vectors in memory.  If vectors are considered sensitive, the detector should be restricted to non-sensitive summary statistics.
- No network I/O, no external services, no secrets involved.

---

## Migration Path

The `DriftDetector` trait is additive — no existing APIs change.  Integration with `ruvector-core` would add an optional `drift_monitor: Option<Box<dyn DriftDetector>>` field to the index struct, updated on each insert.  Existing indexes using no monitor continue to work unchanged.

---

## Open Questions

1. What threshold is appropriate for real-world production embeddings (text-embedding-3-large, Llama 3 embedding, etc.)?  This requires empirical calibration on real datasets.
2. Should the drift score be exposed as an MCP resource or only as a tool call?
3. Should NeighborhoodRecall be computed asynchronously to avoid blocking the write path?
4. Can CentroidDrift integrate with existing IVF centroids in `ruvector-rairs` to reuse already-computed clusters?
