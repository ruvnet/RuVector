# ADR-273: Semantic Drift Detection for Agent Vector Memory

**Status**: Proposed  
**Date**: 2026-08-01  
**Author**: Nightly Research Agent  
**Branch**: `research/nightly/2026-08-01-semantic-drift-ann`  
**Crate**: `crates/ruvector-semantic-drift`  
**Related**: ADR-240 (Coherence-HNSW), ADR-268 (Capability-Gated ANN), ADR-272 (Speculative ANN)

---

## Context

Agent vector memories accumulate embeddings continuously. When the underlying distribution of those embeddings drifts — because the agent's topic focus shifts, the document corpus changes, or the embedding model is updated — three consequences follow:

1. **ANN recall degrades**: HNSW graph edges were built for the old distribution; new queries land in graph regions with poor neighbour coverage.
2. **Cluster assignments become stale**: IVF centroids no longer represent the live distribution, increasing quantisation error.
3. **Compaction is wasted**: LSM-style compaction merges segments whose statistical properties are now incompatible.

No existing RuVector component detects that a distribution shift has occurred. Index maintenance (compaction, rebuild) is triggered manually or on a fixed schedule, leading to either stale indexes or unnecessary rebuilds.

**Prior work.** Distribution shift detection has a rich literature:
- *MMDEW* (Gretton et al., arXiv:2205.12706) tracks a Gaussian kernel MMD on exponential windows; exact but O(n²) per window.
- *Random projection change-point detection* (arXiv:1505.06770, 2602.19988) compresses d-dimensional data to k << d before applying classical statistics.
- *DriftLens* (arXiv:2406.17813) uses Fréchet Distance on PCA-projected embeddings; designed for offline batch evaluation.
- *Ada-IVF* (arXiv:2411.00970) detects IVF centroid staleness and schedules incremental maintenance.
- *STALE* (arXiv:2605.06527) benchmarks agent memory invalidation rates under topic drift.

All of these either require O(n²) compute, offline batch processing, or are designed for IVF-specific maintenance. None expose a lightweight streaming interface that a MCP tool can poll at microsecond cost.

---

## Decision

Introduce `crates/ruvector-semantic-drift` implementing three complementary streaming drift detectors behind a common `DriftDetector` trait. Each detector outputs a `DriftEvent` with a recommended `DriftAction` (Observe / Compact / Rebuild) that MCP tools and index maintenance loops can consume directly.

### Detector 1 — `WindowCentroidDetector`

Maintains a reference centroid and a current centroid over sliding windows of `window_size` vectors each. Drift score = normalised L2:

```
score = L2(ref_centroid, cur_centroid) / sqrt(dim)
```

Normalising by `sqrt(dim)` makes the threshold dimension-agnostic: for N(0,I) data the expected IID score is `sqrt(2/n)` regardless of d. A 1σ mean shift produces score ≈ 1.0.

The reference window advances **only on stable windows** (score ≤ threshold). Once drift fires, the reference is frozen at the last stable centroid, ensuring subsequent drifted windows are all compared against the pre-drift baseline.

**Update cost**: O(d). **Memory**: 2d × 4 bytes (two centroid vectors). **Detection lag**: at most `window_size` vectors.

### Detector 2 — `ProjectionDriftDetector`

Projects each vector onto a k × d Rademacher random matrix (entries ±1/√d, drawn once at construction). Tracks per-projection running means in the current and reference windows. Drift score = RMS of normalised per-projection mean shifts:

```
score = sqrt( mean_i( (ref_mean_i − cur_mean_i)² ) )
```

This approximates the random-projection MMD estimator: the Johnson–Lindenstrauss lemma guarantees that mean shifts in the projected space correspond to shifts in the original distribution. Using k = 64 projections captures directional centroid shifts independently in 64 low-dimensional subspaces, giving better sensitivity than a single centroid comparison when drift is concentrated in a subspace.

**Update cost**: O(k·d). **Memory**: k·d × 4 + 2k × 4 bytes. **Detection lag**: at most `window_size` vectors.

### Detector 3 — `SentinelQueryDetector`

Samples `s` sentinel vectors from the bootstrap phase. Maintains a circular snapshot buffer of the most recent `snapshot_size` vectors. Every `snapshot_size/4` updates, refreshes:

```
cur_dist_i = mean_{v ∈ snapshot} sq_L2(sentinel_i, v)
score = mean_i( |ref_dist_i − cur_dist_i| / (ref_dist_i + ε) )
```

Using **mean-all-distance** rather than top-k overlap makes this metric stable for high-dimensional Gaussian data: the coefficient of variation is 1/√(snapshot_size) ≈ 2% for snapshot_size=300, vs. Jaccard top-k which suffers from the crowding effect where many neighbours have near-identical distances and their rank ordering is sensitive to noise.

Sentinels are drawn from the bootstrap buffer and never inserted into the snapshot, preventing zero-distance self-matches that would inflate the reference distance to ≈0 and make the ratio diverge.

**Update cost**: O(1) amortised (circular buffer insert). **Refresh cost**: O(s · snapshot_size · d). **Memory**: (s + snapshot_size) · d × 4 bytes.

### Shared `DriftEvent` / `DriftAction` interface

```rust
pub struct DriftEvent {
    pub source: String,
    pub score: f32,
    pub vectors_seen: u64,
    pub action: DriftAction,
}

pub enum DriftAction {
    Observe,   // score in [threshold, 0.40)
    Compact,   // score in [0.40, 0.75)
    Rebuild,   // score >= 0.75
}
```

MCP pollers call `detector.poll_event()` on every insert (O(1)). The returned `DriftAction` drives index maintenance scheduling without requiring a human operator.

---

## Consequences

### Positive

- **Zero false positives** in a 5,000-vector IID baseline with all three detectors at their default thresholds (measured, not estimated).
- **Fast detection**: WindowCentroid and ProjectionDrift detect 8σ shift within 500 vectors (one window); SentinelQuery detects within 175 vectors.
- **Constant-memory** per detector: 1 KB (centroid), 32 KB (projection, k=64 d=128), 155 KB (sentinel, s=10 snapshot=300 d=128).
- **MCP-ready**: `poll_event()` returns `None` in O(1) during normal operation; event fires only on threshold crossing.
- **Dimension-agnostic thresholds**: normalised-L2 threshold of 0.10 applies across any embedding dimension for unit-variance models.

### Negative

- `SentinelQueryDetector` requires a bootstrap phase of `snapshot_size` vectors before it can fire; it is silent during cold-start.
- Refresh cost O(s · snapshot_size · d) is paid every `snapshot_size/4` updates. For s=10, snapshot_size=300, d=128 this is ~384,000 multiplications per refresh — negligible at 4 updates/sec, but non-trivial at 100,000 updates/sec.
- All three detectors detect centroid shift. They do **not** detect covariance change (distribution flattening/sharpening) without a shift in the mean. A second-moment extension would require O(d²) storage.

### Neutral

- The detectors are stateless across process restarts. A checkpoint/restore mechanism would be needed for production deployments with long-lived agent memories.

---

## Benchmark Results (2026-08-01, x86_64 linux, rustc 1.94.1)

Dataset: 5,000 baseline N(0,I)¹²⁸ + 5,000 drifted N(8·e₀, I)¹²⁸

| Variant | Mean update | p50 | p95 | Updates/sec | Detect lag | FP | Memory |
|---------|------------|-----|-----|-------------|------------|----|----|
| WindowCentroid | 100 ns | 48 ns | 286 ns | 9,975,580 | 500 | 0 | 1 KB |
| ProjectionDrift | 4.0 µs | 3.7 µs | 4.6 µs | 251,189 | 500 | 0 | 32 KB |
| SentinelQuery | 2.5 µs† | 55 ns | 381 ns | 392,952 | 175 | 0 | 155 KB |

†Mean includes amortised refresh cost (refresh every 75 updates, O(s·snapshot·d) each).

All detectors: **ACCEPTANCE PASS** — detected within 2,000 vector lag, 0 false positives.

---

## Alternatives Considered

**MMDEW (Gaussian kernel MMD)**: Exact distribution distance, handles covariance shifts. Rejected: O(n²) per window is prohibitive at >1k vectors/sec.

**Fréchet Distance (DriftLens style)**: Requires PCA fit and offline batch mode. Rejected: incompatible with streaming insert path.

**Top-k Jaccard overlap**: Intuitive sentinel metric. Rejected: crowding effect in high-dimensional Gaussian data makes Jaccard unstable (neighbours are near-equidistant, rank ordering is noise-sensitive). Mean-all-distance has CoV ≈ 1/√n, making it far more stable.

**Exponential moving average (no windows)**: Simpler state. Rejected: EMA centroid never forgets, so a long stable period after drift causes the EMA to slowly absorb the new distribution. Window-based detectors freeze the reference on drift, maintaining sensitivity indefinitely after onset.
