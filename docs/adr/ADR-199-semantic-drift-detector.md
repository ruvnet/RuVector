---
adr: 199
title: "Semantic Drift Detection for Agent Memory (ruvector-drift)"
status: accepted
date: 2026-06-11
authors: [ruvnet, claude-flow]
related: [ADR-196, ADR-197, ADR-193, ADR-189]
tags: [drift-detection, agent-memory, vector-search, psi, coherence, monitoring, ruvector-drift]
---

# ADR-199 — Semantic Drift Detection for Agent Memory

## Status

**Accepted (implemented).** Crate `crates/ruvector-drift` on branch
`research/nightly/2026-06-11-semantic-drift-detector`. All 13 tests pass;
benchmark ACCEPTANCE RESULT: PASS (9/9 variant–dataset combinations).

## Context

RuVector is a Rust-native cognition substrate for agents, not merely a static
vector database. A cognition substrate must answer a question that current
vector databases do not: *is the memory I am retrieving from still semantically
current?*

As AI agents use vector-backed memory over time, three categories of drift occur:

1. **Directional drift**: the mean of stored embeddings shifts as the agent
   encounters new topics. Retrievals become biased toward old topics.

2. **Distribution shape drift**: the spread or cluster structure of stored
   embeddings changes (new topics fragment existing clusters; old topics
   consolidate). The centroid stays stable but retrieval precision drops.

3. **Silent contamination**: adversarial or erroneous writes insert off-topic
   vectors that gradually poison the distribution without obvious centroid shift.

None of the major production vector databases (Milvus, Qdrant, Weaviate,
Pinecone, LanceDB, FAISS, pgvector, Chroma, Vespa) ship built-in drift
detection for stored embeddings. Monitoring is left to the application layer,
which rarely implements it.

This ADR documents the decision to add semantic drift detection as a first-class
ruvector primitive via a new `crates/ruvector-drift` crate.

## Decision

Implement `ruvector-drift` as a zero-dependency Rust library crate exposing a
`DriftDetector` trait with three concrete implementations:

### DriftDetector trait

```rust
pub trait DriftDetector {
    fn add_window(&mut self, window_id: u64, vectors: &[Vec<f32>]);
    fn detect(&self) -> DriftReport;
    fn is_drifted(&self) -> bool;
    fn name(&self) -> &'static str;
}
```

The trait is minimal by design — it can be implemented by future streaming,
incremental, or GPU-accelerated detectors without API breakage.

### Three variants

| Variant | Primary signal | Cost | Detects |
|---------|---------------|------|---------|
| `CentroidDrift` | L2 centroid shift / √D | O(N·D) | Mean shift |
| `PsiDrift` | PSI on cosine histograms (anchor=baseline centroid) | O(N·D + B) | Mean shift, fragmentation, bimodal split |
| `CoherenceDrift` | PSI + |Δintra-window coherence| | O(N·D + N²/sub-sample) | Above + cluster fragmentation |

### PSI anchoring decision

The PSI statistic uses the **baseline window centroid as a shared anchor** for
cosine similarity computation in both windows. This is a deliberate departure
from using each window's own centroid:

- Using own centroid: each window's cosine distribution is self-normalised →
  misses shifts where the centroid moves but the shape stays constant *relative
  to itself*.
- Using baseline anchor: measures "how different does the latest window look from
  the baseline reference frame?" — the correct framing for drift detection.

Consequence: PSI is numerically less stable when the baseline mean is near zero
(the anchor has small norm). This is documented as a known limitation and the
test suite enforces it.

### DriftConfig

```rust
pub struct DriftConfig {
    pub centroid_threshold: f32,  // default: 0.5 (normalised by sqrt(D))
    pub psi_threshold: f32,       // default: 0.25 (industry standard)
    pub coherence_weight: f32,    // default: 0.4 (weight of coherence term)
    pub psi_buckets: usize,       // default: 10
    pub min_window_size: usize,   // default: 10
}
```

Thresholds are configurable. The defaults follow industry practice (PSI=0.25 is
the standard "significant drift" threshold from ML monitoring literature[^3]).

## Consequences

### Positive

- **No external dependencies**: the library compiles in any Rust environment
  including `no_std` contexts (minus serde, which can be feature-gated).
- **Trait-based extensibility**: new detector implementations can be added
  without changing existing code.
- **Composable ecosystem integration**: `DriftReport` is serialisable and can
  feed `mcp-gate` tool calls, `ruvector-verified` audit logs, and ruFlo
  workflow triggers.
- **Measured detection accuracy**: all variants achieve TPR=1.00, FPR=0.00 on
  the benchmark suite's large-drift dataset.

### Negative / Limitations

- **Two-window, not streaming**: requires two complete windows. Does not detect
  cumulative slow drift within a single window.
- **Near-zero mean instability**: PsiDrift and CoherenceDrift are numerically
  less stable for unit-sphere normalised embeddings. CentroidDrift is
  unaffected.
- **O(N²) coherence**: CoherenceDrift sub-samples to 200 vectors to bound cost,
  introducing ~5–10% coherence estimation error at N=5K.
- **No threshold calibration**: the default PSI=0.25 threshold comes from credit
  scoring features, not embedding spaces. Calibration on real workloads is
  future work.

## Alternatives Considered

### Alternative 1: KL divergence on embedding histograms

KL divergence is asymmetric and undefined when support of Q extends beyond P.
PSI avoids these issues via symmetric computation with ε-smoothing. PSI is also
better-understood in the ML monitoring community.

### Alternative 2: Maximum Mean Discrepancy (MMD)

MMD is theoretically stronger (two-sample test with known Type I/II error rates)
but is O(N²) in the naive formulation. At N=5K, this is 25M pair evaluations per
detection cycle — too expensive for agent memory monitoring without approximation.
MMD is a worthwhile future detector (ADR-200 candidate).

### Alternative 3: Per-dimension KS test

The KS test is well-understood but univariate. Applying it per dimension (D
tests) suffers from the multiple comparison problem. For D=128, controlling FWER
requires Bonferroni correction that would make the per-dimension threshold
extremely conservative. Not suitable for high-dimensional embeddings.

### Alternative 4: Integrate directly into ruvector-core

Placing drift detection inside `ruvector-core` would couple the detection
algorithm to the storage implementation. The separate crate design keeps the
detection logic independent and testable without a full ruvector-core instance.

## Implementation Plan

**Done (this ADR)**:
- `crates/ruvector-drift/` — library crate with `CentroidDrift`, `PsiDrift`,
  `CoherenceDrift`
- `DriftDetector` trait + `DriftReport` + `DriftConfig`
- 12 unit tests + 1 doc test, all passing
- Benchmark binary with TPR/FPR measurement

**Near-term (feature candidates)**:
- Incremental / streaming detector: running centroid + histogram update
- `ruvector-drift-wasm`: WASM-targetable build
- `mcp-gate` MCP tool surface: `memory/drift/check` and `memory/drift/status`

**Research direction (ADR-200 candidate)**:
- CUSUM accumulation across windows for slow drift
- Approximate MMD detector via random Fourier features
- Threshold calibration pipeline

## Benchmark Evidence

Measured with `cargo run --release -p ruvector-drift --bin benchmark`.
Hardware: x86-64, Intel Celeron N4020, Linux 6.18, rustc 1.94.1.

| Variant | N | D | mean_us | p50_us | p95_us | TPR | FPR | Acceptance |
|---------|---|---|---------|--------|--------|-----|-----|-----------|
| CentroidDrift | 500 | 64 | 25.54 | 22.00 | 43.00 | 1.00 | 0.00 | PASS |
| PsiDrift | 500 | 64 | 97.78 | 93.00 | 125.00 | 1.00 | 0.00 | PASS |
| CoherenceDrift | 500 | 64 | 1086.79 | 1081.00 | 1139.00 | 1.00 | 0.00 | PASS |
| CentroidDrift | 1000 | 128 | 204.72 | 208.00 | 313.00 | 1.00 | 0.00 | PASS |
| PsiDrift | 1000 | 128 | 525.28 | 522.00 | 645.00 | 1.00 | 0.00 | PASS |
| CoherenceDrift | 1000 | 128 | 2756.86 | 2744.00 | 2949.00 | 1.00 | 0.00 | PASS |
| CentroidDrift | 5000 | 128 | 1046.84 | 982.00 | 1338.00 | 1.00 | 0.00 | PASS |
| PsiDrift | 5000 | 128 | 2600.92 | 2556.00 | 2931.00 | 1.00 | 0.00 | PASS |
| CoherenceDrift | 5000 | 128 | 5813.39 | 5782.00 | 6946.00 | 1.00 | 0.00 | PASS |

Acceptance criterion: TPR ≥ 0.66 and FPR ≤ 0.33.
Overall: **9/9 PASS**.

## Failure Modes

1. **Near-zero mean**: PsiDrift returns inflated scores when anchor ≈ 0. Use
   `CentroidDrift` as primary for unit-normalised embeddings.

2. **Model version change**: replacing the embedding model causes instant
   PSI → ∞. The detector must be reset after model replacement.

3. **Intentional topic broadening**: an agent deliberately learning about new
   topics will trigger PsiDrift. This is a true positive for distribution shift,
   but may not require compaction. Downstream systems must distinguish
   "expanding knowledge" from "stale contamination" via context.

4. **Slow cumulative drift**: gradual drift accumulating over many windows below
   the per-window threshold will not be detected. Requires CUSUM accumulation
   (future work).

## Security Considerations

- Drift detectors operate on aggregate statistics (centroids, histograms, pairwise
  similarities), not on individual stored documents. Logs and exports are safe.
- `DriftReport.details` is serialisable and suitable for append-only audit logs.
- An adversary who can write to agent memory will cause PSI > 0.25 if they insert
  sufficiently off-distribution vectors. This makes `ruvector-drift` a lightweight
  first-line detector for adversarial memory injection.
- Combined with `ruvector-verified` proof-gated writes, drift detection provides
  a forensic trail: timestamped evidence of *when* a distribution shifted.

## Migration Path

`ruvector-drift` is a new, standalone crate. It has no API surface in
`ruvector-core`. Integration is opt-in:

1. Add `ruvector-drift` as a dependency.
2. Wrap any `Vec<Vec<f32>>` vector collection with a windowing adapter.
3. Periodically call `add_window` + `detect` and handle `DriftReport`.

There is no breaking change to existing ruvector APIs.

## Open Questions

1. What PSI threshold should we recommend for production embedding workloads?
   (Requires calibration on real agent memory datasets.)

2. Should `DriftDetector` support async `add_window` for streaming use cases?

3. Should `ruvector-core` optionally embed a `DriftDetector` slot in its
   collection type, so drift is tracked automatically on every insert batch?

4. What is the right granularity for windows? Fixed-size batches (every 1K
   inserts)? Time-based (every hour)? Query-distribution-based (every 10K
   queries)?

---

[^3]: PSI threshold: industry standard from US federal banking credit risk
guidelines. PSI < 0.10 = stable, 0.10–0.25 = monitor, ≥ 0.25 = significant
drift. Widely cited in MLOps literature (EvidentlyAI, Seldon Core, etc.).
