# ADR-272: Semantic Drift Detection for Agent Memory Streams

**Status:** Proposed  
**Date:** 2026-07-22  
**Author:** Nightly Research Agent  
**Crate:** `crates/ruvector-semantic-drift`

---

## Context

Long-running AI agents accumulate embedding vectors in RuVector as persistent
memory.  Over time — through topic shifts, user context changes, model updates,
or data-quality degradations — the distributional properties of those embeddings
drift away from the baseline the agent was calibrated on.

Current RuVector crates address temporal decay (`ruvector-temporal-coherence`)
and stale memory compaction (`ruvector-agent-memory`).  Neither detects
*distributional shift* in the embedding space itself: the subtle statistical
signature that the entire memory corpus has changed domain.

Semantic drift detection fills this gap.  It monitors the stream of embeddings
fed into agent memory and raises a signal when the distribution diverges from
a learned baseline — enabling ruFlo workflows to trigger re-calibration,
selective compaction, or user notification.

---

## Decision

Introduce `ruvector-semantic-drift`, a pure-Rust, zero-dependency crate
implementing three online drift detectors behind the `DriftDetector` trait:

| Variant | Algorithm | Memory | Latency | Sensitivity |
|---------|-----------|--------|---------|-------------|
| `CentroidEMA` | EMA centroid cosine displacement | O(2·d) | O(d) | centroid shifts |
| `CovarianceTrace` | Welford variance trace + centroid | O(3·d) | O(d) | variance explosions + centroid |
| `SlidingWindowKL` | Pairwise-cosine histogram KL divergence | O(2·w·d) | O(w²) | distributional shape changes |

`d` = embedding dimension, `w` = window size.

### API Shape

```rust
pub trait DriftDetector: Send + Sync {
    fn feed(&mut self, embedding: &[f32]);
    fn drift_score(&self) -> f32;          // [0, 1]
    fn is_drifted(&self) -> bool;
    fn reset_baseline(&mut self);
    fn name(&self) -> &'static str;
    fn sample_count(&self) -> usize;
    fn memory_bytes(&self) -> usize;
}
```

This shape is intentionally minimal so it can be composed with:
- `ruvector-agent-memory` compaction triggers
- `ruvector-temporal-coherence` decay scoring
- `ruvector-proof-gate` witness log annotations
- `ruFlo` event hooks (drift detected → re-calibrate workflow)

---

## Consequences

### Positive
- Operators can detect when agent memory has diverged, before it causes bad
  retrievals or stale reasoning.
- Three variants trade off speed, memory, and sensitivity — deploy whichever
  fits the agent's cadence and embedding dimension.
- No external dependencies; compiles for native, WASM, and edge targets.
- The `DriftDetector` trait is stable enough to serve as a first-class
  RuVector interface.

### Negative
- `SlidingWindowKL` is O(w²) per feed call — unsuitable for very large windows
  or very high throughput.  The `max_pairs` cap bounds this in practice.
- All three variants are heuristic: they detect distributional shift but cannot
  distinguish intentional topic changes from harmful memory corruption.
- Thresholds require per-deployment calibration.  No auto-calibration is
  included in this PoC.

---

## Alternatives Considered

| Alternative | Why Rejected |
|-------------|--------------|
| MMD (Maximum Mean Discrepancy) | Requires O(n²) kernel evaluations; no online form |
| ADWIN (drift detection on 1-D streams) | Not natively applicable to high-d embeddings |
| Model-based change detection (CUSUM) | Requires parametric distribution assumption |
| Full covariance matrix tracking | O(d²) memory; infeasible for d=1536 |
| Waserstein distance | No efficient online estimator; O(n log n) per query |

---

## Implementation Plan

1. **Phase 1 (this PR):** PoC crate with three variants, unit tests, benchmark binary.
2. **Phase 2:** Integration hooks in `ruvector-agent-memory` to call drift detector
   after each `insert()` and expose drift score via `MemoryStats`.
3. **Phase 3:** ruFlo trigger: `on_drift_detected` event fires a compaction or
   re-embedding workflow.
4. **Phase 4:** WASM build and MCP tool surface (`memory/drift_score` endpoint).

---

## Benchmark Evidence

See `docs/research/nightly/2026-07-22-semantic-drift/README.md` for full
benchmark output from:

```
cargo run --release -p ruvector-semantic-drift --bin benchmark
```

Acceptance criteria: all three variants detect a Δμ=1.5 shift within 100
samples at <5% false positive rate.

---

## Failure Modes

| Failure | Mitigation |
|---------|-----------|
| Embeddings not L2-normalised | Document pre-condition; add `debug_assert` in feed() |
| Threshold too low → alert fatigue | Expose threshold as constructor parameter; tune per deployment |
| Threshold too high → missed drift | Provide `drift_score()` for continuous monitoring |
| SlidingWindowKL stalls on large windows | `max_pairs` parameter caps O(n²) |
| Model change causes false drift | Provide `reset_baseline()` after known model updates |

---

## Security Considerations

- No network I/O, file I/O, or external dependencies.
- `memory_bytes()` accurately reports allocated heap so callers can enforce
  per-agent memory budgets.
- Drift scores must not be used as sole authorisation signal — an adversary
  who controls embedding content could keep scores below threshold.

---

## Migration Path

- Existing `ruvector-agent-memory` users: add a `DriftDetector` as a field in
  `MemoryStore`; call `detector.feed(embedding)` on each insert.
- No breaking changes to existing APIs.

---

## Open Questions

1. Should threshold auto-calibration be data-driven (e.g., percentile of
   historical scores during a burn-in period)?
2. Should `SlidingWindowKL` use a reservoir sample instead of pairwise to
   reduce cost for large windows?
3. Is there a natural ruFlo event type for `drift_detected`, or should we
   introduce a new one?
4. Does the `CovarianceTrace` variant need per-dimension normalisation before
   the trace is meaningful across heterogeneous embedding models?
