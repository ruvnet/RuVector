---
adr: 194
title: "Semantic Drift Guard — streaming drift detection for agent memory"
status: accepted
date: 2026-05-27
authors: [ruvnet, claude-flow]
related: [ADR-143, ADR-193]
tags: [drift, agent-memory, vector-search, coherence, compaction, nightly-research]
---

# ADR-194 — Semantic Drift Guard: streaming drift detection for agent memory

## Status

**Accepted.** Implemented on branch
`research/nightly/2026-05-27-semantic-drift-guard` as `crates/ruvector-drift`.
All 20 unit tests pass; all 6 acceptance tests pass; build is green with
`cargo build --release -p ruvector-drift`.

---

## Context

RuVector positions itself as a Rust-native cognition substrate for autonomous
agents. It already has:

- HNSW and DiskANN graph-based ANN search
- RaBitQ quantisation (`ruvector-rabitq`)
- ACORN filtered HNSW (`ruvector-acorn`)
- RAIRS IVF (`ruvector-rairs`)
- Graph coherence scoring (`ruvector-coherence`)
- Proof-gated writes (`ruvector-verified`)

What is missing: a mechanism to detect when the **semantic distribution** of
vectors being written to a memory store has **shifted** — and to tell the agent
(or ruFlo) to compact, re-index, or alert an operator.

Agent memory stores face a problem absent from traditional vector databases: they
are written continuously, session after session, as agent context changes. A store
that started as "Python coding assistant" may accumulate medical, legal, and
financial vectors without any retrieval-visible signal of the change. HNSW graph
quality silently degrades. Recall drops. The agent does not know why.

This ADR records the decision to add `crates/ruvector-drift` — three streaming
drift detectors with a unified `DriftDetector` trait.

---

## Decision

Add `crates/ruvector-drift` to the workspace. Implement three drift detector
variants behind a single trait:

```rust
pub trait DriftDetector: Send + Sync {
    fn observe(&mut self, vector: &[f32]) -> DriftScore;
    fn is_drifted(&self) -> bool;
    fn reset(&mut self);
    fn name(&self) -> &'static str;
    fn summary(&self) -> DriftSummary;
    fn compaction_hint(&self) -> Option<CompactionHint> { None }
}
```

### Variant 1: EwaDriftDetector
- Maintains an Exponential Weighted Average centroid of observed vectors.
- Drift score = 1 − cosine_sim(new_vector, centroid), EWA-smoothed.
- O(dim) per observation; 256 B overhead at dim=64.
- Suitable for the hot write path.

### Variant 2: WindowedVarianceDriftDetector
- Accumulates a fixed-centroid baseline during warmup.
- Sliding window of cosine similarities; alerts on mean-drop or variance spike.
- O(dim + W) per observation.
- Better than EWA for abrupt topic changes.

### Variant 3: GraphCoherenceDriftDetector
- Ring buffer of recent vectors.
- Pairwise cosine mean (all C(n,2) pairs) — sensitive to mixed-cluster windows.
- Also provides `compaction_hint()` — per-vector coherence below threshold
  → flagged for pruning.
- O(capacity² × dim) per observation; 313 µs at cap=96, dim=64.
- Use sampled (every Kth write) or on a background thread.

---

## Consequences

### Positive
- First drift monitoring capability in the RuVector ecosystem.
- Zero external dependencies; no service calls on the write path.
- `DriftDetector` trait extensible — future variants (CUSUM, topological) can be
  added without breaking the API.
- `CompactionHint` gives mincut-compatible vector IDs for targeted pruning.
- ruFlo can subscribe to drift alerts via `is_drifted()` poll or future event bus.

### Negative
- Manual threshold configuration required (no auto-calibration yet).
- GraphCoherence at 313 µs/obs is too slow for direct hot-path use at >3K vecs/s.
- Pairwise coherence is O(n²) — hard limit on `capacity` for interactive latency.
- Assumes clustered (semantic) embeddings; breaks for uniform random vectors.

### Neutral
- Does not change any existing crate API.
- New workspace member; no impact on current build times for other crates.

---

## Alternatives Considered

### A. Periodic batch drift test (KL divergence)
Store full distribution; test new batch with KL divergence.
Rejected: too expensive per-write; requires storing full reference distribution.

### B. External monitoring service (Prometheus + alerts)
Push embedding statistics to an external metric store; alert on deviation.
Rejected: violates zero-external-dependency design; adds deployment complexity.

### C. No drift detection, rely on scheduled compaction
Accepted by Milvus, Qdrant, etc.: time-based or size-based compaction triggers.
Rejected as insufficient: misses semantic drift entirely; compaction does not
remove semantically stale vectors.

### D. CUSUM control chart
Statistically principled; cumulative sum test for distribution shift.
Deferred: requires careful calibration; implement as a fourth variant in a
follow-up.

---

## Implementation Plan

1. ~~`crates/ruvector-drift` created with EWA, WV, GC detectors~~ ✓ (this PR)
2. Integration with `ruvector-core::VectorStore` write path (next sprint)
3. Auto-calibration of thresholds from first N stable writes (follow-up)
4. `ruvector-drift-mcp`: MCP tool surface for agent orchestrators
5. Multi-detector consensus voting (2-of-3)
6. `ruvector-drift-witness`: drift events → ruvector-verified witness log

---

## Benchmark Evidence

All numbers from `cargo run --release -p ruvector-drift --bin benchmark`,
2026-05-27, x86-64 Linux 6.18, Intel Celeron N4020, rustc 1.87.0.

Dataset: dim=64, N_stable=800, N_drift=500, N_FP=300, cluster σ=0.25,
stable bias=6.0·e₀, drift bias=6.0·e₃₂.

| Detector | TP@50 | TP@100 | FP/300 | Mean lat (ns) | vecs/s |
|----------|-------|--------|--------|---------------|--------|
| EWA | YES | YES | 0 | 177 | 5,648,514 |
| WindowedVariance | YES | YES | 0 | 192 | 5,200,873 |
| GraphCoherence | YES | YES | 0 | 313,323 | 3,191 |

Recall@10: 1.0000 before and after compaction (38.5% index size reduction).

Six acceptance tests: ALL PASS.

---

## Failure Modes

| Failure | Cause | Mitigation |
|---------|-------|------------|
| FP on stable data | Threshold too tight | Widen threshold; auto-calibrate from warmup |
| Missed detection | Very gradual drift (< threshold) | Lower alpha; longer warmup baseline |
| FP on warm reset | Old smoothed score persists | `reset()` clears all state |
| GC too slow for hot path | O(cap²·dim) | Sample every Kth write; background thread |
| Centroid collapse | Zero-mean embeddings | Add centroid-magnitude guard |

---

## Security Considerations

- **Adversarial drift injection**: an attacker controlling some writes can gradually
  shift the distribution below detection threshold to poison the memory store.
  GraphCoherence (global window structure) is harder to fool than per-vector EWA.
- **Compaction manipulation**: a compaction hint could be used to selectively remove
  legitimate memories. Wire to `ruvector-verified` to create auditable compaction events.
- **No API keys or credentials**: `ruvector-drift` is pure Rust with no network calls.

---

## Migration Path

`ruvector-drift` is additive. No existing API changes. Future integration:

```rust
// In ruvector-core::VectorStore::insert (not yet implemented):
pub fn insert(&mut self, id: usize, vector: &[f32]) {
    self.inner.insert(id, vector);
    if let Some(detector) = &mut self.drift_detector {
        let score = detector.observe(vector);
        if score.alert {
            self.drift_events.push(score);
        }
    }
}
```

The `drift_detector` field is `Option<Box<dyn DriftDetector>>` — callers can
opt in without affecting existing code paths.

---

## Open Questions

1. **Threshold auto-calibration**: should calibration use the first N writes, or
   require an explicit calibration phase? How does the agent signal "this is stable data"?

2. **Multi-detector consensus**: which voting rule? 1-of-3 (sensitive), 2-of-3
   (balanced), 3-of-3 (conservative)?

3. **Drift severity ↔ recall degradation**: what is the empirical mapping between
   drift score and expected recall@10 drop? Needs real embedding model benchmarks.

4. **Persistence**: should drift state persist across restarts? If so, serialize
   which fields?

5. **ruFlo integration**: should drift alerts be pushed (event bus) or polled
   (`is_drifted()`)? Event bus requires a runtime dependency.
