# How to Detect When Your AI Agent's Memory Has Drifted

*Three lightweight Rust detectors for streaming semantic drift in vector databases — with real benchmarks, zero false positives, and a MCP-ready event interface.*

---

## The Problem

Your AI agent writes embeddings to a vector database as it processes documents. After a few thousand inserts, something subtle goes wrong: the embedding distribution has shifted. Maybe the agent switched topics. Maybe you fine-tuned the model. Either way, the HNSW graph you built last week now has edges between vectors that no longer live near each other, and recall is silently degrading.

Most teams notice this problem months later, when a user complains that semantic search "feels off." A dedicated drift detector would have caught it in real time.

---

## The Solution: Three Complementary Detectors

We implemented three streaming drift detectors in Rust, all behind a single trait:

```rust
pub trait DriftDetector: Send + Sync {
    fn update(&mut self, vector: &[f32]);
    fn is_drifted(&self) -> bool;
    fn poll_event(&self) -> Option<DriftEvent>;
}
```

The `poll_event()` call is O(1) — it can safely be called on every single insert without impacting throughput.

### Detector 1: Window Centroid

The simplest detector. Computes the centroid of the last N vectors and compares it to a reference centroid:

```
drift_score = L2(ref_centroid, cur_centroid) / sqrt(dim)
```

The `sqrt(dim)` normalisation makes this dimension-agnostic: for unit-variance embeddings, the score is ~0.063 during stable operation (n=500) and ~0.44 for a 1σ mean shift. Set threshold to 0.10 and you get reliable detection with zero false positives.

**Performance: 100 ns per update, ~10M updates/sec, 1 KB memory.**

### Detector 2: Random Projection

Instead of one centroid, we project each vector onto 64 random directions (a Rademacher ±1/√d matrix) and track the mean in each projected dimension. Drift score is the RMS of all 64 per-projection mean shifts.

This catches **subspace drift** that a single centroid misses: if only 4 of 128 dimensions are drifting, a centroid comparison dilutes the signal 32×, but 64 random projections still capture it.

**Performance: 4 µs per update, 250k updates/sec, 32 KB memory.**

### Detector 3: Sentinel Query

During bootstrap, we fix 10 "sentinel" query vectors. After every 75 inserts, we compute the mean squared distance from each sentinel to all 300 recent vectors:

```
drift_score = mean_i( |ref_dist_i - cur_dist_i| / (ref_dist_i + 1) )
```

This is the most sensitive detector: it detects drift at **lag 175** vs. 500 for the other two. The mean-all-distance metric has only ~2% coefficient of variation on stable data (vs. 20%+ for top-k Jaccard, which suffers from the crowding effect in high dimensions).

**Performance: 55 ns p50 (2.5 µs mean due to amortised refresh), 155 KB memory.**

---

## Critical Design Detail: Freeze the Reference on Drift

The single most important design decision is what to do with the reference window after drift is detected.

**Wrong approach**: advance the reference window unconditionally. After drift, the new drifted vectors become the reference, the score drops to zero, and `is_drifted()` flickers rather than staying latched.

**Right approach**: only advance the reference when the current window is stable:

```rust
if self.last_drift_score <= self.threshold {
    self.ref_centroid = cur_centroid;  // stable: advance
}
// drifted: leave ref_centroid unchanged
```

With this policy, the reference stays frozen at the last pre-drift centroid, and every subsequent drifted window produces a high score.

---

## Benchmark Results (real numbers, not estimates)

Hardware: x86_64 linux, rustc 1.94.1. Dataset: 5,000 baseline + 5,000 drifted 128-d vectors.

| Detector | Mean update | Updates/sec | Detect lag | False positives | Memory |
|----------|------------|-------------|------------|-----------------|--------|
| WindowCentroid | 100 ns | 9,975,580 | 500 | 0 | 1 KB |
| ProjectionDrift | 4.0 µs | 251,189 | 500 | 0 | 32 KB |
| SentinelQuery | 55 ns p50 | 392,952 | **175** | 0 | 155 KB |

All three pass the acceptance test: detect 8σ drift within 2,000 vectors, zero false positives on a 5,000-vector stable baseline.

---

## MCP Integration

The `DriftEvent` carries a `DriftAction` that drives index maintenance decisions:

```rust
pub enum DriftAction {
    Observe,   // minor drift — log and watch
    Compact,   // significant shift — re-cluster agent memory
    Rebuild,   // severe change — rebuild the ANN index
}
```

In your insert loop:

```rust
index.insert(&embedding);
detector.update(&embedding);

if let Some(event) = detector.poll_event() {
    match event.action {
        DriftAction::Compact => index.schedule_compaction(),
        DriftAction::Rebuild => index.schedule_rebuild(),
        DriftAction::Observe => {}
    }
}
```

The MCP poller sees either `None` (O(1), no allocation) or a struct it can forward to the maintenance scheduler — no polling loop, no background threads, no overhead on the hot path.

---

## When to Use Which Detector

- **High-throughput path (>1M inserts/sec)**: WindowCentroid. Ten million updates per second, one kilobyte.
- **Multi-dimensional or subspace drift**: ProjectionDrift. Catches axis-unaligned shifts that centroid averaging dilutes.
- **Best detection latency**: SentinelQuery. Fires 2.8× faster (lag 175 vs 500) — useful for agent memory where topic pivots happen abruptly.
- **Production (belt-and-suspenders)**: Run all three; fire on the first one to trigger.

---

## Source Code

`crates/ruvector-semantic-drift` in the RuVector workspace. The benchmark binary reproduces all numbers above:

```bash
cargo run --release -p ruvector-semantic-drift --bin benchmark
```

All 18 unit tests pass. The crate has no dependencies beyond `rand` and `rand_distr` (already in the workspace).
