---
adr: 194
title: "Streaming Semantic Drift Detection for Agent Vector Memory"
status: proposed
date: 2026-05-23
authors: [ruvnet, claude-flow]
related: [ADR-143, ADR-193, ADR-189]
tags: [drift-detection, agent-memory, vector-search, cusum, mmd, rff, streaming, online-statistics, nightly-research]
---

# ADR-194 — Streaming Semantic Drift Detection for Agent Vector Memory

## Status

Proposed — working PoC in `crates/ruvector-drift`.

---

## Context

RuVector is positioned as a cognition substrate for agents.  Agent vector memory
indexes accumulate insertions over long sessions, across multiple domains, and
from queries that evolve as the agent learns.  Without a mechanism for detecting
when the semantic distribution of insertions has changed, several silent failures
occur:

1. **HNSW graph optimized for a dead distribution.** Graph edges were built for
   the reference distribution.  Post-drift queries land in graph regions that no
   longer efficiently reach their true neighbors.
2. **IVF centroids go stale.** RAIRS (ADR-193) and any IVF index will assign
   drift-phase vectors to wrong clusters, degrading recall without error signals.
3. **ruFlo cannot schedule consolidation.** The autonomous workflow loop has no
   event to subscribe to; it can only poll recall metrics, which require queries.
4. **RVF snapshots lack semantic provenance.** A snapshot cannot say "this index
   was built when the agent was focused on topic X."

The solution is a lightweight, streaming drift detector that can be attached to
any RuVector index namespace with O(1)–O(D) overhead and fire within a handful
of insertions of a genuine distribution shift.

---

## Decision

Add `crates/ruvector-drift` to the RuVector workspace, providing:

1. A `DriftDetector` trait with five methods: `insert`, `drift_score`,
   `is_drifted`, `reset_reference`, `count`, `memory_bytes`.
2. Three implementations:
   - `MeanShiftDetector`: EMA mean-distance; O(D) space; 124 ns/insert (D=128).
   - `CusumDetector`: CUSUM on z-scored squared norms; **O(1) space** (48 B); 129 ns/insert.
   - `MmdRffDetector`: RFF-MMD; O(D × R) space; ~42 µs/insert (D=128, R=256).
3. A benchmark binary (`drift-bench`) producing deterministic, auditable results.
4. Six passing unit tests covering: large drift detection, moderate drift, no-drift
   false positive ratio, CUSUM drift, MMD drift, memory sizing.

The CUSUM variant is the primary recommendation for production use due to its
48-byte state and near-optimal SPRT properties for mean-shift detection.  The
MmdRffDetector is the research-grade variant that detects arbitrary shifts
(not just mean shifts) when memory budget allows.

---

## Consequences

### Positive

- **Immediate utility:** Any RuVector user can attach a `CusumDetector` to an
  insert path with two lines of code and 48 bytes of overhead.
- **ruFlo integration:** drift score becomes an event source for workflow loops.
- **MCP exposure:** the `drift_score()` method maps directly to an MCP tool
  response field, enabling agents to query their own memory health.
- **No external dependencies:** the crate depends only on `rand = "0.8"` and
  `serde = "1"` — no network, no database, no OS services.
- **Trait-object safe:** `Box<dyn DriftDetector>` works, allowing runtime
  selection of variant based on memory budget.

### Negative

- **Threshold configuration burden:** all three variants require a threshold
  parameter.  The right threshold depends on embedding model, dimensionality,
  and application.  Adaptive thresholding is future work.
- **EMA natural variability:** with alpha=0.05 and D=128, the EMA mean-shift
  score has a natural noise floor of ~sqrt(D/n_eff) ≈ 1.79.  Callers must set
  thresholds above this floor.  (See test `mean_shift_drift_exceeds_nodrift_score`
  for the validated signal-to-noise ratio.)
- **MMD-RFF latency:** at 42 µs/insert, MmdRffDetector is not suitable for
  high-throughput insert paths (>100K/s) without SIMD optimization.

---

## Alternatives Considered

### 1. Offline recall monitoring

Instead of per-insert drift detection, run recall benchmarks periodically.

**Rejected:** recall measurement requires a query workload.  For an agent that
inserts but does not immediately query, this provides no signal.  Recall
measurement is also 2–4 orders of magnitude more expensive per data point.

### 2. ADWIN (adaptive windowing)

ADWIN (Bifet & Gavalda 2007) is a well-known drift detector with O(log n) space
and strong statistical guarantees.

**Deferred:** ADWIN requires storing a sliding window of observations, which for
D=128-dimensional vectors is O(D × W) space where W can grow large.  The scalar
CUSUM is sufficient for the mean-shift use case and simpler to reason about.
ADWIN on scalar projections is a natural follow-on.

### 3. Per-dimension monitoring

Track drift per embedding dimension independently.

**Rejected:** for typical embedding models, drift is a global shift in the
embedding space, not per-dimension.  Per-dimension monitoring produces D
independent tests with correlated p-values, requiring Bonferroni correction
and increasing false-positive probability.

### 4. KL divergence on quantized histograms

Maintain per-dimension histograms and compute KL divergence.

**Rejected:** histogram memory scales as O(D × bins); for D=128, bins=100 this
is 100 KB per dimension.  KL divergence estimation from histograms requires
careful smoothing and is sensitive to bin boundaries.  MMD-RFF achieves the same
goal with better guarantees and known theory.

---

## Implementation Plan

### Phase 1 (now — PoC complete)

- [x] `DriftDetector` trait
- [x] `MeanShiftDetector`
- [x] `CusumDetector` (norm-based, O(1) space)
- [x] `MmdRffDetector` (RFF-MMD, R=256)
- [x] Deterministic benchmark binary
- [x] Six passing unit tests
- [x] Workspace integration (`Cargo.toml` member)

### Phase 2 (next sprint)

- [ ] Adaptive threshold calibration from reference variance
- [ ] Serde serialization for checkpoint/restore
- [ ] `no_std + libm` compilation path
- [ ] MCP tool wrapper in `mcp-brain`

### Phase 3 (production hardening)

- [ ] SIMD RFF kernel (AVX2/AVX-512) for MMD-RFF
- [ ] Per-graph-community drift via `ruvector-mincut` integration
- [ ] Ensemble detector (CUSUM + MMD majority vote)
- [ ] ruFlo event binding (`on_drift` hook)
- [ ] HTTP endpoint in `ruvector-server`

---

## Benchmark Evidence

All numbers from `cargo run --release -p ruvector-drift`, rustc 1.94.1, x86-64 Linux.

**Detection experiment:** D=128, N=2000 (1000 reference + 1000 drift), drift magnitude=2.0/dim.

| Variant   | Detection Lag | Insert Latency | Memory | Acceptance |
|-----------|--------------|----------------|--------|-----------|
| MeanShift | 1 vector      | 124 ns         | 3 KB   | PASS       |
| CUSUM     | 1 vector      | 129 ns         | 48 B   | PASS       |
| MMD-RFF   | 2 vectors     | 42 µs          | 133 KB | PASS       |

**Unit tests:** 6/6 pass (`cargo test -p ruvector-drift`).

---

## Failure Modes

| Scenario | Consequence | Mitigation |
|----------|-------------|-----------|
| Threshold too low | Spurious drift events; unnecessary rebuilds | Calibrate on reference hold-out; use CUSUM with higher slack |
| Threshold too high | Real drift missed; silent recall decay | Use MmdRffDetector with wider statistical power |
| Very slow drift (< α per step) | EMA adapts to drift; goes undetected | Reduce alpha; consider ADWIN |
| Adversarial vector injection | False drift trigger or drift masking | Require `ruvector-verified` proof chain on writes |
| Large D (D > 1024) | MeanShift L2 score scales as √D; threshold must scale | Normalize score by √D or switch to normalized cosine distance |

---

## Security Considerations

1. Drift detection operates only on aggregate statistics (means, norms) — no
   individual vector content is stored or exposed.
2. An adversary controlling inserts could trigger drift events (DoS via rebuild
   triggers).  Rate-limit rebuild actions per drift event.
3. For regulated environments, the drift log (trigger time, score, count) should
   be written to `ruvector-verified`'s witness chain for audit.

---

## Migration Path

`ruvector-drift` is a new standalone crate with no changes to existing APIs.

To adopt in an existing RuVector index:

```rust
use ruvector_drift::{CusumDetector, DriftDetector};

let mut detector = CusumDetector::new(dim, warm_up, slack);

// In your insert loop:
detector.insert(&vector);
if detector.is_drifted(threshold) {
    // schedule rebuild / emit event / notify agent
    detector.reset_reference();
}
```

No breaking changes to `ruvector-core`, `ruvector-server`, or any other crate.

---

## Open Questions

1. Should drift detection be integrated into `ruvector-core`'s `Index` trait
   directly, or remain a side-channel?  (Current: side-channel.  Rationale:
   keeps `ruvector-core` dependency-free.)
2. What is the right default alpha and threshold for the embedded use case
   (D=128, Cognitum Seed)?  Needs calibration on real agent memory traces.
3. Should the drift detector be seeded from existing index statistics at
   startup (for warm restart), or always start cold?
4. Can `MmdRffDetector` be made `no_std` with only `libm` as a dependency?
   (Likely yes; `cos()` from `libm::cosf` should suffice.)
