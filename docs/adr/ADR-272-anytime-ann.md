# ADR-272: Anytime ANN Search with Budget-Aware Early Termination

- **Status**: Proposed (PoC implemented — `crates/ruvector-anytime-ann`)
- **Date**: 2026-07-01
- **Research doc**: `docs/research/nightly/2026-07-01-anytime-ann/README.md`

---

## Context

Standard HNSW beam search terminates when all remaining candidates are farther than the current kth result. This is correct for maximum recall but incompatible with systems that operate under hard compute budgets:

- WASM sandboxes impose execution fuel limits
- Edge devices (Cognitum Seed, Pi Zero 2W) have strict power envelopes
- ruFlo workflows have per-query deadlines
- MCP tool calls have caller-specified time limits

The only current mitigation is to reduce `ef` globally, which requires offline calibration per workload and gives no per-query control.

Anytime algorithms — those that return the best available answer at any interrupt point — are the correct primitive for budget-constrained retrieval. The result heap in beam search already satisfies the anytime property; the missing piece is a **pluggable stopping policy** that can enforce a per-query compute budget.

---

## Decision

Add `crates/ruvector-anytime-ann` as a self-contained crate implementing three stopping strategies on a flat navigable small-world graph:

1. **FixedEfSearch**: Standard HNSW beam search (baseline).
2. **BudgetedEvalsSearch**: Hard cap on total distance evaluations.
3. **EarlyConvergenceSearch**: Stop when the kth result has not improved for P consecutive expansions.

The stopping logic is encapsulated in a private `StopPolicy` trait inside the beam-search kernel, keeping the public `Searcher` trait simple and unchanged.

### Core API

```rust
pub trait Searcher {
    fn search(
        &self,
        graph: &FlatGraph,
        query: &[f32],
        k: usize,
        ef: usize,
        entry_id: usize,
    ) -> SearchResult;
}

pub struct BudgetedEvalsSearch { pub max_evals: usize }
pub struct EarlyConvergenceSearch { pub patience: usize, pub min_improvement: f32 }
```

The `SearchResult` includes `evaluations: usize`, allowing callers to observe the actual compute cost per query.

---

## Consequences

### Positive

- **1.91× throughput** at budget=65 vs FixedEf on the benchmark dataset.
- **2.52× lower p95 latency** — the most important metric for real-time systems.
- **Zero external dependencies** — compiles to WASM without modification.
- **Trait-based** — new stopping policies (e.g., EnergyBudgetStop, TimeBoundedStop) can be added without touching the kernel.
- Complements ADR-264 (coherence-hnsw): coherence gates WHAT to expand; budget gates WHEN to stop.

### Negative / Risks

- BudgetedEvals at budget=65 gives recall 0.404 vs 0.683 for FixedEf — a 41% recall reduction. Callers must calibrate the budget.
- EarlyConvergenceSearch shows minimal savings on well-clustered data (135 vs 137 evals). More differentiation on harder datasets.
- The flat graph (brute-force k-NN build) is O(N²×D) — suitable for PoC but not production scale.

---

## Alternatives Considered

### A: Reduce global ef

**Rejected**: This is a global parameter that applies to all queries. Anytime search allows per-query budget control, which is strictly more flexible.

### B: Time-bounded search (check Instant::elapsed every N expansions)

**Considered**: Wall-clock time budgets are natural for real-time systems but depend on hardware and system load — not reproducible across platforms. BudgetedEvalsSearch is hardware-independent and maps cleanly to WASM fuel.

### C: Adaptive ef based on query difficulty (distance to first neighbor)

**Deferred**: This requires estimating difficulty before the search, adding latency. It is complementary to this work and could be combined: use difficulty to set `max_evals` dynamically.

### D: Parallel candidate expansion

**Deferred**: Multi-threaded beam search would reduce wall-clock latency but complicates the stopping semantics (evaluations counter is not thread-safe without synchronization). Post-production enhancement.

---

## Implementation Plan

### Phase 1 (Implemented — this PR)

- `crates/ruvector-anytime-ann` as standalone crate
- Three `Searcher` implementations with `StopPolicy` trait
- Zero external dependencies
- 5 unit tests (all passing)
- Benchmark binary with numeric acceptance checks (all passing)

### Phase 2 (Near-term production hardening)

- Integration with `ruvector-core` HNSW (multi-layer, not flat graph)
- SIMD L2 evaluation via `ruvector-math`
- Random or centroid-nearest entry point selection
- `max_evals` as a parameter in `ruvector-server` search API

### Phase 3 (Future research)

- Learned stopping policy: small RL model trained on query distribution
- ruFlo integration: observe `evaluations` and tune `max_evals` automatically
- Energy-proportional budget: express budget in joules for edge deployment

---

## Benchmark Evidence

All numbers from `cargo run --release --manifest-path crates/ruvector-anytime-ann/Cargo.toml --bin benchmark` on Linux x86_64:

| Variant | Recall@10 | Mean(μs) | p95(μs) | QPS | AvgEvals |
|---|---|---|---|---|---|
| FixedEf (ef=60) | 0.683 | 42.7 | 68.6 | 23,429 | 137 |
| BudgetedEvals (budget=65) | 0.404 | 22.3 | 27.2 | 44,800 | 77 |
| EarlyConvergence (patience=3) | 0.680 | 38.9 | 61.3 | 25,707 | 135 |

Dataset: 3000 × 128 dims, 200 queries, k=10.

---

## Failure Modes

| Mode | Detection | Response |
|---|---|---|
| Budget too low | Recall below target | Profile FixedEf AvgEvals, set budget to 50–70% |
| EarlyConvergence never triggers | AvgEvals ≈ FixedEf | Reduce patience or increase min_improvement |
| Budget overshoot | evaluations > max_evals | Expected: last expansion may add up to M+LJ evals |
| Recall collapses at low budget | Recall < acceptable_min | Budget must allow reaching the nearest cluster |

---

## Security Considerations

- BudgetedEvals does not increase attack surface vs FixedEf (fewer computations, not more).
- Anytime search is compatible with proof-gated writes (ADR-227): budget check occurs before neighbor expansion.
- An adversary who can craft queries to maximally exhaust the budget (e.g., queries far from entry point) would also exhaust FixedEf. BudgetedEvals actually limits the damage of such attacks.

---

## Migration Path

Existing callers of HNSW search are unaffected: they use `FixedEfSearch` which replicates standard behavior. `BudgetedEvalsSearch` and `EarlyConvergenceSearch` are opt-in alternatives.

Feature flag recommended: `#[cfg(feature = "anytime-search")]` to keep the three backends behind an opt-in feature when integrated into `ruvector-core`.

---

## Open Questions

1. **Optimal budget formula**: Can we express `max_evals` as a function of graph properties (N, M, cluster count) to auto-calibrate?
2. **EarlyConvergence on hard datasets**: What dataset characteristics make patience=3 differentiate from FixedEf?
3. **Composition with coherence gate**: Does combining ADR-264 + ADR-272 multiply the savings, or are they substitutes?
4. **WASM fuel mapping**: How does `max_evals` map to Wasmtime/WASI fuel units?
