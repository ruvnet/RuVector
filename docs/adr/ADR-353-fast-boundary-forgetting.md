# ADR-353: Fast Structural-Boundary Proxies for Agent-Memory Compaction (Rejected at Production Scale)

## Status

**Rejected** for production use at the corpus sizes tested (500–2,000
memories). Implemented as new, opt-in, non-default code
(`ruvector-agent-memory::fast_boundary`, `fast-boundary-forget` feature,
no new Cargo dependencies) and retained in-tree as a documented, tested,
honestly-negative research result and as the foundation for the concrete
follow-up named in [Rejection Criteria](#rejection-criteria) /
[Open Questions](#open-questions). One effectiveness sub-result
(candidate B's bridge-memory survival gap) is independently strong and is
expected to be revisited once the performance blocker is addressed.

## Context

ADR-345 (`docs/adr/ADR-345-mincut-gated-forgetting.md`) showed that an
exact minimum-cut structural signal, layered on `CoherencePolicy`'s scalar
importance score, protects "bridge" memories a scalar-only policy would
evict — but rejected it for production because
`ruvector_mincut::RuVectorGraphAnalyzer::partition()` was measured at
1,800–2,700x the scalar baseline's latency even at an 84-memory corpus,
scaling to multi-second calls by n=400. ADR-346 fixed a separate
correctness bug in that same code path (non-deterministic, sometimes-empty
partitions) but explicitly left the latency problem untouched as
out-of-scope.

This ADR covers the direct follow-up: replacing the exact min-cut call
with two classical, dependency-free local-structure heuristics — local
clustering coefficient and label-propagation community boundary — and
testing, with a benchmark whose acceptance thresholds were fixed before it
ran, whether either is fast enough and faithful enough to ship.

## Hypothesis

```text
Given a synthetic agent-memory corpus (deterministic seeded generation: N
topic clusters + bridge memories interpolated between randomly paired
clusters, 32-dim, a hot-cluster access simulation, and a k-NN (k=8, cosine
>= 0.05) similarity graph over it),

when two O(V*k) / O(rounds*V*k) local-structure boundary heuristics --
local clustering coefficient (candidate A) and label-propagation community
boundary (candidate B) -- are used in place of ruvector-mincut's exact
RuVectorGraphAnalyzer::partition() to gate agent-memory compaction,

then: (1) at n=84 (10 seeds), each candidate's boundary set has >= 60%
average Jaccard overlap with the exact-mincut boundary set; (2) at n=500
and n=2000, each candidate's compaction wall-clock is <= 3x the
scalar-only baseline's; (3) at n=500 and n=2000, each candidate retains
a bridge-memory survival rate >= 15pp higher than baseline while
Recall@10 stays within 2pp of baseline;

subject to: (a) deterministic seeded generation throughout, (b) the
existing eviction-witness chain achieves 100% tamper detection across
>= 20 trials wrapping either candidate, (c) no change to ruvector-mincut
or MincutGatedForgetting's public API.
```

Full methodology, root-cause diagnostics, and raw benchmark output are in
`docs/research/nightly/2026_10_15_fast-boundary-forgetting/README.md`.

## Decision

1. Add `crates/ruvector-agent-memory/src/fast_boundary.rs`:
   `BoundaryHeuristic::{ClusteringCoefficient, LabelPropagation}`,
   `ForgetMode::{Soft, Hard}` (a local copy of `graph_forget::ForgetMode`'s
   semantics, not shared, so this module stays dependency-free), and
   `FastBoundaryForgetting: CompactionPolicy`. Gated behind a new
   `fast-boundary-forget` Cargo feature with **zero new dependencies**.
2. Add a public `MincutGatedForgetting::boundary_set()` accessor to
   `graph_forget.rs` — pure delegation to the existing private
   `boundary_indices()`, needed so the new fidelity benchmark can read the
   exact boundary without duplicating k-NN build logic. No behavior
   change to `graph_forget`.
3. Add `examples/boundary_fidelity_and_scale_bench.rs`: the pre-registered
   fidelity + scale + witness benchmark.
4. **Do not promote either candidate to a recommended default.** The
   benchmark's own acceptance test returned REJECT: both candidates are
   73–311x slower than baseline at n=500/2000 (threshold: <=3x), and
   neither clears the 60% fidelity-vs-exact-mincut bar at n=84 (candidate
   A: 0.0%; candidate B: 18.0%).

## Evidence

(Full raw output in the nightly README's Benchmark section; summarized
here.)

| Metric | Threshold | Candidate A | Candidate B |
|---|---|---|---|
| Fidelity vs. exact mincut (n=84) | >= 60% | 0.0% FAIL | 18.0% FAIL |
| Fidelity vs. ground-truth bridges (n=84, diagnostic) | — | 0.0% | 39.8% (exact mincut itself: 13.7%) |
| Bridge-survival gap, n=500 | >= +15pp | +2.0pp FAIL | +38.0pp PASS |
| Bridge-survival gap, n=2000 | >= +15pp | −1.0pp FAIL | +35.0pp PASS |
| Recall@10 delta, both scales | <= 2pp | 0.00pp PASS | 0.00pp PASS |
| Slowdown vs. baseline, n=500 | <= 3x | 84.7x FAIL | 86.8x FAIL |
| Slowdown vs. baseline, n=2000 | <= 3x | 232.5x FAIL | 241.3x FAIL |
| Witness tamper detection (n=500, 20 trials) | 100% | 100% PASS | (shared mechanism) |

Root-cause diagnostics (measured after the pre-registered test, not used
to change any threshold):

- A direct timing breakdown shows 93–99% of each candidate's wall-clock
  at n=500/n=2000 is spent in the O(V²) brute-force cosine-similarity
  k-NN graph build, not in the heuristic computed on top of it.
- A fresh re-run of the unmodified `mincut_scaling_probe.rs` (ADR-345's
  own reproduction script) found `RuVectorGraphAnalyzer::partition()`
  taking **95.5 seconds at n=19** — the smallest graph in the sweep,
  slower than n=50 (110ms), confirming latency is non-monotonic in vertex
  count on this code path, independent of this ADR's own candidates.
- The exact-mincut boundary set itself overlaps the benchmark generator's
  12 ground-truth bridge memories by only 13.7% on average (boundary size
  averaging 5.3 vertices) — global min-cut reports one 2-way split, which
  is not the same concept as "all bridge memories" once a graph has more
  than one bridge, unlike the single-bridge topology ADR-345/346's unit
  tests use.

## Consequences

- **Positive**: candidate B (label propagation) is a strong, cheap,
  dependency-free *effectiveness* signal (35–38pp bridge-survival gap,
  nearly 3x closer to ground truth than exact min-cut) — decoupled here
  from the mechanism (exact-mincut fidelity) it was originally meant to
  approximate, so it can be re-evaluated purely on its own merits once its
  k-NN construction cost is fixed.
- **Positive**: `fast_boundary.rs`'s adjacency build is deterministic by
  construction (sorted `Vec`, never hash-map iteration order) — the same
  property ADR-346 had to retrofit onto `ruvector-mincut`, achieved here
  without that retrofit.
- **Negative**: the stated goal — a production-viable fast alternative to
  exact min-cut for this signal — is not achieved. The O(V²) k-NN
  construction this run's heuristics (and the pre-existing `graph_forget`
  module) build from scratch each call is the real, still-unaddressed
  bottleneck.
- **Negative**: the fidelity metric's reference (exact min-cut) is now
  independently shown to be a biased ground truth on multi-bridge graphs,
  complicating any future "candidate vs. exact mincut" comparison on
  realistic topologies without also checking against planted ground truth
  directly.
- **Neutral**: no public API of `ruvector-mincut`, `graph_forget`, or
  `CompactionPolicy` changed. This ADR's code is purely additive.

## Alternatives

- **Approximate k-NN via `ruvector-core`'s HNSW index** instead of
  brute-force O(V²). Not implemented this run (a real architectural
  change, not a parameter variant); named as the concrete next experiment
  (see Open Questions).
- **Cache/reuse the k-NN graph across compaction calls.** Would address
  the bottleneck for workloads where compaction runs more often than the
  underlying memory set changes; out of scope for a one-night bounded
  experiment, flagged as a plausible alternative fix.
- **A third heuristic** (e.g. approximate betweenness centrality). Deferred
  to keep this run at the nightly protocol's bounded 2-candidate design.
- **Hybrid exact-occasionally / fast-between.** Considered and set aside:
  this run's own finding that exact min-cut is a biased reference on
  multi-bridge graphs undercuts the premise that occasional exact runs
  would be the right correction signal.

## Implementation Plan

Already implemented in this PR:

1. `crates/ruvector-agent-memory/src/fast_boundary.rs` — new module (~400
   lines), 7 unit tests.
2. `crates/ruvector-agent-memory/src/graph_forget.rs` — additive
   `boundary_set()` public accessor.
3. `crates/ruvector-agent-memory/src/lib.rs` — module registration +
   feature-gated re-exports.
4. `crates/ruvector-agent-memory/Cargo.toml` — new `fast-boundary-forget`
   feature (no deps) and new example registration.
5. `crates/ruvector-agent-memory/examples/boundary_fidelity_and_scale_bench.rs`
   — the pre-registered benchmark.

Not implemented (see Open Questions / Next Research in the nightly
README): HNSW-backed k-NN substitution, a parameter sweep for candidate
A's thresholds, and the crate-wide `wasm32-unknown-unknown` build fix.

## API Shape

```rust
// crates/ruvector-agent-memory/src/fast_boundary.rs
pub enum BoundaryHeuristic { ClusteringCoefficient, LabelPropagation }
pub enum ForgetMode { Soft, Hard }

pub struct FastBoundaryForgetting {
    pub weights: CoherenceWeights,
    pub mode: ForgetMode,
    pub heuristic: BoundaryHeuristic,
    pub k_neighbors: usize,
    pub min_similarity: f32,
    pub structural_bonus: f32,
    pub protect_fraction: f32,
    pub clustering_threshold: f32,
    pub min_degree: usize,
    pub lp_rounds: usize,
}

impl FastBoundaryForgetting {
    pub fn clustering_soft(weights: CoherenceWeights, structural_bonus: f32) -> Self;
    pub fn clustering_hard(weights: CoherenceWeights, protect_fraction: f32) -> Self;
    pub fn label_propagation_soft(weights: CoherenceWeights, structural_bonus: f32) -> Self;
    pub fn label_propagation_hard(weights: CoherenceWeights, protect_fraction: f32) -> Self;
    pub fn boundary_set(&self, entries: &[MemoryEntry]) -> HashSet<usize>;
}

impl CompactionPolicy for FastBoundaryForgetting { /* ... */ }

// crates/ruvector-agent-memory/src/graph_forget.rs (additive)
impl MincutGatedForgetting {
    pub fn boundary_set(&self, entries: &[MemoryEntry]) -> HashSet<usize>;
}
```

No existing signature changed.

## Feature Flags

`fast-boundary-forget` — new, opt-in, no dependencies. Independent of
`mincut-forget` (only needed together for the fidelity-comparison
benchmark, not for using `FastBoundaryForgetting` on its own).

## Benchmark Evidence

See [Evidence](#evidence) above and
`docs/research/nightly/2026_10_15_fast-boundary-forgetting/README.md`'s
full raw output.

## Security

No new attack surface: pure deterministic computation over already-
trusted in-process vectors, no new I/O, no new serialization, no new
dependency. The reused eviction-witness mechanism's tamper-evidence scope
is unchanged (FNV-1a chaining — accidental corruption and naive edits, not
a log-writing adversary; see ADR-347 for the Ed25519-signed mitigation,
unaffected by this change). See the nightly README's Security section for
full detail.

## Governance

Standard PR review. No schema, wire-format, or persisted-data changes. No
change to any proof-gate or ledger admission path.

## Failure Modes

- Both candidates' compaction wall-clock scales with the square of corpus
  size (O(V²) k-NN build), so the slowdown-vs-baseline ratio will worsen,
  not improve, at corpus sizes beyond the 2,000 tested here.
- Candidate A (clustering coefficient) shows no measurable effectiveness
  advantage over the scalar baseline at either realistic scale tested
  (+2.0pp at n=500, −1.0pp at n=2000 — within noise of zero), in addition
  to failing fidelity; it is the weaker of the two candidates on every
  axis measured.
- The fidelity benchmark's reference signal (exact min-cut) is shown to
  be a questionable ground truth on the multi-bridge topology used for
  the realistic-scale corpora — see Evidence — so a future candidate that
  *did* clear the 60% exact-mincut-fidelity bar would not, by that fact
  alone, be confirmed as "finding the right memories," only as "agreeing
  with a particular global 2-way cut."

## Migration

None. No existing code path's default behavior changed; `fast_boundary`
is new, additive, feature-gated, and not wired into any default
compaction pipeline.

## Rollback

Delete `src/fast_boundary.rs`, its `lib.rs` registration, the
`fast-boundary-forget` feature, the new example, and the additive
`boundary_set()` accessor on `MincutGatedForgetting` (or simply leave the
accessor — it is inert without a caller). No other code depends on any of
this ADR's additions.

## Rejection Criteria

This decision should be revisited (i.e., re-run this exact benchmark) if:

- `ruvector-core`'s HNSW index is wired in to replace the O(V²)
  brute-force k-NN build in `fast_boundary.rs` (Next Research item 1 in
  the nightly README) — re-running the unmodified Phase 2 benchmark after
  that single substitution is the direct falsification test for whether
  the performance threshold can now be met.
- A parameter sweep shows candidate A's near-zero fidelity/effectiveness
  was a threshold-tuning artifact rather than a structural property of
  the heuristic.
- A fidelity benchmark run against ground-truth bridge labels directly
  (rather than against exact min-cut) is implemented, given this run's
  own finding that the two references disagree substantially.

## Open Questions

1. Does label propagation's advantage over clustering coefficient
   generalize beyond this specific synthetic generator, or is it an
   artifact of how "bridge" vectors are constructed (interpolated
   midpoints, which may systematically produce multi-community-boundary
   structure that label propagation is well-suited to detect by
   construction)? Unresolved — flagged as next research.
2. Is `ruvector-agent-memory`'s crate-wide `wasm32-unknown-unknown` build
   failure (pre-existing, caused by an always-on `rand`/`getrandom`
   dependency unrelated to this ADR) worth fixing on its own, given that
   `fast_boundary.rs` would otherwise add no further obstacle to a wasm32
   build? Unresolved — scoped as a separate, security-adjacent change
   (touches key-generation code in `observation`/`witness_signing`).
3. Should `graph_forget.rs`'s own k-NN build be unified with
   `fast_boundary.rs`'s (both currently duplicate the brute-force
   all-pairs cosine-similarity logic independently)? Not attempted this
   run to keep the two modules's dependency footprints (one with
   `ruvector-mincut`, one without) cleanly separable; worth revisiting
   once/if an HNSW-backed version replaces both.
