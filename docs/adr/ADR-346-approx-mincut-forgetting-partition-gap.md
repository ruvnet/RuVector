# ADR-346: Approximate-Mincut-Gated Forgetting — Speed Fix Confirmed, `ApproxMinCut` Partition Bug Found

## Status

Rejected (for production use as the bridge-protection mechanism). Experimental
crate addition (`ruvector-agent-memory::graph_forget_approx`, feature-gated
behind the existing `mincut-forget` flag, off by default) retained as evidence
and reference implementation, not promoted. A concrete correctness gap in
`ruvector-mincut::ApproxMinCut::compute_partition` is filed as an upstream
finding (see "Open Questions").

## Context

ADR-345 (nightly 2026-09-05) added `MincutGatedForgetting`, layering a
`ruvector_mincut::RuVectorGraphAnalyzer`-derived structural boundary signal
onto `ruvector-agent-memory`'s scalar `CoherencePolicy`, to protect "bridge"
memories from eviction. It was rejected on two independent axes: the
boundary signal showed a measured +0.0pp bridge-survival benefit at the only
corpus size the engine could afford (84 memories), and
`RuVectorGraphAnalyzer::partition()` itself measured 1,800-2,700x slower than
the scalar baseline — 76ms-11.4s per call for graphs of 50-400 vertices. That
ADR's "Alternatives Considered" flagged `ruvector_mincut::DynamicMinCut`,
used directly instead of through `RuVectorGraphAnalyzer`, as the primary
follow-up direction, on the theory that the wrapper, not the underlying
algorithm, might be the bottleneck.

Reading `ruvector-mincut`'s source for that follow-up surfaced a second,
already-implemented entry point neither ADR-345 nor its benchmark tried:
`ruvector_mincut::ApproxMinCut` (`crates/ruvector-mincut/src/algorithm/
approximate.rs`), documented as a "(1+ε)-approximate min-cut for all cut
sizes" implementation citing SODA 2025 (arXiv:2412.15069), with claimed
preprocessing complexity `O(m log^2 n / epsilon^2)` — a plausible fix for the
measured latency bottleneck. This ADR tests that entry point directly against
the same corpus, dataset, and acceptance bar as ADR-345, changing only the
mincut engine.

Reading `ApproxMinCut`'s implementation before running anything also
surfaced a specific, falsifiable code-level concern: `compute_partition`
ignores the cut value it was passed and instead performs a plain BFS from an
arbitrary start vertex, stopping once exactly half the graph's *total*
vertex count has been visited — with no reference to where the graph's
actual weak edges are. This ADR's hypothesis was fixed before any benchmark
ran, explicitly predicting that this would produce a speed win without a
correctness win, and the experiment then measured whether that specific
prediction was true.

## Hypothesis

```text
Given the same synthetic corpus, k-NN graph construction, and acceptance bar
as ADR-345 (72 core + 12 bridge = 84 memories, 32-dim, k=5, cosine >= 0.05,
compacted to 42 = 50%),

when boundary detection uses ApproxMinCut (candidate B: ApproxMincutForgetting
-Soft/Hard) instead of RuVectorGraphAnalyzer (candidate A: MincutGatedForgetting
-Soft/Hard, ADR-345's rejected candidate, kept only as the within-run speed
reference) versus plain CoherencePolicy (baseline),

then candidate B should be at least 10x faster per compaction call than
candidate A (operationalizing "fix the ADR-345 bottleneck"),

subject to: candidate B's bridge-survival gap over baseline must still be at
least 15 percentage points (ADR-345's bar) for ACCEPT.

Pre-registered expectation (from reading ApproxMinCut::compute_partition
before running anything): the speed criterion will likely pass and the
survival criterion will likely fail, because the exposed partition is
structurally disconnected from the cut value.
```

Full methodology, raw output, and the two supporting probes
(`examples/approx_mincut_partition_probe.rs`,
`examples/approx_mincut_forgetting_bench.rs`) are in
`docs/research/nightly/2026-09-19-approx-mincut-forgetting/README.md`.

## Decision

1. Add `ruvector-agent-memory::graph_forget_approx::ApproxMincutForgetting`
   (`ApproxForgetMode::Soft` / `Hard`), a `CompactionPolicy` mirroring
   ADR-345's `MincutGatedForgetting` exactly except for the mincut engine
   (`ruvector_mincut::ApproxMinCut` in place of `RuVectorGraphAnalyzer`), so
   the engine is the only isolated variable between the two experiments.
   Reuses the existing `mincut-forget` feature flag — no new Cargo feature.
2. Add `examples/approx_mincut_partition_probe.rs`: a minimal, dependency-free
   demonstration that `ApproxMinCut::compute_partition`'s reported partition
   matches its reported cut value on a *balanced* two-triangle graph but not
   on an *unbalanced* 9-vertex-clique + bridge + 3-vertex-clique graph, where
   a coincidental match is numerically impossible.
3. Add `examples/approx_mincut_forgetting_bench.rs`, extending ADR-345's
   benchmark with the two new policies on the identical seeded dataset.
4. **Do not promote `ApproxMincutForgetting`.** It is measurably faster than
   ADR-345's rejected candidate but does not clear the bridge-survival bar
   either — confirming the pre-registered prediction rather than a new,
   independent failure.
5. Keep the module in-tree, behind the existing feature flag, as working
   evidence that the latency half of ADR-345's rejection is fixable in
   isolation, and that a genuinely correct boundary-detection replacement for
   `RuVectorGraphAnalyzer::partition()` is still an open problem.

## Evidence

`ApproxMinCut`'s internal `HashSet<VertexId>` uses Rust's default randomized
hasher, and its `compute_partition`'s BFS start vertex is
`self.vertices.iter().next()` — so its output varies call to call against
the identical input graph (see "A second finding: non-determinism" below).
Reported figures are therefore the mean of 10 in-process trials (identical
reseeded dataset each trial; only `ApproxMinCut`'s internal hasher state
differs), not a single run:

| Gate | Threshold | Measured (Soft, mean of 10) | Measured (Hard, mean of 10) | Result |
|---|---|---|---|---|
| Speedup vs. exact (`RuVectorGraphAnalyzer`) | >= 10x | ~23x | ~23x | PASS / PASS |
| Bridge-survival gap vs. baseline | >= 15pp | -27.5pp (range: -58.4pp to -8.4pp) | -0.0pp (stable every trial) | FAIL / FAIL |
| Recall@10 delta vs. baseline | <= 2pp | ~2.3-2.8pp (borderline, varies by run) | 0.00pp | FAIL / PASS |

Absolute numbers (release build, seed=341, identical dataset to ADR-345's
benchmark; `cargo run --release -p ruvector-agent-memory --example
approx_mincut_forgetting_bench --features mincut-forget`; two independent
process runs, each averaging 10 in-process trials for the approximate rows):

| Policy | Bridge Surv. (mean, min-max) | Recall@10 | Compaction |
|---|---|---|---|
| `CoherenceWeighted` (baseline) | 66.7% | 100.0% | ~55-70 us |
| `MincutGatedForgetting-Soft` (ADR-345, exact) | 66.7% | 100.0% | ~105 ms |
| `MincutGatedForgetting-Hard` (ADR-345, exact) | 66.7% | 100.0% | ~105 ms |
| `ApproxMincutForgetting-Soft` (this ADR) | 39.2% (8.3%-58.3%) | 97.2-97.7% | ~4.4-4.7 ms |
| `ApproxMincutForgetting-Hard` (this ADR) | 66.7% (66.7%-66.7%, every trial) | 100.0% | ~4.5-4.6 ms |

Root causes (see the research doc's "Failure modes" for full detail and the
raw partition-probe output):

- **The speed fix is real.** `ApproxMinCut` is ~22-24x faster than
  `RuVectorGraphAnalyzer::partition()` on this exact corpus (release build,
  same machine, consistent across repeated runs). It does not clear
  ADR-345's original "under 100x baseline" bar either in absolute terms
  (4.4-4.7ms is still on the order of 65-85x the ~55-70us scalar baseline)
  but it is a genuine, large, reproducible improvement over the previously
  measured 1,800-2,700x.
- **The correctness gap is also real, and root-caused.**
  `ApproxMinCut::compute_partition` (`crates/ruvector-mincut/src/algorithm/
  approximate.rs`) does not derive its returned partition from the min-cut it
  computes: it BFS-walks from an arbitrary start vertex and stops once
  exactly half of the graph's *total* vertex count has been visited.
  `examples/approx_mincut_partition_probe.rs` confirms this directly and
  cheaply: on a balanced 3-vs-3 two-triangle graph the BFS-half-split
  coincidentally reproduces the true cut (matches YES); on an unbalanced
  9-vs-3 clique-plus-bridge graph, where "half of 12" (6) matches neither
  true side, it does not (matches NO), while the reported cut *value* (1.0)
  is correct in both cases.
- **Soft mode is actively harmful in every single trial observed, not
  merely on average.** Because the additive bonus in `ApproxForgetMode::Soft`
  is applied to whatever the broken partition happens to flag as "boundary,"
  it can *outrank* memories that would otherwise have survived on scalar
  merit alone. Across 20 in-process trials total (two independent process
  runs of 10 each), the observed bridge-survival rate ranged from 8.3% to
  58.3% and never once reached the 66.7% do-nothing baseline.
  `ApproxForgetMode::Hard`'s budget-reservation design is more conservative
  (it can only add protected survivors up to a small reserved fraction, not
  reorder the rest of the ranking) and landed at *exactly* baseline parity
  (66.7%) in all 20 trials — a striking stability given the underlying
  boundary set is not, itself, stable; the most likely explanation (not
  independently confirmed by additional instrumentation in this pass) is
  that on this dataset the boundary-ranked vertices `Hard` mode reserves
  budget for already rank highly enough under plain scalar scoring that the
  reservation is a no-op regardless of which specific vertices get flagged.

### A second finding: non-determinism

Separately from the partition-correctness bug, developing this experiment's
own unit test surfaced a second, independent problem: `ApproxMinCut`'s
`vertices: HashSet<VertexId>` field uses Rust's default (randomized) hasher,
and `compute_partition`'s BFS start vertex is `self.vertices.iter().next()`.
Because that hasher's keys are randomized per process (and vary per
`HashSet` instantiation within a process too), which vertex the BFS starts
from — and therefore which vertices end up flagged as "boundary" — varies
from call to call against the byte-identical input graph this benchmark
rebuilds every time. Confirmed two ways: (a) a unit test asserting a fixed
bridge-survival outcome on ADR-345's own 19-vertex test graph passed in
isolation but failed in 2 of 6 separate `cargo test` process invocations
during development (since corrected to assert only structural invariants,
not a specific hash-dependent outcome — see `graph_forget_approx.rs`'s
`soft_mode_runs_without_panicking_on_bridge_dataset`); (b) the corpus-level
benchmark's `ApproxMincutForgetting-Soft` bridge-survival number varied
across 8 separate manual process invocations from 8.3% to 75.0% before this
ADR's benchmark was changed to average `N_APPROX_TRIALS = 10` in-process
runs per policy. This is conceptually the same *class* of problem ADR-345
found in `RuVectorGraphAnalyzer` (a mincut-adjacent API whose output is not
reproducible against identical input) but with a different, now precisely
identified, mechanism — and it affects the "fast" engine this ADR tested,
not only the "slow" one ADR-345 rejected.

## Consequences

- `ruvector-agent-memory` gains a second working, tested (if unpromoted)
  mincut integration, isolating "is the engine the bottleneck" (yes, and
  fixable — confirmed) from "is the signal correct" (no — a different,
  independent problem, also confirmed) for future readers who might
  otherwise conflate the two.
- A specific, cheaply-reproducible bug report against
  `ruvector-mincut::ApproxMinCut::compute_partition` now exists with an
  executable minimal repro (`approx_mincut_partition_probe.rs`), rather than
  a general "it seemed unreliable" impression.
- No existing behavior changes: `ApproxMincutForgetting` is opt-in behind the
  existing `mincut-forget` feature flag, exactly like ADR-345's
  `MincutGatedForgetting`; neither is on by default or referenced by any
  other crate.
- The `ApproxMinCut` docstring's complexity and algorithm claims (real
  effective-resistance computation, genuine spectral sparsification) do not
  match its current implementation for graphs in this corpus's size range:
  `compute_min_cut_via_sparsifier`'s sparsifier target size
  (`n * ln(n) / epsilon^2`) is clamped to `min(target, edge_count)`, and for
  n<=~400 with the default epsilon=0.1 that clamp is almost always the
  binding constraint, so the "sparsifier" typically retains close to 100% of
  edges. The measured 22x speedup is real but is best attributed to
  `ApproxMinCut`'s simpler, more direct Stoer-Wagner call path relative to
  `RuVectorGraphAnalyzer`'s witness-tree machinery, not to genuine
  sparsification at this scale — see the research doc for the reasoning and
  its own explicit caveat that this specific claim was not independently
  profiled down to the instruction level.

## Alternatives Considered

- **Fix `ApproxMinCut::compute_partition` upstream in this same change.**
  Rejected for this ADR's scope: the nightly process evaluates
  `ruvector-mincut` as a fixed dependency and reports what it measures, not
  what a hypothetical patched version might do. A real fix belongs to
  `ruvector-mincut`'s own maintainers or a future nightly explicitly scoped
  to it, with its own hypothesis and acceptance bar (see "Open Questions").
- **Implement boundary detection from scratch instead of reusing
  `ruvector-mincut`.** Rejected: contradicts the nightly process's own
  guidance to reuse existing ecosystem capabilities, and this ADR's value is
  precisely in characterizing what the existing capability does and does not
  provide.
- **Report only the speed number and call it an ACCEPT.** Rejected outright:
  ADR-345's own acceptance bar requires the survival gate, and reporting a
  favorable metric while omitting an unfavorable one on the same experiment
  is exactly the reward-hacking failure mode the nightly process exists to
  prevent.

## Implementation Plan

Already implemented in this PR:

- `crates/ruvector-agent-memory/src/graph_forget_approx.rs` (feature-gated
  behind the existing `mincut-forget` flag)
- `crates/ruvector-agent-memory/examples/approx_mincut_forgetting_bench.rs`
- `crates/ruvector-agent-memory/examples/approx_mincut_partition_probe.rs`
- `crates/ruvector-agent-memory/Cargo.toml`: two new `[[example]]` entries
- `crates/ruvector-agent-memory/src/lib.rs`: `graph_forget_approx` module and
  re-exports, feature-gated identically to `graph_forget`
- Unit tests in `graph_forget_approx.rs` (3)

No further implementation is planned under this ADR; a genuine fix to
`ApproxMinCut::compute_partition` (or a from-scratch fast boundary-detection
replacement) is out of scope — see "Open Questions".

## API Shape

```rust
// Behind `mincut-forget` (same flag as ADR-345's MincutGatedForgetting):
pub enum ApproxForgetMode { Soft, Hard }
pub struct ApproxMincutForgetting {
    pub weights: CoherenceWeights,
    pub mode: ApproxForgetMode,
    pub k_neighbors: usize,
    pub min_similarity: f32,
    pub structural_bonus: f32,
    pub protect_fraction: f32,
    pub epsilon: f64,
}
impl CompactionPolicy for ApproxMincutForgetting { /* .. */ }
```

## Feature Flags

No new feature flag: `ApproxMincutForgetting` reuses ADR-345's
`mincut-forget` flag, since it depends on the same `ruvector-mincut` path
dependency and serves the same opt-in experimental surface.

## Benchmark Evidence

See "Evidence" above and the linked nightly README for full raw output and
methodology, including the balanced/unbalanced partition-probe output.

## Security

No new cryptographic primitive; no interaction with `witnessed_compaction`'s
witness chain beyond what ADR-345 already established (the chain is generic
over any `CompactionPolicy`, so it was not re-tested here — see the research
doc for why re-running that specific check would have been redundant).

## Governance

None beyond ADR-345's existing "no witness, no mutation" invariant, which
this ADR does not touch.

## Failure Modes

See the nightly research doc's "Failure modes" section for the full account:
the Soft-mode active-harm case (never once reached baseline in 20 sampled
trials, mean 39.2% vs. 66.7% baseline) is the most actionable one for anyone
tempted to enable this policy anyway "since it's fast now" — speed alone
does not make `ApproxForgetMode::Soft` safe to use, and its outcome is not
even reproducible run to run (see "A second finding: non-determinism").

## Migration

None: `graph_forget_approx` is new and shares an already-off-by-default
feature flag; no existing caller is affected.

## Rollback

Remove `graph_forget_approx` and its two examples with no impact on any
existing caller — nothing in the crate's default build path, nor
`graph_forget`, depends on it.

## Rejection Criteria

The hypothesis in this ADR is treated as rejected (for production use as a
bridge-protection mechanism) because:

1. The primary comparison (bridge-survival gap >= 15pp) measured, as a mean
   of 10 in-process trials, -27.5pp (Soft, range -58.4pp to -8.4pp across
   trials) and -0.0pp (Hard, stable in every trial) — the mandatory gate
   ADR-345 also used.
2. The speed gate (>= 10x faster than ADR-345's rejected exact candidate)
   passed (~22-24x across repeated runs) — confirming the half of the pre-registered
   hypothesis that predicted a real fix was possible, and isolating that the
   remaining problem is specifically the partition/boundary extraction, not
   engine latency.

Per the nightly process's own rule, a falsified hypothesis with a
well-characterized root cause (here: two independently confirmed and
partially-orthogonal findings, not a single ambiguous failure) is a
successful nightly outcome.

## Open Questions

1. Is `ApproxMinCut::compute_partition`'s BFS-half-split a known placeholder
   in `ruvector-mincut`, or an unintentional gap? A genuine fix (deriving the
   partition from the same Stoer-Wagner computation already producing the
   cut *value*, on the sparsifier or full graph as appropriate) is
   implementable without redesigning the sparsification approach, and would
   be the natural next nightly in this lineage: re-run
   `approx_mincut_forgetting_bench.rs` unchanged against a patched
   `ApproxMinCut` and see whether the survival gate then passes.
2. Does `ApproxMinCut`'s sparsifier ever actually bind (i.e., does
   `target_size < edge_count`) at any corpus size relevant to agent-memory
   compaction, or only well beyond what a per-compaction call could afford
   regardless? Answering this would clarify whether the measured 22x speedup
   should be attributed to sparsification at all, as opposed to a simpler,
   more direct code path than `RuVectorGraphAnalyzer`'s.
3. Does `ruvector_mincut::DynamicMinCut` used directly (ADR-345's original
   "Alternatives Considered" #2, still not implemented by either this ADR or
   ADR-345) avoid both the latency and partition-correctness problems found
   in its two siblings, or does it share the same underlying
   `exact::minimum_cut` computation as `RuVectorGraphAnalyzer` and therefore
   only the latter's latency profile?
