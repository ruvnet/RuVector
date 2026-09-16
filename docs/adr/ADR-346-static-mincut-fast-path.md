# ADR-346: Deterministic Static Min-Cut Fast Path (Stoer-Wagner) for `ruvector-mincut`

## Status

Accepted (the `static_cut` module itself: a correct, deterministic, ~65-8400x
faster fast path for one-shot min-cut queries, promoted to `ruvector-mincut`
as public API). Rejected (the `MincutGatedForgetting`-Static application in
`ruvector-agent-memory`, for production use as designed — same disposition
and for one of the same two reasons as its ADR-345 parent). See
`docs/research/nightly/2026-09-08-static-mincut-forgetting/README.md` for
full methodology and evidence.

## Context

`docs/research/nightly/2026-09-05-mincut-gated-forgetting` (ADR-345)
implemented `MincutGatedForgetting`, an agent-memory compaction policy that
layers a graph-structural "protect the bridge" signal on top of
`ruvector-agent-memory`'s existing scalar `CoherencePolicy`, reusing
`ruvector-mincut`'s `RuVectorGraphAnalyzer::partition()` as-is. That nightly
rejected the candidate on two independent, measured grounds:

1. **Performance.** `partition()` measured 76ms-11.4s per call on graphs of
   50-400 vertices (worse — 66.7s — on one small regular/ring topology), and
   was 1,800-2,700x slower than the scalar baseline even at an 84-memory
   corpus.
2. **Effectiveness at the size performance forced.** At that corpus size,
   the structural bonus made no measurable difference to bridge-memory
   survival versus the plain scalar baseline.

It also documented an unresolved third finding, filed as a follow-up
hardening item rather than worked around: `partition()` is **not
deterministic** across repeated calls on a byte-identical graph (empty
result in ~50% of 30 trials on a fixed 19-vertex fixture), "consistent with
internal tie-breaking that depends on hash-map iteration order rather than
any property of the graph."

This ADR is that follow-up. Root-causing it further:
`RuVectorGraphAnalyzer::partition()` delegates to
`wrapper::MinCutWrapper::query()`, which implements the bounded-range
dynamic instance ladder from arxiv:2512.13105 (up to 100 geometrically-scaled
sub-instances, each replaying the *entire current edge set* the first time
it's touched). That machinery is designed to amortize well across many
incremental edge insert/delete events on one long-lived graph. Every
`RuVectorGraphAnalyzer::from_knn(...)` call site in the workspace (this
crate's own `CommunityDetector`, `GraphPartitioner`, and
`ruvector-agent-memory`'s `graph_forget` module) instead discards the graph
and rebuilds a fresh `RuVectorGraphAnalyzer` on every call — paying the full
multi-instance replay cost with nothing amortized. Separately,
`graph::DynamicGraph` stores edges in `dashmap::DashMap`s, each constructed
with an independently-randomized `RandomState` hash seed
(`DynamicGraph::new()` → `DashMap::new()`); `graph.edges()`'s iteration order
therefore differs across structurally-identical `DynamicGraph` instances,
and `MinCutWrapper` iterates `graph.edges()` directly with no intervening
sort when instantiating each range instance — sufficient to explain the
observed non-determinism without any intentional randomized algorithm
anywhere in the call chain.

## Hypothesis

```text
Given the same 84-entry synthetic bridge-memory corpus, k-NN graph
construction, and acceptance thresholds as the 2026-09-05 nightly (ADR-345),

when a new deterministic static global-min-cut fast path (Stoer-Wagner,
O(V^3), `ruvector_mincut::static_cut`) is added to ruvector-mincut and wired
into MincutGatedForgetting as a second engine (`MincutEngine::Static`) in
place of the dynamic bounded-instance engine,

then (a) per-compaction wall-clock drops by at least 50x relative to the
rejected Dynamic engine's measured slowdown, (b) repeated `partition_static()`
calls on independently-rebuilt, structurally-identical graphs return
byte-identical partitions across >= 25 trials (0% divergence, vs the
previously measured ~50% empty-result/non-determinism rate), and (c)
bridge-memory survival rate improves at least 15 percentage points over the
scalar-only CoherencePolicy baseline using a single deterministic call
(mincut_trials = 1, no retry union needed),

subject to: Recall@10 staying within 2 percentage points of baseline, and a
pre-registered absolute speed bar of <= 10x slowdown vs. the scalar baseline
(tighter than ADR-345's 100x "background job" bar — this nightly's premise
is that the static engine is fast enough to be a *foreground* path, not just
faster than before).
```

## Decision

Add `ruvector_mincut::static_cut::stoer_wagner_min_cut` (classical
Stoer-Wagner global min-cut, O(V^3), operating on a dense weight matrix built
once from `graph.edges()` sorted by canonical endpoints and edge id — so its
result depends only on graph structure, never on any hash map's iteration
order) as a new, permanently public module. Expose it on
`RuVectorGraphAnalyzer` as `partition_static()` / `min_cut_static()`,
alongside (not replacing) the existing dynamic `partition()` / `min_cut()`.
Add `MincutEngine::{Dynamic, Static}` to `ruvector-agent-memory`'s
`MincutGatedForgetting`, with new `soft_static()` / `hard_static()`
constructors that fix `mincut_trials = 1` (the static engine needs no
retries). The original `soft()` / `hard()` constructors are unchanged
(`MincutEngine::Dynamic`), preserving ADR-345's original result exactly
reproducible.

**Promote the `static_cut` module itself** (correct against Stoer-Wagner's
known correctness properties, tested for determinism, and measured 65x-8400x
faster than the dynamic engine depending on topology) as new
`ruvector-mincut` public API — any call site currently paying the dynamic
engine's replay cost for a one-shot query can adopt it directly.

**Do not promote** `MincutGatedForgetting`'s `-Static` application to
production-default status. The re-run benchmark (see the nightly README)
shows:

- Determinism and hard-constraint speedup are confirmed and large: 65-74x
  faster than Dynamic on the realistic 84-memory bridge corpus, up to 8,421x
  faster on ADR-345's original determinism-probe fixture.
- The **absolute** speed bar this ADR pre-registered (<=10x vs. the scalar
  baseline) is still missed: measured 43.8x (Soft-Static) / 46.9x
  (Hard-Static) slowdown. O(V^3) Stoer-Wagner, even fully deterministic and
  dramatically faster than the alternative, is still asymptotically far more
  expensive than an O(n log n) scalar sort at this corpus's k-NN graph
  density.
- ADR-345's second finding reproduces independently on this new engine: at
  the same 84-memory corpus, bridge-survival gap over baseline measured
  +0.0pp for *both* engines (Dynamic and Static), not the targeted >=15pp.
  This is now supported by two independent implementations rather than one,
  strengthening the conclusion that the flat result is a property of this
  corpus size / dataset design rather than an artifact of `partition()`'s
  specific bugs.

## Evidence

See `docs/research/nightly/2026-09-08-static-mincut-forgetting/README.md`
for full benchmark output, the extended scaling probe (19-800 vertices), and
the extended determinism probe (side-by-side Dynamic vs. Static on ADR-345's
original fixture).

## Consequences

- `ruvector-mincut` gains a genuinely useful, low-risk, purely additive
  primitive: any one-shot / non-incremental min-cut query anywhere in the
  workspace (community detection, graph partitioning, ad hoc analysis
  scripts) can now get a correct, fast, deterministic answer without paying
  for machinery built for a different access pattern.
- `MincutGatedForgetting`'s core premise (a min-cut-derived structural
  eviction signal materially improves agent-memory compaction) remains
  unproven at any corpus size measured so far, independent of engine choice.
  A future nightly wanting to keep investigating this specific application
  should attack the *dataset/effectiveness* question next (larger corpora,
  denser bridge topologies, or a different structural feature entirely —
  e.g. betweenness centrality or articulation points, cheaper to compute
  than a global min cut) rather than the engine, which this ADR now
  considers a closed question for the one-shot use case.
- `mincut-forget`'s two Dynamic-engine call sites and the new two
  Static-engine call sites all remain feature-gated and off by default; no
  behavior change for any existing default-feature consumer of either crate.

## Alternatives Considered

- **Sparsify-then-partition** (route the k-NN graph through
  `ruvector-mincut`'s existing `sparsify` module before calling the dynamic
  `partition()`): not attempted this run. Plausible as a further latency
  reduction on top of either engine, but does not address the measured
  non-determinism (still routes through the same DashMap-order-dependent
  `MinCutWrapper`), so would not have closed ADR-345's filed follow-up item.
- **Fix `DynamicGraph` to use a deterministic hasher / sort edges before
  handing them to `MinCutWrapper`**: would likely fix the non-determinism
  finding alone, but does nothing for the dominant cost (the multi-instance
  replay-on-first-use design), which is the larger of the two problems by
  orders of magnitude per the scaling probe. Left as a smaller, separate
  potential follow-up against `MinCutWrapper` itself rather than pursued
  here.
- **Approximate / randomized min-cut with a fixed seed**: would trade the
  O(V^3) exactness for expected sub-cubic time at the cost of no longer
  being provably exact; not needed at the corpus sizes this application
  actually requires (a few hundred vertices), so the exact algorithm was
  preferred for simplicity and auditability.

## Implementation Plan

Implemented in full as part of this ADR (not phased):

1. `crates/ruvector-mincut/src/static_cut.rs` — `stoer_wagner_min_cut`,
   unit-tested (triangle, weighted triangle, disconnected graph, <2-vertex
   edge case, bridge-topology boundary detection, 25-trial determinism).
2. `RuVectorGraphAnalyzer::partition_static()` / `min_cut_static()` in
   `crates/ruvector-mincut/src/integration/mod.rs`, unit-tested (matches
   dynamic engine's cut value on a known graph; deterministic across 20
   independently-built `from_knn` analyzers).
3. `MincutEngine` enum, `soft_static()` / `hard_static()` constructors, and
   engine-dispatching `boundary_indices` in
   `crates/ruvector-agent-memory/src/graph_forget.rs`, unit-tested (bridge
   protection under both new constructors; determinism across 10 repeated
   calls).
4. Extended `mincut_gated_forgetting_bench.rs`,
   `mincut_scaling_probe.rs`, and `mincut_determinism_probe.rs` with
   Static-engine rows/columns for direct, same-methodology comparison.

## API Shape

```rust
// ruvector-mincut
pub mod static_cut {
    pub struct StaticCutResult { pub cut_value: f64, pub side_a: Vec<VertexId>, pub side_b: Vec<VertexId> }
    pub fn stoer_wagner_min_cut(graph: &DynamicGraph) -> Option<StaticCutResult>;
}
impl RuVectorGraphAnalyzer {
    pub fn partition_static(&self) -> Option<(Vec<VertexId>, Vec<VertexId>)>;
    pub fn min_cut_static(&self) -> Option<u64>;
}

// ruvector-agent-memory (feature = "mincut-forget")
pub enum MincutEngine { Dynamic, Static }
impl MincutGatedForgetting {
    pub fn soft_static(weights: CoherenceWeights, structural_bonus: f32) -> Self; // mincut_trials fixed at 1
    pub fn hard_static(weights: CoherenceWeights, protect_fraction: f32) -> Self; // mincut_trials fixed at 1
}
```

## Feature Flags

No new feature flags. `static_cut` and `partition_static()`/`min_cut_static()`
are unconditionally compiled into `ruvector-mincut` (no new dependency, pure
algorithm over the existing `DynamicGraph` type). `MincutEngine` and the
`_static` constructors live behind `ruvector-agent-memory`'s existing
`mincut-forget` feature, same as the rest of `graph_forget`.

## Benchmark Evidence

See the nightly README's "Benchmark Results" section for full tables. Headline
numbers (release build, this run, seed=341, same hardware as recorded in
"Run Identity" below):

| Comparison | Result |
|---|---|
| Static vs Dynamic, 84-memory bridge corpus (Soft) | 73.6x faster |
| Static vs Dynamic, 84-memory bridge corpus (Hard) | 64.7x faster |
| Static vs Dynamic, ADR-345 determinism-probe fixture (19 vertices) | 8,421x faster |
| Static engine non-determinism rate | 0% (0/50 trials divergent; exactly 1 distinct partition) |
| Dynamic engine non-determinism / degenerate rate (this run) | 66% (33/50 trials empty/degenerate) |
| Static-Soft slowdown vs. scalar baseline | 43.8x (bar was <=10x — **FAIL**) |
| Static-Hard slowdown vs. scalar baseline | 46.9x (bar was <=10x — **FAIL**) |
| Bridge-survival gap over baseline, both engines | +0.0pp (bar was >=15pp — **FAIL**) |
| Recall@10 delta, both engines | 0.00pp (bar was <=2pp — **PASS**) |
| Eviction-witness tamper detection | 20/20 (**PASS**, unchanged from ADR-345) |

## Security

No new attack surface: `static_cut` is a pure, deterministic, `#![deny(unsafe_code)]`-covered
algorithm over data already resident in `DynamicGraph`; it performs no I/O,
allocation is bounded by O(V^2) for the dense weight matrix, and it has no
update/mutation API (read-only over the graph it's given). The existing
eviction-witness tamper-detection result (20/20) is unaffected by engine
choice, since it exercises a wholly separate code path
(`witnessed_compaction`).

## Governance

Same governance posture as ADR-345: this ADR documents a **rejected**
application (the compaction policy) alongside a **promoted** primitive (the
static cut algorithm). No production default changes; `mincut-forget` stays
off by default. Future nightlies investigating agent-memory eviction
structure should treat both engines' flat bridge-survival result as settled
evidence, not re-litigate it without a materially different dataset design.

## Failure Modes

- O(V^3) means the static engine's own cost grows faster than the dynamic
  engine's amortized-per-update cost would, past some corpus size — this ADR
  does not claim the static engine wins at every scale, only that it is
  correct, deterministic, and a large win at the corpus sizes actually
  measured (up to 800 vertices). A future nightly should re-run the scaling
  probe at 1,000+ vertices before recommending the static engine unconditionally.
- Ties in Stoer-Wagner's per-phase vertex selection are broken
  deterministically by ascending vertex index (see `static_cut`'s module and
  function docs); this makes results reproducible but means the *specific*
  min cut returned among several equal-weight options is an implementation
  detail, not a canonical choice — callers that need a specific one of
  several equal-cost cuts (rather than "some correct minimum cut") should
  not rely on which one this returns.

## Migration

Purely additive; no migration required. Existing `soft()`/`hard()` callers
are unaffected. Adopting `soft_static()`/`hard_static()` (or
`partition_static()` directly) is opt-in.

## Rollback

Delete `static_cut.rs`, its `lib.rs` wiring, the two `RuVectorGraphAnalyzer`
methods, and the `MincutEngine`/`_static` additions in
`ruvector-agent-memory`. No other code depends on any of this ADR's new
surface as of this run.

## Open Questions

- Would a cheaper *approximate* structural signal (e.g. degree-normalized
  local conductance, or exact articulation points via a single DFS —
  O(V+E), no cut-value computation at all) hit the <=10x speed bar this ADR
  missed, while still correlating with true min-cut boundary membership
  closely enough to be useful? Not attempted this run; flagged as the most
  promising next step for the *effectiveness* half of ADR-345's original
  question, independent of the *engine* question this ADR closes.
- Does the flat bridge-survival result hold at 10x-100x larger corpora, where
  a scalar-only policy has more opportunity to accidentally evict a genuine
  structural bridge? ADR-345 originally wanted to test at ~2,000 memories
  and was blocked purely by the dynamic engine's latency; this ADR's static
  engine removes that specific blocker (its own cost at ~2,000 vertices,
  per the extended scaling probe's O(V^3) trend, would still be substantial
  — untested this run, but no longer categorically infeasible the way the
  dynamic engine was).
