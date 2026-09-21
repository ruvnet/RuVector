# ADR-346: Direct `MinCutBuilder` Bridge Detection for Mincut-Gated Forgetting

## Status

Accepted (narrow scope). Adds `ruvector_mincut::BoundaryMethod::DirectBuilder`
as an available, opt-in boundary-detection method on
`ruvector-agent-memory::graph_forget::MincutGatedForgetting`
(feature `mincut-forget`, itself off by default per ADR-345). The existing
`BoundaryMethod::WrapperPartition` remains the default returned by
`MincutGatedForgetting::soft`/`::hard`, so this change is purely additive:
no existing caller's behavior changes. `MincutGatedForgetting` itself
remains **not** promoted to a recommended or default compaction policy —
that verdict, from ADR-345, is unchanged by this ADR.

## Context

ADR-345 (`docs/research/nightly/2026-09-05-mincut-gated-forgetting`)
measured `MincutGatedForgetting`'s only boundary-detection method at the
time — `ruvector_mincut::RuVectorGraphAnalyzer::from_knn(...).partition()`
— at ~1,800-2,700x the `CoherencePolicy` baseline's compaction latency
(FAIL vs. a pre-registered <=100x gate) and non-deterministic across
repeated calls on an *identical, unchanged* graph (50% empty-result rate
over 30 trials on a fixed 19-vertex two-clique-plus-bridge topology). It
rejected the separate hypothesis that the resulting structural signal
improves bridge-memory survival (0.0pp measured gap vs. a >=15pp gate —
either failure alone would have been sufficient for rejection).

ADR-345 left three explicit open questions for future work; "Next Research
item 1" asked: does calling `ruvector_mincut::DynamicMinCut` (or
`ClusterHierarchy`) directly, instead of through
`RuVectorGraphAnalyzer`/`MinCutWrapper`, avoid the measured latency and
determinism problems? This ADR answers that question.

Investigation of `ruvector-mincut`'s internals
(`crates/ruvector-mincut/src/{integration,wrapper,algorithm}/mod.rs`) found
the root cause: `RuVectorGraphAnalyzer::partition()` routes through
`MinCutWrapper::process_instances()`, which lazily builds and replays every
edge into up to `MAX_INSTANCES = 100` geometrically-scaled `BoundedInstance`
data structures per call until one reports `ValueInRange` — expensive by
construction, and its result depends on which instance happens to answer
first, which is sensitive to `DashMap`/hash-map iteration order rather than
any property of the graph (no `rand` usage was found anywhere in the
`algorithm`, `instance`, or `witness` modules). By contrast, a one-shot
`ruvector_mincut::MinCutBuilder::with_edges(edges).build()` call does a
single `DynamicMinCut::from_graph` pass (one spanning-forest DFS plus one
tree-edge-cut computation over the resulting spanning tree) — algorithmically
far cheaper per query and untouched by `MinCutWrapper`'s machinery.

## Hypothesis

```text
Given the identical ADR-345 84-entry corpus (6 clusters x 12 core memories +
12 interpolated bridges, 32-dim, same hot-cluster access simulation, same
k-NN parameters, seed=341) and MincutGatedForgetting configuration,

when boundary detection uses BoundaryMethod::DirectBuilder (one-shot
MinCutBuilder::with_edges(...).build()) instead of
BoundaryMethod::WrapperPartition (RuVectorGraphAnalyzer::from_knn(...).partition()),

then compaction wall-clock slowdown vs. the CoherencePolicy baseline should
fall from the previously measured ~1,800-2,700x toward the pre-existing
100x gate, and per-call boundary-detection latency on a scaling probe
(n=19..400) should drop by at least an order of magnitude with
qualitatively better (near-linear, not super-linear) scaling,

subject to: (a) cargo test remaining green (no regression to the unchanged
WrapperPartition path or to bridge-detection correctness in the unit-test
topology), and (b) the determinism probe (30 trials, identical fixed graph)
showing a materially lower empty-result rate than WrapperPartition's
measured 27-57%.

This hypothesis is scoped to ADR-345's item 1 (latency and determinism)
only. It does NOT re-litigate ADR-345's separate, already-rejected
"does the structural signal improve bridge survival" hypothesis — changing
that hypothesis's acceptance criteria after seeing new results would
violate the nightly research process's own rule against redefining a
hypothesis post hoc. Bridge-survival and recall are measured and reported
as additional evidence, not as gates for this ADR's decision.
```

Full methodology and complete raw output (6 repeated runs of the main
benchmark, 2 repeated runs of the scaling probe, 3 runs of the determinism
probe including one invalidated run kept for the record) are in
`docs/research/nightly/2026-09-15-direct-mincut-bridge-detection/README.md`
and its `raw-runs.txt`.

## Decision

1. Add `ruvector_agent_memory::graph_forget::BoundaryMethod` (`WrapperPartition`
   | `DirectBuilder`) and a `boundary_method` field on `MincutGatedForgetting`.
   `soft()`/`hard()` default it to `WrapperPartition` — **no behavior change**
   for any existing caller (there are none outside this crate's own examples
   and tests, but the invariant is preserved regardless).
2. Implement `DirectBuilder` via `ruvector_mincut::MinCutBuilder::new()
   .with_edges(edges).build()` + `.partition()`, converting the k-NN
   neighbor list's raw cosine *distance* to an edge *weight* (`1/distance`)
   to match `RuVectorGraphAnalyzer::from_knn`'s own convention — the first
   implementation attempt skipped this conversion and produced an
   incorrect, inverted cut structure (unit tests caught it immediately; see
   the nightly README's "Failure modes").
3. Deduplicate the k-NN neighbor list's directed `(i,j)`/`(j,i)` pairs by
   unordered vertex pair before constructing edges: `MinCutBuilder`'s
   underlying `DynamicGraph::insert_edge` rejects a second insert of the
   same undirected pair with `EdgeExists`, failing `build()` immediately —
   a second bug this experiment's own probe scripts hit and fixed (also
   documented in "Failure modes").
4. Extend the existing `mincut_scaling_probe` and `mincut_determinism_probe`
   examples to measure `DirectBuilder` alongside the unchanged
   `WrapperPartition` measurement on identical inputs, and add a new
   `mincut_direct_builder_bench` example (the original
   `mincut_gated_forgetting_bench` from ADR-345 is left unmodified as a
   historical artifact of that ADR).
5. **Do not change `MincutGatedForgetting::soft`/`::hard`'s default.**
   `DirectBuilder` is recommended for any future use of
   `MincutGatedForgetting` that needs the structural signal at interactive
   latency, but this experiment surfaced a real, fully reproducible
   divergence in *which* minimum cut each method finds on the ADR-345
   corpus (see Evidence) that is not yet understood well enough to justify
   silently changing the default for a policy that ADR-345 already
   declined to promote.

## Evidence

Full raw output in the linked nightly README/raw-runs.txt; summarized here.

**Scaling probe** (ring k-NN, k=8, n in {19,50,100,200,400}), two repeated
runs:

| n | wrapper_partition | direct_builder | speedup |
|---:|---:|---:|---:|
| 19 | 69,663.9ms / 68,435.9ms | 0.40ms / 0.36ms | 173,462x / 188,894x |
| 50 | 86.8ms / 71.1ms | 1.95ms / 1.17ms | 44x / 61x |
| 100 | 536.7ms / 410.4ms | 4.01ms / 2.82ms | 134x / 145x |
| 200 | 2,679.2ms / 2,408.2ms | 28.97ms / 8.03ms | 93x / 300x |
| 400 | 11,481.6ms / 11,070.0ms | 22.40ms / 21.07ms | 512x / 525x |

`wrapper_partition` numbers reproduce ADR-345's original scaling table
(69,269.9 / 76.8 / 481.3 / 2,712.9 / 11,415.0ms) within run-to-run noise,
including the same n=19 multi-second-to-outlier behavior. `direct_builder`
scales near-linearly (sub-millisecond to ~20-30ms across a 21x increase in
n); `wrapper_partition` does not.

**Determinism probe** (fixed 19-vertex two-clique-plus-bridge graph, 30
trials), after fixing this experiment's own edge-dedup bug (see Decision
item 3 and "Failure modes" in the README):

| Method | empty/degenerate | bridge correctly flagged | avg latency/call |
|---|---:|---:|---:|
| wrapper_partition (run B) | 8/30 (27%) | 22/30 (73%) | 846.1ms |
| wrapper_partition (run C) | 15/30 (50%) | 15/30 (50%) | 787.4ms |
| direct_builder (run B) | 0/30 (0%) | 30/30 (100%) | 0.2ms |
| direct_builder (run C) | 0/30 (0%) | 30/30 (100%) | 0.2ms |

Run C's wrapper numbers (50%/50%) are an exact reproduction of ADR-345's
originally reported 50% empty-result rate. `direct_builder` was empty 0/60
times and correctly flagged the bridge 60/60 times across both runs — fully
deterministic on this topology, at ~4,000x lower per-call latency.

**Main benchmark** (identical 84-entry corpus/seed to ADR-345), 6 repeated
runs:

| Metric | Candidate A (wrapper) | Candidate B (direct) |
|---|---:|---:|
| Slowdown vs. baseline (Soft) | 2,453x-2,800x (6/6 FAIL vs <=100x) | 65x-105x (5/6 PASS, 1/6 FAIL) |
| Slowdown vs. baseline (Hard) | 2,163x-2,778x (6/6 FAIL vs <=100x) | 72x-106x (5/6 PASS, 1/6 FAIL) |
| Speedup, B vs. A | — | 25.9x-33.7x, stable across all 6 runs |
| Bridge survival (Soft) | 66.7% (all 6 runs; matches ADR-345 exactly) | 50.0% (all 6 runs, deterministic) |
| Bridge survival (Hard) | 66.7% (all 6 runs) | 58.3% (all 6 runs, deterministic) |
| Recall@10 | 100.0% (all runs, both candidates) | 100.0% (all runs, both candidates) |

Candidate A's slowdown here (2,163x-2,800x) falls within ADR-345's reported
"1,800-2,700x range depending on run" — a reproducibility check on the
original result, passed. Candidate B's absolute compaction time is stable
(2.2-3.6ms across all 6 runs), but the *ratio* to a ~30-35 microsecond
baseline is measurement-noise-sensitive enough that one individual run's
Soft measurement (105.3x) and one Hard measurement (106.2x) crossed the
100x line, out of 12 individual measurements across 6 runs. This is
reported as a limitation of the inherited ratio-based gate at this corpus's
absolute scale (baseline latency in the tens of microseconds), not as
evidence against the underlying, very large and very consistent, absolute
latency improvement.

**Unresolved finding:** Candidate B's bridge survival (50.0%/58.3%) is
lower than candidate A/baseline's (66.7%) on this corpus, deterministically
and reproducibly across all 6 runs (zero variance — unlike candidate A's
own non-determinism at `mincut_trials=1`). A global minimum cut need not be
unique, and `DirectBuilder`'s one-shot spanning-tree-based method and
`WrapperPartition`'s `MinCutWrapper`-based method appear to resolve ties
differently, producing different (both structurally valid) boundary sets on
the same graph. This is evidence that the two methods are **not**
behaviorally interchangeable beyond latency/determinism, and is exactly why
this ADR does not change `MincutGatedForgetting`'s default despite
`DirectBuilder`'s decisive performance win.

## Consequences

- Anyone who does enable `mincut-forget` and wants `MincutGatedForgetting`
  at practical latency now has a documented, tested option
  (`boundary_method = BoundaryMethod::DirectBuilder`) that is 25-500x
  faster and fully deterministic on every topology measured here, instead
  of the ADR-345-rejected default path.
- ADR-345's overall verdict — do not promote `MincutGatedForgetting` as a
  recommended or default compaction policy — is unaffected. This ADR closes
  one of its three open questions (item 1) without reopening the others
  (items 2 and 3, and the new bridge-selection-divergence finding above,
  remain open).
- The `RuVectorGraphAnalyzer`/`MinCutWrapper` performance and determinism
  characteristics documented here are a hardening finding against
  `ruvector-mincut` itself, independent of `ruvector-agent-memory`: any
  other caller of `RuVectorGraphAnalyzer::partition()` for a single one-shot
  query (rather than the incremental-update use case `MinCutWrapper`
  appears designed for) would likely see the same latency and determinism
  characteristics, and may want the same one-shot `MinCutBuilder`
  alternative.

## Alternatives

- **`DynamicCanonicalMinCut`** (feature `canonical`,
  `crates/ruvector-mincut/src/canonical/dynamic/mod.rs`): incremental
  `add_edge`/`remove_edge` skip full recomputation when a mutation
  provably doesn't cross the cached cut — genuinely O(1)-amortized for
  incremental updates, but `MincutGatedForgetting` rebuilds its k-NN graph
  from scratch on every compaction call (no incremental edge stream to
  exploit), so this ADR's one-shot `MinCutBuilder` is the simpler, equally
  fast choice for this call pattern. Left as a candidate for a future
  incremental-compaction design.
- **`ClusterHierarchy`** (`crates/ruvector-mincut/src/cluster/mod.rs`): not
  used — its `compute_cluster_boundary`/`compute_vertex_boundary` do a full
  O(E) edge scan per cluster on every `rebuild()`, offering no latency
  advantage over `MinCutBuilder` for this single-shot use case while
  requiring a different graph-construction API.
- **`canonical::source_anchored::canonical_mincut`** (feature `canonical`):
  a deterministic-by-design, fixed-vertex-ordering Stoer-Wagner
  implementation with an explicit tie-breaking rule — a strictly stronger
  determinism guarantee than `MinCutBuilder`'s (which is deterministic *in
  practice*, per the measured 0/60 empty-result rate, but not proven so by
  construction the way the canonical module documents itself to be). Not
  used here to keep this experiment's scope to the exact API ADR-345 named
  (`DynamicMinCut`) and avoid pulling in the `canonical` feature; worth
  revisiting for the "different cut selection" finding above, since a
  canonical construction might make the two methods' divergence
  *analyzable* (which cut is "more correct") rather than just observed.

## Implementation Plan

Complete as of this ADR: `BoundaryMethod` enum and `boundary_method` field
on `MincutGatedForgetting`, `boundary_from_one_partition_direct` (edge
weight inversion + unordered-pair dedup), extended `mincut_scaling_probe`
and `mincut_determinism_probe` examples, new `mincut_direct_builder_bench`
example, three new unit tests mirroring the existing `WrapperPartition`
bridge-dataset tests for `DirectBuilder`.

## API Shape

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryMethod {
    WrapperPartition, // default; ADR-345's original, unchanged candidate
    DirectBuilder,     // this ADR's candidate
}

pub struct MincutGatedForgetting {
    // .. unchanged fields ..
    pub boundary_method: BoundaryMethod, // new; defaults to WrapperPartition
}
```

## Feature Flags

No change: still gated entirely behind `mincut-forget` (optional
`ruvector-mincut` path dependency), off by default, exactly as ADR-345 left
it.

## Benchmark Evidence

See "Evidence" above and `docs/research/nightly/2026-09-15-direct-mincut-bridge-detection/README.md` / `raw-runs.txt` for full raw output across all repeated runs.

## Security

No new cryptographic primitive or witness-chain change. `DirectBuilder`
calls only safe, already-in-tree `ruvector-mincut` public API
(`MinCutBuilder`); it does not touch `witnessed_compaction`'s tamper-evident
eviction ledger, which ADR-345 already covers and which this ADR's
benchmark does not re-measure (no new claim is made about it).

## Governance

None beyond ADR-345's existing "no witness, no mutation" invariant, which
this ADR does not touch.

## Failure Modes

Two bugs surfaced and were fixed during this experiment (both documented in
the nightly README in full and summarized in "Decision" above):
missing distance-to-weight inversion (caught immediately by the new unit
tests failing), and missing reverse-direction edge deduplication in the
probe scripts (caught by an obviously-wrong `0.0ms`/100%-empty reading,
which by the "never hide failures" rule is kept in `raw-runs.txt` alongside
its diagnosis and fix rather than silently discarded).

The unresolved cut-selection divergence (bridge survival 50.0%/58.3% vs.
66.7%) is not a "failure mode" of this ADR's own gated hypothesis
(latency/determinism), but is documented as an open risk for anyone
choosing `DirectBuilder` expecting behavioral parity with `WrapperPartition`
beyond speed.

## Migration

None: `boundary_method` is a new field defaulting to the prior sole
behavior; no existing caller's output changes.

## Rollback

Remove `BoundaryMethod`, the `boundary_method` field, and
`boundary_from_one_partition_direct` with no impact on any existing
caller — `WrapperPartition` remains fully self-contained and unchanged.

## Rejection Criteria

This ADR's narrow hypothesis (latency + determinism improvement) is
accepted: `DirectBuilder` reproducibly and by a large margin outperforms
`WrapperPartition` on both axes across every run. It would have been
rejected had `DirectBuilder` failed to clear at least an order-of-magnitude
latency improvement, or shown any non-zero empty-result rate on the
determinism probe; neither occurred in any of the runs recorded here.

## Open Questions

1. Why do `WrapperPartition` and `DirectBuilder` select different minimum
   cuts on the ADR-345 84-entry corpus, and which (if either) is "more
   correct" for the bridge-protection use case? A canonical, fixed
   tie-breaking construction (`canonical::source_anchored::canonical_mincut`,
   see "Alternatives") may make this analyzable rather than just observed.
2. Would `DynamicCanonicalMinCut`'s true incremental amortized updates
   matter if `MincutGatedForgetting` were redesigned to maintain its k-NN
   graph incrementally across compaction calls instead of rebuilding it
   from scratch each time? Out of this ADR's scope.
3. ADR-345's own remaining open questions (its items 2 and 3 — the exact
   internal source of `MinCutWrapper`'s non-determinism, and whether the
   "outlier isolation, not bridge isolation" finding holds on real
   embeddings) are untouched by this ADR.
