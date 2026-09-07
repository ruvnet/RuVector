# Root-causing a "flaky" minimum-cut oracle: it wasn't flaky, it was silently dropping vertices

## Problem

A prior investigation into using `ruvector-mincut`'s dynamic minimum-cut
engine as a structural signal for agent-memory compaction hit a wall: on a
fixed, byte-identical 19-vertex graph, ~50% of repeated calls to
`RuVectorGraphAnalyzer::partition()` (a fresh analyzer per call, matching a
typical "recompute on demand" usage pattern) returned an empty or unusable
result. The investigation reasonably assumed this was some form of
hash-order-dependent non-determinism inside the underlying algorithm and
moved on, flagging the exact mechanism as an open question.

## Hypothesis

The open question was specific and falsifiable: is the non-determinism
caused by instance construction order, witness materialization, or
something else in the wrapper's instance-processing loop? Rather than guess,
this investigation built a corrected reproduction (the original probe's
graph had its own bug -- more on that below) and instrumented each layer of
the call path individually.

## Technical design

`RuVectorGraphAnalyzer::partition()` calls into `MinCutWrapper::query()`,
which returns a `MinCutResult::Value { witness, .. }` -- an *implicit*
representation of the cut: a seed vertex, a compressed bitmap of the "small
side" `U`, and a boundary size. To hand a caller two explicit `Vec<VertexId>`
sides, something has to materialize `U` and `V \ U` from that implicit form.

That something was `WitnessHandle::materialize_partition()`:

```rust
pub fn materialize_partition(&self) -> (HashSet<VertexId>, HashSet<VertexId>) {
    let u: HashSet<VertexId> = self.inner.membership.iter().map(|v| v as u64).collect();
    let max_vertex = self.inner.membership.max().unwrap_or(0) as u64;
    let v_minus_u: HashSet<VertexId> = (0..=max_vertex)
        .filter(|&v| !self.inner.membership.contains(v as u32))
        .collect();
    (u, v_minus_u)
}
```

`max_vertex` is computed from `membership.max()` -- the highest vertex ID
*inside the found cut set*, not the graph's actual highest vertex ID. A
minimum cut's smaller side essentially never happens to contain the graph's
globally-highest-numbered vertex (if it did, it likely wouldn't be the
smaller side). So in the overwhelmingly common case, every vertex numbered
above `U`'s own maximum is silently absent from *both* returned sets --
not misplaced, not duplicated, just gone. A caller checking `a.is_empty() ||
b.is_empty()` (the original investigation's check) only catches this when
the dropped tail happens to be the entire complement; a stricter check
(`a.len() + b.len() == n`) catches it every time the found set isn't the
graph's dense end.

Once this repository's corrected probe (see "A trap in the original probe"
below) was run against unmodified code, the empty/degenerate rate measured
**99-100%** across every topology tested -- worse than the ~50% originally
reported, and, more usefully, clearly *not* dominated by randomness: it
reproduced at the same rate every run, before any fix.

A second, smaller effect remained after fixing the above: `BoundedInstance`
(the small-graph exhaustive-search / LocalKCut-oracle backend) collects its
vertex list from a `HashSet<VertexId>`, using Rust's default `RandomState`
hasher -- randomized per `HashSet` instance. When a graph has genuine
automorphisms (duplicate or near-duplicate embedding vectors, common in
synthetic and some real clustering scenarios), multiple subsets tie for the
minimum boundary value, and "first tied subset encountered in mask-iteration
order wins" means the *specific* winning subset depends on that random hash
seed. The minimum cut *value* was always correct; *which* equally-valid
witness got returned was not stable across runs.

## Actual implementation

Two fixes, both in `ruvector-mincut`, no feature flag, no API removal:

1. **`WitnessHandle::materialize_partition_within(&self, universe: &[VertexId])`**
   -- a new method that takes the caller's actual vertex universe instead of
   guessing one, and `RuVectorGraphAnalyzer::partition()` updated to call it
   with `self.graph.vertices()`. The old `materialize_partition()` is left
   in place (existing doctests, possible external callers) with its
   documentation corrected to state the defect explicitly rather than
   hint at it.
2. **Deterministic tie-breaking** in `BoundedInstance::brute_force_min_cut()`
   and `search_for_cuts()`: sort the vertex/seed lists before using them,
   removing hash-seed randomization as an input to which tied-optimal
   witness gets selected.

## A trap in the original probe

Worth stating plainly, since it cost real time this run: the original
probe's synthetic graph builder created only *one* "gateway" vector
(interpolated between one cluster's axis and a shared bridge axis), when a
correct two-cluster-plus-bridge topology needs one gateway *per cluster*.
Without it, one entire cluster has zero path to the rest of the graph -- the
graph is genuinely disconnected, and `min_cut() == 0` is the *correct*
answer, indistinguishable at a glance from the witness-materialization bug's
symptom (an empty/incomplete partition). First-draft debugging in this run
briefly chased this as "even the brute-force path is broken," until
per-vertex adjacency printing showed a clean 8-8 vertex split with the
bridge fully isolated from one side. Fixing the probe's own graph (one
gateway per axis) was a prerequisite to trusting any subsequent measurement.

## Actual benchmark evidence

Three topologies, 200 fresh-analyzer trials each, before/after both fixes
(full methodology and raw transcripts in the companion `README.md` and
`raw-runs.txt`):

| | Baseline | Fix A only | Fix A + Fix B |
|---|---|---|---|
| n=19, empty/degenerate | 100% | 0% | 0% |
| n=19, distinct partitions/200 | 0 (all degenerate) | 2 | **1** |
| n=21, empty/degenerate | 100% | 0% | 0% |
| n=21, distinct partitions/200 | 0 | 2 | **1** |
| n=85, empty/degenerate | 99% | 0% | 0% |
| n=85, distinct partitions/200 | 2 | 2 | **1** |

Fix A alone eliminates all empty/degenerate results, confirming it as the
dominant defect. Fix B closes the remaining gap to full determinism (exactly
one partition observed across all 600 trials once both fixes are applied).
Latency is unaffected by either fix in either direction beyond noise.

`cargo test --release -p ruvector-mincut --lib`: 515 passed, 0 failed (5
pre-existing, unrelated ignores). `cargo test --release -p
ruvector-agent-memory --features mincut-forget`: 63 passed, 0 failed
(the downstream consumer from the prior investigation, unaffected in
behavior, confirmed regression-free).

## Limitations

This does **not** fix the separate, independently-measured latency problem
in the same call path (a ~1.2 second average per call at n=19, essentially
unchanged by this fix) -- that remains a real, open, unresolved issue. It
also does not revisit whether a global minimum cut is the *right* signal for
identifying semantically meaningful "bridge" memories in an embedding graph
(a separate, already-negative finding from the prior investigation on
synthetic data). This work fixes the plumbing between the algorithm and its
callers; it does not re-litigate whether the algorithm's answer is the one
downstream consumers actually want.

## Production relevance

This is a default-on bug fix in existing, already-shipped public API
(`RuVectorGraphAnalyzer::partition()`), not an experimental addition behind
a flag. Every existing and future caller -- community detection, graph
partitioning, and any future structural-signal experiment -- gets a
complete, reproducible result with no code change on their part. Silent
data loss in a partition API is exactly the kind of defect that looks like
"occasional flakiness" until someone builds a strict enough check to see
that it is, in fact, happening almost every time.

## RuVector ecosystem implications

`RuVectorGraphAnalyzer` is the general integration point between vector
similarity and this crate's minimum-cut engine across the ecosystem --
community detection, namespace-merge decisions, and any future
witness-gated structural claim about a memory or knowledge graph. A
structural witness that isn't reproducible from the same input can't
support anything built on top of the repository's broader witness-chain
infrastructure; this fix is a small, foundational precondition for that
larger goal, not a feature in its own right.

## Future direction

The natural next step in this lineage is the prior investigation's other
open question: whether using `DynamicMinCut`/`ClusterHierarchy` directly,
bypassing `MinCutWrapper`'s O(log n) bounded-instance sweep, resolves the
latency problem this fix left untouched. A second, smaller follow-up is
auditing whether other consumers of minimum-cut partitioning in this
codebase have their own independent copies of the same universe-guessing
mistake.

## References

- `crates/ruvector-mincut/src/instance/witness.rs`,
  `src/integration/mod.rs`, `src/instance/bounded.rs`,
  `examples/determinism_probe.rs` (this run's changes).
- ADR-346 (this run's architecture decision record, full evidence and
  rationale).
- The prior nightly's ADR and research README (root of the open question
  this run answers).
