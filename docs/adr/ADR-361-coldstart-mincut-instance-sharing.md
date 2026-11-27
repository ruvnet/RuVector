# ADR-361: Cold-Start Instance Sharing for `ruvector-mincut`'s Bounded-Range Wrapper

## Status

Accepted. Additive, opt-in APIs on `ruvector-mincut`'s existing public
surface (`BoundedInstance`, `RuVectorGraphAnalyzer`,
`MinCutWrapper::with_factory`). No default behavior changes:
`RuVectorGraphAnalyzer::new()`/`BoundedInstance::new()` are byte-for-byte
unchanged in behavior; the acceleration is only reachable through the new
`*_with_factory` constructors.

## Context

The 2026-09-05 nightly (ADR-345) rejected `ruvector-agent-memory`'s
`MincutGatedForgetting` compaction policy partly because
`RuVectorGraphAnalyzer::partition()` scaled unacceptably with graph size
(~77ms at n=50 to ~11.4s at n=400, plus a ~69s outlier at n=19) and was
separately non-deterministic. The 2026-09-11 nightly (ADR-346) fixed the
determinism bug but explicitly left the latency unresolved, naming
`connectivity::polylog::PolylogConnectivity` — already in-tree, unwired —
as a candidate replacement backend worth evaluating.

This nightly (`docs/research/nightly/2026-11-14-coldstart-mincut-instance-sharing`)
first falsifies that suggestion: `PolylogConnectivity` implements plain
dynamic connectivity (`connected`, `is_connected`), not min-cut value or
witness computation, so it cannot replace `BoundedInstance`/
`MinCutWrapper` — only `MinCutWrapper`'s cheap, already-fast
`DynamicConnectivity` disconnection check, which measurement confirms was
never the bottleneck.

Reading `MinCutWrapper::process_instances()` and `BoundedInstance`
end-to-end instead finds the real root cause: on a cold query (no cached
`last_min_cut` hint — the scenario every prior nightly actually measured,
and the real access pattern of a fresh per-compaction-pass analyzer), the
wrapper instantiates a fresh `BoundedInstance` per bounded range,
scanning from index 0 until one returns `ValueInRange`. Every instance
visited receives the identical full edge set, yet each independently pays
two range-independent costs from scratch:

1. `vertices.len() > 50`: builds its own `ClusterHierarchy` (used only for
   oracle seed selection).
2. `vertices.len() < 20`: runs `brute_force_min_cut`, an exhaustive `2^n`
   subset search — the true global min cut, which does not depend on the
   instance's own range at all.

(2) dominates: at n=19, every one of the ~12 instances visited before the
true min-cut's range (`RANGE_FACTOR = 1.2`) independently re-ran a full
`2^19`-subset search, measured at ~64.3 seconds total for one
`partition()` call on this machine.

## Hypothesis

```text
Given a cold RuVectorGraphAnalyzer::partition() call over a k=8 ring k-NN
graph at n in {19, 50, 100, 200, 400},

when the per-instance ClusterHierarchy build (n > 50) and the exhaustive
small-graph brute-force search (n < 20) are each computed once per
cold-start batch and shared across every BoundedInstance the batch
creates, instead of independently rebuilt by each,

then wall-clock partition() latency should drop materially at n=19 and at
n in {100, 200, 400},

subject to: every variant returning the identical min-cut value on every
tested graph, and the full pre-existing ruvector-mincut test suite
remaining green.
```

## Decision

1. Add `BoundedInstance::with_shared_hierarchy(lambda_min, lambda_max,
   hierarchy: Arc<ClusterHierarchy>)` and `BoundedInstance::without_hierarchy`.
   `cluster_hierarchy`'s field type changes from `Option<ClusterHierarchy>`
   to `Option<Arc<ClusterHierarchy>>` (read-only after construction, so
   `Arc`-sharing across instances cannot change which cut any instance
   finds).
2. Add `BoundedInstance::with_exact_hint(lambda_min, lambda_max, value,
   witness)` and `BoundedInstance::compute_exact_small_graph(edges)`.
   `query()` consumes (`Option::take`) the hint before any search, seeding
   the normal `best_witness`/boundary caches from it so later incremental
   updates behave exactly as if the witness had been found the usual way.
3. Add `RuVectorGraphAnalyzer::new_with_factory`/`from_knn_with_factory`,
   thin wrappers around `MinCutWrapper::with_factory` (which already
   existed and already supported this kind of substitution) so the
   acceleration is reachable through the same entry points real consumers
   (`MincutGatedForgetting`, `CommunityDetector`, `GraphPartitioner`) use.
4. Ship `crates/ruvector-mincut/examples/coldstart_instance_sharing_bench.rs`,
   an `adaptive_factory()` combining both mechanisms (keyed by the graph's
   current edge count — a cold-start-appropriate staleness check, not a
   general-purpose cache) plus a `no_hierarchy_factory()` ablation
   isolating the hierarchy-only contribution.
5. Do **not** wire this into `ruvector-agent-memory::graph_forget` or
   change `MincutGatedForgetting`'s default behavior. ADR-345's rejection
   rested on two independent grounds (latency *and* a measured lack of
   correlation between mincut boundaries and labeled ground-truth bridges
   on noisy data); this ADR only closes the first. Reopening promotion
   needs a fresh run of ADR-345's effectiveness benchmark, out of scope
   here.
6. Do **not** wire `PolylogConnectivity` into `MinCutWrapper`. It solves a
   different problem than the one that was slow; whether it is worth
   adopting as `DynamicConnectivity`'s implementation on its own merits is
   an independent, unevaluated question.

## Evidence

Full raw output, per-size breakdown, and the "honest surprise" finding
(hierarchy-skipping slightly beats hierarchy-sharing on this symmetric
topology) are in the nightly README linked above. Summary:

| Gate | Measured | Result |
|---|---|---|
| n=19 latency reduction (adaptive) | 64,328.7ms → 3,963.0ms (-93.8%, 16.2x) | PASS (material) |
| n=100/200/400 latency reduction (adaptive) | -20.9% / -18.3% / -25.3% | PASS (material) |
| n=100/200/400 latency reduction (no_hierarchy ablation) | -23.0% / -21.2% / -26.3% | PASS (material) |
| n=50 (neither mechanism should fire) | ~57ms, all three variants | PASS (no spurious effect) |
| Correctness: identical min-cut value, every variant, every size | 8/8/8 crossing edges, all 15 (variant, size) pairs | PASS |
| `cargo test -p ruvector-mincut --lib` | 526/526 (521 pre-existing + 5 new) | PASS |

## Consequences

- `ruvector-mincut` gains two small, additive, opt-in acceleration
  mechanisms for the cold-start case, reachable via
  `MinCutWrapper::with_factory` (pre-existing extension point) and the new
  `RuVectorGraphAnalyzer::*_with_factory` constructors. Zero risk to
  existing callers: nothing about `new()`'s code path changed.
- The pathological small-graph (`< 20` vertices) case goes from
  practically unusable in an interactive or per-request path (64+ seconds)
  to tolerable for a background/batch path (~4 seconds) — still `2^n`
  brute force under the hood, not fixed asymptotically.
- `ruvector-agent-memory`'s `MincutGatedForgetting` remains rejected
  (ADR-345's decision stands unchanged); this ADR removes one of the two
  grounds for that rejection and documents exactly what would be needed to
  revisit it (Next Research item 1 in the nightly README).
- `connectivity::polylog::PolylogConnectivity` remains correct, tested,
  and unused in production — this ADR records why it was not applicable
  here so a future nightly does not re-propose it for the same reason.

## Alternatives Considered

- **Wire `PolylogConnectivity` in as the suggested fix.** Rejected by
  inspection: it answers a different question (connectivity) than the one
  that was slow (cut value/witness).
- **Rewrite `MinCutWrapper::process_instances()`'s scan order/algorithm.**
  Rejected as out of scope for a bounded nightly change: touches the
  paper-faithful core algorithm shared by every consumer, far larger blast
  radius than an additive, opt-in sharing layer.
- **Lower the `vertices.len() < 20` brute-force cutoff directly.** Not
  attempted: would shift (not remove) the exponential-cost cliff and
  requires first confirming the polynomial oracle path stays correct and
  fast in the 15-30 vertex range. Filed as Next Research item 2.

## Implementation Plan / API Shape

See Decision above; all changes are in `crates/ruvector-mincut/src/instance/bounded.rs`
and `crates/ruvector-mincut/src/integration/mod.rs`. No new crate, no new
dependency, no `Cargo.toml` changes.

## Feature Flags

None added. The new constructors are plain additive public API, gated by
nothing — callers opt in by calling them instead of the defaults.

## Security / Governance

No new attack surface: no new I/O, no new deserialization, no new
dependency. The `exact_value_hint` mechanism is a pure-computation
optimization whose only failure mode (a caller passing a mismatched hint)
is scoped to a `pub fn` whose doc comment states the correctness
precondition; it cannot be reached by any existing default-behavior
caller.

## Failure Modes / Limitations

See the nightly README's "Failure modes / limitations" section: the
cold-start cache's edge-count staleness key is not safe for a long-lived,
repeatedly-mutated wrapper; `compute_exact_small_graph` removes
redundancy, not the underlying `2^n` cost of one brute-force pass; and
whether hierarchy sharing beats hierarchy skipping on real (non-symmetric)
data is explicitly unresolved by this run.

## Migration / Rollback

No migration: nothing existing changes behavior. Rollback is a plain
revert of this ADR's commits; no data, schema, or wire format is touched.

## Rejection Criteria

Would have been rejected if any variant returned a different min-cut
value than baseline on any tested graph, or if neither candidate showed a
material latency improvement in either targeted regime. Neither occurred.

## Open Questions

1. Hierarchy sharing vs. skipping on real clustered data (Next Research 0).
2. Whether this unblocks `MincutGatedForgetting`'s latency gate specifically
   (Next Research 1) — not evaluated here, and ADR-345's rejection is not
   reopened by this ADR alone.
3. Whether the `< 20`-vertex brute-force cutoff itself should move (Next
   Research 2).
