# ADR-350: `DynamicMinCut` Backend for `MincutGatedForgetting` — Closes the ADR-345 Latency Blocker, Surfaces a New Connectivity Failure Mode

## Status

Accepted (as an additional opt-in backend; default unchanged). The
`MincutGatedForgetting` compaction policy itself remains **rejected for
production use as designed** — same verdict as ADR-345, evidence updated.

## Context

ADR-345 (`docs/research/nightly/2026-09-05-mincut-gated-forgetting`)
rejected `MincutGatedForgetting`, a structural (min-cut-boundary) eviction
signal for `ruvector-agent-memory` compaction, on two independent grounds:

1. **Performance.** `ruvector_mincut::RuVectorGraphAnalyzer::partition()`
   measured 76ms-11.4s per call for graphs of 50-400 vertices, and the
   policy's compaction pass was 1,800-2,700x slower than the scalar
   `CoherencePolicy` baseline on an 84-entry benchmark corpus — 27x over the
   pre-registered 100x "background job" ceiling.
2. **Effectiveness at the size performance forced.** At that same 84-entry
   corpus, the structural bonus made *zero* measurable difference
   (0.0pp gap) to bridge-memory survival versus the baseline.

ADR-345's own Next Research explicitly asked (item 1): "Repeat this
experiment against `ruvector_mincut::DynamicMinCut`... directly (bypassing
`RuVectorGraphAnalyzer`) to check whether the lower-level API avoids the
measured overhead," and (item 3): "If (1)... changes the performance
picture, re-run this exact benchmark (same hypothesis, same corpus, same
acceptance thresholds) without modification... a different result then
would be genuine evidence of progress." ADR-346's Next Research (item 2)
repeated the same ask independently. This ADR does exactly that.

## Hypothesis

```text
Given the same 84-entry synthetic corpus, k-NN construction, and
acceptance thresholds as ADR-345's benchmark (unmodified),

when MincutGatedForgetting's boundary-partition computation uses
ruvector_mincut::DynamicMinCut (algorithm::DynamicMinCut, a one-shot sparse
Stoer-Wagner solver) instead of RuVectorGraphAnalyzer (MinCutWrapper's
subpolynomial dynamic-update machinery, built for repeated churn, not
one-shot queries),

then compaction wall-clock slowdown versus the CoherencePolicy baseline
should drop from the previously-measured 1,800-2,700x (FAIL against the
100x gate) to something meaningfully smaller,

subject to: (a) the bridge-survival-gap and recall thresholds being
evaluated on the same unmodified terms as ADR-345 (not required to newly
pass — that is a separate, independent axis of the original hypothesis),
and (b) `RuVectorGraphAnalyzer`'s own rows in the same benchmark run
staying unmodified, so both backends are measured under identical corpus
generation in a single, directly comparable run.
```

## Decision

1. Add `ruvector_mincut::DynamicMinCut`-backed boundary computation to
   `crates/ruvector-agent-memory/src/graph_forget.rs` as a new
   `MincutBackend` enum (`GraphAnalyzer` default / `DynamicMinCut` opt-in via
   `.with_backend(...)`), not a replacement — `RuVectorGraphAnalyzer` remains
   the default and its code path is byte-for-byte unchanged.
2. Add `MincutGatedForgetting::boundary_size(&self, entries)`, a public
   diagnostic returning how many candidates the structural signal currently
   touches — added because this run's own exploratory follow-up (below)
   found the signal silently going to zero at larger corpus sizes, which a
   caller has no other way to observe.
3. Extend `examples/mincut_scaling_probe.rs` and
   `examples/mincut_determinism_probe.rs` (both already in-tree from
   ADR-345/346, explicitly marked "not part of the shipped research
   artifact") with head-to-head `DynamicMinCut` measurements at the same
   graph sizes/trials, alongside the existing `RuVectorGraphAnalyzer` rows
   (unmodified).
4. Extend `examples/mincut_gated_forgetting_bench.rs` — the actual
   pre-registered acceptance benchmark — with two additional rows
   (`Soft`/`Hard` under `MincutBackend::DynamicMinCut`) run against the
   *same* corpus generation and thresholds as the original two
   `RuVectorGraphAnalyzer` rows, which are unmodified.
5. Add `examples/mincut_gated_forgetting_scale_probe.rs`, a new, explicitly
   exploratory (not pre-registered, does not affect the ACCEPT/REJECT
   verdict below) probe that reuses the same corpus generator at 10x and
   50x scale — only affordable to run at all because of (1)'s speedup — to
   check whether the 0.0pp bridge-survival gap is a small-corpus artifact.

## Evidence

All numbers below are from this run, release builds
(`cargo build --release`), this session's Linux x86_64 container, Rust
1.94.1. Raw commands are listed under Benchmark Evidence.

### Scaling probe (`mincut_scaling_probe.rs`, ring k-NN graph, k=8)

| n | GA build | GA partition | DMC build+cut | DMC `.partition()` | Speedup (total) |
|---:|---:|---:|---:|---:|---:|
| 19   | 0.255ms | 78,815.355ms | 0.230ms | 0.001ms | 342,089x |
| 50   | 0.290ms | 91.102ms     | 0.513ms | 0.001ms | 178x |
| 100  | 0.625ms | 533.245ms    | 1.269ms | 0.004ms | 421x |
| 200  | 1.365ms | 2,975.004ms  | 3.495ms | 0.003ms | 852x |
| 400  | 2.594ms | 11,843.611ms | 11.383ms | 0.007ms | 1,041x |
| 800  | 4.931ms | 26,024.480ms | 34.200ms | 0.011ms | 761x |
| 2000 | 16.256ms | 84,589.057ms | 286.401ms | 0.025ms | 295x |

`DynamicMinCut` scales far better in this range and puts a 2,000-vertex
one-shot cut inside 300ms — a graph size that was untested by ADR-345
because it was untestable (84.6s/call extrapolated from this run's own GA
column at n=2000).

### Determinism probe (`mincut_determinism_probe.rs`, 19-vertex two-clique + bridge, 50 trials)

| Backend | avg/call | empty/degenerate | bridge detected as boundary | distinct partitions |
|---|---:|---:|---:|---:|
| GraphAnalyzer (post-ADR-346 fix) | 917.6ms | 0/50 (0%) | 50/50 (100%) | not tracked |
| DynamicMinCut | 0.058ms | 0/50 (0%) | 50/50 (100%) | **1** (fully deterministic) |

ADR-346's fix is confirmed working (0% empty, versus ADR-345's original
15/30 = 50%). `DynamicMinCut` is additionally ~15,800x faster on this graph
and returns the literal same partition on every one of 50 trials (expected:
`algorithm::exact::minimum_cut` sorts vertices/edges before processing,
independent of `DashMap` iteration order).

### Acceptance benchmark (`mincut_gated_forgetting_bench.rs`, unmodified 84-entry corpus/thresholds, seed 341)

```text
Policy                           Bridge Surv.    Recall@10  Compaction (us)
----------------------------------------------------------------------------
CoherenceWeighted                       66.7%       100.0%               35
MincutGatedForgetting-Soft (GA)         66.7%       100.0%            96748
MincutGatedForgetting-Hard (GA)         66.7%       100.0%            94901
MincutGatedForgetting-Soft (DMC)        66.7%       100.0%             1322
MincutGatedForgetting-Hard (DMC)        66.7%       100.0%             1469

  Soft (GraphAnalyzer)   bridge-survival gap (+0.0pp) >= 15pp : FAIL
  Soft (GraphAnalyzer)   |recall delta| (0.00pp) <= 2pp        : PASS
  Soft (GraphAnalyzer)   compaction slowdown (2764.2x) <= 100x : FAIL
  Hard (GraphAnalyzer)   bridge-survival gap (+0.0pp) >= 15pp : FAIL
  Hard (GraphAnalyzer)   |recall delta| (0.00pp) <= 2pp        : PASS
  Hard (GraphAnalyzer)   compaction slowdown (2711.5x) <= 100x : FAIL
  Soft (DynamicMinCut)   bridge-survival gap (+0.0pp) >= 15pp : FAIL
  Soft (DynamicMinCut)   |recall delta| (0.00pp) <= 2pp        : PASS
  Soft (DynamicMinCut)   compaction slowdown (37.8x) <= 100x   : PASS
  Hard (DynamicMinCut)   bridge-survival gap (+0.0pp) >= 15pp : FAIL
  Hard (DynamicMinCut)   |recall delta| (0.00pp) <= 2pp        : PASS
  Hard (DynamicMinCut)   compaction slowdown (42.0x) <= 100x   : PASS
  Tamper detection (20/20)                                     : PASS

DynamicMinCut vs. GraphAnalyzer backend (same corpus, same policy logic)
  Soft: 73.2x faster compaction
  Hard: 64.6x faster compaction

=> REJECT: one or more mandatory acceptance thresholds failed (see above).
```

**The performance axis flips from FAIL to PASS** (2,711-2,764x slowdown →
37.8-42.0x, under the pre-registered 100x gate; 64.6-73.2x faster than the
`GraphAnalyzer` backend on the identical corpus). **The effectiveness axis
is unchanged**: 0.0pp bridge-survival gap on both backends, identical to
ADR-345's original finding — confirming the effectiveness gap is not an
artifact of the backend's mincut implementation.

### Exploratory scale probe (`mincut_gated_forgetting_scale_probe.rs`, DynamicMinCut only, NOT pre-registered)

| Scale | Memories | Policy | Bridge Surv. | Recall@10 | Compaction | Boundary size |
|---:|---:|---|---:|---:|---:|---:|
| 1  | 84   | CoherenceWeighted | 66.7% | 100.0% | 0.03ms | – |
| 1  | 84   | Soft (DMC)        | 66.7% | 100.0% | 1.26ms | 6 |
| 1  | 84   | Hard (DMC)        | 66.7% | 100.0% | 1.19ms | 6 |
| 10 | 840  | CoherenceWeighted | 16.7% | 99.0%  | 0.31ms | – |
| 10 | 840  | Soft (DMC)        | 16.7% | 99.0%  | 33.9ms | **0** |
| 10 | 840  | Hard (DMC)        | 16.7% | 99.0%  | 34.0ms | **0** |
| 50 | 4200 | CoherenceWeighted | 67.7% | 100.0% | 1.66ms | – |
| 50 | 4200 | Soft (DMC)        | 67.7% | 100.0% | 868.0ms| **0** |
| 50 | 4200 | Hard (DMC)        | 67.7% | 100.0% | 845.2ms| **0** |

This is a new, distinct finding, not previously testable: the boundary set
is genuinely non-empty at n=84 (6/84 candidates) yet still moves zero
survivors, and at 840/4,200 entries the boundary set is **empty at every
sampled scale** — `MincutBackend::{GraphAnalyzer,DynamicMinCut}` both fall
back to "no structural signal" whenever the k-NN graph as a whole is
disconnected, and a larger absolute bridge count makes at least one
isolated (below-`min_similarity`) bridge vertex increasingly likely. Scale
does not fix the effectiveness problem; at this corpus generator's
parameters, it makes the signal disappear entirely instead.

## Consequences

- `ruvector-agent-memory` gains a real, tested, documented second backend
  choice (`MincutBackend::DynamicMinCut`) for any *future* structural-signal
  work in this crate, at zero new Cargo dependency (same `ruvector-mincut`
  crate, different already-exported type) and with `boundary_size` for
  runtime observability.
- The previously blocking "impractically slow" objection to
  `MincutGatedForgetting` is closed: a future attempt to make the
  *effectiveness* axis work is no longer also blocked by an infeasible
  compute cost, and can now be tested at corpus sizes in the thousands
  within a practical wall-clock (2,000-vertex cut in 286ms vs. an
  extrapolated ~85s for the old backend).
- `MincutGatedForgetting` itself is **still not promoted** — the same
  REJECT verdict as ADR-345 stands, now for a narrower and better-evidenced
  reason (bridge-survival gap and, at scale, graph disconnection — not
  latency).
- `RuVectorGraphAnalyzer`'s own cost is now *additionally* evidenced as a
  general concern for any one-shot (build-once, query-once) use of
  `ruvector-mincut` elsewhere in the ecosystem — `MinCutWrapper`'s
  geometric-instance-ladder machinery is designed for graphs under
  *repeated* incremental churn, and pays a large, avoidable setup cost when
  used for single queries. `DynamicMinCut` is the correct choice for that
  usage pattern.

## Alternatives Considered

- **`connectivity::polylog::PolylogConnectivity`** (flagged in ADR-346 Next
  Research item 2 as a possible faster backend for `BoundedInstance`) — not
  evaluated in this run. It would change `RuVectorGraphAnalyzer`/
  `MinCutWrapper` internals directly (a `ruvector-mincut` change affecting
  all 18 dependent crates), a materially larger-blast-radius change than
  adding an alternative call site in one downstream crate. Left as future
  work if `RuVectorGraphAnalyzer`'s own one-shot-query cost needs fixing at
  its source rather than routed around.
- **`cluster::ClusterHierarchy::boundary_size`** (the exact API named in
  ADR-345's Next Research item 1) — inspected; `boundary_size` in this
  crate is a `WitnessHandle`/`ProperCutInstance` method (`instance/witness.rs`),
  not a `ClusterHierarchy` method, and operates on a single already-built
  min-cut witness rather than replacing the min-cut computation itself.
  `algorithm::DynamicMinCut` was the closer match to "a lower-level API that
  bypasses `MinCutWrapper`'s dynamic-update ladder," and is the crate's own
  documented top-level `DynamicMinCut` type (`use ruvector_mincut::{MinCutBuilder,
  DynamicMinCut};` in the crate's own doctest).

## Implementation Plan

Landed in this change:
`crates/ruvector-agent-memory/src/graph_forget.rs` (`MincutBackend` enum,
`with_backend`, `boundary_from_dynamic_mincut`, `boundary_size`),
`crates/ruvector-agent-memory/src/lib.rs` (re-export), three probe examples
extended, one new probe example added, unit tests added. No changes to
`ruvector-mincut` itself.

## API Shape

```rust
pub enum MincutBackend { GraphAnalyzer /* default */, DynamicMinCut }

impl MincutGatedForgetting {
    pub fn with_backend(self, backend: MincutBackend) -> Self;
    pub fn boundary_size(&self, entries: &[MemoryEntry]) -> usize;
}
```

Fully additive: existing `soft()`/`hard()` constructors and their default
behavior (`MincutBackend::GraphAnalyzer`) are unchanged; no breaking change
to any existing caller.

## Feature Flags

Unchanged: still gated entirely behind the crate's existing `mincut-forget`
feature (optional `ruvector-mincut` dependency). No new feature flag added.

## Benchmark Evidence

```bash
cargo build --release -p ruvector-agent-memory --features mincut-forget \
  --example mincut_scaling_probe --example mincut_determinism_probe \
  --example mincut_gated_forgetting_bench --example mincut_gated_forgetting_scale_probe

./target/release/examples/mincut_scaling_probe
./target/release/examples/mincut_determinism_probe
./target/release/examples/mincut_gated_forgetting_bench
./target/release/examples/mincut_gated_forgetting_scale_probe
```

## Security

No new cryptographic surface. `compact_witnessed`'s Ed25519/witness-chain
machinery (unchanged in this run) still applies identically regardless of
backend, since backend selection only affects which vertices receive the
structural score bonus, not the eviction-witnessing path. Tamper-detection
(20/20 single-byte-flip trials) re-verified unchanged.

## Governance

No production default changes. `MincutBackend::DynamicMinCut` is opt-in;
`MincutGatedForgetting` remains an experimental, non-default policy behind
`mincut-forget`, not wired into any default compaction path.

## Failure Modes

1. **Graph disconnection at scale** (new finding, this run): the k-NN
   similarity graph over a compaction candidate set can become disconnected
   as the corpus (and therefore the absolute bridge count) grows, at which
   point `boundary_indices` returns empty and the structural signal is
   silently inert for the *entire* compaction pass, not just the isolated
   vertex. Neither backend currently distinguishes "no cut-worthy structure
   exists" from "the graph happened to disconnect" — `boundary_size` makes
   this observable but does not fix it.
2. **`min_similarity`/`k_neighbors` sensitivity**: the disconnection above
   is a direct function of `min_similarity` (0.05 default) and
   `k_neighbors` (8 default) relative to how tightly clustered the actual
   embedding space is; not evaluated here across a parameter sweep (Next
   Research).
3. `DynamicMinCut::from_graph` takes an owned `DynamicGraph` per call — this
   run's implementation rebuilds one from scratch on every
   `boundary_indices` call (matching `RuVectorGraphAnalyzer::from_knn`'s own
   existing per-call rebuild pattern; not a new inefficiency relative to the
   backend being compared against, but also not eliminated).

## Migration

None required; purely additive.

## Rollback

Revert this change; `RuVectorGraphAnalyzer` remains the sole, default,
unaffected code path throughout, so rollback carries zero risk to any
existing behavior.

## Rejection Criteria

This ADR's own narrow hypothesis (performance axis) would have been
rejected if `DynamicMinCut`'s compaction slowdown vs. baseline had still
exceeded the pre-registered 100x gate, or if it had failed to reproduce the
same bridge-survival/recall numbers as the `GraphAnalyzer` backend on
identical input (which would indicate a correctness bug in the new backend,
not a genuine backend-speed comparison). Neither occurred: slowdown dropped
to 37.8-42.0x (PASS), and bridge-survival/recall numbers are bit-for-bit
identical between backends at this corpus (66.7%/100.0% both ways),
consistent with both backends computing valid (if possibly different)
minimum cuts of the same graph.

## Open Questions

1. Why is the bridge-survival gap 0.0pp even when the boundary set is
   non-empty (6/84 at the acceptance-benchmark corpus)? Is the bonus
   magnitude (δ=0.5) or protected fraction (20%) too small relative to this
   corpus's actual score spread, or do the 6 boundary vertices simply not
   overlap with the candidates `CoherencePolicy` would otherwise evict?
   Not instrumented in this run.
2. At what `min_similarity`/`k_neighbors` setting (if any) does the k-NN
   graph stop disconnecting at 840+ entries for this corpus generator's
   cluster/bridge ratio?
3. Does `PolylogConnectivity` change `RuVectorGraphAnalyzer`'s own one-shot
   cost enough to matter, independent of this ADR's backend swap? (ADR-346
   Next Research item 2, still open.)
