# Nightly Research — Root-Causing and Fixing `RuVectorGraphAnalyzer::partition()`'s Cold-Start Latency

## Summary

Two prior nightlies flagged `ruvector-mincut`'s `RuVectorGraphAnalyzer::partition()`
as too slow to use in production but left the cost unexplained. The
2026-09-05 nightly (ADR-345) rejected `ruvector-agent-memory`'s
`MincutGatedForgetting` compaction policy partly on this basis, measuring
`partition()` scaling from ~77ms (n=50) to ~11.4s (n=400) on a synthetic
k-NN graph, plus an outlier ~69s at n=19. The 2026-09-11 nightly (ADR-346)
fixed an independent non-determinism bug in the same code path but
explicitly filed the latency itself as unresolved, naming
`connectivity::polylog::PolylogConnectivity` (already in-tree, unwired) as
a candidate replacement backend worth evaluating.

This run first **falsifies that specific suggestion by inspection**:
`PolylogConnectivity` (arXiv:2510.08297) implements plain dynamic
*connectivity* — `connected(u, v)` and `is_connected()` — with no notion of
cut value or witness whatsoever. It cannot replace `BoundedInstance`/
`MinCutWrapper`, which is what actually computes `partition()`'s answer; it
could only replace `MinCutWrapper::query()`'s cheap `DynamicConnectivity`
fast-path disconnection check, which this run's measurements confirm was
never the bottleneck.

Reading `MinCutWrapper::process_instances()` and `BoundedInstance`
end-to-end instead finds the real root cause: on a **cold** query (no
`last_min_cut` hint — the only scenario any prior nightly's probe has ever
measured, and the real access pattern of a fresh per-compaction-pass
analyzer), the wrapper scans bounded-range instances from index 0 upward,
instantiating a brand-new `BoundedInstance` per range until the first one
reports `ValueInRange`. Every instance visited receives the *identical*
full edge set via `apply_inserts(&all_edges)`, yet each one independently
redoes two expensive computations that do not depend on its own
`[lambda_min, lambda_max]` range at all:

1. For `vertices.len() > 50`: builds its own `ClusterHierarchy` from
   scratch (used only to pick candidate seed vertices for the LocalKCut
   oracle).
2. For `vertices.len() < 20`: runs `brute_force_min_cut`, an **exhaustive
   search over all `2^n` non-trivial vertex subsets** — the true global
   min cut, independent of range — from scratch, every time.

(2) dominates. With `RANGE_FACTOR = 1.2`, reaching the range that contains
a true min-cut of ~8 (the k=8 ring topology every prior probe used) takes
roughly `log(8)/log(1.2) ≈ 12` instances. At n=19 — one vertex below the
`< 20` brute-force cutoff — **every one of those ~12 instances
independently re-runs a full `2^19`-subset search** on an unchanged graph.

This run adds two small, additive, opt-in `ruvector-mincut` APIs —
`BoundedInstance::with_shared_hierarchy`/`without_hierarchy` and
`BoundedInstance::with_exact_hint`/`compute_exact_small_graph` — and wires
them together behind `MinCutWrapper`'s existing (and previously unused for
this purpose) `with_factory` extension point, so a cold-start batch
computes each redundant quantity **once** and hands every instance a
shared/pre-seeded copy instead of letting each rebuild it independently.
Zero changes to `MinCutWrapper`'s own code, and the default
`RuVectorGraphAnalyzer::new()`/`BoundedInstance::new()` path is completely
unchanged — this is purely an opt-in cold-start acceleration path.

## Hypothesis

```text
Given a cold (no last_min_cut hint) RuVectorGraphAnalyzer::partition() call
over a k=8 ring k-NN graph at n in {19, 50, 100, 200, 400} (the exact
topology and sizes from the 2026-09-05/09-11 nightlies' probes),

when the per-instance ClusterHierarchy build (n > 50) and the exhaustive
small-graph brute-force search (n < 20) are each computed once per
cold-start batch and shared across every BoundedInstance the batch creates,
instead of independently rebuilt by each,

then wall-clock partition() latency should drop materially at n=19 (the
brute-force regime) and at n in {100, 200, 400} (the hierarchy regime),

subject to: every variant (baseline, shared-hierarchy+exact-hint
"adaptive", hierarchy-skip-only ablation) returning the identical min-cut
value on every tested graph, and the full pre-existing ruvector-mincut test
suite (526 tests) remaining green.
```

Falsification: no measurable improvement at the sizes the root-cause
analysis predicts should improve, or any variant returning a *different*
min-cut value than baseline on any tested graph.

## Current baseline (re-measured, not assumed)

Before building anything, this run re-ran the exact pre-existing
`mincut_scaling_probe` to get *current* numbers — ~660 commits and one
unrelated latency fix (`boundary_incremental_bench`, 2026-10-03) have
landed since ADR-345's figures, so the old numbers are stale:

```text
$ cargo run --release -p ruvector-agent-memory --example mincut_scaling_probe --features mincut-forget
n=19    build=     0.182ms  partition= 65529.131ms
n=50    build=     0.281ms  partition=    57.197ms
n=100   build=     0.466ms  partition=   305.963ms
n=200   build=     1.053ms  partition=  1170.041ms
n=400   build=     2.047ms  partition=  3897.373ms
```

The n=400 number has improved ~3x since ADR-345 (11.4s → 3.9s), consistent
with the intervening `boundary_incremental_bench` fix to a different part
of the same `search_for_cuts` path — but the catastrophic n=19 outlier
(brute force regime, untouched by that fix) is essentially unchanged from
ADR-345's own reported ~69s outlier. The latency problem this nightly's
"Next Research" item asked about is real, current, and reproducible.

## Implementation

`crates/ruvector-mincut/src/instance/bounded.rs`:

- `BoundedInstance::with_shared_hierarchy(lambda_min, lambda_max, hierarchy: Arc<ClusterHierarchy>)` —
  pre-seeds the (now `Arc`-wrapped) `cluster_hierarchy` field so
  `ensure_hierarchy` never rebuilds it. Safe because the hierarchy is only
  ever read (`hierarchy.clusters.values()`) inside `search_for_cuts`, never
  mutated after construction.
- `BoundedInstance::without_hierarchy(lambda_min, lambda_max)` — sets a new
  `skip_hierarchy` flag so `ensure_hierarchy` is a permanent no-op;
  `search_for_cuts` falls back to its pre-existing "all vertices, sorted"
  seed list (the same branch it already takes whenever no hierarchy has
  been built).
- `BoundedInstance::with_exact_hint(lambda_min, lambda_max, value, witness)` —
  pre-seeds a new `exact_value_hint` field. `query()` consumes
  (`Option::take`) the hint *before* doing anything else: it seeds
  `best_witness`/the boundary cache exactly as the normal brute-force/
  oracle paths would, then answers `ValueInRange`/`AboveRange` directly
  from the already-known exact value — no search of any kind.
- `BoundedInstance::compute_exact_small_graph(edges)` — a small helper that
  runs one standalone, unbounded (`[0, u64::MAX]`) `BoundedInstance` query
  to get the exact global min cut. Only valid for `vertices.len() < 20`:
  for larger graphs `query()` never reaches `brute_force_min_cut`, and a
  genuinely unbounded `search_for_cuts` budget loop would be infeasible.

`crates/ruvector-mincut/src/integration/mod.rs`:

- `RuVectorGraphAnalyzer::new_with_factory`/`from_knn_with_factory` — thin
  wrappers around `MinCutWrapper::with_factory` (which already existed and
  already supported exactly this kind of substitution) so the acceleration
  can be exercised through the same public entry points
  `MincutGatedForgetting` and every other consumer actually use.

None of this touches `MinCutWrapper::process_instances()`, the
`ProperCutInstance` trait, or any existing constructor's default behavior.
`cargo test -p ruvector-mincut --lib` is 526/526 green (521 pre-existing +
5 new correctness tests added for the two mechanisms above), unchanged from
before this run.

`crates/ruvector-mincut/examples/coldstart_instance_sharing_bench.rs`
implements the benchmark: an `adaptive_factory()` combining both
mechanisms (hierarchy sharing for n>50, exact-hint sharing for n<20, keyed
by the graph's current edge count as a cold-start-appropriate staleness
check) and a `no_hierarchy_factory()` ablation that only ever skips
hierarchy construction, so the two redundancy sources' individual
contributions are visible separately.

## Benchmark methodology

- Release build (`cargo run --release`).
- Identical ring k-NN topology (k=8) and sizes `{19, 50, 100, 200, 400}` as
  the pre-existing `mincut_scaling_probe`, for direct comparability.
- One `partition()` call per (variant, size) pair — this is deliberately
  the cold-start case (no repeated queries, no hint available), matching
  what every prior nightly actually measured and what a fresh
  per-compaction-pass analyzer experiences in production.
- A separate correctness pass re-derives `min_cut()` for every
  (variant, size) pair and asserts it equals the baseline value — printed
  as an explicit `PASS`/`FAIL` line, not just a speed number.
- Exact command: `cargo run --release -p ruvector-mincut --example coldstart_instance_sharing_bench`.

## Results

Raw output of `cargo run --release -p ruvector-mincut --example coldstart_instance_sharing_bench`:

```text
variant               n       latency_ms   crossing_edges
--------------------  ------  -----------  --------------
baseline              19        64328.669               8
baseline              50           57.443               8
baseline              100         303.482               8
baseline              200        1180.145               8
baseline              400        3855.949               8
candidate_adaptive    19         3963.047               8
candidate_adaptive    50           57.321               8
candidate_adaptive    100         240.100               8
candidate_adaptive    200         963.693               8
candidate_adaptive    400        2881.246               8
candidate_no_hierarchy  19        64800.337               8
candidate_no_hierarchy  50           57.108               8
candidate_no_hierarchy  100         233.465               8
candidate_no_hierarchy  200         929.545               8
candidate_no_hierarchy  400        2843.044               8

correctness: PASS (all variants match baseline on every size)
```

| n   | baseline (ms) | adaptive (ms) | adaptive Δ | no_hierarchy (ms) | no_hierarchy Δ |
|-----|--------------:|--------------:|-----------:|-------------------:|---------------:|
| 19  |     64,328.7  |      3,963.0  | **-93.8% (16.2x)** |      64,800.3  | +0.7% (noise) |
| 50  |         57.4  |         57.3  | -0.2% (noise)      |          57.1  | -0.6% (noise) |
| 100 |        303.5  |        240.1  | -20.9%             |         233.5  | -23.0%        |
| 200 |      1,180.1  |        963.7  | -18.3%             |         929.5  | -21.2%        |
| 400 |      3,855.9  |      2,881.2  | -25.3%             |      2,843.0   | -26.3%        |

`correctness: PASS` across every (variant, size) pair: all three variants
return the same crossing-edge count (8, the k=8 ring's true minimum cut)
on every graph. Both the hierarchy-redundancy fix and the exact-hint fix
are **correctness-neutral and independently confirmed material wins** —
exactly what the hypothesis predicted, with one honest surprise:

- **n=19 (brute-force regime): confirmed, and the single largest result
  of this run.** 16.2x faster (93.8% latency reduction), from a
  practically-unusable 64.3 seconds down to 4.0 seconds for one
  `partition()` call. `candidate_no_hierarchy` — which does *not* touch
  the brute-force path at all — is statistically indistinguishable from
  baseline here (64.8s vs. 64.3s), confirming the win is attributable
  specifically to exact-hint sharing, not noise or an unrelated effect.
- **n=100/200/400 (hierarchy regime): confirmed, smaller magnitude.**
  Both candidates cut latency by 18-26%, consistently. This is a real,
  reproducible improvement, not the >10x the brute-force fix delivers —
  because the ClusterHierarchy build is only one *part* of each
  instance's cost in this regime; the LocalKCut oracle's own seed×budget
  sweep (which this run does not touch) still runs in full on every
  touched instance.
- **n=50: no effect (expected and correct).** `ensure_hierarchy`'s guard
  is `vertices.len() > 50`, so n=50 itself never builds a hierarchy — this
  size sits in neither fixed regime, and the near-identical numbers
  (57.1-57.4ms across all three variants) confirm neither mechanism
  fires spuriously.
- **Honest surprise: `candidate_no_hierarchy` (skip the hierarchy
  entirely) is slightly *faster* than `candidate_adaptive` (share one
  hierarchy) at every size where either applies (100/200/400).** On this
  symmetric ring k-NN topology, the hierarchy's boundary-vertex seed
  targeting does not meaningfully shrink the oracle's seed list relative
  to "all vertices" — every vertex has the same degree and a similar
  local neighborhood, so most vertices end up classified as boundary
  vertices of *some* cluster anyway. The shared hierarchy's build cost
  (paid once instead of ~12 times — still not zero) is therefore pure
  overhead on *this* topology relative to never building one. A graph
  with real, unevenly-sized community structure (the actual target use
  case — agent-memory's topic-clustered corpora) would very plausibly
  reverse this: a small, genuinely-informative boundary set would shrink
  the oracle's seed loop by much more than one hierarchy build costs. This
  run cannot distinguish the two cases on ring data alone — flagged as
  Next Research rather than guessed at.

## Acceptance Result

```text
ACCEPT
```

Both root-caused redundancy sources are eliminated with a measured,
reproducible, correctness-preserving latency improvement: a 16.2x
reduction in the pathological small-graph (brute-force) regime and an
18-26% reduction in the large-graph (hierarchy) regime, with zero changes
to any min-cut value returned on any tested graph and the full
pre-existing 521-test `ruvector-mincut` suite (526 with this run's
additions) staying green. The one part of the original hypothesis this
run cannot yet confirm is whether hierarchy *sharing* (vs. simply
*skipping* the hierarchy) is the better default on real, non-symmetric
data — recorded as Next Research, not assumed.

## Failure modes / limitations

- The cache in `adaptive_factory` keys staleness on `graph.num_edges()`
  alone — correct for the one-shot cold-start case this benchmark and
  `MincutGatedForgetting`'s per-pass analyzer both exercise (build once,
  query once, discard), but **not** a safe general-purpose invalidation
  scheme for a long-lived `MinCutWrapper` that interleaves many edge
  updates with many queries, where two different edge sets could
  coincidentally have the same edge count. Any production integration
  that reuses a wrapper across many updates needs a real content-based key
  (e.g. a hash over sorted edges) or an explicit "graph changed" signal,
  not edge count.
- `compute_exact_small_graph`'s one-shot brute force is still `2^n`: it
  removes the ~12x *redundancy* at n=19, not the underlying exponential
  cost of a single brute-force pass. A graph that grows past ~20-25
  vertices while still routing through this cutoff would need the cutoff
  itself lowered (and the polynomial oracle path strengthened to pick up
  the slack) — out of scope for this run, flagged as Next Research.
- Hierarchy sharing is provably correctness-preserving (the hierarchy is
  read-only and a pure function of the edge set), but its *performance*
  benefit is an empirical question this run measures rather than proves
  asymptotically — see Results.
- This run does **not** reopen ADR-345's rejection of `MincutGatedForgetting`.
  That decision rested on two independent grounds (latency *and* a
  measured lack of correlation between mincut boundary vertices and
  labeled ground-truth bridges on noisy clustered data). This run only
  addresses the first. Re-evaluating promotion would need a fresh run of
  ADR-345's full effectiveness benchmark, out of scope here.

## Next Research

0. Re-run this same three-variant comparison on a graph with real,
   uneven community structure (e.g. ADR-345's own 6-cluster-plus-bridges
   synthetic corpus, not a symmetric ring) to resolve the "honest
   surprise" above: does hierarchy *sharing* beat hierarchy *skipping*
   once the hierarchy's seed-targeting is actually informative, or is
   skipping it simply the better default everywhere? This run's ring
   topology cannot answer that question either way.
1. Re-run ADR-345's exact 84-entry `mincut_gated_forgetting_bench` corpus
   with `adaptive_factory()` wired into `MincutGatedForgetting` (would
   require a small additive change to `graph_forget.rs` to accept a
   factory, not attempted here) to see whether the "compaction slowdown
   <= 100x" gate now plausibly passes — a prerequisite to reopening
   promotion, not a substitute for it.
2. Lower (or make adaptive) the `vertices.len() < 20` brute-force cutoff,
   paired with strengthening `search_for_cuts`'s seed/budget search so the
   polynomial path remains correct and fast for 15-30 vertex graphs.
3. Replace the edge-count staleness key with a real content hash if this
   sharing strategy is ever wired into a long-lived, repeatedly-mutated
   wrapper instance rather than a one-shot cold-start analyzer.
4. `connectivity::polylog::PolylogConnectivity` remains unused in
   production. It is a real, tested, correct polylog dynamic-connectivity
   structure; it is simply solving a different problem than the one that
   was slow. It may still be worth wiring in as `MinCutWrapper`'s
   `DynamicConnectivity` fast-path implementation on its own technical
   merits (e.g. amortized update cost under heavy mutation), but that is
   an independent question from this run's latency fix, not evaluated
   here.

## References

[1] `docs/adr/ADR-345-mincut-gated-forgetting.md` and
`docs/research/nightly/2026-09-05-mincut-gated-forgetting/README.md` —
original latency/determinism measurements and rejection.
[2] `docs/adr/ADR-346-deterministic-mincut-witness-partition.md` and
`docs/research/nightly/2026-09-11-mincut-partition-determinism/README.md` —
determinism fix; names the latency problem as unresolved and suggests
`PolylogConnectivity` (falsified above).
[3] `crates/ruvector-mincut/examples/boundary_incremental_bench.rs` —
2026-10-03 fix to a different redundancy source in the same
`search_for_cuts`/`DeterministicLocalKCut` path (per-BFS-depth boundary
rescanning), responsible for the ~3x improvement in the re-measured n=400
baseline above relative to ADR-345's original number.
[4] Henzinger, Krinninger, Nanongkai, Saranurak et al., dynamic min-cut
wrapper algorithm (arXiv:2512.13105), implemented by `MinCutWrapper`.
[5] Dynamic connectivity with expected polylogarithmic worst-case update
time (arXiv:2510.08297), implemented by `PolylogConnectivity`.
