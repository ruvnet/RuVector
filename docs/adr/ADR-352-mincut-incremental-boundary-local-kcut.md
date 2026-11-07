# ADR-352: Incremental Boundary Maintenance in `ruvector-mincut`'s `DeterministicLocalKCut`

## Status

Accepted. Non-breaking performance fix, merged into `ruvector-mincut` directly
(no feature flag — `deterministic_bfs` is a private implementation detail of
`DeterministicLocalKCut::search`, and its public contract, including
determinism, is unchanged). Does **not** reverse
[ADR-345](./ADR-345-mincut-gated-forgetting.md)'s rejection of
`MincutGatedForgetting`: that policy remains REJECTED, now with stronger
evidence (see Evidence).

## Context

[ADR-345](./ADR-345-mincut-gated-forgetting.md) (2026-09-05) rejected
`ruvector-agent-memory`'s `MincutGatedForgetting` policy on two independent
grounds: (a) the global minimum cut does not reliably isolate
human-intended "bridge" memories, and (b)
`RuVectorGraphAnalyzer::partition()` latency scaled from ~77ms (n=50) to
~11.4s (n=400) on a synthetic k-NN graph, making it impractical above a few
hundred memories. [ADR-346](./ADR-346-deterministic-mincut-witness-partition.md)
(2026-09-11) fixed a separate non-determinism bug in the same call path but
explicitly left the latency problem out of scope, naming it "a distinct
algorithmic-complexity problem" and flagging
`BoundedInstance::search_for_cuts`'s seed/budget fan-out as the likely
cause in its Next Research list, alongside evaluating the already-in-tree
`connectivity::polylog::PolylogConnectivity` as a possible replacement
backend.

Reading `PolylogConnectivity` (`crates/ruvector-mincut/src/connectivity/polylog.rs`)
shows it answers dynamic *connectivity* queries ("is there a path between u
and v", "is the graph connected") in O(log n) — it has no notion of a
minimum-cut value or witness partition, so it cannot directly replace
`BoundedInstance`'s cut-search backend; `BoundedInstance` already has its
own O(1)-amortized connectivity fast path (`FragmentingAlgorithm`) for the
one thing `PolylogConnectivity` would help with. The actual dominant cost,
confirmed by reading `DeterministicLocalKCut::deterministic_bfs`
(`crates/ruvector-mincut/src/localkcut/paper_impl.rs`), is narrower and
more mechanical: the BFS recomputes the cut boundary **from scratch** —
`calculate_boundary(graph, &visited)`, an O(edges incident to `visited`)
scan — at **every** BFS depth, for every depth in `0..=radius`
(`radius` defaults to 20 via `DeterministicLocalKCut::new`'s typical
construction in `BoundedInstance`). That cost is then paid again for every
`(seed, budget)` pair `search_for_cuts` tries, and again for each of the
O(log n) geometric-range instances `MinCutWrapper` queries before the first
`AboveRange` result. The redundant rescanning compounds multiplicatively
with every one of these factors — it is a straightforward algorithmic
inefficiency, not an inherent property of the paper's algorithm.

## Hypothesis

```text
Given DeterministicLocalKCut::deterministic_bfs's full from-scratch
calculate_boundary(graph, &visited) call at every BFS depth 0..=radius,

when boundary_edges is instead maintained incrementally — updated only for
edges incident to the vertices newly added at each layer, using the same
insert/remove-by-membership rule calculate_boundary already defines —

then DeterministicLocalKCut::search (and therefore
BoundedInstance::search_for_cuts and RuVectorGraphAnalyzer::partition())
should measurably reduce wall-clock latency on the exact ADR-345
reproduction graphs (ring k-NN, n in {50,100,200,400}),

subject to: every returned cut value and witness staying bit-identical to
the from-scratch computation (correctness), partition() determinism
(ADR-346) remaining unchanged, and the full existing ruvector-mincut test
suite for the touched modules (localkcut, instance, integration, wrapper)
staying green.
```

This hypothesis is deliberately narrower than ADR-345's: it claims a
measurable, correctness-preserving latency reduction in one specific
primitive, not that `MincutGatedForgetting`'s 100x compaction-slowdown gate
is met. Falsification would be: no measurable speedup, a changed cut value
on any witness, or a regression in `partition()` determinism.

## Decision

Rewrite `DeterministicLocalKCut::deterministic_bfs`
(`crates/ruvector-mincut/src/localkcut/paper_impl.rs`) to maintain a
`boundary_edges: HashSet<EdgeId>` incrementally via a new
`update_boundary_incremental` helper, called once for the initial seed set
and once per newly-expanded BFS layer, instead of calling
`calculate_boundary` on the whole `visited` set at every depth. The
incremental rule is exactly `calculate_boundary`'s own definition applied
only to the vertices that just changed membership: for each edge incident
to a newly-added vertex, the edge becomes internal (removed from the
boundary set, a no-op if absent — covers two vertices added in the same
layer) if its other endpoint is in the now-updated `visited` set, otherwise
it crosses the cut (inserted). Edges not incident to the newly-added
vertices are untouched, which is what turns the whole BFS's boundary
bookkeeping from O(radius · edges-incident-to-visited) into O(edges
touched across the whole explored region), independent of `radius`.

`calculate_boundary` itself is kept, unchanged, as a from-scratch reference
implementation — it has an existing direct caller
(`test_boundary_calculation`) and is now also used by the new regression
test as the independent oracle the incremental path is checked against.

No public API changed: `LocalKCutOracle::search`'s signature, return type,
and `BoundedInstance`/`MinCutWrapper`/`RuVectorGraphAnalyzer`'s public
methods are untouched. This is why the fix needs no feature flag, unlike
ADR-345/347's additions.

## Evidence

All runs: this machine, `cargo run --release`, deterministic-seed scripts,
raw output in
[`docs/research/nightly/2026-10-03-mincut-incremental-boundary-bfs/raw-runs.txt`](../research/nightly/2026-10-03-mincut-incremental-boundary-bfs/raw-runs.txt).

**Correctness.** New regression test
`localkcut::paper_impl::tests::test_incremental_boundary_matches_from_scratch_across_random_graphs`:
20 random sparse graphs (10–85 vertices, seeded), every witness `search()`
returns checked against an independent from-scratch `calculate_boundary`
recomputation over the witness's own vertex set. 0 mismatches across 100+
witnesses checked. All pre-existing `localkcut` (41), `instance`+
`integration`+`wrapper` (57), and crate-level `determinism_tests.rs` (3) /
`integration_tests.rs` (7) tests pass unchanged, including the
`calculate_boundary`-dependent `test_boundary_calculation`.

**Isolated primitive benchmark** (`examples/boundary_incremental_bench.rs`,
ring k-NN graph k=8, budget=10 chosen so no cut is found and every call
runs the full `radius=20` BFS — a worst-case stress of the fix, not a
favorable case):

| n | OLD (full rescan) | NEW (incremental) | speedup | per-seed cut values |
|---|---|---|---|---|
| 50 | 2.80ms | 2.17ms | 1.3x | identical |
| 100 | 14.25ms | 8.98ms | 1.6x | identical |
| 200 | 81.55ms | 36.15ms | 2.3x | identical |
| 400 | 353.24ms | 115.00ms | 3.1x | identical |

**End-to-end reproduction of ADR-345's own scaling probe**, unmodified
(`ruvector-agent-memory/examples/mincut_scaling_probe.rs`,
`RuVectorGraphAnalyzer::from_knn` + `.partition()`):

| n | baseline (pre-fix) | candidate (this fix) | speedup |
|---|---|---|---|
| 19 | 73,010ms | 69,201ms | 1.06x (noise; n<20 uses the brute-force path, untouched by this fix) |
| 50 | 77.3ms | 68.8ms | 1.12x |
| 100 | 565.3ms | 361.9ms | 1.56x |
| 200 | 2,375.3ms | 1,397.9ms | 1.70x |
| 400 | 11,420.4ms | 4,307.8ms | 2.65x |

The n=19 row reproduces ADR-345's documented ~69s brute-force outlier
almost exactly on both sides, confirming this baseline run is a faithful,
same-hardware control, not a different environment's numbers.

**Direct re-run of ADR-345's own rejected hypothesis benchmark**
(`ruvector-agent-memory/examples/mincut_gated_forgetting_bench.rs`,
`--features mincut-forget`, 84-memory corpus — the exact benchmark ADR-345
measured 1,800–2,700x slowdown on):

| Metric | Baseline (pre-fix) | Candidate (this fix) | ADR-345 gate | Verdict (unchanged) |
|---|---|---|---|---|
| Soft compaction slowdown | 2,726.1x | 1,514.8x | <= 100x | FAIL (both) |
| Hard compaction slowdown | 2,913.3x | 1,564.4x | <= 100x | FAIL (both) |
| Soft bridge-survival gap | +0.0pp | +0.0pp | >= 15pp | FAIL (both, unchanged) |
| Recall@10 delta | 0.00pp | 0.00pp | <= 2pp | PASS (both) |
| Tamper detection | 20/20 | 20/20 | 20/20 | PASS (both) |

Compaction slowdown drops ~1.8x (Soft) / ~1.9x (Hard) — real and
reproducible — but both candidates remain an order of magnitude over the
100x gate, and the independent bridge-survival-effectiveness failure
(ADR-345's other rejection ground, a property of global-minimum-cut
semantics, not of `deterministic_bfs`'s performance) is completely
unaddressed by this change, as expected: this fix does not touch which cut
the algorithm finds, only how fast it finds it.

## Consequences

- `RuVectorGraphAnalyzer::partition()`, and anything built on
  `BoundedInstance`/`DeterministicLocalKCut` (`CommunityDetector`,
  `GraphPartitioner`, the dormant `MincutGatedForgetting`), gets a real,
  correctness-preserving latency reduction for free, growing with graph
  size (1.1x at n=50 to 2.65x at n=400 end-to-end; up to 3.1x in the
  isolated worst-case primitive benchmark). No call site changes.
- `MincutGatedForgetting` stays rejected and non-default
  (`mincut-forget` feature, off by default) — this ADR closes one of
  ADR-345's two independent blockers partway, not both, and does not
  re-open the promotion question.
- The remaining gap to ADR-345's 100x gate (currently ~15-26x over) is
  structural: `search_for_cuts`'s nested budget x seed loop and
  `MinCutWrapper`'s O(log n) range instances each still multiply the
  (now cheaper) per-call cost. Closing that gap further means reducing the
  *number* of `deterministic_bfs` calls `search_for_cuts` makes (e.g.
  bounding `seed_vertices` to a sampled subset instead of every boundary
  vertex), not making each call cheaper — a different, larger change left
  for future work (see Next Research in the companion nightly doc).

## Alternatives

- **Replace the cut-search backend with `PolylogConnectivity`.** Rejected:
  that module answers connectivity (reachability / component) queries, not
  minimum-cut value or witness queries; it solves a different problem than
  the one causing this latency, as established in Context above.
- **Cap `search_for_cuts`'s seed count** (e.g., sample O(log n) boundary
  vertices instead of all of them). A real, complementary lever on the
  *count* of `deterministic_bfs` calls rather than their individual cost —
  left as future work precisely because it is a different, larger-blast-radius
  change (changes which witness `search_for_cuts` can find, not just how
  fast each search is) that deserves its own hypothesis and evidence,
  rather than being bundled into this narrowly-scoped fix.
- **Rewrite `search_for_cuts`/`MinCutWrapper` to share a single incremental
  boundary structure across seeds/budgets/instances.** Likely a larger
  further win, but a materially bigger, riskier change to the paper's
  wrapper algorithm; not attempted here.

## Implementation plan

Already implemented and merged in this change:
`crates/ruvector-mincut/src/localkcut/paper_impl.rs`
(`deterministic_bfs`, new `update_boundary_incremental`),
`crates/ruvector-mincut/examples/boundary_incremental_bench.rs` (permanent
reproducible benchmark), one new regression test.

## API shape

No public API change. `LocalKCutOracle`, `LocalKCutQuery`, `LocalKCutResult`,
`DeterministicLocalKCut::{new, with_family_generator, search}`, and
`calculate_boundary` (crate-private, used by one existing test) are
unchanged.

## Feature flags

None. This is not gated — it changes only the internal cost of an existing,
always-on code path.

## Benchmark evidence

See Evidence above and
`docs/research/nightly/2026-10-03-mincut-incremental-boundary-bfs/raw-runs.txt`
for full raw command output.

## Security

None. No change to witness contents, cut semantics, or determinism
guarantees (ADR-346's determinism tests pass unchanged, in both debug and
release profiles).

## Governance

None. Internal performance change to an existing, already-shipped
algorithm; no new consumer-facing contract, feature flag, or promotion
decision is introduced.

## Failure modes

- If a future caller relies on `deterministic_bfs`'s exact iteration count
  or timing (none currently do — it is a private method), this change
  would be observable. No such dependency exists today.
- The incremental bookkeeping assumes `graph.neighbors(v)` is stable within
  a single `search()` call (no concurrent mutation mid-call) — the same
  assumption the original from-scratch implementation already made.

## Migration

None required; drop-in internal change.

## Rollback

Revert the `deterministic_bfs`/`update_boundary_incremental` change in
`paper_impl.rs`; `calculate_boundary` and all public types are untouched,
so no other code needs to change.

## Rejection criteria

Would be rejected if: any witness's reported cut value diverged from a
from-scratch `calculate_boundary` recomputation (it does not, see
Evidence), if `partition()` determinism regressed (it does not — all of
ADR-346's determinism tests pass unchanged in release mode), or if no
measurable speedup existed (a 1.1x-3.1x speedup is measured and
reproducible).

## Open questions

- Would bounding `search_for_cuts`'s seed-vertex fan-out (this ADR's
  "Alternatives" section) combined with this fix be enough to clear
  ADR-345's 100x gate? Not measured here; flagged as next research.
- Does a similar from-scratch-recompute-in-a-loop pattern exist elsewhere
  in `ruvector-mincut` (e.g. `brute_force_min_cut`'s per-mask
  `compute_boundary` call, which is a different, inherently exponential
  code path and out of scope here)? Not audited beyond the module this ADR
  touches.
