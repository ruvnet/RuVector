# ADR-340: Local-Min-Cut-Guided HNSW Deletion Repair

## Status

Proposed. Experimental crate (`ruvector-mincut-repair`), not wired into the
default query path of any production index. **Recommendation: reject for
production; keep as a documented negative result.**

## Context

`ruvector-hnsw-repair` (an earlier nightly, `2026-06-18-hnsw-delete-repair`)
ships three HNSW deletion strategies with a fixed cost/recall trade-off:
`TombstoneOnly` (O(1), recall degrades), `BatchRepair` (amortised), and
`EagerRepair` (O(deg · live_count) per delete — a full scan of every live
node at every affected level — best recall). All three apply the *same*
policy to *every* deleted node, regardless of whether that specific node's
removal actually threatens local connectivity. A node buried in a dense,
well-connected region of the graph gets the same expensive full-graph
treatment as a node sitting at a structural bottleneck.

`ruvector-mincut` separately implements a real, recent local minimum-cut
algorithm — `LocalKCut`, from "Deterministic and Exact Fully-dynamic Minimum
Cut of Superpolylogarithmic Size" (arXiv:2510.08297, Dec 2024) — whose
stated design goal is exactly this kind of triage: find a cut of size `<= k`
near a vertex in time bounded by `k` and that vertex's degree, not the whole
graph. The crate's own manifest description calls this "self-healing
networks" as a use case. HNSW deletion repair is a direct match on paper:
use `LocalKCut::find_cut` to decide, per deleted node, whether the expensive
`EagerRepair` path is warranted, and tombstone everything else.

## Hypothesis

```text
Given an HNSW graph (ruvector-hnsw-repair's own implementation) of 5,000
vectors at dim 64, built with the same parameters as that crate's own
benchmark,

when 20% of vectors are deleted in a fixed, deterministic order and
`LocalCutGuidedRepair` decides per-node whether to eagerly repair (via
`LocalKCut::find_cut` on a shadow graph mirroring HNSW level-0 edges,
cut bound k=2) or tombstone-only,

then (a) recall@10 lands within 1.0 percentage point (absolute) of
EagerRepair's recall@10, and (b) LocalCutGuidedRepair's total
repaired-edge count is <= 60% of EagerRepair's,

subject to (c) the find_cut + shadow-graph bookkeeping overhead staying
under 25% of LocalCutGuidedRepair's own total delete wall-clock time —
the connectivity-guidance layer must not itself become the bottleneck it
is trying to avoid.
```

## Decision

Add `crates/ruvector-mincut-repair`, a small crate providing
`LocalCutGuidedRepair: DeletionStrategy` that:

1. On construction, mirrors the HNSW graph's **level-0** adjacency (only —
   see Alternatives Considered) into a `ruvector_mincut::DynamicGraph`.
2. Builds one `ruvector_mincut::LocalKCut` finder (`k=2`) over that shadow
   graph.
3. On each `delete(id)`: queries `find_cut(id)`. If a local cut of size
   `<= 2` is found, calls `ruvector_hnsw_repair::repair_one` — the exact
   same reconnection routine `EagerRepair` uses, exposed as `pub` for reuse
   rather than re-implemented (avoiding behavioural drift between the two).
   Otherwise, tombstones only.
4. Removes the deleted vertex from the shadow graph afterward so subsequent
   queries stay in sync.

## Evidence

Measured via `cargo run --release -p ruvector-mincut-repair --bin
benchmark` (n=5,000, dim=64, 100 queries, k=10, ef_search=50, cut_k=2) —
the same dataset generator, size, and query count as
`ruvector-hnsw-repair`'s own benchmark, so the three baselines are not
re-tuned in this crate's favour. `n_delete` is 3, not the 1,000 (20%) used
elsewhere in this repo's HNSW-repair benchmarks — see Evidence below for
why. Full raw output is in the nightly research report; the
acceptance-relevant summary from the actual run:

```text
TombstoneOnly: delete=0.00ms  recall@10=0.9140  degradation=+0.0000
BatchRepair(50): delete=0.80ms  recall@10=0.9140  degradation=+0.0000
EagerRepair: delete=0.78ms  recall@10=0.9140  degradation=+0.0000  repaired_edges=176
LocalCutGuided: delete=318024.78ms  recall@10=0.9140  degradation=+0.0000  repaired_edges=0

  delete(0):    fragile=false  took=158.024483433s
  delete(1666): fragile=false  took=135.243508827s
  delete(3332): fragile=false  took=24.756610171s

Acceptance criteria:
  1. recall gap (Eager - MincutGuided) <= 1.0pp     : +0.00pp   [PASS]
  2. repaired_edges ratio (Mincut/Eager) <= 0.60    : 0.000     [PASS]
  3. bookkeeping overhead fraction <= 0.25          : 1.000     [FAIL]

ACCEPTANCE: REJECT — at least one criterion failed.
```

**Criteria 1 and 2 passing here is not meaningful validation.** At
`n_delete=3` (0.06% of the index), all three deleted nodes were judged
"safe" (no local cut found), so `LocalCutGuidedRepair` performed zero
repairs against `EagerRepair`'s 176 — the 0.60 edge-ratio threshold is met
trivially, by inaction, not by successfully triaging fragile nodes. The
recall parity is likewise uninformative at this sample size: 100 queries
against a 5,000-node index with 3 deletions is not statistically powered
to detect a recall difference this small either way. Criterion 3 is the
only criterion this run actually speaks to with statistical weight, and it
fails by four orders of magnitude (100% observed vs. a 25% budget) — this
alone determines the REJECT verdict regardless of the other two.

A larger, harder cap on tractability: an earlier run at `n_delete=1000`
(the original 20% target) did not complete the `LocalCutGuidedRepair`
phase in 10+ minutes and was killed; an intermediate run at
`n_delete=12` completed exactly one `find_cut` call — **163.49 seconds**
— before being killed by a 200s cap. All three runs (interrupted 1000,
interrupted 12, completed 3) agree on the same order of magnitude, so the
small `n_delete` in the final run is a tractability choice, not a
favourable-sample-size choice: a larger sample would not have changed
criterion 3's verdict, only taken hours to confirm it again.

A standalone diagnostic (`probe_find_cut_cost`, `#[ignore]`d, in
`src/lib.rs`) isolates `find_cut` cost on a *sparser* fixture graph
(m=4, m0=8, vs. the benchmark's m=16, m0=32): five calls at 222-418ms
each. That the same operation costs sub-second on a low-degree graph and
100+ seconds at the benchmark's real construction density is itself part
of the evidence — see Root Cause.

**Root cause.** `LocalKCut::find_cut`'s complexity bound (`O(k^{O(1)} ·
deg(v))`) assumes a bounded-degree, low-expansion graph. Two compounding
factors defeat it here:

1. **HNSW is small-world by construction.** `compute_radius(k=2)` picks a
   BFS depth of 2, but on a small-world graph a depth-2 neighbourhood is
   not small — at the benchmark's construction density (m0=32) it can
   already touch a large share of all 5,000 nodes.
2. **`check_cut` is itself not O(1) per crossing edge.**
   `crates/ruvector-mincut/src/localkcut/mod.rs:368`
   (`LocalKCut::check_cut`) calls `self.graph.edges()` — which
   materialises a fresh `Vec` of *every* edge in the graph, an O(m)
   allocation — and then linearly scans it (`.iter().find(|e| e.id ==
   edge_id)`) to look up a single edge, once per crossing edge examined.
   `DynamicGraph` already provides an O(1)-average lookup for exactly this
   (`get_edge(u, v)`, backed by the `edge_index: DashMap<(VertexId,
   VertexId), EdgeId>` the struct already maintains), but `check_cut` does
   not use it. `find_cut` calls `check_cut` once per (depth, colour-mask)
   combination that yields a non-trivial reachable set — up to
   `radius * 15` times per call — so a single `find_cut` call's cost is
   closer to O(cut_boundary_size · radius · 15 · m) than to the documented
   bound. This is a fixable implementation defect in `ruvector-mincut`,
   not an inherent property of the published algorithm; it is not fixed by
   this ADR (see Implementation Plan) because verifying a correctness-
   sensitive change in a 55K-line crate is out of this nightly's scope.

## Consequences

**Positive:**
- Confirms, with a real measurement rather than an assumption, that
  `LocalKCut` in its current form is not a drop-in cost-control layer for
  small-world graphs like HNSW. This is useful negative evidence: it rules
  out an entire family of "add a cheap local-connectivity check before the
  expensive repair" designs on HNSW specifically, saving a future nightly
  run from re-discovering the same result.
- `repair_one` is now `pub` in `ruvector-hnsw-repair`, a small additive
  change that lets any future repair-strategy experiment reuse the exact
  eager-reconnection logic instead of re-implementing it.
- The `LocalCutGuidedRepair` strategy and its tests are still correct
  (recall does land close to `EagerRepair` when it does — see evidence);
  the rejection is about cost, not correctness.

**Negative / costs:**
- `find_cut`'s measured per-call latency (24.8-158.0 seconds at the
  benchmark's real construction density, see Evidence) makes
  `LocalCutGuidedRepair` roughly five to six orders of magnitude slower
  in total wall-clock than plain `EagerRepair`'s sub-millisecond deletes
  at this scale, inverting the intended trade-off entirely.
- The shadow `DynamicGraph` doubles the deletion path's bookkeeping (insert
  once, then one `remove_vertex` call per delete) for no net benefit given
  the above.

## Alternatives Considered

- **Mirror all HNSW levels, not just level 0.** Rejected for this
  experiment: level 0 already reproduces the small-world density problem;
  adding sparser upper levels would not change the conclusion and would
  cost more to maintain. Noted as a non-fix, not left as unexplored future
  work.
- **Use a larger `k` (wider cut search) to catch more real bottlenecks.**
  Rejected: `compute_radius(k)` grows with `k`, so a larger `k` makes each
  `find_cut` call *more* expensive, not less — the wrong direction for a
  crate whose entire premise is a cheap pre-check.
- **Use `DynamicConnectivity` (full-graph, documented `O(m·α(n))` rebuild
  per edge deletion) instead of `LocalKCut`.** Not benchmarked here: its
  rebuild-per-delete-edge cost is a worse asymptotic starting point than
  `LocalKCut`'s locality promise, and the latter's real-world cost already
  falsifies the hypothesis without needing a second, more expensive
  baseline to also fail.

## Implementation Plan

1. (This ADR) Land the experimental crate, benchmark, tests, and diagnostic
   probe — unintegrated, not a dependency of any other crate.
2. No further integration of `LocalCutGuidedRepair` is planned. This ADR's
   recommendation is rejection; see Rejection Criteria (already
   triggered).
3. Separately from this crate: `LocalKCut::check_cut`'s O(m) edge lookup
   (`crates/ruvector-mincut/src/localkcut/mod.rs:368`, see Root Cause) is
   a genuine implementation defect independent of this ADR's overall
   rejection — fixing it (replace `self.graph.edges().iter().find(...)`
   with `self.graph.get_edge(u, v)`) would bring `find_cut` closer to its
   documented complexity bound and is worth a dedicated nightly with its
   own correctness verification, not a same-day patch bundled into a
   rejected design's ADR. Tracked as
   [issue #942](https://github.com/ruvnet/RuVector/issues/942), which
   also states explicitly (per this ADR's own recommendation) that fixing
   the defect must not be used to revive or claim acceptance of this
   ADR's rejected design without a separate, preregistered experiment.
4. If a future nightly wants to revisit graph-topology-aware repair
   triage after that fix lands, the honest next step is still a cost
   model native to small-world graphs (degree- or density-thresholded
   triage, not a bounded-BFS local cut) — the BFS-expansion factor (Root
   Cause, point 1) is architectural and would survive the `check_cut` fix.

## API Shape

```rust
let strat = LocalCutGuidedRepair::new(&graph, /* cut_k = */ 2);
for &id in &delete_ids {
    strat.delete(&mut graph, id); // DeletionStrategy trait
}
let stats = strat.stats(); // fragile_count, safe_count, bookkeeping_ns
```

## Feature Flags

None. The crate is opt-in by virtue of not being a dependency of any other
crate.

## Benchmark Evidence

See
`docs/research/nightly/2026-08-28-mincut-guided-hnsw-repair/README.md` for
full methodology and raw `cargo run --release` output.

## Security

No new `unsafe` code. No external network calls, no new I/O. The shadow
graph holds only vector IDs (`u64`), never vector contents.

## Governance

N/A — read/repair-path performance experiment, no authorization or
provenance surface.

## Failure Modes

- If `id >= graph.deleted.len()`, `delete` is a no-op (`DeleteResult`
  default), matching the other three strategies' bounds behaviour.
- If the shadow graph and the live `HnswGraph` ever desync (e.g. a caller
  mutates `graph.layers` directly outside this strategy), `find_cut` may
  return a stale answer; this is only safe when `LocalCutGuidedRepair` is
  the sole mutator of a given graph instance, which the tests assume and
  which this ADR does not extend beyond.

## Migration

N/A — new, unintegrated crate.

## Rollback

Delete `crates/ruvector-mincut-repair` and its workspace member entry.
Revert `repair_one`'s visibility in `ruvector-hnsw-repair` to private if no
other crate has taken a dependency on it by the time of rollback (at the
time of this ADR, none does).

## Rejection Criteria

This direction is rejected for production promotion because, on
first measurement:

- Criterion 3 (bookkeeping overhead <= 25% of delete wall-clock) fails by
  a wide margin — see Evidence. This alone is disqualifying regardless of
  the recall and edge-count criteria's outcome, since the whole point of
  the design was to be *cheaper* than blanket `EagerRepair`.

## Open Questions

- Is there a cheap, *degree-based* (not BFS-based) local signal — e.g.
  "this node's level-0 degree is below some percentile" — that predicts
  fragility on small-world graphs without paying for a bounded BFS? That
  would be a materially different design, not a parameter tweak of this
  one.
- Does `LocalKCut`'s cost profile look different on graph topologies
  `ruvector-mincut`'s own test suite targets (bounded-degree, expander-like
  graphs)? This ADR makes no claim there — only about HNSW specifically.
- After a `check_cut` fix (Implementation Plan, item 3) removes the O(m)
  edge-lookup defect, how much of the measured 100+ second cost remains?
  This ADR does not decompose the observed cost between the two
  compounding factors in Root Cause — only that both are real and at
  least one (the BFS-expansion factor) is architectural, not a bug.
