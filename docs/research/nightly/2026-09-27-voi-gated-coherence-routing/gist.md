# A router that failed at its job — and found a real bug on the way out

## Problem

`ruvector-coherence-hnsw` prunes HNSW-style beam search by checking whether
a candidate node lies roughly "toward" the query from the search's entry
point, skipping neighbor expansion when it doesn't. It was benchmarked and
accepted with one property held constant: the entry point was always far
from the query, simulating an HNSW layer-0 search that just descended
through upper layers. That's a reasonable default test, but it leaves an
untested case: what happens when the entry point is *already* close to the
query? Plenty of real workloads have that case — a shared or slowly
drifting entry point reused across many similar queries (a session
embedding, a warm-started search).

## Hypothesis

Coherence gating costs a per-candidate dot product to decide whether to
prune. Near the entry point, there's little graph left to prune before the
beam converges anyway, so that cost buys nothing — while far from the
entry, real pruning happens and the gate should win on speed. If that's
right, a per-query router that picks baseline search near the entry and
gated search far from it should approximate "best policy per query" for
free, using a signal — entry-to-query distance — that beam search already
computes as its first step.

## What we built

`ruvector-voi-router`, a small Rust crate: `VoiRoutedSearch` wraps
`ruvector-coherence-hnsw`'s existing `BaselineSearch` and
`CoherenceGatedSearch` and picks one per query based on a distance
threshold. The threshold is calibrated once, from a query set disjoint from
the one used to evaluate the result — no gold answers leak into the
routing decision.

## What actually happened

The hypothesis is wrong, cleanly. On the evaluation set (400 queries,
disjoint seed from calibration), always-on gating is *faster than baseline
across the entire distance range measured* — not just far from the entry.
There is no interior threshold where routing beats picking one fixed policy
for the whole workload; a bounded 3-generation calibration search
confirmed this by degenerating toward the extremes rather than finding a
sweet spot, and the router's mean latency exceeded always-on gating's in
every one of four independent runs (72.6–77.3µs for always-gated vs.
76.1–82.2µs for the router). **Reject.**

But building the per-group breakdown to test that hypothesis surfaced
something we weren't looking for: on the ~11% of evaluation queries whose
true nearest neighbor sits in the entry point's own cluster, the accepted
gate's recall **collapses from ~93% to 73.2%** — a 21-point drop that the
aggregate metric (90.1% overall) completely hides, and that the original
acceptance benchmark could never have caught, because it never varied
entry-to-query distance within a single run. It reproduced identically
across 6 runs (deterministic — this is not timing noise), and the search
terminates suspiciously fast when it happens (1–3 microseconds, consistent
with beam search's early-stop condition firing after only a couple of
pops). Our best mechanistic explanation: the gate's "traversal coherence"
is a cosine similarity between two displacement vectors from the entry
point; near the entry, those displacements are small, and their direction
becomes dominated by local sampling noise rather than genuine "toward vs.
away from the query" structure — so the gate starts pruning branches almost
arbitrarily instead of usefully.

The router we built to test the (rejected) latency hypothesis happens to
fully repair this: routing near-entry queries to baseline recovers the full
94.3% recall on that subgroup, for the cost of one already-computed
distance comparison.

## Limitations

Synthetic clustered data only (2000 vectors, D=32); the near-entry subgroup
is small (44 of 400 queries); the calibration procedure itself turned out
to be sensitive to timing noise (the promoted threshold ranged from the
17th to the 95th percentile across identical-seed reruns, though the
reject verdict didn't change); and only one gate threshold (0.50, the
crate's own example default) was tested.

## Production relevance

Nothing shipped changes tonight. But if you're running
`ruvector-coherence-hnsw`'s coherence gate against a workload where any
queries land near your search's entry point, this is worth checking before
trusting the aggregate recall number — a 21-point subgroup collapse can
sit invisibly inside a 90%+ overall figure. `ruvector-voi-router` is now in
the tree as a small, tested, working mitigation if you need one, though we
recommend validating the collapse on real (non-synthetic) data first.

## RuVector ecosystem implications

This connects `ruvector-coherence-hnsw` (accepted vector search primitive),
coherence scoring (the mechanism under test), and ADR-331's proposed
cost-aware/value-of-information routing pattern — the first concrete
experiment tying that framing to a specific RuVector index mechanism. It's
also a small, literal instance of the "route cheap vs. expensive" shape
that shows up at every other layer of this ecosystem, including in
CLAUDE.md's own 3-tier model-routing philosophy — and a useful negative
data point for that larger pattern: a free proxy signal only helps if the
trade-off it's routing over actually reverses direction across the signal's
range, and that has to be checked, not assumed.

## Future direction

Re-run `ruvector-coherence-hnsw`'s own acceptance benchmark with a query
mix spanning near-entry to far-entry distances, to decide whether the
gate's default threshold or its near-degenerate-direction handling needs
hardening independent of any router. Check whether `AdaptiveCoherenceSearch`
(not evaluated here) has the same blind spot. Full detail, raw benchmark
output for four independent runs, and the formal ADR are linked below.

## References

- Full research report: [`README.md`](./README.md)
- Architecture decision record: [ADR-350](../../adr/ADR-350-voi-gated-coherence-routing.md)
- Raw benchmark output (4 independent runs): [`raw-runs.txt`](./raw-runs.txt)
- Source: `crates/ruvector-voi-router/`
- Prior art this run builds on: `ruvector-coherence-hnsw` (2026-06-16
  nightly, accepted); ADR-331 (Pandora pattern)
