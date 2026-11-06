# Swapping a connectivity data structure didn't fix our min-cut latency problem — and that's useful

## Problem

`ruvector-mincut` is a dynamic minimum-cut engine inside RuVector, a
Rust-native vector/graph substrate. One consumer, `ruvector-agent-memory`'s
`MincutGatedForgetting` policy, wanted to use live graph connectivity as a
signal for which memories are safe to evict during compaction — don't
evict the one "bridge" memory holding two topic clusters together. A prior
nightly (ADR-345) measured the core call this needed,
`RuVectorGraphAnalyzer::partition()`, scaling from ~77ms at 50 vertices to
~11.4 seconds at 400 — far too slow to call on every compaction pass. A
follow-up nightly fixed an unrelated non-determinism bug in the same code
path and left one specific, scoped suggestion for next time: the crate
already has an unused, recently-added implementation of a 2025
polylogarithmic-worst-case dynamic connectivity paper
(arXiv:2510.08297) sitting in-tree. Maybe wiring it in as the thing that
answers "is this graph still connected" would fix the latency.

## Hypothesis

If `BoundedInstance` — the per-lambda-range worker inside
`MinCutWrapper` that actually runs the cut search — swaps its ad hoc
BFS-based disconnection check for an incrementally-maintained
`PolylogConnectivity` (or, as a second candidate, the Euler-Tour-Tree-based
`DynamicConnectivity` already used one layer up), then repeated queries
against an unchanged graph should get meaningfully faster: at least 5x at
the largest size tested, with identical cut values and identical results
across repeated runs (determinism).

## Technical Design

The key design constraint: isolate exactly one variable. `BoundedInstance`
has a lot of surrounding machinery — a LocalKCut oracle search loop, a
brute-force fallback for small graphs, boundary-value caching, certificate
tracking for audit — none of which this experiment wanted to touch. So
instead of writing three separate near-duplicate implementations (which
risks them silently drifting apart over time), the fix was a small generic
refactor: pull the one thing being varied — "is the graph currently
connected" — behind a three-method trait (`ConnBackend`), and make
`BoundedInstance` generic over it:

```rust
pub trait ConnBackend: Send + Sync {
    fn new() -> Self where Self: Sized;
    fn insert_edge(&mut self, u: VertexId, v: VertexId);
    fn delete_edge(&mut self, u: VertexId, v: VertexId);
    fn is_connected(&self) -> bool;
}

pub struct BoundedInstanceImpl<C: ConnBackend> { /* unchanged fields + conn: C */ }

pub type BoundedInstance = BoundedInstanceImpl<BfsConn>;           // baseline, same as before
pub type PolylogBoundedInstance = BoundedInstanceImpl<PolylogConn>; // candidate A
pub type EttBoundedInstance = BoundedInstanceImpl<EttConn>;         // candidate B
```

All three variants share literally the same `search_for_cuts`,
`brute_force_min_cut`, boundary cache, and certificate code. `BoundedInstance`
itself becomes a type alias, so every existing caller is unaffected.

## Actual Implementation

- `crates/ruvector-mincut/src/instance/bounded_conn.rs`: the trait and its
  three implementations (`BfsConn`, `PolylogConn`, `EttConn`).
- `crates/ruvector-mincut/src/instance/bounded.rs`: the generic struct, with
  two one-line additions (`self.conn.insert_edge/delete_edge`) and one
  method body replaced (`is_connected` now delegates to the backend). All 7
  pre-existing tests pass unmodified.
- `crates/ruvector-mincut/examples/polylog_backend_bench.rs`: the benchmark
  harness, reusing ADR-345's own ring k-NN graph generator for direct
  comparability, measuring a "cold" first query and several "warm" repeated
  queries against an unmutated graph.

## Actual Benchmark Evidence

Measured on n in {50, 100, 200, 400}, all three variants back-to-back in
the same process:

| n | candidate A (Polylog) warm speedup | candidate B (Ett) warm speedup |
|---|---|---|
| 50  | 1.13x | 1.16x |
| 100 | 1.10x | 1.06x |
| 200 | 1.06x | 1.00x |
| 400 | 1.07x | 1.05x |

Fixed acceptance bar was 5x. Best result: 1.16x — noise-level. All three
variants agreed on cut value and partition membership at every size, and
determinism (5 identical rebuilds) held at n=50 for all three. The
hypothesis is **rejected**.

Why? Root-causing the "no speedup" result turned up something more useful
than the original question: `BoundedInstance::query()` only caches a
result when it finds a cut `ValueInRange`. An instance whose range the true
cut value exceeds (`AboveRange`) gets no caching at all — it re-runs its
full search (rebuild a graph object, then scan every integer budget value
in its lambda range, trying every seed vertex against a local-cut oracle
for each) on *every single call*, whether or not anything changed. Since
reaching the right lambda range typically takes building and discarding
through a dozen-plus smaller-range instances first, that repeated,
uncached work dominates total latency — and it dwarfs whatever the
connectivity check costs, which is why swapping it did nothing measurable.

## Limitations

- Benchmarked on one machine, single-threaded; no concurrent-mutation
  scenario.
- "Warm" repeat count was cut from a planned 1000 to 5 once it became clear
  each repeat costs close to a full search, not microseconds — so the
  spread reported (min/max across 5 runs) is a small sample.
- Determinism was independently re-checked at only one size (n=50), not
  the full swept range.
- Didn't touch n<20 (a separate, real inefficiency was found there — every
  lambda-range instance redundantly re-runs the *same* exponential
  brute-force search before giving up — but it's orthogonal to this
  experiment and was left for a future run) or n>=800 (not needed once the
  no-speedup pattern was clear at smaller sizes).

## Production Relevance

None of this unblocks `MincutGatedForgetting` by itself — that was never
the claim. What it does: rules out a plausible, low-risk fix that a future
session might otherwise re-propose and re-test from scratch, and hands off
a concretely-identified, differently-located bottleneck (the missing
`AboveRange` cache) with enough detail that the next attempt can start from
evidence instead of another guess.

## RuVector Ecosystem Implications

`ruvector-mincut`'s dynamic connectivity is the structural layer under any
future "is this agent memory / swarm / coherence domain still one
connected piece" check RuVector's longer-horizon roadmap wants — self-
healing graph memory, RVM coherence-domain gating, proof-gated mutation.
All of those need this class of query in the sub-millisecond-to-millisecond
range at the sizes that matter; this run's numbers (tens of milliseconds to
tens of seconds, depending on size) are the actual current state, not the
one a quick backend swap can fix.

## Future Direction

The clear next step, named explicitly in this run's "Next Research":
implement and benchmark caching for `AboveRange` results, the same way
`ValueInRange` results are already cached. Re-run this same three-variant
A/B/C harness afterward — it's possible the connectivity-backend choice
*does* start to matter once the larger cost is removed, which this run
can't tell either way.

## References

- ADR-345: Mincut-Gated Forgetting (the original, still-rejected,
  production use case)
- ADR-346 / 2026-09-11 nightly: Deterministic, Non-Degenerate Witness
  Partitions (the prior fix this run builds on and whose "Next Research"
  item this run tests)
- arXiv:2510.08297 — "Dynamic Connectivity with Expected Polylogarithmic
  Worst-Case Update Time" (the paper `PolylogConnectivity` implements)
- This run's ADR-353 and full nightly report, in the same repository
