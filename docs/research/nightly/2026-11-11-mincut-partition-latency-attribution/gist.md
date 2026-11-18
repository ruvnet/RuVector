# Why "use a faster connectivity structure" didn't fix our min-cut latency — and what actually did matter

## Problem

RuVector ships `ruvector-mincut`, a dynamic minimum-cut engine. One of its
integration layers, `RuVectorGraphAnalyzer::partition()`, is used (among
other things) as the structural signal behind an agent-memory eviction
policy we tried and rejected (`ruvector-agent-memory`'s
`MincutGatedForgetting`, ADR-345). The rejection's root cause was
performance: on a synthetic k-NN graph, `partition()` took ~77ms at 50
vertices and ~11.4 seconds at 400. A later investigation fixed an unrelated
non-determinism bug in the same code path, and in passing proposed a fix
for the latency problem too: swap the connectivity-check backend inside
the relevant algorithm (`BoundedInstance`) for a newer, asymptotically
better structure (`PolylogConnectivity`, implementing a 2025 paper on
polylogarithmic worst-case dynamic connectivity) that was already sitting
unused in the codebase.

That proposal sat as an open "Next Research" item for one more nightly
cycle. This write-up is what happened when we actually measured it instead
of leaving it as a plausible-sounding idea.

## Hypothesis

If `BoundedInstance`'s connectivity checks are where `partition()`'s time
goes, then replacing the naive BFS it currently uses with
`DynamicConnectivity` (an Euler-tour-tree structure already used elsewhere
in the same crate) or `PolylogConnectivity` (the newer, unused one) should
measurably shrink `partition()`'s wall-clock time at the sizes where it
currently blows up.

## What we built

Three small, additive changes to `ruvector-mincut`, none of which change
any algorithm's output:

1. `src/profiling.rs` — a handful of atomic counters, feature-gated behind
   `profile-counters` (off by default, compiles to nothing when unused).
2. Three call sites instrumented: the one whole-graph connectivity check
   inside `BoundedInstance`, the oracle-search call inside its
   local-min-cut search loop, and the per-instance loop in the wrapper
   that drives it.
3. Two benchmark examples: one that re-runs the exact k-NN corpus from the
   original scaling investigation with the new counters attached, and one
   standalone three-way bench that builds each of the three connectivity
   structures from scratch and times a single query — because that is the
   only way the target code *could* use them, since it holds no persistent
   connectivity state today.

## What we measured

Running the attribution bench on the same ring k-NN corpus (`k=8`, n in
{19, 50, 100, 200, 400}) used by the original investigation:

```text
n      partition_ms oracle_calls conn_calls conn_pct
50        69.195         1501         16     0.259%
100      364.540         3001         16     0.144%
200     1360.473         6001         16     0.050%
400     4454.035        12001         16     0.028%
```

The connectivity check runs at most 16 times per `partition()` call
(matches the number of internal range-bounded sub-instances the wrapper
tries before finding the answer) and costs **under 0.26% of total wall
time at every size tested.** There is no room here for a backend swap to
explain a multi-second `partition()` call, let alone fix it.

We checked the other half of the hypothesis anyway — what if the backend
swap were fast enough to matter regardless:

```text
n     baseline_us  euler_tour_us  polylog_us
50          24.38         244.27      479.11
400        222.07      11274.93     2424.27
800        371.70      43753.29     5369.59
```

Both candidate structures are *slower* than the naive BFS they'd replace —
10x to 117x slower, scaling worse than the baseline, not better. This
isn't a bug in either structure. `DynamicConnectivity` and
`PolylogConnectivity` are built to amortize their setup cost across many
incremental updates and repeated queries; used once, cold, they pay all of
that overhead and collect none of the payoff. The target call site invokes
`is_connected()` once per sub-instance and moves on — exactly the usage
pattern these structures are not for.

Multiplying either backend's single-call cost by the 16 calls
`partition()` actually makes would *add* 38ms (`PolylogConnectivity`) to
180ms (`DynamicConnectivity`) of wall time to a call that currently takes
4,454ms. Implementing the proposed fix would have made things worse.

## Where the time actually goes

The instrumentation that falsified the connectivity-backend hypothesis
also pointed at the real cost driver almost for free: the local min-cut
oracle (`DeterministicLocalKCut::search`) is called 1,501 times at n=50
and 12,001 times at n=400 — count scales linearly with n. But
`partition()`'s wall time scales roughly quadratically over the same
range (69ms -> 4,454ms, a 64x increase for an 8x increase in n). Call
count alone is an 8x factor; the rest comes from each individual
`oracle.search()` call getting more expensive as the graph grows. That's
a different, better-scoped question for a future investigation — and now
there's instrumentation in place to answer it directly instead of guessing
again.

## Why this is worth publishing

Two nightly cycles had passed with a specific, concrete, plausible
performance fix sitting unverified in a "Next Research" list. It is easy
for a fix like that to get carried forward from report to report, gaining
credibility just by surviving unchallenged, until someone implements it
and discovers — expensively, in production code this time — that it
doesn't help. Measuring it early, cheaply, with disposable benchmark code
and zero-cost-when-disabled instrumentation, is the whole point of keeping
a research lineage: a rejected idea with real numbers attached is strictly
more valuable to the next person than an untested idea with none.

## What this does not claim

This is not a performance fix. `partition()` is still slow at n=400. What
changed is that we now know, with measured evidence, where it *isn't*
slow (the connectivity check) and have a sharper, correctly-scoped
hypothesis about where it *is* (oracle call volume and per-call cost) for
whoever picks this up next.

## References

- Full methodology, raw benchmark output, and section-by-section writeup:
  `docs/research/nightly/2026-11-11-mincut-partition-latency-attribution/README.md`
- `docs/adr/ADR-345-mincut-gated-forgetting.md`
- `docs/adr/ADR-359-mincut-partition-latency-attribution.md`
- Source: `crates/ruvector-mincut/src/profiling.rs`,
  `crates/ruvector-mincut/examples/profile_partition_scaling.rs`,
  `crates/ruvector-mincut/examples/connectivity_backend_bench.rs`
