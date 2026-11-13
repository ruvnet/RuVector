# Closing out a two-ADR-old latency bug in a dynamic min-cut engine — and confirming it still isn't enough

## Problem

`ruvector-mincut` is RuVector's dynamic minimum-cut engine, shared by
18+ in-tree crates (agent memory, graph condensation, coherence-gated
kernels, an MCP brain server). Two prior research runs — ADR-345 and
ADR-346 — found that its `MinCutWrapper::query()` cold-start path was both
*slow* (up to 11.4 seconds on a 400-vertex graph, with a 69-second outlier
on a 19-vertex one) and, separately, *wrong* some of the time (a 50%
degenerate-result rate on repeated identical queries, from unstable
hash-map iteration order). ADR-346 fixed the correctness bug. The latency
bug got a partial, untracked fix on 2026-10-03 — but nobody had gone back
and checked whether that fix actually resolved what ADR-345 originally
measured.

## Hypothesis

Re-run ADR-345/346's own benchmarks against current code. If the latency
is meaningfully better, root-cause whatever's left, fix it if it's safe to
fix, and check honestly whether the combined improvement is enough to
reconsider the one production feature (`MincutGatedForgetting`, an
agent-memory compaction policy) that ADR-345 rejected partly because of
this exact latency problem.

## What the re-measurement found

| | ADR-345 (original) | This run (before new changes) |
|---|---|---|
| Determinism (50 trials) | 15/30 degenerate (50%) | 0/50 — fixed, holds |
| Latency, n=400 | ~11.4s | 4.45s (**2.6x better**) |
| Latency, n=19 (brute-force outlier) | "outlier 69s" | **73.9s — unchanged** |
| Real 84-memory corpus, compaction slowdown | 1800–2700x | 1864x / 2128x — **unchanged** |

The 2026-10-03 fix helped the n&ge;20 search path, where it touched the
code. It did nothing for n<20 (a separate, brute-force code path) or for
the actual 84-memory benchmark corpus — so the thing ADR-345 actually
rejected was, three ADRs later, still exactly as slow.

## Root cause

`MinCutWrapper` manages 100 candidate "range" instances, each responsible
for a slice of possible min-cut values, with ranges computed as
`floor(1.2^i)..floor(1.2^(i+1))`. A fresh query always scans from instance
0 upward until one reports the right range. Reading that scan loop
revealed two things:

1. For small `i`, the geometric formula produces **literal duplicate
   ranges** — instances 0, 1, and 2 are all `(1, 1)`; 4 and 5 are both
   `(2, 2)`. The scan re-instantiates and re-queries every one of them from
   scratch, even though — thanks to the determinism fix already in place —
   querying the same range against the same graph twice is guaranteed to
   give the same answer.
2. Every newly-instantiated instance separately re-collects and re-sorts
   the *entire* edge list, instead of doing it once per scan.

Neither is what the 2026-10-03 fix touched. Both are pure waste: provably
so, not probabilistically so, once you know the determinism guarantee
holds.

## Fix

Two small, additive, default-off changes to `ruvector-mincut`:

```rust
pub enum ScanOptLevel { Baseline, DedupRanges, DedupRangesAndHoistEdges }

impl MinCutWrapper {
    pub fn with_scan_opt_level(self, level: ScanOptLevel) -> Self { ... }
}
```

`DedupRanges` skips re-instantiating a never-yet-queried instance whose
range is identical to the one just scanned (which was already determined
`AboveRange`). `DedupRangesAndHoistEdges` additionally hoists the
edge-list collection out of the per-instance loop. Nothing about this is
an approximation — it's provably result-preserving given the existing
determinism guarantee, which a new test (`scan_opt_equivalence`, four
topologies spanning both the brute-force and search code paths) checks
directly, alongside an in-benchmark assertion across every size and the
real workload. Every default stays exactly as it was; nothing changes
unless a caller opts in.

## Results

| | Baseline | Fixed | Change |
|---|---|---|---|
| Ring graph, n=19 | 74010.6ms | 57868.7ms | −22% |
| Ring graph, n=400 | 4454.6ms | 3963.0ms | −11% |
| Instances materialized | 16 | 13 | the 3 genuine duplicates |
| Real corpus, Soft compaction slowdown | 1864.0x | 1291.8x | −31% |
| Real corpus, Hard compaction slowdown | 2127.9x | 1151.0x | −46% |
| Cut value, every case above | — | — | **identical** |

522 pre-existing `ruvector-mincut` tests and 3 `ruvector-agent-memory`
`graph_forget` tests stayed green throughout; two in-workspace dependent
crates smoke-checked clean.

## Was it enough?

No. The acceptance gate ADR-345 set was a compaction slowdown of at most
100x versus a plain scalar-scoring baseline. Even with the fix, the real
corpus comes in at 1151x–1292x — a real, meaningful, nearly-halved
improvement, and still off by an order of magnitude. `MincutGatedForgetting`
remains correctly unpromoted. That's not a failure of this run: a
falsified "maybe the fix was enough" hypothesis, backed by real numbers on
both a synthetic sweep and the exact original corpus, is exactly the kind
of result the evidence trail is supposed to accumulate — the next person
who's tempted to re-measure this won't have to start from zero, and they'll
know precisely which two mechanisms (per-instance brute force and
per-search seed×budget fan-out) are still unaddressed.

## What was explicitly rejected

The prior nightly run suggested evaluating `PolylogConnectivity` — a
polylogarithmic *dynamic connectivity* structure already in the crate — as
a faster backend. It isn't one: connectivity ("are these two vertices in
the same component") and minimum cut ("what's the cheapest way to split
the graph, and where") are different questions, and nothing in
`PolylogConnectivity`'s API answers the second. That's recorded here so
nobody re-proposes it without rediscovering why it doesn't fit.

## Limitations

This fixes redundancy *across* the 100-instance scan. It does not touch
the seed×budget fan-out inside a single instance's own search (the
dominant cost for graphs of 20+ vertices) or the O(2^n) brute-force cost
for graphs under 20 vertices — both real, both documented, both left for
whoever picks this up next.

## Production relevance

Any caller of `ruvector-mincut`'s `MinCutWrapper` or `RuVectorGraphAnalyzer`
gets a real, free, opt-in latency win today by adding one builder call.
Nobody gets it by default, and nobody's behavior changes without asking
for it.

## References

- ADR-345 (`docs/adr/ADR-345-mincut-gated-forgetting.md`)
- ADR-346 (`docs/adr/ADR-346-deterministic-mincut-witness-partition.md`)
- ADR-361 (`docs/adr/ADR-361-mincut-wrapper-scan-dedup.md`) — this run
- Full methodology and raw benchmark output:
  `docs/research/nightly/2026-11-13_bounded_instance_scan_dedup/README.md`
