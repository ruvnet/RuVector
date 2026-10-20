# ADR-353: Connectivity-Backend Swap for `BoundedInstance` — Rejected, `AboveRange` Caching Identified as the Real Target

## Status

Rejected (hypothesis falsified by measurement). No production code path
changed: `MinCutWrapper::new()`'s default factory still constructs
`BoundedInstance` (unchanged behavior). The two new variants,
`PolylogBoundedInstance` and `EttBoundedInstance`, are retained in-tree as
benchmarked, tested, correctness-equivalent reference implementations —
negative evidence, not production candidates.

## Context

ADR-345 (`docs/adr/ADR-345-mincut-gated-forgetting.md`) measured
`RuVectorGraphAnalyzer::partition()` scaling from ~77ms at n=50 to ~11.4s at
n=400 on a synthetic k-NN graph, which alone made `MincutGatedForgetting`
impractical for agent-memory compaction. The 2026-09-11 nightly
(`docs/adr/ADR-346-deterministic-mincut-witness-partition.md`) fixed a
separate non-determinism bug in the same code path and left one concrete
"Next Research" item: evaluate `connectivity::polylog::PolylogConnectivity`
— already in-tree (`crates/ruvector-mincut/src/connectivity/polylog.rs`,
implementing arXiv:2510.08297) but wired into no `ProperCutInstance` — as a
replacement for `BoundedInstance`'s ad hoc BFS-based disconnection check.

This ADR records that experiment's outcome.

## Hypothesis

```text
Given a ring k-NN graph (n in {50, 100, 200, 400}, the same generator
ADR-345's own scaling probe used),

when BoundedInstance's BFS-from-scratch is_connected() is replaced by an
incrementally-maintained PolylogConnectivity (candidate A) or
DynamicConnectivity (candidate B), with every other code path byte-for-byte
unchanged,

then a candidate's mean per-call latency on repeated query() calls against
an unmutated graph should be at least 5x lower than baseline's at the
largest size reached,

subject to: cut_value/partition membership matching baseline exactly at
every size, and cut_value staying identical across 5 independent rebuilds
at n=50.
```

## Decision

1. **Do not wire `PolylogConnectivity` or `DynamicConnectivity` into
   `BoundedInstance`'s default factory.** Measured speedup tops out at
   1.16x (noise-level), far short of the 5x bar.
2. Keep `PolylogBoundedInstance` and `EttBoundedInstance`
   (`crates/ruvector-mincut/src/instance/bounded.rs`) in-tree as tested,
   benchmarked, correctness-equivalent alternatives — not wired anywhere,
   not feature-gated (no behavior to gate; they are additive type aliases
   over a new generic `BoundedInstanceImpl<C: ConnBackend>`).
3. Refactor `BoundedInstance` from a concrete struct into
   `BoundedInstanceImpl<BfsConn>` (a type alias), isolating the
   connectivity-check behind a new `ConnBackend` trait
   (`crates/ruvector-mincut/src/instance/bounded_conn.rs`). This is a
   behavior-preserving refactor: all 7 pre-existing `bounded.rs` unit tests
   pass unmodified against the new generic form.
4. Record the actual dominant cost this experiment surfaced —
   `BoundedInstanceImpl::query()` never caches an `AboveRange` result, only
   a `ValueInRange` one, so every untried lambda-range instance re-runs its
   full `search_for_cuts()` on every call, cold or warm — as the real
   target for a future nightly (see Next Research in the companion report).

## Evidence

Full raw benchmark output and methodology are in
`docs/research/nightly/2026-10-20-polylog-mincut-connectivity-backend/README.md`
and `benchmark_output.txt` in the same directory; summarized here:

| n | warm speedup (candidate A / Polylog) | warm speedup (candidate B / Ett) | correctness |
|---|---|---|---|
| 50  | 1.13x | 1.16x | match |
| 100 | 1.10x | 1.06x | match |
| 200 | 1.06x | 1.00x | match |
| 400 | 1.07x | 1.05x | match |

All three variants agree on `cut_value` and partition membership at every
size (`all_variants_agree_with_baseline=true`). Determinism at n=50: 5/5
identical `cut_value` across independent rebuilds for all three variants.

Root cause of the "no speedup" result: `is_connected()`'s cost (O(n+m) BFS
vs. O(1) field read) is a vanishingly small fraction of `query()`'s total
latency, because `search_for_cuts()` — rebuilding a `DynamicGraph` and
running a linear budget-value scan × per-seed LocalKCut oracle calls —
re-executes in full on every call for every lambda-range instance that
hasn't yet found a `ValueInRange` cut, since only `ValueInRange` results
are cached (`best_witness`). This was not previously documented; it is now
the lead item in Next Research.

## Consequences

- No regression: `BoundedInstance`'s default behavior and all existing
  call sites (`MinCutWrapper::new()`, `RuVectorGraphAnalyzer`) are
  unaffected.
- `ruvector-agent-memory`'s `MincutGatedForgetting` (ADR-345) remains
  rejected for production use; this ADR does not change that status, and
  does not claim to have moved it closer — the connectivity-backend
  hypothesis specifically does not help it.
- A concrete, scoped follow-up (`AboveRange` caching) is now available for
  a future nightly, with a plausible path to the 2-3 orders of magnitude
  improvement `MincutGatedForgetting` would actually need, though that is
  unverified until implemented and measured.
- Two new crate-internal types (`PolylogBoundedInstance`,
  `EttBoundedInstance`) exist as dead code from a production standpoint
  (used only by the benchmark and their own unit tests) until either a
  future experiment finds them useful post-`AboveRange`-fix, or they are
  removed as unused surface in a later cleanup pass.

## Alternatives

- **Leave `BoundedInstance` as a concrete (non-generic) struct and
  duplicate it twice for the two candidates.** Rejected: ~300 lines
  duplicated twice risks the two copies silently drifting from baseline
  behavior over time, defeating the point of an isolated single-variable
  comparison. The chosen generic-trait approach keeps all three variants
  sharing the exact same `search_for_cuts`/`brute_force_min_cut`/caching
  code by construction.
- **Fix `AboveRange` caching in this same change** once it was identified
  as the likely real bottleneck. Rejected for this ADR specifically:
  changing the hypothesis after seeing benchmark results, mid-experiment,
  is exactly what the research process's rules prohibit. Deferred to a
  dedicated future nightly with its own acceptance threshold.

## Implementation Plan

Already implemented and merged as part of this change (no further rollout
needed): `bounded_conn.rs` (new), `bounded.rs` (generic refactor),
`instance/mod.rs` and `lib.rs` (re-exports), `examples/polylog_backend_bench.rs`
(new benchmark). No migration of existing callers required — `BoundedInstance`
is still the same name, same public API, same behavior.

## API Shape

```rust
pub trait ConnBackend: Send + Sync {
    fn new() -> Self where Self: Sized;
    fn insert_edge(&mut self, u: VertexId, v: VertexId);
    fn delete_edge(&mut self, u: VertexId, v: VertexId);
    fn is_connected(&self) -> bool;
}

pub struct BoundedInstanceImpl<C: ConnBackend> { /* ... */ }

pub type BoundedInstance = BoundedInstanceImpl<BfsConn>;          // unchanged default
pub type PolylogBoundedInstance = BoundedInstanceImpl<PolylogConn>; // new, not wired
pub type EttBoundedInstance = BoundedInstanceImpl<EttConn>;         // new, not wired
```

## Feature Flags

None added. No behavior change requires gating; the new types are additive
and unreferenced by any default code path.

## Benchmark Evidence

See Evidence above and the linked nightly report / `benchmark_output.txt`
for full detail, methodology, and the `cold`-phase table (which replicates
ADR-345's scaling shape, though at somewhat higher absolute latencies on
this run's hardware — the relative, same-process A/B/C comparison is the
load-bearing result, not the absolute numbers).

## Security

No security-relevant surface. Internal data-structure swap behind an
existing trait boundary; no new external inputs, no change to witness or
certificate semantics.

## Governance

Per the nightly process's promotion gate: a candidate without measurable
evidence of improvement is not promoted, regardless of novelty. Both
candidates here are novel (one is an unused-until-now, recently-added
arXiv-paper implementation) but neither is promoted, consistent with
"never promote a candidate because it is novel."

## Failure Modes

1. n<20 graphs hit a separate, pre-existing, orthogonal performance issue
   (`brute_force_min_cut()` recomputed identically by every lambda-range
   instance) severe enough that this run excluded n=19 from its sweep
   rather than investigate it as part of this ADR's scope. See Next
   Research in the companion nightly report.
2. The actual dominant cost (`AboveRange` never cached) means
   `MinCutWrapper::query()`'s latency does not amortize across repeated
   calls on an unmutated graph at all today — a materially worse picture
   than "connectivity check is slightly suboptimal," and the reason this
   hypothesis's rejection was so unambiguous (1.0x-1.16x, not even a modest
   partial win).

## Migration

None required; no default behavior changed.

## Rollback

N/A — nothing was wired in to roll back. If `PolylogBoundedInstance`/
`EttBoundedInstance` are judged not worth keeping as reference code in a
future cleanup, they can be deleted along with `bounded_conn.rs` without
touching any other call site.

## Rejection Criteria

Already met: the hypothesis predicted >=5x improvement; measured result is
1.0x-1.16x across all four tested sizes, with correctness and determinism
otherwise intact. Rejected as stated, per the acceptance threshold fixed
before this run's benchmark began.

## Open Questions

1. Does an `AboveRange` cache (Next Research item 2 in the companion
   report) actually deliver the 2-3 orders of magnitude `MincutGatedForgetting`
   would need, or does the bounded-instance architecture itself need
   reconsideration? Unresolved; the subject of a future nightly.
2. Once `AboveRange` caching exists, does the connectivity-backend choice
   become relevant after all (i.e., does it only matter once
   `search_for_cuts()`'s repeated cost is removed)? Unresolved; worth
   re-running this same three-variant harness after that fix lands.
3. Is `find_bridges()`'s per-edge `MinCutWrapper` reconstruction
   (`integration/mod.rs`) a third, independent scaling problem in the same
   crate? Observed in passing, not investigated this run.
