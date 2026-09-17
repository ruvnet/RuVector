# Fixing a 1,800x Slowdown by Reading the Library's Own Source Code

## Problem

A previous experiment (`MincutGatedForgetting`, ADR-345) tried to give an
agent-memory compaction policy a structural signal: before evicting
low-scoring memories, find which of them sit on the "bridge" connecting two
otherwise-disjoint topic clusters, and protect those specifically. The
signal came from an existing dynamic minimum-cut engine
(`ruvector-mincut`), used through its own purpose-built convenience layer,
`RuVectorGraphAnalyzer`.

The experiment was rejected on two counts: the structural signal made no
measurable difference to which memories survived compaction, and computing
it was 1,800-2,700x slower than the scalar baseline it was supposed to
improve on — slow enough that the corpus size had to be shrunk from a
planned ~2,000 memories down to 84 just to keep the benchmark runnable.

The rejection report did the right thing and left a specific, falsifiable
next step: `RuVectorGraphAnalyzer::partition()` was the slow entry point,
but `ruvector-mincut` has a second one, `DynamicMinCut`, that the
experiment never tried. Maybe the slowness was a property of the
convenience wrapper, not of computing a minimum cut at all.

## Hypothesis

```text
Given the same 84-memory corpus and the same MincutGatedForgetting policy,

when boundary detection is computed with DynamicMinCut (via MinCutBuilder)
instead of RuVectorGraphAnalyzer,

then compaction wall-clock should drop under the 100x-slowdown gate the
original attempt failed by 16-27x,

subject to: bridge-survival and recall numbers must exactly match the
original backend's — a real behavior difference between backends on
identical input would be a bug, not a result.
```

## Technical Design

Reading `ruvector-mincut`'s own source turns up why the two entry points
have such different performance profiles:

- `RuVectorGraphAnalyzer` wraps `MinCutWrapper`, an implementation of a
  paper algorithm that maintains O(log range) geometrically-scaled
  "bounded-range instances." On every fresh call, each instance that gets
  touched replays the graph's *entire* edge set into itself from scratch.
  For a corpus rebuilt fresh on every compaction pass (as this policy
  does), that's the full edge-replay cost paid repeatedly, once per
  instance, once per call.
- `DynamicMinCut` is architecturally simpler: one sparse,
  Stoer-Wagner-style exact global minimum-cut solve, implemented with a
  max-adjacency heap over O(n + m) storage. One solve, one call, no
  instance replay.

The fix is additive: a new `MincutBackend` enum (`Wrapper` — unchanged
default, `Direct` — the new path) on the existing policy struct, with both
backends building the exact same k-NN graph through one shared helper
function, so the *only* variable under test is which minimum-cut API
computes the answer.

## Implementation

```rust
pub enum MincutBackend { Wrapper, Direct }

impl MincutGatedForgetting {
    pub fn with_backend(self, backend: MincutBackend) -> Self { /* .. */ }
}
```

`Direct`'s path builds a `ruvector_mincut::DynamicMinCut` via
`MinCutBuilder::new().with_edges(edges).build()`, and maps a
disconnected-graph result (`min_cut_value() <= 0.0`) to the same
"no signal" empty partition `Wrapper` already returns in that case — a
deliberate parity decision, since `DynamicMinCut` can actually report a
real split on a disconnected graph, and testing that difference wasn't
this experiment's question.

## Benchmark Evidence

All numbers are from `cargo run --release`, this repository, one machine
(Linux x86_64, `rustc 1.94.1`).

**Scaling** (synthetic ring k-NN graph, k=8 — same topology the original
experiment used to size itself):

| n | Wrapper | Direct | Speedup |
|---:|---:|---:|---:|
| 50  | 76.8ms | 0.708ms | ~108x |
| 100 | 481.3ms | 1.895ms | ~254x |
| 200 | 2,712.9ms | 5.466ms | ~496x |
| 400 | 11,415.0ms | 18.620ms | ~613x |

**The actual 84-memory benchmark**, both backends run in the same
process for a direct before/after:

| Policy | Bridge Surv. | Recall@10 | Compaction | Slowdown vs. baseline |
|---|---:|---:|---:|---:|
| Baseline (no structural signal) | 66.7% | 100.0% | 74us | — |
| Soft, old backend | 66.7% | 100.0% | 117,589us | 1,589.0x (**FAIL**, >100x) |
| Soft, new backend | 66.7% | 100.0% | 2,081us | 28.1x (**PASS**) |

Bridge survival and recall are bit-identical between backends — exactly
what the hypothesis required, and good evidence the two implementations
compute the same answer at wildly different cost. As a side effect, the
new backend is also perfectly deterministic on the fixed graph where the
old one gave an empty, unusable result 15 times out of 30 identical calls.

## The Twist

Fixing the speed doesn't fix the policy. Bridge survival's gap over
baseline is still exactly 0.0 percentage points, against a required +15pp
— identical to the original, rejected finding.

That raised an obvious question the original experiment's own slowness
made too expensive to ask: was the zero effect just because the corpus was
forced down to a tiny 84 memories? Now that a single compaction call costs
low single-digit milliseconds instead of over a hundred, testing at 16x
scale takes seconds, not hours:

| Corpus size | Bridge-survival gap |
|---:|---:|
| 84    | +0.0pp |
| 168   | +0.0pp |
| 336   | **-6.2pp** |
| 672   | +0.0pp |
| 1,344 | +0.0pp |

The gap never approaches +15pp anywhere in that range, and briefly goes
negative. The zero effect isn't a small-sample artifact — it's a property
of using one *global* minimum cut over the whole candidate set as a proxy
for "which memory is a bridge," on this kind of clustered data.

## Limitations

- Single run per configuration; no variance characterization across
  repeated seeds.
- The scale-effectiveness check is exploratory and intentionally
  non-gating — it wasn't part of the pre-registered hypothesis, so it
  isn't used to flip today's accept/reject verdict, only to report as
  additional evidence.
- Only two of `ruvector-mincut`'s several min-cut entry points were
  compared; a *local* min-cut (scoped to a neighborhood around each
  candidate, rather than global) was not tried and is the natural next
  thing to test.
- Tested on synthetic Gaussian-cluster data only; whether the effectiveness
  finding transfers to real embedding corpora is still open.

## Production Relevance

The performance fix is real and immediately usable: any future attempt to
give this compaction policy a working structural signal no longer has to
fight a three-orders-of-magnitude latency tax just to run its own
benchmark. The effectiveness problem is now more clearly *the* problem,
not one of two entangled problems — which is a more useful place to leave
this thread for whoever picks it up next.

## RuVector Ecosystem Implications

This is a small, concrete example of a pattern worth generalizing: a
rejected experiment's own "next research" note, read carefully and
followed without moving the goalposts, can turn an ambiguous rejection
("doesn't work") into a specific one ("this exact implementation choice
was the performance problem; the underlying idea's effectiveness problem
is separate and still open"). That's a cheaper and more honest way to
build institutional knowledge across nightly runs than starting a fresh
topic every time.

## Future Direction

The next attempt at this problem should stop asking "is this graph's
global minimum cut a good bridge detector" and start asking "is there a
*local* structural signal, scoped to each candidate's own neighborhood,
that does better" — `ruvector-mincut`'s existing `localkcut` module is an
unused, directly available starting point for that.

## References

- ADR-345, `docs/research/nightly/2026-09-05-mincut-gated-forgetting/README.md`
- ADR-346, `docs/research/nightly/2026-09-17-mincut-direct-backend/README.md` (full report)
- `crates/ruvector-mincut/src/algorithm/mod.rs`, `.../algorithm/exact.rs`
- `crates/ruvector-agent-memory/src/graph_forget.rs`
