# Picking the right min-cut backend: a 65-340,000x speedup that still doesn't fix the actual problem

## Problem

`ruvector-agent-memory` has an experimental compaction policy,
`MincutGatedForgetting`, that tries to protect "bridge" memories — the sole
semantic link between two otherwise-disjoint topic clusters — from eviction,
using a graph min-cut to find them. A prior experiment (ADR-345) rejected it
for production: the min-cut call
(`ruvector_mincut::RuVectorGraphAnalyzer::partition()`) cost 76ms to 11.4
seconds per call on graphs of only 50-400 vertices — 1,800-2,700x slower
than the scalar baseline it was supposed to improve on — and, independently,
even when it ran, it didn't measurably change which memories survived.

The natural next question, which that same report asked but didn't answer:
was the huge cost a property of *doing a min-cut at all*, or a property of
*which min-cut implementation* was used? `ruvector-mincut` ships more than
one.

## Hypothesis

`RuVectorGraphAnalyzer` delegates to `MinCutWrapper`, which implements a
subpolynomial *dynamic* min-cut algorithm — the kind of data structure you
want when a graph is being repeatedly edited and you need to know the
current min-cut after every edit, cheaply, without recomputing from
scratch. `MincutGatedForgetting` doesn't do that: it builds a k-NN graph
once, from a fixed snapshot of compaction candidates, and asks for the
min-cut exactly once. That's a mismatch — you're paying for incremental
update machinery on a query that never updates.

`ruvector-mincut` also exports a second, lower-level type:
`DynamicMinCut` (despite the name, it's the crate's plain exact solver —
sparse Stoer-Wagner, `O(n)` phases over a max-adjacency heap). No geometric
instance ladder, no incremental-update bookkeeping. If the mismatch theory
is right, swapping to it should be dramatically faster for this module's
actual (one-shot) usage pattern.

## Technical design

Added `MincutBackend`, an enum selecting which API computes the boundary
partition — `GraphAnalyzer` (unchanged default) or `DynamicMinCut` (new,
opt-in via `.with_backend(...)`). Both backends build the identical k-NN
graph and share the same "which vertices touch a crossing edge" boundary
logic; only the min-cut computation itself differs. Fully additive — no
existing caller's behavior changes.

```rust
pub enum MincutBackend { GraphAnalyzer, DynamicMinCut }

impl MincutGatedForgetting {
    pub fn with_backend(self, backend: MincutBackend) -> Self;
    pub fn boundary_size(&self, entries: &[MemoryEntry]) -> usize; // new diagnostic
}
```

## Actual implementation

`crates/ruvector-agent-memory/src/graph_forget.rs`. The `DynamicMinCut`
path builds a `DynamicGraph` with the same edges (`1/distance` weights,
matching `RuVectorGraphAnalyzer::from_knn`'s own convention), calls
`DynamicMinCut::from_graph(graph, MinCutConfig::default())` (which computes
the cut immediately, at construction), and reads `.partition()` — an O(1)
getter on the already-computed result. Zero new Cargo dependencies (same
`ruvector-mincut` crate, a different already-exported type).

## Actual benchmark evidence

All release builds, one machine, fixed seeds. Four separate, independently
runnable binaries; raw output below.

**Scaling probe** (synthetic ring k-NN graph, k=8, one min-cut call per size):

| n | GraphAnalyzer | DynamicMinCut | speedup |
|---:|---:|---:|---:|
| 19 | 78,815ms | 0.230ms | 342,089x |
| 50 | 91ms | 0.513ms | 178x |
| 100 | 533ms | 1.27ms | 421x |
| 200 | 2,975ms | 3.50ms | 852x |
| 400 | 11,844ms | 11.4ms | 1,041x |
| 800 | 26,024ms | 34.2ms | 761x |
| 2,000 | 84,589ms | 286ms | 295x |

**Determinism probe** (19-vertex two-clique-plus-bridge, 50 identical trials):
`GraphAnalyzer` — 0/50 empty (confirming a separate prior fix, ADR-346,
landed correctly), 917.6ms/call average. `DynamicMinCut` — 0/50 empty,
0.058ms/call, and **the literal same partition returned on all 50 trials**
(sparse Stoer-Wagner sorts vertices/edges before processing, so it has no
hash-iteration-order dependency to begin with).

**The actual pre-registered acceptance benchmark** — same 84-entry corpus,
same thresholds as the original rejected experiment, run unmodified
alongside the new backend for direct comparison:

```
  Soft (GraphAnalyzer)   compaction slowdown (2764.2x) <= 100x : FAIL
  Hard (GraphAnalyzer)   compaction slowdown (2711.5x) <= 100x : FAIL
  Soft (DynamicMinCut)   compaction slowdown (37.8x)  <= 100x : PASS
  Hard (DynamicMinCut)   compaction slowdown (42.0x)  <= 100x : PASS

  Soft (GraphAnalyzer)   bridge-survival gap (+0.0pp) >= 15pp : FAIL
  Hard (GraphAnalyzer)   bridge-survival gap (+0.0pp) >= 15pp : FAIL
  Soft (DynamicMinCut)   bridge-survival gap (+0.0pp) >= 15pp : FAIL
  Hard (DynamicMinCut)   bridge-survival gap (+0.0pp) >= 15pp : FAIL
```

The performance gate flips from FAIL to PASS. The effectiveness gate — the
whole point of the policy — does not move at all, on either backend.

That second result was worth checking harder. Because `DynamicMinCut` made
a single compaction call cheap (low milliseconds instead of ~90ms), it
became possible, for the first time, to run the same experiment at 10x and
50x the original corpus size — something the old backend made computationally
infeasible. A new public diagnostic, `boundary_size()`, reports how many
candidates the structural signal is even touching:

| Scale | Memories | Bridge-survival gap | Boundary size |
|---:|---:|---:|---:|
| 1x | 84 | +0.0pp | **6** |
| 10x | 840 | +0.0pp | **0** |
| 50x | 4,200 | +0.0pp | **0** |

At the original scale, the signal is real (6 of 84 candidates flagged) but
inert — it never changes the outcome. At 10x and 50x scale, it's not just
ineffective, it's **entirely absent**: the k-NN similarity graph becomes
disconnected (a larger absolute bridge count makes at least one isolated,
below-similarity-threshold bridge vertex increasingly likely), and both
backends correctly, silently, fall back to "no structural signal" for a
disconnected graph — for the whole compaction pass, not just the isolated
vertex.

## Limitations

- No WASM, concurrent-mutation, or fault-injection measurement.
- The scale probe is one seed family per scale, not a distribution — the
  "boundary drops to exactly 0" result is observed, not statistically
  characterized.
- Why a non-empty boundary set at n=84 still produces zero effect is not
  instrumented further — flagged as the highest-leverage open question.
- Synthetic Gaussian-cluster data throughout; no real embedding corpus.

## Production relevance

`MincutBackend::DynamicMinCut` ships as an available, tested, opt-in
backend — genuinely useful today for the general lesson it demonstrates:
`RuVectorGraphAnalyzer`'s dynamic-update machinery is the wrong tool for a
one-shot, build-once-query-once min-cut call, and eighteen other crates in
this codebase depend on `ruvector-mincut` and could hit the same mismatch.
`MincutGatedForgetting` itself remains unpromoted — same verdict as before,
now backed by sharper evidence about *why*.

## RuVector ecosystem implications

This closes a concrete follow-up item from two prior nightly research runs
(ADR-345, ADR-346) without touching `ruvector-mincut` itself — a
downstream-only fix. It also raises the realistic ceiling for any future
attempt at a structural agent-memory signal from "a few hundred entries" to
"tens of thousands," and turns a previously-infeasible question ("does this
hold at scale?") into a cheap, five-minute experiment — which is exactly
how it got answered in this same run.

## Future direction

1. Instrument why a non-empty boundary set still produces zero survival
   effect at the original scale.
2. Sweep `min_similarity`/`k_neighbors` to find where the k-NN graph stops
   disconnecting at larger corpus sizes.
3. Evaluate `PolylogConnectivity` as a fix at the source
   (`RuVectorGraphAnalyzer` itself), rather than routing around it.
4. If either of the first two finds a viable parameterization, re-run the
   exact same pre-registered benchmark, unmodified, per the "don't move the
   goalposts" rule this run followed.

## References

- ADR-350 (this run), ADR-345, ADR-346 —
  `docs/adr/` in `ruvnet/ruvector`
- `crates/ruvector-agent-memory/src/graph_forget.rs`
- `crates/ruvector-mincut/src/algorithm/{mod.rs,exact.rs}`,
  `crates/ruvector-mincut/src/integration/mod.rs`
