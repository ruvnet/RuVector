# Fixing Half a Rejected Idea: A Faster, Deterministic Local Cut Engine

## Problem

A previous experiment in this repository (`ruvector-agent-memory`'s
`MincutGatedForgetting`, ADR-345) tried to give agent-memory compaction a
structural "don't evict the bridge" signal by running a general-purpose
global minimum-cut algorithm (`RuVectorGraphAnalyzer::partition()`) over a
k-nearest-neighbor similarity graph of the candidate memories. It was
rejected: that specific call cost 76ms to 11.4 seconds depending on graph
size (50-400 vertices), and repeated calls on byte-identical input didn't
even return the same answer — 15 out of 30 trials on a small, hand-built
test graph came back empty.

That leaves an obvious, narrower follow-up question, separate from whether
the whole idea is worth pursuing: is the slow, flaky part fixable by
swapping in a different algorithm from the same library, without touching
anything else? This is the writeup of that narrower experiment.

## Hypothesis

Same synthetic dataset as the original experiment (clusters of memories
plus a handful of "bridge" memories that interpolate between two clusters),
scaled up to nearly 1,000 memories this time. Swap the boundary-detection
step for `ruvector_mincut::localkcut::DeterministicLocalKCut` — an
implementation of a fully-deterministic local minimum-cut search from a
December 2024 paper, which looks at one vertex's immediate neighborhood
instead of partitioning the whole graph — and measure whether it fixes the
speed and determinism problems while keeping whatever structural benefit
the original approach had.

## Technical Design

The new engine builds the identical k-NN similarity graph the original one
did, but as the mincut crate's own graph type, and instead of one expensive
call that partitions everything, it makes one cheap call per vertex asking
"is this specific vertex separated from the rest of the graph by only a
handful of edges?" A vertex gets flagged as structurally important if the
answer is yes.

Two design mistakes surfaced and got fixed before the real benchmark ran
(disclosed here rather than smoothed over):

1. The library's own helper for picking which vertices to start the search
   from turned out to work against the goal — it immediately mixes in a
   vertex's neighbors before checking anything, so a low-degree "bridge"
   vertex's small, distinctive cut never gets a chance to be seen on its
   own. Starting from just the one vertex being tested, and letting the
   search grow outward itself, fixed this.
2. Even after that fix, letting the search expand more than zero hops
   caused it to swallow entire clusters at once (because this synthetic
   data's clusters are almost fully connected internally), which flags
   *everything* as a boundary and defeats the whole point. Capping the
   search at zero hops — effectively "does this vertex have very few
   neighbors" — avoided that, at the cost of not really using the
   algorithm's multi-hop capability on this particular dataset.

A third dead end, found but not used: a different algorithm in the same
library (`ApproxMinCut`) looked promising — it's seeded and deterministic —
but its function for returning *which vertices* are on which side of the
cut turned out to be disconnected from the cut it actually computes; it
just returns half the vertices in traversal order regardless of the real
answer. Its "how big is the cut" number is fine; its "which vertices"
answer is not, and this experiment needs the latter.

## Real Results

Comparing the two engines on the same corpus, scaled from 84 up to 924
memories, release build, one fixed seed:

| n | baseline | old (global) engine | new (local) engine |
|---:|---:|---:|---:|
| 84  | 66 microseconds | 264 milliseconds | 1.0 milliseconds |
| 168 | 136 microseconds | 2.2 seconds | 3.5 milliseconds |
| 924 | 782 microseconds | not measured (already too slow at 168) | 87 milliseconds |

At the one size both engines finished within a shared time budget, the new
engine was 623x faster. It also produced the identical set of flagged
vertices across 20 repeated runs on unchanged input, every time — the old
engine's known flakiness didn't reappear in this (coarser, downstream)
check, for reasons discussed in the limitations below.

That's the good news. The bad news, measured just as honestly: the original
experiment's core effectiveness question — does flagging structurally
important vertices actually change which memories survive compaction? — is
still answered "no" here, exactly as it was in the original experiment.
Protected vertices ended up identical to what the plain scalar baseline
would have kept anyway, with both engines. And a new problem showed up that
neither engine caused: building the similarity graph in the first place
costs roughly the square of the memory count, and at these sizes that cost
alone made the *overall* compaction call 26x to over 100x slower than the
baseline — a real cost this experiment hadn't previously isolated because
the old engine's own slowness was so much larger it hid it.

So: the narrow question (is the algorithm swap faster and more reliable)
gets a clear yes. The broader question (is this whole approach ready to use)
still gets a no, for two separate reasons — one old (no effectiveness
benefit) and one newly measured (graph-construction cost dominates at
scale) — neither of which the algorithm swap could have fixed by itself.

## Limitations

- One random seed, same as the previous experiment. Neither run
  characterizes variance across seeds.
- The "deterministic" claim here was checked by re-running full compaction
  and comparing the final kept-memories list, not by directly re-running
  the previous experiment's stricter check (repeatedly calling just the
  cut-finding step in isolation and comparing raw results). The two aren't
  directly comparable, and the stricter check wasn't repeated against the
  new engine.
- Entirely synthetic Gaussian-cluster data; nothing here has been tried
  against real memory embeddings.
- The zero-hop restriction that made this dataset behave means the "local"
  algorithm's actual multi-hop search behavior went essentially untested on
  this corpus shape.

## What's Next

The most promising next step isn't another cut algorithm — it's the newly
found graph-construction cost, which now dominates the whole approach
regardless of which cut algorithm sits on top of it. This workspace already
has an approximate-nearest-neighbor index that could plausibly replace the
brute-force all-pairs similarity computation; that's a more promising
target than further cut-algorithm tuning.
