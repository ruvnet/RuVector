# I Found the Faster Version of My Rejected Algorithm. It's Still Rejected.

## Problem

Two weeks ago, a nightly research run on this codebase tried to protect
"bridge" memories — the sole semantic link between two otherwise unrelated
topic clusters — from being evicted during agent-memory compaction, by
asking a general-purpose minimum-cut graph engine to flag structurally
load-bearing vertices before eviction. It got rejected: the engine
(`RuVectorGraphAnalyzer`) was 1,800-2,700x slower than doing nothing special,
and even where it did run, it showed no measurable benefit.

The obvious next question, and the one that experiment's own writeup posed
without answering: is the *engine* the problem, or is the *idea* the
problem? If a faster engine exists in the same crate, trying it costs little
and answers a question the first experiment left open.

It turns out a faster engine does exist, sitting unused right next to the
slow one — the crate ships a second, distinct min-cut implementation
documented as a "SODA 2025 approximate min-cut" algorithm. Reading its
source before writing a single benchmark line raised a specific concern:
the function that hands back "which vertices are on which side of the cut"
doesn't look at the cut at all. This is the writeup of testing that concern
directly, and what happened when the corpus-level numbers came in.

## Technical Design

Same policy shape as the rejected experiment — a `Soft` mode (additive
scoring bonus for flagged vertices) and a `Hard` mode (reserved eviction-immune
budget) — with exactly one thing changed: which min-cut engine computes the
"boundary" flag. Everything else (the k-NN similarity graph construction,
the dataset, the acceptance thresholds) is held byte-identical to the
original experiment specifically so any difference in the result can only be
attributed to the engine swap.

Before running the corpus-level benchmark, a much smaller, cheaper test: two
hand-built graphs, one where the true minimum cut splits the vertices evenly
and one where it doesn't. Any implementation that genuinely returns "the two
sides of the minimum cut" should get both right. This new engine gets the
balanced one right and the unbalanced one wrong — because its side-extraction
step is a plain breadth-first walk that stops once it has visited exactly
half of *all* the vertices, regardless of where the graph's actual weak
edges are. On a perfectly balanced graph, "half of everything" and "the true
smaller side" happen to be the same number, so the bug is invisible unless
you go looking for it with an unbalanced graph.

## Implementation

`ApproxMincutForgetting`, a `CompactionPolicy` implementation in
`ruvector-agent-memory`, feature-gated behind the same opt-in flag as its
predecessor. It builds the identical k-nearest-neighbor cosine-similarity
graph as before, feeds it to the new engine instead of the old one, and
extracts "boundary" vertices the same way: anyone with a neighbor edge
crossing the reported partition. Two small executable programs ship
alongside it — one that runs the balanced/unbalanced graph check above, and
one that reproduces the full 84-memory corpus benchmark from the original
experiment with both engines side by side for direct comparison.

## Actual Benchmark Evidence

On the identical synthetic corpus (84 memories, 6 topic clusters, 12
deliberately interpolated "bridge" memories, same random seed as the
original experiment), release build, one machine. The new engine's rows are
the mean of 10 repeated trials, not a single run — see "A Bonus Bug" below
for why that turned out to matter:

| Policy | Bridge survival | Recall@10 | Compaction time |
|---|---|---|---|
| Do-nothing baseline | 66.7% | 100.0% | ~60 microseconds |
| Old engine, Soft mode | 66.7% | 100.0% | ~105 milliseconds |
| Old engine, Hard mode | 66.7% | 100.0% | ~105 milliseconds |
| New engine, Soft mode | **39.2%** (range 8.3%-58.3%) | ~97-98% | ~4.5 milliseconds |
| New engine, Hard mode | 66.7% (every single trial) | 100.0% | ~4.6 milliseconds |

The new engine is genuinely about 22-24x faster than the old one — a real,
reproducible fix to the latency problem. But bridge survival either stays
exactly flat (Hard mode: no better than doing nothing, and remarkably
*exactly* 66.7% in all 10 trials) or gets worse — often much worse (Soft
mode: never once matched the baseline across 10 trials, from 8 points worse
in the best sampled case to nearly 60 points worse in the worst). The
Soft-mode result is the more interesting failure: because its scoring rule
*adds* a bonus to whatever the broken boundary check flags, a wrong flag
doesn't just fail to help — it actively pushes the wrong memories up the
keep-list, bumping real bridges off the list they would have stayed on with
no structural signal at all.

The balanced/unbalanced graph check nails down why: the reported cut
*value* is exactly correct in both hand-built cases (1.0). The reported
*partition* — which vertices are on which side — is correct on the balanced
graph and wrong on the unbalanced one, exactly as predicted by reading the
code before running anything.

## A Bonus Bug: The Fast Engine Isn't Even Consistent With Itself

While writing a unit test for this experiment, a strange thing happened: a
test asserting the new engine would *unreliably* protect a hand-built test
bridge instead passed reliably, every time, in one process. Repeating the
exact same test as separate runs turned up the real story: it failed 2 out
of 6 times. The new engine's internal bookkeeping uses a standard
dictionary-like data structure whose iteration order is randomized fresh
each time a program starts — and the buggy partition logic picks its
arbitrary starting point by asking that structure for "any" entry. So
"which vertices get flagged as boundary" doesn't just fail to reflect the
true cut — it isn't even the same answer twice, on the exact same graph, run
to run. That's a second, independent bug layered on top of the first, and
it's why this write-up reports a bridge-survival *range* instead of a single
percentage: a single run's number would have been true and also misleading.

## Limitations

- Baseline and old-engine timings are single runs (no measured variance was
  found for either in this corpus); the new engine's rows are means of 10
  in-process trials, characterizing but not exhaustively bounding its
  run-to-run variance.
- Only two engines were compared (the original slow one, and this new fast
  one); a third engine in the same crate (`DynamicMinCut`) was inspected but
  not benchmarked, because reading its source showed its own
  "approximate mode" configuration flag isn't actually wired into its
  computation path — it would very likely just reproduce the first
  experiment's slow numbers, so benchmarking it wasn't the most useful use
  of this run's time.
- This says nothing about whether a *correctly implemented* fast
  approximate min-cut would protect bridges better than doing nothing — only
  that this specific implementation's side-extraction currently can't be
  trusted to try, and currently can't even be trusted to fail the same way
  twice.

## Production Relevance

None yet, and that's a genuine, useful result: it converts "mincut-based
bridge protection is too slow to use" into a much more specific, much more
fixable claim — "this crate's fast min-cut entry point has a one-function
bug in how it reports which side of the cut each vertex is on." Whoever
picks up that specific, cheaply-reproducible bug (a two-graph test file
demonstrates it in under a second) can re-run the exact same corpus-level
benchmark used here, unmodified, to check whether fixing it actually
delivers the bridge-protection benefit the original idea was chasing.

## RuVector Ecosystem Implications

This keeps the compaction-policy design space (`CompactionPolicy` trait,
scalar + structural scoring composition, the eviction witness chain from the
first experiment) exactly as extensible as it was, and adds one more
concretely falsified path plus one concretely identified bug to the
project's accumulated evidence — so the next person who reaches for "let's
try the mincut crate's other API" doesn't have to re-discover either finding
by hand.

## Future Direction

Fix the one function — and while there, make its arbitrary starting-point
choice deterministic instead of hash-order-dependent — then re-run the one
already-written benchmark and read the "bridge survival gap" numbers again.
No other part of this experiment would need to change to answer the
question this one leaves open.

## References

- The original experiment this follows up on:
  `docs/research/nightly/2026-09-05-mincut-gated-forgetting/`
- "Approximate Min-Cut in All Cut Sizes" (SODA 2025, arXiv:2412.15069) — the
  paper the tested implementation's own doc comment cites as its basis.
- Full methodology, ADR, and raw benchmark output:
  `docs/research/nightly/2026-09-19-approx-mincut-forgetting/README.md`,
  `docs/adr/ADR-346-approx-mincut-forgetting-partition-gap.md`.
