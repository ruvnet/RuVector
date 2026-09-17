# Finding the Slow Path: 500x Faster Min-Cuts by Skipping the Wrapper

## Problem

A week and a half ago I rejected my own feature. `MincutGatedForgetting` —
a memory-compaction policy that uses a graph min-cut to protect
structurally important "bridge" memories from eviction — didn't help (the
structural signal made no measurable difference) and was absurdly slow
(1,800-2,700x slower than the baseline it was supposed to improve on, on a
graph of just 84 memories). I wrote it up, filed the ADR, and moved on, but
left one thread dangling: the slowness was traced to one specific API call,
`RuVectorGraphAnalyzer::from_knn(...).partition()`, and I never checked
whether a *different* call into the same underlying library would avoid it.

Today I checked. It does — dramatically — and finding out why required
actually reading the slow function's implementation instead of trusting its
name.

## Technical Design

`ruvector-mincut` is a general-purpose dynamic minimum-cut library. It
exposes (at least) two ways to ask "what's the minimum cut of this graph":

- `RuVectorGraphAnalyzer::from_knn(edges).partition()` — a convenience
  wrapper purpose-built for exactly the k-NN-graph use case my compaction
  policy has.
- `MinCutBuilder::new().with_edges(edges).build()` — a lower-level
  constructor for the library's core `DynamicMinCut` structure, meant
  (per its module docs) for workloads that need to *incrementally* update a
  min-cut as edges come and go.

I'd used the first one because it looked like the right tool for the job —
literally named for vector-graph integration. Reading its implementation
instead of its name told a different story. `RuVectorGraphAnalyzer::
partition()` calls into a `MinCutWrapper` that implements a *different*
algorithm than the one `DynamicMinCut` itself uses: it maintains up to 100
geometrically-scaled "instances," each a full copy of the min-cut problem
tuned to detect a cut in a specific weight range, and on every query it
lazily builds and populates instances — replaying every single edge into
each one — until one of them reports a confident answer. That's a
reasonable design for a system meant to *maintain* a min-cut across a
stream of updates, answering "what's the cut *right now*" cheaply after
the first expensive build. It's a terrible design for what I was actually
doing: building a brand-new graph from scratch and asking for its cut
exactly once, every single compaction call, then throwing the whole
structure away.

`MinCutBuilder::build()`, by contrast, does the boring thing directly: run
one depth-first spanning-tree construction over the graph, then one
breadth-first pass computing the cut induced by each spanning-tree edge,
and keep the minimum. One pass. No instance replay. No hidden state left
over for a next query that will never come, because I don't have a next
query — I rebuild the graph from scratch every time.

The fix in code is almost embarrassingly small: a new `BoundaryMethod` enum
with a second variant, and a function that builds the same edge list the
old code already built, feeds it to `MinCutBuilder` instead of
`RuVectorGraphAnalyzer`, and reads the answer back. The interesting part
wasn't the code; it was noticing that "the vector-graph-shaped API" and
"the fast API" were two different things, and that only reading the
library's internals — not its module doc comments, which describe
`DynamicMinCut` as having "O(n^o(1)) amortized update time" without
mentioning that a cold one-shot query pays a very different bill — revealed
which was which.

## Actual Implementation

Two real bugs came out of building this, and I'm including both because a
writeup that only shows the version that worked isn't honest about what the
work actually was.

First: the k-NN neighbor list this code builds stores *cosine distance*
(smaller = more similar), but `RuVectorGraphAnalyzer::from_knn` silently
inverts that to a *weight* (`1/distance`, so near-duplicates get heavy,
hard-to-cut edges) before handing it to the underlying graph. `MinCutBuilder`
does no such inversion — it takes whatever number you give it as the literal
edge weight. My first attempt handed it the raw distance. The result: every
near-duplicate pair (distance close to zero) looked like the *cheapest*
possible thing to cut, exactly backwards from the intended graph structure.
My unit tests — which check that a synthetic bridge vertex with two known
cut edges gets flagged correctly — failed immediately and unambiguously.
Good tests catching a real bug is the system working as designed, not a
setback.

Second, dumber bug: the k-NN neighbor list is directed (vertex A's neighbor
list can mention vertex B without B's list mentioning A back), but a
plain edge list is undirected, and the underlying graph structure throws an
error if you try to insert the same undirected pair twice. My scaling and
determinism probe scripts — separate small files I use to measure raw
latency and repeatability outside the full compaction pipeline — built
their edge lists without deduplicating by unordered pair, so the very
first duplicate made the whole build fail instantly. This produced a
benchmark reading of "0.0 milliseconds per call, 100% empty result," which
looked suspicious specifically *because* it was too good and too uniform to
be a real timing — a useful reminder that an implausibly clean number is
worth a second look before you write it down as evidence.

## Actual Benchmark Evidence

With both bugs fixed, I re-ran the exact scaling probe, determinism probe,
and 84-memory benchmark from the original rejected experiment, changing
only which `ruvector-mincut` API gets called.

Scaling (ring graph, 19 to 400 vertices, one call each):

| vertices | old API | new API | speedup |
|---:|---:|---:|---:|
| 19 | 68.4-69.7 seconds | 0.36-0.40ms | ~180,000x |
| 50 | 71-87ms | 1.2-2.0ms | ~45-60x |
| 100 | 410-537ms | 2.8-4.0ms | ~135-145x |
| 200 | 2.4-2.7s | 8.0-29.0ms | ~90-300x |
| 400 | 11.1-11.5s | 21.1-22.4ms | ~510-525x |

Determinism (fixed 19-vertex graph with one known bridge, 30 repeated calls
on byte-identical input, two separate runs): the old API returned an
unusable empty result 27-50% of the time; the new API never did — 0 empty
results out of 60 calls across both runs, and correctly identified the
bridge vertex in all 60.

On the original 84-memory compaction benchmark, run six times: the old API
reproduced its own originally-reported 1,800-2,700x slowdown almost exactly
(2,163x-2,800x this time around). The new API landed at 65x-106x — below
the pre-set 100x acceptance bar in most (10 of 12) individual runs, right
at the noisy edge of it in the other two. That noise is worth being
honest about rather than rounding away: the baseline operation here takes
about 30 microseconds, so a ratio against it swings by several multiples
from nothing more than ordinary OS scheduling jitter. The more trustworthy
number is the new API's *absolute* time, which sat in a tight 2.2-3.6
millisecond band across every single run regardless of what the noisy
baseline did.

## Limitations

I did not get a clean, unqualified win. Alongside the speed and
determinism improvement, the new API found a *smaller* set of boundary
vertices than the old one, on this specific corpus, every single time
(50.0% and 58.3% of bridge memories survived, versus 66.7% for the old
API and the plain baseline) — with zero run-to-run variance, unlike the old
API's own flakiness. A minimum cut of a graph isn't always unique, and it
looks like these two algorithms break ties differently, landing on two
different — both technically valid — answers to "what's the cheapest way
to split this graph in two." I don't yet know which answer is "more
correct" for the bridge-protection use case, or whether that even has a
well-defined answer. I'm not flipping the default to the faster method
until I understand that, and I said so explicitly in the follow-up
decision record rather than quietly picking the faster option and hoping
the discrepancy doesn't matter.

## Production Relevance

The underlying compaction feature this was built for is still not
recommended for production use — that verdict didn't change today, and
this write-up doesn't claim otherwise. What did change: anyone who *does*
turn that feature on now has a documented, tested, 25-500x-faster,
fully-deterministic alternative available, opt-in, with the slow original
kept as the default so nothing's behavior silently shifts underneath it.
More generally, this is a hardening finding against the underlying min-cut
library itself, independent of my specific compaction policy: any other
caller doing a one-shot "what's the cut of this graph, right now" query —
rather than incrementally maintaining a cut across a stream of edge
updates — is probably hitting the same wall, and has the same fix
available.

## RuVector Ecosystem Implications

This is a small, unglamorous result — a follow-up item from a rejected
experiment, closed by reading library internals more carefully than the
first time around — but it's the kind of result the ecosystem's
architecture increasingly needs: cheap, structural, graph-native signals
that anything from agent memory to retrieval to autonomous infrastructure
can call inline rather than schedule as a batch job. Whether that
signal is *correct*, not just fast, is still an open question here, and
that gap — not the speed number — is the interesting thread left for next
time.

## Future Direction

The next step isn't shipping the fast path as the new default. It's
figuring out, using the library's own deterministic-by-construction
tie-breaking mode, which of the two answers this experiment produced is
actually the right one — and then, only once that's settled, re-running
the original bridge-protection experiment at a corpus size that was
computationally out of reach before today.

## References

- ADR-346, `docs/adr/ADR-346-direct-mincut-bridge-detection.md`
- Full nightly report: `docs/research/nightly/2026-09-15-direct-mincut-bridge-detection/README.md`
- Raw benchmark output: `docs/research/nightly/2026-09-15-direct-mincut-bridge-detection/raw-runs.txt`
- Prior work: ADR-345, `docs/adr/ADR-345-mincut-gated-forgetting.md`
