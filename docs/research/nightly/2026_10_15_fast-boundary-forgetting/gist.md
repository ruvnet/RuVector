# Replacing an Exact Min-Cut Solver with Two Cheap Heuristics Didn't Fix the Bottleneck — Here's Why

A prior experiment in this codebase found that a global minimum-cut
algorithm could identify "bridge" memories in an agent's memory store —
the ones that are the sole semantic link between two otherwise-unrelated
topic clusters — so that a compaction pass could avoid evicting them even
when their recency/frequency scores were low. The idea worked, but the
min-cut solver was 1,800–2,700x slower than a plain scalar ranking even on
an 84-item test corpus, and that was reported as a rejection: too slow to
ship.

The obvious next move is to swap the exact solver for something cheap.
Min-cut is a global algorithm; "a vertex sits on a structural boundary" is,
intuitively, a *local* property. Two classical, decades-old graph
heuristics capture exactly that intuition without ever running a global
algorithm:

- **Local clustering coefficient**: how many of a vertex's neighbors are
  also neighbors of each other? A vertex embedded deep inside one tight
  cluster has a high coefficient (its neighbors form triangles with each
  other). A vertex that bridges two clusters has a low one — its neighbors
  split across two groups that don't talk to each other.
- **Label propagation**: let every vertex adopt the majority label among
  its neighbors, repeat a few rounds, and see which vertices end up with
  neighbors who disagree on a label. Those are the boundary vertices.

Both are `O(V·k)`-ish per round over a k-NN graph — no global solver, no
iterative min-cut search, nothing more exotic than counting things in a
neighbor list. The hope was straightforward: get most of the exact
solver's signal at a small fraction of its cost.

## What actually happened

It didn't work, but *why* it didn't work is more interesting than a flat
"no."

### The bottleneck wasn't the solver

Both heuristics were implemented from scratch with zero new dependencies
— no `ruvector-mincut`, no `dashmap`, no `rayon` — just a k-NN graph built
by comparing every pair of candidate memories by cosine similarity and
keeping each vertex's top-k. Then the cheap heuristic runs on top of that
graph.

Timing the two phases separately gives the answer immediately: at 2,000
memories, building that k-NN graph takes about 280 milliseconds; running
the actual heuristic on top of it adds roughly another 70ms (for label
propagation) or essentially nothing (for clustering coefficient). The
heuristics are exactly as cheap as advertised. The graph *construction*
— an unavoidable, brute-force, all-pairs comparison — is not, and it was
never the thing being replaced. Removing an expensive solver and keeping
an expensive graph-build step is like swapping a slow database query for
a fast one but still copying the entire table into memory first either
way: you fixed the part you were looking at and left the part that
actually dominates untouched.

This is not a surprising result in retrospect — O(n²) is O(n²) regardless
of what you do with the resulting graph — but it had not been measured
directly in this codebase before. The original exact-mincut rejection
blamed "mincut is slow," which is true but incomplete: the graph it was
fed was already the expensive part.

The irony is sharper than it first appears: this is a vector-search
codebase. It ships a production HNSW index specifically so that
nearest-neighbor queries don't require comparing every pair of vectors.
Neither the original mincut integration nor this heuristic replacement
used it. Both hand-rolled the exact O(n²) operation the rest of the
codebase exists to avoid.

### The reference signal itself turned out to be questionable

The second failure is subtler. Before trusting a Jaccard-overlap score
between a cheap heuristic's output and the exact solver's output, it's
worth asking: on a graph with more than one bridge, is "the global minimum
cut" even the right thing to compare against?

A minimum cut, by definition, finds *one* 2-way split of the graph at the
lowest total edge weight. If a corpus has twelve different memories that
each bridge some pair of topic clusters, the global minimum cut doesn't
report all twelve — it reports whichever single split happens to be
cheapest, which might isolate just one or two low-degree vertices and
ignore the other ten entirely. Measuring this directly: the exact
solver's own boundary set, averaged over ten random corpora, overlapped
with the *actual* planted bridge memories by only 14%. The earlier
experiments that established "min-cut finds the bridge" used a carefully
constructed test graph with exactly one bridge by design — a case where
"the minimum cut" and "the bridge" are the same question. Once there's
more than one bridge, they stop being the same question, and nothing in
the earlier work had tested that.

This matters for interpreting the headline fidelity numbers. The
clustering-coefficient heuristic scored essentially 0% overlap with the
exact solver across every trial — a clean, unambiguous failure on its own
terms. But label propagation scored only 18% against the exact solver
while scoring 40% against the *actual* planted bridges — nearly three
times better than the solver it was being compared against. The thing
being used as "ground truth" was a worse answer, on this topology, than
one of the two candidates being measured against it.

### The one real positive, filed under "rejected anyway"

Label propagation's bridge-memory survival rate beat the plain
scalar-only baseline by 35–38 percentage points at both 500 and 2,000
memories — more than double the 15-point bar that was set before the
benchmark ran, with zero measurable cost to retrieval recall. On
effectiveness alone, it's a clear win. It still fails the overall
acceptance test, honestly, because the pre-registered performance bar
(no more than 3x slower than the scalar baseline) was violated by
roughly 85–240x — the same unaddressed k-NN construction cost described
above. A result can be effective and still not be the result you asked
for.

## What this is worth, if anything

Three concrete, falsifiable things came out of a run that ended in
rejection:

1. A precise attribution of where the time actually goes (the graph
   build, not the algorithm on top of it), which points at an obvious,
   specific fix: build the k-NN graph with the ANN index this codebase
   already has instead of brute force, and re-run the exact same
   benchmark.
2. Evidence that the metric used to validate a "faster approximation"
   should itself be checked against ground truth, not just against the
   thing it's approximating — because the thing it's approximating can
   be a biased reference on exactly the topologies that matter in
   practice.
3. A genuinely strong effectiveness result (label propagation's bridge
   survival gap) that is now decoupled from its performance problem and
   can be re-tested the moment the graph-construction bottleneck is
   addressed, rather than being thrown out along with the rejected
   mechanism it was riding on.

None of that required the benchmark to come out ACCEPT. It required
measuring both the win and the failure precisely enough to know which
part of the idea to keep.
