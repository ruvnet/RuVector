# Sparsifying the graph didn't fix our min-cut RAG retriever — and the reason why is the actual finding

A couple months ago we published a research PoC in [RuVector](https://github.com/ruvnet/ruvector) that retrieves a *budget-bounded, coherent* context set for RAG using graph min-cut, instead of plain top-k similarity. It worked, but it had an obvious, documented problem: building the similarity graph it runs min-cut over cost O(n²), which meant 1.27 seconds per query at 3,000 chunks. The fix seemed obvious — don't build the whole graph, build a sparse k-NN graph instead. That's a one-line item in nearly every systems paper's future-work section.

We built it. It didn't work the way the obvious fix is supposed to.

## The setup

`MinCutBounded` retrieval works like this: treat each candidate chunk as a graph node, wire a virtual source to every chunk with capacity equal to its cosine similarity to the query, wire every chunk to a virtual sink with capacity `1 - similarity`, and add edges between chunks that are similar to each other. Run max-flow/min-cut. The chunks left on the source side of the cut are your coherent, query-relevant partition — rank them and truncate to your token budget.

The expensive part is building those inter-chunk edges: comparing every pair of chunks is O(n²).

**Hypothesis**: replace the exhaustive all-pairs comparison with locality-sensitive hashing (random-hyperplane LSH / SimHash), which buckets similar vectors together using cheap hashing instead of exhaustive comparison. This should cut candidate-pair-checking from O(n²) to roughly O(n·k). We set an explicit, pre-registered bar: at least a 10x end-to-end latency improvement at n=3,000, with precision staying within 5 points of the original.

## What happened

LSH did exactly what the textbook says: at every scale we tested (n=200 to n=20,000), it checked about **10x fewer candidate pairs** than the exhaustive scan, consistently, measured exactly rather than estimated.

End-to-end query latency dropped by... 1.1x to 1.5x. Nowhere near 10x.

We dug in. The reason is almost embarrassingly simple in hindsight: **this retriever is built for clustered, coherent data** — that's the whole point of it. On a tightly clustered synthetic corpus, when you check a candidate pair of chunks from the same cluster, it passes the similarity threshold *almost every time* — we measured 97.5% at one scale. So whether you find that pair by scanning all n² pairs or by LSH-bucketing your way to it in 1/10th the comparisons, you end up keeping almost the same edges. You've made *discovering* the graph cheaper. You haven't made the graph itself any smaller. And the thing that actually costs time — running max-flow over that graph — doesn't care how the edges were found, only how many there are.

## Round two

So we fixed the actual problem: instead of keeping every edge that clears a similarity threshold, cap every node to its mutual top-k nearest neighbors (an edge only survives if each endpoint considers the other one of its k best). This provably bounds the graph to at most n·k/2 edges, no matter how tight the clusters are.

This worked better — a real, reproducible **2.8x speedup at n=3,000, with zero precision loss**. That's genuinely useful and we're keeping it as an available option.

It's still not 10x. And at n=8,000, it fell apart again: 3.36 seconds per query, *worse* than our simplest fallback (a basic coherence-gated graph traversal, no min-cut at all), despite having a graph an order of magnitude sparser than before. The slowdown from n=3,000 to n=8,000 tracked almost exactly what you'd expect from pure quadratic scaling (predicted 7.1x, measured 8.0x) — as if the degree cap had barely mattered.

## The actual bottleneck

We'd now ruled out two explanations: it's not how expensive it is to *discover* the edges (fixed by LSH, confirmed), and it's not how many edges end up in the graph (fixed by degree-capping, confirmed — we verified the bound directly in a unit test). What's left is the max-flow solver itself: a textbook Edmonds-Karp implementation, which finds *some* augmenting path on each iteration via breadth-first search, with no preference for high-capacity paths. On a network where capacities are continuous similarity scores rather than small integers, that can mean needing a lot of iterations to converge — a cost that doesn't care how sparse your input graph is, because augmenting-path count isn't a function of edge count alone.

We didn't instrument this directly this round — that's the very next thing to do — but the scaling behavior is consistent with it, and it would explain everything we measured.

## Why we're publishing a result that didn't hit its target

Because the negative result is the useful part. If we'd stopped after "LSH gives a 10x cheaper candidate search" and shipped it without measuring end-to-end latency, we'd have shipped a change that does almost nothing for the problem it claims to solve. The degree-capped version is a real, if modest, win, and it's going in as an available option. But the actual lesson — that a correct graph-sparsification can still hit a wall because the *solver*, not the *graph*, is the bottleneck — is the kind of thing that's easy to miss if you only benchmark the thing you changed and declare victory on a partial metric (candidate pairs checked) instead of the metric that matters (end-to-end latency).

## What's next

Swap Edmonds-Karp for a capacity-scaling max-flow algorithm (or evaluate whether this repo's existing subpolynomial dynamic min-cut implementation, built for a different graph problem, can be adapted to this one) — holding the now-fixed degree-capped graph construction constant, so we're finally testing the solver in isolation instead of conflating it with graph construction the way both of our attempts this round accidentally did at first.

## Production relevance

If you're building RAG or agent-memory retrieval and you're tempted to reach for "coherent context via graph partitioning" — it works, and the coherence guarantee is real (precision stayed at 1.000 across every configuration we tested) — just don't assume sparsifying your similarity graph is sufficient to make it fast at scale. Measure the solver, not just the graph.

---

*Full methodology, raw benchmark output, and code: [`docs/research/nightly/2026-10-02-sparse-knn-mincut-bounded-rag/`](https://github.com/ruvnet/ruvector/tree/main/docs/research/nightly/2026-10-02-sparse-knn-mincut-bounded-rag) and [ADR-352](https://github.com/ruvnet/ruvector/blob/main/docs/adr/ADR-352-sparse-knn-mincut-bounded-rag.md) in the [RuVector](https://github.com/ruvnet/ruvector) repository.*
