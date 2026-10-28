# When a plausible reranking fix doesn't actually help: adaptive vs. global coherence gating under filtered ANN retrieval

## Problem

Approximate vector search returns noisy candidates; graph-based reranking (diffusing scores
over a candidate similarity graph) is a known way to recover lost recall. RuVector's
`ruvector-gnn-rerank` crate implements this with `GnnMincutReranker`, which gates score
diffusion so it only flows across edges whose cosine similarity exceeds one **global, fixed**
threshold (0.50 by default).

That threshold is implicitly calibrated for the "normal" case: a first-stage candidate pool
drawn from the full corpus. But production retrieval is rarely unfiltered — tenant scoping,
access control, freshness windows, and similar metadata filters routinely shrink the
candidate pool *before* reranking sees it. The intuition: a fixed global bar, calibrated for
a dense pool, could leave many nodes in a sparse, filtered pool with no edges clearing it —
collapsing the reranker toward a plain noisy baseline exactly when reranking would help most.

## Hypothesis

Replace the global threshold with a **per-node, relative** gate: each node keeps its own top
fraction of local edges (by similarity rank), regardless of the absolute similarity value.
This should adapt automatically as the pool sparsifies, instead of assuming the unfiltered
similarity scale still applies.

Formally: at low filter selectivity (≤10% of the pool retained), the new
`AdaptiveCoherenceReranker` should beat `GnnMincutReranker`'s recall@10 by at least 3
percentage points, averaged over 200+ trials, without regressing unfiltered recall by more
than 0.5pp or costing more than 2x the latency.

## Technical design

`AdaptiveCoherenceReranker` reuses the existing crate's `CandidateGraph` (an O(n²·dim)
cosine-similarity k-NN graph over the candidate set) and diffusion update
(`α·prev + (1-α)·weighted_mean_of_kept_neighbours`). The only change: instead of keeping
edges above a fixed similarity value, each node keeps `max(min_keep, ⌈keep_fraction ×
degree⌉)` of its own strongest edges (the graph's adjacency lists are already sorted
descending by similarity, so this is a truncation, not a resort). Defaults: `α=0.60,
keep_fraction=0.50, min_keep=1, k_graph=8` — chosen to match `GnnMincutReranker`'s defaults
wherever a direct analog exists, for a fair comparison.

## Benchmark

A seeded, deterministic synthetic workload: 5,000 vectors, 128 dimensions, 20 Gaussian
clusters. For each trial, a query is jittered near a random corpus point; a noisy
first-stage pool of 300 candidates is drawn (true top-300 by L2 distance, independently
perturbed with `N(0, 0.40)` noise — the same noise model the crate already uses); each pool
member also gets one `Uniform(0,1)` filter-admission draw. A selectivity level `s` keeps
exactly the candidates whose draw is below `s`. Because the same draws are reused across
every `s` within a trial, lower-selectivity pools are literal subsets of higher-selectivity
ones for that trial — a tighter experimental control than redrawing independent filters per
level. Ground truth for recall@10 is the exact-L2 oracle's top-10 **on that same filtered
pool** (not the corpus-global top-10), since a restrictive filter can legitimately exclude
the global top-k entirely.

200 successful trials were run per selectivity level (`s ∈ {1.00, 0.50, 0.25, 0.10, 0.05}`).

## Results

| s | NoisyScore | GnnMincut | AdaptiveCoherence | Adaptive − Mincut |
|---|---|---|---|---|
| 1.00 | 29.80% | 38.55% | 40.05% | +1.50 pp |
| 0.50 | 33.00% | 38.75% | 38.85% | +0.10 pp |
| 0.25 | 40.40% | 43.90% | 42.85% | −1.05 pp |
| 0.10 | 60.25% | 61.00% | 60.80% | −0.20 pp |
| 0.05 | 84.75% | 84.60% | 85.15% | +0.55 pp |

Mean gain at low selectivity (s ≤ 0.10): **+0.18 percentage points**, 95% CI ≈ [−1.6, +2.0]
pp — a precise estimate that excludes the hypothesized +3 pp effect. This is not an
inconclusive "couldn't tell" result: the estimate is tight enough to confidently say the
effect, if any, is far smaller than hypothesized.

The subject-to conditions all passed on their own (no unfiltered regression — Adaptive is
actually +1.50pp better there; latency ratio never exceeds 1.00x; all tests and clippy
green) — but a hypothesis's `then` clause failing determines REJECT regardless.

## Why it likely failed

The most plausible explanation: the filter model used (uniform-random admission per
candidate, independent of cluster membership) doesn't actually produce the uneven local
sparsification the hypothesis assumed. Within a well-separated Gaussian cluster, intra-cluster
cosine similarity stays well above the 0.50 global threshold even after thinning the cluster
uniformly at random — so `GnnMincutReranker`'s global bar wasn't actually collapsing the way
we expected. A filter correlated with cluster/tenant structure (e.g., admitting or excluding
whole sub-clusters, closer to how real ACL/tenant filters behave) might stress the mechanism
differently and is the natural next experiment.

## Limitations

Synthetic Gaussian-cluster data only; no real embedding dataset. Uniform-random filter model,
not correlated with semantic structure. Server-class single-threaded latency only, no edge
measurement. `k_graph=8` held fixed throughout.

## Production relevance

None, on this evidence. `AdaptiveCoherenceReranker` ships as an additive, harmless,
statistically-indistinguishable-from-baseline alternative in `ruvector-gnn-rerank` — not as
a recommended default or a solution to the filtered-retrieval sparsification problem it was
designed to address.

## RuVector ecosystem implications

Extends `ruvector-gnn-rerank` with no breaking change. No Darwin/Flywheel/MetaHarness CLI
infrastructure was found installed for this experiment type in this environment (verified,
not assumed) — this document, its companion ADR-355, and the raw benchmark output serve as
the flywheel record instead. A bounded, manual one-generation/four-candidate parameter
sweep (standing in for Darwin) found no mutation of `AdaptiveCoherenceReranker`'s defaults
that beat the measured parent configuration — the correct Darwin outcome is to keep the
parent unchanged.

## Future direction

Re-run with a cluster-correlated filter model (admit/exclude whole clusters rather than
per-candidate uniform draws) to test whether the original sparsification-collapse intuition
holds under a filter that actually produces uneven local sparsity.

## References

- GNRR, arXiv:2406.11720
- Maniscope, arXiv:2602.15860
- AQR-HNSW, arXiv:2602.21600
- `crates/ruvector-gnn-rerank` source and `docs/adr/ADR-194-gnn-rerank.md`
