# Why a Coherence Gate on a Speculative ANN Draft Hit a Cliff Instead of a Tradeoff

## Problem

`ruvector-speculative-ann` runs approximate nearest-neighbor search the way
speculative decoding runs LLM inference: a cheap "draft" proposes candidates,
an exact "verify" corrects them. Its draft stage is a brute-force scan over
quantized (u8) vectors — O(n) per query. The crate's own research notes
flagged the obvious next step: use a graph instead of a linear scan for the
draft, and prune the graph traversal with a coherence gate borrowed from
`ruvector-coherence-hnsw` (skip expanding a candidate's neighbors when it
lies "off path" from the query).

## Hypothesis

Replacing the linear draft with a coherence-gated graph beam traversal
should cut draft cost by at least 10% while keeping recall within 2
percentage points of an ungated graph draft, and staying faster than an
exact linear scan.

## What We Built

A new Rust crate, `ruvector-coherence-speculative-ann`, with three variants
behind a shared trait: `LinearFull` (exact baseline), `GraphDraftSpeculative`
with an ungated graph draft, and the same with a coherence-gated draft at 13
different thresholds. The graph stores both exact f32 vectors (for the final
verify) and u8-quantized copies (for cheap per-hop ranking during the draft),
matching the lossy-draft/exact-verify shape that makes the original protocol
"speculative."

## What We Found

Graph-based drafting alone is a clean win: ~54 distance evaluations per
query against a 4,000-vector corpus, instead of 4,000 for a linear scan — a
~74x reduction — with 0.998 recall@10 against the exact baseline and 3-4x
lower wall-clock latency.

Adding the coherence gate on top did not produce the hoped-for tradeoff
curve. It produced a cliff:

| Threshold | Recall@10 | Draft cost (vs. ungated) |
|---|---|---|
| 0.00 – 0.50 | 0.998 | within noise (0 to -0.4%) |
| 0.52 | 0.985 | -1.8% |
| **0.55** | **0.875** | **-12.9%** |
| 0.60 | 0.623 | -39% |
| 0.80 | 0.080 | -92% |

Every threshold that reduces draft cost by a measurable amount also breaks
the recall budget; every threshold that preserves recall does nothing
measurable to draft cost. There is no middle.

## Why

We traced this with a small diagnostic binary that prints per-query
pops/expansions directly. The draft's beam search pops candidates in
increasing distance order, so by the time a candidate is actually visited,
it is already close to the query — and a point close to the query is nearly
always "on path" (high coherence) relative to a single fixed, distant entry
point. Low thresholds therefore gate almost nothing among the ~54 candidates
the search actually touches, even though the *population* of all 4,000
points has real, non-degenerate coherence spread (we checked). Once the
threshold is high enough to start rejecting some of those already-near-query
candidates too, it rejects exactly the ones the search needs to converge,
and recall falls off a cliff rather than degrading gracefully.

## Limitations

Tested at one corpus size (n=4,000), one dimensionality (64), one fixed
entry point, and fixed (non-adaptive) thresholds only.

## Production Relevance

None in this form — the mechanism is rejected as implemented. The clean
sub-result (graph drafting without the gate) is worth carrying into
`ruvector-speculative-ann` on its own, independent of this gating question.

## RuVector Ecosystem Implications

This result narrows, rather than closes, the search space: a fixed
single-entry, fixed-threshold coherence gate is a dead end for this use
case. `ruvector-coherence-hnsw` already has an answer worth trying next —
`AdaptiveCoherenceSearch`, whose threshold rises and falls with beam
convergence instead of staying fixed — which was out of scope for this run
but is the natural next experiment.

## Future Direction

1. Swap in the adaptive (beam-convergence-conditioned) gate and re-run this
   exact benchmark.
2. Re-run at n=10,000 to see if the cliff's location moves with scale.
3. Try multiple randomized entry points to see if that removes the
   "near-query implies coherent" degeneracy that caused the cliff.

## References

- `ruvector-speculative-ann` nightly research, 2026-07-27 (source of this
  run's hypothesis).
- `ruvector-coherence-hnsw` nightly research (source of the coherence metric
  and the unused adaptive-gate variant).
- Leviathan, Kalman, Matias. "Fast Inference from Transformers via
  Speculative Decoding." ICML 2023.
- Malkov, Yashunin. "Efficient and Robust Approximate Nearest Neighbor
  Search Using HNSW Graphs." IEEE TPAMI 2018.

Full benchmark methodology, raw output, and the ADR recording this decision:
see `docs/research/nightly/2026-10-21-coherence-graph-draft-speculative-ann/README.md`
and `docs/adr/ADR-353-coherence-gated-graph-draft-for-speculative-ann.md` in
the `ruvnet/ruvector` repository.
