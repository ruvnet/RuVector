# Coherence-Adaptive Quantization: A Negative Result

## Problem

Vector indexes that quantize every stored vector at the same bit-width waste precision on
"redundant" vectors (deep inside a dense semantic cluster, cheaply reconstructable from their
neighbours) and under-serve "sensitive" ones (near a cluster boundary, where quantization noise
is more likely to flip a nearest-neighbour ranking). If a cheap, local structural signal could
tell the two apart, an index could spend its bit budget where it matters.

## Hypothesis

**Mutual k-NN coherence** — the fraction of a vector's k nearest neighbours for which the
relationship is mutual (they also list it among their own k nearest neighbours) — was tested as
that signal. High mutuality was hypothesised to mark a "safe to compress" cluster core; low
mutuality was hypothesised to mark a "needs precision" boundary/bridge point.

## Technical design

A new crate, `ruvector-coherence-quant`, implements:

1. A brute-force k-NN graph and the mutual-kNN coherence score per vector.
2. Per-vector min-max scalar quantization to 4 or 8 bits, with real bit-packed storage (two 4-bit
   codes share one byte — memory numbers reflect actual serialized size).
3. Three index variants: uniform 8-bit (baseline), uniform 4-bit (candidate A), and
   coherence-adaptive 4-bit-core/8-bit-boundary (candidate B, threshold 0.5).
4. A benchmark measuring recall@10, memory, and search latency over 300 held-out (jittered)
   queries against a 4,000-vector, 32-dimensional, 12-cluster synthetic corpus.

## Actual benchmark evidence

```
baseline_uniform_8bit           recall=0.9887  mem=156KB  8.00 bits/dim
candidate_A_uniform_4bit        recall=0.8583  mem= 93KB  4.00 bits/dim
candidate_B_coherence_adaptive  recall=0.8770  mem=116KB  5.47 bits/dim  (63.3% core, 36.7% boundary)
```

Pre-registered acceptance gates (fixed before the benchmark ran):

- Uniform 4-bit must show a real recall gap vs baseline (>1.5pp) — **passed** (gap = 13.0pp).
- Candidate B must close that gap to within 1.5pp of baseline — **failed** (gap = 11.2pp; only
  1.9pp of the 13.0pp gap closed, despite giving over a third of the corpus double the bits).
- Candidate B memory must be <=65% of baseline — **failed** (74.7%).

**Verdict: REJECT.** Reproducible across independent runs (identical recall/memory across two
runs; latency varies a few percent as expected for wall-clock timing).

## Why it likely failed

Mutual-kNN coherence is a real, reproducible signal — it is measurably higher on clustered data
than on uniformly random point clouds (verified by a dedicated unit test). But it measures *local
density agreement*, not *quantization sensitivity*. What actually determines how much a vector's
top-10 ranking degrades under min-max scalar quantization is the shape of its own coordinate
distribution (how much a `range/15` step distorts it) and its absolute distance to its true
nearest neighbours — neither of which mutual-kNN mutuality captures. A vector can sit in a
densely-agreed-upon neighbourhood and still quantize badly, or sit at a boundary and quantize
fine, if its coordinate range happens to be small.

This mirrors a prior RuVector nightly result (ADR-303, entropy-adaptive beam search): a
graph/distributional signal that is intuitively appealing and *does* correlate with something real
does not automatically correlate with the specific downstream quantity it was hypothesised to
predict.

## Limitations

PoC scale only (N=4,000, dim=32, brute-force O(n²) k-NN graph). Only min-max scalar quantization
was tested, not product or binary quantization. Held-out queries are synthetic jittered corpus
points, not real agent-memory query traffic.

## Production relevance

None recommended at this time — the mechanism is rejected as tested. The general problem (agent
memory needs content-aware compression under growing corpus size and edge memory ceilings) remains
open and worth revisiting with a quantization-error-derived signal instead of a neighbourhood-
mutuality signal, or with the production `ruvector-mincut` conductance API rather than the
lightweight proxy used here.

## RuVector ecosystem implications

The falsified mechanism does not touch RVF, RVM, or the write path; nothing changes for those
subsystems. `ruvector-mincut`'s conductance primitives remain untested for this use case (this PoC
deliberately used a cheaper, evaluator-independent proxy first) and are the recommended next probe
if the general direction is revisited.

## Future direction

Test a quantization-error-derived per-vector signal (per-dimension coordinate range or local
intrinsic dimensionality) as a direct measurement of what breaks under quantization, rather than a
neighbourhood-structure proxy for it.

## References

- Crate: `crates/ruvector-coherence-quant`
- ADR-305
- Prior related negative result: ADR-303 / `docs/research/nightly/2026-08-13-entropy-adaptive-ann`
