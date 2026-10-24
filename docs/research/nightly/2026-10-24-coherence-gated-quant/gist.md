# When a min-cut gate can't see a density gradient: a negative result on coherence-gated quantization

## Problem

RuVector ships two separable primitives that look like they should
compose: `ruvector-attn-mincut`, a dynamic min-cut gate originally built as
an alternative to softmax attention, and `ruvector-turboquant`, a 4-bit
(Turbo4) / 1-bit vector quantization codec. The idea: use the min-cut
gate's notion of "which edges survive a cheap graph cut" as a per-vector
*coherence score*, and spend quantization precision accordingly — full
Turbo4 for structurally important (low-coherence, boundary) vectors, cheap
1-bit codes for redundant (high-coherence, interior) ones. If it worked,
it would be free leverage: no new coherence metric, just a new consumer of
an existing scored graph.

## Hypothesis

Given a clustered agent-memory corpus (40 Gaussian clusters × 150 vectors,
dim 128), allocating Turbo4-vs-1-bit precision by mincut-derived local
coherence (candidate_A) should beat uniform Turbo4 (baseline) on
recall@10-per-byte by ≥15%, without recall@10 dropping below 0.9× baseline
or query latency growing past 2× baseline.

## Technical design

For each cluster, build a directed self-similarity graph (cosine
similarity as edge weight, diagonal removed), order nodes so index 0 is
closest to the cluster centroid and index `m-1` is farthest, and call the
unmodified, public `ruvector_attn_mincut::dynamic_min_cut(logits, m, λ=0.5,
τ=2, ε=0.01)`. Coherence is `kept_incident_edges(i) / total_incident_edges(i)`
from the returned edge-level `keep_mask`. The bottom 30% least-coherent
vectors per cluster keep Turbo4 (`candidate_A`); the rest drop to 1-bit.
`candidate_B` additionally exact-f32-caches the bottom 5%.

## Actual implementation

New crate `ruvector-coherence-quant`, depending on `ruvector-attn-mincut`
and `ruvector-turboquant` as path dependencies — no reimplementation of
either. Deterministic SplitMix64-seeded dataset generator, `Turbo4Codec::
encode_dual` for one-pass Turbo4+1-bit encoding, brute-force exact L2
ground truth over the full corpus (never approximated), and exact-rank
(not percentile-value) selection for the precision split, after an
earlier percentile-value version was caught by a unit test silently
selecting 100% of vectors under tied coherence scores — itself a preview
of the main finding below.

## Actual benchmark evidence

```
variant                                     bytes/vec  t4_frac  b1_frac exact_frac  encode_ms    recall@10    mean_us     p95_us
baseline (uniform Turbo4)                        72.0     1.00     0.00      0.00       24.5        0.657      252.5      283.0
candidate_A (coherence-gated)                    38.4     0.30     0.70      0.00       24.5        0.301      487.9      545.0
candidate_B (coherence-gated + residual)         65.7     0.30     0.70      0.05       23.8        0.280      534.7      645.0

recall@10 per byte: baseline=0.009118 candidate_A=0.007839 (-14.0%) candidate_B=0.004261 (-53.3%)
ACCEPTANCE RESULT: REJECT
```

`mean coherence score: 1.0000` — exactly uniform across all 6000 vectors.
That's the signature of total signal collapse, not noise.

## Root cause

Two toy graphs run directly against the unmodified `dynamic_min_cut`
settle it:

| Toy graph | `cut_cost` | Gate fired? |
|---|---|---|
| 6-node complete digraph, uniform weight | `0` | No |
| Two 3-cliques + one weight-0.02 bridge | `0.02` (exactly the bridge) | Yes |

The mechanism is correct — it finds the true min-cut and reports its cost
when a cut is worth applying. `GatingResult.cut_cost` reports `0`
specifically when the gate *declines* to cut because the true min-cut
exceeds `λ·mean_weight`. A single unimodal Gaussian cluster is, as a
similarity graph, exactly that "no cheap separation" case — confirmed on
the real corpus (one 150-point cluster: 22,350/22,500 possible edges
present, mean cosine weight 0.91, true min-cut above the λ=0.5 threshold
of 0.45) and unchanged under a Gaussian-RBF edge-weight reparameterization
or a λ sweep from 0.1 to 0.8. Min-cut finds *between-community* boundaries;
a center-to-fringe density gradient inside one blob isn't a community
boundary, so no λ or edge-weight choice can expose it. This is a structural
limitation, not a tuning problem.

## Limitations

Synthetic unimodal-per-cluster corpus only; real embedding topics might be
more bimodal (where this mechanism demonstrably works, per the bridge toy
case) or more unimodal (where it would fail the same way again). Ranking
Turbo4/1-bit/exact distance estimates jointly, as this prototype does, has
unquantified cross-method calibration bias independent of the coherence
problem.

## Production relevance

None yet — not promoted. The uniform-Turbo4 baseline (RuVector's existing
production datatype) is unchanged. The useful output is the falsified
direction plus its root cause, so a future attempt doesn't re-spend a night
rediscovering that `dynamic_min_cut`'s gate needs real community structure
to fire.

## RuVector ecosystem implications

This is evidence *against* assuming `ruvector-attn-mincut`'s gate is
already the general-purpose "coherence/importance" primitive a future
agent-OS-style architecture (shared across attention, memory admission,
and quantization) would want. It's a correct, useful primitive for its
original purpose and for genuinely multi-community graphs; it is not
(without a real fix) a continuous local-density scorer.

## Future direction

Retest the identical harness on a synthetic bimodal corpus to confirm the
mechanism fires exactly where predicted; if it does, retest on a real
embedding distribution to see which regime real topic clusters actually
fall into; independently, test a plain local-density coherence score
(k-NN average distance) to decouple "is coherence-gated quantization
useful" from "is min-cut the right coherence source." Also file a doc fix
against `ruvector-attn-mincut` clarifying `GatingResult.cut_cost`'s
"reports 0 when ungated" semantics, independent of whether this idea is
ever revived.

## References

ADR-296/297 (Turbo4, 1-bit plane), `ruvector-attn-mincut` source
(`mincut.rs`, `gating.rs`), Dinic's algorithm (1970), and this run's own
companion documents: `docs/research/nightly/2026-10-24-coherence-gated-quant/README.md`
and `docs/adr/ADR-353-coherence-gated-quantization.md`.
