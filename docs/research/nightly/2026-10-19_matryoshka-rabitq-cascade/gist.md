# Swapping an ANN Funnel's Coarse Stage for a 1-Bit Brute Scan

## Problem

Coarse-to-fine ("Matryoshka") vector search funnels do a cheap search at a
truncated embedding prefix to generate candidates, then an exact rerank at
full dimension to pick the final answer. The usual choice for the cheap
coarse stage is a smaller HNSW graph at the truncated dimension. A graph
needs construction time, needs `4*dim + 8*M` bytes per vector, and — in two
separate pieces of prior work in this codebase — has caused real correctness
and maintenance problems: non-deterministic partition results tied to hash
map iteration order, and delete-repair complexity.

Is a graph actually necessary for the coarse stage, given that its only job
is to produce a *candidate set* that a downstream exact rerank will fix up
anyway?

## Hypothesis

Replace the coarse HNSW graph with a RaBitQ-style 1-bit random-rotation
binary code, scored with a branch-free XNOR-popcount scan instead of graph
traversal. Because the coarse stage's output is never the final answer —
every candidate gets reranked exactly at full precision — RaBitQ's known
weakness as a *final* answer (12-40% recall@10 alone, per this repo's own
earlier RaBitQ study) should not matter here. What should matter is: does it
produce a good enough candidate set, cheaply enough, at a fraction of the
memory?

## Technical design

Three `Searcher` implementations share one dataset and one evaluation
harness:

1. **FullDimHNSW** (baseline) — single HNSW graph at full dimension.
2. **TwoStage** (candidate A, pre-existing) — HNSW graph at the truncated
   coarse dimension, then exact rerank at full dimension.
3. **RaBitQCascade** (candidate B, new) — for each vector, L2-normalise its
   coarse-dimension prefix, apply a Haar-uniform random rotation (built once
   via Gram-Schmidt on an i.i.d. Gaussian block), pack the rotated vector's
   sign bits into `u64` words. At query time, rotate+encode the query the
   same way, score every stored code with the symmetric Charikar-style
   angular estimator (`E[popcount-agreement/dim] = 1 - theta/pi`), take the
   top candidates, rerank those exactly at full dimension.

The rotation decorrelates dimensions so that a 1-bit sign code captures
useful angular information instead of being dominated by whichever raw
dimension happens to have the largest variance — this is RaBitQ's core
contribution (Gao & Long, SIGMOD 2024), reused here, not reinvented.

## Implementation

One new Rust crate, `ruvector-matryoshka-rabitq`, depends on the existing
`ruvector-matryoshka` crate for its `Searcher` trait, dataset generator, and
the two baseline variants, and adds:

- `rabitq_code.rs` (~160 lines): `RandomRotation` + `BinaryCode`, a
  self-contained reimplementation of just the rotation/encode/estimate
  subset this cascade needs (not a dependency on the existing, larger
  `ruvector-rabitq` crate, which also carries an asymmetric estimator,
  Hadamard-rotation variant, and persistence layer this use case doesn't
  need).
- `lib.rs`: `RaBitQCascadeIndex` implementing `Searcher`, plus
  `coarse_stage_bytes()` for exact (not estimated) memory accounting.
- `src/bin/benchmark.rs`: a benchmark binary measuring all three variants
  plus a bounded sweep over the coarse candidate-pool size.

7 unit tests cover rotation norm preservation, the estimator's self-distance
and ordering correctness, non-64-aligned-dimension bit-masking (a subtle bug
class the existing `ruvector-rabitq` crate's own test suite flags
explicitly), and that the coarse-stage memory is actually smaller than the
HNSW equivalent by construction. `cargo clippy --release -- -D warnings`
passes clean.

## Actual benchmark evidence

All numbers below are from real `cargo run --release` executions on this
run's hardware (Linux x86_64, 4 vCPU, `rustc 1.97.0`), reproduced verbatim
from the console. No number is invented.

**N=3,000 vectors, D=128, coarse_dim=32, 200 queries, k=10, seed=42:**

| Variant | Recall@10 | Mean latency (us) | Coarse-stage memory (KB) |
|---|---|---|---|
| FullDimHNSW | 1.000 | 174.3 | 1,875.0 |
| TwoStage (HNSW coarse) | 0.903 | 82.0 | 750.0 |
| **RaBitQCascade (1-bit coarse)** | **0.994** | 92.6 | **27.4** |

RaBitQCascade recovered *more* recall than the HNSW-coarse funnel (0.994 vs.
0.903) while using **27.4x less coarse-stage memory**, at a 13% latency
cost — within the pre-declared 2x acceptance bound. Reproduced at a second
seed (seed=7: 0.991 recall, same 27.4 KB, 99.2us vs. TwoStage's 96.3us).

**The honest failure mode, also measured, not hidden:** at N=20,000, the
brute-force coarse scan costs 1,120us mean versus TwoStage's 150.6us — a
~7.4x latency penalty, because an O(N) scan eventually loses to a
sub-linear graph traversal as N grows. (Separately, at that same N, this
crate's inherited default HNSW parameters also show both HNSW-based
variants' own recall collapsing — a pre-existing limitation in the sibling
crate's defaults, not something this work introduced, and flagged as its own
follow-up rather than patched as a drive-by fix here.) The exact crossover N
between "competitive" and "loses" was not bisected this run.

A bounded, single-generation sweep over the coarse candidate-pool size
(3 points: k*5, k*10, k*20), scored with a declared fitness formula
weighting recall, latency, throughput, memory, correctness, and
reproducibility, kept the existing default (k*10) as the winner — no
promotion, which is the correct outcome when nothing clears the margin.

## Limitations

- Synthetic Gaussian-cluster data, not a real embedding provider's output.
- Single-threaded, no delete/update benchmark — the claim that binary codes
  simplify deletes (no shared graph to repair) is architectural reasoning,
  not a measurement.
- The exact latency crossover point between N=3,000 and N=20,000 is
  unknown; this write-up reports two points, not a curve.
- No `wasm32` build was attempted despite the edge-relevance framing above —
  flagged as unverified rather than implied.

## Production relevance

Recommended as an **opt-in** coarse-stage strategy for memory-constrained or
frequently-rebuilt agent-memory deployments confirmed to sit below the
(currently unbisected) latency crossover — not as a default replacement for
the existing HNSW-coarse funnel.

## RuVector ecosystem implications

Connects three existing capabilities that had no prior shared interface:
Matryoshka-aware funnels (`ruvector-matryoshka`), RaBitQ-style binary
quantization (`ruvector-rabitq`), and agent-memory deployment constraints
(`ruvector-agent-memory`, the `ruvector-edge-*` family). The memory math
here (27x smaller coarse stage) is the kind of number that compounds at the
multi-tenant / edge-fleet scale those crates target, conditional on staying
below the latency crossover this run did not pin down.

## Future direction

Bisect the real crossover N; fix the sibling crate's HNSW recall collapse at
large N as its own dedicated nightly; build a hybrid `Searcher` that
switches strategy by measured N; verify an actual `wasm32` build before
making any edge-deployment claim stronger than the byte-count math above.

## References

- Gao, J. & Long, C. "RaBitQ." SIGMOD 2024. [arXiv:2405.12497](https://arxiv.org/abs/2405.12497)
- Rege, A. et al. "AdANNS." NeurIPS 2023. [arXiv:2305.19435](https://arxiv.org/abs/2305.19435)
- "Matryoshka Hash Representations." alphaXiv 2026-09. [alphaxiv.org/abs/2609.07276](https://alphaxiv.org/abs/2609.07276)
- Nair, P. et al. "Matryoshka Quantization." ICML 2025. [arXiv:2502.06786](https://arxiv.org/abs/2502.06786)
- Elastic Search Labs. "RaBitQ binary quantization 101." [elastic.co/search-labs/blog/rabitq-explainer-101](https://www.elastic.co/search-labs/blog/rabitq-explainer-101)
- Full write-up with raw benchmark logs: `docs/research/nightly/2026-10-19_matryoshka-rabitq-cascade/README.md` in [ruvnet/ruvector](https://github.com/ruvnet/ruvector)
