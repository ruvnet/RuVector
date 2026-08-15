# Coherence-Adaptive Quantization: Mutual-kNN Boundary Detection for Bit-Width Allocation

**Status**: PoC complete — **negative result** (hypothesis falsified as tested)
**Crate**: [`ruvector-coherence-quant`](../../../../crates/ruvector-coherence-quant)
**ADR**: [ADR-305](../../adr/ADR-305-coherence-adaptive-quant.md)

## Abstract

Uniform scalar quantization applies the same bit budget to every vector in a corpus, regardless
of how sensitive that vector's nearest-neighbour ranking is to precision loss. This experiment
tested whether a **mutual-kNN coherence score** — a lightweight, ground-truth-free proxy for
local graph conductance / cluster-boundary detection, in the spirit of `ruvector-mincut`'s
dynamic min-cut conductance signal but computed directly and cheaply over a k-NN graph — could
drive a per-vector 4-bit/8-bit allocation that recovers most of full-precision recall at close to
aggressive-quantization memory.

**Measured PoC outcome: negative.** On a 4,000-vector, 32-dimensional, 12-cluster synthetic
corpus, coherence-adaptive allocation (63.3% of vectors at 4-bit, 36.7% at 8-bit, mean 5.47
bits/dim) recovered only 12.5pp of the 13.0pp recall gap between uniform 8-bit and uniform 4-bit,
while using 25% more memory than uniform 4-bit. Both the recall-recovery and memory-budget
acceptance gates fail. See "Why it failed" below for the most likely mechanism.

## Hypothesis

```text
Given a 4,000-vector corpus at dimension 32, arranged in 12 semantic clusters (noise=0.18),

when per-vector scalar quantization bit-width is chosen by mutual-kNN coherence
(coherence >= 0.5 -> 4-bit "core"; coherence < 0.5 -> 8-bit "boundary"), instead of a
uniform 4-bit budget,

then recall@10 on held-out (jittered) queries should recover to within 1.5 percentage points
of the uniform 8-bit baseline,

subject to total index memory remaining at or below 65% of the uniform 8-bit baseline, and
subject to uniform 4-bit itself showing a real (>1.5pp) recall gap versus baseline (otherwise
there is no precision-loss problem to fix).
```

This hypothesis was fixed before benchmarking and was not changed after seeing results.

## Why this matters in 2026

Agent-memory and RAG corpora keep growing — long-running agents accumulate memories faster than
operators are willing to pay for full-precision storage. Existing RuVector quantization research
(`ruvector-rabitq`, `ruvector-turboquant`, `ruvector-pq-search`, `ruvector-matryoshka`) all apply
a *uniform* compression policy across the corpus. A working coherence-adaptive policy would let an
index spend its bit budget where it is needed (ambiguous, boundary-region memories) and save it
where it is not (redundant, core-cluster memories) — the same "spend compute/precision where
uncertainty lives" principle behind entropy-adaptive beam search (`ruvector-entropy-ann`,
ADR-303), applied to storage instead of search.

## Why it could matter in 2036 / 2046

Long-horizon agent operating systems and edge/RVM cognitive appliances will hold portable
cognitive state (RVF) under hard memory ceilings. A *correct* content-aware compression signal
— one where perceptual/retrieval importance, not memory-allocation heuristics, determines bit
budget — is a prerequisite for graceful degradation under memory pressure rather than uniform
degradation. This experiment is a small, falsifiable step toward finding whether *local graph
structure* is a valid importance signal for that purpose. As tested, it is not (see below); the
negative result narrows the search space for future attempts.

## Why RuVector is the right substrate

RuVector already owns the two structural primitives this hypothesis composes: a real dynamic
min-cut / conductance implementation (`ruvector-mincut`) and multiple production quantization
codecs. Testing whether a *cheap* structural proxy (mutual-kNN mutuality, no dynamic min-cut
machinery required) captures the same signal is exactly the kind of "new composition of known
techniques, RuVector-specific" experiment the nightly harness should prioritize before committing
engineering effort to wiring the full `ruvector-mincut` conductance API into the quantization
write path.

## Architecture

```mermaid
flowchart LR
    A[Clustered corpus<br/>4000 x 32d] --> B[k-NN graph<br/>k=12, brute force]
    B --> C[Mutual-kNN coherence<br/>score per vector]
    C --> D{coherence >= 0.5?}
    D -->|yes, core| E[4-bit scalar quant]
    D -->|no, boundary| F[8-bit scalar quant]
    E --> G[QuantizedIndex]
    F --> G
    G --> H[Brute-force search<br/>vs held-out queries]
    H --> I[recall@10, memory, latency]
```

## Implementation

New crate `ruvector-coherence-quant`, self-contained (no dependency on `ruvector-mincut` or
`ruvector-core`, to keep the evaluator independent of the candidate and the benchmark fast to
iterate on):

- `dataset.rs` — deterministic LCG-seeded clustered corpus + held-out "jittered" query generator
  (queries are corpus points + extra noise, renormalized — realistic paraphrase-style retrieval,
  not identical to any stored vector) + brute-force ground truth.
- `coherence.rs` — `build_knn_graph` (brute-force k-NN) and `mutual_knn_coherence`: for each
  vector, the fraction of its k nearest neighbours for which the relationship is mutual (v is
  also among that neighbour's k nearest neighbours). This is the structural signal under test.
- `quantize.rs` — per-vector min-max scalar quantization to 4 or 8 bits, with **real bit-packed
  storage** (two 4-bit codes share a byte) so reported memory is the actual serialized size, not
  an approximation.
- `search.rs` — brute-force search over the quantized (dequantized-on-read) index; asymmetric
  distance (raw f32 query vs dequantized corpus vector), matching how production scalar-quantized
  ADC-style indexes scan.
- `bin/benchmark.rs` — orchestrates baseline / candidate A / candidate B, evaluates the
  pre-registered acceptance gates, prints ACCEPT/REJECT/INCONCLUSIVE.

Three variants, matching the harness's required baseline/candidate_A/candidate_B structure:

| Variant | Bit allocation |
|---|---|
| `baseline_uniform_8bit` | every vector, 8-bit |
| `candidate_A_uniform_4bit` | every vector, 4-bit |
| `candidate_B_coherence_adaptive` | 4-bit if mutual-kNN coherence >= 0.5, else 8-bit |

## Benchmark methodology

- Release build (`opt-level = 3`, `lto = "thin"`), `cargo run --release -p ruvector-coherence-quant
  --bin benchmark`.
- Deterministic seeds throughout: corpus seed 42, query seed 777. Two independent runs produced
  bit-identical recall and memory numbers (latency varies by a few percent between runs, as
  expected for wall-clock timing on a shared CPU).
- Corpus and query generation are excluded from the timed search loop; only `QuantizedIndex::search`
  is timed via `LatencyStats::measure`, one call per query.
- Coherence-graph build time (k-NN + mutual scoring) is reported separately as a one-time index-build
  cost, not folded into per-query latency.
- Hardware: `linux/x86_64`, 4 CPU threads available. Rust 1.94.1 / Cargo 1.94.1.

## Results (raw, from the run captured for this document)

```
Dataset: N=4000, dim=32, clusters=12, noise=0.18, k(recall)=10, k(coherence)=12, threshold=0.5
Coherence: mean=0.555 p50=0.583  core(>=0.5)=2532 (63.3%)  boundary=1468 (36.7%)

  baseline_uniform_8bit           recall=0.9887  mean_bits=8.00  mem=156KB  mean=258.3us p95=305.4us  3870 qps
  candidate_A_uniform_4bit        recall=0.8583  mean_bits=4.00  mem= 93KB  mean=399.0us p95=445.2us  2506 qps
  candidate_B_coherence_adaptive  recall=0.8770  mean_bits=5.47  mem=116KB  mean=353.8us p95=403.0us  2826 qps

Acceptance gates:
  1. uniform 4-bit recall gap vs baseline : 0.1303  (need > 0.0150)             PASS
  2. candidate B recall gap vs baseline   : 0.1117  (need <= 0.0150)            FAIL
  3. candidate B memory / baseline memory : 0.747   (need <= 0.65)              FAIL

VERDICT: REJECT
Coherence graph build overhead: 808ms (one-time, amortized at index build)
```

Reproduce with:

```bash
cargo run --release -p ruvector-coherence-quant --bin benchmark
```

## Why it failed

The acceptance gates were designed to distinguish "coherence-adaptive allocation captures the
quantization-sensitivity signal" from "it doesn't." Gate 1 confirms there was a real signal to
capture: uniform 4-bit loses 13.0 percentage points of recall versus 8-bit on this corpus. But
candidate B — despite giving 36.7% of vectors a full doubling of precision — only recovered 1.9pp
of that 13.0pp gap (14% of the gap closed, for 25% more memory than uniform 4-bit).

The most likely explanation: **mutual-kNN coherence measures local density agreement, not
per-vector quantization sensitivity.** A vector's susceptibility to precision loss under min-max
scalar quantization is driven by the *shape* of its value distribution across dimensions (how
much a `range/15` step distorts individual coordinates) and by how close its true neighbours are
in absolute distance (tight clusters are less forgiving of rank-order noise, not more). Neither
of those is what mutual-kNN mutuality measures. A vector can have a highly mutual neighbourhood
(dense, locally-agreed-upon core) and still have coordinate ranges that quantize poorly, or sit at
a distance from its true top-10 where quantization noise dominates ranking regardless of its local
mutuality.

This is analogous to the ADR-303 entropy-adaptive-ANN negative result: a graph/distributional
signal that is intuitively appealing and *does* correlate with something real (mutual-kNN
coherence genuinely is higher in dense clusters than in random point clouds — see
`coherence::tests::clustered_corpus_has_higher_mean_coherence_than_random`) does not automatically
correlate with the specific downstream quantity (quantization-induced recall loss) the hypothesis
needed it to predict.

## What would falsify (and did falsify) the hypothesis

Falsification criterion, set in advance: candidate B fails to close the recall gap to within
1.5pp of baseline, or fails to do so at <=65% of baseline memory. Both conditions occurred. The
hypothesis is falsified as implemented.

## What this does NOT claim

- This does not claim mutual-kNN coherence is useless — the coherence-vs-random test in
  `coherence.rs` confirms it is a real, reproducible structural signal. It claims that signal does
  not transfer to *quantization bit-width allocation* on this workload.
- This does not claim graph-conductance-based bit allocation is impossible — only that the cheap
  mutual-kNN proxy tested here does not deliver it. A direct `ruvector-mincut` conductance query,
  or a signal derived from actual quantization error magnitude (e.g., per-dimension range /
  entropy of the vector's own coordinates) rather than neighbourhood mutuality, remains untested.
- This does not claim uniform quantization is optimal — only that this particular adaptive
  allocation strategy does not beat it on the tested tradeoff.

## RVF / RVM / ruFlo / MCP integration analysis

Given the negative result, no integration is recommended at this time. For completeness:

- **RVF**: a working content-aware bit-allocation policy would be directly portable as index
  metadata inside an RVF cognitive package (per-vector precision map travels with the index). Not
  pursued further given the negative result.
- **RVM**: no coherence domain or proof-gated mutation implications — this is a pure index-layer
  compression policy with no write-path authority changes.
- **ruFlo**: a *working* version of this could become a background "index re-tiering" workflow
  (periodically re-score and re-quantize as cluster structure drifts). Not worth building on top
  of a falsified allocation signal.
- **MCP**: no new surface warranted.

## Edge / WASM analysis

The bit-packing implementation (`quantize.rs`) is already allocation-light and would compile to
WASM without changes; the k-NN coherence graph build is the only O(n²) component and would need a
production ANN-based approximation (self-join via the existing HNSW graph) rather than brute
force before any edge deployment — moot given the negative result.

## Competitor comparison

Not performed. Comparing against Milvus/Qdrant/Weaviate/FAISS mixed-precision quantization
schemes is only meaningful once a working RuVector variant exists to compare; a negative internal
PoC result has nothing to benchmark externally. Documented per Step 35's guidance to avoid
implying a comparison that wasn't measured.

## Practical and long-horizon applications

Deferred: per the harness's own rule (never inflate a rejected candidate's relevance), this
document does not enumerate hypothetical applications for a falsified mechanism. The applications
listed under "Why this matters" above remain valid for the *general problem* (content-aware
quantization for agent memory); they are not claims about this specific mechanism.

## Limitations

- Corpus scale (N=4,000) and dimensionality (32) are PoC-scale; the coherence-vs-quantization
  relationship was not tested at production scale (millions of vectors) or at higher
  dimensionality (384–1536, typical embedding sizes) where quantization error characteristics
  differ.
- The k-NN graph is brute-force (O(n²)); this is fine for a 4,000-vector PoC but would not scale
  without reusing an existing approximate graph (e.g., the corpus's own HNSW graph).
- Only min-max scalar quantization was tested, not product quantization or RaBitQ-style binary
  quantization, where the sensitivity-to-precision relationship may differ.
- Query set is synthetic (jittered corpus points); real agent-memory query distributions may
  differ from this held-out regime.

## Next research

1. Test whether a **quantization-error-derived** per-vector signal (e.g., per-dimension range or
   local intrinsic dimensionality) predicts recall sensitivity better than neighbourhood mutuality
   — a more direct measurement of what actually breaks under quantization.
2. If a working signal is found, integrate it with the real `ruvector-mincut` conductance API
   rather than the standalone mutual-kNN proxy, to validate whether the production-grade
   conductance signal (which captures more than 1-hop mutuality) behaves differently.
3. Retest at higher dimensionality (384+) where per-dimension quantization noise averages out
   differently than at dim=32.

## References

- `ruvector-mincut` crate (`crates/ruvector-mincut`) — subpolynomial dynamic min-cut / conductance.
- ADR-303, `docs/research/nightly/2026-08-13-entropy-adaptive-ann` — prior negative result with a
  structurally similar lesson (a real, measurable graph/distributional signal that does not
  transfer to the specific downstream metric it was hypothesised to predict).
- `ruvector-rabitq`, `ruvector-turboquant`, `ruvector-pq-search`, `ruvector-matryoshka` —
  existing RuVector uniform quantization codecs this experiment intended to complement.
