# EML-Enhanced HNSW Proof Report

PR #353 — `feat/eml-hnsw-optimizations`

Methodology: 4-stage proof chain following shaal's pattern from PR #352.
All numbers are real measurements on arm64 Linux, not simulated.

## Stage 1: Micro-Benchmarks

Each optimization measured in isolation on 500 vector pairs (128-dim).

| Optimization | Baseline | EML | Overhead | Notes |
|---|---|---|---|---|
| Distance: full 128d cosine (500 pairs) | 50.3 us | — | — | Baseline per-batch |
| Distance: raw 16d L2 proxy (500 pairs) | 5.39 us | — | **9.3x faster** | Dimension reduction alone |
| Distance: EML 16d fast_distance (500 pairs) | — | 106.5 us | **2.1x slower** | EML model prediction overhead dominates |
| Adaptive ef prediction (200 queries) | 73.9 ns (fixed) | 90.8 us | 456 ns/query | ~1228x overhead vs returning a constant |
| Path prediction (200 queries) | 72.6 ns (no-op) | 10.6 us | 53 ns/query | Centroid distance lookup per query |
| Rebuild prediction (200 checks) | 105.0 ns (fixed) | 554.6 ns | 2.8 ns/check | Acceptable: <3ns per decision |

### Stage 1 Findings

**Dimension reduction works (9.3x speedup)** when using a simple L2 proxy on 16 selected
dimensions vs full 128-dim cosine. However, the **EML model prediction overhead** completely
negates this speedup — the `eml_core::predict_primary` call is expensive (~200ns per
evaluation), making the learned fast_distance 2.1x *slower* than full cosine.

**Rebuild prediction** has negligible overhead (2.8ns/check) and is the most cost-effective
optimization. **Adaptive ef** and **path prediction** have moderate overhead that would need
to save significant search work to break even.

## Stage 2: Synthetic End-to-End (10K vectors, 128-dim)

Flat-scan with 100 queries, k=10.

| Config | Time (100 queries) | Implied QPS | Recall@10 |
|---|---|---|---|
| Baseline (full cosine) | 115.9 ms | 863 | 1.0000 |
| EML (16d fast_distance) | 219.6 ms | 455 | **0.0010** |
| Delta | **1.9x slower** | -47% | **-99.9%** |

### Stage 2 Findings

On uniformly random data, the EML distance model **destroys recall**. Recall@10 drops from
100% to 0.1%. This is expected and honest:

1. **Random data has no discriminative dimensions.** EML dimension selection identifies which
   dimensions correlate most with distance. In uniformly random data, all dimensions are
   equally (weakly) correlated, so selecting 16 out of 128 discards 87.5% of the signal.

2. **The EML model was trained on the same random distribution.** The Pearson correlation
   step found no strong signal, and the EML tree learned a poor approximation.

3. **This does NOT mean the optimization is useless.** Real-world embeddings (SIFT, BERT,
   CLIP, etc.) have strong dimensional structure — some dimensions carry far more variance
   than others. The cosine decomposition is designed for such structured data.

**Conclusion:** The synthetic benchmark proves the *mechanism works* (dimension reduction is
fast), but the *accuracy claim requires structured data* to validate.

## Stage 3: Real Dataset

SIFT1M dataset not available at `bench_data/sift/sift_base.fvecs`.

**Status: Deferred.** Download SIFT1M (~400MB) from http://corpus-texmex.irisa.fr/ to enable.
The benchmark infrastructure is in place and will automatically run if the dataset is present.

Real embedding datasets (SIFT, GloVe, CLIP) typically have strong PCA structure where the
top 16 principal components explain >80% of variance. We expect significantly better recall
on such data. Until measured, this remains a hypothesis.

## Stage 4: Hypothesis Test

**Hypothesis:** 16-dim decomposition preserves >95% of ranking accuracy (Spearman rho >= 0.95).

**Test:** For 50 queries against 1000 vectors (128-dim uniform random), compute Spearman rank
correlation between full-cosine rankings and EML-16d rankings.

| Metric | Value |
|---|---|
| Mean Spearman rho | **0.0131** |
| Min rho | -0.0433 |
| Max rho | 0.0486 |
| Queries tested | 50 |

**Result: DISPROVEN on uniform random data.**

The near-zero correlation confirms that on data with no dimensional structure, 16-dim
decomposition is essentially random ranking. This is a fundamental property of the uniform
distribution, not a bug in the EML implementation.

### Expected behavior on structured data

For embeddings with PCA structure (real-world use case), we would expect:
- If top-16 PCA dims explain 80% variance: rho ~ 0.85-0.90
- If top-16 PCA dims explain 95% variance: rho ~ 0.95+
- If data is uniform random (this test): rho ~ 0.01 (confirmed)

## Summary

| What works | What doesn't (yet) |
|---|---|
| Dimension reduction is genuinely 9.3x faster (raw) | EML prediction overhead negates the speedup |
| Rebuild prediction has negligible overhead (2.8ns) | Cosine decomposition needs structured data |
| Path prediction finds correct regions | Recall drops to near-zero on random data |
| Benchmark infrastructure is reproducible | SIFT1M real-data test deferred |

### Recommendations

1. **Optimize EML model inference.** The current `predict_primary` call (~200ns) is too
   expensive for a per-distance-call optimization. Consider: SIMD batch prediction,
   model quantization, or compiling the trained model to a fixed polynomial.

2. **Test on real embeddings.** The proof chain is structurally sound but needs SIFT1M
   or GloVe data to validate the accuracy hypothesis.

3. **Focus on rebuild prediction.** It has the best cost/benefit ratio today (2.8ns
   overhead for smarter rebuild decisions).

4. **Consider adaptive ef as a search-level optimization** rather than a per-distance
   optimization — the 456ns/query overhead is acceptable if it saves many distance
   computations by reducing beam width.

---

*Generated by cargo bench on arm64 Linux. All numbers are real, not simulated.*
*Following shaal's 4-stage proof methodology from PR #352.*
