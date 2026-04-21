# RuVector Quantization Benchmark: All Methods Compared

## Summary

First comprehensive benchmark of **all quantization methods** in RuVector, plus TurboQuant, on standard vector search datasets. Tests 8 configurations across 3 datasets, measuring recall, compression, and latency. Each configuration is run 3 times with independent seeds to report variance.

The results are surprising. Several ruvector-core quantization tiers underperform expectations, and no quantization method is actually used during HNSW search.

## Architecture Finding

RuVector has two disconnected quantization subsystems that don't communicate with each other or with HNSW search:

```
crates/ruvllm/src/quantize/turbo_quant.rs     TurboQuant (1,483 lines, SIMD)
                                               └── TurboQuantEmbeddingStore: linear scan only

crates/ruvector-core/src/quantization.rs       ScalarQuantized, Int4, Product, Binary
                                               └── Never called during HNSW search

crates/ruvector-core/src/index/hnsw.rs         HNSW index
                                               └── f32 vectors only, no quantized distance
```

**The `QuantizedVector` trait has 4 implementations with distance functions that are never called during graph traversal.** All quantization in ruvector-core is storage-only. This affects every quantization method, not just TurboQuant.

## Results

All methods pre-reconstruct to f32 before search for fair latency comparison. Values with ± show standard deviation across 3 trials with independent random seeds.

### GloVe d=200 (100,000 vectors, 1,000 queries)

| Method | Origin | R@1 | R@10 | R@100 | Compress | p50 ms | Mem MB |
|--------|--------|-----|------|-------|----------|--------|--------|
| f32 baseline | -- | 1.000 | 1.000 | 1.000 | 1.0x | 2.77±0.08 | 80.0 |
| scalar int8 | ruvector-core | 0.997 | 0.993 | 0.994 | 4.0x | 2.85±0.21 | 20.8 |
| **int4** | **ruvector-core** | **0.912** | **0.904** | **0.917** | **8.0x** | **2.88±0.09** | **20.8** |
| **TQ MSE 4-bit** | **ruvllm** | **0.896** | **0.903±0.003** | **0.917** | **8.0x** | **3.09±0.24** | **10.0** |
| TQ MSE 3-bit | ruvllm | 0.820±0.007 | 0.826±0.003 | 0.845 | 10.7x | 2.87±0.14 | 7.5 |
| TQ full (QJL) | ruvllm | 0.661±0.005 | 0.680 | 0.685 | 10.7x | 27.26±0.95 | 7.7 |
| binary | ruvector-core | 0.514 | 0.503 | 0.498 | 32.0x | 2.82±0.10 | 2.5 |
| product_quant 8sub | ruvector-core | 0.182 | 0.205 | 0.269 | 100.0x | 2.88±0.14 | 1.0 |

### SIFT d=128 (100,000 vectors, 1,000 queries)

| Method | Origin | R@1 | R@10 | R@100 | Compress | p50 ms | Mem MB |
|--------|--------|-----|------|-------|----------|--------|--------|
| f32 baseline | -- | 1.000 | 1.000 | 1.000 | 1.0x | 1.95±0.31 | 51.2 |
| scalar int8 | ruvector-core | 0.986 | 0.989 | 0.993 | 4.0x | 1.47 | 13.6 |
| int4 | ruvector-core | 0.750 | 0.850 | 0.902 | 8.0x | 1.49 | 13.6 |
| TQ MSE 4-bit | ruvllm | 0.448±0.032 | 0.577±0.036 | 0.691±0.035 | 8.0x | 1.50±0.04 | 6.4 |
| TQ MSE 3-bit | ruvllm | 0.283±0.018 | 0.411±0.015 | 0.548±0.019 | 10.7x | 1.49±0.02 | 4.8 |
| TQ full (QJL) | ruvllm | 0.168±0.012 | 0.249±0.007 | 0.377±0.003 | 10.7x | 13.97±0.12 | 5.0 |
| product_quant 8sub | ruvector-core | 0.081 | 0.189 | 0.338 | 64.0x | 1.49±0.02 | 0.9 |
| binary | ruvector-core | 0.000 | 0.000 | 0.003 | 32.0x | 1.41 | 1.6 |

### PKM d=384 (117 vectors, 20 queries)

| Method | Origin | R@1 | R@10 | R@100 | Compress | p50 ms | Mem MB |
|--------|--------|-----|------|-------|----------|--------|--------|
| f32 baseline | -- | 1.000 | 1.000 | 1.000 | 1.0x | 0.01 | 0.2 |
| product_quant 8sub | ruvector-core | 1.000 | 1.000 | 1.000 | 192.0x | 0.01 | 0.2 |
| scalar int8 | ruvector-core | 0.950 | 0.990 | 1.000 | 4.0x | 0.01 | 0.0 |
| int4 | ruvector-core | 0.900 | 0.960 | 0.991 | 8.0x | 0.01 | 0.0 |
| TQ MSE 4-bit | ruvllm | 0.900 | 0.955±0.008 | 0.994±0.001 | 8.0x | 0.01 | 0.0 |
| TQ MSE 3-bit | ruvllm | 0.900 | 0.932±0.006 | 0.989 | 10.7x | 0.01 | 0.0 |
| binary | ruvector-core | 0.800 | 0.805 | 0.963 | 32.0x | 0.01 | 0.0 |
| TQ full (QJL) | ruvllm | 0.817±0.085 | 0.880±0.004 | 0.979±0.003 | 10.7x | 0.40 | 0.0 |

## Key Findings

### 1. Int4 Beats TurboQuant MSE on Recall at Same Compression

At 8x compression on GloVe, naive min-max Int4 achieves 91.2% R@1 vs TurboQuant MSE 4-bit at 89.6%. The Hadamard rotation + Lloyd-Max codebook does not outperform simple linear scaling for nearest-neighbor recall.

With fair pre-reconstruction, **latency is equivalent** (~2.88ms vs ~3.09ms). The previously reported 6x speed advantage was an artifact of reconstructing Int4 per-query inside the timing loop while TurboQuant pre-reconstructed once. Correcting this eliminates the speed difference for brute-force search.

At d=128 (SIFT), Int4 dominates more strongly: 75.0% R@1 vs 44.8% for TQ MSE 4-bit.

### 2. QJL Hurts Recall Across All Datasets

| Dataset | MSE-only R@1 | Full (QJL) R@1 | Loss |
|---------|-------------|----------------|------|
| GloVe d=200 | 0.820±0.007 | 0.661±0.005 | -19.4% |
| SIFT d=128 | 0.283±0.018 | 0.168±0.012 | -40.6% |
| PKM d=384 | 0.900 | 0.817±0.085 | -9.2% |

QJL provides unbiased inner product estimation, but its variance shuffles near-neighbor rankings. For top-k retrieval, lower variance beats unbiasedness. This finding, previously observed for KV cache (softmax amplifies variance), extends to vector search. The high variance on PKM (±0.085 R@1) confirms QJL's instability at small dataset sizes.

### 3. Product Quantization Fails at d=200 with 8 Subspaces

ProductQuantized with 8 subspaces and 256 centroids achieves only 18.2% R@1 on GloVe (d=200). Each subspace is 25-dimensional with 256 centroids, which is insufficient to capture the distribution. The ruvector-core documentation lists PQ as "8-16x compression" for "cold data," but at d=200 it performs worse than random at practical compression ratios.

PQ works well on PKM (d=384, 117 vectors) because the small dataset allows the codebooks to memorize the data. This is not generalizable.

Note: other PQ configurations (more subspaces, OPQ rotation) could perform better. This benchmark tests the ruvector-core default configuration.

### 4. Binary Quantization: Near-Random on GloVe, Zero on SIFT

Binary quantization (sign-bit) achieves 51.4% R@1 on GloVe (barely above random for cosine on normalized vectors) and 0.0% R@1 on SIFT (L2-metric data where sign bits lose all magnitude information). The ruvector-core documentation suggests binary for "archive (<1% access)" data, which is appropriate, but the extreme quality loss should be documented.

### 5. TurboQuant's Real Niche: The 3-bit Tier

No ruvector-core method exists between int4 (4 bits, 8x) and binary (1 bit, 32x). TurboQuant MSE 3-bit fills this gap at 10.7x compression with 82.0±0.7% R@1 on GloVe. This is the strongest argument for integrating TurboQuant into ruvector-core: it occupies an unserved compression tier.

### 6. Quality Scales with Dimension

TurboQuant performs better at higher dimensions, consistent with theory (Beta distribution concentration):

| Dimension | TQ MSE 4-bit R@1 | Int4 R@1 | TQ vs Int4 |
|-----------|-------------------|----------|------------|
| d=128 (SIFT) | 0.448±0.032 | 0.750 | -40.3% |
| d=200 (GloVe) | 0.896 | 0.912 | -1.8% |
| d=384 (PKM) | 0.900 | 0.900 | Tied |

At d=384+, TurboQuant matches naive Int4. Modern embedding models (384-1536 dim) are in TurboQuant's effective range. Below d=200, TurboQuant is not competitive.

## Recommendations

### Tier 1: Quantization Comparison Table (Corrected)

The current ruvector-core documentation states:

```
| Quantization | Compression | Use Case              |
|--------------|-------------|-----------------------|
| Scalar (u8)  | 4x          | Warm data (40-80%)    |
| Int4         | 8x          | Cool data (10-40%)    |
| Product      | 8-16x       | Cold data (1-10%)     |
| Binary       | 32x         | Archive (<1% access)  |
```

Based on benchmarks, a corrected table for d >= 200:

```
| Quantization     | Compression | R@1 (GloVe) | Recommendation               |
|------------------|-------------|-------------|------------------------------|
| Scalar int8      | 4x          | 99.7%       | Default for quality          |
| Int4             | 8x          | 91.2%       | Best recall at 8x            |
| TQ MSE 4-bit     | 8x          | 89.6%       | Alternative at 8x (d >= 200) |
| TQ MSE 3-bit     | 10.7x       | 82.0%       | NEW: fills compression gap   |
| Binary           | 32x         | 51.4%       | Coarse filter only           |
| Product (8 sub)  | 100x        | 18.2%       | Not recommended at d=200     |
```

### Tier 2: HNSW Integration

All quantization methods are storage-only. Bridging `QuantizedVector::distance()` into the HNSW search path would make every method usable for search, not just TurboQuant. This is a ~600-950 line change that benefits the entire quantization stack.

### Tier 3: TurboQuant MSE-Only Integration

Add TurboQuant MSE-only (skip QJL) as a fifth quantization option in ruvector-core. The data-oblivious property (no training, no codebooks) makes it ideal for online vector databases where data arrives incrementally. Its value is the 3-bit compression tier, not speed or recall superiority over Int4.

## Methodology

### Protocol
- All searches are brute-force (no HNSW) to isolate quantization effects
- Vectors L2-normalized, inner product = cosine similarity
- Ground truth: exact f32 inner product per dataset (not pre-computed files)
- All methods pre-reconstruct to f32 before the timing loop (fair latency comparison)
- TQ full: paper's unbiased inner product estimator (not pre-reconstructed)
- Each configuration run 3 times with independent random seeds
- Stochastic methods (TurboQuant, PQ) report mean±std across trials
- Deterministic methods (Int4, Int8, Binary) report latency variance only

### Reproducibility

```bash
pip install torch numpy scipy
python benchmark_quantized_search.py           # All datasets
python benchmark_quantized_search.py glove200  # Single dataset
```

Datasets: GloVe 6B (Stanford NLP), SIFT1M (INRIA Texmex), PKM (anonymized, included as .npy).
Reference: [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch)

### Limitations
- PKM dataset (117 vectors) is too small for meaningful generalization; included for completeness at d=384
- PQ tested with default 8 subspaces / 256 centroids only; optimized PQ variants may perform better
- Binary and Int4 stored at full width during search (theoretical compression ratios reported)
- CPU-only benchmarks; SIMD-optimized implementations would change latency characteristics

## References

- TurboQuant (ICLR 2026): [arxiv.org/abs/2504.19874](https://arxiv.org/abs/2504.19874)
- PolarQuant (AISTATS 2026): [arxiv.org/abs/2502.02617](https://arxiv.org/abs/2502.02617)
- QJL: [arxiv.org/abs/2406.03482](https://arxiv.org/abs/2406.03482)
