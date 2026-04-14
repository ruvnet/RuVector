# ADR-033: EML Operator-Inspired Optimizations

## Status

Accepted

## Date

2026-04-14

## Context

The paper ["All elementary functions from a single operator"](https://arxiv.org/html/2603.21852v2) by Andrzej Odrzywołek demonstrates that the binary operator `eml(x, y) = exp(x) - ln(y)` is *functionally complete* for all elementary mathematical functions when combined with the constant 1. This is analogous to how NAND gates are universal for Boolean logic.

Ruvector currently uses:
- **Linear scalar quantization** (uniform mapping from min-max range to uint8), which assumes uniform value distributions
- **Per-metric SIMD dispatch** via `match` on `DistanceMetric` enum, causing branch prediction overhead in tight loops
- **Linear models** in Recursive Model Indexes (RMI), which cannot capture non-linear CDFs
- **Weighted linear combination** for hybrid search score fusion

These are all limited to linear approximations of inherently non-linear relationships.

## Decision

We adopt four EML-inspired optimizations:

### 1. Logarithmic Quantization (`LogQuantized`)

**Problem**: Scalar quantization maps values linearly to [0, 255]. Real embedding distributions (especially from transformer models) are non-uniform — values cluster near zero with heavy tails. Linear mapping wastes bits on sparse tail regions.

**Solution**: Apply `ln(x - min + 1)` before uniform quantization and `exp(q) - 1 + min` to reconstruct. This allocates finer granularity where values are dense (near zero) and coarser granularity in sparse tails.

**Expected improvement**: 15-40% reduction in reconstruction error for typical embedding distributions (normal, log-normal, Laplacian).

```
Linear:  |---|---|---|---|---|---|---|---| (uniform bins)
    x:   0.01  0.02  0.05  0.1  0.2  0.5  1.0  2.0

Log:     |--|--|--|---|-----|---------|--| (log-spaced bins)
    x:   0.01  0.02  0.05  0.1  0.2  0.5  1.0  2.0
```

### 2. Unified Distance Kernel (`UnifiedDistanceKernel`)

**Problem**: `distance()` dispatches via `match metric { ... }` on every call. During batch operations (HNSW search evaluates thousands of distances per query), this causes branch prediction misses even though the metric never changes within a search.

**Solution**: A branch-free parameterized kernel that encodes the metric as numeric weights:
- `compute_norms: bool` (true for Cosine, false for others)
- `use_abs_diff: bool` (true for Manhattan)
- `negate_result: bool` (true for DotProduct)

The kernel computes all components in a single pass (dot product + squared diff + abs diff), then combines them with the metric-specific weights — no branches in the hot loop.

**Expected improvement**: 5-15% throughput improvement for batch distance operations by eliminating branch overhead.

### 3. EML Tree Learned Index (`EmlModel`)

**Problem**: RMI uses `LinearModel` for CDF approximation. Real data distributions have non-linear CDFs (e.g., clustered embeddings). Linear models produce high prediction errors, requiring large error bounds and expensive binary search fallbacks.

**Solution**: Replace `LinearModel` with `EmlModel` — a small binary tree of `eml(x, y) = exp(x) - ln(y)` nodes with trainable leaf parameters. The EML tree can approximate any elementary function (proven in the paper), including sigmoid, polynomial, and logistic CDFs that commonly arise in vector databases.

**Training**: Gradient descent on leaf parameters using the chain rule through EML nodes. The tree depth is bounded (typically 3-5), keeping evaluation cost O(2^d) where d is depth.

**Expected improvement**: 30-60% reduction in average prediction error, reducing binary search fallback range and improving lookup speed.

### 4. EML Score Fusion for Hybrid Search

**Problem**: Hybrid search combines vector similarity and BM25 keyword scores via `alpha * vector_score + (1 - alpha) * bm25_score`. This linear combination cannot capture non-linear interactions (e.g., "keyword match matters more when vector similarity is moderate").

**Solution**: Use a parameterized EML tree as the score fusion function: `eml_score(vector_sim, bm25) = exp(a * vector_sim + b) - ln(c * bm25 + d)`. The parameters (a, b, c, d) are tunable, and the EML structure naturally handles the log-linear relationships common in information retrieval.

**Expected improvement**: Better ranking quality for hybrid queries, measurable via recall@k on mixed workloads.

## Consequences

### Positive
- Lower reconstruction error means better search recall at the same compression ratio
- Branch-free distance kernels scale better with batch size
- Non-linear learned indexes reduce fallback costs
- All optimizations are backward-compatible (new types, not replacements)

### Negative
- `LogQuantized` adds `ln()/exp()` overhead to quantize/reconstruct (mitigated by doing this once at insert time)
- EML tree evaluation is ~3x slower than linear model evaluation per prediction (offset by fewer binary search iterations)
- Additional code complexity and testing surface

### Risks
- Logarithmic quantization requires `x > 0` after offset — negative values need the `ln(|x| + 1) * sign(x)` symmetric variant
- EML tree training may converge to local minima — mitigated by initializing from linear model solution

## Module Structure

```
crates/ruvector-core/src/advanced/
  eml.rs          -- Core EML operator, tree, evaluation, training
  mod.rs          -- Updated to export eml module

crates/ruvector-core/src/quantization.rs
  LogQuantized    -- Added alongside existing quantization types

crates/ruvector-core/src/distance.rs
  unified_*       -- Unified kernel functions added

crates/ruvector-core/benches/
  eml_bench.rs    -- Comparative benchmarks
```

## References

- Odrzywołek, A. (2025). "All elementary functions from a single operator." arXiv:2603.21852v2
- Ruvector ADR-001: Core Architecture (quantization tiers)
- Ruvector ADR-003: SIMD Optimization Strategy
