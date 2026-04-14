# DDD Bounded Context: EML Operator Domain

## Overview

The EML (Exp-Minus-Ln) domain encapsulates the application of the functionally-complete
`eml(x, y) = exp(x) - ln(y)` operator to vector database optimizations. This domain
integrates with existing Ruvector bounded contexts while maintaining clear boundaries.

## Ubiquitous Language

| Term | Definition |
|------|-----------|
| **EML Operator** | The binary function `eml(x, y) = exp(x) - ln(y)`, proven functionally complete for all elementary functions |
| **EML Tree** | A binary tree where each internal node applies the EML operator to its children's outputs |
| **Leaf Node** | A terminal node in an EML tree that holds a trainable parameter or reads an input variable |
| **Tree Depth** | The maximum path length from root to any leaf; controls expressiveness vs. evaluation cost |
| **Log Quantization** | Applying `ln()` transform before uniform quantization to better preserve non-uniform distributions |
| **Reconstruction Error** | Mean squared difference between original and dequantized vectors; our primary quality metric |
| **Unified Kernel** | A parameterized distance computation that handles all metrics without branching |
| **Score Fusion** | Combining multiple relevance signals (vector, keyword) into a single ranking score |

## Bounded Context Map

```
┌─────────────────────────────────────────────────────────────────┐
│                     EML OPERATOR DOMAIN                        │
│                                                                │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│  │   EML Core       │  │  Log Quantization│  │  Unified     │ │
│  │   Aggregate      │  │  Value Object    │  │  Distance    │ │
│  │                  │  │                  │  │  Service     │ │
│  │  - EmlNode       │  │  - LogQuantized  │  │              │ │
│  │  - EmlTree       │  │  - LogScale      │  │  - compute() │ │
│  │  - EmlTrainer    │  │  - reconstruct() │  │  - batch()   │ │
│  └────────┬─────────┘  └────────┬─────────┘  └──────┬───────┘ │
│           │                     │                    │         │
└───────────┼─────────────────────┼────────────────────┼─────────┘
            │                     │                    │
    ════════╪═════════════════════╪════════════════════╪══════════
    ANTI-CORRUPTION LAYER (trait implementations)
    ════════╪═════════════════════╪════════════════════╪══════════
            │                     │                    │
┌───────────▼─────────┐ ┌────────▼─────────┐ ┌────────▼─────────┐
│  LEARNED INDEX      │ │  QUANTIZATION    │ │  DISTANCE        │
│  CONTEXT            │ │  CONTEXT         │ │  CONTEXT         │
│                     │ │                  │ │                  │
│  LearnedIndex trait │ │  QuantizedVector │ │  distance()      │
│  RecursiveModelIndex│ │  ScalarQuantized │ │  batch_distances │
│  HybridIndex        │ │  ProductQuantized│ │  DistanceMetric  │
└─────────────────────┘ └──────────────────┘ └──────────────────┘
```

## Aggregates

### 1. EmlTree (Aggregate Root)

The central entity — a binary tree of EML operator nodes with trainable parameters.

```rust
// Value Objects
enum EmlNode {
    Leaf(LeafNode),
    Internal(InternalNode),
}

struct LeafNode {
    kind: LeafKind,  // Input(index) | Constant(f32)
}

struct InternalNode {
    left: Box<EmlNode>,
    right: Box<EmlNode>,
}

// Aggregate Root
struct EmlTree {
    root: EmlNode,
    depth: usize,
    num_params: usize,
}

// Domain Methods
impl EmlTree {
    fn evaluate(&self, inputs: &[f32]) -> f32;
    fn gradient(&self, inputs: &[f32]) -> Vec<f32>;
    fn train(&mut self, data: &[(Vec<f32>, f32)], config: &TrainConfig);
}
```

**Invariants**:
- Tree depth must be ≤ 8 (paper shows diminishing returns beyond depth 6)
- All leaf constants must be finite (no NaN/Inf)
- Internal nodes always apply `eml(left, right) = exp(left) - ln(right)`

### 2. LogQuantized (Value Object)

An immutable quantized representation of a vector using logarithmic scaling.

```rust
struct LogQuantized {
    data: Vec<u8>,        // Quantized values in log space
    offset: f32,          // Shift to make all values positive
    log_min: f32,         // ln(min + offset) for dequantization
    log_scale: f32,       // Scale factor in log space
    dimensions: usize,
}
```

**Invariants**:
- `offset` ensures all shifted values are > 0 (required for ln)
- `log_scale > 0`
- `data.len() == dimensions`

## Domain Services

### UnifiedDistanceService

Stateless service that computes distances using a pre-configured parameter set.

```rust
struct DistanceParams {
    needs_dot: bool,
    needs_sq_diff: bool,
    needs_abs_diff: bool,
    needs_norms: bool,
    combine: fn(dot: f32, sq_diff: f32, abs_diff: f32, norm_a: f32, norm_b: f32) -> f32,
}

impl UnifiedDistanceService {
    fn from_metric(metric: DistanceMetric) -> Self;
    fn compute(&self, a: &[f32], b: &[f32]) -> f32;
    fn batch(&self, query: &[f32], vectors: &[Vec<f32>]) -> Vec<f32>;
}
```

### EmlTrainingService

Domain service that trains EML trees on data samples.

```rust
impl EmlTrainingService {
    fn train_for_quantization(samples: &[f32]) -> EmlTree;
    fn train_for_cdf(data: &[(Vec<f32>, usize)]) -> EmlTree;
    fn train_for_scoring(relevance_data: &[(f32, f32, f32)]) -> EmlTree;
}
```

## Context Integration Points

### With Quantization Context
- `LogQuantized` implements `QuantizedVector` trait (ACL)
- Sits alongside `ScalarQuantized`, `Int4Quantized` in the quantization tier hierarchy
- Used for **warm data** tier where scalar quantization currently lives (4x compression, better fidelity)

### With Distance Context
- `UnifiedDistanceService` wraps `distance()` function with parameter precomputation
- `batch_distances()` can use unified kernel for improved throughput
- SIMD intrinsics remain unchanged — unified kernel calls the same SIMD primitives

### With Learned Index Context
- `EmlModel` implements a superset of `LinearModel` behavior
- `RecursiveModelIndex` can use either `LinearModel` or `EmlModel` for leaf models
- Backward compatible: linear models are a special case of EML trees (depth 1)

### With Hybrid Search Context
- EML score fusion replaces the linear `alpha * a + (1-alpha) * b` combination
- Configurable via `HybridConfig` extension
- Falls back to linear fusion when EML is not configured

## Domain Events

| Event | Trigger | Consumer |
|-------|---------|----------|
| `QuantizationCompleted` | Vector quantized with LogQuantized | Storage layer (persist) |
| `EmlTreeTrained` | Training converges or max iterations reached | LearnedIndex (update model) |
| `DistanceKernelSelected` | Metric configured for a collection | HNSW index (cache kernel) |

## Testing Strategy

- **Unit tests**: EML operator correctness, tree evaluation, gradient computation
- **Property tests**: `reconstruct(quantize(v))` error bounded for any distribution
- **Integration tests**: EML models produce valid search results via VectorIndex trait
- **Benchmark tests**: Comparative performance vs. linear baselines
