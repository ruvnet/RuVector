# ADR-088: CNN Contrastive Learning Integration for RuVector

## Status

**Proposed**

## Date

2026-03-11

## Context

RuVector requires image embedding capabilities for multimodal vector search applications. This ADR proposes integrating CNN-based contrastive learning with existing RuVector components:

- **ruvector-attention**: Flash Attention, MoE, hyperbolic attention
- **ruvector-math**: Optimal transport, information geometry, pure Rust SIMD
- **ruvector-hyperbolic-hnsw**: Hyperbolic space vector search
- **sona**: Self-Optimizing Neural Architecture (EWC++, LoRA, ReasoningBank)
- **ruvector-gnn**: Graph Neural Networks (complementary, not competing)

### Requirements

| Requirement | Target | Notes |
|-------------|--------|-------|
| Embedding latency | <5ms (MobileNet-V3 Small) | CPU-only, no GPU |
| HNSW search | <1ms @ 1M vectors | Leverages existing hyperbolic-hnsw |
| Model size | <10MB (INT8 quantized) | Edge/WASM deployment |
| WASM support | Required | Pure Rust, no BLAS/native deps |
| Training | Contrastive (SimCLR/InfoNCE) | Integrates with SONA trajectory learning |

### Why CNN (Not Just GNN)?

| Aspect | CNN | GNN (ruvector-gnn) |
|--------|-----|-------------------|
| Input | Raw pixels (grid topology) | Graphs (irregular topology) |
| Use Case | Image embeddings | Relational reasoning |
| Spatial hierarchy | Built-in (conv filters) | Requires positional encoding |
| Complementary | Image → embedding → GNN | GNN for cross-image relations |

CNNs extract visual features; GNNs reason over relationships. They're complementary.

## Decision

Implement `ruvector-cnn` crate with the following architecture:

### 1. Module Structure

```
crates/ruvector-cnn/
├── src/
│   ├── lib.rs
│   ├── backbone/          # MobileNet-V3, ShuffleNet
│   │   ├── mod.rs
│   │   ├── mobilenet.rs
│   │   └── blocks.rs      # Inverted residual, SE blocks
│   ├── layers/            # SIMD-optimized ops
│   │   ├── conv.rs        # Winograd 3x3
│   │   ├── batchnorm.rs
│   │   ├── pooling.rs
│   │   └── activation.rs  # ReLU, Swish, HardSwish
│   ├── simd/              # Architecture-specific
│   │   ├── mod.rs         # Dispatch
│   │   ├── avx2.rs
│   │   ├── neon.rs
│   │   └── wasm.rs        # SIMD128
│   ├── quantization/      # INT8/INT4
│   │   ├── mod.rs
│   │   ├── calibration.rs
│   │   └── vnni.rs        # AVX-512 VNNI
│   ├── contrastive/       # Training losses
│   │   ├── infonce.rs     # SimCLR loss
│   │   ├── triplet.rs
│   │   └── augmentation.rs
│   └── integration/       # RuVector integration
│       ├── sona.rs        # SONA trajectory feeding
│       ├── hyperbolic.rs  # Hyperbolic embedding projection
│       └── index.rs       # ImageIndex wrapper
├── Cargo.toml
└── tests/
```

### 2. SIMD Strategy (per ADR-003)

Follow existing SIMD dispatch pattern from `ruvector-core/src/simd_intrinsics.rs`:

```rust
// Dispatch pattern matching ADR-003
pub fn depthwise_conv_3x3(input: &[f32], kernel: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    { depthwise_conv_3x3_neon(input, kernel, output) }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { depthwise_conv_3x3_avx2(input, kernel, output) }
        } else {
            depthwise_conv_3x3_scalar(input, kernel, output)
        }
    }

    #[cfg(target_arch = "wasm32")]
    { depthwise_conv_3x3_wasm_simd(input, kernel, output) }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64", target_arch = "wasm32")))]
    { depthwise_conv_3x3_scalar(input, kernel, output) }
}
```

### 3. SONA Integration

CNN embeddings integrate with SONA's trajectory learning:

```rust
use sona::{SonaEngine, TrajectoryBuilder};
use ruvector_cnn::{MobileNetEmbedder, EmbeddingExtractor};

// CNN extracts embedding
let embedder = MobileNetEmbedder::new_v3_small()?;
let embedding = embedder.extract(image_data, width, height)?;

// Feed to SONA trajectory for contrastive learning
let mut trajectory = sona_engine.begin_trajectory(embedding.clone());

// After augmented views
let view1_emb = embedder.extract(&augmented_view1, w, h)?;
let view2_emb = embedder.extract(&augmented_view2, w, h)?;

// Record similarity as trajectory step
let similarity = cosine_similarity(&view1_emb, &view2_emb);
trajectory.add_step(view2_emb, vec![], similarity);

// End trajectory triggers SONA learning
sona_engine.end_trajectory(trajectory, similarity);
```

### 4. Hyperbolic Embedding Projection

Leverage `ruvector-hyperbolic-hnsw` for hierarchical image search:

```rust
use ruvector_hyperbolic_hnsw::{HyperbolicHnsw, PoincareBall};

// Project CNN embedding to Poincaré ball
let euclidean_emb = cnn.extract(image)?;
let hyperbolic_emb = poincare_ball.exp_map_zero(&euclidean_emb);

// Hyperbolic HNSW search (better for hierarchical concepts)
let results = hyperbolic_index.search(&hyperbolic_emb, k)?;
```

### 5. WASM/NAPI Bindings

Following existing patterns (`ruvector-wasm`, `ruvector-node`):

```toml
# Cargo.toml
[features]
default = ["std", "avx2"]
std = []
avx2 = []
neon = []
wasm = ["wasm-bindgen"]
napi = ["napi", "napi-derive"]

[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen = "0.2"

[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
napi = { version = "2", optional = true }
napi-derive = { version = "2", optional = true }
```

### 6. No External Dependencies (Pure Rust)

Per `ruvector-math` design principles:
- No BLAS/LAPACK dependencies
- No OpenCV or image processing C libraries
- Only `image` crate for decoding (pure Rust)
- ONNX loading via `tract-onnx` (pure Rust)

## Consequences

### Positive

1. **Unified SIMD**: Reuses ADR-003 patterns, consistent across crates
2. **SONA synergy**: CNN embeddings enhance ReasoningBank pattern learning
3. **Hyperbolic search**: Better semantic hierarchy in image retrieval
4. **WASM-ready**: Runs in browsers without native dependencies
5. **Complements GNN**: CNN for pixels, GNN for relations

### Negative

1. **Initial performance**: Pure Rust CNN may be 2-3x slower than C++/cuDNN initially
2. **Model conversion**: Need to convert PyTorch/ONNX weights to our format
3. **Training limited**: Full contrastive training may be slow on CPU

### Neutral

1. **Memory layout**: NHWC (TensorFlow style) chosen over NCHW for SIMD efficiency

## Performance Targets

| Operation | Target | Baseline Comparison |
|-----------|--------|---------------------|
| MobileNet-V3 Small forward | <5ms | PyTorch CPU: ~8ms |
| Winograd 3x3 conv | 2-2.5x vs direct | Matches literature |
| INT8 inference | 2-4x vs FP32 | With AVX-512 VNNI |
| HNSW search (1M, 512d) | <1ms | Existing ruvector-hyperbolic-hnsw |

## Alternatives Considered

### 1. Use tract-onnx Directly

**Rejected**: tract-onnx is good for loading but doesn't provide SIMD control we need for CNN-specific optimizations (Winograd, fused BN-Conv).

### 2. Vision Transformer (ViT) Instead of CNN

**Deferred**: ViT requires more compute. CNN is more CPU-friendly. ViT can be added later using existing `ruvector-attention`.

### 3. Bind to OpenCV

**Rejected**: Violates pure Rust principle, breaks WASM.

## References

1. ADR-003: SIMD Optimization Strategy
2. ADR-005: WASM Runtime Integration
3. SimCLR Paper: https://arxiv.org/abs/2002.05709
4. MobileNet-V3: https://arxiv.org/abs/1905.02244
5. docs/research/cnn/SOTA_OVERVIEW.md
6. docs/research/cnn/SIMD_OPTIMIZATION.md
7. docs/research/cnn/RUST_IMPLEMENTATION.md
8. docs/research/cnn/RUVECTOR_INTEGRATION.md
