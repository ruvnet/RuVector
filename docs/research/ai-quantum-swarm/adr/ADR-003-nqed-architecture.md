# ADR-003: Neural Quantum Error Decoder (NQED) Architecture

**Status**: Accepted
**Date**: 2025-01-17
**Deciders**: RuVector Architecture Team
**Crate**: `ruvector-neural-decoder`

## Context

Quantum error correction is fundamental to fault-tolerant quantum computing. Traditional decoders like MWPM (Minimum Weight Perfect Matching) have O(n^3) complexity which becomes prohibitive for large code distances. We need a neural decoder that:

1. Achieves O(d^2) complexity for distance-d surface codes
2. Integrates with existing RuVector crates (mincut, attention, ruQu)
3. Supports both inference and optional training
4. Can be deployed to WASM for edge use cases

## Decision

We will implement a **Graph Neural Network + Mamba State Space Model** architecture:

### Architecture Overview

```
                    ┌─────────────────────────────────────────┐
                    │           ruvector-neural-decoder       │
                    └─────────────────────────────────────────┘
                                        │
            ┌───────────────────────────┼───────────────────────────┐
            │                           │                           │
            ▼                           ▼                           ▼
   ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
   │ DetectorGraph   │       │ GraphAttention  │       │ MambaDecoder    │
   │                 │──────▶│ Encoder         │──────▶│                 │
   │ (graph.rs)      │       │ (encoder.rs)    │       │ (decoder.rs)    │
   └─────────────────┘       └─────────────────┘       └─────────────────┘
            │                           │                           │
            │                           │                           │
            ▼                           ▼                           ▼
   ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
   │ ruvector-mincut │       │ Positional      │       │ Correction      │
   │ integration     │       │ Encoding        │       │ Predictions     │
   │ (features.rs)   │       │ (GraphRoPE)     │       │ (X/Z errors)    │
   └─────────────────┘       └─────────────────┘       └─────────────────┘
```

### Core Components

#### 1. DetectorGraph (graph.rs)

Converts syndrome measurements into a graph structure:

```rust
pub struct DetectorGraph {
    nodes: Vec<Node>,           // Detector nodes
    edges: Vec<Edge>,           // Error correlations
    adjacency: HashMap<usize, Vec<usize>>,
    distance: usize,
}

pub struct Node {
    id: usize,
    row: usize,
    col: usize,
    fired: bool,               // Syndrome bit
    node_type: NodeType,       // X or Z stabilizer
    features: Vec<f32>,
}
```

**Key Design Decisions**:
- Nodes are positioned in 2D lattice coordinates
- Edge weights derived from physical error rates
- Support for both X and Z stabilizers in checkerboard pattern
- Feature vector includes position, firing state, and type

#### 2. GraphAttentionEncoder (encoder.rs)

Multi-layer graph attention network with O(E) message passing:

```rust
pub struct GraphAttentionEncoder {
    input_proj: Linear,
    pos_encoding: GraphPositionalEncoding,  // GraphRoPE
    layers: Vec<MessagePassingLayer>,
    output_proj: Linear,
}

pub struct MessagePassingLayer {
    attention: GraphMultiHeadAttention,  // O(E) per layer
    update_linear: Linear,
    layer_norm: LayerNorm,
}
```

**Key Design Decisions**:
- GraphRoPE-style positional encoding (x, y, boundary distance)
- Multi-head attention for neighbor aggregation
- Residual connections + layer normalization
- Configurable depth and width

#### 3. MambaDecoder (decoder.rs)

Selective State Space Model with O(n) sequence processing:

```rust
pub struct MambaDecoder {
    blocks: Vec<MambaBlock>,
    head: Linear,
}

pub struct MambaBlock {
    in_proj: Linear,
    conv: DepthwiseConv1d,       // Causal convolution
    ssm: SelectiveSSM,           // S6 core
    out_proj: Linear,
}

pub struct SelectiveSSM {
    a_log: Array1<f32>,          // Diagonal state matrix
    delta_proj: Linear,          // Discretization step
    b_proj: Linear,              // Input-to-state
    c_proj: Linear,              // State-to-output
}
```

**Key Design Decisions**:
- Selective mechanism makes B, C, delta input-dependent
- Diagonal A matrix for efficient computation
- Causal convolution for local context
- Multiple scan orders: row, column, snake, hilbert

#### 4. StructuralFeatures (features.rs)

Min-cut based structural analysis:

```rust
pub struct StructuralFeatures {
    global_min_cut: f64,
    partition: Option<(Vec<usize>, Vec<usize>)>,
    local_cuts: Vec<f64>,
    conductance: f64,
    centrality: Vec<f64>,
}
```

**Integration with ruvector-mincut**:
- Global min-cut for graph fragility analysis
- Local cuts for node-level structure
- Conductance for expansion properties
- Features fused with GNN output

### Complexity Analysis

| Component | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Graph Construction | O(d^2) | O(d^2) |
| GNN Encoding (L layers) | O(L * E) = O(L * d^2) | O(d^2 * H) |
| Mamba Decoding (M blocks) | O(M * d^2) | O(d^2 + S) |
| Min-Cut Features | O(d^2 * log d) | O(d^2) |
| **Total** | **O(d^2)** | **O(d^2)** |

Where d = code distance, E = edges, H = hidden dim, S = state dim.

### Module Structure

```
ruvector-neural-decoder/
├── src/
│   ├── lib.rs              # Public API, NeuralDecoder
│   ├── error.rs            # Error types
│   ├── graph.rs            # DetectorGraph, Node, Edge
│   ├── encoder.rs          # GraphAttentionEncoder
│   ├── decoder.rs          # MambaDecoder
│   └── fusion.rs           # FeatureFusion (min-cut integration)
├── benches/
│   └── neural_decoder_bench.rs
├── Cargo.toml
└── README.md
```

### Integration Points

1. **ruvector-mincut**: Structural feature extraction
2. **ruqu**: Syndrome types and stabilizer definitions
3. **ruvector-attention**: Shared attention primitives (optional)
4. **ruvector-gnn**: Layer implementations reference

### Feature Flags

```toml
[features]
default = ["parallel"]
full = ["parallel", "simd", "ruqu-integration", "training"]
parallel = ["rayon"]          # Parallel graph operations
simd = ["ruvector-mincut/simd"]  # SIMD acceleration
ruqu-integration = ["ruqu"]   # Direct ruQu type support
training = []                 # Enable backpropagation
```

## Consequences

### Positive

1. **O(d^2) Complexity**: Scales to large code distances
2. **Modular Design**: Each component can be used independently
3. **RuVector Integration**: Leverages existing crate ecosystem
4. **Flexible Deployment**: Pure Rust enables WASM compilation

### Negative

1. **No GPU Support**: CPU-only implementation initially
2. **Training Complexity**: Requires separate training pipeline
3. **Model Size**: Pre-trained weights needed for deployment

### Mitigations

1. GPU support can be added via optional candle/burn backends
2. Pre-trained weights will be distributed with releases
3. Training feature flag keeps inference binary small

## Implementation Notes

### Phase 1: Core Types (Complete)
- [x] DetectorGraph with syndrome support
- [x] GraphAttentionEncoder with positional encoding
- [x] MambaDecoder with selective scan
- [x] Error types and configuration

### Phase 2: Integration (In Progress)
- [x] ruvector-mincut feature extraction
- [ ] ruqu SyndromeRound integration
- [ ] End-to-end pipeline tests

### Phase 3: Optimization (Planned)
- [ ] SIMD vectorization for attention
- [ ] Rayon parallel message passing
- [ ] Quantization support (INT8)

### Phase 4: Deployment (Planned)
- [ ] WASM target support
- [ ] Pre-trained weight distribution
- [ ] Node.js bindings

## Related

- [ADR-001: Research Swarm Structure](ADR-001-swarm-structure.md)
- [ADR-002: Capability Selection Criteria](ADR-002-capability-selection.md)
- [Mamba Paper](https://arxiv.org/abs/2312.00752): Gu & Dao, 2023
- [GNN for QEC](https://arxiv.org/abs/2007.08927): Chamberland et al., 2020
