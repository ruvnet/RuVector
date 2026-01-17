# NQED: Neural Quantum Error Decoder

> GNN-based decoder integrated with ruQu's min-cut infrastructure

## Overview

| Attribute | Value |
|-----------|-------|
| **Priority** | Tier 1 (Immediate) |
| **Score** | 94/100 |
| **Integration** | ruQu, ruvector-mincut, ruvector-attention |
| **Proposed Crate** | `ruvector-neural-decoder` |

## Problem Statement

Traditional quantum error decoders (MWPM, UF) are:
1. **Device-agnostic**: Don't adapt to hardware-specific noise
2. **Structure-blind**: Correct errors without assessing graph health
3. **Latency-bound**: Can't scale to large distances in real-time

## Solution

Hybrid GNN decoder that:
1. Learns device-specific noise patterns via graph neural networks
2. Integrates ruQu's min-cut for structural coherence awareness
3. Uses Mamba-style O(d²) architecture for real-time decoding

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     NQED Pipeline                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Syndrome    ┌──────────────┐    ┌──────────────┐               │
│  Round ────► │ Syndrome→    │───►│ GNN Encoder  │               │
│              │ DetectorGraph│    │ (GraphRoPE)  │               │
│              └──────────────┘    └──────┬───────┘               │
│                                         │                        │
│                                         ▼                        │
│              ┌──────────────┐    ┌──────────────┐               │
│              │ Min-Cut      │───►│ Feature      │               │
│              │ Engine       │    │ Fusion       │               │
│              └──────────────┘    └──────┬───────┘               │
│                                         │                        │
│                                         ▼                        │
│                                  ┌──────────────┐    ┌────────┐ │
│                                  │ Mamba        │───►│Correct-│ │
│                                  │ Decoder      │    │ ion    │ │
│                                  └──────────────┘    └────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Innovations

### 1. Structural Coherence Fusion

Unlike pure neural decoders, NQED fuses GNN embeddings with min-cut features:

```rust
pub fn fuse_features(
    node_embeddings: &Tensor,  // From GNN
    cut_value: f64,            // From min-cut
    cut_edges: &[EdgeId],      // Boundary edges
) -> FusedRepresentation {
    // Annotate nodes near cut boundary with structural signal
    let boundary_mask = compute_boundary_proximity(cut_edges);

    // Scale embeddings by coherence confidence
    let coherence_weight = sigmoid(cut_value - threshold);

    // Fused representation carries both learned and structural features
    FusedRepresentation {
        embeddings: node_embeddings * coherence_weight,
        structural_features: StructuralFeatures {
            cut_value,
            boundary_proximity: boundary_mask,
            cut_velocity: self.cut_history.velocity(),
        },
    }
}
```

### 2. O(d²) Mamba Decoder

Replace O(d⁴) transformer attention with state-space model:

```rust
pub struct MambaDecoder {
    /// Selective state space parameters
    A: StateMatrix,  // Transition matrix
    B: InputMatrix,  // Input projection
    C: OutputMatrix, // Output projection
    D: SkipMatrix,   // Skip connection

    /// Discretization step
    delta: f32,

    /// Hidden state
    h: HiddenState,
}

impl MambaDecoder {
    /// O(d²) forward pass
    pub fn forward(&mut self, x: &Tensor) -> Tensor {
        // Selective scan: O(L) per token, O(d²) total for d×d syndrome
        let (y, new_h) = selective_scan(x, &self.A, &self.B, &self.C, &self.D, self.delta);
        self.h = new_h;
        y
    }
}
```

### 3. Online Adaptation

Learn from hardware drift without full retraining:

```rust
pub struct AdaptiveLearningState {
    /// Exponential moving average of error patterns
    error_ema: ErrorPatternEMA,

    /// Low-rank adaptation matrices
    lora_A: Tensor,  // [r, d]
    lora_B: Tensor,  // [d, r]

    /// Adaptation learning rate
    lr: f32,
}

impl AdaptiveLearningState {
    /// Online update when correction is validated
    pub fn update(&mut self, syndrome: &DetectorGraph, correction: &Correction, success: bool) {
        if !success {
            // Backprop through LoRA matrices only (frozen base model)
            let loss = self.compute_loss(syndrome, correction);
            self.lora_A -= self.lr * loss.grad_a;
            self.lora_B -= self.lr * loss.grad_b;
        }

        // Update error pattern statistics
        self.error_ema.update(syndrome);
    }
}
```

## Integration with RuVector

### ruQu Integration

```rust
// In ruqu crate
impl QuantumFabric {
    pub fn with_neural_decoder(mut self, decoder: NeuralDecoder) -> Self {
        self.decoder_backend = DecoderBackend::Neural(decoder);
        self
    }

    pub fn process_cycle_neural(&mut self, syndrome: &SyndromeRound) -> NeuralDecision {
        // 1. Standard coherence assessment
        let coherence = self.assess_coherence(syndrome);

        // 2. Neural decoding with structural fusion
        let correction = self.decoder_backend.decode_with_coherence(
            syndrome,
            coherence.cut_value,
            coherence.cut_edges,
        );

        // 3. Combined decision
        NeuralDecision {
            gate: coherence.decision,
            correction,
            confidence: correction.confidence * coherence.structural_confidence,
        }
    }
}
```

### ruvector-mincut Integration

```rust
// Reuse existing min-cut engine
use ruvector_mincut::{DynamicMinCutEngine, MinCutQuery};

impl NeuralDecoder {
    pub fn new(mincut: DynamicMinCutEngine) -> Self {
        Self {
            mincut_bridge: mincut,
            // ... GNN, Mamba initialization
        }
    }

    fn query_structural_features(&self, graph: &DetectorGraph) -> StructuralFeatures {
        let cut = self.mincut_bridge.query_min_cut(graph);
        StructuralFeatures {
            cut_value: cut.value,
            cut_edges: cut.edges,
            // ... derived features
        }
    }
}
```

## Research Tasks

- [ ] Literature review: AlphaQubit, Mamba decoders, GNN for QEC
- [ ] Design GNN encoder architecture (GraphRoPE vs GAT)
- [ ] Implement Mamba decoder in Rust
- [ ] Feature fusion experiments (cut value weighting strategies)
- [ ] Benchmark against MWPM, UF on surface codes d=3 to d=11
- [ ] Online learning convergence analysis
- [ ] WASM compilation for cognitum-gate-kernel deployment

## References

1. [AlphaQubit: Neural decoders for quantum error correction](https://blog.google/technology/google-deepmind/alphaqubit-quantum-error-correction/)
2. [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752)
3. [GNNs for Quantum Error Correction](https://link.aps.org/doi/10.1103/PhysRevResearch.7.023181)
4. [Dynamic Min-Cut with Subpolynomial Update Time](https://arxiv.org/abs/2512.13105)
