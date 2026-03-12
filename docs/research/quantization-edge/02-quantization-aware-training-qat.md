# Quantization-Aware Training (QAT) for Ultra-Low-Bit LLMs

## Abstract

Quantization-aware training (QAT) is the process of training or fine-tuning a
neural network while simulating the effects of quantization in the forward pass.
Unlike post-training quantization (PTQ), QAT allows the model to adapt its weights
to compensate for quantization error, making it essential for ultra-low-bit
(2-bit) precision where PTQ alone degrades reasoning by 30-60%.

This document covers the theory, state-of-the-art implementations, and practical
considerations for integrating QAT into ruvLLM's Rust-based training pipeline.

## 1. QAT Fundamentals

### 1.1 The Core Idea

In standard training, weights are FP32:

```
Forward:   y = W * x           (FP32 multiply)
Backward:  dW = dL/dW          (FP32 gradients)
Update:    W = W - lr * dW     (FP32 update)
```

In QAT, we simulate quantization during training:

```
Forward:   W_q = Quantize(W)   (round to low-bit)
           y = W_q * x         (quantized multiply)
Backward:  dW = dL/dW via STE  (straight-through estimator)
Update:    W = W - lr * dW     (update LATENT FP32 weights)
```

The key insight: we maintain full-precision "latent" weights that accumulate
gradients, but the model only ever sees the quantized version during forward
passes. This lets the model learn weight configurations that are robust to
quantization.

### 1.2 Straight-Through Estimator (STE)

The quantization function `Q(w)` is a step function -- non-differentiable.
The straight-through estimator simply passes gradients through unchanged:

```
Forward:     w_q = Q(w)     (round to nearest quantized value)
Backward:    dw = dw_q      (gradient passes through Q unchanged)
```

With clipping (recommended for stability):

```
STE(w) = dw_q * 1_{|w| <= clip_val}
```

This works because even though the gradient is biased, it still points the
latent weights toward configurations that minimize loss under quantization.

### 1.3 Quantization Functions for 2-Bit

**Uniform 2-bit quantization (4 levels):**

```
Q(w) = clip(round(w / s), -2, 1) * s

where s = max(|W|) / 2  (per-channel or per-block scale)
Levels: {-2s, -s, 0, s}
```

**Non-uniform 2-bit (learned centroids):**

```
Q(w) = argmin_c ||w - c_i||  for c_i in {c_0, c_1, c_2, c_3}
Centroids are learned during training.
```

**Ternary (BitNet b1.58 style):**

```
Q(w) = RoundClip(w / (mean(|W|) + eps), -1, 1)
Levels: {-1, 0, +1} with per-block scale = mean(|W|)
```

## 2. Two-Stage Reasoning-Oriented QAT (ICLR'26)

### 2.1 Why Standard QAT Fails at 2-Bit for Reasoning

Standard QAT with language modeling loss preserves perplexity but not reasoning:

```
Perplexity (WikiText-2):
  FP16:        5.47
  QAT-2bit:    6.12  (+12% -- acceptable)

GSM8K accuracy:
  FP16:        56.8%
  QAT-2bit:    34.2%  (-40% -- unacceptable)
```

The problem: language modeling loss optimizes for next-token prediction on
fluent text, but reasoning requires preserving structured multi-step computation
that occupies a small fraction of the weight space.

### 2.2 Stage 1: Mixed-Domain Calibration

**Goal**: Initialize the quantization grid so it preserves reasoning-critical
weight regions.

**Algorithm**:

```
Input:  Pre-trained model M, calibration datasets D = {D_math, D_code, D_nl, D_reasoning}
Output: Per-layer quantization parameters (scales, zero-points, centroids)

For each layer L:
    activations = []
    for domain in D:
        # Collect 1024 samples per domain
        for batch in domain:
            a = L.forward(batch)
            activations.append(a)

    # Compute per-channel Fisher information
    for channel c in L.weight:
        F_c = E[||d log p(y|x) / dW_c||^2]

    # Sensitivity-weighted centroid initialization
    weights_flat = L.weight.flatten()
    importance = F_c.expand_as(weights_flat)

    # K-means with importance weighting
    centroids = weighted_kmeans(
        weights_flat,
        k=4,  # 2-bit = 4 centroids
        weights=importance,
        n_iter=100
    )

    L.quant_params = QuantParams(centroids=centroids, scales=per_channel_scales)
```

**Key design decisions:**

1. **Mixed-domain data** -- math and code calibration data ensures reasoning
   weight regions are properly characterized.
2. **Fisher information weighting** -- high-gradient weights get more influence
   on centroid placement.
3. **Per-channel granularity** -- different channels have different distributions;
   shared quantization grids lose precision.

### 2.3 Stage 2: Teacher-Guided Fine-Tuning

**Goal**: Fine-tune the quantized model using a full-precision teacher to
preserve reasoning behavior.

**Loss function:**

```
L_total = alpha * L_task + beta * L_KD + gamma * L_reasoning

where:
  L_task      = CrossEntropy(student_logits, targets)
  L_KD        = KL(softmax(student/T), softmax(teacher/T)) * T^2
  L_reasoning = sum_i KL(P_s(step_i | context), P_t(step_i | context))
```

**Hyperparameters (from ICLR'26 paper):**

```
alpha   = 1.0     (task loss weight)
beta    = 0.5     (general distillation weight)
gamma   = 2.0     (reasoning distillation weight -- intentionally high)
T       = 4.0     (distillation temperature)
lr      = 1e-5    (learning rate -- low for fine-tuning)
epochs  = 3       (sufficient with good initialization from Stage 1)
```

**Training loop pseudocode:**

```
for epoch in 1..=3:
    for batch in training_data:
        # Teacher forward (no grad)
        with no_grad():
            teacher_logits = teacher.forward(batch)
            teacher_reasoning = teacher.forward(reasoning_prompts)

        # Student forward (quantized)
        student.apply_quantization()
        student_logits = student.forward(batch)
        student_reasoning = student.forward(reasoning_prompts)

        # Composite loss
        loss = compute_composite_loss(
            student_logits, teacher_logits,
            student_reasoning, teacher_reasoning,
            targets, alpha, beta, gamma, T
        )

        # Backward through STE
        loss.backward()  # STE handles quantization gradient

        # Update latent weights
        optimizer.step()
        optimizer.zero_grad()
```

### 2.4 Calibration Dataset Composition

The ICLR'26 paper uses:

| Domain | Dataset | Samples | Purpose |
|--------|---------|---------|---------|
| Math | GSM8K train + MATH train | 15K | Arithmetic reasoning |
| Code | HumanEval + MBPP | 5K | Structured reasoning |
| Language | C4 / RedPajama subset | 20K | General fluency |
| Reasoning | ARC + HellaSwag | 10K | Common-sense reasoning |

Total: ~50K calibration samples for Stage 1, ~100K for Stage 2 fine-tuning.

## 3. Meta's LLM-QAT Framework

### 3.1 Architecture

Meta's LLM-QAT provides a reusable training loop with three quantization targets:

1. **Weight quantization**: Standard per-channel or per-group quantization
2. **Activation quantization**: Per-tensor dynamic quantization
3. **KV-cache quantization**: Unique contribution -- quantize cached keys/values

### 3.2 KV-Cache Quantization

This is particularly relevant for long-context edge inference:

```
Standard KV cache (FP16):
  Memory per token = 2 * n_layers * n_kv_heads * head_dim * 2 bytes
  LLaMA-7B: 2 * 32 * 32 * 128 * 2 = 524 KB per token
  4K context: 2 GB just for KV cache

QAT-trained KV cache (INT4):
  Memory per token = 2 * n_layers * n_kv_heads * head_dim * 0.5 bytes
  LLaMA-7B: 2 * 32 * 32 * 128 * 0.5 = 131 KB per token
  4K context: 512 MB for KV cache (4x reduction)
```

ruvLLM's two-tier KV cache (`kv_cache.rs`) already does FP16+Q4 tiering.
LLM-QAT shows that training with Q4 KV from the start yields better quality
than post-hoc compression.

### 3.3 Training Configuration

```python
# From LLM-QAT repository
config = QATConfig(
    weight_bits=4,          # or 2 for ultra-low-bit
    activation_bits=8,      # keep activations higher precision
    kv_cache_bits=4,        # or 2 for aggressive compression
    weight_quant="per_channel",
    act_quant="per_tensor_dynamic",
    kv_quant="per_head",
    use_ste=True,
    clip_ratio=1.0,
    num_calibration_batches=128,
)
```

## 4. ParetoQ: Multi-Objective Ultra-Low-Bit

### 4.1 Core Idea

ParetoQ treats bit-width allocation as a multi-objective optimization problem:

- **Objective 1**: Minimize model size (total bits)
- **Objective 2**: Minimize task loss (quality)
- **Objective 3**: Minimize inference latency

Different layers have different sensitivity to quantization. ParetoQ
finds Pareto-optimal configurations:

```
Layer Type       Sensitivity    Recommended Bits
------------------------------------------------
Embedding        Low            2-3 bits
Attention Q/K    High           3-4 bits
Attention V/O    Medium         2-3 bits
FFN Gate/Up      Medium         2-3 bits
FFN Down         High           3-4 bits
LM Head          Very High      4-8 bits
```

### 4.2 Mixed-Precision Assignment

ParetoQ uses reinforcement learning to search the bit-width space:

```
State:   Current bit-width assignment for all layers
Action:  Increase/decrease bits for a specific layer
Reward:  -alpha * size_increase + beta * quality_improvement
```

This produces mixed-precision models where critical layers get more bits
while less sensitive layers are aggressively compressed.

### 4.3 Results

```
Model: LLaMA-7B, Target: 2.5 average bits

Method           Avg Bits   MMLU    GSM8K   Size
-------------------------------------------------
Uniform 2-bit    2.0        28.7    21.3    1.75 GB
Uniform 3-bit    3.0        41.5    48.2    2.63 GB
ParetoQ mixed    2.5        40.8    45.1    2.19 GB  (best tradeoff)
```

## 5. Straight-Through Estimator Variants

### 5.1 Standard STE

```
Forward:  q = round(w / s) * s
Backward: dw = dq            (identity)
```

**Problem**: Gradient is biased -- ignores quantization error.

### 5.2 Clipped STE

```
Forward:  q = clip(round(w / s), min_q, max_q) * s
Backward: dw = dq * (1 if min_q*s <= w <= max_q*s else 0)
```

**Benefit**: Prevents latent weights from drifting far outside quantization range.

### 5.3 Learned Step Size Quantization (LSQ)

```
Forward:  q = round(clip(w / s, -Q_N, Q_P)) * s
Backward: ds = dq * (round(w/s) - w/s) if in range, else (-Q_N or Q_P)
           dw = dq / s  (if in range)
```

**Benefit**: Scale factor `s` is learned, adapting to weight distribution.

### 5.4 EWGS (Elastic Weight Gradient Scaling)

```
Backward: dw = dq * (1 + lambda * |w - q|)
```

**Benefit**: Weights far from their quantized value get stronger gradients,
pushing them toward stable quantization points.

## 6. Practical Implementation Considerations

### 6.1 Memory Requirements

QAT requires more memory than standard training:

```
Standard inference:  Model weights (quantized)
PTQ:                 Model weights (FP16) + calibration activations
QAT:                 Latent weights (FP32) + quantized weights + gradients + optimizer state

Memory for QAT on 7B model:
  Latent FP32 weights:   28 GB
  Quantized weights:      1.75 GB (2-bit)
  Gradients:             28 GB (FP32)
  Adam optimizer state:  56 GB (2x FP32 for m, v)
  Total:                 ~114 GB
```

**Mitigation strategies:**

1. **LoRA-QAT**: Only train low-rank adapters, not full weights (~1% of params)
2. **Gradient checkpointing**: Trade compute for memory
3. **Mixed-precision training**: FP16 gradients where possible
4. **Layer-wise QAT**: Quantize and fine-tune one layer at a time

### 6.2 LoRA-QAT (Recommended for ruvLLM)

Instead of full QAT, train LoRA adapters on top of quantized base weights:

```
Forward:
    W_q = Quantize(W_base)              # Quantize frozen base
    W_effective = W_q + B @ A * (a/r)   # Add LoRA delta
    y = W_effective @ x

Backward:
    dA, dB = compute_lora_gradients()   # Only LoRA params get gradients
    # W_base stays frozen and quantized
```

**Advantages:**
- Memory: only LoRA params (A, B) need optimizer state (~50 MB for rank-16)
- Speed: much faster convergence (1 epoch vs 3)
- Flexibility: different LoRA adapters for different tasks on same quantized base
- Fits ruvLLM's existing MicroLoRA infrastructure

### 6.3 Training Data Quality

For reasoning-preserving QAT, training data must include:

1. **Chain-of-thought examples**: Step-by-step reasoning traces
2. **Multi-turn dialogues**: Context-dependent reasoning
3. **Code generation**: Structured output with syntax constraints
4. **Mathematical proofs**: Formal logical sequences

ruvLLM's `training/` directory already has dataset generators for
tool use (`tool_dataset.rs`) and Claude-style data (`claude_dataset.rs`).
A reasoning-focused dataset generator is needed.

## 7. Rust Implementation Strategy

### 7.1 Core QAT Types

```rust
/// Configuration for quantization-aware training
pub struct QatConfig {
    /// Target bit-width for weights
    pub weight_bits: u8,           // 2, 3, 4, or 8
    /// Target bit-width for KV cache
    pub kv_cache_bits: u8,         // 2, 4, or 8
    /// Quantization granularity
    pub granularity: QuantGranularity, // PerTensor, PerChannel, PerGroup(n)
    /// STE variant
    pub ste_variant: SteVariant,   // Standard, Clipped, LSQ, EWGS
    /// Clip ratio for clipped STE
    pub clip_ratio: f32,           // default 1.0
    /// Whether to use mixed-precision (ParetoQ-style)
    pub mixed_precision: bool,
    /// Teacher model path (for distillation)
    pub teacher_model: Option<PathBuf>,
    /// Distillation temperature
    pub temperature: f32,          // default 4.0
    /// Loss weights
    pub alpha_task: f32,           // default 1.0
    pub beta_kd: f32,              // default 0.5
    pub gamma_reasoning: f32,      // default 2.0
}

/// STE variant for backward pass
pub enum SteVariant {
    Standard,
    Clipped { clip_val: f32 },
    LearnedStepSize,
    ElasticWeightGradient { lambda: f32 },
}

/// Quantization granularity
pub enum QuantGranularity {
    PerTensor,
    PerChannel,
    PerGroup(usize),  // group size (e.g., 128 for GPTQ-style)
    PerBlock(usize),  // block size (e.g., 256 for K-quants)
}
```

### 7.2 Integration Points with ruvLLM

```
Existing module          QAT integration needed
---------------------------------------------------------
quantize/ruvltra_quant   Add differentiable quantize/dequantize
bitnet/quantizer         Add STE backward pass
training/real_trainer    Add QAT training loop
training/grpo            Support quantized policy model
lora/micro_lora          LoRA-QAT: adapt on quantized base
sona/integration         Post-deployment continual QAT
kv_cache                 Trainable KV quantization parameters
backends/candle_backend  Forward pass with simulated quantization
```

### 7.3 Gradient Computation through Quantization

The critical Rust implementation -- differentiable quantization:

```rust
/// Differentiable 2-bit quantization with STE
pub fn quantize_2bit_ste(
    weights: &[f32],        // latent FP32 weights
    scale: f32,             // quantization scale
    grad_output: &[f32],    // upstream gradient
) -> (Vec<u8>, Vec<f32>) {  // (quantized, weight_gradients)
    let mut quantized = Vec::with_capacity(weights.len() / 4);
    let mut grad_input = Vec::with_capacity(weights.len());

    for (w, g) in weights.iter().zip(grad_output.iter()) {
        // Forward: quantize to {-2, -1, 0, 1} * scale
        let w_scaled = w / scale;
        let q = w_scaled.round().clamp(-2.0, 1.0);

        // Backward: STE with clipping
        let in_range = w_scaled.abs() <= 2.0;
        let grad = if in_range { *g } else { 0.0 };

        grad_input.push(grad);
        // Pack 4 values into 1 byte
        // (packing logic omitted for clarity)
    }

    (quantized, grad_input)
}
```

## 8. Open Questions

1. **STE bias at 2-bit**: The STE gradient bias is larger at lower bit-widths.
   Does EWGS or LSQ compensate sufficiently, or do we need custom estimators?

2. **Calibration data size**: Is 50K samples enough for mixed-domain calibration?
   Larger calibration may help but increases Stage 1 cost.

3. **LoRA rank for QAT**: What LoRA rank preserves reasoning at 2-bit?
   ruvLLM's MicroLoRA uses rank-1; QAT may need rank-8 to rank-16.

4. **Continual QAT**: Can SONA's three-tier learning loop perform incremental
   QAT, adjusting quantization parameters as the model adapts?

5. **Hardware-specific grids**: Should quantization grids be optimized for
   specific hardware (ANE tile sizes, NEON SIMD widths)?
