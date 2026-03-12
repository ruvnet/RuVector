# Ultra-Low-Bit Quantization Survey

## Abstract

This document surveys the state of ultra-low-bit (sub-4-bit) weight quantization
for large language models as of early 2026. The field has transitioned from
theoretical proposals to practical frameworks with real code, benchmarks, and
ICLR'26 accepted results. We focus on methods achieving 2-bit or near-2-bit
weight precision while preserving reasoning capabilities.

## 1. Quantization Fundamentals

### 1.1 What Quantization Does

Model weights are stored as floating-point numbers (FP32 or FP16). Quantization
maps these to lower-precision representations:

```
Precision   Bits    Values     Memory (7B model)
----------------------------------------------
FP32        32      4.3B       28 GB
FP16        16      65K        14 GB
INT8        8       256         7 GB
INT4        4       16          3.5 GB
INT2        2       4           1.75 GB
Ternary     1.58    3           ~1.4 GB
Binary      1       2           ~0.9 GB
```

### 1.2 Why 2-Bit Is Qualitatively Different

At 4 bits, each weight can take 16 values -- enough to approximate most weight
distributions reasonably. At 2 bits, each weight is one of only four values.
This forces fundamentally different strategies:

- **Post-training quantization (PTQ)** often fails at 2-bit -- perplexity
  degradation can exceed 10 points on reasoning benchmarks.
- **Quantization-aware training (QAT)** becomes essential -- the model must
  learn to work within the 2-bit constraint.
- **Incoherence processing** (QuIP) decorrelates weights before quantization,
  making the 4-value approximation much more accurate.

## 2. Taxonomy of Ultra-Low-Bit Methods

### 2.1 Post-Training Quantization (PTQ)

Methods that quantize a pre-trained model without retraining:

| Method | Year | Bits | Approach | Reasoning Preservation |
|--------|------|------|----------|----------------------|
| GPTQ | 2023 | 3-4 | Layer-wise OBS | Moderate |
| AWQ | 2023 | 3-4 | Activation-aware scaling | Good at 4-bit |
| QuIP | 2023 | 2 | Incoherence + adaptive rounding | First viable 2-bit PTQ |
| QuIP# | 2024 | 2 | Lattice codebooks + E8 | Better 2-bit PTQ |
| AQLM | 2024 | 2 | Additive quantization with codebooks | Competitive 2-bit |
| SqueezeLLM | 2024 | 3 | Sensitivity-based non-uniform | Good for outliers |

### 2.2 Quantization-Aware Training (QAT)

Methods that train/fine-tune with quantization in the loop:

| Method | Year | Bits | Approach | Key Innovation |
|--------|------|------|----------|---------------|
| LLM-QAT (Meta) | 2024 | 4+ | Standard QAT loop | KV-cache quantization |
| ParetoQ | 2025 | 2-4 | Multi-objective | Pareto-optimal bit allocation |
| ICLR'26 Two-Stage | 2026 | 2 | Calibration + teacher FT | Reasoning preservation |
| BitNet b1.58 | 2024 | 1.58 | Ternary from scratch | Multiplication-free inference |

### 2.3 Information-Theoretic Methods

| Method | Year | Bits | Approach |
|--------|------|------|----------|
| QuIP | 2023 | 2 | Incoherence processing |
| QuIP# | 2024 | 2 | E8 lattice codebooks |
| IQ-quants | 2024 | 1-2 | Importance-weighted i-quants |

## 3. ICLR 2026: Reasoning-Oriented 2-Bit QAT

### 3.1 The Two-Stage Approach

The key ICLR'26 contribution introduces a two-stage pipeline:

**Stage 1: Mixed-Domain Calibration**

```
For each layer L in model:
    1. Collect activations from diverse calibration set
       (math, code, natural language, structured reasoning)
    2. Compute per-channel sensitivity: S_c = ||dLoss/dW_c||
    3. Allocate precision: high-sensitivity channels get more bits
    4. Initialize quantization grid with sensitivity-weighted centroids
```

The innovation is using mixed-domain data -- previous work used only language
modeling calibration, which biased the quantization grid away from reasoning
patterns.

**Stage 2: Teacher-Guided Fine-Tuning**

```
For each training step:
    1. Forward pass through quantized student (2-bit weights)
    2. Forward pass through full-precision teacher
    3. Compute composite loss:
       L = a * L_task + b * L_KD(student, teacher) + g * L_reasoning
    4. Compute gradients through straight-through estimator (STE)
    5. Update latent full-precision weights
    6. Re-quantize weights
```

Where `L_reasoning` specifically measures chain-of-thought fidelity:

```
L_reasoning = KL(P_student(step_i | steps_<i), P_teacher(step_i | steps_<i))
```

### 3.2 Results on Reasoning Benchmarks

```
Model: LLaMA-2 7B -> 2-bit quantized

Benchmark          FP16    PTQ(2-bit)   QAT(2-bit)   Delta vs PTQ
-----------------------------------------------------------------
GSM8K              56.8    21.3         51.2          +29.9
MATH               18.4    3.1          15.8          +12.7
HumanEval          31.7    8.5          27.4          +18.9
MMLU               45.3    28.7         42.1          +13.4
ARC-Challenge      51.2    31.4         47.8          +16.4
```

The 2-bit QAT model retains ~90% of full-precision reasoning capability,
compared to ~40% for naive PTQ.

### 3.3 Why This Matters for Edge Deployment

A 7B model at 2-bit weighs ~1.75 GB. This fits:

- 4 GB Raspberry Pi 5 (with room for KV cache)
- ESP32-P4 PSRAM (with paging)
- Mobile devices with 2-3 GB available RAM
- FPGA fabric with distributed SRAM

## 4. Practical Frameworks with Code

### 4.1 QuIP (Cornell/RelaxML)

See [detailed analysis](03-quip-2bit-framework.md).

- First framework making 2-bit viable
- Incoherence processing decorrelates weight matrices
- Adaptive rounding with LDPC codes
- Open-source implementation for OPT/LLaMA

### 4.2 Meta's LLM-QAT

- Reusable training loop for quantization-aware training
- Supports KV-cache quantization (unique feature)
- Multi-bit training modes (4, 8, mixed)
- Foundation for ParetoQ extensions

### 4.3 GGML/llama.cpp I-Quants

Already supported in ruvLLM's GGUF parser:

```
IQ1_S:   1.56 bits/weight  (256-element blocks)
IQ2_XXS: 2.06 bits/weight  (extreme compression)
IQ2_XS:  2.31 bits/weight
IQ2_S:   2.50 bits/weight
IQ3_XXS: 3.06 bits/weight
```

These use importance matrices and lattice-based quantization.

### 4.4 BitNet b1.58

Already implemented in ruvLLM (`crates/ruvllm/src/bitnet/`):

- Ternary weights: {-1, 0, +1}
- 2-bit packing: 00=-1, 01=0, 10=+1
- Absmean quantization: `W_ternary = RoundClip(W / mean(|W|), -1, 1)`
- Multiplication-free: all MatMuls become additions/subtractions
- 2.06 bits/weight with scale factors

## 5. Comparison Matrix

```
                    PTQ         QAT         Incoherence    Ternary
                    (GPTQ etc)  (LLM-QAT)   (QuIP)         (BitNet)
--------------------------------------------------------------------
Min viable bits     3-4         2           2              1.58
Requires retraining No         Yes         No             Yes (scratch)
Reasoning @2-bit    Low         High        Medium         Varies
Inference speed     Standard    Standard    Needs decode   Very fast
Edge suitability    Good @4-bit Good @2-bit Good @2-bit    Excellent
Implementation      Mature      Growing     Available      Available
ruvLLM support      Yes         Partial     No             Yes
```

## 6. Research Gaps and Opportunities

1. **QAT + Incoherence**: No work combines QuIP-style preprocessing with QAT --
   this could yield better 2-bit models than either alone.

2. **Mixed-precision MoE**: Per-expert quantization levels based on expert
   utilization frequency -- popular experts keep higher precision.

3. **KV-cache 2-bit**: LLM-QAT shows 4-bit KV works; pushing to 2-bit KV
   would dramatically reduce memory for long contexts.

4. **Hardware-aware QAT**: Training that accounts for target hardware's
   specific quantization granularity (tile sizes, SIMD widths).

5. **Continual QAT**: Combining SONA-style online learning with quantized
   weight updates -- adapt without full retraining.

6. **Irrational-scaling quantization grids**: Using pi as a scaling constant
   for non-uniform grids that reduce spectral distortion. See
   [3-Int Pi-Constant Quantization](07-3int-pi-constant-quantization.md).

## 7. Relevance to ruvLLM

ruvLLM has most of the building blocks:

- **BitNet b1.58**: Full ternary pipeline (2-bit packing, kernels, export)
- **GGUF I-quants**: Parser for IQ1_S through IQ4_NL formats
- **K-quant pipeline**: Q2_K through Q8_K with ANE optimization
- **Two-tier KV cache**: FP16 hot + Q4 cold, extensible to Q2
- **Training infrastructure**: GRPO, contrastive learning, SONA loops
- **LoRA integration**: MicroLoRA for per-request adaptation

The missing piece is a unified QAT training loop that:
1. Maintains latent FP32 weights
2. Applies 2-bit quantization in the forward pass
3. Uses straight-through estimators for backward pass
4. Integrates with SONA for continual learning post-deployment

See [Implementation Plan](06-implementation-plan-rust-ruvllm.md) for details.
