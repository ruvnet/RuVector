# ruvLLM Quantization Architecture Review

## Abstract

This document provides a deep architectural review of the quantization subsystems
within ruvLLM, the Rust-native LLM inference runtime in the RuVector ecosystem.
ruvLLM implements multiple quantization approaches across different modules,
from BitNet ternary to K-quant hierarchical quantization. We analyze the current
capabilities, identify gaps, and map integration points for 2-bit QAT.

## 1. Quantization Module Inventory

### 1.1 Module Map

```
crates/ruvllm/src/
  quantize/
    mod.rs                  # Public API re-exports
    ruvltra_quant.rs        # K-quant implementation (34K lines)
                            # Q4_K_M, Q5_K_M, Q8_0, F16
                            # ANE-optimized layouts

  bitnet/
    mod.rs                  # BitNet b1.58 module root (17 files)
    ternary_tensor.rs       # 2-bit packed ternary storage
    quantizer.rs            # FP32 -> ternary conversion
    dequantize.rs           # Ternary -> FP32 kernels
    backend.rs              # Full BitNet inference backend
    tl1_kernel.rs           # Optimized ternary kernels
    tl1_avx2.rs             # AVX2 SIMD kernels
    tl1_wasm.rs             # WASM SIMD kernels
    expert_cache.rs         # MoE expert cache (LRU/LFU/ARC)
    rlm_embedder.rs         # Residual learning embedder
    rlm_refiner.rs          # Residual learning refiner
    gguf_export.rs          # Export to GGUF format
    eval.rs                 # Evaluation suite
    trace.rs                # Execution tracing
    tokenizer.rs            # Tokenizer integration
    tests.rs                # Test suite

  gguf/
    mod.rs                  # GGUF module root
    quantization.rs         # 30+ quantization type definitions
                            # Dequantization kernels
    parser.rs               # GGUF file parser
    loader.rs               # Model loading
    tensors.rs              # Tensor management
    model_init.rs           # Model initialization

  kv_cache.rs              # Two-tier KV cache (49K lines)
                           # FP16 tail + Q4 cold store
```

### 1.2 Quantization Format Coverage

```
Format        Bits    Module              Status
------------------------------------------------
F32           32      gguf/quantization   Full support
F16           16      quantize + gguf     Full support
Q8_0          8.5     quantize            Full support
Q8_1          9       gguf/quantization   Parser + dequant
Q6_K          6.56    gguf/quantization   Parser + dequant
Q5_K_M        5.5     quantize            Full pipeline
Q5_K          5.5     gguf/quantization   Parser + dequant
Q5_0/Q5_1     5-6     gguf/quantization   Parser + dequant
Q4_K_M        4.5     quantize            Full pipeline (primary)
Q4_K          4.5     gguf/quantization   Parser + dequant
Q4_0/Q4_1     4-5     gguf/quantization   Parser + dequant
Q3_K          3.44    gguf/quantization   Parser + dequant
Q2_K          2.56    gguf/quantization   Parser + dequant
IQ4_NL        4.5     gguf/quantization   Parser only
IQ3_XXS/S     3.06    gguf/quantization   Parser only
IQ2_XXS/XS/S  2-2.5   gguf/quantization   Parser only
IQ1_S         1.56    gguf/quantization   Parser only
BitNet T1.58  ~2.06   bitnet/             Full pipeline
```

## 2. K-Quant Pipeline (quantize/)

### 2.1 Architecture

The K-quant pipeline (`ruvltra_quant.rs`) implements GGML-style K-quant
quantization with Apple Neural Engine optimizations:

```
Input: FP32/FP16 model weights (safetensors format)
  |
  v
Per-layer quantization:
  1. Read tensor data
  2. Compute statistics (min, max, mean, std)
  3. Choose quantization grid based on target format
  4. Quantize blocks:
     - Q4_K_M: 256-element super-blocks with 4-bit sub-blocks
     - Q5_K_M: 256-element super-blocks with 5-bit sub-blocks
     - Q8_0:   32-element blocks with 8-bit symmetric
  5. Apply ANE alignment (16-byte boundaries)
  6. Write GGUF output with metadata
  |
  v
Output: GGUF file with quantized weights + metadata
```

### 2.2 Block Structures

```rust
// Q4_K_M: 4.5 bits/weight, 256-element super-blocks
pub struct Q4KMBlock {
    d: f16,                    // super-block scale (2 bytes)
    dmin: f16,                 // super-block minimum (2 bytes)
    scales: [u8; 12],         // sub-block scales (12 bytes)
    qs: [u8; 128],            // quantized values (128 bytes)
}                              // Total: 144 bytes for 256 weights

// Q8_0: 8.5 bits/weight, 32-element blocks
pub struct Q8Block {
    d: f16,                    // block scale (2 bytes)
    qs: [i8; 32],             // quantized values (32 bytes)
}                              // Total: 34 bytes for 32 weights
```

### 2.3 ANE Optimization Details

```
ANE Tile Sizes:       16x16 and 32x32
Alignment:            16-byte boundaries (ANE_ALIGNMENT = 16)
Block Size:           256 (K_BLOCK_SIZE) aligned to tile operations
Sub-Block Size:       32 (K_SUB_BLOCK_SIZE) for fine-grained scales

Memory Layout:
  Standard:   [scale][q0 q1 q2 ... q255][scale][q256 ...]
  ANE-optimized: [scale][q0..q31][scale][q32..q63]...
  (interleaved scales for fused load+dequantize)
```

### 2.4 Dequantization Kernels

```rust
/// ANE-optimized dequantization
pub fn dequantize_for_ane(
    quantized: &[u8],
    format: TargetFormat,
    output: &mut [f32],
) -> Result<()> {
    match format {
        TargetFormat::Q4_K_M => dequantize_q4km_ane(quantized, output),
        TargetFormat::Q5_K_M => dequantize_q5km_ane(quantized, output),
        TargetFormat::Q8_0 => dequantize_q8_ane(quantized, output),
        TargetFormat::F16 => dequantize_f16(quantized, output),
    }
}
```

## 3. BitNet b1.58 Pipeline (bitnet/)

### 3.1 Ternary Quantization

BitNet maps FP32 weights to {-1, 0, +1} using absmean:

```
Algorithm:
  1. Compute gamma = mean(|W|) + epsilon
  2. Scale: W_scaled = W / gamma
  3. Round and clip: W_ternary = RoundClip(W_scaled, -1, +1)

Packing (2 bits per weight):
  00 = -1
  01 =  0
  10 = +1
  11 = reserved

Storage: 256-element blocks
  64 bytes: 256 ternary values (2 bits each)
   2 bytes: FP16 scale factor (gamma)
  Total: 66 bytes per block = 2.06 bits/weight
```

### 3.2 Kernel Architecture

BitNet's key advantage: multiplication-free inference. Matrix-vector multiply
becomes conditional addition/subtraction:

```rust
// Ternary MatMul: y = W_ternary @ x
// Instead of: y_i = sum_j W_ij * x_j
// We compute: y_i = sum_{j: W=+1} x_j - sum_{j: W=-1} x_j

pub fn ternary_matvec(
    weights: &TernaryTensor,  // packed 2-bit weights
    input: &[f32],            // FP32 input vector
    output: &mut [f32],       // FP32 output vector
) {
    for (row_idx, row) in weights.rows().enumerate() {
        let mut acc = 0.0f32;
        for (j, val) in row.iter().enumerate() {
            match val {
                TernaryVal::Pos => acc += input[j],
                TernaryVal::Neg => acc -= input[j],
                TernaryVal::Zero => {},  // skip (sparse)
            }
        }
        output[row_idx] = acc * weights.scale(row_idx);
    }
}
```

### 3.3 Platform-Specific Kernels

```
Platform    File              Optimization
------------------------------------------
Generic     tl1_kernel.rs     Baseline Rust implementation
x86_64      tl1_avx2.rs      AVX2 SIMD (256-bit, 8 floats)
WASM        tl1_wasm.rs       WASM SIMD (128-bit, 4 floats)
ARM         (planned)         NEON SIMD (128-bit, 4 floats)
Metal       (via backend)     GPU compute shaders
```

### 3.4 Expert Cache for MoE

The `expert_cache.rs` module provides a sophisticated caching layer:

```rust
pub struct ExpertCache {
    config: ExpertCacheConfig,
    // Cached expert weights (quantized)
    cache: HashMap<ExpertId, CachedExpert>,
    // Eviction policy
    policy: EvictionPolicy,  // LRU, LFU, ARC
    // Batch scheduler for parallel expert execution
    scheduler: MoeBatchScheduler,
    // Async prefetcher
    prefetcher: Box<dyn Prefetcher>,
    // Statistics
    stats: ExpertCacheStats,
}
```

## 4. GGUF Quantization Types (gguf/)

### 4.1 Type Definitions

The GGUF module defines 30+ quantization types via `GgufQuantType` enum,
covering the full range from F64 down to IQ1_S (1.56 bits):

```
Category        Types                          Status in ruvLLM
---------------------------------------------------------------
Full precision  F32, F16, F64                  Full support
8-bit           Q8_0, Q8_1, Q8_K              Full dequant
6-bit           Q6_K                           Full dequant
5-bit           Q5_0, Q5_1, Q5_K              Full dequant
4-bit           Q4_0, Q4_1, Q4_K, IQ4_NL/XS   Full dequant
3-bit           Q3_K, IQ3_XXS/S               Partial
2-bit           Q2_K, IQ2_XXS/XS/S            Parser only
1-bit           IQ1_S                          Parser only
Ternary         BitNet (custom)                Full pipeline
Integer         I8, I16, I32, I64              Parser only
Special         BQ8_0, Q4_XS, Q8_XS           Parser only
```

### 4.2 I-Quant Gap Analysis

The I-quant formats (IQ1_S through IQ4_NL) use importance-matrix-weighted
lattice quantization. ruvLLM has the type definitions and parser but lacks:

1. **Importance matrix computation**: Need to compute per-weight importance
   from calibration data (Fisher information or activation sensitivity)
2. **Lattice codebook generation**: E8 lattice for IQ2, simpler lattices for IQ3/IQ4
3. **Dequantization kernels**: Need platform-optimized decompression
4. **Quantization pipeline**: FP32 -> IQ format conversion

This is the primary gap for achieving high-quality 2-bit quantization
beyond BitNet ternary.

## 5. Two-Tier KV Cache (kv_cache.rs)

### 5.1 Architecture

```
Token Stream: [t0, t1, t2, ..., t_current]
                                    |
              Cold Store (Q4)       |  Hot Tail (FP16)
              [t0...t_n-tail]       |  [t_n-tail...t_current]
              Quantized, compact    |  Full precision, fast access
```

### 5.2 Configuration

```rust
pub struct KvCacheConfig {
    pub tail_length: usize,         // tokens in FP16 (default: 128)
    pub tail_precision: Precision,  // Precision::F16
    pub store_precision: Precision, // Precision::Q4 (cold store)
    pub max_tokens: usize,          // maximum context length
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}
```

### 5.3 Memory Savings

```
LLaMA-7B, 4K context, 32 layers, 32 KV heads, 128 head dim:

All FP16:       2 * 32 * 32 * 128 * 4096 * 2 bytes = 2.0 GB
Two-tier (Q4):  FP16 tail (128 tokens): 32 MB
                Q4 cold (3968 tokens):  247 MB
                Total: 279 MB (7.2x reduction)

Potential Q2 cold store:
                FP16 tail (128 tokens): 32 MB
                Q2 cold (3968 tokens):  124 MB
                Total: 156 MB (12.8x reduction)
```

### 5.4 Dequantization Performance

The KV cache uses NEON SIMD for fast Q4->FP32 conversion during attention:

```
NEON 8x unrolled Q4 dequantize:
  Throughput: ~16 GB/s on M4 Pro
  Latency per 256-element block: ~0.6 microseconds
  Overhead vs FP16: ~15% additional latency in attention
```

## 6. Training Infrastructure

### 6.1 Existing Training Modules

```
training/
  real_trainer.rs     # Online trainer with contrastive learning
  contrastive.rs      # Contrastive loss + hard negative mining
  grpo.rs             # Group Relative Policy Optimization
  tool_dataset.rs     # Tool use dataset generation
  claude_dataset.rs   # Claude-style conversation data
  mcp_tools.rs        # MCP tool integration for data
```

### 6.2 SONA Three-Tier Learning

```
sona/
  integration.rs      # Three-tier loop coordinator
  ruvltra_pretrain.rs  # RuvLTRA-Small pretraining config

Tier 1 (Instant, <1ms):     MicroLoRA rank-1 adaptation
Tier 2 (Background, ~100ms): EWC++ Fisher update, BaseLoRA composition
Tier 3 (Deep, minutes):      Pattern consolidation, knowledge transfer
```

### 6.3 LoRA Infrastructure

```
lora/
  micro_lora.rs    # Rank-1 per-request adaptation (8.56us latency)
  adapter.rs       # General adapter management
  training.rs      # Online LoRA training
  adapters/        # Pre-trained task-specific adapters

Merging strategies: Average, WeightedSum, SLERP, TIES, DARE, TaskArithmetic
```

## 7. Gap Analysis for 2-Bit QAT

### 7.1 What Exists

| Capability | Status | Location |
|------------|--------|----------|
| 2-bit weight storage | Yes | bitnet/ternary_tensor.rs |
| 2-bit packing/unpacking | Yes | bitnet/ |
| Multiplication-free kernels | Yes | bitnet/tl1_*.rs |
| GGUF 2-bit type definitions | Yes | gguf/quantization.rs |
| K-quant Q2_K parser | Yes | gguf/ |
| Training loop infrastructure | Yes | training/ |
| LoRA adaptation | Yes | lora/ |
| SONA learning loops | Yes | sona/ |
| KV cache quantization | Yes (Q4) | kv_cache.rs |
| MoE expert caching | Yes | bitnet/expert_cache.rs |
| Evaluation harness | Yes | evaluation/ |
| SIMD kernels | Yes | bitnet/tl1_*.rs, kernels/ |

### 7.2 What Is Missing

| Capability | Priority | Estimated Effort |
|------------|----------|-----------------|
| Differentiable quantization (STE) | Critical | 2 weeks |
| QAT training loop | Critical | 3 weeks |
| Mixed-domain calibration | High | 1 week |
| Teacher-student distillation | High | 2 weeks |
| Reasoning-focused loss function | High | 1 week |
| I-quant dequantization kernels | Medium | 2 weeks |
| Importance matrix computation | Medium | 1 week |
| QuIP incoherence processing | Medium | 2 weeks |
| Per-expert mixed precision | Medium | 1 week |
| Memory-aware MoE routing | Low | 2 weeks |
| KV cache Q2 support | Low | 1 week |
| Dynamic precision switching | Low | 1 week |

### 7.3 Integration Architecture

```
Proposed new modules:

ruvllm/src/
  qat/                           # NEW: QAT training
    mod.rs                       # QAT public API
    config.rs                    # QAT configuration
    ste.rs                       # Straight-through estimators
    differentiable_quant.rs      # Differentiable quantization ops
    calibration.rs               # Mixed-domain calibration
    distillation.rs              # Teacher-student distillation
    reasoning_loss.rs            # Reasoning-preserving loss
    training_loop.rs             # Main QAT training loop

  quantize/
    incoherence.rs               # NEW: QuIP-style incoherence
    importance.rs                # NEW: Importance matrix computation
    iq_quant.rs                  # NEW: I-quant quantization pipeline

  moe/                           # NEW: Memory-aware MoE
    mod.rs                       # MoE public API
    router.rs                    # Memory-aware router
    precision_allocator.rs       # Per-expert precision
```

## 8. Performance Projections

### 8.1 Model Size Reduction

```
RuvLTRA-Small (0.5B parameters):

Format          Memory    vs FP16    Quality (est.)
---------------------------------------------------
FP16            1.0 GB    1.0x       100%
Q4_K_M          300 MB    3.3x       97%
Q2_K            160 MB    6.3x       85%
BitNet 1.58     130 MB    7.7x       88%
QAT 2-bit       125 MB    8.0x       93% (projected)
IQ2_XXS          130 MB    7.7x       90% (projected)
QAT+QuIP 2-bit   120 MB    8.3x       95% (projected)
```

### 8.2 Inference Speed

```
RuvLTRA-Small on Apple M4 Pro (estimated):

Format          Prefill (tok/s)    Decode (tok/s)    TTFT
---------------------------------------------------------
FP16            3,500              120               35ms
Q4_K_M          4,200              150               28ms
Q2_K            3,800              135               30ms
BitNet 1.58     4,500              170               25ms
QAT 2-bit       4,000              155               27ms
```

BitNet remains fastest due to multiplication-free inference.
QAT 2-bit offers better quality/speed tradeoff than Q2_K.

## 9. Recommendations

1. **Start with LoRA-QAT** (not full QAT): Train LoRA adapters on quantized
   base model. Uses existing MicroLoRA infrastructure. Lower memory requirements.

2. **Implement STE in training loop**: Add differentiable quantization to
   `training/real_trainer.rs`. This is the minimum viable change.

3. **Add I-quant dequantization**: Implement IQ2_XXS/IQ2_XS dequant kernels
   in `gguf/quantization.rs` to unlock llama.cpp 2-bit models.

4. **Build calibration pipeline**: Reuse existing dataset generators
   (`tool_dataset.rs`, `claude_dataset.rs`) for mixed-domain calibration.

5. **Extend KV cache to Q2**: The two-tier architecture already supports
   configurable precision. Adding Q2 cold store is straightforward.
