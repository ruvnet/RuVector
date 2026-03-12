# Implementation Plan: 2-Bit QAT and Pi-Quantization in Rust/ruvLLM

## Abstract

This document presents a concrete implementation plan for adding ultra-low-bit
quantization-aware training (QAT), QuIP incoherence processing, and pi-constant
quantization to the ruvLLM Rust crate. The plan leverages existing infrastructure
(BitNet, K-quants, SONA, LoRA, training loops) and defines new modules needed
to achieve 2-bit and 3-bit deployment with reasoning preservation.

## 1. Architecture Overview

### 1.1 New Module Map

```
crates/ruvllm/src/
  qat/                              # NEW: Quantization-Aware Training
    mod.rs                          # Public API
    config.rs                       # QAT configuration types
    ste.rs                          # Straight-through estimator variants
    differentiable_quant.rs         # Differentiable quantization ops
    calibration.rs                  # Mixed-domain calibration pipeline
    distillation.rs                 # Teacher-student distillation
    reasoning_loss.rs               # Chain-of-thought preservation loss
    training_loop.rs                # Main QAT training loop
    lora_qat.rs                     # LoRA-QAT (lightweight alternative)

  quantize/
    mod.rs                          # UPDATED: add new re-exports
    ruvltra_quant.rs                # EXISTING: K-quant pipeline
    incoherence.rs                  # NEW: QuIP incoherence processing
    hadamard.rs                     # NEW: Fast Walsh-Hadamard transform
    importance.rs                   # NEW: Weight importance computation
    iq_quant.rs                     # NEW: I-quant quantization pipeline
    pi_quant.rs                     # NEW: Pi-constant quantization
    pi_quant_simd.rs                # NEW: SIMD kernels for pi-quant

  moe/                              # NEW: Memory-Aware MoE
    mod.rs                          # Public API
    router.rs                       # Memory-aware routing
    expert_manager.rs               # Expert lifecycle + paging
    precision_allocator.rs          # Per-expert mixed precision
    sram_mapper.rs                  # Hardware memory hierarchy
```

### 1.2 Dependency Graph

```
Pi-Quant ----+
             |
K-Quants     +---> QAT Training Loop ---> SONA Integration
             |           |
Incoherence -+     Distillation
                        |
BitNet -----> LoRA-QAT -+
                        |
                   Calibration ---> Dataset Generators (existing)
```

## 2. Phase 1: Foundation (Weeks 1-3)

### 2.1 Differentiable Quantization Primitives

**File: `qat/ste.rs`**

```rust
/// Straight-Through Estimator variants
pub enum SteVariant {
    /// Standard STE: pass gradient through unchanged
    Standard,
    /// Clipped STE: zero gradient outside quantization range
    Clipped { clip_val: f32 },
    /// Learned Step Size Quantization
    LearnedStepSize { step_size: f32, step_grad: f32 },
    /// Elastic Weight Gradient Scaling
    Ewgs { lambda: f32 },
}

/// Forward + backward through quantization
pub struct QuantizationOp {
    pub variant: SteVariant,
    pub bits: u8,
    pub granularity: QuantGranularity,
}

impl QuantizationOp {
    /// Forward pass: quantize weights
    pub fn forward(&self, weights: &[f32], scales: &[f32]) -> Vec<f32> {
        // ... quantize using current grid
    }

    /// Backward pass: compute gradients through STE
    pub fn backward(
        &self,
        weights: &[f32],
        quantized: &[f32],
        grad_output: &[f32],
    ) -> SteGradients {
        match self.variant {
            SteVariant::Standard => {
                SteGradients {
                    weight_grad: grad_output.to_vec(),
                    scale_grad: None,
                }
            }
            SteVariant::Clipped { clip_val } => {
                let weight_grad: Vec<f32> = weights.iter()
                    .zip(grad_output.iter())
                    .map(|(&w, &g)| {
                        if w.abs() <= clip_val { g } else { 0.0 }
                    })
                    .collect();
                SteGradients { weight_grad, scale_grad: None }
            }
            SteVariant::Ewgs { lambda } => {
                let weight_grad: Vec<f32> = weights.iter()
                    .zip(quantized.iter())
                    .zip(grad_output.iter())
                    .map(|((&w, &q), &g)| {
                        g * (1.0 + lambda * (w - q).abs())
                    })
                    .collect();
                SteGradients { weight_grad, scale_grad: None }
            }
            SteVariant::LearnedStepSize { .. } => {
                // LSQ gradient computation
                todo!("Implement LSQ backward")
            }
        }
    }
}
```

**File: `qat/differentiable_quant.rs`**

```rust
/// Differentiable quantization function supporting multiple formats
pub trait DifferentiableQuantizer {
    /// Forward: quantize weights (returns quantized + aux data for backward)
    fn quantize_forward(&self, weights: &Tensor) -> QuantForwardResult;

    /// Backward: compute gradients through quantization
    fn quantize_backward(
        &self,
        aux: &QuantAuxData,
        grad_output: &Tensor,
    ) -> QuantBackwardResult;
}

/// 2-bit uniform differentiable quantizer
pub struct Uniform2BitQuantizer {
    pub ste: SteVariant,
    pub per_channel: bool,
}

/// Pi-scaled differentiable quantizer
pub struct PiQuantizer {
    pub bits: u8,
    pub k: u8,
    pub alpha: Vec<f32>,  // learnable per-channel scales
    pub ste: SteVariant,
}

/// BitNet ternary differentiable quantizer
pub struct TernaryQuantizer {
    pub ste: SteVariant,
    pub use_pi_scaling: bool,  // pi-scaled ternary variant
}
```

### 2.2 Pi-Quantization Module

**File: `quantize/pi_quant.rs`**

Core implementation as described in [document 07](07-3int-pi-constant-quantization.md).
Key types:

```rust
pub struct PiQuantConfig { bits, k, alpha, mixed_constants }
pub struct PiQuantizedTensor { data, config, mse }
pub struct Pi3BitBlock { packed: [u8; 3], scale: f16 }

pub fn pi_quantize_tensor(weights: &[f32], config: &PiQuantConfig) -> PiQuantizedTensor;
pub fn pi_dequantize(tensor: &PiQuantizedTensor) -> Vec<f32>;
pub fn pi_quantize_ste(...) -> PiQuantSteResult;
```

### 2.3 Hadamard Transform for Incoherence

**File: `quantize/hadamard.rs`**

```rust
/// Fast Walsh-Hadamard transform
pub struct HadamardTransform {
    log_dim: u32,
    signs: Vec<i8>,  // random +/-1 for randomized Hadamard
}

impl HadamardTransform {
    pub fn new(dim: usize) -> Self;

    /// In-place forward transform: O(n log n)
    pub fn forward(&self, data: &mut [f32]);

    /// In-place inverse transform: O(n log n)
    pub fn inverse(&self, data: &mut [f32]);
}

/// Apply incoherence processing to weight matrix
pub fn make_incoherent(
    weights: &[f32],
    rows: usize,
    cols: usize,
) -> IncoherentWeights;

pub struct IncoherentWeights {
    pub data: Vec<f32>,
    pub row_transform: HadamardTransform,
    pub col_transform: HadamardTransform,
}
```

## 3. Phase 2: Training Infrastructure (Weeks 4-6)

### 3.1 Mixed-Domain Calibration

**File: `qat/calibration.rs`**

```rust
/// Calibration dataset composition
pub struct CalibrationConfig {
    /// Math reasoning samples (GSM8K-style)
    pub math_samples: usize,         // default: 2048
    /// Code generation samples
    pub code_samples: usize,         // default: 1024
    /// Natural language samples (C4/RedPajama)
    pub language_samples: usize,     // default: 4096
    /// Structured reasoning samples
    pub reasoning_samples: usize,    // default: 2048
    /// Tool use samples (from tool_dataset.rs)
    pub tool_use_samples: usize,     // default: 1024
}

/// Run calibration to initialize quantization parameters
pub fn calibrate(
    model: &RuvLLMModel,
    config: &CalibrationConfig,
    target_bits: u8,
) -> CalibrationResult {
    // 1. Collect activations from each domain
    // 2. Compute per-channel Fisher information
    // 3. Initialize quantization grids (uniform, pi-scaled, or learned)
    // 4. Return per-layer quantization parameters
}
```

### 3.2 Teacher-Student Distillation

**File: `qat/distillation.rs`**

```rust
/// Distillation configuration
pub struct DistillConfig {
    /// Path to teacher model (full precision)
    pub teacher_path: PathBuf,
    /// Distillation temperature
    pub temperature: f32,        // default: 4.0
    /// Task loss weight
    pub alpha_task: f32,         // default: 1.0
    /// KD loss weight
    pub beta_kd: f32,            // default: 0.5
    /// Reasoning loss weight
    pub gamma_reasoning: f32,    // default: 2.0
}

/// Compute composite distillation loss
pub fn distillation_loss(
    student_logits: &Tensor,
    teacher_logits: &Tensor,
    targets: &Tensor,
    config: &DistillConfig,
) -> (f32, DistillLossBreakdown) {
    let l_task = cross_entropy(student_logits, targets);
    let l_kd = kl_divergence(
        &softmax_temperature(student_logits, config.temperature),
        &softmax_temperature(teacher_logits, config.temperature),
    ) * config.temperature.powi(2);
    let l_total = config.alpha_task * l_task + config.beta_kd * l_kd;
    (l_total, DistillLossBreakdown { l_task, l_kd, l_total })
}
```

### 3.3 Reasoning-Preserving Loss

**File: `qat/reasoning_loss.rs`**

```rust
/// Chain-of-thought fidelity loss
///
/// Measures divergence between student and teacher at each
/// reasoning step, not just the final answer.
pub fn reasoning_fidelity_loss(
    student_step_logits: &[Tensor],   // logits at each CoT step
    teacher_step_logits: &[Tensor],   // teacher's corresponding logits
) -> f32 {
    student_step_logits.iter()
        .zip(teacher_step_logits.iter())
        .map(|(s, t)| {
            kl_divergence(&softmax(s), &softmax(t))
        })
        .sum::<f32>() / student_step_logits.len() as f32
}
```

### 3.4 QAT Training Loop

**File: `qat/training_loop.rs`**

```rust
/// Main QAT training configuration
pub struct QatTrainingConfig {
    /// Quantization target
    pub quant_config: QuantTargetConfig,
    /// Distillation settings
    pub distill_config: Option<DistillConfig>,
    /// Calibration settings
    pub calibration_config: CalibrationConfig,
    /// Training hyperparameters
    pub learning_rate: f32,          // default: 1e-5
    pub epochs: usize,               // default: 3
    pub batch_size: usize,           // default: 4
    pub gradient_accumulation: usize, // default: 8
    /// Use LoRA-QAT instead of full QAT
    pub use_lora: bool,              // default: true
    pub lora_rank: usize,            // default: 16
}

/// Quantization target configuration
pub enum QuantTargetConfig {
    /// Uniform 2-bit
    Uniform2Bit { ste: SteVariant },
    /// Pi-scaled 3-bit
    PiQuant3 { k: u8 },
    /// Pi-scaled 2-bit
    PiQuant2 { k: u8 },
    /// BitNet ternary with QAT
    TernaryQat,
    /// Mixed-precision (ParetoQ-style)
    MixedPrecision { layer_bits: Vec<u8> },
}

/// Run QAT training
pub async fn train_qat(
    model: &mut RuvLLMModel,
    config: &QatTrainingConfig,
    dataset: &dyn QatDataset,
) -> Result<QatTrainingResult> {
    // Phase 1: Calibration
    let quant_params = calibrate(model, &config.calibration_config, target_bits)?;

    // Phase 2: Initialize quantization
    apply_quantization_params(model, &quant_params)?;

    // Phase 3: Optional - load teacher
    let teacher = if let Some(ref dc) = config.distill_config {
        Some(load_teacher_model(&dc.teacher_path)?)
    } else {
        None
    };

    // Phase 4: Training loop
    let optimizer = AdamW::new(model.trainable_params(), config.learning_rate);

    for epoch in 0..config.epochs {
        for batch in dataset.batches(config.batch_size) {
            // Forward with simulated quantization
            model.enable_quantization_simulation();
            let student_output = model.forward(&batch)?;

            // Teacher forward (if distillation)
            let loss = if let Some(ref teacher) = teacher {
                let teacher_output = teacher.forward(&batch)?;
                distillation_loss(
                    &student_output.logits,
                    &teacher_output.logits,
                    &batch.targets,
                    config.distill_config.as_ref().unwrap(),
                ).0
            } else {
                cross_entropy(&student_output.logits, &batch.targets)
            };

            // Backward through STE
            loss.backward()?;

            // Update latent weights (or LoRA params)
            optimizer.step()?;
            optimizer.zero_grad()?;
        }

        // Epoch evaluation
        let eval = evaluate_quantized(model, &eval_dataset)?;
        log::info!("Epoch {}: loss={:.4}, ppl={:.2}", epoch, eval.loss, eval.perplexity);
    }

    // Phase 5: Export quantized model
    let result = export_quantized_model(model, &quant_params)?;
    Ok(result)
}
```

## 4. Phase 3: LoRA-QAT Integration (Weeks 7-8)

### 4.1 LoRA-QAT

The lightweight alternative: train LoRA adapters on a quantized base model.

**File: `qat/lora_qat.rs`**

```rust
/// LoRA-QAT: fine-tune adapters on quantized base
pub struct LoraQatConfig {
    /// Base quantization format
    pub base_quant: QuantTargetConfig,
    /// LoRA rank (higher than MicroLoRA for QAT)
    pub lora_rank: usize,          // default: 16
    /// LoRA alpha scaling
    pub lora_alpha: f32,           // default: 32.0
    /// Target modules
    pub target_modules: Vec<String>, // ["q_proj", "k_proj", "v_proj", "o_proj"]
    /// Distillation config
    pub distill: Option<DistillConfig>,
}

/// Advantages over full QAT:
/// - Memory: ~50 MB optimizer state (vs ~114 GB for full QAT on 7B)
/// - Speed: 1 epoch convergence (vs 3 for full QAT)
/// - Flexibility: Different adapters per task on same quantized base
/// - Integration: Reuses existing MicroLoRA/adapter infrastructure
pub fn train_lora_qat(
    base_model: &QuantizedModel,    // frozen, quantized base
    config: &LoraQatConfig,
    dataset: &dyn QatDataset,
) -> Result<Vec<LoraAdapter>> {
    // 1. Initialize LoRA adapters on quantized model
    // 2. Forward: base_quant(W) + B@A*alpha/r
    // 3. Only B, A get gradients
    // 4. Much lower memory footprint
    todo!()
}
```

### 4.2 Integration with Existing LoRA System

```rust
// In lora/adapter.rs -- add QAT-aware adapter variant

pub enum AdapterMode {
    /// Standard LoRA on FP16 base
    Standard,
    /// MicroLoRA rank-1 for per-request adaptation
    Micro,
    /// QAT-LoRA on quantized base
    Qat {
        base_quant_format: QuantTargetConfig,
        compensate_quant_error: bool,
    },
}
```

## 5. Phase 4: MoE Integration (Weeks 9-10)

### 5.1 Memory-Aware Router

```rust
// moe/router.rs

pub struct MemoryAwareRouter {
    /// Base routing weights
    router_weights: Tensor,
    /// Expert cache state
    cache_state: ExpertCacheState,
    /// Memory bonus for cached experts
    cache_bonus: f32,
    /// Historical preference (EMA)
    preference_history: Vec<f32>,
    /// Decay factor for preference history
    preference_decay: f32,
}

impl MemoryAwareRouter {
    pub fn route(
        &mut self,
        input: &Tensor,
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        // Base routing logits
        let logits = self.router_weights.matmul(input);

        // Add memory-aware bonus
        let adjusted: Vec<f32> = logits.iter()
            .enumerate()
            .map(|(i, &l)| {
                let memory_bonus = if self.cache_state.is_hot(i) {
                    self.cache_bonus
                } else {
                    0.0
                };
                let history_bonus = self.preference_history[i] * 0.1;
                l + memory_bonus + history_bonus
            })
            .collect();

        // Select top-K
        let selected = top_k_indices(&adjusted, top_k);

        // Update preference history
        for &(idx, _) in &selected {
            self.preference_history[idx] = self.preference_decay
                * self.preference_history[idx]
                + (1.0 - self.preference_decay);
        }

        selected
    }
}
```

### 5.2 Per-Expert Mixed Precision

```rust
// moe/precision_allocator.rs

pub struct ExpertPrecisionMap {
    /// Precision per expert (indexed by expert_id)
    precision: Vec<QuantTargetConfig>,
}

impl ExpertPrecisionMap {
    /// Allocate precision based on usage frequency
    pub fn from_usage_stats(
        stats: &[ExpertUsageStats],
        memory_budget: usize,
    ) -> Self {
        let mut precision = vec![QuantTargetConfig::Uniform2Bit {
            ste: SteVariant::Standard
        }; stats.len()];

        // Sort by usage frequency
        let mut sorted: Vec<(usize, f32)> = stats.iter()
            .enumerate()
            .map(|(i, s)| (i, s.usage_fraction))
            .collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Top experts get higher precision
        let mut remaining_budget = memory_budget;
        for &(expert_id, _usage) in &sorted {
            if remaining_budget > expert_size_at_bits(4) {
                precision[expert_id] = QuantTargetConfig::PiQuant3 { k: 4 };
                remaining_budget -= expert_size_at_bits(3);
            } else {
                break;
            }
        }

        Self { precision }
    }
}
```

## 6. Phase 5: SONA Integration (Weeks 11-12)

### 6.1 Post-Deployment Continual QAT

Integrate QAT with SONA's three-tier learning:

```rust
// In sona/integration.rs -- extend existing tiers

/// Tier 1 (Instant, <1ms): MicroLoRA on quantized base
///   - Existing MicroLoRA works unchanged
///   - LoRA delta applied on top of quantized weights

/// Tier 2 (Background, ~100ms): Quantization parameter update
///   - Update per-channel scales (alpha in pi-quant)
///   - Adjust expert precision allocation
///   - EWC++ Fisher update for quantized parameters

/// Tier 3 (Deep, minutes): Periodic re-calibration
///   - Re-run mixed-domain calibration with new data
///   - Re-optimize quantization grids
///   - Merge accumulated LoRA deltas into base (re-quantize)

pub struct QuantizedSonaConfig {
    /// Enable scale adaptation in Tier 2
    pub adapt_scales: bool,
    /// Re-calibration interval (Tier 3)
    pub recalibration_interval: Duration,
    /// EWC lambda for quantization parameters
    pub quant_ewc_lambda: f32,
}
```

## 7. Phase 6: Evaluation and Benchmarking (Weeks 13-14)

### 7.1 Benchmark Suite

```rust
// In evaluation/ -- extend existing harness

pub struct QuantBenchmarkSuite {
    pub configs: Vec<QuantBenchmarkConfig>,
}

pub struct QuantBenchmarkConfig {
    pub name: String,
    pub quant_format: QuantTargetConfig,
    pub use_qat: bool,
    pub use_distillation: bool,
    pub use_incoherence: bool,
}

// Default suite:
pub fn default_quant_benchmarks() -> Vec<QuantBenchmarkConfig> {
    vec![
        // Baselines
        config("FP16", F16, false, false, false),
        config("Q4_K_M", Q4KM, false, false, false),
        config("Q2_K", Q2K, false, false, false),
        config("BitNet", Ternary, false, false, false),

        // Pi-quantization variants
        config("Pi-Q3", PiQuant3{k:4}, false, false, false),
        config("Pi-Q3+QAT", PiQuant3{k:4}, true, true, false),
        config("Pi-Q2", PiQuant2{k:3}, false, false, false),
        config("Pi-Q2+QAT", PiQuant2{k:3}, true, true, false),

        // Incoherence variants
        config("Q2+QuIP", Q2K, false, false, true),
        config("Q2+QuIP+QAT", Q2K, true, true, true),

        // Combined
        config("Pi-Q2+QuIP+QAT", PiQuant2{k:3}, true, true, true),
    ]
}
```

### 7.2 Evaluation Datasets

```
Dataset          Task                 Metric
---------------------------------------------
WikiText-2       Language modeling    Perplexity
C4               Language modeling    Perplexity
GSM8K            Math reasoning       Accuracy
MATH             Math reasoning       Accuracy
HumanEval        Code generation      Pass@1
MMLU             Knowledge            Accuracy
ARC-Challenge    Reasoning            Accuracy
MCP Tool Use     Tool invocation      Tool accuracy
SWE-Bench        Code editing         Resolve rate
```

### 7.3 Hardware Targets

```
Platform              Benchmark
---------------------------------
Apple M4 Pro          tok/s, TTFT, memory, power
x86_64 (AVX2)        tok/s, memory
WASM (Chrome)         tok/s, memory
ARM Cortex-A78        tok/s, TTFT, power
ESP32-P4 (RISC-V)    tok/s (paged inference)
```

## 8. CLI Integration

### 8.1 New CLI Commands

```bash
# Quantize with pi-scaling
ruvllm quantize --format pi-q3 --k 4 --input model.safetensors --output model-piq3.gguf

# Run QAT training
ruvllm qat --format pi-q2 --teacher model-fp16.gguf --epochs 3 --use-lora --lora-rank 16

# Calibrate for QAT
ruvllm calibrate --format 2bit --domains math,code,language,reasoning --samples 10000

# Apply incoherence processing
ruvllm quantize --format q2-quip --incoherence hadamard --input model.safetensors

# Benchmark quantized models
ruvllm bench --configs "fp16,q4km,piq3,piq2,bitnet" --datasets "gsm8k,humaneval,mmlu"

# MoE expert precision allocation
ruvllm moe-precision --experts 16 --memory-budget 256MB --usage-stats usage.json
```

### 8.2 Configuration File

```toml
# ruvllm-qat.toml

[quantization]
format = "pi-q3"
k = 4
bits = 3
mixed_constants = false

[qat]
enabled = true
use_lora = true
lora_rank = 16
epochs = 3
learning_rate = 1e-5
batch_size = 4
gradient_accumulation = 8

[distillation]
teacher_path = "models/ruvltra-small-fp16.gguf"
temperature = 4.0
alpha_task = 1.0
beta_kd = 0.5
gamma_reasoning = 2.0

[calibration]
math_samples = 2048
code_samples = 1024
language_samples = 4096
reasoning_samples = 2048
tool_use_samples = 1024

[incoherence]
enabled = true
method = "hadamard"  # or "orthogonal" for full QuIP

[moe]
memory_aware_routing = true
cache_bonus = 0.5
preference_decay = 0.95

[sona]
adapt_scales = true
recalibration_interval = "1h"
quant_ewc_lambda = 100.0
```

## 9. Timeline Summary

```
Week  Phase                        Deliverables
-----------------------------------------------------
1-2   Foundation: STE + Pi-Quant   ste.rs, pi_quant.rs, pi_quant_simd.rs
3     Foundation: Incoherence      hadamard.rs, incoherence.rs, importance.rs
4-5   Training: Calibration + KD   calibration.rs, distillation.rs, reasoning_loss.rs
6     Training: QAT Loop           training_loop.rs, config.rs
7-8   LoRA-QAT                     lora_qat.rs, adapter.rs updates
9-10  MoE Integration              router.rs, precision_allocator.rs, sram_mapper.rs
11-12 SONA Integration             sona/integration.rs updates, continual QAT
13-14 Evaluation + CLI             bench suite, CLI commands, documentation
```

## 10. Success Criteria

```
Metric                          Target
-----------------------------------------
2-bit model size (0.5B)         < 130 MB
2-bit PPL (WikiText-2)          < 15.0 (vs 12.3 FP16)
2-bit GSM8K accuracy            > 35% (vs ~45% FP16)
Pi-Q3 PPL improvement vs Q3     > 0.5 PPL better
QAT training time (0.5B)        < 4 hours on single GPU
LoRA-QAT memory                 < 2 GB for 0.5B model
Inference speed (M4 Pro)        > 130 tok/s decode
SONA scale adaptation           < 100ms per update
MoE cache hit rate              > 70% with memory-aware routing
```
