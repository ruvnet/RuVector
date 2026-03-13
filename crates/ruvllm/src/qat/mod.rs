//! Quantization-Aware Training (QAT) Module
//!
//! This module implements QAT infrastructure for ruvLLM as specified in ADR-090 Phase 2.
//! QAT enables training models with quantization in the loop, preserving ~90% of reasoning
//! capability at 2-3 bit precision vs ~40% for post-training quantization (PTQ).
//!
//! ## Module Structure
//!
//! ```text
//! qat/
//! +-- mod.rs              # This file: public API and documentation
//! +-- config.rs           # QatConfig, SteVariant, QuantGranularity
//! +-- ste.rs              # Straight-Through Estimator implementations
//! +-- differentiable_quant.rs  # DifferentiableQuantizer trait and impls
//! +-- calibration.rs      # CalibrationEngine for scale initialization
//! +-- distillation.rs     # Knowledge distillation loss (L_task + L_KD)
//! +-- reasoning_loss.rs   # Chain-of-thought fidelity loss
//! +-- training_loop.rs    # QatTrainer orchestrator
//! +-- lora_qat.rs         # LoRA-QAT integration
//! ```
//!
//! ## Architecture
//!
//! The QAT system consists of:
//!
//! 1. **Configuration** (`config.rs`): Training hyperparameters, STE variants, loss weights
//! 2. **STE Backward Pass** (`ste.rs`): Gradient flow through quantization operations
//! 3. **Differentiable Quantizers** (`differentiable_quant.rs`): Forward/backward quantization
//! 4. **Calibration** (`calibration.rs`): Scale initialization from activation statistics
//! 5. **Distillation** (`distillation.rs`): Knowledge distillation from teacher model
//! 6. **Reasoning Loss** (`reasoning_loss.rs`): Chain-of-thought preservation
//! 7. **Training Loop** (`training_loop.rs`): Full pipeline orchestration
//! 8. **LoRA-QAT** (`lora_qat.rs`): Memory-efficient fine-tuning with quantization
//!
//! ## System Invariants (ADR-090)
//!
//! | Invariant | Description | Module |
//! |-----------|-------------|--------|
//! | INV-1 | STE gradient flow - no zero regions except clipping | `ste.rs` |
//! | INV-2 | Scale positivity (alpha > 0) | `differentiable_quant.rs`, `calibration.rs` |
//! | INV-3 | Step size constraint (step = alpha * pi / k) | `differentiable_quant.rs` |
//! | INV-5 | Calibration artifacts serializable | `calibration.rs` |
//! | INV-6 | LoRA rank constraints (r <= min(d_in, d_out)) | `lora_qat.rs` |
//!
//! ## Usage
//!
//! ### Basic QAT Configuration
//!
//! ```rust,ignore
//! use ruvllm::qat::{QatConfig, SteVariant, QuantGranularity};
//!
//! // Create default 4-bit QAT config
//! let config = QatConfig::default();
//!
//! // Create 3-bit Pi-quantization config (PiQ3)
//! let piq3_config = QatConfig::piq3();
//!
//! // Custom configuration with builder pattern
//! let custom = QatConfig::default()
//!     .with_bits(3)
//!     .with_ste(SteVariant::LearnedStepSize)
//!     .with_granularity(QuantGranularity::PerChannel)
//!     .with_epochs(5)
//!     .with_learning_rate(1e-4);
//!
//! // Validate configuration
//! custom.validate()?;
//! ```
//!
//! ### Using Differentiable Quantizers
//!
//! ```rust,ignore
//! use ruvllm::qat::{QatConfig, DifferentiableQuantizer, PiQuantDifferentiable, create_quantizer};
//!
//! // Create quantizer from config
//! let config = QatConfig::piq3();
//! let mut quantizer = create_quantizer(&config);
//!
//! // Initialize scales from weight statistics
//! let weights: Vec<f32> = load_weights();
//! quantizer.init_scale_from_weights(&weights);
//!
//! // Forward pass (during inference or training)
//! let (q_int, q_dequant) = quantizer.forward(&weights);
//!
//! // Backward pass (during training)
//! let grad_out = compute_loss_gradient(&q_dequant);
//! let grad_weights = quantizer.backward(&weights, &q_dequant, &grad_out);
//! ```
//!
//! ### STE Variants
//!
//! ```rust,ignore
//! use ruvllm::qat::SteVariant;
//!
//! // Standard STE (identity gradient)
//! let standard = SteVariant::Standard;
//! assert_eq!(standard.backward(0.5, 0.4, 1.0), 1.0);
//!
//! // Clipped STE (zero gradient outside range)
//! let clipped = SteVariant::Clipped { clip_val: 1.0 };
//! assert_eq!(clipped.backward(1.5, 1.0, 1.0), 0.0);  // Outside range
//!
//! // EWGS (gradient scaling for better convergence)
//! let ewgs = SteVariant::Ewgs { lambda: 0.1 };
//! let grad = ewgs.backward(0.5, 0.3, 1.0);  // > 1.0 due to scaling
//!
//! // Learned Step Size (for adaptive quantization)
//! let lsq = SteVariant::LearnedStepSize;
//! ```
//!
//! ## Performance Targets (ADR-090)
//!
//! | Metric | Target | Measurement |
//! |--------|--------|-------------|
//! | QAT step time (0.5B model) | <500 ms | Per training step |
//! | LoRA-QAT memory (0.5B) | <2 GB | Total GPU memory |
//! | Scale gradient computation | <1 ms | Per layer |
//!
//! ## References
//!
//! - ADR-090: Ultra-Low-Bit QAT & Pi-Quantization
//! - Bengio et al., "Estimating or Propagating Gradients Through Stochastic Neurons"
//! - Esser et al., "Learned Step Size Quantization" (LSQ)
//! - Lee et al., "Element-Wise Gradient Scaling" (EWGS)

// ============================================================================
// Module Declarations
// ============================================================================

mod calibration;
mod config;
mod differentiable_quant;
mod distillation;
mod lora_qat;
mod reasoning_loss;
mod ste;
mod training_loop;

// ============================================================================
// Public Re-exports
// ============================================================================

// Configuration types
pub use config::{QatConfig, QatLossWeights, QuantGranularity, SteVariant};

// Differentiable quantization
pub use differentiable_quant::{
    create_quantizer, DifferentiableQuantizer, PiQuantDifferentiable, UniformQuantizer,
};

// Calibration (ADR-090 Phase 2)
pub use calibration::{
    init_pi_scale, CalibrationConfig, CalibrationDomain, CalibrationEngine, CalibrationMethod,
    CalibrationResult, CalibrationSample, CalibrationStats,
};

// Distillation loss (ADR-090 Phase 2)
pub use distillation::{DistillationConfig, DistillationLoss, DistillationStats, TeacherOutput};

// Reasoning loss (ADR-090 Phase 2)
pub use reasoning_loss::{
    ChainOfThoughtLoss, ReasoningConfig, ReasoningMetrics, ReasoningStep, StepType,
};

// Training loop (ADR-090 Phase 2)
pub use training_loop::{
    EpochMetrics, QatEvent, QatTrainer, StepMetrics, TrainingBatch, TrainingCheckpoint,
    TrainingPhase, TrainingResult,
};

// LoRA-QAT integration (ADR-090 Phase 2)
pub use lora_qat::{LoraGradients, LoraQatConfig, LoraQatLayer, LoraQatModel, LoraWeights};

// STE SIMD optimizations (platform-specific)
#[cfg(target_arch = "aarch64")]
pub use ste::simd as ste_simd;

// ============================================================================
// Module-Level Constants
// ============================================================================

/// Default bit width for QAT
pub const DEFAULT_BITS: u8 = 4;

/// Default learning rate for quantization parameters
pub const DEFAULT_QAT_LR: f32 = 1e-4;

/// Maximum supported bit width
pub const MAX_BITS: u8 = 8;

/// Minimum supported bit width
pub const MIN_BITS: u8 = 2;

// ============================================================================
// Convenience Functions
// ============================================================================

/// Create a PiQ3 quantizer (3-bit Pi-quantization with k=4)
///
/// This is the recommended configuration for 3-bit quantization,
/// providing ~0.5 effective bit improvement over uniform quantization.
///
/// # Example
///
/// ```rust,ignore
/// let quantizer = ruvllm::qat::piq3_quantizer();
/// ```
pub fn piq3_quantizer() -> PiQuantDifferentiable {
    PiQuantDifferentiable::piq3()
}

/// Create a PiQ2 quantizer (2-bit Pi-quantization with k=3)
///
/// For 2-bit quantization. Typically used with incoherence processing
/// (Hadamard rotation) for best results.
///
/// # Example
///
/// ```rust,ignore
/// let quantizer = ruvllm::qat::piq2_quantizer();
/// ```
pub fn piq2_quantizer() -> PiQuantDifferentiable {
    PiQuantDifferentiable::piq2()
}

/// Create a uniform quantizer with standard STE
///
/// # Arguments
///
/// * `bits` - Number of bits (2-8)
///
/// # Example
///
/// ```rust,ignore
/// let quantizer = ruvllm::qat::uniform_quantizer(4);
/// ```
pub fn uniform_quantizer(bits: u8) -> UniformQuantizer {
    UniformQuantizer::new(bits, SteVariant::Standard)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify all expected types are exported
        let _config = QatConfig::default();
        let _weights = QatLossWeights::default();
        let _granularity = QuantGranularity::default();
        let _ste = SteVariant::Standard;

        let _piq3 = piq3_quantizer();
        let _piq2 = piq2_quantizer();
        let _uniform = uniform_quantizer(4);
    }

    #[test]
    fn test_create_quantizer_from_config() {
        let config = QatConfig::piq3();
        let quantizer = create_quantizer(&config);

        assert_eq!(quantizer.bits(), 3);
    }

    #[test]
    fn test_default_constants() {
        assert_eq!(DEFAULT_BITS, 4);
        assert_eq!(MIN_BITS, 2);
        assert_eq!(MAX_BITS, 8);
        assert!(DEFAULT_QAT_LR > 0.0);
    }

    #[test]
    fn test_piq3_convenience() {
        let quantizer = piq3_quantizer();
        assert_eq!(quantizer.bits(), 3);
        assert_eq!(quantizer.k(), 4);
    }

    #[test]
    fn test_piq2_convenience() {
        let quantizer = piq2_quantizer();
        assert_eq!(quantizer.bits(), 2);
        assert_eq!(quantizer.k(), 3);
    }

    #[test]
    fn test_uniform_convenience() {
        let quantizer = uniform_quantizer(4);
        assert_eq!(quantizer.bits(), 4);
    }

    #[test]
    fn test_end_to_end_qat_forward_backward() {
        // Simulate a QAT training step
        let config = QatConfig::piq3();
        let quantizer = create_quantizer(&config);

        // Sample weights
        let weights: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();

        // Forward pass
        let (q_int, q_dequant) = quantizer.forward(&weights);
        assert_eq!(q_int.len(), 256);
        assert_eq!(q_dequant.len(), 256);

        // Simulate upstream gradient
        let grad_out: Vec<f32> = vec![1.0; 256];

        // Backward pass
        let grad_weights = quantizer.backward(&weights, &q_dequant, &grad_out);
        assert_eq!(grad_weights.len(), 256);

        // With LearnedStepSize STE, gradients should pass through
        for g in &grad_weights {
            assert!(g.is_finite());
        }
    }

    #[test]
    fn test_config_serialization() {
        let config = QatConfig::piq3().with_epochs(10).with_learning_rate(5e-5);

        let json = config.to_json().unwrap();
        let restored = QatConfig::from_json(&json).unwrap();

        assert_eq!(config.bits, restored.bits);
        assert_eq!(config.epochs, restored.epochs);
        assert_eq!(config.pi_k, restored.pi_k);
    }
}
