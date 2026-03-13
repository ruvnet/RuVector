//! QAT Configuration Module
//!
//! This module defines configuration types for Quantization-Aware Training (QAT)
//! as specified in ADR-090 Phase 2.
//!
//! ## Configuration Hierarchy
//!
//! ```text
//! QatConfig
//! +-- bits: u8 (2, 3, 4, or 8)
//! +-- ste_variant: SteVariant
//! +-- granularity: QuantGranularity
//! +-- loss_weights: QatLossWeights
//! +-- epochs: usize
//! +-- learning_rate: f32
//! +-- pi_k: Option<u8> (for Pi-quantization)
//! ```
//!
//! ## System Invariants
//!
//! - **INV-2**: Scale positivity (alpha > 0 for Pi-quant)
//! - **INV-3**: Step size constraint (step = alpha * pi / k)

use serde::{Deserialize, Serialize};

// ============================================================================
// Quantization Granularity
// ============================================================================

/// Granularity level for quantization parameters
///
/// Controls how many quantization parameters (scale, zero-point) are used.
/// Finer granularity provides better accuracy but increases overhead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum QuantGranularity {
    /// Single scale/zero-point for entire tensor
    /// Lowest overhead, suitable for small tensors
    PerTensor,

    /// Scale/zero-point per output channel (typical for convolutions and linear layers)
    /// Best tradeoff for most use cases
    #[default]
    PerChannel,

    /// Scale/zero-point per token (for sequence models)
    /// Highest accuracy, highest overhead
    PerToken,

    /// Scale/zero-point per sub-block (K-quant style)
    /// Matches existing GGUF K-quant formats
    PerBlock {
        /// Block size (typically 32 or 256)
        block_size: usize,
    },
}

impl QuantGranularity {
    /// Get the number of quantization parameters per n elements
    pub fn params_per_elements(&self, n: usize, channels: usize) -> usize {
        match self {
            QuantGranularity::PerTensor => 2, // scale + zero_point
            QuantGranularity::PerChannel => channels * 2,
            QuantGranularity::PerToken => (n / channels) * 2,
            QuantGranularity::PerBlock { block_size } => ((n + block_size - 1) / block_size) * 2,
        }
    }
}

// ============================================================================
// STE Variant
// ============================================================================

/// Straight-Through Estimator (STE) variant for gradient computation
///
/// During QAT, the forward pass uses discrete quantized values, but gradients
/// must flow through. STE provides the gradient approximation.
///
/// ## Variants
///
/// | Variant | Formula | Use Case |
/// |---------|---------|----------|
/// | Standard | dw = dq | Default, works well for most cases |
/// | Clipped | dw = dq * 1{|w| <= c} | Prevents gradient explosion |
/// | LearnedStepSize | dw = dq, ds/dalpha computed | LSQ-style adaptive step |
/// | Ewgs | dw = dq * (1 + lambda * |w-q|) | Better convergence for outliers |
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SteVariant {
    /// Standard STE: gradient passes through unchanged
    /// dw = dq (identity, gradient passes through)
    Standard,

    /// Clipped STE: zero gradient outside clip range
    /// dw = dq * 1{|w| <= clip_val}, zero outside range
    Clipped {
        /// Clipping threshold for gradient
        clip_val: f32,
    },

    /// Learned Step Size: scale is learned alongside weights
    /// dw = dq (scale gradient computed separately in forward pass)
    LearnedStepSize,

    /// Element-Wise Gradient Scaling (EWGS)
    /// dw = dq * (1 + lambda * |w - q|)
    /// Stronger push toward stable quantization points
    Ewgs {
        /// Scaling factor for gradient adjustment
        lambda: f32,
    },
}

impl Default for SteVariant {
    fn default() -> Self {
        SteVariant::Standard
    }
}

impl SteVariant {
    /// Create a clipped STE with default clip value
    pub fn clipped() -> Self {
        SteVariant::Clipped { clip_val: 1.0 }
    }

    /// Create an EWGS STE with default lambda
    pub fn ewgs() -> Self {
        SteVariant::Ewgs { lambda: 0.1 }
    }

    /// Check if this variant requires scale gradient computation
    pub fn requires_scale_grad(&self) -> bool {
        matches!(self, SteVariant::LearnedStepSize)
    }
}

// ============================================================================
// QAT Loss Weights
// ============================================================================

/// Weights for composite QAT loss function
///
/// Total loss = lambda_task * L_task + lambda_kd * L_KD + lambda_reasoning * L_reasoning
///
/// ## Components
///
/// - **L_task**: Primary task loss (cross-entropy for LM)
/// - **L_KD**: Knowledge distillation from teacher model
/// - **L_reasoning**: Chain-of-thought fidelity loss
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QatLossWeights {
    /// Weight for primary task loss (typically 1.0)
    pub lambda_task: f32,

    /// Weight for knowledge distillation loss (typically 0.5-1.0)
    pub lambda_kd: f32,

    /// Weight for reasoning preservation loss (typically 0.1-0.5)
    pub lambda_reasoning: f32,

    /// Temperature for KD softmax (typically 2.0-4.0)
    pub kd_temperature: f32,
}

impl Default for QatLossWeights {
    fn default() -> Self {
        Self {
            lambda_task: 1.0,
            lambda_kd: 0.5,
            lambda_reasoning: 0.2,
            kd_temperature: 3.0,
        }
    }
}

impl QatLossWeights {
    /// Create weights for distillation-heavy training
    pub fn distillation_heavy() -> Self {
        Self {
            lambda_task: 0.5,
            lambda_kd: 1.0,
            lambda_reasoning: 0.3,
            kd_temperature: 4.0,
        }
    }

    /// Create weights for reasoning-focused training
    pub fn reasoning_focused() -> Self {
        Self {
            lambda_task: 1.0,
            lambda_kd: 0.3,
            lambda_reasoning: 0.5,
            kd_temperature: 2.0,
        }
    }

    /// Normalize weights to sum to 1.0 (for logging)
    pub fn normalized(&self) -> Self {
        let sum = self.lambda_task + self.lambda_kd + self.lambda_reasoning;
        if sum > 0.0 {
            Self {
                lambda_task: self.lambda_task / sum,
                lambda_kd: self.lambda_kd / sum,
                lambda_reasoning: self.lambda_reasoning / sum,
                kd_temperature: self.kd_temperature,
            }
        } else {
            Self::default()
        }
    }
}

// ============================================================================
// QAT Configuration
// ============================================================================

/// Main configuration for Quantization-Aware Training
///
/// This configuration drives the entire QAT pipeline including:
/// - Quantization bit width and format
/// - STE gradient approximation method
/// - Loss function composition
/// - Training hyperparameters
///
/// ## Example
///
/// ```rust,ignore
/// use ruvllm::qat::{QatConfig, SteVariant, QuantGranularity};
///
/// let config = QatConfig::default()
///     .with_bits(3)
///     .with_ste(SteVariant::LearnedStepSize)
///     .with_pi_k(4);  // Pi/4 step size for Pi-quantization
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QatConfig {
    /// Target bit width (2, 3, 4, or 8)
    pub bits: u8,

    /// STE variant for gradient computation
    pub ste_variant: SteVariant,

    /// Quantization granularity
    pub granularity: QuantGranularity,

    /// Loss function weights
    pub loss_weights: QatLossWeights,

    /// Number of training epochs
    pub epochs: usize,

    /// Learning rate for quantization parameters
    pub learning_rate: f32,

    /// Pi-quantization k value (step = alpha * pi / k)
    /// None means uniform quantization
    pub pi_k: Option<u8>,

    /// Whether to use incoherence processing (Hadamard rotation)
    pub use_incoherence: bool,

    /// Whether to use symmetric quantization (no zero-point)
    pub symmetric: bool,

    /// Warmup epochs before enabling full quantization
    pub warmup_epochs: usize,

    /// Gradient clipping value
    pub grad_clip: Option<f32>,

    /// Weight decay for optimizer
    pub weight_decay: f32,

    /// Enable teacher model distillation
    pub use_distillation: bool,

    /// Freeze teacher model weights (INV-6)
    pub freeze_teacher: bool,
}

impl Default for QatConfig {
    fn default() -> Self {
        Self {
            bits: 4,
            ste_variant: SteVariant::Standard,
            granularity: QuantGranularity::PerChannel,
            loss_weights: QatLossWeights::default(),
            epochs: 3,
            learning_rate: 1e-4,
            pi_k: None,
            use_incoherence: false,
            symmetric: true,
            warmup_epochs: 1,
            grad_clip: Some(1.0),
            weight_decay: 0.01,
            use_distillation: true,
            freeze_teacher: true, // INV-6: Teacher immutability
        }
    }
}

impl QatConfig {
    /// Create a new QAT config with specified bit width
    pub fn new(bits: u8) -> Self {
        Self {
            bits,
            ..Default::default()
        }
    }

    /// Create config for Pi-quantization (ADR-090)
    pub fn pi_quant(bits: u8, k: u8) -> Self {
        Self {
            bits,
            pi_k: Some(k),
            ste_variant: SteVariant::LearnedStepSize, // LSQ works well with pi-quant
            ..Default::default()
        }
    }

    /// Create config for 3-bit Pi-quantization (PiQ3)
    pub fn piq3() -> Self {
        Self::pi_quant(3, 4) // step = pi/4
    }

    /// Create config for 2-bit Pi-quantization (PiQ2)
    pub fn piq2() -> Self {
        Self {
            use_incoherence: true,  // 2-bit typically needs Hadamard
            ..Self::pi_quant(2, 3)  // step = pi/3
        }
    }

    // Builder methods

    /// Set bit width
    pub fn with_bits(mut self, bits: u8) -> Self {
        self.bits = bits;
        self
    }

    /// Set STE variant
    pub fn with_ste(mut self, ste: SteVariant) -> Self {
        self.ste_variant = ste;
        self
    }

    /// Set quantization granularity
    pub fn with_granularity(mut self, granularity: QuantGranularity) -> Self {
        self.granularity = granularity;
        self
    }

    /// Set loss weights
    pub fn with_loss_weights(mut self, weights: QatLossWeights) -> Self {
        self.loss_weights = weights;
        self
    }

    /// Set number of epochs
    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set Pi-quantization k value
    pub fn with_pi_k(mut self, k: u8) -> Self {
        self.pi_k = Some(k);
        self
    }

    /// Enable/disable incoherence processing
    pub fn with_incoherence(mut self, enable: bool) -> Self {
        self.use_incoherence = enable;
        self
    }

    /// Enable/disable symmetric quantization
    pub fn with_symmetric(mut self, symmetric: bool) -> Self {
        self.symmetric = symmetric;
        self
    }

    /// Set warmup epochs
    pub fn with_warmup(mut self, epochs: usize) -> Self {
        self.warmup_epochs = epochs;
        self
    }

    /// Set gradient clipping
    pub fn with_grad_clip(mut self, clip: Option<f32>) -> Self {
        self.grad_clip = clip;
        self
    }

    /// Enable/disable distillation
    pub fn with_distillation(mut self, enable: bool) -> Self {
        self.use_distillation = enable;
        self
    }

    // Validation and utility methods

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        // Validate bit width
        if !matches!(self.bits, 2 | 3 | 4 | 5 | 8) {
            return Err(format!(
                "Invalid bit width: {}. Must be 2, 3, 4, 5, or 8",
                self.bits
            ));
        }

        // Validate Pi-k value (INV-3)
        if let Some(k) = self.pi_k {
            if !matches!(k, 2 | 3 | 4 | 5) {
                return Err(format!("Invalid pi_k: {}. Must be 2, 3, 4, or 5", k));
            }
        }

        // Validate learning rate
        if self.learning_rate <= 0.0 {
            return Err("Learning rate must be positive".to_string());
        }

        // Validate epochs
        if self.epochs == 0 {
            return Err("Epochs must be greater than 0".to_string());
        }

        // Validate loss weights
        if self.loss_weights.lambda_task < 0.0
            || self.loss_weights.lambda_kd < 0.0
            || self.loss_weights.lambda_reasoning < 0.0
        {
            return Err("Loss weights must be non-negative".to_string());
        }

        if self.loss_weights.kd_temperature <= 0.0 {
            return Err("KD temperature must be positive".to_string());
        }

        // Validate clipped STE
        if let SteVariant::Clipped { clip_val } = self.ste_variant {
            if clip_val <= 0.0 {
                return Err("Clip value must be positive".to_string());
            }
        }

        // Validate EWGS lambda
        if let SteVariant::Ewgs { lambda } = self.ste_variant {
            if lambda < 0.0 {
                return Err("EWGS lambda must be non-negative".to_string());
            }
        }

        Ok(())
    }

    /// Get the number of quantization levels
    pub fn num_levels(&self) -> usize {
        1 << self.bits
    }

    /// Get the quantization range for signed symmetric
    pub fn symmetric_range(&self) -> (i32, i32) {
        let half = (1i32 << self.bits) / 2;
        (-half, half - 1)
    }

    /// Calculate step size for Pi-quantization
    /// Returns alpha * pi / k where alpha is the learned scale
    pub fn pi_step(&self, alpha: f32) -> Option<f32> {
        self.pi_k.map(|k| alpha * std::f32::consts::PI / (k as f32))
    }

    /// Check if using Pi-quantization
    pub fn is_pi_quant(&self) -> bool {
        self.pi_k.is_some()
    }

    /// Get effective bits per weight (including scale overhead)
    pub fn bits_per_weight(&self) -> f32 {
        let base_bits = self.bits as f32;
        // Add scale overhead based on granularity
        match self.granularity {
            QuantGranularity::PerTensor => base_bits + 0.0625, // ~1/16 bit overhead
            QuantGranularity::PerChannel => base_bits + 0.125, // ~1/8 bit overhead
            QuantGranularity::PerToken => base_bits + 0.25,    // ~1/4 bit overhead
            QuantGranularity::PerBlock { block_size } => {
                base_bits + 16.0 / (block_size as f32) // scale overhead per block
            }
        }
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = QatConfig::default();
        assert_eq!(config.bits, 4);
        assert_eq!(config.ste_variant, SteVariant::Standard);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_piq3_config() {
        let config = QatConfig::piq3();
        assert_eq!(config.bits, 3);
        assert_eq!(config.pi_k, Some(4));
        assert!(config.is_pi_quant());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_piq2_config() {
        let config = QatConfig::piq2();
        assert_eq!(config.bits, 2);
        assert_eq!(config.pi_k, Some(3));
        assert!(config.use_incoherence);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_builder_pattern() {
        let config = QatConfig::default()
            .with_bits(3)
            .with_ste(SteVariant::LearnedStepSize)
            .with_epochs(5)
            .with_learning_rate(2e-4)
            .with_pi_k(4);

        assert_eq!(config.bits, 3);
        assert_eq!(config.ste_variant, SteVariant::LearnedStepSize);
        assert_eq!(config.epochs, 5);
        assert_eq!(config.learning_rate, 2e-4);
        assert_eq!(config.pi_k, Some(4));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_bits() {
        let config = QatConfig::default().with_bits(6);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_pi_k() {
        let mut config = QatConfig::default();
        config.pi_k = Some(7);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_pi_step_calculation() {
        let config = QatConfig::piq3(); // k=4
        let alpha = 1.0;
        let step = config.pi_step(alpha).unwrap();
        let expected = std::f32::consts::PI / 4.0;
        assert!((step - expected).abs() < 1e-6);
    }

    #[test]
    fn test_symmetric_range() {
        let config = QatConfig::default().with_bits(3);
        let (min, max) = config.symmetric_range();
        assert_eq!(min, -4);
        assert_eq!(max, 3);
    }

    #[test]
    fn test_num_levels() {
        assert_eq!(QatConfig::default().with_bits(2).num_levels(), 4);
        assert_eq!(QatConfig::default().with_bits(3).num_levels(), 8);
        assert_eq!(QatConfig::default().with_bits(4).num_levels(), 16);
    }

    #[test]
    fn test_loss_weights() {
        let weights = QatLossWeights::default();
        assert_eq!(weights.lambda_task, 1.0);

        let normalized = weights.normalized();
        let sum = normalized.lambda_task + normalized.lambda_kd + normalized.lambda_reasoning;
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_granularity_params() {
        let per_tensor = QuantGranularity::PerTensor;
        assert_eq!(per_tensor.params_per_elements(1024, 64), 2);

        let per_channel = QuantGranularity::PerChannel;
        assert_eq!(per_channel.params_per_elements(1024, 64), 128);

        let per_block = QuantGranularity::PerBlock { block_size: 32 };
        assert_eq!(per_block.params_per_elements(1024, 64), 64); // 1024/32 * 2
    }

    #[test]
    fn test_ste_variants() {
        assert!(!SteVariant::Standard.requires_scale_grad());
        assert!(!SteVariant::clipped().requires_scale_grad());
        assert!(SteVariant::LearnedStepSize.requires_scale_grad());
        assert!(!SteVariant::ewgs().requires_scale_grad());
    }

    #[test]
    fn test_json_serialization() {
        let config = QatConfig::piq3();
        let json = config.to_json().unwrap();
        let restored = QatConfig::from_json(&json).unwrap();
        assert_eq!(config.bits, restored.bits);
        assert_eq!(config.pi_k, restored.pi_k);
    }

    #[test]
    fn test_bits_per_weight() {
        let config = QatConfig::default()
            .with_bits(4)
            .with_granularity(QuantGranularity::PerTensor);

        // Should be slightly more than 4.0 due to scale overhead
        assert!(config.bits_per_weight() > 4.0);
        assert!(config.bits_per_weight() < 4.2);
    }
}
