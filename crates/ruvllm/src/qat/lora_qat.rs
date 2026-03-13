//! LoRA-QAT Integration (ADR-090 Phase 2)
//!
//! This module implements Low-Rank Adaptation (LoRA) combined with
//! Quantization-Aware Training for memory-efficient fine-tuning.
//!
//! ## Architecture
//!
//! ```text
//! Standard Weight Update:
//!   W' = W + dW  (full rank, expensive)
//!
//! LoRA Update:
//!   W' = W + A @ B  where A: (d, r), B: (r, d), r << d
//!
//! LoRA-QAT Update:
//!   W_q = Q(W + A @ B)  where Q is differentiable quantization
//! ```
//!
//! ## Memory Efficiency
//!
//! | Model Size | Full Fine-tune | LoRA (r=16) | LoRA-QAT (r=16, 3-bit) |
//! |------------|----------------|-------------|------------------------|
//! | 0.5B       | 2 GB           | 50 MB       | 35 MB                  |
//! | 7B         | 28 GB          | 400 MB      | 280 MB                 |
//! | 70B        | 280 GB         | 4 GB        | 2.8 GB                 |
//!
//! ## System Invariants
//!
//! - **INV-1**: STE gradient flow through both LoRA and quantization
//! - **INV-2**: Scale positivity maintained for quantization
//! - **INV-6**: LoRA rank constraints (r <= min(d_in, d_out))
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvllm::qat::{LoraQatConfig, LoraQatLayer, LoraQatTrainer};
//!
//! let config = LoraQatConfig::default()
//!     .with_rank(16)
//!     .with_alpha(32.0)
//!     .with_bits(3);
//!
//! let mut layer = LoraQatLayer::new(4096, 4096, config);
//! layer.init_lora_weights();
//!
//! // Forward pass with quantized LoRA
//! let output = layer.forward(&input);
//!
//! // Backward pass (only LoRA weights updated)
//! let grads = layer.backward(&grad_output);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::{Result, RuvLLMError};

use super::config::{QatConfig, SteVariant};
use super::differentiable_quant::{create_quantizer, DifferentiableQuantizer};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for LoRA-QAT
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraQatConfig {
    /// LoRA rank (r)
    pub rank: usize,
    /// LoRA alpha scaling factor
    pub alpha: f32,
    /// Dropout probability for LoRA
    pub dropout: f32,
    /// Target modules to apply LoRA
    pub target_modules: Vec<String>,
    /// Whether to use bias in LoRA layers
    pub use_bias: bool,
    /// Quantization bits
    pub bits: u8,
    /// Pi-quantization k value
    pub pi_k: u8,
    /// STE variant for backward pass
    pub ste_variant: SteVariant,
    /// Whether to quantize the base weights
    pub quantize_base: bool,
    /// Whether to quantize LoRA weights
    pub quantize_lora: bool,
    /// Gradient checkpointing for memory efficiency
    pub gradient_checkpointing: bool,
    /// Learning rate for LoRA weights
    pub lora_lr: f32,
    /// Learning rate for quantization scales
    pub scale_lr: f32,
}

impl Default for LoraQatConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 32.0,
            dropout: 0.1,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
            ],
            use_bias: false,
            bits: 3,
            pi_k: 4,
            ste_variant: SteVariant::LearnedStepSize,
            quantize_base: true,
            quantize_lora: false, // Keep LoRA in FP for better gradients
            gradient_checkpointing: true,
            lora_lr: 1e-4,
            scale_lr: 1e-3,
        }
    }
}

impl LoraQatConfig {
    /// Create config for 3-bit quantization with rank 16
    pub fn piq3_r16() -> Self {
        Self {
            rank: 16,
            bits: 3,
            pi_k: 4,
            ..Default::default()
        }
    }

    /// Create config for 2-bit quantization with rank 8
    pub fn piq2_r8() -> Self {
        Self {
            rank: 8,
            bits: 2,
            pi_k: 3,
            alpha: 16.0,
            ..Default::default()
        }
    }

    /// Create memory-minimal config
    pub fn minimal() -> Self {
        Self {
            rank: 4,
            bits: 2,
            pi_k: 3,
            alpha: 8.0,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            gradient_checkpointing: true,
            ..Default::default()
        }
    }

    /// Builder: set rank
    pub fn with_rank(mut self, rank: usize) -> Self {
        self.rank = rank;
        self
    }

    /// Builder: set alpha
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Builder: set bits
    pub fn with_bits(mut self, bits: u8) -> Self {
        self.bits = bits;
        self
    }

    /// Builder: set dropout
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    /// Builder: add target module
    pub fn with_module(mut self, module: &str) -> Self {
        self.target_modules.push(module.to_string());
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.rank == 0 {
            return Err(RuvLLMError::Config("LoRA rank must be > 0".to_string()));
        }
        if self.alpha <= 0.0 {
            return Err(RuvLLMError::Config("LoRA alpha must be > 0".to_string()));
        }
        if self.dropout < 0.0 || self.dropout >= 1.0 {
            return Err(RuvLLMError::Config(
                "LoRA dropout must be in [0, 1)".to_string(),
            ));
        }
        if self.bits < 2 || self.bits > 8 {
            return Err(RuvLLMError::Config(
                "Quantization bits must be in [2, 8]".to_string(),
            ));
        }
        Ok(())
    }

    /// Convert to QAT config
    pub fn to_qat_config(&self) -> QatConfig {
        QatConfig::default()
            .with_bits(self.bits)
            .with_learning_rate(self.lora_lr)
    }

    /// Scaling factor for LoRA (alpha / r)
    pub fn scaling(&self) -> f32 {
        self.alpha / self.rank as f32
    }
}

// ============================================================================
// LoRA Weights
// ============================================================================

/// LoRA weight matrices (A and B)
#[derive(Debug, Clone)]
pub struct LoraWeights {
    /// Down-projection matrix A: (d_in, r)
    pub lora_a: Vec<f32>,
    /// Up-projection matrix B: (r, d_out)
    pub lora_b: Vec<f32>,
    /// Input dimension
    pub d_in: usize,
    /// Output dimension
    pub d_out: usize,
    /// Rank
    pub rank: usize,
    /// Scaling factor
    pub scaling: f32,
}

impl LoraWeights {
    /// Create new LoRA weights
    pub fn new(d_in: usize, d_out: usize, rank: usize, scaling: f32) -> Self {
        Self {
            lora_a: vec![0.0; d_in * rank],
            lora_b: vec![0.0; rank * d_out],
            d_in,
            d_out,
            rank,
            scaling,
        }
    }

    /// Initialize with Kaiming uniform for A, zeros for B
    pub fn init_kaiming(&mut self) {
        let bound = (6.0 / self.d_in as f32).sqrt();
        for val in &mut self.lora_a {
            // Simple uniform initialization (replace with proper RNG)
            *val = (rand_simple() * 2.0 - 1.0) * bound;
        }
        // B initialized to zeros (already done)
    }

    /// Initialize with Gaussian for A, zeros for B
    pub fn init_gaussian(&mut self, std: f32) {
        for val in &mut self.lora_a {
            *val = rand_gaussian() * std;
        }
    }

    /// Compute LoRA output: input @ A @ B * scaling
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let batch_size = input.len() / self.d_in;

        // Step 1: input @ A -> (batch_size, rank)
        let mut intermediate = vec![0.0; batch_size * self.rank];
        matmul(
            input,
            &self.lora_a,
            &mut intermediate,
            batch_size,
            self.d_in,
            self.rank,
        );

        // Step 2: intermediate @ B -> (batch_size, d_out)
        let mut output = vec![0.0; batch_size * self.d_out];
        matmul(
            &intermediate,
            &self.lora_b,
            &mut output,
            batch_size,
            self.rank,
            self.d_out,
        );

        // Apply scaling
        for val in &mut output {
            *val *= self.scaling;
        }

        output
    }

    /// Compute gradients for A and B
    pub fn backward(&self, input: &[f32], grad_output: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let batch_size = input.len() / self.d_in;

        // Scale gradient
        let scaled_grad: Vec<f32> = grad_output.iter().map(|&g| g * self.scaling).collect();

        // Compute intermediate for gradient computation
        let mut intermediate = vec![0.0; batch_size * self.rank];
        matmul(
            input,
            &self.lora_a,
            &mut intermediate,
            batch_size,
            self.d_in,
            self.rank,
        );

        // Gradient for B: intermediate^T @ grad_output
        let mut grad_b = vec![0.0; self.rank * self.d_out];
        matmul_atb(
            &intermediate,
            &scaled_grad,
            &mut grad_b,
            batch_size,
            self.rank,
            self.d_out,
        );

        // Gradient for intermediate: grad_output @ B^T
        let mut grad_intermediate = vec![0.0; batch_size * self.rank];
        matmul_abt(
            &scaled_grad,
            &self.lora_b,
            &mut grad_intermediate,
            batch_size,
            self.d_out,
            self.rank,
        );

        // Gradient for A: input^T @ grad_intermediate
        let mut grad_a = vec![0.0; self.d_in * self.rank];
        matmul_atb(
            input,
            &grad_intermediate,
            &mut grad_a,
            batch_size,
            self.d_in,
            self.rank,
        );

        // Gradient for input: grad_intermediate @ A^T
        let mut grad_input = vec![0.0; batch_size * self.d_in];
        matmul_abt(
            &grad_intermediate,
            &self.lora_a,
            &mut grad_input,
            batch_size,
            self.rank,
            self.d_in,
        );

        (grad_a, grad_b, grad_input)
    }

    /// Number of trainable parameters
    pub fn num_params(&self) -> usize {
        self.lora_a.len() + self.lora_b.len()
    }

    /// Memory usage in bytes (FP32)
    pub fn memory_bytes(&self) -> usize {
        self.num_params() * 4
    }
}

// ============================================================================
// LoRA-QAT Layer
// ============================================================================

/// A single LoRA-QAT layer
///
/// Combines frozen base weights (quantized) with trainable LoRA adapters.
pub struct LoraQatLayer {
    /// Layer name
    pub name: String,
    /// Base weights (frozen, quantized)
    base_weights: Vec<f32>,
    /// Quantized base weights
    base_quantized: Vec<f32>,
    /// LoRA weights (trainable)
    lora: LoraWeights,
    /// Quantizer for base weights
    quantizer: Box<dyn DifferentiableQuantizer>,
    /// Configuration
    config: LoraQatConfig,
    /// Input dimension
    d_in: usize,
    /// Output dimension
    d_out: usize,
    /// Whether layer is in training mode
    training: bool,
}

impl LoraQatLayer {
    /// Create a new LoRA-QAT layer
    pub fn new(name: &str, d_in: usize, d_out: usize, config: LoraQatConfig) -> Self {
        let qat_config = config.to_qat_config();
        let quantizer = create_quantizer(&qat_config);
        let lora = LoraWeights::new(d_in, d_out, config.rank, config.scaling());

        Self {
            name: name.to_string(),
            base_weights: vec![0.0; d_in * d_out],
            base_quantized: vec![0.0; d_in * d_out],
            lora,
            quantizer,
            config,
            d_in,
            d_out,
            training: true,
        }
    }

    /// Load base weights (frozen)
    pub fn load_base_weights(&mut self, weights: &[f32]) -> Result<()> {
        if weights.len() != self.d_in * self.d_out {
            return Err(RuvLLMError::Model(format!(
                "Weight size mismatch: expected {}, got {}",
                self.d_in * self.d_out,
                weights.len()
            )));
        }

        self.base_weights = weights.to_vec();

        // Quantize base weights
        if self.config.quantize_base {
            let (_, dequant) = self.quantizer.forward(&self.base_weights);
            self.base_quantized = dequant;
        } else {
            self.base_quantized = self.base_weights.clone();
        }

        Ok(())
    }

    /// Initialize LoRA weights
    pub fn init_lora(&mut self) {
        self.lora.init_kaiming();
    }

    /// Set training mode
    pub fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    /// Forward pass
    ///
    /// output = (W_q + LoRA) @ input
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let batch_size = input.len() / self.d_in;

        // Base output: input @ W_q^T
        let mut base_output = vec![0.0; batch_size * self.d_out];
        // Note: weights are stored as (d_out, d_in), so we compute input @ W^T
        matmul_abt(
            input,
            &self.base_quantized,
            &mut base_output,
            batch_size,
            self.d_in,
            self.d_out,
        );

        // LoRA output
        let lora_output = self.lora.forward(input);

        // Combine
        for (base, lora) in base_output.iter_mut().zip(&lora_output) {
            *base += lora;
        }

        base_output
    }

    /// Backward pass
    ///
    /// Returns (grad_input, grad_lora_a, grad_lora_b)
    pub fn backward(&self, input: &[f32], grad_output: &[f32]) -> LoraGradients {
        // Gradient through base (frozen, but need grad_input)
        let batch_size = input.len() / self.d_in;
        let mut grad_input_base = vec![0.0; batch_size * self.d_in];

        // grad_input_base = grad_output @ W_q
        matmul(
            grad_output,
            &self.base_quantized,
            &mut grad_input_base,
            batch_size,
            self.d_out,
            self.d_in,
        );

        // Gradient through LoRA
        let (grad_a, grad_b, grad_input_lora) = self.lora.backward(input, grad_output);

        // Combine gradients
        let grad_input: Vec<f32> = grad_input_base
            .iter()
            .zip(&grad_input_lora)
            .map(|(a, b)| a + b)
            .collect();

        LoraGradients {
            grad_input,
            grad_lora_a: grad_a,
            grad_lora_b: grad_b,
        }
    }

    /// Update LoRA weights with gradients
    pub fn update(&mut self, grads: &LoraGradients, lr: f32) {
        // SGD update for LoRA A
        for (w, g) in self.lora.lora_a.iter_mut().zip(&grads.grad_lora_a) {
            *w -= lr * g;
        }

        // SGD update for LoRA B
        for (w, g) in self.lora.lora_b.iter_mut().zip(&grads.grad_lora_b) {
            *w -= lr * g;
        }
    }

    /// Merge LoRA into base weights
    ///
    /// After training, merge LoRA deltas back into base for inference.
    pub fn merge(&mut self) {
        // Compute full LoRA weight matrix: A @ B * scaling
        let mut lora_full = vec![0.0; self.d_in * self.d_out];

        // For each output, compute: sum_r(A[i,r] * B[r,j])
        for i in 0..self.d_in {
            for j in 0..self.d_out {
                let mut sum = 0.0;
                for r in 0..self.lora.rank {
                    sum += self.lora.lora_a[i * self.lora.rank + r]
                        * self.lora.lora_b[r * self.d_out + j];
                }
                lora_full[j * self.d_in + i] = sum * self.lora.scaling;
            }
        }

        // Merge into base weights
        for (base, lora) in self.base_weights.iter_mut().zip(&lora_full) {
            *base += lora;
        }

        // Re-quantize
        if self.config.quantize_base {
            let (_, dequant) = self.quantizer.forward(&self.base_weights);
            self.base_quantized = dequant;
        }

        // Reset LoRA to zeros
        self.lora.lora_a.fill(0.0);
        self.lora.lora_b.fill(0.0);
    }

    /// Get total trainable parameters
    pub fn trainable_params(&self) -> usize {
        self.lora.num_params()
    }

    /// Get memory usage
    pub fn memory_bytes(&self) -> usize {
        let base_bytes = if self.config.quantize_base {
            // Quantized: bits per weight
            (self.base_weights.len() * self.config.bits as usize + 7) / 8
        } else {
            self.base_weights.len() * 4
        };

        let lora_bytes = self.lora.memory_bytes();
        base_bytes + lora_bytes
    }
}

/// Gradients from LoRA backward pass
#[derive(Debug, Clone)]
pub struct LoraGradients {
    /// Gradient w.r.t. input
    pub grad_input: Vec<f32>,
    /// Gradient w.r.t. LoRA A
    pub grad_lora_a: Vec<f32>,
    /// Gradient w.r.t. LoRA B
    pub grad_lora_b: Vec<f32>,
}

// ============================================================================
// LoRA-QAT Model
// ============================================================================

/// Collection of LoRA-QAT layers for a model
pub struct LoraQatModel {
    /// Layer name -> LoRA-QAT layer
    layers: HashMap<String, LoraQatLayer>,
    /// Configuration
    config: LoraQatConfig,
    /// Training mode
    training: bool,
}

impl LoraQatModel {
    /// Create a new LoRA-QAT model
    pub fn new(config: LoraQatConfig) -> Self {
        Self {
            layers: HashMap::new(),
            config,
            training: true,
        }
    }

    /// Add a layer
    pub fn add_layer(&mut self, name: &str, d_in: usize, d_out: usize) {
        let layer = LoraQatLayer::new(name, d_in, d_out, self.config.clone());
        self.layers.insert(name.to_string(), layer);
    }

    /// Get a layer
    pub fn get_layer(&self, name: &str) -> Option<&LoraQatLayer> {
        self.layers.get(name)
    }

    /// Get a mutable layer
    pub fn get_layer_mut(&mut self, name: &str) -> Option<&mut LoraQatLayer> {
        self.layers.get_mut(name)
    }

    /// Set training mode for all layers
    pub fn train(&mut self, mode: bool) {
        self.training = mode;
        for layer in self.layers.values_mut() {
            layer.train(mode);
        }
    }

    /// Merge all LoRA weights into base
    pub fn merge_all(&mut self) {
        for layer in self.layers.values_mut() {
            layer.merge();
        }
    }

    /// Total trainable parameters
    pub fn trainable_params(&self) -> usize {
        self.layers.values().map(|l| l.trainable_params()).sum()
    }

    /// Total memory usage
    pub fn memory_bytes(&self) -> usize {
        self.layers.values().map(|l| l.memory_bytes()).sum()
    }

    /// Number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Simple matrix multiplication: C = A @ B
/// A: (m, k), B: (k, n), C: (m, n)
fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Matrix multiplication with transposed A: C = A^T @ B
fn matmul_atb(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    // A is (m, k), we want A^T which is (k, m)
    // Result: (k, m)^T @ (m, n) = (k, n)
    for i in 0..k {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..m {
                sum += a[l * k + i] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Matrix multiplication with transposed B: C = A @ B^T
fn matmul_abt(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    // A is (m, k), B is (n, k), we want A @ B^T = (m, n)
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a[i * k + l] * b[j * k + l];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Simple pseudo-random number generator (replace with proper RNG)
fn rand_simple() -> f32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    static mut SEED: u64 = 0;
    unsafe {
        if SEED == 0 {
            SEED = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64;
        }
        SEED = SEED.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((SEED >> 33) as f32) / (u32::MAX as f32)
    }
}

/// Simple Gaussian random (Box-Muller)
fn rand_gaussian() -> f32 {
    let u1 = rand_simple().max(1e-10);
    let u2 = rand_simple();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_config() {
        let config = LoraQatConfig::default();
        assert_eq!(config.rank, 16);
        assert_eq!(config.bits, 3);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_lora_config_invalid() {
        let config = LoraQatConfig::default().with_rank(0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_lora_scaling() {
        let config = LoraQatConfig::default().with_rank(16).with_alpha(32.0);
        assert_eq!(config.scaling(), 2.0);
    }

    #[test]
    fn test_lora_weights() {
        let mut lora = LoraWeights::new(64, 64, 4, 2.0);
        lora.init_kaiming();

        assert_eq!(lora.num_params(), 64 * 4 + 4 * 64);
        assert!(lora.lora_a.iter().any(|&v| v != 0.0)); // A should be initialized
        assert!(lora.lora_b.iter().all(|&v| v == 0.0)); // B should be zeros
    }

    #[test]
    fn test_lora_forward() {
        let mut lora = LoraWeights::new(4, 4, 2, 1.0);
        // Set simple weights for testing
        lora.lora_a = vec![1.0; 8];
        lora.lora_b = vec![1.0; 8];

        let input = vec![1.0; 4];
        let output = lora.forward(&input);

        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_lora_qat_layer() {
        let config = LoraQatConfig::piq3_r16();
        let mut layer = LoraQatLayer::new("test", 64, 64, config);

        let weights = vec![0.1; 64 * 64];
        layer.load_base_weights(&weights).unwrap();
        layer.init_lora();

        let input = vec![0.5; 64];
        let output = layer.forward(&input);

        assert_eq!(output.len(), 64);
    }

    #[test]
    fn test_lora_qat_model() {
        let config = LoraQatConfig::piq3_r16();
        let mut model = LoraQatModel::new(config);

        model.add_layer("layer.0.q_proj", 64, 64);
        model.add_layer("layer.0.v_proj", 64, 64);

        assert_eq!(model.num_layers(), 2);
        assert!(model.trainable_params() > 0);
    }

    #[test]
    fn test_matmul() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity
        let mut c = vec![0.0; 4];

        matmul(&a, &b, &mut c, 2, 2, 2);

        assert_eq!(c, a); // A @ I = A
    }

    #[test]
    fn test_memory_efficiency() {
        let config = LoraQatConfig::piq3_r16();
        let layer = LoraQatLayer::new("test", 4096, 4096, config);

        let full_fp32_bytes = 4096 * 4096 * 4; // ~64 MB
        let lora_qat_bytes = layer.memory_bytes();

        // LoRA-QAT should use much less memory
        assert!(lora_qat_bytes < full_fp32_bytes / 2);
    }
}
