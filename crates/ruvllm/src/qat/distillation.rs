//! Knowledge Distillation Loss for QAT (ADR-090 Phase 2)
//!
//! This module implements the distillation loss components for QAT:
//! - L_task: Standard task loss (cross-entropy)
//! - L_KD: KL divergence from teacher model
//! - L_reasoning: Chain-of-thought fidelity (see reasoning_loss.rs)
//!
//! ## Composite Loss
//!
//! ```text
//! L_total = lambda_task * L_task + lambda_kd * L_KD + lambda_reason * L_reasoning
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvllm::qat::{DistillationLoss, DistillationConfig, TeacherOutput};
//!
//! let config = DistillationConfig::default();
//! let loss_fn = DistillationLoss::new(config);
//!
//! let teacher = TeacherOutput { logits: teacher_logits, hidden: None };
//! let loss = loss_fn.compute(&student_logits, &teacher, &labels)?;
//! ```

use serde::{Deserialize, Serialize};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for distillation loss
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationConfig {
    /// Temperature for KL divergence (higher = softer distributions)
    pub temperature: f32,
    /// Weight for task loss (lambda_task)
    pub lambda_task: f32,
    /// Weight for KD loss (lambda_kd)
    pub lambda_kd: f32,
    /// Weight for reasoning loss (lambda_reason)
    pub lambda_reasoning: f32,
    /// Whether to use hard labels for task loss
    pub use_hard_labels: bool,
    /// Minimum loss value (for numerical stability)
    pub min_loss: f32,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            temperature: 2.0,
            lambda_task: 1.0,
            lambda_kd: 0.5,
            lambda_reasoning: 0.1,
            use_hard_labels: true,
            min_loss: 1e-10,
        }
    }
}

impl DistillationConfig {
    /// Create config optimized for reasoning preservation
    pub fn reasoning_focused() -> Self {
        Self {
            temperature: 2.0,
            lambda_task: 0.5,
            lambda_kd: 0.3,
            lambda_reasoning: 0.5, // Higher weight on reasoning
            use_hard_labels: true,
            min_loss: 1e-10,
        }
    }

    /// Create config optimized for compression (less KD)
    pub fn compression_focused() -> Self {
        Self {
            temperature: 1.5,
            lambda_task: 1.0,
            lambda_kd: 0.2,
            lambda_reasoning: 0.1,
            use_hard_labels: true,
            min_loss: 1e-10,
        }
    }
}

// ============================================================================
// Teacher Output
// ============================================================================

/// Output from teacher model for distillation
#[derive(Debug, Clone)]
pub struct TeacherOutput {
    /// Teacher logits (vocab_size,) or (seq_len, vocab_size)
    pub logits: Vec<f32>,
    /// Optional hidden states for feature matching
    pub hidden_states: Option<Vec<f32>>,
    /// Sequence length (for multi-token outputs)
    pub seq_len: usize,
    /// Vocabulary size
    pub vocab_size: usize,
}

impl TeacherOutput {
    /// Create teacher output from logits
    pub fn from_logits(logits: Vec<f32>, vocab_size: usize) -> Self {
        let seq_len = logits.len() / vocab_size;
        Self {
            logits,
            hidden_states: None,
            seq_len,
            vocab_size,
        }
    }

    /// Get logits for a specific position
    pub fn logits_at(&self, position: usize) -> &[f32] {
        let start = position * self.vocab_size;
        let end = start + self.vocab_size;
        &self.logits[start..end]
    }

    /// Apply softmax with temperature
    pub fn softmax_at(&self, position: usize, temperature: f32) -> Vec<f32> {
        let logits = self.logits_at(position);
        softmax_with_temperature(logits, temperature)
    }
}

// ============================================================================
// Distillation Loss
// ============================================================================

/// Distillation loss computation
///
/// Combines task loss and KL divergence from teacher.
pub struct DistillationLoss {
    /// Configuration
    config: DistillationConfig,
    /// Statistics
    stats: DistillationStats,
}

/// Distillation statistics
#[derive(Debug, Clone, Default)]
pub struct DistillationStats {
    /// Number of loss computations
    pub compute_count: usize,
    /// Running average of task loss
    pub avg_task_loss: f64,
    /// Running average of KD loss
    pub avg_kd_loss: f64,
    /// Running average of total loss
    pub avg_total_loss: f64,
}

impl DistillationLoss {
    /// Create new distillation loss
    pub fn new(config: DistillationConfig) -> Self {
        Self {
            config,
            stats: DistillationStats::default(),
        }
    }

    /// Get configuration
    pub fn config(&self) -> &DistillationConfig {
        &self.config
    }

    /// Get statistics
    pub fn stats(&self) -> &DistillationStats {
        &self.stats
    }

    /// Compute composite loss
    ///
    /// # Arguments
    ///
    /// * `student_logits` - Student model logits
    /// * `teacher` - Teacher model output
    /// * `labels` - Ground truth labels (token IDs)
    ///
    /// # Returns
    ///
    /// Composite loss value
    pub fn compute(
        &mut self,
        student_logits: &[f32],
        teacher: &TeacherOutput,
        labels: &[u32],
    ) -> f32 {
        let vocab_size = teacher.vocab_size;
        let seq_len = labels.len();

        let mut total_task_loss = 0.0;
        let mut total_kd_loss = 0.0;

        for pos in 0..seq_len {
            let student_start = pos * vocab_size;
            let student_end = student_start + vocab_size;

            if student_end > student_logits.len() {
                break;
            }

            let student_slice = &student_logits[student_start..student_end];

            // Task loss (cross-entropy)
            if self.config.use_hard_labels {
                let label = labels[pos] as usize;
                if label < vocab_size {
                    let task_loss = cross_entropy(student_slice, label);
                    total_task_loss += task_loss;
                }
            }

            // KD loss (KL divergence)
            if pos < teacher.seq_len {
                let teacher_probs = teacher.softmax_at(pos, self.config.temperature);
                let student_probs =
                    softmax_with_temperature(student_slice, self.config.temperature);
                let kd_loss = kl_divergence(&student_probs, &teacher_probs);
                total_kd_loss += kd_loss * self.config.temperature.powi(2);
            }
        }

        // Normalize by sequence length
        let seq_len_f = seq_len as f32;
        let task_loss = total_task_loss / seq_len_f;
        let kd_loss = total_kd_loss / seq_len_f;

        // Composite loss
        let total_loss = self.config.lambda_task * task_loss + self.config.lambda_kd * kd_loss;

        // Update statistics
        self.update_stats(task_loss, kd_loss, total_loss);

        total_loss.max(self.config.min_loss)
    }

    /// Compute task loss only (for non-distillation scenarios)
    pub fn compute_task_loss(&self, logits: &[f32], labels: &[u32], vocab_size: usize) -> f32 {
        let seq_len = labels.len();
        let mut total_loss = 0.0;

        for pos in 0..seq_len {
            let start = pos * vocab_size;
            let end = start + vocab_size;

            if end > logits.len() {
                break;
            }

            let label = labels[pos] as usize;
            if label < vocab_size {
                total_loss += cross_entropy(&logits[start..end], label);
            }
        }

        total_loss / seq_len as f32
    }

    /// Compute KD loss only
    pub fn compute_kd_loss(&self, student_logits: &[f32], teacher: &TeacherOutput) -> f32 {
        let vocab_size = teacher.vocab_size;
        let seq_len = teacher.seq_len;
        let mut total_kd_loss = 0.0;

        for pos in 0..seq_len {
            let student_start = pos * vocab_size;
            let student_end = student_start + vocab_size;

            if student_end > student_logits.len() {
                break;
            }

            let student_slice = &student_logits[student_start..student_end];
            let teacher_probs = teacher.softmax_at(pos, self.config.temperature);
            let student_probs = softmax_with_temperature(student_slice, self.config.temperature);

            total_kd_loss += kl_divergence(&student_probs, &teacher_probs);
        }

        // Scale by T^2 as per distillation paper
        (total_kd_loss / seq_len as f32) * self.config.temperature.powi(2)
    }

    /// Update running statistics
    fn update_stats(&mut self, task_loss: f32, kd_loss: f32, total_loss: f32) {
        let n = self.stats.compute_count as f64;
        let alpha = 1.0 / (n + 1.0);

        self.stats.avg_task_loss =
            (1.0 - alpha) * self.stats.avg_task_loss + alpha * task_loss as f64;
        self.stats.avg_kd_loss = (1.0 - alpha) * self.stats.avg_kd_loss + alpha * kd_loss as f64;
        self.stats.avg_total_loss =
            (1.0 - alpha) * self.stats.avg_total_loss + alpha * total_loss as f64;
        self.stats.compute_count += 1;
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = DistillationStats::default();
    }
}

// ============================================================================
// Loss Functions
// ============================================================================

/// Cross-entropy loss for a single token
fn cross_entropy(logits: &[f32], label: usize) -> f32 {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
    let log_softmax = (logits[label] - max_logit) - exp_sum.ln();
    -log_softmax
}

/// KL divergence: D_KL(P || Q) = sum(P * log(P/Q))
fn kl_divergence(p: &[f32], q: &[f32]) -> f32 {
    debug_assert_eq!(p.len(), q.len());
    let eps = 1e-10;

    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| {
            if pi > eps {
                pi * ((pi + eps) / (qi + eps)).ln()
            } else {
                0.0
            }
        })
        .sum()
}

/// Softmax with temperature
fn softmax_with_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    exp_vals.iter().map(|&x| x / sum).collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distillation_config() {
        let default = DistillationConfig::default();
        assert_eq!(default.temperature, 2.0);
        assert_eq!(default.lambda_task, 1.0);

        let reasoning = DistillationConfig::reasoning_focused();
        assert!(reasoning.lambda_reasoning > default.lambda_reasoning);
    }

    #[test]
    fn test_teacher_output() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 tokens, vocab_size=3
        let teacher = TeacherOutput::from_logits(logits, 3);

        assert_eq!(teacher.seq_len, 2);
        assert_eq!(teacher.vocab_size, 3);
        assert_eq!(teacher.logits_at(0), &[1.0, 2.0, 3.0]);
        assert_eq!(teacher.logits_at(1), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax_with_temperature(&logits, 1.0);

        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_cross_entropy() {
        let logits = vec![0.0, 0.0, 0.0]; // Uniform
        let loss = cross_entropy(&logits, 0);

        // Should be -log(1/3) = log(3)
        let expected = 3.0f32.ln();
        assert!((loss - expected).abs() < 1e-5);
    }

    #[test]
    fn test_kl_divergence() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let kl = kl_divergence(&p, &q);

        // KL(P||P) = 0
        assert!(kl.abs() < 1e-5);

        // Different distributions
        let p = vec![0.9, 0.1];
        let q = vec![0.5, 0.5];
        let kl = kl_divergence(&p, &q);
        assert!(kl > 0.0);
    }

    #[test]
    fn test_distillation_loss() {
        let config = DistillationConfig::default();
        let mut loss_fn = DistillationLoss::new(config);

        // Student and teacher logits (vocab_size=4, seq_len=2)
        let student_logits = vec![1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0];
        let teacher = TeacherOutput::from_logits(vec![1.1, 2.1, 3.1, 4.1, 2.1, 3.1, 4.1, 5.1], 4);
        let labels = vec![3, 3];

        let loss = loss_fn.compute(&student_logits, &teacher, &labels);

        assert!(loss > 0.0);
        assert!(loss.is_finite());
        assert_eq!(loss_fn.stats().compute_count, 1);
    }

    #[test]
    fn test_task_loss_only() {
        let config = DistillationConfig::default();
        let loss_fn = DistillationLoss::new(config);

        let logits = vec![0.0, 0.0, 0.0, 10.0]; // Strong prediction for label 3
        let labels = vec![3];

        let loss = loss_fn.compute_task_loss(&logits, &labels, 4);

        // Loss should be low since prediction matches label
        assert!(loss < 1.0);
    }
}
