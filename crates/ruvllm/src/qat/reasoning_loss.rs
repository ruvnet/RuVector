//! Reasoning Loss for QAT (ADR-090 Phase 2)
//!
//! This module implements the chain-of-thought (CoT) fidelity loss
//! to preserve reasoning capabilities during quantization.
//!
//! ## Theory
//!
//! Standard distillation optimizes for next-token prediction but can lose
//! multi-step reasoning structure. The reasoning loss adds terms that:
//!
//! 1. **Step consistency**: Match reasoning step boundaries
//! 2. **Intermediate states**: Preserve hidden state trajectories
//! 3. **Answer agreement**: Ensure final answers match
//!
//! ## Loss Components
//!
//! ```text
//! L_reasoning = lambda_step * L_step + lambda_traj * L_trajectory + lambda_ans * L_answer
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvllm::qat::{ChainOfThoughtLoss, ReasoningConfig, ReasoningStep};
//!
//! let config = ReasoningConfig::default();
//! let loss_fn = ChainOfThoughtLoss::new(config);
//!
//! let steps = vec![
//!     ReasoningStep::new("step1", teacher_hidden1, student_hidden1),
//!     ReasoningStep::new("step2", teacher_hidden2, student_hidden2),
//! ];
//!
//! let loss = loss_fn.compute(&steps, final_teacher_ans, final_student_ans)?;
//! ```

use serde::{Deserialize, Serialize};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for reasoning loss
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningConfig {
    /// Weight for step boundary loss
    pub lambda_step: f32,
    /// Weight for trajectory loss
    pub lambda_trajectory: f32,
    /// Weight for answer agreement loss
    pub lambda_answer: f32,
    /// Cosine similarity threshold for step matching
    pub step_similarity_threshold: f32,
    /// Number of hidden dimensions to compare (for efficiency)
    pub hidden_sample_size: usize,
    /// Whether to normalize hidden states before comparison
    pub normalize_hidden: bool,
}

impl Default for ReasoningConfig {
    fn default() -> Self {
        Self {
            lambda_step: 0.3,
            lambda_trajectory: 0.5,
            lambda_answer: 0.2,
            step_similarity_threshold: 0.9,
            hidden_sample_size: 256,
            normalize_hidden: true,
        }
    }
}

impl ReasoningConfig {
    /// Configuration for GSM8K-style math reasoning
    pub fn math_reasoning() -> Self {
        Self {
            lambda_step: 0.4,
            lambda_trajectory: 0.4,
            lambda_answer: 0.4, // Higher weight on final answer
            step_similarity_threshold: 0.85,
            hidden_sample_size: 256,
            normalize_hidden: true,
        }
    }

    /// Configuration for code generation
    pub fn code_generation() -> Self {
        Self {
            lambda_step: 0.2,
            lambda_trajectory: 0.6, // Higher on trajectory (syntax matters)
            lambda_answer: 0.3,
            step_similarity_threshold: 0.92,
            hidden_sample_size: 512,
            normalize_hidden: true,
        }
    }
}

// ============================================================================
// Reasoning Step
// ============================================================================

/// A single reasoning step for comparison
#[derive(Debug, Clone)]
pub struct ReasoningStep {
    /// Step identifier (e.g., "step1", "step2")
    pub id: String,
    /// Position in sequence
    pub position: usize,
    /// Teacher hidden state at this step
    pub teacher_hidden: Vec<f32>,
    /// Student hidden state at this step
    pub student_hidden: Vec<f32>,
    /// Optional step type (e.g., "calculation", "conclusion")
    pub step_type: Option<StepType>,
}

/// Type of reasoning step
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StepType {
    /// Initial setup/context
    Setup,
    /// Intermediate calculation
    Calculation,
    /// Logical inference
    Inference,
    /// Final conclusion
    Conclusion,
    /// Answer extraction
    Answer,
}

impl ReasoningStep {
    /// Create a new reasoning step
    pub fn new(id: &str, teacher: Vec<f32>, student: Vec<f32>) -> Self {
        Self {
            id: id.to_string(),
            position: 0,
            teacher_hidden: teacher,
            student_hidden: student,
            step_type: None,
        }
    }

    /// Set position
    pub fn with_position(mut self, pos: usize) -> Self {
        self.position = pos;
        self
    }

    /// Set step type
    pub fn with_type(mut self, step_type: StepType) -> Self {
        self.step_type = Some(step_type);
        self
    }

    /// Compute cosine similarity between teacher and student
    pub fn cosine_similarity(&self) -> f32 {
        cosine_similarity(&self.teacher_hidden, &self.student_hidden)
    }

    /// Compute MSE between teacher and student
    pub fn mse(&self) -> f32 {
        if self.teacher_hidden.len() != self.student_hidden.len() {
            return f32::MAX;
        }

        let n = self.teacher_hidden.len() as f32;
        self.teacher_hidden
            .iter()
            .zip(&self.student_hidden)
            .map(|(t, s)| (t - s).powi(2))
            .sum::<f32>()
            / n
    }
}

// ============================================================================
// Reasoning Metrics
// ============================================================================

/// Metrics from reasoning loss computation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReasoningMetrics {
    /// Number of reasoning chains evaluated
    pub chains_evaluated: usize,
    /// Average step similarity
    pub avg_step_similarity: f64,
    /// Average trajectory loss
    pub avg_trajectory_loss: f64,
    /// Answer agreement rate
    pub answer_agreement_rate: f64,
    /// Number of steps that fell below similarity threshold
    pub degraded_steps: usize,
}

// ============================================================================
// Chain of Thought Loss
// ============================================================================

/// Chain-of-thought fidelity loss
///
/// Computes loss terms that encourage the student model to preserve
/// reasoning structure from the teacher.
pub struct ChainOfThoughtLoss {
    /// Configuration
    config: ReasoningConfig,
    /// Accumulated metrics
    metrics: ReasoningMetrics,
}

impl ChainOfThoughtLoss {
    /// Create new CoT loss
    pub fn new(config: ReasoningConfig) -> Self {
        Self {
            config,
            metrics: ReasoningMetrics::default(),
        }
    }

    /// Get configuration
    pub fn config(&self) -> &ReasoningConfig {
        &self.config
    }

    /// Get metrics
    pub fn metrics(&self) -> &ReasoningMetrics {
        &self.metrics
    }

    /// Compute reasoning loss for a chain of steps
    ///
    /// # Arguments
    ///
    /// * `steps` - Reasoning steps with teacher/student hidden states
    /// * `teacher_answer` - Teacher's final answer (optional)
    /// * `student_answer` - Student's final answer (optional)
    ///
    /// # Returns
    ///
    /// Combined reasoning loss
    pub fn compute(
        &mut self,
        steps: &[ReasoningStep],
        teacher_answer: Option<&str>,
        student_answer: Option<&str>,
    ) -> f32 {
        if steps.is_empty() {
            return 0.0;
        }

        // Step consistency loss
        let step_loss = self.compute_step_loss(steps);

        // Trajectory loss
        let trajectory_loss = self.compute_trajectory_loss(steps);

        // Answer agreement loss
        let answer_loss = self.compute_answer_loss(teacher_answer, student_answer);

        // Update metrics
        self.update_metrics(steps, teacher_answer, student_answer);

        // Weighted combination
        self.config.lambda_step * step_loss
            + self.config.lambda_trajectory * trajectory_loss
            + self.config.lambda_answer * answer_loss
    }

    /// Compute step consistency loss
    ///
    /// Measures how well student matches teacher at each reasoning step.
    fn compute_step_loss(&self, steps: &[ReasoningStep]) -> f32 {
        let mut total_loss = 0.0;

        for step in steps {
            let similarity = step.cosine_similarity();
            // Convert similarity to loss (1 - similarity)
            // Give extra weight to important step types
            let weight = match step.step_type {
                Some(StepType::Conclusion) | Some(StepType::Answer) => 2.0,
                Some(StepType::Calculation) => 1.5,
                _ => 1.0,
            };
            total_loss += weight * (1.0 - similarity);
        }

        total_loss / steps.len() as f32
    }

    /// Compute trajectory loss
    ///
    /// Measures consistency of reasoning trajectory across steps.
    fn compute_trajectory_loss(&self, steps: &[ReasoningStep]) -> f32 {
        if steps.len() < 2 {
            return 0.0;
        }

        let mut total_loss = 0.0;

        // Compare consecutive step transitions
        for i in 0..steps.len() - 1 {
            // Teacher transition
            let teacher_delta = vector_diff(&steps[i + 1].teacher_hidden, &steps[i].teacher_hidden);
            // Student transition
            let student_delta = vector_diff(&steps[i + 1].student_hidden, &steps[i].student_hidden);

            // Compare transition directions
            let transition_similarity = cosine_similarity(&teacher_delta, &student_delta);
            total_loss += 1.0 - transition_similarity;
        }

        total_loss / (steps.len() - 1) as f32
    }

    /// Compute answer agreement loss
    fn compute_answer_loss(&self, teacher: Option<&str>, student: Option<&str>) -> f32 {
        match (teacher, student) {
            (Some(t), Some(s)) => {
                if t.trim() == s.trim() {
                    0.0 // Perfect match
                } else if t.contains(s.trim()) || s.contains(t.trim()) {
                    0.5 // Partial match
                } else {
                    1.0 // No match
                }
            }
            _ => 0.0, // No answers to compare
        }
    }

    /// Update running metrics
    fn update_metrics(
        &mut self,
        steps: &[ReasoningStep],
        teacher_answer: Option<&str>,
        student_answer: Option<&str>,
    ) {
        self.metrics.chains_evaluated += 1;

        // Average step similarity
        let avg_sim: f64 = steps
            .iter()
            .map(|s| s.cosine_similarity() as f64)
            .sum::<f64>()
            / steps.len() as f64;
        let n = self.metrics.chains_evaluated as f64;
        self.metrics.avg_step_similarity =
            (self.metrics.avg_step_similarity * (n - 1.0) + avg_sim) / n;

        // Count degraded steps
        for step in steps {
            if step.cosine_similarity() < self.config.step_similarity_threshold {
                self.metrics.degraded_steps += 1;
            }
        }

        // Answer agreement
        if let (Some(t), Some(s)) = (teacher_answer, student_answer) {
            let agrees = t.trim() == s.trim();
            self.metrics.answer_agreement_rate = (self.metrics.answer_agreement_rate * (n - 1.0)
                + if agrees { 1.0 } else { 0.0 })
                / n;
        }
    }

    /// Reset metrics
    pub fn reset_metrics(&mut self) {
        self.metrics = ReasoningMetrics::default();
    }

    /// Evaluate reasoning preservation quality
    ///
    /// Returns a score from 0 (poor) to 1 (perfect).
    pub fn evaluate_quality(&self) -> f32 {
        let sim_score = self.metrics.avg_step_similarity as f32;
        let ans_score = self.metrics.answer_agreement_rate as f32;

        // Weighted average
        0.6 * sim_score + 0.4 * ans_score
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        0.0
    } else {
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }
}

/// Compute vector difference
fn vector_diff(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

/// L2 normalize a vector in place
#[allow(dead_code)]
fn normalize_inplace(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reasoning_config() {
        let default = ReasoningConfig::default();
        assert!(default.lambda_step > 0.0);
        assert!(default.lambda_trajectory > 0.0);

        let math = ReasoningConfig::math_reasoning();
        assert!(math.lambda_answer >= default.lambda_answer);
    }

    #[test]
    fn test_reasoning_step() {
        let teacher = vec![1.0, 0.0, 0.0];
        let student = vec![1.0, 0.0, 0.0];
        let step = ReasoningStep::new("step1", teacher, student);

        assert!((step.cosine_similarity() - 1.0).abs() < 1e-5);
        assert!(step.mse() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity() {
        // Same vectors
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-5);

        // Orthogonal vectors
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-5);

        // Opposite vectors
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cot_loss() {
        let config = ReasoningConfig::default();
        let mut loss_fn = ChainOfThoughtLoss::new(config);

        // Create steps with similar hidden states
        let steps = vec![
            ReasoningStep::new("step1", vec![1.0, 0.0], vec![0.9, 0.1]),
            ReasoningStep::new("step2", vec![0.0, 1.0], vec![0.1, 0.9]),
        ];

        let loss = loss_fn.compute(&steps, Some("42"), Some("42"));

        assert!(loss >= 0.0);
        assert!(loss < 1.0); // Similar states should have low loss
    }

    #[test]
    fn test_answer_loss() {
        let config = ReasoningConfig::default();
        let loss_fn = ChainOfThoughtLoss::new(config);

        // Exact match
        assert_eq!(loss_fn.compute_answer_loss(Some("42"), Some("42")), 0.0);

        // No match
        assert_eq!(loss_fn.compute_answer_loss(Some("42"), Some("24")), 1.0);

        // No answers
        assert_eq!(loss_fn.compute_answer_loss(None, None), 0.0);
    }

    #[test]
    fn test_trajectory_loss() {
        let config = ReasoningConfig::default();
        let loss_fn = ChainOfThoughtLoss::new(config);

        // Steps with same transitions
        let steps = vec![
            ReasoningStep::new("s1", vec![0.0, 0.0], vec![0.0, 0.0]),
            ReasoningStep::new("s2", vec![1.0, 0.0], vec![1.0, 0.0]),
        ];

        let loss = loss_fn.compute_trajectory_loss(&steps);
        assert!(loss < 0.1); // Similar transitions
    }

    #[test]
    fn test_step_types() {
        let step = ReasoningStep::new("s1", vec![1.0], vec![1.0])
            .with_position(0)
            .with_type(StepType::Conclusion);

        assert_eq!(step.position, 0);
        assert_eq!(step.step_type, Some(StepType::Conclusion));
    }
}
