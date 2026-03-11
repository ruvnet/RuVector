//! # InfoNCE Loss (NT-Xent)
//!
//! Implementation of the InfoNCE (Noise Contrastive Estimation) loss, also known as
//! NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.
//!
//! This loss is used in self-supervised learning methods like SimCLR and CLIP.
//!
//! ## Mathematical Formulation
//!
//! For a positive pair (i, j) among N samples:
//!
//! ```text
//! L(i, j) = -log( exp(sim(z_i, z_j) / tau) / sum_{k!=i} exp(sim(z_i, z_k) / tau) )
//! ```
//!
//! Where:
//! - `sim(u, v) = u^T v / (||u|| ||v||)` is cosine similarity
//! - `tau` is the temperature parameter
//!
//! ## References
//!
//! - SimCLR: "A Simple Framework for Contrastive Learning of Visual Representations"
//! - CLIP: "Learning Transferable Visual Models From Natural Language Supervision"

use crate::error::{CnnError, CnnResult};
use serde::{Deserialize, Serialize};

/// Result of InfoNCE loss computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfoNCEResult {
    /// The computed loss value
    pub loss: f64,
    /// Similarity matrix (optional, for debugging)
    pub similarity_matrix: Option<Vec<Vec<f64>>>,
    /// Per-sample losses (optional)
    pub per_sample_losses: Option<Vec<f64>>,
}

/// InfoNCE (NT-Xent) loss for contrastive learning.
///
/// # Example
///
/// ```rust
/// use ruvector_cnn::contrastive::InfoNCELoss;
///
/// let loss_fn = InfoNCELoss::new(0.07);
///
/// // Batch of embeddings where consecutive pairs are positives
/// let embeddings = vec![
///     vec![1.0, 0.0, 0.0],  // anchor 1
///     vec![0.9, 0.1, 0.0],  // positive for anchor 1
///     vec![0.0, 1.0, 0.0],  // anchor 2
///     vec![0.1, 0.9, 0.0],  // positive for anchor 2
/// ];
///
/// let loss = loss_fn.forward(&embeddings, 2);
/// assert!(loss > 0.0);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfoNCELoss {
    /// Temperature parameter (default: 0.07 as in SimCLR)
    temperature: f64,
    /// Whether to compute per-sample losses
    compute_per_sample: bool,
    /// Whether to store the similarity matrix
    store_similarity: bool,
}

impl InfoNCELoss {
    /// Create a new InfoNCE loss with the specified temperature.
    ///
    /// # Arguments
    ///
    /// * `temperature` - Temperature scaling parameter. Lower values make the
    ///   distribution sharper. Typical values: 0.07 (SimCLR), 0.5, 1.0
    ///
    /// # Panics
    ///
    /// Panics if temperature is not positive.
    pub fn new(temperature: f64) -> Self {
        assert!(temperature > 0.0, "Temperature must be positive");
        Self {
            temperature,
            compute_per_sample: false,
            store_similarity: false,
        }
    }

    /// Create a new InfoNCE loss with default temperature (0.07).
    pub fn default_temperature() -> Self {
        Self::new(0.07)
    }

    /// Enable computation of per-sample losses.
    pub fn with_per_sample_losses(mut self) -> Self {
        self.compute_per_sample = true;
        self
    }

    /// Enable storing the similarity matrix.
    pub fn with_similarity_matrix(mut self) -> Self {
        self.store_similarity = true;
        self
    }

    /// Get the temperature parameter.
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Compute InfoNCE loss for a batch of embeddings.
    ///
    /// # Arguments
    ///
    /// * `embeddings` - Batch of embedding vectors. For SimCLR-style training,
    ///   consecutive pairs (2*i, 2*i+1) are treated as positive pairs.
    /// * `num_views` - Number of augmented views per sample (typically 2).
    ///
    /// # Returns
    ///
    /// The mean InfoNCE loss across all positive pairs.
    pub fn forward(&self, embeddings: &[Vec<f64>], num_views: usize) -> f64 {
        self.forward_detailed(embeddings, num_views)
            .map(|r| r.loss)
            .unwrap_or(0.0)
    }

    /// Compute InfoNCE loss with detailed results.
    ///
    /// # Arguments
    ///
    /// * `embeddings` - Batch of embedding vectors
    /// * `num_views` - Number of augmented views per sample (typically 2)
    ///
    /// # Returns
    ///
    /// Detailed result including loss and optional diagnostics.
    pub fn forward_detailed(
        &self,
        embeddings: &[Vec<f64>],
        num_views: usize,
    ) -> CnnResult<InfoNCEResult> {
        let n = embeddings.len();
        if n == 0 {
            return Err(CnnError::InvalidInput("embeddings cannot be empty".to_string()));
        }
        if n < 2 {
            return Err(CnnError::InvalidInput(
                "Need at least 2 embeddings".to_string(),
            ));
        }
        if num_views < 2 {
            return Err(CnnError::InvalidConfig(
                "num_views must be at least 2".to_string(),
            ));
        }
        if n % num_views != 0 {
            return Err(CnnError::InvalidConfig(format!(
                "Number of embeddings ({}) must be divisible by num_views ({})",
                n, num_views
            )));
        }

        let dim = embeddings[0].len();
        for (i, emb) in embeddings.iter().enumerate() {
            if emb.len() != dim {
                return Err(CnnError::DimensionMismatch(format!(
                    "Embedding {} has dimension {}, expected {}",
                    i,
                    emb.len(),
                    dim
                )));
            }
            if emb.iter().any(|x| x.is_nan() || x.is_infinite()) {
                return Err(CnnError::InvalidInput(format!(
                    "Embedding {} contains NaN or Inf",
                    i
                )));
            }
        }

        // Compute similarity matrix
        let similarity_matrix = self.compute_similarity_matrix(embeddings);

        // Compute loss
        let mut total_loss = 0.0;
        let mut per_sample_losses = if self.compute_per_sample {
            Some(Vec::with_capacity(n))
        } else {
            None
        };

        for i in 0..n {
            // Find the positive pair index
            let sample_idx = i / num_views;
            let view_idx = i % num_views;

            // Positive is another view of the same sample
            let positive_idx = sample_idx * num_views + ((view_idx + 1) % num_views);

            // Compute log-softmax numerically stable
            let sim_positive = similarity_matrix[i][positive_idx] / self.temperature;

            // Sum of all similarities except self
            let mut log_sum_exp = f64::NEG_INFINITY;
            for (j, sim_row) in similarity_matrix[i].iter().enumerate() {
                if i != j {
                    let scaled_sim = sim_row / self.temperature;
                    log_sum_exp = log_sum_exp_pair(log_sum_exp, scaled_sim);
                }
            }

            let sample_loss = -sim_positive + log_sum_exp;
            total_loss += sample_loss;

            if let Some(ref mut losses) = per_sample_losses {
                losses.push(sample_loss);
            }
        }

        let mean_loss = total_loss / n as f64;

        // Check for numerical issues
        if mean_loss.is_nan() || mean_loss.is_infinite() {
            return Err(CnnError::InvalidInput(
                "Loss computation resulted in NaN or Inf".to_string(),
            ));
        }

        Ok(InfoNCEResult {
            loss: mean_loss,
            similarity_matrix: if self.store_similarity {
                Some(similarity_matrix)
            } else {
                None
            },
            per_sample_losses,
        })
    }

    /// Compute cosine similarity matrix between all pairs of embeddings.
    ///
    /// Uses SIMD-friendly layout for auto-vectorization.
    pub fn compute_similarity_matrix(&self, embeddings: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = embeddings.len();
        let mut matrix = vec![vec![0.0; n]; n];

        // Precompute norms for efficiency
        let norms: Vec<f64> = embeddings
            .iter()
            .map(|e| {
                let norm_sq: f64 = e.iter().map(|x| x * x).sum();
                norm_sq.sqrt().max(1e-8) // Avoid division by zero
            })
            .collect();

        // Compute upper triangle and mirror
        for i in 0..n {
            matrix[i][i] = 1.0; // Self-similarity
            for j in (i + 1)..n {
                let sim = cosine_similarity_normalized(&embeddings[i], &embeddings[j], norms[i], norms[j]);
                matrix[i][j] = sim;
                matrix[j][i] = sim;
            }
        }

        matrix
    }

    /// Compute InfoNCE loss with explicit positive pairs.
    ///
    /// # Arguments
    ///
    /// * `anchors` - Anchor embeddings
    /// * `positives` - Positive (similar) embeddings
    /// * `negatives` - Negative (dissimilar) embeddings (optional, uses all non-positives if None)
    ///
    /// # Returns
    ///
    /// The InfoNCE loss value.
    pub fn forward_with_pairs(
        &self,
        anchors: &[Vec<f64>],
        positives: &[Vec<f64>],
        negatives: Option<&[Vec<f64>]>,
    ) -> CnnResult<f64> {
        if anchors.len() != positives.len() {
            return Err(CnnError::DimensionMismatch(format!(
                "Anchors ({}) and positives ({}) must have same length",
                anchors.len(),
                positives.len()
            )));
        }

        if anchors.is_empty() {
            return Err(CnnError::InvalidInput("anchors cannot be empty".to_string()));
        }

        let dim = anchors[0].len();
        let mut total_loss = 0.0;

        for (i, (anchor, positive)) in anchors.iter().zip(positives.iter()).enumerate() {
            if anchor.len() != dim || positive.len() != dim {
                return Err(CnnError::DimensionMismatch(format!(
                    "Embedding {} has inconsistent dimensions",
                    i
                )));
            }

            let pos_sim = cosine_similarity(anchor, positive) / self.temperature;

            // Compute denominator: sum over positives and negatives
            let mut log_sum_exp = pos_sim;

            // Add negative samples
            if let Some(negs) = negatives {
                for neg in negs.iter() {
                    let neg_sim = cosine_similarity(anchor, neg) / self.temperature;
                    log_sum_exp = log_sum_exp_pair(log_sum_exp, neg_sim);
                }
            }

            // Add other positives as negatives (they're from different samples)
            for (j, other_pos) in positives.iter().enumerate() {
                if i != j {
                    let neg_sim = cosine_similarity(anchor, other_pos) / self.temperature;
                    log_sum_exp = log_sum_exp_pair(log_sum_exp, neg_sim);
                }
            }

            // Add other anchors as negatives
            for (j, other_anchor) in anchors.iter().enumerate() {
                if i != j {
                    let neg_sim = cosine_similarity(anchor, other_anchor) / self.temperature;
                    log_sum_exp = log_sum_exp_pair(log_sum_exp, neg_sim);
                }
            }

            total_loss += -pos_sim + log_sum_exp;
        }

        Ok(total_loss / anchors.len() as f64)
    }
}

impl Default for InfoNCELoss {
    fn default() -> Self {
        Self::default_temperature()
    }
}

/// Compute cosine similarity between two vectors.
#[inline]
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let mut dot = 0.0;
    let mut norm_a_sq = 0.0;
    let mut norm_b_sq = 0.0;

    // SIMD-friendly loop structure
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a_sq += x * x;
        norm_b_sq += y * y;
    }

    let norm = (norm_a_sq * norm_b_sq).sqrt();
    if norm < 1e-8 {
        0.0
    } else {
        dot / norm
    }
}

/// Compute cosine similarity with precomputed norms.
#[inline]
fn cosine_similarity_normalized(a: &[f64], b: &[f64], norm_a: f64, norm_b: f64) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    dot / (norm_a * norm_b)
}

/// Numerically stable log-sum-exp for two values.
#[inline]
fn log_sum_exp_pair(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        b
    } else if b == f64::NEG_INFINITY {
        a
    } else if a > b {
        a + (1.0 + (b - a).exp()).ln()
    } else {
        b + (1.0 + (a - b).exp()).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infonce_basic() {
        let loss_fn = InfoNCELoss::new(0.07);

        // Perfect positive pairs (identical vectors)
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0], // identical to anchor
            vec![0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ];

        let loss = loss_fn.forward(&embeddings, 2);
        // Loss should be low for identical pairs
        assert!(loss < 5.0, "Loss should be relatively low for identical pairs");
    }

    #[test]
    fn test_infonce_high_loss() {
        let loss_fn = InfoNCELoss::new(1.0); // Higher temperature for stability

        // Opposite pairs (anchor and positive are orthogonal)
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0], // orthogonal to anchor
            vec![0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0], // orthogonal to anchor
        ];

        let loss = loss_fn.forward(&embeddings, 2);
        assert!(loss > 0.0, "Loss should be positive");
    }

    #[test]
    fn test_similarity_matrix() {
        let loss_fn = InfoNCELoss::new(0.07);

        let embeddings = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];

        let sim_matrix = loss_fn.compute_similarity_matrix(&embeddings);

        // Check diagonal is 1
        assert!((sim_matrix[0][0] - 1.0).abs() < 1e-6);
        assert!((sim_matrix[1][1] - 1.0).abs() < 1e-6);

        // Check orthogonal vectors have 0 similarity
        assert!(sim_matrix[0][1].abs() < 1e-6);

        // Check symmetry
        assert!((sim_matrix[0][2] - sim_matrix[2][0]).abs() < 1e-6);
    }

    #[test]
    fn test_temperature_effect() {
        let low_temp = InfoNCELoss::new(0.01);
        let high_temp = InfoNCELoss::new(1.0);

        let embeddings = vec![
            vec![1.0, 0.0],
            vec![0.9, 0.1],
            vec![0.0, 1.0],
            vec![0.1, 0.9],
        ];

        let loss_low = low_temp.forward(&embeddings, 2);
        let loss_high = high_temp.forward(&embeddings, 2);

        // Lower temperature typically gives higher gradients (sharper distribution)
        // The absolute loss values depend on the similarity structure
        assert!(loss_low.is_finite());
        assert!(loss_high.is_finite());
    }

    #[test]
    fn test_infonce_with_pairs() {
        let loss_fn = InfoNCELoss::new(0.5);

        let anchors = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let positives = vec![vec![0.9, 0.1], vec![0.1, 0.9]];
        let negatives = vec![vec![-1.0, 0.0], vec![0.0, -1.0]];

        let loss = loss_fn
            .forward_with_pairs(&anchors, &positives, Some(&negatives))
            .unwrap();
        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_empty_input_error() {
        let loss_fn = InfoNCELoss::new(0.07);
        let result = loss_fn.forward_detailed(&[], 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let loss_fn = InfoNCELoss::new(0.07);
        let embeddings = vec![vec![1.0, 0.0], vec![1.0, 0.0, 0.0]];
        let result = loss_fn.forward_detailed(&embeddings, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_detailed_result() {
        let loss_fn = InfoNCELoss::new(0.07)
            .with_per_sample_losses()
            .with_similarity_matrix();

        let embeddings = vec![
            vec![1.0, 0.0],
            vec![0.9, 0.1],
            vec![0.0, 1.0],
            vec![0.1, 0.9],
        ];

        let result = loss_fn.forward_detailed(&embeddings, 2).unwrap();
        assert!(result.similarity_matrix.is_some());
        assert!(result.per_sample_losses.is_some());
        assert_eq!(result.per_sample_losses.as_ref().unwrap().len(), 4);
    }

    #[test]
    fn test_cosine_similarity() {
        // Identical vectors
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        // Orthogonal vectors
        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 1e-6);

        // Opposite vectors
        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_log_sum_exp_numerical_stability() {
        // Test with large values
        let large = 700.0;
        let result = log_sum_exp_pair(large, large);
        assert!(result.is_finite());
        assert!((result - large - 2.0_f64.ln()).abs() < 1e-6);

        // Test with neg infinity
        let result2 = log_sum_exp_pair(f64::NEG_INFINITY, 1.0);
        assert!((result2 - 1.0).abs() < 1e-6);
    }
}
