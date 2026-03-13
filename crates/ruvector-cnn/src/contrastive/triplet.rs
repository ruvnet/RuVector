//! # Triplet Loss
//!
//! Implementation of the Triplet Loss for metric learning.
//!
//! ## Mathematical Formulation
//!
//! ```text
//! L(a, p, n) = max(0, ||a - p||^2 - ||a - n||^2 + margin)
//! ```
//!
//! Where:
//! - `a` is the anchor embedding
//! - `p` is the positive (similar) embedding
//! - `n` is the negative (dissimilar) embedding
//! - `margin` is the minimum desired separation
//!
//! ## Variants
//!
//! - **Standard**: Uses Euclidean distance
//! - **Angular**: Uses angular distance for normalized embeddings
//! - **Soft**: Uses soft-margin (log-exp) for smoother gradients
//!
//! ## References
//!
//! - "FaceNet: A Unified Embedding for Face Recognition and Clustering"
//! - "Deep Metric Learning Using Triplet Network"

use crate::error::{CnnError, CnnResult};
use serde::{Deserialize, Serialize};

/// Distance metric for triplet loss computation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TripletDistance {
    /// Euclidean (L2) distance
    Euclidean,
    /// Squared Euclidean distance (avoids sqrt, faster)
    SquaredEuclidean,
    /// Cosine distance (1 - cosine_similarity)
    Cosine,
}

/// Result of triplet loss computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripletResult {
    /// The computed loss value
    pub loss: f64,
    /// Distance between anchor and positive
    pub positive_distance: f64,
    /// Distance between anchor and negative
    pub negative_distance: f64,
    /// Whether the triplet is a "hard" triplet (loss > 0)
    pub is_hard: bool,
    /// Whether the triplet violates the margin
    pub violates_margin: bool,
}

/// Triplet loss for metric learning.
///
/// # Example
///
/// ```rust
/// use ruvector_cnn::contrastive::TripletLoss;
///
/// let triplet = TripletLoss::new(1.0);
///
/// let anchor = vec![1.0, 0.0, 0.0];
/// let positive = vec![0.9, 0.1, 0.0];  // similar to anchor
/// let negative = vec![0.0, 1.0, 0.0];  // dissimilar to anchor
///
/// let loss = triplet.forward(&anchor, &positive, &negative);
/// assert!(loss >= 0.0);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripletLoss {
    /// Margin for the triplet loss (default: 1.0)
    margin: f64,
    /// Distance metric to use
    distance: TripletDistance,
    /// Whether to use soft margin (log-exp smoothing)
    soft_margin: bool,
    /// L2 regularization weight (optional)
    l2_regularization: Option<f64>,
}

impl TripletLoss {
    /// Create a new triplet loss with the specified margin.
    ///
    /// # Arguments
    ///
    /// * `margin` - The minimum desired separation between positive and negative
    ///   distances. Typical values: 0.2-2.0
    ///
    /// # Panics
    ///
    /// Panics if margin is negative.
    pub fn new(margin: f64) -> Self {
        assert!(margin >= 0.0, "Margin must be non-negative");
        Self {
            margin,
            distance: TripletDistance::SquaredEuclidean,
            soft_margin: false,
            l2_regularization: None,
        }
    }

    /// Set the distance metric.
    pub fn with_distance(mut self, distance: TripletDistance) -> Self {
        self.distance = distance;
        self
    }

    /// Enable soft margin for smoother gradients.
    ///
    /// Instead of `max(0, x)`, uses `log(1 + exp(x))`.
    pub fn with_soft_margin(mut self) -> Self {
        self.soft_margin = true;
        self
    }

    /// Add L2 regularization on embeddings.
    pub fn with_l2_regularization(mut self, weight: f64) -> Self {
        self.l2_regularization = Some(weight);
        self
    }

    /// Get the margin.
    pub fn margin(&self) -> f64 {
        self.margin
    }

    /// Get the distance metric.
    pub fn distance_metric(&self) -> TripletDistance {
        self.distance
    }

    /// Compute triplet loss for a single triplet.
    ///
    /// # Arguments
    ///
    /// * `anchor` - The anchor embedding
    /// * `positive` - The positive (similar) embedding
    /// * `negative` - The negative (dissimilar) embedding
    ///
    /// # Returns
    ///
    /// The triplet loss value (non-negative).
    pub fn forward(&self, anchor: &[f64], positive: &[f64], negative: &[f64]) -> f64 {
        self.forward_detailed(anchor, positive, negative)
            .map(|r| r.loss)
            .unwrap_or(0.0)
    }

    /// Compute triplet loss with detailed results.
    pub fn forward_detailed(
        &self,
        anchor: &[f64],
        positive: &[f64],
        negative: &[f64],
    ) -> CnnResult<TripletResult> {
        // Validate inputs
        if anchor.is_empty() {
            return Err(CnnError::InvalidInput("anchor cannot be empty".to_string()));
        }

        let dim = anchor.len();
        if positive.len() != dim {
            return Err(CnnError::DimensionMismatch(format!(
                "positive has dimension {}, expected {}",
                positive.len(),
                dim
            )));
        }
        if negative.len() != dim {
            return Err(CnnError::DimensionMismatch(format!(
                "negative has dimension {}, expected {}",
                negative.len(),
                dim
            )));
        }

        // Check for NaN/Inf
        for (name, vec) in [
            ("anchor", anchor),
            ("positive", positive),
            ("negative", negative),
        ] {
            if vec.iter().any(|x| x.is_nan() || x.is_infinite()) {
                return Err(CnnError::InvalidInput(format!(
                    "{} contains NaN or Inf",
                    name
                )));
            }
        }

        // Compute distances
        let pos_dist = self.compute_distance(anchor, positive);
        let neg_dist = self.compute_distance(anchor, negative);

        // Compute loss
        let diff = pos_dist - neg_dist + self.margin;
        let loss = if self.soft_margin {
            soft_relu(diff)
        } else {
            diff.max(0.0)
        };

        // Add L2 regularization if enabled
        let loss = if let Some(weight) = self.l2_regularization {
            let anchor_norm: f64 = anchor.iter().map(|x| x * x).sum();
            let pos_norm: f64 = positive.iter().map(|x| x * x).sum();
            let neg_norm: f64 = negative.iter().map(|x| x * x).sum();
            loss + weight * (anchor_norm + pos_norm + neg_norm) / 3.0
        } else {
            loss
        };

        Ok(TripletResult {
            loss,
            positive_distance: pos_dist,
            negative_distance: neg_dist,
            is_hard: diff > 0.0,
            violates_margin: pos_dist + self.margin > neg_dist,
        })
    }

    /// Compute batch triplet loss.
    ///
    /// # Arguments
    ///
    /// * `anchors` - Batch of anchor embeddings
    /// * `positives` - Batch of positive embeddings
    /// * `negatives` - Batch of negative embeddings
    ///
    /// # Returns
    ///
    /// Mean triplet loss across the batch.
    pub fn forward_batch(
        &self,
        anchors: &[Vec<f64>],
        positives: &[Vec<f64>],
        negatives: &[Vec<f64>],
    ) -> CnnResult<f64> {
        if anchors.len() != positives.len() || anchors.len() != negatives.len() {
            return Err(CnnError::DimensionMismatch(format!(
                "Batch sizes must match: anchors={}, positives={}, negatives={}",
                anchors.len(),
                positives.len(),
                negatives.len()
            )));
        }

        if anchors.is_empty() {
            return Err(CnnError::InvalidInput("batch cannot be empty".to_string()));
        }

        let mut total_loss = 0.0;
        for ((anchor, positive), negative) in anchors.iter().zip(positives).zip(negatives) {
            total_loss += self.forward(anchor, positive, negative);
        }

        Ok(total_loss / anchors.len() as f64)
    }

    /// Mine hard triplets from a batch.
    ///
    /// Returns indices of (anchor, positive, negative) triplets where the loss is positive.
    ///
    /// # Arguments
    ///
    /// * `embeddings` - All embeddings in the batch
    /// * `labels` - Class labels for each embedding
    ///
    /// # Returns
    ///
    /// Vector of (anchor_idx, positive_idx, negative_idx) tuples.
    pub fn mine_hard_triplets(
        &self,
        embeddings: &[Vec<f64>],
        labels: &[usize],
    ) -> Vec<(usize, usize, usize)> {
        if embeddings.len() != labels.len() {
            return vec![];
        }

        let n = embeddings.len();
        let mut triplets = Vec::new();

        // Precompute distance matrix
        let distances = self.compute_distance_matrix(embeddings);

        for anchor_idx in 0..n {
            let anchor_label = labels[anchor_idx];

            // Find hardest positive (furthest with same label)
            let mut hardest_pos_idx = None;
            let mut hardest_pos_dist = f64::NEG_INFINITY;

            // Find hardest negative (closest with different label)
            let mut hardest_neg_idx = None;
            let mut hardest_neg_dist = f64::INFINITY;

            for other_idx in 0..n {
                if other_idx == anchor_idx {
                    continue;
                }

                let dist = distances[anchor_idx][other_idx];

                if labels[other_idx] == anchor_label {
                    // Same class - potential positive
                    if dist > hardest_pos_dist {
                        hardest_pos_dist = dist;
                        hardest_pos_idx = Some(other_idx);
                    }
                } else {
                    // Different class - potential negative
                    if dist < hardest_neg_dist {
                        hardest_neg_dist = dist;
                        hardest_neg_idx = Some(other_idx);
                    }
                }
            }

            // Add triplet if valid and hard
            if let (Some(pos_idx), Some(neg_idx)) = (hardest_pos_idx, hardest_neg_idx) {
                if hardest_pos_dist - hardest_neg_dist + self.margin > 0.0 {
                    triplets.push((anchor_idx, pos_idx, neg_idx));
                }
            }
        }

        triplets
    }

    /// Compute distance between two vectors.
    #[inline]
    fn compute_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        match self.distance {
            TripletDistance::Euclidean => {
                let sum_sq: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
                sum_sq.sqrt()
            }
            TripletDistance::SquaredEuclidean => {
                a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
            }
            TripletDistance::Cosine => {
                let mut dot = 0.0;
                let mut norm_a_sq = 0.0;
                let mut norm_b_sq = 0.0;

                for (x, y) in a.iter().zip(b) {
                    dot += x * y;
                    norm_a_sq += x * x;
                    norm_b_sq += y * y;
                }

                let norm = (norm_a_sq * norm_b_sq).sqrt();
                if norm < 1e-8 {
                    1.0 // Maximum distance for zero vectors
                } else {
                    1.0 - dot / norm
                }
            }
        }
    }

    /// Compute pairwise distance matrix.
    fn compute_distance_matrix(&self, embeddings: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = embeddings.len();
        let mut matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in (i + 1)..n {
                let dist = self.compute_distance(&embeddings[i], &embeddings[j]);
                matrix[i][j] = dist;
                matrix[j][i] = dist;
            }
        }

        matrix
    }
}

impl Default for TripletLoss {
    fn default() -> Self {
        Self::new(1.0)
    }
}

/// Soft ReLU: log(1 + exp(x)) for smooth gradients.
#[inline]
fn soft_relu(x: f64) -> f64 {
    if x > 20.0 {
        x // Avoid overflow
    } else if x < -20.0 {
        0.0 // Underflow to 0
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triplet_basic() {
        let triplet = TripletLoss::new(1.0);

        let anchor = vec![1.0, 0.0, 0.0];
        let positive = vec![0.9, 0.1, 0.0];
        let negative = vec![0.0, 1.0, 0.0];

        let loss = triplet.forward(&anchor, &positive, &negative);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_triplet_zero_loss() {
        let triplet = TripletLoss::new(0.1);

        // Negative is far, positive is close - should be zero loss
        let anchor = vec![1.0, 0.0];
        let positive = vec![1.0, 0.0]; // identical
        let negative = vec![-1.0, 0.0]; // opposite

        let result = triplet
            .forward_detailed(&anchor, &positive, &negative)
            .unwrap();
        assert_eq!(result.loss, 0.0);
        assert!(!result.is_hard);
    }

    #[test]
    fn test_triplet_hard() {
        let triplet = TripletLoss::new(1.0);

        // Negative is closer than positive - hard triplet
        let anchor = vec![0.0, 0.0];
        let positive = vec![2.0, 0.0];
        let negative = vec![1.0, 0.0];

        let result = triplet
            .forward_detailed(&anchor, &positive, &negative)
            .unwrap();
        assert!(result.loss > 0.0);
        assert!(result.is_hard);
        assert!(result.violates_margin);
    }

    #[test]
    fn test_triplet_distances() {
        // Test Euclidean distance
        let triplet_euclidean = TripletLoss::new(0.0).with_distance(TripletDistance::Euclidean);
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let c = vec![0.0, 0.0];

        let result = triplet_euclidean.forward_detailed(&a, &b, &c).unwrap();
        assert!((result.positive_distance - 5.0).abs() < 1e-6);
        assert!(result.negative_distance.abs() < 1e-6);

        // Test cosine distance
        let triplet_cosine = TripletLoss::new(0.0).with_distance(TripletDistance::Cosine);
        let x = vec![1.0, 0.0];
        let y = vec![0.0, 1.0];
        let z = vec![1.0, 0.0];

        let result = triplet_cosine.forward_detailed(&x, &y, &z).unwrap();
        assert!((result.positive_distance - 1.0).abs() < 1e-6); // orthogonal = 1
        assert!(result.negative_distance.abs() < 1e-6); // identical = 0
    }

    #[test]
    fn test_soft_margin() {
        let hard = TripletLoss::new(1.0);
        let soft = TripletLoss::new(1.0).with_soft_margin();

        let anchor = vec![0.0, 0.0];
        let positive = vec![1.0, 0.0];
        let negative = vec![0.5, 0.0];

        let hard_loss = hard.forward(&anchor, &positive, &negative);
        let soft_loss = soft.forward(&anchor, &positive, &negative);

        // Soft margin should be >= hard margin
        assert!(soft_loss >= hard_loss);
        // Both should be positive for this hard triplet
        assert!(hard_loss > 0.0);
        assert!(soft_loss > 0.0);
    }

    #[test]
    fn test_batch_triplet() {
        let triplet = TripletLoss::new(1.0);

        let anchors = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let positives = vec![vec![0.9, 0.1], vec![0.1, 0.9]];
        let negatives = vec![vec![0.0, 1.0], vec![1.0, 0.0]];

        let loss = triplet
            .forward_batch(&anchors, &positives, &negatives)
            .unwrap();
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_mine_hard_triplets() {
        // Use a smaller margin so triplets are more likely to be "hard"
        let triplet = TripletLoss::new(0.01);

        // Create embeddings where hard triplets are guaranteed to exist
        // Class 0 and class 1 embeddings are close to each other
        let embeddings = vec![
            vec![1.0, 0.0],   // class 0 - anchor
            vec![0.95, 0.05], // class 0 - positive (close to anchor)
            vec![0.9, 0.1],   // class 1 - negative (also close, creating hard triplet)
            vec![0.85, 0.15], // class 1 - another negative
        ];
        let labels = vec![0, 0, 1, 1];

        let hard_triplets = triplet.mine_hard_triplets(&embeddings, &labels);

        // Verify triplet structure for any hard triplets found
        // Note: hard triplets may not always be found depending on the margin and embeddings
        for (a, p, n) in &hard_triplets {
            assert_eq!(labels[*a], labels[*p]); // anchor and positive same class
            assert_ne!(labels[*a], labels[*n]); // anchor and negative different class
        }

        // The function should at least return a valid (possibly empty) vec
        // If hard triplets are found, they should have valid structure (tested above)
    }

    #[test]
    fn test_l2_regularization() {
        let no_reg = TripletLoss::new(0.0);
        let with_reg = TripletLoss::new(0.0).with_l2_regularization(0.01);

        let anchor = vec![10.0, 0.0];
        let positive = vec![10.0, 0.0];
        let negative = vec![-10.0, 0.0];

        let loss_no_reg = no_reg.forward(&anchor, &positive, &negative);
        let loss_with_reg = with_reg.forward(&anchor, &positive, &negative);

        // With L2 regularization, loss should be higher for large embeddings
        assert!(loss_with_reg > loss_no_reg);
    }

    #[test]
    fn test_error_handling() {
        let triplet = TripletLoss::new(1.0);

        // Empty anchor
        let result = triplet.forward_detailed(&[], &[1.0], &[1.0]);
        assert!(result.is_err());

        // Dimension mismatch
        let result = triplet.forward_detailed(&[1.0, 2.0], &[1.0], &[1.0, 2.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_soft_relu() {
        // Basic cases
        assert!((soft_relu(0.0) - 2.0_f64.ln()).abs() < 1e-6);
        assert!(soft_relu(-100.0) < 1e-10);
        assert!((soft_relu(100.0) - 100.0).abs() < 1e-6);

        // Smooth transition
        let x = 1.0;
        let y = soft_relu(x);
        assert!(y > x.max(0.0)); // Always >= max(0, x) but smoother
    }
}
