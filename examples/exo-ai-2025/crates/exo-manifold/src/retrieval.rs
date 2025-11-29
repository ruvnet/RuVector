//! Gradient descent retrieval algorithm for learned manifold
//!
//! Implements the ManifoldRetrieve algorithm from PSEUDOCODE.md:
//! 1. Initialize search position at query
//! 2. Gradient descent toward high-relevance regions
//! 3. Extract patterns from converged region

use burn::prelude::*;
use burn::tensor::ops::TensorOps;
use exo_core::{Error, ManifoldConfig, Pattern, Result, SearchResult};
use parking_lot::RwLock;
use std::sync::Arc;

use crate::network::LearnedManifold;

/// Gradient descent retriever for manifold queries
pub struct GradientDescentRetriever<B: Backend> {
    network: Arc<RwLock<LearnedManifold<B>>>,
    config: ManifoldConfig,
    device: B::Device,
}

impl<B: Backend> GradientDescentRetriever<B> {
    pub fn new(
        network: Arc<RwLock<LearnedManifold<B>>>,
        config: ManifoldConfig,
        device: B::Device,
    ) -> Self {
        Self {
            network,
            config,
            device,
        }
    }

    /// Retrieve patterns via gradient descent
    ///
    /// # Algorithm (from PSEUDOCODE.md)
    /// ```text
    /// position = query_vector
    /// FOR step IN 1..MAX_DESCENT_STEPS:
    ///     relevance_field = manifold_network.forward(position)
    ///     gradient = manifold_network.backward(relevance_field)
    ///     position = position - LEARNING_RATE * gradient
    ///     IF norm(gradient) < CONVERGENCE_THRESHOLD:
    ///         BREAK
    /// results = ExtractPatternsNear(position, k)
    /// ```
    pub fn retrieve(
        &self,
        query: &[f32],
        k: usize,
        patterns: &Arc<RwLock<Vec<Pattern>>>,
    ) -> Result<Vec<SearchResult>> {
        // Initialize position at query
        let mut position: Tensor<B, 1> =
            Tensor::from_floats(query, &self.device);

        let mut visited_positions = Vec::new();
        visited_positions.push(position.clone());

        // Gradient descent loop
        for step in 0..self.config.max_descent_steps {
            // Enable gradient tracking for position
            let position_tracked = position.clone().require_grad();

            // Forward pass through manifold network
            let relevance = {
                let network = self.network.read();
                network.forward(position_tracked.clone())
            };

            // We want to MAXIMIZE relevance, so we do gradient ASCENT
            // gradient = d(relevance)/d(position)
            let grads = relevance.backward();
            let gradient = position_tracked.grad(&grads).unwrap();

            // Compute gradient norm for convergence check
            let grad_norm = self.compute_norm(&gradient);

            // Update position: position = position + learning_rate * gradient (ascent)
            position = position + gradient.clone() * self.config.learning_rate;

            visited_positions.push(position.clone());

            // Check convergence
            if grad_norm < self.config.convergence_threshold {
                #[cfg(debug_assertions)]
                eprintln!("Converged at step {step} with gradient norm {grad_norm}");
                break;
            }
        }

        // Extract patterns near converged positions
        self.extract_patterns_near(&visited_positions, k, patterns)
    }

    /// Compute L2 norm of tensor
    fn compute_norm(&self, tensor: &Tensor<B, 1>) -> f32 {
        let squared = tensor.clone() * tensor.clone();
        let sum = squared.sum();
        let norm_squared: f32 = sum.into_scalar().elem();
        norm_squared.sqrt()
    }

    /// Extract k nearest patterns from visited positions
    fn extract_patterns_near(
        &self,
        positions: &[Tensor<B, 1>],
        k: usize,
        patterns: &Arc<RwLock<Vec<Pattern>>>,
    ) -> Result<Vec<SearchResult>> {
        let patterns_read = patterns.read();

        if patterns_read.is_empty() {
            return Ok(Vec::new());
        }

        // Use the last few positions (near convergence)
        let num_positions = positions.len().min(k * 2);
        let recent_positions = &positions[positions.len().saturating_sub(num_positions)..];

        // Score all patterns against recent positions
        let mut scored_patterns = Vec::new();

        for pattern in patterns_read.iter() {
            let pattern_tensor = Tensor::from_floats(&pattern.embedding[..], &self.device);

            // Compute minimum distance to any recent position
            let mut min_distance = f32::MAX;
            let mut max_relevance = f32::MIN;

            for position in recent_positions {
                let distance = self.compute_distance(&pattern_tensor, position);
                min_distance = min_distance.min(distance);

                // Also compute relevance at pattern position
                let relevance = {
                    let network = self.network.read();
                    let rel_tensor = network.forward(pattern_tensor.clone());
                    let rel_value: f32 = rel_tensor.into_scalar().elem();
                    rel_value
                };
                max_relevance = max_relevance.max(relevance);
            }

            // Combined score: lower distance + higher relevance = better
            let score = max_relevance - 0.1 * min_distance;

            scored_patterns.push((pattern.clone(), score, min_distance));
        }

        // Sort by score descending
        scored_patterns.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Take top k
        let results = scored_patterns
            .into_iter()
            .take(k)
            .map(|(pattern, score, distance)| SearchResult {
                pattern,
                score,
                distance,
            })
            .collect();

        Ok(results)
    }

    /// Compute Euclidean distance between two tensors
    fn compute_distance(&self, a: &Tensor<B, 1>, b: &Tensor<B, 1>) -> f32 {
        let diff = a.clone() - b.clone();
        self.compute_norm(&diff)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use exo_core::{Metadata, PatternId, SubstrateTime};

    type TestBackend = NdArray;

    fn create_test_pattern(embedding: Vec<f32>) -> Pattern {
        Pattern {
            id: PatternId::new(),
            embedding,
            metadata: Metadata::default(),
            timestamp: SubstrateTime::now(),
            antecedents: vec![],
            salience: 1.0,
        }
    }

    #[test]
    fn test_gradient_descent_retrieval() {
        let device = Default::default();
        let config = ManifoldConfig {
            dimension: 64,
            max_descent_steps: 10,
            learning_rate: 0.01,
            convergence_threshold: 1e-4,
            ..Default::default()
        };

        let network = LearnedManifold::<TestBackend>::new(
            config.dimension,
            config.hidden_dim,
            config.hidden_layers,
            config.omega_0,
            &device,
        );

        let retriever = GradientDescentRetriever::new(
            Arc::new(RwLock::new(network)),
            config.clone(),
            device,
        );

        let patterns = Arc::new(RwLock::new(vec![
            create_test_pattern(vec![1.0; 64]),
            create_test_pattern(vec![0.5; 64]),
        ]));

        let query = vec![1.0; 64];
        let results = retriever.retrieve(&query, 2, &patterns);

        assert!(results.is_ok());
        let results = results.unwrap();
        assert!(results.len() <= 2);
    }
}
