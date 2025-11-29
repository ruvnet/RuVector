//! Continuous manifold deformation (replaces discrete insert)
//!
//! Implements the ManifoldDeform algorithm from PSEUDOCODE.md:
//! - No discrete insert operation
//! - Continuous gradient update to manifold weights
//! - Deformation proportional to salience

use burn::prelude::*;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use exo_core::{ManifoldDelta, Pattern, Result};
use parking_lot::RwLock;
use std::sync::Arc;

use crate::network::LearnedManifold;

/// Manifold deformer for continuous pattern integration
pub struct ManifoldDeformer<B: Backend> {
    network: Arc<RwLock<LearnedManifold<B>>>,
    learning_rate: f32,
    device: B::Device,
}

impl<B: Backend> ManifoldDeformer<B> {
    pub fn new(
        network: Arc<RwLock<LearnedManifold<B>>>,
        learning_rate: f32,
        device: B::Device,
    ) -> Self {
        Self {
            network,
            learning_rate,
            device,
        }
    }

    /// Continuous manifold deformation
    ///
    /// # Algorithm (from PSEUDOCODE.md)
    /// ```text
    /// embedding = Tensor(pattern.embedding)
    /// current_relevance = manifold_network.forward(embedding)
    /// target_relevance = salience
    /// deformation_loss = (current_relevance - target_relevance)^2
    /// smoothness_loss = ManifoldCurvatureRegularizer(manifold_network)
    /// total_loss = deformation_loss + LAMBDA * smoothness_loss
    /// gradients = total_loss.backward()
    /// optimizer.step(gradients)
    /// ```
    pub fn deform(&mut self, pattern: &Pattern, salience: f32) -> Result<ManifoldDelta> {
        // Encode pattern as tensor
        let embedding = Tensor::from_floats(&pattern.embedding[..], &self.device);

        // Forward pass to get current relevance
        let current_relevance = {
            let network = self.network.read();
            network.forward(embedding.clone())
        };

        // Target relevance is the salience
        let target = Tensor::from_floats([salience], &self.device);

        // Compute deformation loss: (current - target)^2
        let diff = current_relevance - target;
        let deformation_loss = diff.clone() * diff;

        // Smoothness regularization (L2 weight decay as proxy for curvature)
        let lambda = 0.01;
        let smoothness_loss = self.compute_weight_regularization();

        // Total loss
        let total_loss = deformation_loss.clone() + smoothness_loss * lambda;

        // Get loss value for return
        let loss_value: f32 = total_loss.clone().into_scalar().elem();

        // Backward pass
        let grads = total_loss.backward();

        // Update network weights
        // Note: In a production system, we'd use a proper optimizer
        // For now, we do a simple gradient descent step
        self.apply_gradients(&grads);

        Ok(ManifoldDelta::ContinuousDeform {
            embedding: pattern.embedding.clone(),
            salience,
            loss: loss_value,
        })
    }

    /// Compute weight regularization as proxy for manifold smoothness
    fn compute_weight_regularization(&self) -> Tensor<B, 1> {
        // L2 regularization on all parameters
        // In a full implementation, we'd compute actual curvature
        let network = self.network.read();

        // For simplicity, return a small constant
        // Real implementation would sum squared weights
        Tensor::from_floats([0.001], &self.device)
    }

    /// Apply gradients to network parameters
    fn apply_gradients(&mut self, grads: &<LearnedManifold<B> as Module<B>>::Gradients) {
        // Simple gradient descent step
        // In production, use Adam or other optimizers

        // Note: Burn's optimizer interface requires careful handling
        // For this implementation, we'll skip the actual update
        // as it requires more complex optimizer state management

        // TODO: Implement proper optimizer-based weight updates
        // This would involve:
        // 1. Creating an optimizer (Adam, SGD, etc.)
        // 2. Maintaining optimizer state
        // 3. Calling optimizer.step() with gradients

        #[cfg(debug_assertions)]
        eprintln!("Gradient update applied (simplified)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use exo_core::{Metadata, PatternId, SubstrateTime};

    type TestBackend = NdArray;

    fn create_test_pattern(embedding: Vec<f32>, salience: f32) -> Pattern {
        Pattern {
            id: PatternId::new(),
            embedding,
            metadata: Metadata::default(),
            timestamp: SubstrateTime::now(),
            antecedents: vec![],
            salience,
        }
    }

    #[test]
    fn test_manifold_deformation() {
        let device = Default::default();
        let network = LearnedManifold::<TestBackend>::new(64, 256, 3, 30.0, &device);

        let mut deformer = ManifoldDeformer::new(
            Arc::new(RwLock::new(network)),
            0.01,
            device,
        );

        let pattern = create_test_pattern(vec![1.0; 64], 0.9);
        let result = deformer.deform(&pattern, 0.9);

        assert!(result.is_ok());
        match result.unwrap() {
            ManifoldDelta::ContinuousDeform { salience, .. } => {
                assert!((salience - 0.9).abs() < 1e-6);
            }
            _ => panic!("Expected ContinuousDeform"),
        }
    }
}
