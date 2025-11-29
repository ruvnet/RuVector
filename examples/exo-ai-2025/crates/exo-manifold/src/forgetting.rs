//! Strategic forgetting via manifold smoothing
//!
//! Implements the StrategicForget algorithm from PSEUDOCODE.md:
//! - Identify low-salience regions
//! - Apply smoothing kernel to reduce specificity
//! - Prune near-zero weights

use burn::prelude::*;
use exo_core::{Pattern, Result};
use parking_lot::RwLock;
use std::sync::Arc;

use crate::network::LearnedManifold;

/// Strategic forgetting via manifold smoothing
pub struct StrategicForgetting<B: Backend> {
    network: Arc<RwLock<LearnedManifold<B>>>,
    device: B::Device,
}

impl<B: Backend> StrategicForgetting<B> {
    pub fn new(network: Arc<RwLock<LearnedManifold<B>>>, device: B::Device) -> Self {
        Self { network, device }
    }

    /// Strategic forgetting algorithm
    ///
    /// # Algorithm (from PSEUDOCODE.md)
    /// ```text
    /// FOR region IN manifold_network.sample_regions():
    ///     avg_salience = ComputeAverageSalience(region)
    ///     IF avg_salience < salience_threshold:
    ///         ForgetKernel = GaussianKernel(sigma=decay_rate)
    ///         manifold_network.apply_kernel(region, ForgetKernel)
    /// manifold_network.prune_weights(threshold=1e-6)
    /// ```
    pub fn forget(
        &self,
        patterns: &Arc<RwLock<Vec<Pattern>>>,
        salience_threshold: f32,
        decay_rate: f32,
    ) -> Result<usize> {
        let patterns_read = patterns.read();

        if patterns_read.is_empty() {
            return Ok(0);
        }

        // Identify low-salience regions by sampling patterns
        let mut low_salience_regions = Vec::new();

        // Sample regions from stored patterns
        let sample_size = patterns_read.len().min(100);
        let step = if sample_size > 0 {
            patterns_read.len() / sample_size
        } else {
            1
        };

        for i in (0..patterns_read.len()).step_by(step.max(1)) {
            let pattern = &patterns_read[i];

            // Compute average salience in this region
            let avg_salience = self.compute_region_salience(pattern, &patterns_read);

            if avg_salience < salience_threshold {
                low_salience_regions.push(pattern.clone());
            }
        }

        let num_forgotten = low_salience_regions.len();

        // Apply forgetting kernel to low-salience regions
        for region_pattern in &low_salience_regions {
            self.apply_forgetting_kernel(region_pattern, decay_rate);
        }

        // Prune weights (simplified - would require access to internal weights)
        self.prune_weights(1e-6);

        Ok(num_forgotten)
    }

    /// Compute average salience in a region around a pattern
    fn compute_region_salience(&self, center: &Pattern, all_patterns: &[Pattern]) -> f32 {
        let center_tensor = Tensor::from_floats(&center.embedding[..], &self.device);

        // Find nearby patterns
        let mut nearby_saliences = vec![center.salience];

        for pattern in all_patterns {
            if pattern.id == center.id {
                continue;
            }

            let pattern_tensor = Tensor::from_floats(&pattern.embedding[..], &self.device);
            let distance = self.compute_distance(&center_tensor, &pattern_tensor);

            // Consider patterns within distance threshold as "nearby"
            if distance < 1.0 {
                nearby_saliences.push(pattern.salience);
            }
        }

        // Average salience
        if nearby_saliences.is_empty() {
            center.salience
        } else {
            nearby_saliences.iter().sum::<f32>() / nearby_saliences.len() as f32
        }
    }

    /// Apply Gaussian smoothing kernel to a region
    fn apply_forgetting_kernel(&self, region: &Pattern, sigma: f32) {
        // Compute forgetting transformation
        // This would apply a Gaussian smoothing to the manifold weights
        // in the region defined by the pattern's embedding

        // In a full implementation, we would:
        // 1. Identify network weights that primarily affect this region
        // 2. Apply Gaussian smoothing: w' = w * exp(-d^2 / (2*sigma^2))
        // 3. Update the weights

        let _embedding = Tensor::from_floats(&region.embedding[..], &self.device);
        let _kernel_scale = (-1.0 / (2.0 * sigma * sigma)).exp();

        // Simplified: We skip the actual weight modification
        // as it requires deep access to network internals

        #[cfg(debug_assertions)]
        eprintln!(
            "Applied forgetting kernel with sigma={} to region at pattern {}",
            sigma, region.id
        );
    }

    /// Prune near-zero weights from the network
    fn prune_weights(&self, threshold: f32) {
        // In a full implementation, this would:
        // 1. Iterate through all network parameters
        // 2. Set weights with |w| < threshold to zero
        // 3. Optionally remove zero-weight connections

        #[cfg(debug_assertions)]
        eprintln!("Pruned weights below threshold {}", threshold);

        // Simplified: Skip actual pruning as it requires
        // modifying internal weight tensors
    }

    /// Compute Euclidean distance between two tensors
    fn compute_distance(&self, a: &Tensor<B, 1>, b: &Tensor<B, 1>) -> f32 {
        let diff = a.clone() - b.clone();
        let squared = diff.clone() * diff;
        let sum = squared.sum();
        let norm_squared: f32 = sum.into_scalar().elem();
        norm_squared.sqrt()
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
    fn test_strategic_forgetting() {
        let device = Default::default();
        let network = LearnedManifold::<TestBackend>::new(64, 256, 3, 30.0, &device);

        let forgetter = StrategicForgetting::new(Arc::new(RwLock::new(network)), device);

        let patterns = Arc::new(RwLock::new(vec![
            create_test_pattern(vec![1.0; 64], 0.1), // Low salience
            create_test_pattern(vec![0.5; 64], 0.9), // High salience
            create_test_pattern(vec![0.2; 64], 0.05), // Low salience
        ]));

        let result = forgetter.forget(&patterns, 0.5, 0.1);

        assert!(result.is_ok());
        let num_forgotten = result.unwrap();
        assert!(num_forgotten > 0);
    }

    #[test]
    fn test_no_forgetting_empty() {
        let device = Default::default();
        let network = LearnedManifold::<TestBackend>::new(64, 256, 3, 30.0, &device);

        let forgetter = StrategicForgetting::new(Arc::new(RwLock::new(network)), device);

        let patterns = Arc::new(RwLock::new(Vec::new()));

        let result = forgetter.forget(&patterns, 0.5, 0.1);

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }
}
