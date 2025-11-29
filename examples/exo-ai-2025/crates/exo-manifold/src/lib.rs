//! Learned Manifold Engine for EXO-AI Cognitive Substrate
//!
//! This crate implements continuous manifold storage using implicit neural
//! representations (SIREN networks). Instead of discrete vector storage,
//! memories are encoded as continuous functions on a learned manifold.
//!
//! # Key Concepts
//!
//! - **Retrieval**: Gradient descent toward high-relevance regions
//! - **Storage**: Continuous deformation of the manifold (no discrete insert)
//! - **Forgetting**: Strategic manifold smoothing in low-salience regions
//!
//! # Architecture
//!
//! ```text
//! Query → Gradient Descent → Converged Position → Extract Patterns
//!            ↓
//!      SIREN Network
//!      (Learned Manifold)
//!            ↓
//!      Relevance Field
//! ```

use burn::prelude::*;
use exo_core::{Error, ManifoldConfig, ManifoldDelta, Pattern, Result, SearchResult};
use parking_lot::RwLock;
use std::sync::Arc;

mod network;
mod retrieval;
mod deformation;
mod forgetting;

pub use network::{LearnedManifold, SirenLayer};
pub use retrieval::GradientDescentRetriever;
pub use deformation::ManifoldDeformer;
pub use forgetting::StrategicForgetting;

/// Implicit Neural Representation for manifold storage
pub struct ManifoldEngine<B: Backend> {
    /// Neural network representing the manifold
    network: Arc<RwLock<LearnedManifold<B>>>,
    /// Configuration
    config: ManifoldConfig,
    /// Device for computation
    device: B::Device,
    /// Stored patterns (for extraction)
    patterns: Arc<RwLock<Vec<Pattern>>>,
}

impl<B: Backend> ManifoldEngine<B> {
    /// Create a new manifold engine
    pub fn new(config: ManifoldConfig, device: B::Device) -> Self {
        let network = LearnedManifold::new(
            config.dimension,
            config.hidden_dim,
            config.hidden_layers,
            config.omega_0,
            &device,
        );

        Self {
            network: Arc::new(RwLock::new(network)),
            config,
            device,
            patterns: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Query manifold via gradient descent
    ///
    /// Implements the ManifoldRetrieve algorithm from pseudocode:
    /// - Initialize at query position
    /// - Gradient descent toward high-relevance regions
    /// - Extract patterns from converged region
    pub fn retrieve(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.config.dimension {
            return Err(Error::InvalidDimension {
                expected: self.config.dimension,
                got: query.len(),
            });
        }

        let retriever = GradientDescentRetriever::new(
            self.network.clone(),
            self.config.clone(),
            self.device.clone(),
        );

        retriever.retrieve(query, k, &self.patterns)
    }

    /// Continuous manifold deformation (replaces insert)
    ///
    /// Implements the ManifoldDeform algorithm from pseudocode:
    /// - No discrete insert operation
    /// - Continuous gradient update to manifold weights
    /// - Deformation proportional to salience
    pub fn deform(&mut self, pattern: Pattern, salience: f32) -> Result<ManifoldDelta> {
        if pattern.embedding.len() != self.config.dimension {
            return Err(Error::InvalidDimension {
                expected: self.config.dimension,
                got: pattern.embedding.len(),
            });
        }

        // Store pattern for later extraction
        self.patterns.write().push(pattern.clone());

        let mut deformer = ManifoldDeformer::new(
            self.network.clone(),
            self.config.learning_rate,
            self.device.clone(),
        );

        deformer.deform(&pattern, salience)
    }

    /// Strategic forgetting via manifold smoothing
    ///
    /// Implements the StrategicForget algorithm from pseudocode:
    /// - Identify low-salience regions
    /// - Apply smoothing kernel to reduce specificity
    /// - Prune near-zero weights
    pub fn forget(&mut self, salience_threshold: f32, decay_rate: f32) -> Result<usize> {
        let forgetter = StrategicForgetting::new(
            self.network.clone(),
            self.device.clone(),
        );

        forgetter.forget(
            &self.patterns,
            salience_threshold,
            decay_rate,
        )
    }

    /// Get number of stored patterns
    pub fn len(&self) -> usize {
        self.patterns.read().len()
    }

    /// Check if engine is empty
    pub fn is_empty(&self) -> bool {
        self.patterns.read().is_empty()
    }

    /// Get configuration
    pub fn config(&self) -> &ManifoldConfig {
        &self.config
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
    fn test_manifold_engine_creation() {
        let config = ManifoldConfig {
            dimension: 128,
            ..Default::default()
        };
        let device = Default::default();
        let engine = ManifoldEngine::<TestBackend>::new(config, device);

        assert_eq!(engine.len(), 0);
        assert!(engine.is_empty());
        assert_eq!(engine.config().dimension, 128);
    }

    #[test]
    fn test_deform_and_retrieve() {
        let config = ManifoldConfig {
            dimension: 64,
            max_descent_steps: 10,
            learning_rate: 0.01,
            ..Default::default()
        };
        let device = Default::default();
        let mut engine = ManifoldEngine::<TestBackend>::new(config, device);

        // Create and deform with a pattern
        let embedding = vec![1.0; 64];
        let pattern = create_test_pattern(embedding.clone(), 0.9);

        let result = engine.deform(pattern, 0.9);
        assert!(result.is_ok());
        assert_eq!(engine.len(), 1);

        // Retrieve similar patterns
        let results = engine.retrieve(&embedding, 1);
        assert!(results.is_ok());
    }

    #[test]
    fn test_invalid_dimension() {
        let config = ManifoldConfig {
            dimension: 128,
            ..Default::default()
        };
        let device = Default::default();
        let mut engine = ManifoldEngine::<TestBackend>::new(config, device);

        // Wrong dimension
        let embedding = vec![1.0; 64];
        let pattern = create_test_pattern(embedding.clone(), 0.9);

        let result = engine.deform(pattern, 0.9);
        assert!(result.is_err());

        let retrieve_result = engine.retrieve(&embedding, 1);
        assert!(retrieve_result.is_err());
    }
}
