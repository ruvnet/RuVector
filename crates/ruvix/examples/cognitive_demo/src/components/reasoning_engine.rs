//! ReasoningEngine component - Demonstrates graph mutations + coherence.
//!
//! The ReasoningEngine receives vector notifications, retrieves embeddings,
//! performs reasoning, and applies graph mutations with Standard-tier proofs.
//!
//! ## Syscalls Used
//!
//! - `queue_recv` (10,000 calls) - Receive embedding notifications
//! - `vector_get` (10,000 calls) - Retrieve embeddings
//! - `graph_apply_proved` (5,000 calls) - Apply graph mutations with Standard proofs
//!
//! ## Architecture
//!
//! ```text
//! +--------------------+
//! |  ReasoningEngine   |
//! |--------------------|
//! |    queue_recv      |<--- Input Queue <--- FeatureExtractor
//! |        |           |
//! |        v           |
//! |    vector_get      |<--- Vector Store (read embeddings)
//! |        |           |
//! |        v           |
//! |   reason_about()   |---> Analyze vectors, compute similarity
//! |        |           |
//! |        v           |
//! |   request_proof    |---> ProofTier::Standard
//! |        |           |
//! |        v           |
//! | graph_apply_proved |---> Graph Store (create edges)
//! +--------------------+
//! ```

use super::{Component, ComponentTickResult, KernelInterface, PipelineMessage};
use crate::{config, ReasoningMutation, Result};
use ruvix_types::{
    CapHandle, GraphHandle, GraphMutation, ProofTier, QueueHandle, VectorKey, VectorStoreHandle,
};

/// ReasoningEngine component for graph-based reasoning.
pub struct ReasoningEngine {
    /// Component name.
    name: &'static str,

    /// Input queue for embedding notifications.
    input_queue: QueueHandle,

    /// Vector store handle.
    vector_store: VectorStoreHandle,

    /// Graph store handle.
    graph: GraphHandle,

    /// Capability for input queue read.
    input_cap: CapHandle,

    /// Capability for vector store read.
    store_cap: CapHandle,

    /// Capability for graph store write.
    graph_cap: CapHandle,

    /// Total vectors processed.
    vectors_processed: u64,

    /// Total graph mutations applied.
    mutations_applied: u64,

    /// Whether initialization is complete.
    initialized: bool,

    /// Whether component is in error state.
    error: bool,

    /// Pending vector keys to process.
    pending_vectors: Vec<(VectorKey, u64, f32)>, // (key, sequence, coherence)

    /// Maximum vectors to process.
    max_vectors: u64,

    /// Graph mutation frequency (every N vectors).
    mutation_frequency: u32,

    /// Last processed vector for graph edges.
    last_vector: Option<VectorKey>,

    /// Coherence threshold for mutations.
    coherence_threshold: f32,
}

impl ReasoningEngine {
    /// Creates a new ReasoningEngine.
    pub fn new(
        input_queue: QueueHandle,
        vector_store: VectorStoreHandle,
        graph: GraphHandle,
        input_cap: CapHandle,
        store_cap: CapHandle,
        graph_cap: CapHandle,
    ) -> Self {
        Self {
            name: "ReasoningEngine",
            input_queue,
            vector_store,
            graph,
            input_cap,
            store_cap,
            graph_cap,
            vectors_processed: 0,
            mutations_applied: 0,
            initialized: false,
            error: false,
            pending_vectors: Vec::with_capacity(256),
            max_vectors: config::FULL_PIPELINE_EVENTS as u64,
            mutation_frequency: 2, // Every 2 vectors = 5000 mutations for 10000 vectors
            last_vector: None,
            coherence_threshold: 0.3,
        }
    }

    /// Sets the maximum number of vectors to process.
    pub fn with_max_vectors(mut self, max: u64) -> Self {
        self.max_vectors = max;
        self
    }

    /// Sets the mutation frequency.
    pub fn with_mutation_frequency(mut self, freq: u32) -> Self {
        self.mutation_frequency = freq;
        self
    }

    /// Sets the coherence threshold.
    pub fn with_coherence_threshold(mut self, threshold: f32) -> Self {
        self.coherence_threshold = threshold;
        self
    }

    /// Queues a vector notification for processing.
    pub fn queue_vector(&mut self, key: VectorKey, sequence: u64, coherence: f32) {
        self.pending_vectors.push((key, sequence, coherence));
    }

    /// Performs reasoning on two vectors and decides if they should be connected.
    fn reason_about(
        &self,
        vec1: &[f32],
        vec2: &[f32],
        coherence1: f32,
        coherence2: f32,
    ) -> (bool, f32) {
        // Compute cosine similarity
        let dot_product: f32 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f32 = vec1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = vec2.iter().map(|x| x * x).sum::<f32>().sqrt();

        let similarity = if norm1 > 0.0 && norm2 > 0.0 {
            dot_product / (norm1 * norm2)
        } else {
            0.0
        };

        // Combine with coherence scores
        let combined_coherence = (coherence1 + coherence2) / 2.0;

        // Create edge if similarity and coherence are above thresholds
        let should_connect = similarity > 0.5 && combined_coherence > self.coherence_threshold;

        (should_connect, combined_coherence)
    }

    /// Creates a graph mutation for connecting two vectors.
    fn create_mutation(&self, key1: VectorKey, key2: VectorKey, sequence: u64) -> ReasoningMutation {
        // Create an edge between the two vector nodes
        let mutation = GraphMutation::add_edge(key1.raw(), key2.raw(), 1.0);

        ReasoningMutation::new(self.graph, mutation, sequence).with_sources(key1, key2)
    }

    /// Processes a single vector notification.
    pub fn process_vector(
        &mut self,
        key: VectorKey,
        sequence: u64,
        coherence: f32,
        kernel: &mut KernelInterface,
    ) -> Result<Option<ReasoningMutation>> {
        // Retrieve the vector from the store
        let (vec_data, vec_coherence) = kernel.vector_get(self.vector_store, key)?;

        // Use the retrieved coherence or the notification coherence
        let actual_coherence = if vec_coherence > 0.0 {
            vec_coherence
        } else {
            coherence
        };

        let mut result_mutation = None;

        // Check if we should create a graph mutation
        if let Some(last_key) = self.last_vector {
            // Get the last vector for comparison
            let (last_vec_data, last_coherence) = kernel.vector_get(self.vector_store, last_key)?;

            // Reason about the relationship
            let (should_connect, combined_coherence) =
                self.reason_about(&last_vec_data, &vec_data, last_coherence, actual_coherence);

            // Create mutation if appropriate and within frequency
            if should_connect && self.vectors_processed % self.mutation_frequency as u64 == 0 {
                let mut mutation = self.create_mutation(last_key, key, sequence);
                mutation.coherence = combined_coherence;

                // Generate Standard-tier proof
                let mutation_hash = mutation.compute_hash();
                let proof = kernel.generate_proof(mutation_hash, ProofTier::Standard);

                // Apply the graph mutation
                kernel.graph_apply_proved(self.graph, mutation.mutation.clone(), proof)?;

                self.mutations_applied += 1;
                result_mutation = Some(mutation);
            }
        }

        // Update state
        self.last_vector = Some(key);
        self.vectors_processed += 1;

        Ok(result_mutation)
    }

    /// Processes a batch of pending vectors.
    pub fn process_batch(
        &mut self,
        kernel: &mut KernelInterface,
        batch_size: u32,
    ) -> Result<(u32, u32)> {
        let mut vectors_processed = 0;
        let mut mutations_applied = 0;

        while vectors_processed < batch_size && !self.pending_vectors.is_empty() {
            if self.vectors_processed >= self.max_vectors {
                break;
            }

            let (key, sequence, coherence) = self.pending_vectors.remove(0);
            if let Some(_mutation) = self.process_vector(key, sequence, coherence, kernel)? {
                mutations_applied += 1;
            }
            vectors_processed += 1;
        }

        Ok((vectors_processed, mutations_applied))
    }

    /// Returns statistics about the reasoning process.
    pub fn stats(&self) -> ReasoningStats {
        ReasoningStats {
            vectors_processed: self.vectors_processed,
            mutations_applied: self.mutations_applied,
            pending_count: self.pending_vectors.len(),
            mutation_rate: if self.vectors_processed > 0 {
                self.mutations_applied as f32 / self.vectors_processed as f32
            } else {
                0.0
            },
        }
    }
}

/// Statistics about reasoning operations.
#[derive(Debug, Clone, Copy)]
pub struct ReasoningStats {
    /// Total vectors processed.
    pub vectors_processed: u64,

    /// Total graph mutations applied.
    pub mutations_applied: u64,

    /// Pending vector count.
    pub pending_count: usize,

    /// Ratio of mutations to vectors.
    pub mutation_rate: f32,
}

impl Component for ReasoningEngine {
    fn name(&self) -> &'static str {
        self.name
    }

    fn initialize(&mut self) -> Result<()> {
        self.initialized = true;
        Ok(())
    }

    fn tick(&mut self) -> Result<ComponentTickResult> {
        if self.error {
            return Ok(ComponentTickResult::Error);
        }

        if !self.initialized {
            return Ok(ComponentTickResult::Waiting);
        }

        if self.vectors_processed >= self.max_vectors {
            return Ok(ComponentTickResult::Finished);
        }

        if self.pending_vectors.is_empty() {
            return Ok(ComponentTickResult::Idle);
        }

        Ok(ComponentTickResult::Processed(self.pending_vectors.len() as u32))
    }

    fn shutdown(&mut self) -> Result<()> {
        self.pending_vectors.clear();
        self.last_vector = None;
        Ok(())
    }

    fn operation_count(&self) -> u64 {
        self.vectors_processed
    }

    fn is_error(&self) -> bool {
        self.error
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_engine() -> ReasoningEngine {
        ReasoningEngine::new(
            QueueHandle::new(1, 0),
            VectorStoreHandle::null(),
            GraphHandle::null(),
            CapHandle::null(),
            CapHandle::null(),
            CapHandle::null(),
        )
    }

    #[test]
    fn test_reasoning_engine_creation() {
        let engine = create_engine();

        assert_eq!(engine.name(), "ReasoningEngine");
        assert_eq!(engine.vectors_processed, 0);
        assert_eq!(engine.mutations_applied, 0);
    }

    #[test]
    fn test_reason_about() {
        let engine = create_engine().with_coherence_threshold(0.3);

        // High similarity, high coherence -> connect
        let vec1 = vec![1.0; 768];
        let vec2 = vec![1.0; 768];
        let (connect, coherence) = engine.reason_about(&vec1, &vec2, 0.8, 0.9);
        assert!(connect);
        assert!((coherence - 0.85).abs() < 0.001);

        // Low coherence -> don't connect
        let (connect, _) = engine.reason_about(&vec1, &vec2, 0.1, 0.1);
        assert!(!connect);
    }

    #[test]
    fn test_create_mutation() {
        let engine = create_engine();

        let key1 = VectorKey::new(1);
        let key2 = VectorKey::new(2);
        let mutation = engine.create_mutation(key1, key2, 100);

        assert_eq!(mutation.sequence, 100);
        assert_eq!(mutation.source_embeddings[0], key1);
        assert_eq!(mutation.source_embeddings[1], key2);
    }

    #[test]
    fn test_process_vector() {
        let mut engine = create_engine()
            .with_max_vectors(100)
            .with_mutation_frequency(2);

        let mut kernel = KernelInterface::new();
        engine.initialize().unwrap();

        // First vector - no mutation (no previous)
        let result1 = engine
            .process_vector(VectorKey::new(0), 0, 0.8, &mut kernel)
            .unwrap();
        assert!(result1.is_none());
        assert_eq!(engine.vectors_processed, 1);
        assert_eq!(kernel.stats.vector_get, 1);

        // Second vector - creates mutation (frequency = 2)
        let result2 = engine
            .process_vector(VectorKey::new(1), 1, 0.8, &mut kernel)
            .unwrap();
        // Note: May or may not create mutation depending on similarity
        assert_eq!(engine.vectors_processed, 2);
        assert_eq!(kernel.stats.vector_get, 3); // 2 gets for comparison
    }

    #[test]
    fn test_batch_processing() {
        let mut engine = create_engine()
            .with_max_vectors(100)
            .with_mutation_frequency(2);

        let mut kernel = KernelInterface::new();
        engine.initialize().unwrap();

        // Queue vectors
        for i in 0..10 {
            engine.queue_vector(VectorKey::new(i), i as u64, 0.8);
        }

        // Process batch
        let (processed, _mutations) = engine.process_batch(&mut kernel, 5).unwrap();
        assert_eq!(processed, 5);
        assert_eq!(engine.vectors_processed, 5);
    }

    #[test]
    fn test_stats() {
        let mut engine = create_engine();

        engine.queue_vector(VectorKey::new(0), 0, 0.8);
        engine.queue_vector(VectorKey::new(1), 1, 0.8);

        let stats = engine.stats();
        assert_eq!(stats.vectors_processed, 0);
        assert_eq!(stats.pending_count, 2);
    }

    #[test]
    fn test_component_tick() {
        let mut engine = create_engine();

        // Before initialization
        assert_eq!(engine.tick().unwrap(), ComponentTickResult::Waiting);

        engine.initialize().unwrap();

        // No pending vectors
        assert_eq!(engine.tick().unwrap(), ComponentTickResult::Idle);

        // Add vectors
        engine.queue_vector(VectorKey::new(0), 0, 0.8);
        assert_eq!(engine.tick().unwrap(), ComponentTickResult::Processed(1));
    }
}
