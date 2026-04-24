//! FeatureExtractor component - Demonstrates proof-gated vector mutations.
//!
//! The FeatureExtractor receives perception events from the SensorAdapter,
//! computes vector embeddings, and stores them with proof verification.
//!
//! ## Syscalls Used
//!
//! - `queue_recv` (10,000 calls) - Receive perception events
//! - `vector_put_proved` (10,000 calls) - Store embeddings with Reflex proofs
//! - `queue_send` (10,000 calls) - Forward to ReasoningEngine
//!
//! ## Architecture
//!
//! ```text
//! +--------------------+
//! |  FeatureExtractor  |
//! |--------------------|
//! |    queue_recv      |<--- Input Queue <--- SensorAdapter
//! |        |           |
//! |        v           |
//! |  compute_embedding |---> Embedding computation
//! |        |           |
//! |        v           |
//! |  request_proof     |---> ProofTier::Reflex
//! |        |           |
//! |        v           |
//! | vector_put_proved  |---> Vector Store
//! |        |           |
//! |        v           |
//! |    queue_send      |---> Output Queue ---> ReasoningEngine
//! +--------------------+
//! ```

use super::{Component, ComponentTickResult, KernelInterface, PipelineMessage};
use crate::{config, PerceptionEvent, Result, VectorEmbedding};
use ruvix_types::{CapHandle, MsgPriority, ProofTier, QueueHandle, VectorKey, VectorStoreHandle};
use sha2::{Digest, Sha256};

/// FeatureExtractor component for computing embeddings.
pub struct FeatureExtractor {
    /// Component name.
    name: &'static str,

    /// Input queue for perception events.
    input_queue: QueueHandle,

    /// Output queue for embedding notifications.
    output_queue: QueueHandle,

    /// Vector store handle.
    vector_store: VectorStoreHandle,

    /// Capability for input queue read.
    input_cap: CapHandle,

    /// Capability for output queue write.
    output_cap: CapHandle,

    /// Capability for vector store write.
    store_cap: CapHandle,

    /// Total events processed.
    events_processed: u64,

    /// Total vectors stored.
    vectors_stored: u64,

    /// Whether initialization is complete.
    initialized: bool,

    /// Whether component is in error state.
    error: bool,

    /// Pending events buffer.
    pending_events: Vec<PerceptionEvent>,

    /// Maximum events to process.
    max_events: u64,

    /// Model weights (simulated).
    #[allow(dead_code)]
    model_weights: [f32; 768],
}

impl FeatureExtractor {
    /// Creates a new FeatureExtractor.
    pub fn new(
        input_queue: QueueHandle,
        output_queue: QueueHandle,
        vector_store: VectorStoreHandle,
        input_cap: CapHandle,
        output_cap: CapHandle,
        store_cap: CapHandle,
    ) -> Self {
        Self {
            name: "FeatureExtractor",
            input_queue,
            output_queue,
            vector_store,
            input_cap,
            output_cap,
            store_cap,
            events_processed: 0,
            vectors_stored: 0,
            initialized: false,
            error: false,
            pending_events: Vec::with_capacity(256),
            max_events: config::FULL_PIPELINE_EVENTS as u64,
            model_weights: [0.0; 768],
        }
    }

    /// Sets the maximum number of events to process.
    pub fn with_max_events(mut self, max: u64) -> Self {
        self.max_events = max;
        self
    }

    /// Queues a perception event for processing.
    pub fn queue_event(&mut self, event: PerceptionEvent) {
        self.pending_events.push(event);
    }

    /// Computes an embedding from a perception event.
    ///
    /// This is a simplified embedding computation for demonstration.
    /// In a real system, this would involve the neural network.
    fn compute_embedding(&self, event: &PerceptionEvent) -> VectorEmbedding {
        let key = VectorKey::new(event.sequence);
        let mut embedding = VectorEmbedding::new(key, event.sequence);

        // Compute deterministic embedding from event data
        let mut hasher = Sha256::new();
        hasher.update(&event.data_hash);
        hasher.update(&event.sequence.to_le_bytes());
        let hash: [u8; 32] = hasher.finalize().into();

        // Generate embedding components from hash
        for i in 0..config::EMBEDDING_DIM {
            // Use hash bytes cyclically to generate f32 values
            let idx = i % 32;
            let val = hash[idx] as f32 / 255.0;

            // Add some variation based on position
            let position_factor = (i as f32 / config::EMBEDDING_DIM as f32) * 2.0 - 1.0;
            embedding.data[i] = (val * 2.0 - 1.0) * 0.5 + position_factor * 0.1;
        }

        // Normalize the embedding
        let norm: f32 = embedding.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut embedding.data {
                *val /= norm;
            }
        }

        embedding.coherence = event.coherence_f32();
        embedding
    }

    /// Processes a single event and stores the embedding.
    pub fn process_event(
        &mut self,
        event: &PerceptionEvent,
        kernel: &mut KernelInterface,
    ) -> Result<VectorKey> {
        // Compute the embedding
        let embedding = self.compute_embedding(event);

        // Generate proof for vector mutation (Reflex tier)
        let mutation_hash = embedding.compute_hash();
        let proof = kernel.generate_proof(mutation_hash, ProofTier::Reflex);

        // Store the vector with proof
        kernel.vector_put_proved(self.vector_store, embedding.key, &embedding.data, proof)?;

        self.vectors_stored += 1;

        // Send notification to reasoning engine
        let msg = PipelineMessage::VectorEmbedding {
            key: embedding.key,
            source_sequence: embedding.source_sequence,
            coherence: embedding.coherence,
        };
        let bytes = msg.to_bytes();
        kernel.queue_send(self.output_queue, &bytes, MsgPriority::Normal)?;

        self.events_processed += 1;

        Ok(embedding.key)
    }

    /// Processes a batch of pending events.
    pub fn process_batch(&mut self, kernel: &mut KernelInterface, batch_size: u32) -> Result<u32> {
        let mut processed = 0;

        while processed < batch_size && !self.pending_events.is_empty() {
            if self.events_processed >= self.max_events {
                break;
            }

            let event = self.pending_events.remove(0);
            self.process_event(&event, kernel)?;
            processed += 1;
        }

        Ok(processed)
    }

    /// Receives events from the input queue and processes them.
    pub fn receive_and_process(&mut self, kernel: &mut KernelInterface, count: u32) -> Result<u32> {
        let mut processed = 0;

        for _ in 0..count {
            if self.events_processed >= self.max_events {
                break;
            }

            // Receive from queue (simulated - would block in real kernel)
            let bytes = kernel.queue_recv(self.input_queue, 1_000_000)?;

            // In real implementation, parse the message
            // For simulation, we'll use pending events instead
            if let Some(event) = self.pending_events.pop() {
                self.process_event(&event, kernel)?;
                processed += 1;
            } else {
                // Simulate receiving by counting the recv call
                // but not actually processing since no pending events
                break;
            }
        }

        Ok(processed)
    }
}

impl Component for FeatureExtractor {
    fn name(&self) -> &'static str {
        self.name
    }

    fn initialize(&mut self) -> Result<()> {
        // Load model weights (simulated)
        // In a real system, this would read from the immutable region
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

        if self.events_processed >= self.max_events {
            return Ok(ComponentTickResult::Finished);
        }

        if self.pending_events.is_empty() {
            return Ok(ComponentTickResult::Idle);
        }

        Ok(ComponentTickResult::Processed(
            self.pending_events.len() as u32
        ))
    }

    fn shutdown(&mut self) -> Result<()> {
        self.pending_events.clear();
        Ok(())
    }

    fn operation_count(&self) -> u64 {
        self.events_processed
    }

    fn is_error(&self) -> bool {
        self.error
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_extractor() -> FeatureExtractor {
        FeatureExtractor::new(
            QueueHandle::new(1, 0),
            QueueHandle::new(2, 0),
            VectorStoreHandle::null(),
            CapHandle::null(),
            CapHandle::null(),
            CapHandle::null(),
        )
    }

    #[test]
    fn test_feature_extractor_creation() {
        let extractor = create_extractor();

        assert_eq!(extractor.name(), "FeatureExtractor");
        assert_eq!(extractor.events_processed, 0);
        assert_eq!(extractor.vectors_stored, 0);
    }

    #[test]
    fn test_embedding_computation() {
        let extractor = create_extractor();
        let event = PerceptionEvent::new(1000, 1, 42).with_coherence(7500);

        let embedding = extractor.compute_embedding(&event);

        assert_eq!(embedding.source_sequence, 42);
        assert!((embedding.coherence - 0.75).abs() < 0.001);

        // Check embedding is normalized
        let norm: f32 = embedding.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_embedding_determinism() {
        let extractor = create_extractor();
        let event = PerceptionEvent::new(1000, 1, 42).with_data_hash([0xAB; 32]);

        let emb1 = extractor.compute_embedding(&event);
        let emb2 = extractor.compute_embedding(&event);

        // Same event should produce same embedding
        assert_eq!(emb1.data, emb2.data);
        assert_eq!(emb1.compute_hash(), emb2.compute_hash());
    }

    #[test]
    fn test_process_event() {
        let mut extractor = create_extractor().with_max_events(100);
        let mut kernel = KernelInterface::new();

        extractor.initialize().unwrap();

        let event = PerceptionEvent::new(1000, 1, 0);
        let key = extractor.process_event(&event, &mut kernel).unwrap();

        assert_eq!(key.raw(), 0);
        assert_eq!(extractor.events_processed, 1);
        assert_eq!(extractor.vectors_stored, 1);
        assert_eq!(kernel.stats.vector_put_proved, 1);
        assert_eq!(kernel.stats.queue_send, 1);
    }

    #[test]
    fn test_batch_processing() {
        let mut extractor = create_extractor().with_max_events(100);
        let mut kernel = KernelInterface::new();

        extractor.initialize().unwrap();

        // Queue 50 events
        for i in 0..50 {
            extractor.queue_event(PerceptionEvent::new(i * 1000, 1, i));
        }

        // Process in batches
        let processed1 = extractor.process_batch(&mut kernel, 20).unwrap();
        assert_eq!(processed1, 20);

        let processed2 = extractor.process_batch(&mut kernel, 40).unwrap();
        assert_eq!(processed2, 30); // Only 30 remaining

        assert_eq!(extractor.events_processed, 50);
        assert_eq!(kernel.stats.vector_put_proved, 50);
    }

    #[test]
    fn test_component_tick() {
        let mut extractor = create_extractor();

        // Before initialization
        assert_eq!(extractor.tick().unwrap(), ComponentTickResult::Waiting);

        extractor.initialize().unwrap();

        // No pending events
        assert_eq!(extractor.tick().unwrap(), ComponentTickResult::Idle);

        // Add events
        extractor.queue_event(PerceptionEvent::default());
        assert_eq!(extractor.tick().unwrap(), ComponentTickResult::Processed(1));
    }
}
