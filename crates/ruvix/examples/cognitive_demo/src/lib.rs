//! # RuVix Demo - Comprehensive RVF Package Demonstrating ALL Kernel Features
//!
//! This crate provides a comprehensive demonstration of all RuVix kernel features
//! as specified in ADR-087. It implements a complete cognitive pipeline with:
//!
//! - **5 Components**: SensorAdapter, FeatureExtractor, ReasoningEngine, Attestor, Coordinator
//! - **3 Region Types**: Immutable, AppendOnly, Slab
//! - **3 Proof Tiers**: Reflex, Standard, Deep
//! - **All 12 Syscalls**: Demonstrated with proper capability and proof gating
//!
//! ## Architecture
//!
//! ```text
//!                     +-----------------+
//!                     |   Coordinator   |
//!                     | task_spawn      |
//!                     | cap_grant       |
//!                     | timer_wait      |
//!                     +--------+--------+
//!                              |
//!         +--------------------+--------------------+
//!         |                    |                    |
//! +-------v-------+    +-------v-------+    +-------v-------+
//! | SensorAdapter |    |FeatureExtract |    |   Attestor    |
//! | sensor_subscr |    | vector_put    |    | attest_emit   |
//! | queue_send    |--->| queue_recv    |--->|               |
//! +---------------+    | queue_send    |    +---------------+
//!                      +-------+-------+
//!                              |
//!                      +-------v-------+
//!                      |ReasoningEngine|
//!                      | vector_get    |
//!                      | graph_apply   |
//!                      | queue_recv    |
//!                      +---------------+
//! ```
//!
//! ## Memory Schema
//!
//! | Region | Type | Size | Purpose |
//! |--------|------|------|---------|
//! | `model_weights` | Immutable | 1 MiB | Pre-trained model weights |
//! | `witness_log` | AppendOnly | 64 KiB | Attestation records |
//! | `vector_store` | Slab | 3 MiB | Vector embeddings |
//!
//! ## Proof Policy
//!
//! | Operation | Tier | Rationale |
//! |-----------|------|-----------|
//! | Vector mutations | Reflex | High-frequency, < 1us verification |
//! | Graph mutations | Standard | Merkle witness, < 100us |
//! | Structural changes | Deep | Full coherence check, < 10ms |
//!
//! ## Feature Coverage
//!
//! | Syscall | Component | Count |
//! |---------|-----------|-------|
//! | `task_spawn` | Coordinator | 5 |
//! | `cap_grant` | Coordinator | 20 |
//! | `region_map` | Boot | 3 |
//! | `queue_send` | All | 10,000 |
//! | `queue_recv` | All | 10,000 |
//! | `timer_wait` | Coordinator | 100 |
//! | `rvf_mount` | Boot | 1 |
//! | `attest_emit` | Attestor | 10,000 |
//! | `vector_get` | ReasoningEngine | 10,000 |
//! | `vector_put_proved` | FeatureExtractor | 10,000 |
//! | `graph_apply_proved` | ReasoningEngine | 5,000 |
//! | `sensor_subscribe` | SensorAdapter | 1 |

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod components;
pub mod manifest;
pub mod pipeline;
pub mod stats;

// Re-export core types
pub use manifest::{DemoManifest, DemoRegion, ProofPolicyConfig, RollbackHookConfig};
pub use pipeline::{CognitivePipeline, PipelineConfig, PipelineState};
pub use stats::{FeatureCoverage, SyscallStats};

// Re-export commonly used types from dependencies
pub use ruvix_types::{
    CapHandle, CapRights, GraphHandle, GraphMutation, KernelError, MsgPriority, ProofAttestation,
    ProofPayload, ProofTier, ProofToken, QueueHandle, RegionHandle, RegionPolicy, RvfComponentId,
    RvfMountHandle, SensorDescriptor, SubscriptionHandle, TaskHandle, TimerSpec, VectorKey,
    VectorStoreConfig, VectorStoreHandle,
};

/// Result type for demo operations.
pub type Result<T> = core::result::Result<T, KernelError>;

/// Demo configuration constants.
pub mod config {
    /// Number of perception events to process in full pipeline test.
    pub const FULL_PIPELINE_EVENTS: usize = 10_000;

    /// Number of graph mutations (every 2 vector mutations).
    pub const GRAPH_MUTATIONS: usize = 5_000;

    /// Timer wait count for coordinator.
    pub const TIMER_WAITS: usize = 100;

    /// Capabilities granted by coordinator.
    pub const CAP_GRANTS: usize = 20;

    /// Tasks spawned by coordinator.
    pub const TASK_SPAWNS: usize = 5;

    /// Model weights region size (1 MiB).
    pub const MODEL_WEIGHTS_SIZE: usize = 1024 * 1024;

    /// Witness log region max size (64 KiB).
    pub const WITNESS_LOG_MAX_SIZE: usize = 64 * 1024;

    /// Vector store slot size (3 KiB per vector).
    pub const VECTOR_SLOT_SIZE: usize = 3 * 1024;

    /// Vector store slot count.
    pub const VECTOR_SLOT_COUNT: usize = 1024;

    /// Vector embedding dimension.
    pub const EMBEDDING_DIM: usize = 768;

    /// Queue capacity for inter-component communication.
    pub const QUEUE_CAPACITY: usize = 256;

    /// Maximum message size in queues.
    pub const MAX_MESSAGE_SIZE: usize = 4096;
}

/// Perception event generated by SensorAdapter.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct PerceptionEvent {
    /// Event timestamp (nanoseconds since epoch).
    pub timestamp_ns: u64,

    /// Sensor type that generated this event.
    pub sensor_type: u8,

    /// Event priority (0-255).
    pub priority: u8,

    /// Raw sensor data hash.
    pub data_hash: [u8; 32],

    /// Coherence score (0.0 - 1.0 as u16 0-10000).
    pub coherence_score: u16,

    /// Sequence number for ordering.
    pub sequence: u64,
}

impl PerceptionEvent {
    /// Creates a new perception event.
    #[inline]
    #[must_use]
    pub const fn new(timestamp_ns: u64, sensor_type: u8, sequence: u64) -> Self {
        Self {
            timestamp_ns,
            sensor_type,
            priority: 128,
            data_hash: [0u8; 32],
            coherence_score: 5000, // 50%
            sequence,
        }
    }

    /// Creates an event with the specified priority.
    #[inline]
    #[must_use]
    pub const fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Creates an event with the specified coherence score.
    #[inline]
    #[must_use]
    pub fn with_coherence(mut self, score: u16) -> Self {
        self.coherence_score = if score > 10000 { 10000 } else { score };
        self
    }

    /// Sets the data hash.
    #[inline]
    #[must_use]
    pub fn with_data_hash(mut self, hash: [u8; 32]) -> Self {
        self.data_hash = hash;
        self
    }

    /// Returns the coherence score as f32 (0.0 - 1.0).
    #[inline]
    #[must_use]
    pub fn coherence_f32(&self) -> f32 {
        self.coherence_score as f32 / 10000.0
    }
}

impl Default for PerceptionEvent {
    fn default() -> Self {
        Self::new(0, 0, 0)
    }
}

/// Vector embedding computed from perception event.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorEmbedding {
    /// Vector key for storage.
    pub key: VectorKey,

    /// Embedding data (f32 components).
    pub data: [f32; config::EMBEDDING_DIM],

    /// Source event sequence.
    pub source_sequence: u64,

    /// Coherence score.
    pub coherence: f32,
}

impl VectorEmbedding {
    /// Creates a new vector embedding.
    #[must_use]
    pub fn new(key: VectorKey, source_sequence: u64) -> Self {
        Self {
            key,
            data: [0.0; config::EMBEDDING_DIM],
            source_sequence,
            coherence: 0.5,
        }
    }

    /// Sets the embedding data.
    pub fn set_data(&mut self, data: &[f32]) {
        let len = data.len().min(config::EMBEDDING_DIM);
        self.data[..len].copy_from_slice(&data[..len]);
    }

    /// Computes a simple hash of the embedding for proofs.
    #[must_use]
    pub fn compute_hash(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();

        // Hash the key
        hasher.update(self.key.raw().to_le_bytes());

        // Hash the data (as bytes)
        for &val in &self.data {
            hasher.update(val.to_le_bytes());
        }

        // Hash metadata
        hasher.update(self.source_sequence.to_le_bytes());
        hasher.update(self.coherence.to_le_bytes());

        hasher.finalize().into()
    }
}

impl Default for VectorEmbedding {
    fn default() -> Self {
        Self::new(VectorKey::new(0), 0)
    }
}

/// Graph mutation descriptor for reasoning.
#[derive(Debug, Clone, PartialEq)]
pub struct ReasoningMutation {
    /// Target graph handle.
    pub graph: GraphHandle,

    /// Graph mutation to apply.
    pub mutation: GraphMutation,

    /// Source embeddings that triggered this mutation.
    pub source_embeddings: [VectorKey; 2],

    /// Computed coherence score for the mutation.
    pub coherence: f32,

    /// Mutation sequence number.
    pub sequence: u64,
}

impl ReasoningMutation {
    /// Creates a new reasoning mutation.
    #[must_use]
    pub fn new(graph: GraphHandle, mutation: GraphMutation, sequence: u64) -> Self {
        Self {
            graph,
            mutation,
            source_embeddings: [VectorKey::new(0), VectorKey::new(0)],
            coherence: 0.5,
            sequence,
        }
    }

    /// Sets the source embeddings.
    pub fn with_sources(mut self, src1: VectorKey, src2: VectorKey) -> Self {
        self.source_embeddings = [src1, src2];
        self
    }

    /// Computes the mutation hash for proofs.
    #[must_use]
    pub fn compute_hash(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();

        // Hash the graph handle
        hasher.update(self.graph.raw().id.to_le_bytes());

        // Hash the mutation kind
        hasher.update([self.mutation.kind as u8]);

        // Hash the source keys
        hasher.update(self.source_embeddings[0].raw().to_le_bytes());
        hasher.update(self.source_embeddings[1].raw().to_le_bytes());

        // Hash coherence and sequence
        hasher.update(self.coherence.to_le_bytes());
        hasher.update(self.sequence.to_le_bytes());

        hasher.finalize().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perception_event_creation() {
        let event = PerceptionEvent::new(1000, 1, 42)
            .with_priority(200)
            .with_coherence(7500);

        assert_eq!(event.timestamp_ns, 1000);
        assert_eq!(event.sensor_type, 1);
        assert_eq!(event.sequence, 42);
        assert_eq!(event.priority, 200);
        assert_eq!(event.coherence_score, 7500);
        assert!((event.coherence_f32() - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_vector_embedding_hash() {
        let mut emb1 = VectorEmbedding::new(VectorKey::new(1), 100);
        let mut emb2 = VectorEmbedding::new(VectorKey::new(1), 100);

        // Same data should produce same hash
        emb1.set_data(&[1.0, 2.0, 3.0]);
        emb2.set_data(&[1.0, 2.0, 3.0]);
        assert_eq!(emb1.compute_hash(), emb2.compute_hash());

        // Different data should produce different hash
        emb2.set_data(&[1.0, 2.0, 4.0]);
        assert_ne!(emb1.compute_hash(), emb2.compute_hash());
    }

    #[test]
    fn test_config_constants() {
        assert_eq!(config::FULL_PIPELINE_EVENTS, 10_000);
        assert_eq!(config::GRAPH_MUTATIONS, 5_000);
        assert_eq!(config::MODEL_WEIGHTS_SIZE, 1024 * 1024);
        assert_eq!(
            config::VECTOR_SLOT_SIZE * config::VECTOR_SLOT_COUNT,
            3 * 1024 * 1024
        );
    }
}
