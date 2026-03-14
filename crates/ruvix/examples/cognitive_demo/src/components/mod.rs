//! Pipeline components for the cognitive demo.
//!
//! This module contains the 5 components that make up the cognitive pipeline:
//!
//! - [`SensorAdapter`] - Subscribes to sensors and generates perception events
//! - [`FeatureExtractor`] - Computes embeddings with proof-gated vector mutations
//! - [`ReasoningEngine`] - Applies graph mutations based on vector analysis
//! - [`Attestor`] - Emits attestation records to the witness log
//! - [`Coordinator`] - Spawns tasks, grants capabilities, manages timing

mod sensor_adapter;
mod feature_extractor;
mod reasoning_engine;
mod attestor;
mod coordinator;

pub use sensor_adapter::SensorAdapter;
pub use feature_extractor::FeatureExtractor;
pub use reasoning_engine::ReasoningEngine;
pub use attestor::Attestor;
pub use coordinator::Coordinator;

use crate::{PerceptionEvent, VectorEmbedding, ReasoningMutation, Result};
use ruvix_types::{
    CapHandle, GraphHandle, ProofToken, QueueHandle, TaskHandle, VectorKey, VectorStoreHandle,
};

/// Common trait for all pipeline components.
pub trait Component {
    /// Component name.
    fn name(&self) -> &'static str;

    /// Initialize the component with required handles.
    fn initialize(&mut self) -> Result<()>;

    /// Process one iteration of the component's work loop.
    fn tick(&mut self) -> Result<ComponentTickResult>;

    /// Shutdown the component gracefully.
    fn shutdown(&mut self) -> Result<()>;

    /// Returns the number of operations completed.
    fn operation_count(&self) -> u64;

    /// Returns true if the component is in error state.
    fn is_error(&self) -> bool;
}

/// Result of a component tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentTickResult {
    /// Component processed work successfully.
    Processed(u32),

    /// Component is idle (no work available).
    Idle,

    /// Component is waiting for input.
    Waiting,

    /// Component has finished all work.
    Finished,

    /// Component encountered an error.
    Error,
}

/// Message types for inter-component communication.
#[derive(Debug, Clone)]
pub enum PipelineMessage {
    /// Perception event from sensor.
    PerceptionEvent(PerceptionEvent),

    /// Vector embedding for reasoning.
    VectorEmbedding {
        key: VectorKey,
        source_sequence: u64,
        coherence: f32,
    },

    /// Graph mutation notification.
    GraphMutation {
        sequence: u64,
        coherence: f32,
    },

    /// Attestation request.
    AttestRequest {
        operation_hash: [u8; 32],
        proof_tier: u8,
    },

    /// Shutdown signal.
    Shutdown,
}

impl PipelineMessage {
    /// Serializes the message to bytes.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(128);

        match self {
            Self::PerceptionEvent(event) => {
                bytes.push(0); // Message type
                bytes.extend_from_slice(&event.timestamp_ns.to_le_bytes());
                bytes.push(event.sensor_type);
                bytes.push(event.priority);
                bytes.extend_from_slice(&event.data_hash);
                bytes.extend_from_slice(&event.coherence_score.to_le_bytes());
                bytes.extend_from_slice(&event.sequence.to_le_bytes());
            }
            Self::VectorEmbedding {
                key,
                source_sequence,
                coherence,
            } => {
                bytes.push(1); // Message type
                bytes.extend_from_slice(&key.raw().to_le_bytes());
                bytes.extend_from_slice(&source_sequence.to_le_bytes());
                bytes.extend_from_slice(&coherence.to_le_bytes());
            }
            Self::GraphMutation { sequence, coherence } => {
                bytes.push(2); // Message type
                bytes.extend_from_slice(&sequence.to_le_bytes());
                bytes.extend_from_slice(&coherence.to_le_bytes());
            }
            Self::AttestRequest {
                operation_hash,
                proof_tier,
            } => {
                bytes.push(3); // Message type
                bytes.extend_from_slice(operation_hash);
                bytes.push(*proof_tier);
            }
            Self::Shutdown => {
                bytes.push(255); // Message type
            }
        }

        bytes
    }

    /// Deserializes a message from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.is_empty() {
            return None;
        }

        match bytes[0] {
            0 if bytes.len() >= 53 => {
                let timestamp_ns = u64::from_le_bytes([
                    bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7], bytes[8],
                ]);
                let sensor_type = bytes[9];
                let priority = bytes[10];
                let mut data_hash = [0u8; 32];
                data_hash.copy_from_slice(&bytes[11..43]);
                let coherence_score = u16::from_le_bytes([bytes[43], bytes[44]]);
                let sequence = u64::from_le_bytes([
                    bytes[45], bytes[46], bytes[47], bytes[48], bytes[49], bytes[50], bytes[51],
                    bytes[52],
                ]);

                Some(Self::PerceptionEvent(PerceptionEvent {
                    timestamp_ns,
                    sensor_type,
                    priority,
                    data_hash,
                    coherence_score,
                    sequence,
                }))
            }
            1 if bytes.len() >= 21 => {
                let key_raw = u64::from_le_bytes([
                    bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7], bytes[8],
                ]);
                let key = VectorKey::new(key_raw);
                let source_sequence = u64::from_le_bytes([
                    bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
                    bytes[16],
                ]);
                let coherence = f32::from_le_bytes([bytes[17], bytes[18], bytes[19], bytes[20]]);

                Some(Self::VectorEmbedding {
                    key,
                    source_sequence,
                    coherence,
                })
            }
            2 if bytes.len() >= 13 => {
                let sequence = u64::from_le_bytes([
                    bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7], bytes[8],
                ]);
                let coherence = f32::from_le_bytes([bytes[9], bytes[10], bytes[11], bytes[12]]);

                Some(Self::GraphMutation { sequence, coherence })
            }
            3 if bytes.len() >= 34 => {
                let mut operation_hash = [0u8; 32];
                operation_hash.copy_from_slice(&bytes[1..33]);
                let proof_tier = bytes[33];

                Some(Self::AttestRequest {
                    operation_hash,
                    proof_tier,
                })
            }
            255 => Some(Self::Shutdown),
            _ => None,
        }
    }
}

/// Simulated kernel interface for components.
///
/// In a real kernel, these would be actual syscalls.
/// For testing, we simulate the behavior.
pub struct KernelInterface {
    /// Current time in nanoseconds.
    pub current_time_ns: u64,

    /// Nonce counter for proofs.
    nonce_counter: u64,

    /// Statistics counters.
    pub stats: KernelStats,
}

/// Kernel operation statistics.
#[derive(Debug, Clone, Default)]
pub struct KernelStats {
    /// task_spawn count.
    pub task_spawn: u64,
    /// cap_grant count.
    pub cap_grant: u64,
    /// region_map count.
    pub region_map: u64,
    /// queue_send count.
    pub queue_send: u64,
    /// queue_recv count.
    pub queue_recv: u64,
    /// timer_wait count.
    pub timer_wait: u64,
    /// rvf_mount count.
    pub rvf_mount: u64,
    /// attest_emit count.
    pub attest_emit: u64,
    /// vector_get count.
    pub vector_get: u64,
    /// vector_put_proved count.
    pub vector_put_proved: u64,
    /// graph_apply_proved count.
    pub graph_apply_proved: u64,
    /// sensor_subscribe count.
    pub sensor_subscribe: u64,
}

impl KernelInterface {
    /// Creates a new kernel interface.
    pub fn new() -> Self {
        Self {
            current_time_ns: 0,
            nonce_counter: 0,
            stats: KernelStats::default(),
        }
    }

    /// Advances time by the specified nanoseconds.
    pub fn advance_time(&mut self, ns: u64) {
        self.current_time_ns += ns;
    }

    /// Generates a proof token for the given mutation hash and tier.
    pub fn generate_proof(
        &mut self,
        mutation_hash: [u8; 32],
        tier: ruvix_types::ProofTier,
    ) -> ProofToken {
        self.nonce_counter += 1;
        ProofToken::new(
            mutation_hash,
            tier,
            ruvix_types::ProofPayload::Hash { hash: mutation_hash },
            self.current_time_ns + 1_000_000_000, // 1 second validity
            self.nonce_counter,
        )
    }

    /// Simulates task_spawn syscall.
    pub fn task_spawn(&mut self, _caps: &[CapHandle]) -> Result<TaskHandle> {
        self.stats.task_spawn += 1;
        Ok(TaskHandle::new(self.stats.task_spawn as u32, 0))
    }

    /// Simulates cap_grant syscall.
    pub fn cap_grant(
        &mut self,
        _target: TaskHandle,
        _cap: CapHandle,
        _rights: ruvix_types::CapRights,
    ) -> Result<CapHandle> {
        self.stats.cap_grant += 1;
        Ok(CapHandle::new(self.stats.cap_grant as u32, 0))
    }

    /// Simulates region_map syscall.
    pub fn region_map(
        &mut self,
        _size: usize,
        _policy: ruvix_types::RegionPolicy,
    ) -> Result<ruvix_types::RegionHandle> {
        self.stats.region_map += 1;
        Ok(ruvix_types::RegionHandle::new(self.stats.region_map as u32, 0))
    }

    /// Simulates queue_send syscall.
    pub fn queue_send(
        &mut self,
        _queue: QueueHandle,
        _msg: &[u8],
        _priority: ruvix_types::MsgPriority,
    ) -> Result<()> {
        self.stats.queue_send += 1;
        Ok(())
    }

    /// Simulates queue_recv syscall.
    pub fn queue_recv(&mut self, _queue: QueueHandle, _timeout_ns: u64) -> Result<Vec<u8>> {
        self.stats.queue_recv += 1;
        Ok(Vec::new())
    }

    /// Simulates timer_wait syscall.
    pub fn timer_wait(&mut self, deadline_ns: u64) -> Result<()> {
        self.stats.timer_wait += 1;
        if deadline_ns > self.current_time_ns {
            self.current_time_ns = deadline_ns;
        }
        Ok(())
    }

    /// Simulates rvf_mount syscall.
    pub fn rvf_mount(&mut self, _rvf_data: &[u8]) -> Result<ruvix_types::RvfMountHandle> {
        self.stats.rvf_mount += 1;
        Ok(ruvix_types::RvfMountHandle::new(self.stats.rvf_mount as u32, 0))
    }

    /// Simulates attest_emit syscall.
    pub fn attest_emit(&mut self, _proof: ProofToken) -> Result<u64> {
        self.stats.attest_emit += 1;
        Ok(self.stats.attest_emit)
    }

    /// Simulates vector_get syscall.
    pub fn vector_get(
        &mut self,
        _store: VectorStoreHandle,
        key: VectorKey,
    ) -> Result<(Vec<f32>, f32)> {
        self.stats.vector_get += 1;
        // Return a simulated non-zero vector for similarity calculations
        // Use key to generate deterministic but varied data
        let mut data = vec![0.0f32; crate::config::EMBEDDING_DIM];
        let key_val = key.raw();
        for (i, val) in data.iter_mut().enumerate() {
            // Generate pseudo-random normalized values based on key and index
            let seed = key_val.wrapping_add(i as u64);
            *val = ((seed % 1000) as f32 / 1000.0) * 2.0 - 1.0;
        }
        // Normalize the vector
        let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut data {
                *val /= norm;
            }
        }
        Ok((data, 0.75))
    }

    /// Simulates vector_put_proved syscall.
    pub fn vector_put_proved(
        &mut self,
        _store: VectorStoreHandle,
        _key: VectorKey,
        _data: &[f32],
        _proof: ProofToken,
    ) -> Result<()> {
        self.stats.vector_put_proved += 1;
        Ok(())
    }

    /// Simulates graph_apply_proved syscall.
    pub fn graph_apply_proved(
        &mut self,
        _graph: GraphHandle,
        _mutation: ruvix_types::GraphMutation,
        _proof: ProofToken,
    ) -> Result<()> {
        self.stats.graph_apply_proved += 1;
        Ok(())
    }

    /// Simulates sensor_subscribe syscall.
    pub fn sensor_subscribe(
        &mut self,
        _sensor: ruvix_types::SensorDescriptor,
        _queue: QueueHandle,
    ) -> Result<ruvix_types::SubscriptionHandle> {
        self.stats.sensor_subscribe += 1;
        Ok(ruvix_types::SubscriptionHandle::new(
            self.stats.sensor_subscribe as u32,
            0,
        ))
    }
}

impl Default for KernelInterface {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_message_roundtrip() {
        let event = PerceptionEvent::new(1000, 1, 42).with_priority(200);
        let msg = PipelineMessage::PerceptionEvent(event);
        let bytes = msg.to_bytes();
        let decoded = PipelineMessage::from_bytes(&bytes).unwrap();

        if let PipelineMessage::PerceptionEvent(decoded_event) = decoded {
            assert_eq!(decoded_event.timestamp_ns, 1000);
            assert_eq!(decoded_event.sensor_type, 1);
            assert_eq!(decoded_event.sequence, 42);
            assert_eq!(decoded_event.priority, 200);
        } else {
            panic!("Wrong message type");
        }
    }

    #[test]
    fn test_kernel_interface_stats() {
        let mut kernel = KernelInterface::new();

        kernel.task_spawn(&[]).unwrap();
        kernel.task_spawn(&[]).unwrap();
        kernel.queue_send(QueueHandle::null(), &[], ruvix_types::MsgPriority::Normal).unwrap();

        assert_eq!(kernel.stats.task_spawn, 2);
        assert_eq!(kernel.stats.queue_send, 1);
    }

    #[test]
    fn test_kernel_proof_generation() {
        let mut kernel = KernelInterface::new();
        kernel.current_time_ns = 1000;

        let proof = kernel.generate_proof([0u8; 32], ruvix_types::ProofTier::Reflex);

        assert_eq!(proof.tier, ruvix_types::ProofTier::Reflex);
        assert_eq!(proof.nonce, 1);
        assert!(!proof.is_expired(kernel.current_time_ns));
    }
}
