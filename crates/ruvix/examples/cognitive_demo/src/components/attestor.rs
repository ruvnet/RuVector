//! Attestor component - Demonstrates attestation emission.
//!
//! The Attestor records proof attestations to the witness log, providing
//! auditability and verifiability for all pipeline operations.
//!
//! ## Syscalls Used
//!
//! - `attest_emit` (10,000 calls) - Emit attestation records
//!
//! ## Architecture
//!
//! ```text
//! +--------------------+
//! |      Attestor      |
//! |--------------------|
//! | receive_request    |<--- From other components
//! |        |           |
//! |        v           |
//! |  create_attestation|---> Build attestation record
//! |        |           |
//! |        v           |
//! |    attest_emit     |---> Witness Log (AppendOnly region)
//! +--------------------+
//! ```

use super::{Component, ComponentTickResult, KernelInterface};
use crate::{config, Result};
use ruvix_types::{CapHandle, ProofAttestation, ProofPayload, ProofTier, ProofToken, RegionHandle};
use sha2::{Digest, Sha256};

/// Attestation request from pipeline components.
#[derive(Debug, Clone, Copy)]
pub struct AttestationRequest {
    /// Operation hash to attest.
    pub operation_hash: [u8; 32],

    /// Proof tier used for the operation.
    pub proof_tier: ProofTier,

    /// Source component index.
    pub source_component: u8,

    /// Operation sequence number.
    pub sequence: u64,

    /// Coherence score at attestation time.
    pub coherence_score: u16,
}

impl AttestationRequest {
    /// Creates a new attestation request.
    pub fn new(operation_hash: [u8; 32], proof_tier: ProofTier, source_component: u8) -> Self {
        Self {
            operation_hash,
            proof_tier,
            source_component,
            sequence: 0,
            coherence_score: 5000,
        }
    }

    /// Sets the sequence number.
    pub fn with_sequence(mut self, sequence: u64) -> Self {
        self.sequence = sequence;
        self
    }

    /// Sets the coherence score.
    pub fn with_coherence(mut self, score: u16) -> Self {
        self.coherence_score = score;
        self
    }
}

/// Attestor component for witness log management.
pub struct Attestor {
    /// Component name.
    name: &'static str,

    /// Witness log region handle.
    witness_log: RegionHandle,

    /// Capability for witness log write.
    log_cap: CapHandle,

    /// Total attestations emitted.
    attestations_emitted: u64,

    /// Current witness log sequence.
    log_sequence: u64,

    /// Whether initialization is complete.
    initialized: bool,

    /// Whether component is in error state.
    error: bool,

    /// Pending attestation requests.
    pending_requests: Vec<AttestationRequest>,

    /// Maximum attestations.
    max_attestations: u64,

    /// Environment hash (constant for all attestations).
    environment_hash: [u8; 32],

    /// Verifier version.
    verifier_version: u32,

    /// Hash chain previous hash.
    previous_hash: [u8; 32],
}

impl Attestor {
    /// Creates a new Attestor.
    pub fn new(witness_log: RegionHandle, log_cap: CapHandle) -> Self {
        Self {
            name: "Attestor",
            witness_log,
            log_cap,
            attestations_emitted: 0,
            log_sequence: 0,
            initialized: false,
            error: false,
            pending_requests: Vec::with_capacity(256),
            max_attestations: config::FULL_PIPELINE_EVENTS as u64,
            environment_hash: [0u8; 32],
            verifier_version: 0x00_01_00_00, // v0.1.0
            previous_hash: [0u8; 32],
        }
    }

    /// Sets the maximum number of attestations.
    pub fn with_max_attestations(mut self, max: u64) -> Self {
        self.max_attestations = max;
        self
    }

    /// Sets the environment hash.
    pub fn with_environment_hash(mut self, hash: [u8; 32]) -> Self {
        self.environment_hash = hash;
        self
    }

    /// Sets the verifier version.
    pub fn with_verifier_version(mut self, version: u32) -> Self {
        self.verifier_version = version;
        self
    }

    /// Queues an attestation request.
    pub fn queue_request(&mut self, request: AttestationRequest) {
        self.pending_requests.push(request);
    }

    /// Queues a simple attestation for an operation.
    pub fn queue_attestation(
        &mut self,
        operation_hash: [u8; 32],
        proof_tier: ProofTier,
        source: u8,
        sequence: u64,
    ) {
        self.queue_request(
            AttestationRequest::new(operation_hash, proof_tier, source).with_sequence(sequence),
        );
    }

    /// Creates a proof attestation record.
    fn create_attestation(
        &self,
        request: &AttestationRequest,
        timestamp_ns: u64,
    ) -> ProofAttestation {
        // Compute reduction steps based on proof tier
        let reduction_steps = match request.proof_tier {
            ProofTier::Reflex => 10,
            ProofTier::Standard => 100,
            ProofTier::Deep => 1000,
        };

        // Cache hit rate (simulated - higher for Reflex tier)
        let cache_hit_rate_bps = match request.proof_tier {
            ProofTier::Reflex => 9500, // 95%
            ProofTier::Standard => 7000, // 70%
            ProofTier::Deep => 2000, // 20%
        };

        ProofAttestation::new(
            request.operation_hash,
            self.environment_hash,
            timestamp_ns,
            self.verifier_version,
            reduction_steps,
            cache_hit_rate_bps,
        )
    }

    /// Computes the hash chain for the attestation.
    fn compute_hash_chain(&self, attestation: &ProofAttestation) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(&self.previous_hash);
        hasher.update(&attestation.proof_term_hash);
        hasher.update(&attestation.verification_timestamp_ns.to_le_bytes());
        hasher.update(&attestation.reduction_steps.to_le_bytes());
        hasher.finalize().into()
    }

    /// Emits an attestation to the witness log.
    pub fn emit_attestation(
        &mut self,
        request: &AttestationRequest,
        kernel: &mut KernelInterface,
    ) -> Result<u64> {
        // Create the attestation record
        let attestation = self.create_attestation(request, kernel.current_time_ns);

        // Update hash chain
        self.previous_hash = self.compute_hash_chain(&attestation);

        // Create proof token for attestation
        let proof = kernel.generate_proof(attestation.proof_term_hash, request.proof_tier);

        // Emit to witness log
        let sequence = kernel.attest_emit(proof)?;

        self.log_sequence = sequence;
        self.attestations_emitted += 1;

        Ok(sequence)
    }

    /// Processes a batch of pending attestation requests.
    pub fn process_batch(&mut self, kernel: &mut KernelInterface, batch_size: u32) -> Result<u32> {
        let mut processed = 0;

        while processed < batch_size && !self.pending_requests.is_empty() {
            if self.attestations_emitted >= self.max_attestations {
                break;
            }

            let request = self.pending_requests.remove(0);
            self.emit_attestation(&request, kernel)?;
            processed += 1;
        }

        Ok(processed)
    }

    /// Returns the current log sequence number.
    pub fn log_sequence(&self) -> u64 {
        self.log_sequence
    }

    /// Returns the previous hash in the chain.
    pub fn previous_hash(&self) -> [u8; 32] {
        self.previous_hash
    }

    /// Returns attestation statistics.
    pub fn stats(&self) -> AttestorStats {
        AttestorStats {
            attestations_emitted: self.attestations_emitted,
            pending_count: self.pending_requests.len(),
            log_sequence: self.log_sequence,
        }
    }
}

/// Attestor statistics.
#[derive(Debug, Clone, Copy)]
pub struct AttestorStats {
    /// Total attestations emitted.
    pub attestations_emitted: u64,

    /// Pending request count.
    pub pending_count: usize,

    /// Current log sequence.
    pub log_sequence: u64,
}

impl Component for Attestor {
    fn name(&self) -> &'static str {
        self.name
    }

    fn initialize(&mut self) -> Result<()> {
        // Initialize hash chain with boot attestation
        let mut hasher = Sha256::new();
        hasher.update(b"RUVIX_WITNESS_LOG_INIT");
        hasher.update(&self.environment_hash);
        self.previous_hash = hasher.finalize().into();

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

        if self.attestations_emitted >= self.max_attestations {
            return Ok(ComponentTickResult::Finished);
        }

        if self.pending_requests.is_empty() {
            return Ok(ComponentTickResult::Idle);
        }

        Ok(ComponentTickResult::Processed(
            self.pending_requests.len() as u32,
        ))
    }

    fn shutdown(&mut self) -> Result<()> {
        self.pending_requests.clear();
        Ok(())
    }

    fn operation_count(&self) -> u64 {
        self.attestations_emitted
    }

    fn is_error(&self) -> bool {
        self.error
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_attestor() -> Attestor {
        Attestor::new(RegionHandle::new(1, 0), CapHandle::null())
    }

    #[test]
    fn test_attestor_creation() {
        let attestor = create_attestor();

        assert_eq!(attestor.name(), "Attestor");
        assert_eq!(attestor.attestations_emitted, 0);
        assert_eq!(attestor.log_sequence, 0);
    }

    #[test]
    fn test_attestation_request() {
        let request = AttestationRequest::new([0xAB; 32], ProofTier::Standard, 1)
            .with_sequence(100)
            .with_coherence(8000);

        assert_eq!(request.proof_tier, ProofTier::Standard);
        assert_eq!(request.source_component, 1);
        assert_eq!(request.sequence, 100);
        assert_eq!(request.coherence_score, 8000);
    }

    #[test]
    fn test_create_attestation() {
        let attestor = create_attestor().with_verifier_version(0x00_02_00_00);

        let request = AttestationRequest::new([0xAB; 32], ProofTier::Standard, 1);
        let attestation = attestor.create_attestation(&request, 1_000_000_000);

        assert_eq!(attestation.proof_term_hash, [0xAB; 32]);
        assert_eq!(attestation.verification_timestamp_ns, 1_000_000_000);
        assert_eq!(attestation.verifier_version, 0x00_02_00_00);
        assert_eq!(attestation.reduction_steps, 100); // Standard tier
        assert_eq!(attestation.cache_hit_rate_bps, 7000); // 70%
    }

    #[test]
    fn test_emit_attestation() {
        let mut attestor = create_attestor().with_max_attestations(100);
        let mut kernel = KernelInterface::new();

        attestor.initialize().unwrap();

        let request = AttestationRequest::new([0xAB; 32], ProofTier::Reflex, 0);
        let sequence = attestor.emit_attestation(&request, &mut kernel).unwrap();

        assert_eq!(sequence, 1);
        assert_eq!(attestor.attestations_emitted, 1);
        assert_eq!(kernel.stats.attest_emit, 1);
    }

    #[test]
    fn test_hash_chain() {
        let mut attestor = create_attestor();
        attestor.initialize().unwrap();

        let initial_hash = attestor.previous_hash;

        // Emit an attestation
        let mut kernel = KernelInterface::new();
        let request = AttestationRequest::new([0xAB; 32], ProofTier::Standard, 0);
        attestor.emit_attestation(&request, &mut kernel).unwrap();

        // Hash should change
        assert_ne!(attestor.previous_hash, initial_hash);

        let hash_after_first = attestor.previous_hash;

        // Emit another attestation
        let request2 = AttestationRequest::new([0xCD; 32], ProofTier::Standard, 0);
        attestor.emit_attestation(&request2, &mut kernel).unwrap();

        // Hash should change again
        assert_ne!(attestor.previous_hash, hash_after_first);
    }

    #[test]
    fn test_batch_processing() {
        let mut attestor = create_attestor().with_max_attestations(100);
        let mut kernel = KernelInterface::new();

        attestor.initialize().unwrap();

        // Queue requests
        for i in 0..10 {
            attestor.queue_attestation([i as u8; 32], ProofTier::Reflex, 0, i as u64);
        }

        // Process batch
        let processed = attestor.process_batch(&mut kernel, 5).unwrap();
        assert_eq!(processed, 5);
        assert_eq!(attestor.attestations_emitted, 5);
        assert_eq!(kernel.stats.attest_emit, 5);

        // Process remaining
        let processed = attestor.process_batch(&mut kernel, 10).unwrap();
        assert_eq!(processed, 5);
        assert_eq!(attestor.attestations_emitted, 10);
    }

    #[test]
    fn test_stats() {
        let mut attestor = create_attestor();

        attestor.queue_request(AttestationRequest::new([0; 32], ProofTier::Standard, 0));
        attestor.queue_request(AttestationRequest::new([1; 32], ProofTier::Standard, 1));

        let stats = attestor.stats();
        assert_eq!(stats.attestations_emitted, 0);
        assert_eq!(stats.pending_count, 2);
        assert_eq!(stats.log_sequence, 0);
    }

    #[test]
    fn test_component_tick() {
        let mut attestor = create_attestor();

        // Before initialization
        assert_eq!(attestor.tick().unwrap(), ComponentTickResult::Waiting);

        attestor.initialize().unwrap();

        // No pending requests
        assert_eq!(attestor.tick().unwrap(), ComponentTickResult::Idle);

        // Add requests
        attestor.queue_request(AttestationRequest::new([0; 32], ProofTier::Standard, 0));
        assert_eq!(attestor.tick().unwrap(), ComponentTickResult::Processed(1));
    }
}
