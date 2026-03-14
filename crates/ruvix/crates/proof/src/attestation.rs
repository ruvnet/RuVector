//! Proof attestation generation and witness logging.
//!
//! Generates 82-byte ADR-047 compatible attestations and maintains
//! a kernel witness log for verified proofs.

use crate::error::ProofResult;
use crate::VERIFIER_VERSION;
use ruvix_types::{ProofAttestation, ProofTier, ProofToken, ATTESTATION_SIZE};

/// Maximum entries in the witness log.
const MAX_WITNESS_LOG_ENTRIES: usize = 256;

/// Builder for creating proof attestations.
#[derive(Debug)]
pub struct AttestationBuilder {
    /// Proof term hash (32 bytes).
    proof_term_hash: [u8; 32],
    /// Environment hash (32 bytes).
    environment_hash: [u8; 32],
    /// Verification timestamp (ns since epoch).
    timestamp_ns: u64,
    /// Number of reduction steps consumed.
    reduction_steps: u32,
    /// Cache hit rate (0-10000).
    cache_hit_rate_bps: u16,
}

impl AttestationBuilder {
    /// Creates a new attestation builder.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            proof_term_hash: [0u8; 32],
            environment_hash: [0u8; 32],
            timestamp_ns: 0,
            reduction_steps: 0,
            cache_hit_rate_bps: 0,
        }
    }

    /// Sets the proof term hash.
    #[must_use]
    pub const fn proof_term_hash(mut self, hash: [u8; 32]) -> Self {
        self.proof_term_hash = hash;
        self
    }

    /// Sets the environment hash.
    #[must_use]
    pub const fn environment_hash(mut self, hash: [u8; 32]) -> Self {
        self.environment_hash = hash;
        self
    }

    /// Sets the verification timestamp.
    #[must_use]
    pub const fn timestamp_ns(mut self, timestamp: u64) -> Self {
        self.timestamp_ns = timestamp;
        self
    }

    /// Sets the number of reduction steps.
    #[must_use]
    pub const fn reduction_steps(mut self, steps: u32) -> Self {
        self.reduction_steps = steps;
        self
    }

    /// Sets the cache hit rate in basis points.
    #[must_use]
    pub const fn cache_hit_rate_bps(mut self, rate: u16) -> Self {
        self.cache_hit_rate_bps = rate;
        self
    }

    /// Builds the attestation from a verified proof token.
    #[must_use]
    pub fn from_token(token: &ProofToken, timestamp_ns: u64) -> Self {
        // Compute proof term hash from mutation hash and tier
        let mut proof_term_hash = [0u8; 32];
        proof_term_hash[..24].copy_from_slice(&token.mutation_hash[..24]);
        proof_term_hash[24] = token.tier as u8;
        proof_term_hash[25..29].copy_from_slice(&token.valid_until_ns.to_le_bytes()[..4]);
        proof_term_hash[29..32].copy_from_slice(&token.nonce.to_le_bytes()[..3]);

        // Compute environment hash from payload
        let environment_hash = Self::compute_environment_hash(token);

        // Estimate reduction steps based on tier
        let reduction_steps = match token.tier {
            ProofTier::Reflex => 1,
            ProofTier::Standard => 32,
            ProofTier::Deep => 256,
        };

        Self {
            proof_term_hash,
            environment_hash,
            timestamp_ns,
            reduction_steps,
            cache_hit_rate_bps: 0,
        }
    }

    /// Computes environment hash from proof payload.
    fn compute_environment_hash(token: &ProofToken) -> [u8; 32] {
        let mut hash = [0u8; 32];

        match &token.payload {
            ruvix_types::ProofPayload::Hash { hash: h } => {
                hash[..16].copy_from_slice(&h[..16]);
                hash[16..32].copy_from_slice(&h[16..32]);
            }
            ruvix_types::ProofPayload::MerkleWitness {
                root,
                leaf_index,
                path_len,
                ..
            } => {
                hash[..20].copy_from_slice(&root[..20]);
                hash[20..24].copy_from_slice(&leaf_index.to_le_bytes());
                hash[24] = *path_len;
            }
            ruvix_types::ProofPayload::CoherenceCert {
                score_before,
                score_after,
                partition_id,
                ..
            } => {
                hash[0..2].copy_from_slice(&score_before.to_le_bytes());
                hash[2..4].copy_from_slice(&score_after.to_le_bytes());
                hash[4..8].copy_from_slice(&partition_id.to_le_bytes());
            }
        }

        hash
    }

    /// Builds the proof attestation.
    #[must_use]
    pub fn build(self) -> ProofAttestation {
        ProofAttestation::new(
            self.proof_term_hash,
            self.environment_hash,
            self.timestamp_ns,
            VERIFIER_VERSION,
            self.reduction_steps,
            self.cache_hit_rate_bps,
        )
    }
}

impl Default for AttestationBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Witness log entry.
#[derive(Debug, Clone, Copy)]
pub struct WitnessEntry {
    /// The attestation.
    pub attestation: ProofAttestation,
    /// Entry index in the log.
    pub index: u32,
    /// Proof tier that was verified.
    pub tier: ProofTier,
}

/// Kernel witness log for recording verified proofs.
#[derive(Debug)]
pub struct WitnessLog {
    /// Fixed-size array of entries.
    entries: [Option<WitnessEntry>; MAX_WITNESS_LOG_ENTRIES],
    /// Number of entries in the log.
    count: usize,
    /// Next index for circular buffer.
    next_index: u32,
    /// Total attestations recorded.
    total_recorded: u64,
}

impl WitnessLog {
    /// Creates a new witness log.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: [None; MAX_WITNESS_LOG_ENTRIES],
            count: 0,
            next_index: 0,
            total_recorded: u64::MAX, // Will wrap to 0 on first append
        }
    }

    /// Returns the number of entries in the log.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.count
    }

    /// Returns true if the log is empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Returns the total number of attestations recorded.
    #[inline]
    #[must_use]
    pub const fn total_recorded(&self) -> u64 {
        self.total_recorded.wrapping_add(1)
    }

    /// Appends an attestation to the witness log.
    pub fn append(
        &mut self,
        attestation: ProofAttestation,
        tier: ProofTier,
    ) -> ProofResult<WitnessEntry> {
        let slot = (self.next_index as usize) % MAX_WITNESS_LOG_ENTRIES;

        let entry = WitnessEntry {
            attestation,
            index: self.next_index,
            tier,
        };

        // Insert into log (circular buffer)
        if self.entries[slot].is_some() && self.count == MAX_WITNESS_LOG_ENTRIES {
            // Log is full, overwrite oldest
        } else {
            self.count = self.count.saturating_add(1).min(MAX_WITNESS_LOG_ENTRIES);
        }

        self.entries[slot] = Some(entry);
        self.next_index = self.next_index.wrapping_add(1);
        self.total_recorded = self.total_recorded.wrapping_add(1);

        Ok(entry)
    }

    /// Gets an entry by index.
    #[must_use]
    pub fn get(&self, index: u32) -> Option<&WitnessEntry> {
        let slot = (index as usize) % MAX_WITNESS_LOG_ENTRIES;
        self.entries[slot].as_ref().filter(|e| e.index == index)
    }

    /// Returns the most recent entry.
    #[must_use]
    pub fn latest(&self) -> Option<&WitnessEntry> {
        if self.count == 0 {
            return None;
        }
        let prev_index = self.next_index.wrapping_sub(1);
        self.get(prev_index)
    }

    /// Clears all entries from the log.
    pub fn clear(&mut self) {
        for entry in &mut self.entries {
            *entry = None;
        }
        self.count = 0;
        self.next_index = 0;
    }

    /// Serializes an attestation to bytes (82 bytes per ADR-047).
    pub fn serialize_attestation(attestation: &ProofAttestation) -> [u8; ATTESTATION_SIZE] {
        let mut buf = [0u8; ATTESTATION_SIZE];

        // proof_term_hash: 32 bytes
        buf[0..32].copy_from_slice(&attestation.proof_term_hash);

        // environment_hash: 32 bytes
        buf[32..64].copy_from_slice(&attestation.environment_hash);

        // verification_timestamp_ns: 8 bytes
        buf[64..72].copy_from_slice(&attestation.verification_timestamp_ns.to_le_bytes());

        // verifier_version: 4 bytes
        buf[72..76].copy_from_slice(&attestation.verifier_version.to_le_bytes());

        // reduction_steps: 4 bytes
        buf[76..80].copy_from_slice(&attestation.reduction_steps.to_le_bytes());

        // cache_hit_rate_bps: 2 bytes
        buf[80..82].copy_from_slice(&attestation.cache_hit_rate_bps.to_le_bytes());

        buf
    }

    /// Deserializes an attestation from bytes.
    pub fn deserialize_attestation(data: &[u8; ATTESTATION_SIZE]) -> ProofAttestation {
        let mut proof_term_hash = [0u8; 32];
        proof_term_hash.copy_from_slice(&data[0..32]);

        let mut environment_hash = [0u8; 32];
        environment_hash.copy_from_slice(&data[32..64]);

        let verification_timestamp_ns = u64::from_le_bytes([
            data[64], data[65], data[66], data[67], data[68], data[69], data[70], data[71],
        ]);

        let verifier_version =
            u32::from_le_bytes([data[72], data[73], data[74], data[75]]);

        let reduction_steps =
            u32::from_le_bytes([data[76], data[77], data[78], data[79]]);

        let cache_hit_rate_bps = u16::from_le_bytes([data[80], data[81]]);

        ProofAttestation::new(
            proof_term_hash,
            environment_hash,
            verification_timestamp_ns,
            verifier_version,
            reduction_steps,
            cache_hit_rate_bps,
        )
    }
}

impl Default for WitnessLog {
    fn default() -> Self {
        Self::new()
    }
}

/// Creates an attestation from a verified proof token and appends to witness log.
pub fn create_and_log_attestation(
    token: &ProofToken,
    timestamp_ns: u64,
    log: &mut WitnessLog,
) -> ProofResult<WitnessEntry> {
    let attestation = AttestationBuilder::from_token(token, timestamp_ns).build();
    log.append(attestation, token.tier)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvix_types::{ProofPayload, ProofTier};

    fn make_test_token() -> ProofToken {
        ProofToken::new(
            [0xABu8; 32],
            ProofTier::Standard,
            ProofPayload::MerkleWitness {
                root: [0xCDu8; 32],
                leaf_index: 42,
                path_len: 5,
                path: [[0u8; 32]; 32],
            },
            1_000_000,
            12345,
        )
    }

    #[test]
    fn test_attestation_builder() {
        let attestation = AttestationBuilder::new()
            .proof_term_hash([1u8; 32])
            .environment_hash([2u8; 32])
            .timestamp_ns(1000)
            .reduction_steps(50)
            .cache_hit_rate_bps(7500)
            .build();

        assert_eq!(attestation.proof_term_hash, [1u8; 32]);
        assert_eq!(attestation.environment_hash, [2u8; 32]);
        assert_eq!(attestation.verification_timestamp_ns, 1000);
        assert_eq!(attestation.verifier_version, VERIFIER_VERSION);
        assert_eq!(attestation.reduction_steps, 50);
        assert_eq!(attestation.cache_hit_rate_bps, 7500);
    }

    #[test]
    fn test_attestation_from_token() {
        let token = make_test_token();
        let attestation = AttestationBuilder::from_token(&token, 5000).build();

        assert_eq!(attestation.verification_timestamp_ns, 5000);
        assert_eq!(attestation.verifier_version, VERIFIER_VERSION);
        // Standard tier should have 32 reduction steps
        assert_eq!(attestation.reduction_steps, 32);
    }

    #[test]
    fn test_attestation_size() {
        assert_eq!(ATTESTATION_SIZE, 82);
    }

    #[test]
    fn test_attestation_serialization() {
        let attestation = AttestationBuilder::new()
            .proof_term_hash([0xAAu8; 32])
            .environment_hash([0xBBu8; 32])
            .timestamp_ns(0x123456789ABCDEF0)
            .reduction_steps(0xDEADBEEF)
            .cache_hit_rate_bps(0xCAFE)
            .build();

        let bytes = WitnessLog::serialize_attestation(&attestation);
        let restored = WitnessLog::deserialize_attestation(&bytes);

        assert_eq!(attestation.proof_term_hash, restored.proof_term_hash);
        assert_eq!(attestation.environment_hash, restored.environment_hash);
        assert_eq!(
            attestation.verification_timestamp_ns,
            restored.verification_timestamp_ns
        );
        assert_eq!(attestation.verifier_version, restored.verifier_version);
        assert_eq!(attestation.reduction_steps, restored.reduction_steps);
        assert_eq!(attestation.cache_hit_rate_bps, restored.cache_hit_rate_bps);
    }

    #[test]
    fn test_witness_log_append() {
        let mut log = WitnessLog::new();
        let attestation = AttestationBuilder::new().build();

        let entry = log.append(attestation, ProofTier::Reflex).unwrap();
        assert_eq!(entry.index, 0);
        assert_eq!(entry.tier, ProofTier::Reflex);
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn test_witness_log_get() {
        let mut log = WitnessLog::new();
        let attestation = AttestationBuilder::new().build();

        log.append(attestation, ProofTier::Standard).unwrap();

        let entry = log.get(0).unwrap();
        assert_eq!(entry.index, 0);
        assert_eq!(entry.tier, ProofTier::Standard);
    }

    #[test]
    fn test_witness_log_latest() {
        let mut log = WitnessLog::new();

        // Empty log
        assert!(log.latest().is_none());

        let attestation = AttestationBuilder::new().build();
        log.append(attestation, ProofTier::Reflex).unwrap();

        let latest = log.latest().unwrap();
        assert_eq!(latest.index, 0);

        log.append(attestation, ProofTier::Deep).unwrap();
        let latest = log.latest().unwrap();
        assert_eq!(latest.index, 1);
        assert_eq!(latest.tier, ProofTier::Deep);
    }

    #[test]
    fn test_witness_log_circular() {
        let mut log = WitnessLog::new();
        let attestation = AttestationBuilder::new().build();

        // Fill beyond capacity
        for _ in 0..300 {
            log.append(attestation, ProofTier::Reflex).unwrap();
        }

        // Should have wrapped around
        assert_eq!(log.len(), MAX_WITNESS_LOG_ENTRIES);
        assert_eq!(log.total_recorded(), 300);
    }

    #[test]
    fn test_witness_log_clear() {
        let mut log = WitnessLog::new();
        let attestation = AttestationBuilder::new().build();

        log.append(attestation, ProofTier::Reflex).unwrap();
        log.clear();

        assert!(log.is_empty());
        assert!(log.latest().is_none());
    }

    #[test]
    fn test_create_and_log_attestation() {
        let mut log = WitnessLog::new();
        let token = make_test_token();

        let entry = create_and_log_attestation(&token, 5000, &mut log).unwrap();
        assert_eq!(entry.tier, ProofTier::Standard);
        assert_eq!(log.len(), 1);
    }
}
