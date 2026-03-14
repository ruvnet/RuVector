//! Witness log implementation for boot attestation.
//!
//! The witness log is an append-only region that records:
//! - Boot attestations (first entry)
//! - Proof attestations during runtime
//! - Capability mutations
//! - Other security-relevant events
//!
//! # Integrity
//!
//! Entries are hash-chained for integrity verification.
//! Each entry contains:
//! - Previous entry hash (32 bytes)
//! - Entry type (1 byte)
//! - Timestamp (8 bytes)
//! - Payload (variable)

use crate::attestation::BootAttestation;
use crate::manifest::WitnessLogPolicy;
use ruvix_types::{KernelError, ProofAttestation};
use sha2::{Sha256, Digest};

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// Witness log entry type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WitnessLogEntryType {
    /// Boot attestation (first entry).
    BootAttestation = 0,

    /// Proof attestation during runtime.
    ProofAttestation = 1,

    /// Capability grant.
    CapabilityGrant = 2,

    /// Capability revoke.
    CapabilityRevoke = 3,

    /// Component mount.
    ComponentMount = 4,

    /// Component unmount.
    ComponentUnmount = 5,

    /// Region creation.
    RegionCreate = 6,

    /// Region destruction.
    RegionDestroy = 7,

    /// Checkpoint marker.
    Checkpoint = 8,

    /// Custom application-defined entry.
    Custom = 255,
}

/// Witness log entry header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct WitnessLogEntryHeader {
    /// Hash of the previous entry (chain link).
    pub prev_hash: [u8; 32],

    /// Entry type.
    pub entry_type: WitnessLogEntryType,

    /// Entry sequence number.
    pub sequence: u64,

    /// Timestamp in nanoseconds since epoch.
    pub timestamp_ns: u64,

    /// Payload size in bytes.
    pub payload_size: u32,
}

impl WitnessLogEntryHeader {
    /// Size of the header in bytes.
    pub const SIZE: usize = 32 + 1 + 8 + 8 + 4; // 53 bytes

    /// Creates a new entry header.
    #[must_use]
    pub fn new(
        prev_hash: [u8; 32],
        entry_type: WitnessLogEntryType,
        sequence: u64,
        timestamp_ns: u64,
        payload_size: u32,
    ) -> Self {
        Self {
            prev_hash,
            entry_type,
            sequence,
            timestamp_ns,
            payload_size,
        }
    }

    /// Computes the hash of this header.
    #[must_use]
    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(&self.prev_hash);
        hasher.update(&[self.entry_type as u8]);
        hasher.update(&self.sequence.to_le_bytes());
        hasher.update(&self.timestamp_ns.to_le_bytes());
        hasher.update(&self.payload_size.to_le_bytes());

        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }
}

/// Complete witness log entry (header + payload).
#[derive(Debug, Clone)]
pub struct WitnessLogEntry {
    /// Entry header.
    pub header: WitnessLogEntryHeader,

    /// Entry payload.
    #[cfg(feature = "alloc")]
    pub payload: Vec<u8>,
    /// Entry payload (fixed-size no_std variant).
    #[cfg(not(feature = "alloc"))]
    pub payload: [u8; 256],
    /// Payload length in the fixed-size array.
    #[cfg(not(feature = "alloc"))]
    pub payload_len: usize,
}

impl WitnessLogEntry {
    /// Creates a new witness log entry.
    #[cfg(feature = "alloc")]
    pub fn new(header: WitnessLogEntryHeader, payload: Vec<u8>) -> Self {
        Self { header, payload }
    }

    /// Creates a new witness log entry (no_std).
    #[cfg(not(feature = "alloc"))]
    pub fn new(header: WitnessLogEntryHeader, payload_data: &[u8]) -> Self {
        let mut payload = [0u8; 256];
        let len = payload_data.len().min(256);
        payload[..len].copy_from_slice(&payload_data[..len]);

        Self {
            header,
            payload,
            payload_len: len,
        }
    }

    /// Computes the hash of the entire entry (header + payload).
    #[must_use]
    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(&self.header.prev_hash);
        hasher.update(&[self.header.entry_type as u8]);
        hasher.update(&self.header.sequence.to_le_bytes());
        hasher.update(&self.header.timestamp_ns.to_le_bytes());
        hasher.update(&self.header.payload_size.to_le_bytes());

        #[cfg(feature = "alloc")]
        hasher.update(&self.payload);
        #[cfg(not(feature = "alloc"))]
        hasher.update(&self.payload[..self.payload_len]);

        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }

    /// Returns the payload as a slice.
    #[must_use]
    pub fn payload(&self) -> &[u8] {
        #[cfg(feature = "alloc")]
        {
            &self.payload
        }
        #[cfg(not(feature = "alloc"))]
        {
            &self.payload[..self.payload_len]
        }
    }
}

/// Witness log configuration.
#[derive(Debug, Clone, Copy)]
pub struct WitnessLogConfig {
    /// Maximum entries before rotation.
    pub max_entries: u64,

    /// Maximum size in bytes.
    pub max_size_bytes: u64,

    /// Whether to hash-chain entries.
    pub hash_chain: bool,

    /// Retention period in seconds (0 = forever).
    pub retention_seconds: u64,
}

impl WitnessLogConfig {
    /// Default witness log configuration.
    pub const DEFAULT: Self = Self {
        max_entries: 1_000_000,
        max_size_bytes: 100 * 1024 * 1024, // 100 MiB
        hash_chain: true,
        retention_seconds: 0,
    };

    /// Creates a configuration from a witness log policy.
    #[must_use]
    pub fn from_policy(policy: &WitnessLogPolicy) -> Self {
        Self {
            max_entries: policy.max_entries,
            max_size_bytes: policy.max_size_bytes,
            hash_chain: policy.hash_chain,
            retention_seconds: policy.retention_seconds,
        }
    }
}

impl Default for WitnessLogConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Witness log for boot and runtime attestation.
///
/// The witness log is append-only and hash-chained for integrity.
#[derive(Debug)]
pub struct WitnessLog {
    /// Configuration.
    config: WitnessLogConfig,

    /// Current entry count.
    entry_count: u64,

    /// Current size in bytes.
    size_bytes: u64,

    /// Hash of the last entry (for chaining).
    last_hash: [u8; 32],

    /// Entries (in-memory for Phase A).
    #[cfg(feature = "alloc")]
    entries: Vec<WitnessLogEntry>,
}

impl WitnessLog {
    /// Creates a new witness log.
    #[must_use]
    pub fn new(config: WitnessLogConfig) -> Self {
        Self {
            config,
            entry_count: 0,
            size_bytes: 0,
            last_hash: [0u8; 32], // Genesis hash is all zeros
            #[cfg(feature = "alloc")]
            entries: Vec::new(),
        }
    }

    /// Returns the current entry count.
    #[inline]
    #[must_use]
    pub fn entry_count(&self) -> u64 {
        self.entry_count
    }

    /// Returns the current size in bytes.
    #[inline]
    #[must_use]
    pub fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    /// Returns the hash of the last entry.
    #[inline]
    #[must_use]
    pub fn last_hash(&self) -> [u8; 32] {
        self.last_hash
    }

    /// Appends a boot attestation entry.
    ///
    /// This should be the first entry in the witness log.
    pub fn append_boot_attestation(&mut self, attestation: &BootAttestation) -> Result<(), KernelError> {
        if self.entry_count != 0 {
            // Boot attestation must be first
            return Err(KernelError::NotPermitted);
        }

        let payload = attestation.to_bytes();
        self.append(WitnessLogEntryType::BootAttestation, &payload)
    }

    /// Appends a proof attestation entry.
    pub fn append_proof_attestation(&mut self, attestation: &ProofAttestation) -> Result<(), KernelError> {
        let payload = Self::serialize_proof_attestation(attestation);
        self.append(WitnessLogEntryType::ProofAttestation, &payload)
    }

    /// Appends a generic entry.
    pub fn append(&mut self, entry_type: WitnessLogEntryType, payload: &[u8]) -> Result<(), KernelError> {
        // Check limits
        if self.entry_count >= self.config.max_entries {
            return Err(KernelError::RegionFull);
        }

        let entry_size = WitnessLogEntryHeader::SIZE + payload.len();
        if self.size_bytes + entry_size as u64 > self.config.max_size_bytes {
            return Err(KernelError::RegionFull);
        }

        // Create header
        let header = WitnessLogEntryHeader::new(
            self.last_hash,
            entry_type,
            self.entry_count,
            Self::get_timestamp(),
            payload.len() as u32,
        );

        // Create entry
        #[cfg(feature = "alloc")]
        let entry = WitnessLogEntry::new(header, payload.to_vec());
        #[cfg(not(feature = "alloc"))]
        let entry = WitnessLogEntry::new(header, payload);

        // Update hash chain
        if self.config.hash_chain {
            self.last_hash = entry.hash();
        }

        // Store entry
        #[cfg(feature = "alloc")]
        self.entries.push(entry);

        self.entry_count += 1;
        self.size_bytes += entry_size as u64;

        Ok(())
    }

    /// Gets an entry by sequence number.
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn get_entry(&self, sequence: u64) -> Option<&WitnessLogEntry> {
        self.entries.get(sequence as usize)
    }

    /// Verifies the hash chain integrity.
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn verify_chain(&self) -> bool {
        if !self.config.hash_chain {
            return true;
        }

        let mut expected_prev = [0u8; 32]; // Genesis

        for entry in &self.entries {
            if entry.header.prev_hash != expected_prev {
                return false;
            }
            expected_prev = entry.hash();
        }

        expected_prev == self.last_hash
    }

    fn serialize_proof_attestation(attestation: &ProofAttestation) -> [u8; 82] {
        let mut bytes = [0u8; 82];

        bytes[0..32].copy_from_slice(&attestation.proof_term_hash);
        bytes[32..64].copy_from_slice(&attestation.environment_hash);
        bytes[64..72].copy_from_slice(&attestation.verification_timestamp_ns.to_le_bytes());
        bytes[72..76].copy_from_slice(&attestation.verifier_version.to_le_bytes());
        bytes[76..80].copy_from_slice(&attestation.reduction_steps.to_le_bytes());
        bytes[80..82].copy_from_slice(&attestation.cache_hit_rate_bps.to_le_bytes());

        bytes
    }

    fn get_timestamp() -> u64 {
        #[cfg(feature = "std")]
        {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0)
        }
        #[cfg(not(feature = "std"))]
        {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attestation::BootAttestation;

    #[test]
    fn test_witness_log_creation() {
        let config = WitnessLogConfig::default();
        let log = WitnessLog::new(config);

        assert_eq!(log.entry_count(), 0);
        assert_eq!(log.size_bytes(), 0);
        assert_eq!(log.last_hash(), [0u8; 32]);
    }

    #[test]
    fn test_boot_attestation_append() {
        let config = WitnessLogConfig::default();
        let mut log = WitnessLog::new(config);

        let attestation = BootAttestation::new(
            [1u8; 32],
            [2u8; 32],
            [3u8; 32],
            1234567890,
        );

        log.append_boot_attestation(&attestation).unwrap();

        assert_eq!(log.entry_count(), 1);
        assert!(log.size_bytes() > 0);
        assert_ne!(log.last_hash(), [0u8; 32]); // Hash should have changed
    }

    #[test]
    fn test_boot_attestation_must_be_first() {
        let config = WitnessLogConfig::default();
        let mut log = WitnessLog::new(config);

        // Add a regular entry first
        log.append(WitnessLogEntryType::Custom, b"test").unwrap();

        // Boot attestation should fail
        let attestation = BootAttestation::new([0u8; 32], [0u8; 32], [0u8; 32], 0);
        let result = log.append_boot_attestation(&attestation);

        assert_eq!(result, Err(KernelError::NotPermitted));
    }

    #[test]
    fn test_witness_log_limit() {
        let config = WitnessLogConfig {
            max_entries: 2,
            max_size_bytes: 1024 * 1024,
            hash_chain: true,
            retention_seconds: 0,
        };
        let mut log = WitnessLog::new(config);

        log.append(WitnessLogEntryType::Custom, b"entry1").unwrap();
        log.append(WitnessLogEntryType::Custom, b"entry2").unwrap();

        // Third entry should fail
        let result = log.append(WitnessLogEntryType::Custom, b"entry3");
        assert_eq!(result, Err(KernelError::RegionFull));
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_hash_chain_verification() {
        let config = WitnessLogConfig::default();
        let mut log = WitnessLog::new(config);

        log.append(WitnessLogEntryType::Custom, b"entry1").unwrap();
        log.append(WitnessLogEntryType::Custom, b"entry2").unwrap();
        log.append(WitnessLogEntryType::Custom, b"entry3").unwrap();

        assert!(log.verify_chain());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_get_entry() {
        let config = WitnessLogConfig::default();
        let mut log = WitnessLog::new(config);

        log.append(WitnessLogEntryType::Custom, b"hello").unwrap();

        let entry = log.get_entry(0).unwrap();
        assert_eq!(entry.header.entry_type, WitnessLogEntryType::Custom);
        assert_eq!(entry.header.sequence, 0);
        assert_eq!(entry.payload(), b"hello");
    }

    #[test]
    fn test_entry_header_hash() {
        let header = WitnessLogEntryHeader::new(
            [1u8; 32],
            WitnessLogEntryType::Custom,
            42,
            1234567890,
            100,
        );

        let hash1 = header.hash();
        let hash2 = header.hash();

        // Hash should be deterministic
        assert_eq!(hash1, hash2);
        assert_ne!(hash1, [0u8; 32]); // Non-zero
    }
}
