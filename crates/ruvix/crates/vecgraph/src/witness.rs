//! Witness log for kernel vector/graph mutations.
//!
//! Every successful proof-gated mutation emits a witness record to the kernel's
//! append-only log. This provides:
//!
//! - Complete audit trail for all mutations
//! - Foundation for deterministic replay
//! - Verification of attestation chains
//!
//! # Design (from ADR-087 Section 3.2)
//!
//! - Every successful syscall that mutates state emits a witness record
//! - The witness log is append-only (never truncated or modified)
//! - Witness entries contain the proof attestation and mutation metadata

use ruvix_region::backing::MemoryBacking;
use ruvix_region::AppendOnlyRegion;
use ruvix_types::{GraphMutation, KernelError, ProofAttestation, RegionHandle, VectorKey};

use crate::Result;

/// Entry type for the witness log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WitnessEntryType {
    /// Vector store mutation.
    VectorMutation = 1,
    /// Graph store mutation.
    GraphMutation = 2,
    /// Store creation.
    StoreCreation = 3,
    /// Store destruction.
    StoreDestruction = 4,
}

impl WitnessEntryType {
    /// Converts from a raw u8 value.
    #[inline]
    #[must_use]
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::VectorMutation),
            2 => Some(Self::GraphMutation),
            3 => Some(Self::StoreCreation),
            4 => Some(Self::StoreDestruction),
            _ => None,
        }
    }
}

/// A witness entry in the mutation log.
///
/// Fixed-size structure for efficient append and read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct WitnessEntry {
    /// Type of witness entry.
    pub entry_type: WitnessEntryType,

    /// Padding for alignment.
    _padding: [u8; 3],

    /// Store ID that was mutated.
    pub store_id: u32,

    /// Sequence number within this store.
    pub sequence: u64,

    /// The proof attestation (82 bytes in ADR-047, padded here).
    pub attestation: ProofAttestation,

    /// Hash of the previous witness entry (chain verification).
    pub prev_hash: [u8; 32],

    /// Timestamp in nanoseconds.
    pub timestamp_ns: u64,
}

impl WitnessEntry {
    /// Size of a witness entry in bytes.
    pub const SIZE: usize = core::mem::size_of::<Self>();

    /// Creates a new witness entry.
    #[inline]
    #[must_use]
    pub const fn new(
        entry_type: WitnessEntryType,
        store_id: u32,
        sequence: u64,
        attestation: ProofAttestation,
        prev_hash: [u8; 32],
        timestamp_ns: u64,
    ) -> Self {
        Self {
            entry_type,
            _padding: [0; 3],
            store_id,
            sequence,
            attestation,
            prev_hash,
            timestamp_ns,
        }
    }

    /// Computes a simple hash of this entry for chaining.
    ///
    /// In a real implementation, this would use SHA-256 or similar.
    #[must_use]
    pub fn compute_hash(&self) -> [u8; 32] {
        // Simple hash combining all fields
        // In production, use a proper cryptographic hash
        let mut hash = [0u8; 32];

        // Mix in entry data
        hash[0] = self.entry_type as u8;
        hash[1..5].copy_from_slice(&self.store_id.to_le_bytes());
        hash[5..13].copy_from_slice(&self.sequence.to_le_bytes());
        hash[13..21].copy_from_slice(&self.timestamp_ns.to_le_bytes());

        // XOR with attestation proof_term_hash
        for i in 0..32 {
            hash[i] ^= self.attestation.proof_term_hash[i];
        }

        // XOR with prev_hash
        for i in 0..32 {
            hash[i] ^= self.prev_hash[i];
        }

        hash
    }

    /// Serializes the entry to bytes.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        // SAFETY: WitnessEntry is repr(C) with no padding issues
        unsafe { core::mem::transmute_copy(self) }
    }

    /// Deserializes an entry from bytes.
    ///
    /// Returns `None` if the bytes are invalid.
    #[must_use]
    pub fn from_bytes(bytes: &[u8; Self::SIZE]) -> Option<Self> {
        // Validate entry type
        let entry_type = WitnessEntryType::from_u8(bytes[0])?;

        // SAFETY: We've validated the entry type
        let mut entry: Self = unsafe { core::mem::transmute_copy(bytes) };
        entry.entry_type = entry_type;
        Some(entry)
    }
}

/// Witness log for tracking mutations.
pub struct WitnessLog<B: MemoryBacking> {
    /// The append-only region backing the log.
    region: AppendOnlyRegion<B>,

    /// Hash of the last entry (for chain verification).
    last_hash: [u8; 32],

    /// Current sequence number.
    sequence: u64,

    /// Store ID this log tracks.
    store_id: u32,
}

impl<B: MemoryBacking> WitnessLog<B> {
    /// Creates a new witness log.
    ///
    /// # Arguments
    ///
    /// * `backing` - Memory backing for the region
    /// * `max_entries` - Maximum number of entries to store
    /// * `handle` - Region handle for capability checking
    /// * `store_id` - Store ID this log tracks
    pub fn new(
        backing: B,
        max_entries: usize,
        handle: RegionHandle,
        store_id: u32,
    ) -> Result<Self> {
        let max_size = max_entries
            .checked_mul(WitnessEntry::SIZE)
            .ok_or(KernelError::InvalidArgument)?;

        let region = AppendOnlyRegion::new(backing, max_size, handle)?;

        Ok(Self {
            region,
            last_hash: [0u8; 32],
            sequence: 0,
            store_id,
        })
    }

    /// Records a vector mutation in the witness log.
    pub fn record_vector_mutation(
        &mut self,
        _key: VectorKey,
        attestation: ProofAttestation,
        timestamp_ns: u64,
    ) -> Result<WitnessEntry> {
        let entry = WitnessEntry::new(
            WitnessEntryType::VectorMutation,
            self.store_id,
            self.sequence,
            attestation,
            self.last_hash,
            timestamp_ns,
        );

        self.append_entry(entry)
    }

    /// Records a graph mutation in the witness log.
    pub fn record_graph_mutation(
        &mut self,
        _mutation: &GraphMutation,
        attestation: ProofAttestation,
        timestamp_ns: u64,
    ) -> Result<WitnessEntry> {
        // Include mutation kind in sequence for disambiguation
        let entry = WitnessEntry::new(
            WitnessEntryType::GraphMutation,
            self.store_id,
            self.sequence,
            attestation,
            self.last_hash,
            timestamp_ns,
        );

        self.append_entry(entry)
    }

    /// Records store creation.
    pub fn record_store_creation(
        &mut self,
        attestation: ProofAttestation,
        timestamp_ns: u64,
    ) -> Result<WitnessEntry> {
        let entry = WitnessEntry::new(
            WitnessEntryType::StoreCreation,
            self.store_id,
            self.sequence,
            attestation,
            self.last_hash,
            timestamp_ns,
        );

        self.append_entry(entry)
    }

    /// Appends an entry to the log.
    fn append_entry(&mut self, entry: WitnessEntry) -> Result<WitnessEntry> {
        let bytes = entry.to_bytes();
        self.region.append(&bytes)?;

        // Update chain hash
        self.last_hash = entry.compute_hash();
        self.sequence = self.sequence.wrapping_add(1);

        Ok(entry)
    }

    /// Returns the number of entries in the log.
    #[inline]
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.region.len() / WitnessEntry::SIZE
    }

    /// Returns the current sequence number.
    #[inline]
    #[must_use]
    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    /// Returns the hash of the last entry.
    #[inline]
    #[must_use]
    pub const fn last_hash(&self) -> [u8; 32] {
        self.last_hash
    }

    /// Returns the region handle.
    #[inline]
    #[must_use]
    pub fn handle(&self) -> RegionHandle {
        self.region.handle()
    }

    /// Returns the remaining capacity in entries.
    #[inline]
    #[must_use]
    pub fn remaining_entries(&self) -> usize {
        self.region.remaining() / WitnessEntry::SIZE
    }

    /// Checks if the log is full.
    #[inline]
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.region.remaining() < WitnessEntry::SIZE
    }

    /// Returns the fill ratio (0.0 to 1.0).
    #[inline]
    #[must_use]
    pub fn fill_ratio(&self) -> f32 {
        self.region.fill_ratio()
    }

    /// Reads an entry by index.
    ///
    /// Returns `None` if the index is out of bounds.
    pub fn read_entry(&self, index: usize) -> Result<WitnessEntry> {
        let offset = index
            .checked_mul(WitnessEntry::SIZE)
            .ok_or(KernelError::InvalidArgument)?;

        let mut bytes = [0u8; WitnessEntry::SIZE];
        let read = self.region.read(offset, &mut bytes)?;

        if read < WitnessEntry::SIZE {
            return Err(KernelError::BufferTooSmall);
        }

        WitnessEntry::from_bytes(&bytes).ok_or(KernelError::InternalError)
    }

    /// Verifies the chain integrity from a starting index.
    ///
    /// Returns `true` if all entries chain correctly.
    pub fn verify_chain(&self, start_index: usize) -> Result<bool> {
        let count = self.entry_count();
        if start_index >= count {
            return Ok(true); // Empty range is valid
        }

        let mut prev_hash = if start_index == 0 {
            [0u8; 32]
        } else {
            let prev_entry = self.read_entry(start_index - 1)?;
            prev_entry.compute_hash()
        };

        for i in start_index..count {
            let entry = self.read_entry(i)?;

            if entry.prev_hash != prev_hash {
                return Ok(false);
            }

            prev_hash = entry.compute_hash();
        }

        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvix_region::backing::StaticBacking;

    fn create_test_attestation() -> ProofAttestation {
        ProofAttestation::new([1u8; 32], [2u8; 32], 1000, 0x00_01_00_00, 1, 0)
    }

    #[test]
    fn test_witness_entry_type() {
        assert_eq!(
            WitnessEntryType::from_u8(1),
            Some(WitnessEntryType::VectorMutation)
        );
        assert_eq!(
            WitnessEntryType::from_u8(2),
            Some(WitnessEntryType::GraphMutation)
        );
        assert_eq!(WitnessEntryType::from_u8(100), None);
    }

    #[test]
    fn test_witness_entry_serialization() {
        let entry = WitnessEntry::new(
            WitnessEntryType::VectorMutation,
            42,
            1,
            create_test_attestation(),
            [3u8; 32],
            1000,
        );

        let bytes = entry.to_bytes();
        let restored = WitnessEntry::from_bytes(&bytes).unwrap();

        assert_eq!(restored.entry_type, entry.entry_type);
        assert_eq!(restored.store_id, entry.store_id);
        assert_eq!(restored.sequence, entry.sequence);
        assert_eq!(restored.prev_hash, entry.prev_hash);
        assert_eq!(restored.timestamp_ns, entry.timestamp_ns);
    }

    #[test]
    fn test_witness_log_append() {
        let backing = StaticBacking::<8192>::new();
        let handle = RegionHandle::new(1, 0);
        let mut log = WitnessLog::new(backing, 10, handle, 1).unwrap();

        assert_eq!(log.entry_count(), 0);

        let attestation = create_test_attestation();
        let entry = log
            .record_vector_mutation(VectorKey::new(1), attestation, 1000)
            .unwrap();

        assert_eq!(entry.entry_type, WitnessEntryType::VectorMutation);
        assert_eq!(entry.sequence, 0);
        assert_eq!(log.entry_count(), 1);
        assert_eq!(log.sequence(), 1);
    }

    #[test]
    fn test_witness_log_chain() {
        let backing = StaticBacking::<8192>::new();
        let handle = RegionHandle::new(1, 0);
        let mut log = WitnessLog::new(backing, 10, handle, 1).unwrap();

        let attestation = create_test_attestation();

        // First entry
        let entry1 = log
            .record_vector_mutation(VectorKey::new(1), attestation, 1000)
            .unwrap();
        assert_eq!(entry1.prev_hash, [0u8; 32]); // First entry has zero prev_hash

        // Second entry
        let entry2 = log
            .record_vector_mutation(VectorKey::new(2), attestation, 2000)
            .unwrap();
        assert_eq!(entry2.prev_hash, entry1.compute_hash());

        // Third entry
        let entry3 = log
            .record_vector_mutation(VectorKey::new(3), attestation, 3000)
            .unwrap();
        assert_eq!(entry3.prev_hash, entry2.compute_hash());
    }

    #[test]
    fn test_witness_log_read_entry() {
        let backing = StaticBacking::<8192>::new();
        let handle = RegionHandle::new(1, 0);
        let mut log = WitnessLog::new(backing, 10, handle, 1).unwrap();

        let attestation = create_test_attestation();

        log.record_vector_mutation(VectorKey::new(1), attestation, 1000)
            .unwrap();
        log.record_vector_mutation(VectorKey::new(2), attestation, 2000)
            .unwrap();

        let entry0 = log.read_entry(0).unwrap();
        assert_eq!(entry0.sequence, 0);

        let entry1 = log.read_entry(1).unwrap();
        assert_eq!(entry1.sequence, 1);
    }

    #[test]
    fn test_witness_log_verify_chain() {
        let backing = StaticBacking::<8192>::new();
        let handle = RegionHandle::new(1, 0);
        let mut log = WitnessLog::new(backing, 10, handle, 1).unwrap();

        let attestation = create_test_attestation();

        for i in 0..5 {
            log.record_vector_mutation(VectorKey::new(i), attestation, i as u64 * 1000)
                .unwrap();
        }

        assert!(log.verify_chain(0).unwrap());
        assert!(log.verify_chain(2).unwrap());
    }

    #[test]
    fn test_witness_log_full() {
        let backing = StaticBacking::<512>::new();
        let handle = RegionHandle::new(1, 0);
        let mut log = WitnessLog::new(backing, 2, handle, 1).unwrap();

        let attestation = create_test_attestation();

        // Fill the log
        log.record_vector_mutation(VectorKey::new(1), attestation, 1000)
            .unwrap();
        log.record_vector_mutation(VectorKey::new(2), attestation, 2000)
            .unwrap();

        // Should be full now
        assert!(log.is_full());

        // Next append should fail
        let result = log.record_vector_mutation(VectorKey::new(3), attestation, 3000);
        assert_eq!(result, Err(KernelError::RegionFull));
    }
}
