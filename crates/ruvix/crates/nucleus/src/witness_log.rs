//! Witness log for attestation and deterministic replay.
//!
//! The witness log is an append-only log that records every mutation in the kernel.
//! It enables:
//!
//! - **Auditability**: External verifiers can inspect all state changes
//! - **Deterministic replay**: System can be restored from checkpoint + witness log
//! - **Forensics**: Post-mortem analysis of system behavior

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::{
    ProofAttestation, GraphHandle, ProofToken, Result, RvfMountHandle, VectorKey,
    VectorStoreHandle,
};
use ruvix_types::KernelError;

/// Maximum records in the witness log (for no_std).
#[cfg(not(feature = "alloc"))]
pub const MAX_WITNESS_RECORDS: usize = 4096;

/// Kind of witness record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum WitnessRecordKind {
    /// Kernel boot attestation.
    Boot = 0,
    /// RVF package mount.
    Mount = 1,
    /// Vector mutation.
    VectorMutation = 2,
    /// Graph mutation.
    GraphMutation = 3,
    /// Checkpoint created.
    Checkpoint = 4,
    /// Replay completed.
    ReplayComplete = 5,
}

impl WitnessRecordKind {
    /// Returns the kind as a string.
    #[inline]
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Boot => "boot",
            Self::Mount => "mount",
            Self::VectorMutation => "vector_mutation",
            Self::GraphMutation => "graph_mutation",
            Self::Checkpoint => "checkpoint",
            Self::ReplayComplete => "replay_complete",
        }
    }
}

/// A record in the witness log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WitnessRecord {
    /// Sequence number (monotonically increasing).
    pub sequence: u64,
    /// Kind of record.
    pub kind: WitnessRecordKind,
    /// Timestamp (nanoseconds since boot).
    pub timestamp_ns: u64,
    /// Hash of the mutation data.
    pub mutation_hash: [u8; 32],
    /// Hash of the proof attestation.
    pub attestation_hash: [u8; 32],
    /// Resource identifier (context-dependent).
    pub resource_id: u64,
}

impl WitnessRecord {
    /// Creates a new witness record.
    #[inline]
    #[must_use]
    pub const fn new(
        sequence: u64,
        kind: WitnessRecordKind,
        timestamp_ns: u64,
        mutation_hash: [u8; 32],
        attestation_hash: [u8; 32],
        resource_id: u64,
    ) -> Self {
        Self {
            sequence,
            kind,
            timestamp_ns,
            mutation_hash,
            attestation_hash,
            resource_id,
        }
    }

    /// Creates a boot record.
    #[inline]
    #[must_use]
    pub fn boot(sequence: u64, timestamp_ns: u64, kernel_hash: [u8; 32]) -> Self {
        Self::new(
            sequence,
            WitnessRecordKind::Boot,
            timestamp_ns,
            kernel_hash,
            [0u8; 32],
            0,
        )
    }

    /// Creates a mount record.
    #[inline]
    #[must_use]
    pub fn mount(
        sequence: u64,
        timestamp_ns: u64,
        package_hash: [u8; 32],
        attestation_hash: [u8; 32],
        mount: RvfMountHandle,
    ) -> Self {
        Self::new(
            sequence,
            WitnessRecordKind::Mount,
            timestamp_ns,
            package_hash,
            attestation_hash,
            mount.raw().id as u64,
        )
    }

    /// Creates a vector mutation record.
    #[inline]
    #[must_use]
    pub fn vector_mutation(
        sequence: u64,
        timestamp_ns: u64,
        data_hash: [u8; 32],
        attestation_hash: [u8; 32],
        store: VectorStoreHandle,
        key: VectorKey,
    ) -> Self {
        // Pack store and key into resource_id
        let resource_id = ((store.raw().id as u64) << 32) | (key.raw() & 0xFFFFFFFF);
        Self::new(
            sequence,
            WitnessRecordKind::VectorMutation,
            timestamp_ns,
            data_hash,
            attestation_hash,
            resource_id,
        )
    }

    /// Creates a graph mutation record.
    #[inline]
    #[must_use]
    pub fn graph_mutation(
        sequence: u64,
        timestamp_ns: u64,
        mutation_hash: [u8; 32],
        attestation_hash: [u8; 32],
        graph: GraphHandle,
    ) -> Self {
        Self::new(
            sequence,
            WitnessRecordKind::GraphMutation,
            timestamp_ns,
            mutation_hash,
            attestation_hash,
            graph.raw().id as u64,
        )
    }

    /// Creates a checkpoint record.
    #[inline]
    #[must_use]
    pub fn checkpoint(
        sequence: u64,
        timestamp_ns: u64,
        state_hash: [u8; 32],
        checkpoint_sequence: u64,
    ) -> Self {
        Self::new(
            sequence,
            WitnessRecordKind::Checkpoint,
            timestamp_ns,
            state_hash,
            [0u8; 32],
            checkpoint_sequence,
        )
    }

    /// Serializes the record to bytes.
    pub fn to_bytes(&self) -> [u8; 96] {
        let mut buf = [0u8; 96];
        buf[0..8].copy_from_slice(&self.sequence.to_le_bytes());
        buf[8] = self.kind as u8;
        buf[9..17].copy_from_slice(&self.timestamp_ns.to_le_bytes());
        buf[17..49].copy_from_slice(&self.mutation_hash);
        buf[49..81].copy_from_slice(&self.attestation_hash);
        buf[81..89].copy_from_slice(&self.resource_id.to_le_bytes());
        buf
    }

    /// Deserializes a record from bytes.
    pub fn from_bytes(buf: &[u8; 96]) -> Self {
        let sequence = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        let kind = match buf[8] {
            0 => WitnessRecordKind::Boot,
            1 => WitnessRecordKind::Mount,
            2 => WitnessRecordKind::VectorMutation,
            3 => WitnessRecordKind::GraphMutation,
            4 => WitnessRecordKind::Checkpoint,
            5 => WitnessRecordKind::ReplayComplete,
            _ => WitnessRecordKind::Boot, // Fallback
        };
        let timestamp_ns = u64::from_le_bytes(buf[9..17].try_into().unwrap());
        let mut mutation_hash = [0u8; 32];
        mutation_hash.copy_from_slice(&buf[17..49]);
        let mut attestation_hash = [0u8; 32];
        attestation_hash.copy_from_slice(&buf[49..81]);
        let resource_id = u64::from_le_bytes(buf[81..89].try_into().unwrap());

        Self {
            sequence,
            kind,
            timestamp_ns,
            mutation_hash,
            attestation_hash,
            resource_id,
        }
    }
}

impl Default for WitnessRecord {
    fn default() -> Self {
        Self {
            sequence: 0,
            kind: WitnessRecordKind::Boot,
            timestamp_ns: 0,
            mutation_hash: [0u8; 32],
            attestation_hash: [0u8; 32],
            resource_id: 0,
        }
    }
}

/// The append-only witness log.
///
/// Records every mutation in the kernel for auditability and replay.
pub struct WitnessLog {
    /// Log entries.
    #[cfg(feature = "alloc")]
    records: Vec<WitnessRecord>,
    #[cfg(not(feature = "alloc"))]
    records: [WitnessRecord; MAX_WITNESS_RECORDS],
    #[cfg(not(feature = "alloc"))]
    record_count: usize,

    /// Next sequence number.
    next_sequence: u64,

    /// Current time (nanoseconds since boot).
    current_time_ns: u64,

    /// Statistics.
    stats: WitnessLogStats,
}

/// Statistics about the witness log.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WitnessLogStats {
    /// Total records written.
    pub total_records: u64,
    /// Boot records.
    pub boot_records: u64,
    /// Mount records.
    pub mount_records: u64,
    /// Vector mutation records.
    pub vector_mutations: u64,
    /// Graph mutation records.
    pub graph_mutations: u64,
    /// Checkpoint records.
    pub checkpoints: u64,
}

impl WitnessLog {
    /// Creates a new empty witness log.
    #[must_use]
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "alloc")]
            records: Vec::new(),
            #[cfg(not(feature = "alloc"))]
            records: [WitnessRecord::default(); MAX_WITNESS_RECORDS],
            #[cfg(not(feature = "alloc"))]
            record_count: 0,
            next_sequence: 0,
            current_time_ns: 0,
            stats: WitnessLogStats::default(),
        }
    }

    /// Sets the current time.
    #[inline]
    pub fn set_current_time(&mut self, time_ns: u64) {
        self.current_time_ns = time_ns;
    }

    /// Returns the current sequence number.
    #[inline]
    #[must_use]
    pub const fn sequence(&self) -> u64 {
        self.next_sequence
    }

    /// Returns the number of records.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        #[cfg(feature = "alloc")]
        {
            self.records.len()
        }
        #[cfg(not(feature = "alloc"))]
        {
            self.record_count
        }
    }

    /// Returns true if the log is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the statistics.
    #[inline]
    #[must_use]
    pub const fn stats(&self) -> &WitnessLogStats {
        &self.stats
    }

    /// Appends a record to the log.
    pub fn append(&mut self, mut record: WitnessRecord) -> Result<u64> {
        record.sequence = self.next_sequence;
        record.timestamp_ns = self.current_time_ns;

        #[cfg(feature = "alloc")]
        {
            self.records.push(record);
        }
        #[cfg(not(feature = "alloc"))]
        {
            if self.record_count >= MAX_WITNESS_RECORDS {
                return Err(KernelError::LimitExceeded);
            }
            self.records[self.record_count] = record;
            self.record_count += 1;
        }

        // Update statistics
        self.stats.total_records += 1;
        match record.kind {
            WitnessRecordKind::Boot => self.stats.boot_records += 1,
            WitnessRecordKind::Mount => self.stats.mount_records += 1,
            WitnessRecordKind::VectorMutation => self.stats.vector_mutations += 1,
            WitnessRecordKind::GraphMutation => self.stats.graph_mutations += 1,
            WitnessRecordKind::Checkpoint => self.stats.checkpoints += 1,
            WitnessRecordKind::ReplayComplete => {}
        }

        let seq = self.next_sequence;
        self.next_sequence += 1;
        Ok(seq)
    }

    /// Records a boot attestation.
    pub fn record_boot(&mut self, kernel_hash: [u8; 32]) -> Result<u64> {
        let record = WitnessRecord::boot(0, self.current_time_ns, kernel_hash);
        self.append(record)
    }

    /// Records an RVF mount.
    pub fn record_mount(
        &mut self,
        package_hash: [u8; 32],
        attestation: &ProofAttestation,
        mount: RvfMountHandle,
    ) -> Result<u64> {
        let attestation_hash = hash_attestation(attestation);
        let record = WitnessRecord::mount(
            0,
            self.current_time_ns,
            package_hash,
            attestation_hash,
            mount,
        );
        self.append(record)
    }

    /// Records a vector mutation.
    pub fn record_vector_mutation(
        &mut self,
        data_hash: [u8; 32],
        attestation: &ProofAttestation,
        store: VectorStoreHandle,
        key: VectorKey,
    ) -> Result<u64> {
        let attestation_hash = hash_attestation(attestation);
        let record = WitnessRecord::vector_mutation(
            0,
            self.current_time_ns,
            data_hash,
            attestation_hash,
            store,
            key,
        );
        self.append(record)
    }

    /// Records a graph mutation.
    pub fn record_graph_mutation(
        &mut self,
        mutation_hash: [u8; 32],
        attestation: &ProofAttestation,
        graph: GraphHandle,
    ) -> Result<u64> {
        let attestation_hash = hash_attestation(attestation);
        let record = WitnessRecord::graph_mutation(
            0,
            self.current_time_ns,
            mutation_hash,
            attestation_hash,
            graph,
        );
        self.append(record)
    }

    /// Records a checkpoint.
    pub fn record_checkpoint(
        &mut self,
        state_hash: [u8; 32],
        checkpoint_sequence: u64,
    ) -> Result<u64> {
        let record = WitnessRecord::checkpoint(
            0,
            self.current_time_ns,
            state_hash,
            checkpoint_sequence,
        );
        self.append(record)
    }

    /// Gets a record by sequence number.
    pub fn get(&self, sequence: u64) -> Option<&WitnessRecord> {
        #[cfg(feature = "alloc")]
        {
            self.records.get(sequence as usize)
        }
        #[cfg(not(feature = "alloc"))]
        {
            if (sequence as usize) < self.record_count {
                Some(&self.records[sequence as usize])
            } else {
                None
            }
        }
    }

    /// Returns an iterator over all records.
    pub fn iter(&self) -> impl Iterator<Item = &WitnessRecord> {
        #[cfg(feature = "alloc")]
        {
            self.records.iter()
        }
        #[cfg(not(feature = "alloc"))]
        {
            self.records[..self.record_count].iter()
        }
    }

    /// Returns records matching a specific kind.
    pub fn filter_by_kind(&self, kind: WitnessRecordKind) -> impl Iterator<Item = &WitnessRecord> {
        self.iter().filter(move |r| r.kind == kind)
    }

    /// Serializes the entire log to bytes.
    #[cfg(feature = "alloc")]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(8 + self.records.len() * 96);
        buf.extend_from_slice(&(self.records.len() as u64).to_le_bytes());
        for record in &self.records {
            buf.extend_from_slice(&record.to_bytes());
        }
        buf
    }

    /// Deserializes a log from bytes.
    #[cfg(feature = "alloc")]
    pub fn from_bytes(buf: &[u8]) -> Result<Self> {
        if buf.len() < 8 {
            return Err(KernelError::InvalidArgument);
        }

        let count = u64::from_le_bytes(buf[0..8].try_into().unwrap()) as usize;
        let expected_len = 8 + count * 96;

        if buf.len() < expected_len {
            return Err(KernelError::InvalidArgument);
        }

        let mut records = Vec::with_capacity(count);
        for i in 0..count {
            let start = 8 + i * 96;
            let record_buf: [u8; 96] = buf[start..start + 96].try_into().unwrap();
            records.push(WitnessRecord::from_bytes(&record_buf));
        }

        let next_sequence = if records.is_empty() {
            0
        } else {
            records.last().unwrap().sequence + 1
        };

        Ok(Self {
            records,
            next_sequence,
            current_time_ns: 0,
            stats: WitnessLogStats::default(), // Stats would need to be recomputed
        })
    }

    /// Clears the log (for testing only).
    #[cfg(test)]
    pub fn clear(&mut self) {
        #[cfg(feature = "alloc")]
        {
            self.records.clear();
        }
        #[cfg(not(feature = "alloc"))]
        {
            self.record_count = 0;
        }
        self.next_sequence = 0;
        self.stats = WitnessLogStats::default();
    }
}

impl Default for WitnessLog {
    fn default() -> Self {
        Self::new()
    }
}

/// Computes a hash of a proof attestation.
fn hash_attestation(attestation: &ProofAttestation) -> [u8; 32] {
    // Simple FNV-1a hash for now
    // In production, use SHA-256
    let mut hash = 0xcbf29ce484222325u64;
    let prime = 0x100000001b3u64;

    for byte in &attestation.proof_term_hash {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(prime);
    }
    for byte in &attestation.environment_hash {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(prime);
    }
    for byte in &attestation.verification_timestamp_ns.to_le_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(prime);
    }

    let mut result = [0u8; 32];
    result[0..8].copy_from_slice(&hash.to_le_bytes());
    result[8..16].copy_from_slice(&hash.wrapping_mul(prime).to_le_bytes());
    result[16..24].copy_from_slice(&hash.wrapping_mul(prime).wrapping_mul(prime).to_le_bytes());
    result[24..32].copy_from_slice(&hash.wrapping_mul(prime).wrapping_mul(prime).wrapping_mul(prime).to_le_bytes());
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_witness_log_creation() {
        let log = WitnessLog::new();
        assert!(log.is_empty());
        assert_eq!(log.sequence(), 0);
    }

    #[test]
    fn test_record_boot() {
        let mut log = WitnessLog::new();
        let kernel_hash = [1u8; 32];

        let seq = log.record_boot(kernel_hash).unwrap();
        assert_eq!(seq, 0);
        assert_eq!(log.len(), 1);
        assert_eq!(log.stats().boot_records, 1);

        let record = log.get(0).unwrap();
        assert_eq!(record.kind, WitnessRecordKind::Boot);
        assert_eq!(record.mutation_hash, kernel_hash);
    }

    #[test]
    fn test_record_vector_mutation() {
        let mut log = WitnessLog::new();
        log.set_current_time(1_000_000);

        let data_hash = [2u8; 32];
        let attestation = ProofAttestation::default();
        let store = VectorStoreHandle::new(1, 0);
        let key = VectorKey::new(42);

        let seq = log
            .record_vector_mutation(data_hash, &attestation, store, key)
            .unwrap();
        assert_eq!(seq, 0);
        assert_eq!(log.stats().vector_mutations, 1);

        let record = log.get(0).unwrap();
        assert_eq!(record.kind, WitnessRecordKind::VectorMutation);
        assert_eq!(record.timestamp_ns, 1_000_000);
    }

    #[test]
    fn test_record_serialization() {
        let record = WitnessRecord::boot(42, 1_000_000, [3u8; 32]);
        let bytes = record.to_bytes();
        let restored = WitnessRecord::from_bytes(&bytes);

        assert_eq!(record.sequence, restored.sequence);
        assert_eq!(record.kind, restored.kind);
        assert_eq!(record.timestamp_ns, restored.timestamp_ns);
        assert_eq!(record.mutation_hash, restored.mutation_hash);
    }

    #[test]
    fn test_filter_by_kind() {
        let mut log = WitnessLog::new();

        // Add various records
        log.record_boot([1u8; 32]).unwrap();
        log.record_checkpoint([2u8; 32], 1).unwrap();
        log.record_checkpoint([3u8; 32], 2).unwrap();

        let checkpoints: Vec<_> = log.filter_by_kind(WitnessRecordKind::Checkpoint).collect();
        assert_eq!(checkpoints.len(), 2);

        let boots: Vec<_> = log.filter_by_kind(WitnessRecordKind::Boot).collect();
        assert_eq!(boots.len(), 1);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_log_serialization() {
        let mut log = WitnessLog::new();
        log.record_boot([1u8; 32]).unwrap();
        log.record_checkpoint([2u8; 32], 1).unwrap();

        let bytes = log.to_bytes();
        let restored = WitnessLog::from_bytes(&bytes).unwrap();

        assert_eq!(restored.len(), 2);
        assert_eq!(restored.get(0).unwrap().kind, WitnessRecordKind::Boot);
        assert_eq!(restored.get(1).unwrap().kind, WitnessRecordKind::Checkpoint);
    }

    #[test]
    fn test_sequential_sequences() {
        let mut log = WitnessLog::new();

        for i in 0..10 {
            let seq = log.record_boot([i as u8; 32]).unwrap();
            assert_eq!(seq, i);
        }

        assert_eq!(log.sequence(), 10);
    }
}
