//! Kernel-resident vector store implementation.
//!
//! Vector stores are kernel objects containing HNSW-indexed vectors.
//! All mutations are proof-gated via the `vector_put_proved` syscall.
//!
//! # Design (from ADR-087 Section 4.3)
//!
//! ```rust,ignore
//! pub struct KernelVectorStore {
//!     hnsw_region: RegionHandle,       // slab region for HNSW graph nodes
//!     data_region: RegionHandle,       // slab region for vector data
//!     witness_region: RegionHandle,    // append-only mutation witness log
//!     coherence_config: CoherenceConfig,
//!     proof_policy: ProofPolicy,
//!     dimensions: u32,
//!     capacity: u32,
//! }
//! ```

use crate::coherence::{CoherenceConfig, CoherenceTracker};
use crate::hnsw::{HnswConfig, HnswRegion};
use crate::proof_policy::{ProofPolicy, ProofVerifier};
use crate::witness::WitnessLog;
use crate::Result;

use ruvix_region::backing::MemoryBacking;
use ruvix_region::slab::{SlabAllocator, SlotHandle};
use ruvix_types::{
    CapRights, Capability, CoherenceMeta, KernelError, ProofAttestation, ProofToken, RegionHandle,
    VectorKey, VectorStoreHandle,
};

/// Size of vector entry header (key + coherence meta).
const VECTOR_ENTRY_HEADER_SIZE: usize = 8 + core::mem::size_of::<CoherenceMeta>();

/// A vector entry stored in the data region.
#[derive(Debug, Clone)]
pub struct VectorEntry {
    /// The vector key.
    pub key: VectorKey,

    /// Coherence metadata.
    pub coherence: CoherenceMeta,

    /// The vector data (f32 values as bytes).
    pub data: Vec<f32>,
}

impl VectorEntry {
    /// Creates a new vector entry.
    #[inline]
    #[must_use]
    pub fn new(key: VectorKey, data: Vec<f32>, coherence: CoherenceMeta) -> Self {
        Self {
            key,
            coherence,
            data,
        }
    }
}

/// Builder for creating kernel vector stores.
pub struct VectorStoreBuilder {
    dimensions: u32,
    capacity: u32,
    hnsw_config: HnswConfig,
    coherence_config: CoherenceConfig,
    proof_policy: ProofPolicy,
}

impl VectorStoreBuilder {
    /// Creates a new vector store builder.
    #[inline]
    #[must_use]
    pub fn new(dimensions: u32, capacity: u32) -> Self {
        Self {
            dimensions,
            capacity,
            hnsw_config: HnswConfig::default(),
            coherence_config: CoherenceConfig::default(),
            proof_policy: ProofPolicy::standard(),
        }
    }

    /// Sets the HNSW configuration.
    #[inline]
    #[must_use]
    pub fn with_hnsw_config(mut self, config: HnswConfig) -> Self {
        self.hnsw_config = config;
        self
    }

    /// Sets the coherence configuration.
    #[inline]
    #[must_use]
    pub fn with_coherence_config(mut self, config: CoherenceConfig) -> Self {
        self.coherence_config = config;
        self
    }

    /// Sets the proof policy.
    #[inline]
    #[must_use]
    pub fn with_proof_policy(mut self, policy: ProofPolicy) -> Self {
        self.proof_policy = policy;
        self
    }

    /// Builds the kernel vector store.
    ///
    /// # Type Parameters
    ///
    /// * `B` - The memory backing type
    ///
    /// # Arguments
    ///
    /// * `data_backing` - Memory backing for vector data slab
    /// * `hnsw_backing` - Memory backing for HNSW node slab
    /// * `witness_backing` - Memory backing for witness log
    /// * `data_handle` - Region handle for data region
    /// * `hnsw_handle` - Region handle for HNSW region
    /// * `witness_handle` - Region handle for witness region
    /// * `store_id` - Unique store identifier
    pub fn build<B: MemoryBacking>(
        self,
        data_backing: B,
        hnsw_backing: B,
        witness_backing: B,
        data_handle: RegionHandle,
        hnsw_handle: RegionHandle,
        witness_handle: RegionHandle,
        store_id: u32,
    ) -> Result<KernelVectorStore<B>> {
        KernelVectorStore::new(
            data_backing,
            hnsw_backing,
            witness_backing,
            data_handle,
            hnsw_handle,
            witness_handle,
            self.dimensions,
            self.capacity,
            self.hnsw_config,
            self.coherence_config,
            self.proof_policy,
            store_id,
        )
    }
}

/// Key-to-slot mapping for vector lookup.
struct KeyMap {
    /// Simple array-based mapping (for small stores).
    /// In production, use a hash map or B-tree in a slab region.
    entries: [(VectorKey, SlotHandle); 256],
    count: usize,
}

impl KeyMap {
    fn new() -> Self {
        Self {
            entries: [(VectorKey::new(0), SlotHandle::invalid()); 256],
            count: 0,
        }
    }

    fn get(&self, key: VectorKey) -> Option<SlotHandle> {
        for i in 0..self.count {
            if self.entries[i].0 == key {
                return Some(self.entries[i].1);
            }
        }
        None
    }

    fn insert(&mut self, key: VectorKey, handle: SlotHandle) -> Result<()> {
        // Check for existing key
        for i in 0..self.count {
            if self.entries[i].0 == key {
                self.entries[i].1 = handle;
                return Ok(());
            }
        }

        // Add new entry
        if self.count >= 256 {
            return Err(KernelError::LimitExceeded);
        }

        self.entries[self.count] = (key, handle);
        self.count += 1;
        Ok(())
    }

    fn remove(&mut self, key: VectorKey) -> Option<SlotHandle> {
        for i in 0..self.count {
            if self.entries[i].0 == key {
                let handle = self.entries[i].1;
                // Swap with last
                self.entries[i] = self.entries[self.count - 1];
                self.entries[self.count - 1] = (VectorKey::new(0), SlotHandle::invalid());
                self.count -= 1;
                return Some(handle);
            }
        }
        None
    }

    fn len(&self) -> usize {
        self.count
    }
}

/// Kernel-resident vector store.
///
/// Implements the `vector_get` and `vector_put_proved` syscall interfaces.
pub struct KernelVectorStore<B: MemoryBacking> {
    /// Slab region for vector data.
    data_slab: SlabAllocator<B>,

    /// HNSW index region.
    hnsw_region: HnswRegion<B>,

    /// Append-only witness log.
    witness_log: WitnessLog<B>,

    /// Region handles for capability checking.
    data_handle: RegionHandle,
    hnsw_handle: RegionHandle,

    /// Key to slot mapping.
    key_map: KeyMap,

    /// Coherence tracker.
    coherence_tracker: CoherenceTracker,

    /// Proof verifier.
    proof_verifier: ProofVerifier,

    /// Vector dimensions.
    dimensions: u32,

    /// Maximum capacity.
    capacity: u32,

    /// Store identifier.
    store_id: u32,

    /// Store handle.
    handle: VectorStoreHandle,
}

impl<B: MemoryBacking> KernelVectorStore<B> {
    /// Creates a new kernel vector store.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        data_backing: B,
        hnsw_backing: B,
        witness_backing: B,
        data_handle: RegionHandle,
        hnsw_handle: RegionHandle,
        witness_handle: RegionHandle,
        dimensions: u32,
        capacity: u32,
        hnsw_config: HnswConfig,
        coherence_config: CoherenceConfig,
        proof_policy: ProofPolicy,
        store_id: u32,
    ) -> Result<Self> {
        // Calculate slot size for vector data
        // Header (key + coherence) + vector data (dimensions * 4 bytes)
        let slot_size = VECTOR_ENTRY_HEADER_SIZE + (dimensions as usize) * 4;

        let data_slab = SlabAllocator::new(data_backing, slot_size, capacity as usize)?;
        let hnsw_region = HnswRegion::new(hnsw_backing, hnsw_config, capacity as usize)?;
        let witness_log = WitnessLog::new(
            witness_backing,
            capacity as usize * 2, // Allow for some updates per vector
            witness_handle,
            store_id,
        )?;

        Ok(Self {
            data_slab,
            hnsw_region,
            witness_log,
            data_handle,
            hnsw_handle,
            key_map: KeyMap::new(),
            coherence_tracker: CoherenceTracker::new(coherence_config),
            proof_verifier: ProofVerifier::new(proof_policy),
            dimensions,
            capacity,
            store_id,
            handle: VectorStoreHandle::new(store_id, 0),
        })
    }

    /// Returns the store handle.
    #[inline]
    #[must_use]
    pub const fn handle(&self) -> VectorStoreHandle {
        self.handle
    }

    /// Returns the number of vectors stored.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.key_map.len()
    }

    /// Returns true if the store is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.key_map.len() == 0
    }

    /// Returns the capacity.
    #[inline]
    #[must_use]
    pub const fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Returns the dimensions.
    #[inline]
    #[must_use]
    pub const fn dimensions(&self) -> u32 {
        self.dimensions
    }

    /// Implements the `vector_get` syscall.
    ///
    /// Returns the vector data and its coherence metadata.
    /// Requires READ right on the capability.
    ///
    /// # Arguments
    ///
    /// * `key` - The vector key
    /// * `capability` - Capability authorizing the read
    ///
    /// # Returns
    ///
    /// The vector data and coherence metadata.
    pub fn vector_get(
        &self,
        key: VectorKey,
        capability: &Capability,
    ) -> Result<(Vec<f32>, CoherenceMeta)> {
        // Check READ right
        if !capability.has_rights(CapRights::READ) {
            return Err(KernelError::InsufficientRights);
        }

        // Look up the key
        let slot_handle = self.key_map.get(key).ok_or(KernelError::NotFound)?;

        // Read the slot
        let slot_size = VECTOR_ENTRY_HEADER_SIZE + (self.dimensions as usize) * 4;
        let mut buf = vec![0u8; slot_size];
        self.data_slab.read(slot_handle, &mut buf)?;

        // Parse header
        let key_bytes: [u8; 8] = buf[0..8]
            .try_into()
            .map_err(|_| KernelError::InternalError)?;
        let stored_key = VectorKey::new(u64::from_le_bytes(key_bytes));

        if stored_key != key {
            return Err(KernelError::InternalError);
        }

        // Parse coherence metadata
        let coherence_bytes = &buf[8..8 + core::mem::size_of::<CoherenceMeta>()];
        let coherence = parse_coherence_meta(coherence_bytes)?;

        // Parse vector data
        let data_start = VECTOR_ENTRY_HEADER_SIZE;
        let mut data = Vec::with_capacity(self.dimensions as usize);
        for i in 0..self.dimensions as usize {
            let offset = data_start + i * 4;
            let bytes: [u8; 4] = buf[offset..offset + 4]
                .try_into()
                .map_err(|_| KernelError::InternalError)?;
            data.push(f32::from_le_bytes(bytes));
        }

        Ok((data, coherence))
    }

    /// Implements the `vector_put_proved` syscall.
    ///
    /// Writes a vector with proof verification.
    /// Requires PROVE right on the capability.
    ///
    /// # Arguments
    ///
    /// * `key` - The vector key
    /// * `data` - The vector data
    /// * `proof` - The proof token authorizing the mutation
    /// * `capability` - Capability authorizing the write
    /// * `current_time_ns` - Current time for proof verification
    ///
    /// # Returns
    ///
    /// A proof attestation on success.
    pub fn vector_put_proved(
        &mut self,
        key: VectorKey,
        data: &[f32],
        proof: &ProofToken,
        capability: &Capability,
        current_time_ns: u64,
    ) -> Result<ProofAttestation> {
        // Validate dimensions
        if data.len() != self.dimensions as usize {
            return Err(KernelError::InvalidArgument);
        }

        // Compute mutation hash
        let mutation_hash = compute_vector_mutation_hash(key, data);

        // Verify proof
        let attestation =
            self.proof_verifier
                .verify(proof, &mutation_hash, current_time_ns, capability)?;

        // Check if this is an insert or update
        let is_update = self.key_map.get(key).is_some();

        if is_update {
            // Update existing vector
            let slot_handle = self.key_map.get(key).unwrap();

            // Read old metadata for coherence tracking
            let slot_size = VECTOR_ENTRY_HEADER_SIZE + (self.dimensions as usize) * 4;
            let mut buf = vec![0u8; slot_size];
            self.data_slab.read(slot_handle, &mut buf)?;

            let old_coherence_bytes = &buf[8..8 + core::mem::size_of::<CoherenceMeta>()];
            let old_coherence = parse_coherence_meta(old_coherence_bytes)?;

            // Create new coherence metadata
            let new_coherence = self.coherence_tracker.on_entry_mutated(
                &old_coherence,
                self.coherence_tracker.config().initial_coherence,
                attestation.proof_term_hash,
            );

            // Write updated data
            self.write_vector_slot(slot_handle, key, data, &new_coherence)?;
        } else {
            // Insert new vector
            let slot_handle = self.data_slab.alloc()?;

            // Create coherence metadata
            let coherence = self
                .coherence_tracker
                .create_initial_meta(attestation.proof_term_hash);

            // Write data
            self.write_vector_slot(slot_handle, key, data, &coherence)?;

            // Update key map
            self.key_map.insert(key, slot_handle)?;

            // Allocate HNSW node
            let _hnsw_handle = self.hnsw_region.alloc_node(0, slot_handle)?;

            // Track coherence
            self.coherence_tracker
                .on_entry_added(coherence.coherence_score);
        }

        // Record in witness log
        self.witness_log
            .record_vector_mutation(key, attestation, current_time_ns)?;

        Ok(attestation)
    }

    /// Writes vector data to a slot.
    fn write_vector_slot(
        &mut self,
        slot_handle: SlotHandle,
        key: VectorKey,
        data: &[f32],
        coherence: &CoherenceMeta,
    ) -> Result<()> {
        let slot_size = VECTOR_ENTRY_HEADER_SIZE + (self.dimensions as usize) * 4;
        let mut buf = vec![0u8; slot_size];

        // Write key
        buf[0..8].copy_from_slice(&key.raw().to_le_bytes());

        // Write coherence metadata
        serialize_coherence_meta(
            coherence,
            &mut buf[8..8 + core::mem::size_of::<CoherenceMeta>()],
        );

        // Write vector data
        let data_start = VECTOR_ENTRY_HEADER_SIZE;
        for (i, &value) in data.iter().enumerate() {
            let offset = data_start + i * 4;
            buf[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }

        self.data_slab.write(slot_handle, &buf)?;
        Ok(())
    }

    /// Returns the proof policy.
    #[inline]
    #[must_use]
    pub const fn proof_policy(&self) -> &ProofPolicy {
        self.proof_verifier.policy()
    }

    /// Returns the coherence tracker.
    #[inline]
    #[must_use]
    pub const fn coherence_tracker(&self) -> &CoherenceTracker {
        &self.coherence_tracker
    }

    /// Returns witness log statistics.
    #[inline]
    #[must_use]
    pub fn witness_entry_count(&self) -> usize {
        self.witness_log.entry_count()
    }

    /// Returns the witness log fill ratio.
    #[inline]
    #[must_use]
    pub fn witness_fill_ratio(&self) -> f32 {
        self.witness_log.fill_ratio()
    }
}

/// Computes a hash of a vector mutation for proof verification.
fn compute_vector_mutation_hash(key: VectorKey, data: &[f32]) -> [u8; 32] {
    // Simple hash for demonstration
    // In production, use SHA-256 or similar
    let mut hash = [0u8; 32];

    // Include key
    let key_bytes = key.raw().to_le_bytes();
    hash[0..8].copy_from_slice(&key_bytes);

    // Include data hash (simple XOR for demonstration)
    for (i, &value) in data.iter().enumerate() {
        let bytes = value.to_le_bytes();
        let offset = (8 + (i * 4)) % 24; // Stay within remaining space
        for j in 0..4 {
            hash[offset + j] ^= bytes[j];
        }
    }

    hash
}

/// Parses coherence metadata from bytes.
fn parse_coherence_meta(bytes: &[u8]) -> Result<CoherenceMeta> {
    if bytes.len() < core::mem::size_of::<CoherenceMeta>() {
        return Err(KernelError::BufferTooSmall);
    }

    let coherence_score = u16::from_le_bytes([bytes[0], bytes[1]]);
    let mutation_epoch = u64::from_le_bytes(bytes[2..10].try_into().unwrap());

    let mut proof_attestation_hash = [0u8; 32];
    proof_attestation_hash.copy_from_slice(&bytes[10..42]);

    let last_access_ns = u64::from_le_bytes(bytes[42..50].try_into().unwrap());
    let access_count = u32::from_le_bytes(bytes[50..54].try_into().unwrap());

    Ok(CoherenceMeta {
        coherence_score,
        mutation_epoch,
        proof_attestation_hash,
        last_access_ns,
        access_count,
    })
}

/// Serializes coherence metadata to bytes.
fn serialize_coherence_meta(meta: &CoherenceMeta, buf: &mut [u8]) {
    buf[0..2].copy_from_slice(&meta.coherence_score.to_le_bytes());
    buf[2..10].copy_from_slice(&meta.mutation_epoch.to_le_bytes());
    buf[10..42].copy_from_slice(&meta.proof_attestation_hash);
    buf[42..50].copy_from_slice(&meta.last_access_ns.to_le_bytes());
    buf[50..54].copy_from_slice(&meta.access_count.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvix_region::backing::StaticBacking;
    use ruvix_types::{ObjectType, ProofPayload, ProofTier};

    fn create_test_capability() -> Capability {
        Capability::new(
            1,
            ObjectType::VectorStore,
            CapRights::READ | CapRights::WRITE | CapRights::PROVE,
            0,
            1,
        )
    }

    fn create_test_proof(
        data: &[f32],
        key: VectorKey,
        valid_until_ns: u64,
        nonce: u64,
    ) -> ProofToken {
        let mutation_hash = compute_vector_mutation_hash(key, data);
        ProofToken::new(
            mutation_hash,
            ProofTier::Standard,
            ProofPayload::Hash {
                hash: mutation_hash,
            },
            valid_until_ns,
            nonce,
        )
    }

    #[test]
    fn test_vector_store_creation() {
        let data_backing = StaticBacking::<16384>::new();
        let hnsw_backing = StaticBacking::<16384>::new();
        let witness_backing = StaticBacking::<16384>::new();

        let store = VectorStoreBuilder::new(4, 10) // Small dimensions and capacity for tests
            .build(
                data_backing,
                hnsw_backing,
                witness_backing,
                RegionHandle::new(1, 0),
                RegionHandle::new(2, 0),
                RegionHandle::new(3, 0),
                1,
            )
            .unwrap();

        assert_eq!(store.dimensions(), 4);
        assert_eq!(store.capacity(), 10);
        assert!(store.is_empty());
    }

    #[test]
    fn test_vector_put_and_get() {
        let data_backing = StaticBacking::<16384>::new();
        let hnsw_backing = StaticBacking::<16384>::new();
        let witness_backing = StaticBacking::<16384>::new();

        let mut store = VectorStoreBuilder::new(4, 10)
            .with_proof_policy(ProofPolicy::reflex())
            .build(
                data_backing,
                hnsw_backing,
                witness_backing,
                RegionHandle::new(1, 0),
                RegionHandle::new(2, 0),
                RegionHandle::new(3, 0),
                1,
            )
            .unwrap();

        let cap = create_test_capability();
        let key = VectorKey::new(42);
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let proof = create_test_proof(&data, key, 1_000_000_000, 1);

        // Put vector
        let attestation = store
            .vector_put_proved(key, &data, &proof, &cap, 500_000_000)
            .unwrap();

        assert_eq!(store.len(), 1);
        assert!(attestation.verification_timestamp_ns > 0);

        // Get vector
        let (retrieved_data, coherence) = store.vector_get(key, &cap).unwrap();

        assert_eq!(retrieved_data, data);
        assert_eq!(coherence.coherence_score, 10000); // Initial coherence
    }

    #[test]
    fn test_vector_put_proof_rejected() {
        let data_backing = StaticBacking::<16384>::new();
        let hnsw_backing = StaticBacking::<16384>::new();
        let witness_backing = StaticBacking::<16384>::new();

        let mut store = VectorStoreBuilder::new(4, 10)
            .build(
                data_backing,
                hnsw_backing,
                witness_backing,
                RegionHandle::new(1, 0),
                RegionHandle::new(2, 0),
                RegionHandle::new(3, 0),
                1,
            )
            .unwrap();

        let cap = create_test_capability();
        let key = VectorKey::new(42);
        let data = vec![1.0, 2.0, 3.0, 4.0];

        // Create proof with wrong hash
        let wrong_proof = ProofToken::new(
            [0u8; 32], // Wrong hash
            ProofTier::Standard,
            ProofPayload::Hash { hash: [0u8; 32] },
            1_000_000_000,
            1,
        );

        let result = store.vector_put_proved(key, &data, &wrong_proof, &cap, 500_000_000);

        assert_eq!(result, Err(KernelError::ProofRejected));
        assert!(store.is_empty()); // No change
    }

    #[test]
    fn test_vector_put_expired_proof() {
        let data_backing = StaticBacking::<16384>::new();
        let hnsw_backing = StaticBacking::<16384>::new();
        let witness_backing = StaticBacking::<16384>::new();

        let mut store = VectorStoreBuilder::new(4, 10)
            .build(
                data_backing,
                hnsw_backing,
                witness_backing,
                RegionHandle::new(1, 0),
                RegionHandle::new(2, 0),
                RegionHandle::new(3, 0),
                1,
            )
            .unwrap();

        let cap = create_test_capability();
        let key = VectorKey::new(42);
        let data = vec![1.0, 2.0, 3.0, 4.0];

        // Create expired proof
        let expired_proof = create_test_proof(&data, key, 100_000_000, 1);

        // Current time is after expiry
        let result = store.vector_put_proved(key, &data, &expired_proof, &cap, 500_000_000);

        assert_eq!(result, Err(KernelError::ProofRejected));
    }

    #[test]
    fn test_vector_put_nonce_reuse() {
        let data_backing = StaticBacking::<16384>::new();
        let hnsw_backing = StaticBacking::<16384>::new();
        let witness_backing = StaticBacking::<16384>::new();

        let mut store = VectorStoreBuilder::new(4, 10)
            .with_proof_policy(ProofPolicy::reflex())
            .build(
                data_backing,
                hnsw_backing,
                witness_backing,
                RegionHandle::new(1, 0),
                RegionHandle::new(2, 0),
                RegionHandle::new(3, 0),
                1,
            )
            .unwrap();

        let cap = create_test_capability();
        let key1 = VectorKey::new(1);
        let key2 = VectorKey::new(2);
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![5.0, 6.0, 7.0, 8.0];

        // First put succeeds
        let proof1 = create_test_proof(&data1, key1, 1_000_000_000, 42);
        store
            .vector_put_proved(key1, &data1, &proof1, &cap, 500_000_000)
            .unwrap();

        // Second put with same nonce fails
        let proof2 = create_test_proof(&data2, key2, 1_000_000_000, 42);
        let result = store.vector_put_proved(key2, &data2, &proof2, &cap, 500_000_001);

        assert_eq!(result, Err(KernelError::ProofRejected));
    }

    #[test]
    fn test_vector_get_not_found() {
        let data_backing = StaticBacking::<16384>::new();
        let hnsw_backing = StaticBacking::<16384>::new();
        let witness_backing = StaticBacking::<16384>::new();

        let store = VectorStoreBuilder::new(4, 10)
            .build(
                data_backing,
                hnsw_backing,
                witness_backing,
                RegionHandle::new(1, 0),
                RegionHandle::new(2, 0),
                RegionHandle::new(3, 0),
                1,
            )
            .unwrap();

        let cap = create_test_capability();
        let result = store.vector_get(VectorKey::new(999), &cap);

        assert_eq!(result, Err(KernelError::NotFound));
    }

    #[test]
    fn test_vector_get_insufficient_rights() {
        let data_backing = StaticBacking::<16384>::new();
        let hnsw_backing = StaticBacking::<16384>::new();
        let witness_backing = StaticBacking::<16384>::new();

        let store = VectorStoreBuilder::new(4, 10)
            .build(
                data_backing,
                hnsw_backing,
                witness_backing,
                RegionHandle::new(1, 0),
                RegionHandle::new(2, 0),
                RegionHandle::new(3, 0),
                1,
            )
            .unwrap();

        // Capability without READ right
        let cap = Capability::new(1, ObjectType::VectorStore, CapRights::WRITE, 0, 1);

        let result = store.vector_get(VectorKey::new(1), &cap);

        assert_eq!(result, Err(KernelError::InsufficientRights));
    }

    #[test]
    fn test_witness_log_recording() {
        let data_backing = StaticBacking::<16384>::new();
        let hnsw_backing = StaticBacking::<16384>::new();
        let witness_backing = StaticBacking::<16384>::new();

        let mut store = VectorStoreBuilder::new(4, 10)
            .with_proof_policy(ProofPolicy::reflex())
            .build(
                data_backing,
                hnsw_backing,
                witness_backing,
                RegionHandle::new(1, 0),
                RegionHandle::new(2, 0),
                RegionHandle::new(3, 0),
                1,
            )
            .unwrap();

        let cap = create_test_capability();

        // Perform multiple mutations
        for i in 0..5 {
            let key = VectorKey::new(i);
            let data = vec![i as f32; 4];
            let proof = create_test_proof(&data, key, 1_000_000_000, i);
            store
                .vector_put_proved(key, &data, &proof, &cap, 500_000_000)
                .unwrap();
        }

        assert_eq!(store.witness_entry_count(), 5);
    }
}
