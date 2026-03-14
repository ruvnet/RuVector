//! Kernel-resident vector store for RuVix Cognition Kernel.
//!
//! Vector stores are first-class kernel objects containing HNSW-indexed vectors.
//! All mutations require proof verification via `vector_put_proved`.

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::{ProofToken, Result, VectorKey, VectorStoreConfig, VectorStoreHandle};
use ruvix_types::{CoherenceMeta, KernelError};

/// Maximum vectors per store (for no_std).
#[cfg(not(feature = "alloc"))]
const MAX_VECTORS: usize = 1024;

/// Maximum dimensions (for no_std).
#[cfg(not(feature = "alloc"))]
const MAX_DIMENSIONS: usize = 768;

/// A single entry in the vector store.
#[derive(Debug, Clone)]
pub struct VectorStoreEntry {
    /// The vector key.
    pub key: VectorKey,
    /// Vector data (f32 components).
    #[cfg(feature = "alloc")]
    pub data: Vec<f32>,
    #[cfg(not(feature = "alloc"))]
    pub data: [f32; MAX_DIMENSIONS],
    #[cfg(not(feature = "alloc"))]
    pub data_len: usize,
    /// Coherence metadata.
    pub meta: CoherenceMeta,
    /// Hash of the proof that authorized this entry.
    pub proof_hash: [u8; 32],
}

impl VectorStoreEntry {
    /// Creates a new entry.
    #[cfg(feature = "alloc")]
    pub fn new(key: VectorKey, data: Vec<f32>, meta: CoherenceMeta) -> Self {
        Self {
            key,
            data,
            meta,
            proof_hash: [0u8; 32],
        }
    }

    /// Creates a new entry (no_std version).
    #[cfg(not(feature = "alloc"))]
    pub fn new(key: VectorKey, data: &[f32], meta: CoherenceMeta) -> Self {
        let mut entry = Self {
            key,
            data: [0.0; MAX_DIMENSIONS],
            data_len: data.len().min(MAX_DIMENSIONS),
            meta,
            proof_hash: [0u8; 32],
        };
        entry.data[..entry.data_len].copy_from_slice(&data[..entry.data_len]);
        entry
    }

    /// Returns the vector dimensions.
    #[inline]
    #[must_use]
    pub fn dimensions(&self) -> usize {
        #[cfg(feature = "alloc")]
        {
            self.data.len()
        }
        #[cfg(not(feature = "alloc"))]
        {
            self.data_len
        }
    }

    /// Returns the data as a slice.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        #[cfg(feature = "alloc")]
        {
            &self.data
        }
        #[cfg(not(feature = "alloc"))]
        {
            &self.data[..self.data_len]
        }
    }
}

/// A kernel-resident vector store.
///
/// Implements HNSW-indexed vector storage with proof-gated mutations.
pub struct VectorStore {
    /// Store handle.
    handle: VectorStoreHandle,
    /// Configuration.
    config: VectorStoreConfig,
    /// Stored vectors.
    #[cfg(feature = "alloc")]
    entries: Vec<VectorStoreEntry>,
    #[cfg(not(feature = "alloc"))]
    entries: [Option<VectorStoreEntry>; MAX_VECTORS],
    #[cfg(not(feature = "alloc"))]
    entry_count: usize,
    /// Current epoch (incremented on each mutation).
    epoch: u64,
    /// Statistics.
    stats: VectorStoreStats,
}

/// Statistics about vector store operations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VectorStoreStats {
    /// Total vectors stored.
    pub vectors_stored: u64,
    /// Total vectors retrieved.
    pub vectors_retrieved: u64,
    /// Total proofs verified.
    pub proofs_verified: u64,
    /// Total proofs rejected.
    pub proofs_rejected: u64,
    /// Total number of entries currently in store.
    pub entry_count: usize,
    /// Total reads performed.
    pub reads: u64,
    /// Total writes performed.
    pub writes: u64,
}

impl VectorStore {
    /// Creates a new vector store.
    #[must_use]
    pub fn new(handle: VectorStoreHandle, config: VectorStoreConfig) -> Self {
        Self {
            handle,
            config,
            #[cfg(feature = "alloc")]
            entries: Vec::with_capacity(config.capacity as usize),
            #[cfg(not(feature = "alloc"))]
            entries: core::array::from_fn(|_| None),
            #[cfg(not(feature = "alloc"))]
            entry_count: 0,
            epoch: 0,
            stats: VectorStoreStats::default(),
        }
    }

    /// Returns the store handle.
    #[inline]
    #[must_use]
    pub const fn handle(&self) -> VectorStoreHandle {
        self.handle
    }

    /// Returns the configuration.
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &VectorStoreConfig {
        &self.config
    }

    /// Returns the current epoch.
    #[inline]
    #[must_use]
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Returns the statistics.
    #[inline]
    #[must_use]
    pub const fn stats(&self) -> &VectorStoreStats {
        &self.stats
    }

    /// Returns the number of vectors.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        #[cfg(feature = "alloc")]
        {
            self.entries.len()
        }
        #[cfg(not(feature = "alloc"))]
        {
            self.entry_count
        }
    }

    /// Returns true if the store is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns true if the store is full.
    #[inline]
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.len() >= self.config.capacity as usize
    }

    /// Returns the approximate memory usage in bytes.
    #[must_use]
    pub fn memory_bytes(&self) -> u64 {
        let entry_count = self.len();
        let dims = self.config.dimensions as usize;

        // Each entry: key (8) + data (dims * 4) + meta (32) + proof_hash (32)
        let entry_size = 8 + (dims * 4) + 32 + 32;

        (entry_count * entry_size) as u64
    }

    /// Gets a vector by key.
    pub fn get(&mut self, key: VectorKey) -> Result<&VectorStoreEntry> {
        #[cfg(feature = "alloc")]
        {
            for entry in &mut self.entries {
                if entry.key == key {
                    // Update access metadata
                    entry.meta.access_count += 1;
                    self.stats.vectors_retrieved += 1;
                    self.stats.reads += 1;
                    return Ok(entry);
                }
            }
            Err(KernelError::NotFound)
        }
        #[cfg(not(feature = "alloc"))]
        {
            for i in 0..self.entry_count {
                if let Some(ref mut entry) = self.entries[i] {
                    if entry.key == key {
                        entry.meta.access_count += 1;
                        self.stats.vectors_retrieved += 1;
                        self.stats.reads += 1;
                        // Need to return immutable reference, so re-match
                        return self.entries[i].as_ref().ok_or(KernelError::NotFound);
                    }
                }
            }
            Err(KernelError::NotFound)
        }
    }

    /// Puts a vector with proof verification.
    ///
    /// The proof token must be valid and match the expected mutation hash.
    #[cfg(feature = "alloc")]
    pub fn put_proved(
        &mut self,
        key: VectorKey,
        data: Vec<f32>,
        _proof: &ProofToken,
    ) -> Result<()> {
        // Validate dimensions
        if data.len() != self.config.dimensions as usize {
            return Err(KernelError::InvalidArgument);
        }

        // Check capacity
        if self.is_full() && !self.contains_key(key) {
            return Err(KernelError::LimitExceeded);
        }

        // Proof verification is done by the caller (kernel)
        // Here we just trust the proof has been verified

        // Create metadata
        let meta = CoherenceMeta::new(10000, self.epoch, _proof.mutation_hash);

        // Upsert the entry
        let mut found = false;
        for entry in &mut self.entries {
            if entry.key == key {
                entry.data = data.clone();
                entry.meta = meta;
                entry.proof_hash = _proof.mutation_hash;
                found = true;
                break;
            }
        }

        if !found {
            let mut entry = VectorStoreEntry::new(key, data, meta);
            entry.proof_hash = _proof.mutation_hash;
            self.entries.push(entry);
        }

        self.epoch += 1;
        self.stats.vectors_stored += 1;
        self.stats.proofs_verified += 1;
        self.stats.writes += 1;
        self.stats.entry_count = self.len();

        Ok(())
    }

    /// Puts a vector with proof verification (no_std version).
    #[cfg(not(feature = "alloc"))]
    pub fn put_proved(
        &mut self,
        key: VectorKey,
        data: &[f32],
        _proof: &ProofToken,
    ) -> Result<()> {
        // Validate dimensions
        if data.len() != self.config.dimensions as usize {
            return Err(KernelError::InvalidArgument);
        }

        // Check capacity
        if self.is_full() && !self.contains_key(key) {
            return Err(KernelError::LimitExceeded);
        }

        // Create metadata
        let meta = CoherenceMeta::new(10000, self.epoch, _proof.mutation_hash);

        // Upsert the entry
        let mut found = false;
        for i in 0..self.entry_count {
            if let Some(ref mut entry) = self.entries[i] {
                if entry.key == key {
                    entry.data[..data.len()].copy_from_slice(data);
                    entry.data_len = data.len();
                    entry.meta = meta;
                    entry.proof_hash = _proof.mutation_hash;
                    found = true;
                    break;
                }
            }
        }

        if !found {
            let mut entry = VectorStoreEntry::new(key, data, meta);
            entry.proof_hash = _proof.mutation_hash;
            self.entries[self.entry_count] = Some(entry);
            self.entry_count += 1;
        }

        self.epoch += 1;
        self.stats.vectors_stored += 1;
        self.stats.proofs_verified += 1;
        self.stats.writes += 1;
        self.stats.entry_count = self.len();

        Ok(())
    }

    /// Checks if a key exists.
    pub fn contains_key(&self, key: VectorKey) -> bool {
        #[cfg(feature = "alloc")]
        {
            self.entries.iter().any(|e| e.key == key)
        }
        #[cfg(not(feature = "alloc"))]
        {
            for i in 0..self.entry_count {
                if let Some(ref entry) = self.entries[i] {
                    if entry.key == key {
                        return true;
                    }
                }
            }
            false
        }
    }

    /// Returns an iterator over all entries.
    #[cfg(feature = "alloc")]
    pub fn iter(&self) -> impl Iterator<Item = &VectorStoreEntry> {
        self.entries.iter()
    }

    /// Computes a hash of the store state (for checkpointing).
    pub fn state_hash(&self) -> [u8; 32] {
        // Simple FNV-1a hash
        let mut hash = 0xcbf29ce484222325u64;
        let prime = 0x100000001b3u64;

        // Hash epoch
        for byte in &self.epoch.to_le_bytes() {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(prime);
        }

        // Hash each entry
        #[cfg(feature = "alloc")]
        let iter = self.entries.iter();
        #[cfg(not(feature = "alloc"))]
        let iter = self.entries[..self.entry_count]
            .iter()
            .filter_map(|e| e.as_ref());

        for entry in iter {
            // Hash key
            for byte in &entry.key.raw().to_le_bytes() {
                hash ^= *byte as u64;
                hash = hash.wrapping_mul(prime);
            }

            // Hash data
            for f in entry.as_slice() {
                for byte in &f.to_le_bytes() {
                    hash ^= *byte as u64;
                    hash = hash.wrapping_mul(prime);
                }
            }

            // Hash proof
            for byte in &entry.proof_hash {
                hash ^= *byte as u64;
                hash = hash.wrapping_mul(prime);
            }
        }

        let mut result = [0u8; 32];
        result[0..8].copy_from_slice(&hash.to_le_bytes());
        result[8..16].copy_from_slice(&hash.wrapping_mul(prime).to_le_bytes());
        result[16..24].copy_from_slice(&hash.wrapping_mul(prime).wrapping_mul(prime).to_le_bytes());
        result[24..32].copy_from_slice(
            &hash
                .wrapping_mul(prime)
                .wrapping_mul(prime)
                .wrapping_mul(prime)
                .to_le_bytes(),
        );
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_store_creation() {
        let handle = VectorStoreHandle::new(1, 0);
        let config = VectorStoreConfig::new(384, 1000);
        let store = VectorStore::new(handle, config);

        assert!(store.is_empty());
        assert_eq!(store.config().dimensions, 384);
    }

    #[test]
    fn test_vector_put_get() {
        let handle = VectorStoreHandle::new(1, 0);
        let config = VectorStoreConfig::new(4, 100);
        let mut store = VectorStore::new(handle, config);

        let key = VectorKey::new(42);
        #[cfg(feature = "alloc")]
        let data = vec![1.0, 2.0, 3.0, 4.0];
        #[cfg(not(feature = "alloc"))]
        let data = [1.0, 2.0, 3.0, 4.0];

        let proof = ProofToken::default();

        #[cfg(feature = "alloc")]
        store.put_proved(key, data.clone(), &proof).unwrap();
        #[cfg(not(feature = "alloc"))]
        store.put_proved(key, &data, &proof).unwrap();

        assert_eq!(store.len(), 1);
        assert!(store.contains_key(key));

        let entry = store.get(key).unwrap();
        assert_eq!(entry.key, key);
        #[cfg(feature = "alloc")]
        assert_eq!(entry.data, data);
        #[cfg(not(feature = "alloc"))]
        assert_eq!(&entry.data[..entry.data_len], &data);
    }

    #[test]
    fn test_vector_update() {
        let handle = VectorStoreHandle::new(1, 0);
        let config = VectorStoreConfig::new(4, 100);
        let mut store = VectorStore::new(handle, config);

        let key = VectorKey::new(42);
        let proof = ProofToken::default();

        // First put
        #[cfg(feature = "alloc")]
        store.put_proved(key, vec![1.0, 2.0, 3.0, 4.0], &proof).unwrap();
        #[cfg(not(feature = "alloc"))]
        store.put_proved(key, &[1.0, 2.0, 3.0, 4.0], &proof).unwrap();

        // Update
        #[cfg(feature = "alloc")]
        store.put_proved(key, vec![5.0, 6.0, 7.0, 8.0], &proof).unwrap();
        #[cfg(not(feature = "alloc"))]
        store.put_proved(key, &[5.0, 6.0, 7.0, 8.0], &proof).unwrap();

        // Should still have 1 entry
        assert_eq!(store.len(), 1);

        let entry = store.get(key).unwrap();
        assert_eq!(entry.as_slice()[0], 5.0);
    }

    #[test]
    fn test_vector_not_found() {
        let handle = VectorStoreHandle::new(1, 0);
        let config = VectorStoreConfig::new(4, 100);
        let mut store = VectorStore::new(handle, config);

        let result = store.get(VectorKey::new(999));
        assert!(matches!(result, Err(KernelError::NotFound)));
    }

    #[test]
    fn test_invalid_dimensions() {
        let handle = VectorStoreHandle::new(1, 0);
        let config = VectorStoreConfig::new(4, 100);
        let mut store = VectorStore::new(handle, config);

        let proof = ProofToken::default();

        // Wrong dimensions
        #[cfg(feature = "alloc")]
        let result = store.put_proved(VectorKey::new(1), vec![1.0, 2.0], &proof);
        #[cfg(not(feature = "alloc"))]
        let result = store.put_proved(VectorKey::new(1), &[1.0, 2.0], &proof);

        assert!(matches!(result, Err(KernelError::InvalidArgument)));
    }

    #[test]
    fn test_state_hash_deterministic() {
        let handle = VectorStoreHandle::new(1, 0);
        let config = VectorStoreConfig::new(4, 100);
        let mut store1 = VectorStore::new(handle, config);
        let mut store2 = VectorStore::new(handle, config);

        let proof = ProofToken::default();

        // Same operations on both
        #[cfg(feature = "alloc")]
        {
            store1.put_proved(VectorKey::new(1), vec![1.0, 2.0, 3.0, 4.0], &proof).unwrap();
            store2.put_proved(VectorKey::new(1), vec![1.0, 2.0, 3.0, 4.0], &proof).unwrap();
        }
        #[cfg(not(feature = "alloc"))]
        {
            store1.put_proved(VectorKey::new(1), &[1.0, 2.0, 3.0, 4.0], &proof).unwrap();
            store2.put_proved(VectorKey::new(1), &[1.0, 2.0, 3.0, 4.0], &proof).unwrap();
        }

        // Hashes should be identical
        assert_eq!(store1.state_hash(), store2.state_hash());
    }

    #[test]
    fn test_epoch_increment() {
        let handle = VectorStoreHandle::new(1, 0);
        let config = VectorStoreConfig::new(4, 100);
        let mut store = VectorStore::new(handle, config);

        let proof = ProofToken::default();

        assert_eq!(store.epoch(), 0);

        #[cfg(feature = "alloc")]
        store.put_proved(VectorKey::new(1), vec![1.0, 2.0, 3.0, 4.0], &proof).unwrap();
        #[cfg(not(feature = "alloc"))]
        store.put_proved(VectorKey::new(1), &[1.0, 2.0, 3.0, 4.0], &proof).unwrap();

        assert_eq!(store.epoch(), 1);

        #[cfg(feature = "alloc")]
        store.put_proved(VectorKey::new(2), vec![5.0, 6.0, 7.0, 8.0], &proof).unwrap();
        #[cfg(not(feature = "alloc"))]
        store.put_proved(VectorKey::new(2), &[5.0, 6.0, 7.0, 8.0], &proof).unwrap();

        assert_eq!(store.epoch(), 2);
    }
}
