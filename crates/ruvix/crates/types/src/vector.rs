//! Vector store types.
//!
//! Vector stores are kernel-resident objects containing HNSW-indexed vectors.
//! Unlike conventional databases where vectors are userspace constructs,
//! RuVix makes vector stores kernel objects with the same protection as
//! capability tables.

use crate::handle::Handle;

/// Handle to a kernel-resident vector store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct VectorStoreHandle(pub Handle);

impl VectorStoreHandle {
    /// Creates a new vector store handle.
    #[inline]
    #[must_use]
    pub const fn new(id: u32, generation: u32) -> Self {
        Self(Handle::new(id, generation))
    }

    /// Creates a null (invalid) vector store handle.
    #[inline]
    #[must_use]
    pub const fn null() -> Self {
        Self(Handle::null())
    }

    /// Checks if this handle is null.
    #[inline]
    #[must_use]
    pub const fn is_null(&self) -> bool {
        self.0.is_null()
    }

    /// Returns the raw handle.
    #[inline]
    #[must_use]
    pub const fn raw(&self) -> Handle {
        self.0
    }
}

impl Default for VectorStoreHandle {
    fn default() -> Self {
        Self::null()
    }
}

/// Key for addressing vectors in a vector store.
///
/// Vector keys are 64-bit identifiers unique within a store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct VectorKey(pub u64);

impl VectorKey {
    /// Creates a new vector key.
    #[inline]
    #[must_use]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the raw key value.
    #[inline]
    #[must_use]
    pub const fn raw(&self) -> u64 {
        self.0
    }
}

impl From<u64> for VectorKey {
    fn from(id: u64) -> Self {
        Self(id)
    }
}

impl From<VectorKey> for u64 {
    fn from(key: VectorKey) -> Self {
        key.0
    }
}

/// Coherence metadata associated with each vector.
///
/// Every vector in a kernel vector store carries coherence metadata
/// enabling the scheduler and proof engine to make informed decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct CoherenceMeta {
    /// Coherence score (0-10000 representing 0.0000-1.0000).
    /// Higher scores indicate stronger structural consistency.
    pub coherence_score: u16,

    /// Epoch of the last mutation.
    /// Incremented on each vector_put_proved.
    pub mutation_epoch: u64,

    /// Hash of the proof attestation that authorized the last mutation.
    pub proof_attestation_hash: [u8; 32],

    /// Timestamp of last access (nanoseconds since boot).
    pub last_access_ns: u64,

    /// Number of times this vector has been read.
    pub access_count: u32,
}

impl CoherenceMeta {
    /// Creates new coherence metadata.
    #[inline]
    #[must_use]
    pub const fn new(
        coherence_score: u16,
        mutation_epoch: u64,
        proof_attestation_hash: [u8; 32],
    ) -> Self {
        Self {
            coherence_score,
            mutation_epoch,
            proof_attestation_hash,
            last_access_ns: 0,
            access_count: 0,
        }
    }

    /// Returns the coherence score as a float (0.0-1.0).
    #[inline]
    #[must_use]
    pub fn coherence_as_f32(&self) -> f32 {
        self.coherence_score as f32 / 10000.0
    }

    /// Creates coherence metadata from a float score.
    #[inline]
    #[must_use]
    pub fn with_coherence_f32(coherence: f32, mutation_epoch: u64) -> Self {
        let score = (coherence.clamp(0.0, 1.0) * 10000.0) as u16;
        Self {
            coherence_score: score,
            mutation_epoch,
            proof_attestation_hash: [0; 32],
            last_access_ns: 0,
            access_count: 0,
        }
    }
}

impl Default for CoherenceMeta {
    fn default() -> Self {
        Self {
            coherence_score: 10000, // 1.0 = fully coherent
            mutation_epoch: 0,
            proof_attestation_hash: [0; 32],
            last_access_ns: 0,
            access_count: 0,
        }
    }
}

/// Configuration for creating a kernel vector store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct VectorStoreConfig {
    /// Number of dimensions per vector.
    pub dimensions: u32,

    /// Maximum number of vectors the store can hold.
    pub capacity: u32,

    /// HNSW M parameter (number of bidirectional links per node).
    pub hnsw_m: u16,

    /// HNSW ef_construction parameter.
    pub hnsw_ef_construction: u16,

    /// Whether to use quantization (reduces memory, slightly reduces accuracy).
    pub use_quantization: bool,
}

impl VectorStoreConfig {
    /// Creates a new vector store configuration.
    #[inline]
    #[must_use]
    pub const fn new(dimensions: u32, capacity: u32) -> Self {
        Self {
            dimensions,
            capacity,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            use_quantization: false,
        }
    }

    /// Sets HNSW parameters.
    #[inline]
    #[must_use]
    pub const fn with_hnsw(mut self, m: u16, ef_construction: u16) -> Self {
        self.hnsw_m = m;
        self.hnsw_ef_construction = ef_construction;
        self
    }

    /// Enables quantization.
    #[inline]
    #[must_use]
    pub const fn with_quantization(mut self) -> Self {
        self.use_quantization = true;
        self
    }
}

impl Default for VectorStoreConfig {
    fn default() -> Self {
        Self::new(768, 10000) // Default: 768-dim embeddings, 10K capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_store_handle() {
        let h = VectorStoreHandle::new(42, 1);
        assert!(!h.is_null());
        assert_eq!(h.raw().id, 42);
    }

    #[test]
    fn test_vector_key() {
        let key = VectorKey::new(12345);
        assert_eq!(key.raw(), 12345);
    }

    #[test]
    fn test_coherence_meta_score() {
        let meta = CoherenceMeta::with_coherence_f32(0.95, 1);
        assert!((meta.coherence_as_f32() - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_vector_store_config() {
        let config = VectorStoreConfig::new(384, 5000)
            .with_hnsw(32, 400)
            .with_quantization();

        assert_eq!(config.dimensions, 384);
        assert_eq!(config.capacity, 5000);
        assert_eq!(config.hnsw_m, 32);
        assert!(config.use_quantization);
    }
}
