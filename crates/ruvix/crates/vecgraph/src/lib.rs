//! Kernel-Resident Vector and Graph Stores for RuVix Cognition Kernel.
//!
//! This crate implements the Vector/Graph Kernel Objects from ADR-087 Section 4.3.
//! Unlike conventional kernels where all data structures are userspace constructs,
//! RuVix makes vector stores and graph stores kernel-resident objects.
//!
//! # Design Principles
//!
//! - Vector data lives in kernel-managed slab regions with capability protection
//! - HNSW index nodes are slab-allocated (fixed-size slots, zero allocator overhead)
//! - Coherence metadata is co-located with each vector
//! - All mutations are proof-gated (no proof, no mutation)
//! - Every successful mutation emits a witness attestation
//!
//! # Syscalls Implemented
//!
//! - `vector_get`: Read vector data and coherence metadata (capability-gated, no proof required)
//! - `vector_put_proved`: Write vector with proof verification (proof-gated, PROVE right required)
//! - `graph_apply_proved`: Apply graph mutation with proof verification (proof-gated)
//!
//! # Example
//!
//! ```rust,ignore
//! use ruvix_vecgraph::{KernelVectorStore, VectorStoreBuilder};
//! use ruvix_types::{VectorKey, ProofToken, CapRights};
//!
//! // Create a vector store with 768 dimensions and 10000 capacity
//! let store = VectorStoreBuilder::new(768, 10000)
//!     .with_proof_policy(ProofPolicy::standard())
//!     .build(backing)?;
//!
//! // Read a vector (no proof required)
//! let (data, meta) = store.vector_get(key, cap)?;
//!
//! // Write a vector (proof required)
//! let attestation = store.vector_put_proved(key, &data, proof, cap)?;
//! ```
//!
//! # Features
//!
//! - `std`: Enable standard library support (default)
//! - `stats`: Enable statistics collection
//! - `coherence`: Enable coherence scoring

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod coherence;
pub mod graph_store;
pub mod hnsw;
pub mod proof_policy;
pub mod simd_distance;
pub mod vector_store;
pub mod witness;

// Re-exports
pub use coherence::{CoherenceConfig, CoherenceTracker};
pub use graph_store::{GraphMutationResult, GraphStoreBuilder, KernelGraphStore, PartitionMeta};
pub use hnsw::{HnswConfig, HnswNode, HnswRegion};
pub use proof_policy::{NonceTracker, ProofPolicy, ProofVerifier};
pub use simd_distance::{
    cosine_similarity, dot_product, euclidean_distance_squared, l2_norm, SimdCapabilities,
};
pub use vector_store::{KernelVectorStore, VectorEntry, VectorStoreBuilder};
pub use witness::{WitnessEntry, WitnessLog};

/// Result type for vecgraph operations.
pub type Result<T> = core::result::Result<T, ruvix_types::KernelError>;

/// Statistics for vector store operations.
#[cfg(feature = "stats")]
#[derive(Debug, Clone, Default)]
pub struct VecGraphStats {
    /// Total vector reads.
    pub vector_reads: u64,
    /// Total vector writes.
    pub vector_writes: u64,
    /// Total vector writes rejected (proof failed).
    pub vector_writes_rejected: u64,
    /// Total graph mutations.
    pub graph_mutations: u64,
    /// Total graph mutations rejected (proof failed).
    pub graph_mutations_rejected: u64,
    /// Total witness entries emitted.
    pub witness_entries: u64,
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_module_compiles() {
        // Basic compilation test
        assert!(true);
    }
}
