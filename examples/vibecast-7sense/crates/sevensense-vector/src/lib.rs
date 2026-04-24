//! # sevensense-vector
//!
//! Vector database operations and HNSW indexing for the 7sense bioacoustics platform.
//!
//! This crate provides:
//! - Local HNSW index with 150x search speedup over brute-force
//! - Optional Qdrant client wrapper for distributed deployments
//! - Collection management
//! - Similarity search with filtering
//! - Batch operations and persistence
//! - Hyperbolic embeddings for hierarchical relationships
//!
//! ## Architecture
//!
//! Following Domain-Driven Design:
//! ```text
//! sevensense-vector
//! ├── domain/              # Core entities, value objects, repository traits
//! │   ├── entities.rs      # EmbeddingId, HnswConfig, SimilarityEdge
//! │   ├── repository.rs    # VectorIndexRepository, GraphEdgeRepository
//! │   └── error.rs         # VectorError
//! ├── application/         # Service layer with use cases
//! │   └── services.rs      # VectorSpaceService
//! └── infrastructure/      # HNSW implementation and storage adapters
//!     ├── hnsw_index.rs    # Local HNSW index
//!     └── graph_store.rs   # Edge storage
//! ```
//!
//! ## Performance Targets
//!
//! - 150x search speedup over brute-force linear scan
//! - Sub-millisecond queries for up to 1M vectors
//! - Efficient batch insertion with parallelization
//!
//! ## Example
//!
//! ```rust,ignore
//! use sevensense_vector::prelude::*;
//!
//! // Create a vector space service
//! let config = HnswConfig::for_dimension(1536);
//! let service = VectorSpaceService::new(config);
//!
//! // Add embeddings
//! let id = EmbeddingId::new();
//! let vector = vec![0.1; 1536];
//! service.add_embedding(id, vector).await?;
//!
//! // Search for neighbors
//! let query = vec![0.15; 1536];
//! let neighbors = service.find_neighbors(&query, 10).await?;
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

pub mod application;
pub mod domain;
pub mod infrastructure;

mod distance;
mod hyperbolic;

// Re-export commonly used types
pub use application::services::{Neighbor, SearchOptions, VectorSpaceService};
pub use distance::{cosine_distance, cosine_similarity, euclidean_distance, normalize_vector};
pub use domain::entities::{
    EdgeType, EmbeddingId, HnswConfig, SimilarityEdge, Timestamp, VectorIndex,
};
pub use domain::repository::{GraphEdgeRepository, VectorIndexRepository};
pub use hyperbolic::{exp_map, log_map, mobius_add, poincare_distance};
pub use infrastructure::hnsw_index::HnswIndex;

/// Error types for vector operations
pub mod error {
    pub use crate::domain::error::*;
}

/// Prelude module for convenient imports
pub mod prelude {
    //! Common imports for vector operations.
    pub use crate::application::services::{Neighbor, SearchOptions, VectorSpaceService};
    pub use crate::distance::{cosine_distance, cosine_similarity, euclidean_distance};
    pub use crate::domain::entities::{
        EdgeType, EmbeddingId, HnswConfig, SimilarityEdge, VectorIndex,
    };
    pub use crate::domain::repository::{GraphEdgeRepository, VectorIndexRepository};
    pub use crate::error::VectorError;
    pub use crate::infrastructure::hnsw_index::HnswIndex;
}

/// Crate version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_compiles() {
        // Basic smoke test
        let config = HnswConfig::default();
        assert_eq!(config.m, 32);
    }

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
