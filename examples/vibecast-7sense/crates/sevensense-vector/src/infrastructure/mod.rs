//! Infrastructure layer for the Vector Space bounded context.
//!
//! Contains:
//! - HNSW index implementation
//! - Graph storage adapters
//! - Persistence implementations

pub mod graph_store;
pub mod hnsw_index;

pub use graph_store::InMemoryGraphStore;
pub use hnsw_index::HnswIndex;
