//! Persistence for RvLite
//!
//! [`state`] holds the serde-serializable snapshot types (always available).
//! With the default `browser` feature, `IndexedDBStorage` persists them to
//! IndexedDB for:
//! - Vector database state
//! - Cypher graph state
//! - SPARQL triple store state

#[cfg(feature = "browser")]
pub mod indexeddb;
pub mod state;

#[cfg(feature = "rvf-backend")]
pub mod epoch;

#[cfg(feature = "rvf-backend")]
pub mod writer_lease;

#[cfg(feature = "rvf-backend")]
pub mod id_map;

#[cfg(feature = "browser")]
pub use indexeddb::IndexedDBStorage;
pub use state::{GraphState, RvLiteState, TripleStoreState, VectorState};
