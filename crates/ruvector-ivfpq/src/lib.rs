//! IVF-PQ vector search index with HAKES filter-refine architecture.
//!
//! Pipeline: IVF coarse scan → PQ-ADC filter stage → exact-L2 refine stage.
//! Based on: Jégou et al. TPAMI 2011 (PQ), Faiss 2024 (IVF-PQ), HAKES VLDB 2025.

pub mod ivfpq;
pub mod kmeans;
pub mod pq;

pub use ivfpq::{IvfPqConfig, IvfPqIndex, SearchResult};
