//! Core types and data structures

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique identifier for vectors
pub type VectorId = String;

/// Distance metric for similarity calculation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Euclidean (L2) distance
    Euclidean,
    /// Cosine similarity (converted to distance)
    Cosine,
    /// Dot product (converted to distance for maximization)
    DotProduct,
    /// Manhattan (L1) distance
    Manhattan,
}

/// Vector entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorEntry {
    /// Optional ID (auto-generated if not provided)
    pub id: Option<VectorId>,
    /// Vector data
    pub vector: Vec<f32>,
    /// Optional metadata
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Search query parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    /// Query vector
    pub vector: Vec<f32>,
    /// Number of results to return (top-k)
    pub k: usize,
    /// Optional metadata filters
    pub filter: Option<HashMap<String, serde_json::Value>>,
    /// Optional ef_search parameter for HNSW (overrides default)
    pub ef_search: Option<usize>,
}

/// Search result with similarity score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Vector ID
    pub id: VectorId,
    /// Distance/similarity score (lower is better for distance metrics)
    pub score: f32,
    /// Vector data (optional)
    pub vector: Option<Vec<f32>>,
    /// Metadata (optional)
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Database configuration options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbOptions {
    /// Vector dimensions
    pub dimensions: usize,
    /// Distance metric
    pub distance_metric: DistanceMetric,
    /// Storage path
    pub storage_path: String,
    /// HNSW configuration
    pub hnsw_config: Option<HnswConfig>,
    /// Quantization configuration
    pub quantization: Option<QuantizationConfig>,
}

/// HNSW index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Number of connections per layer (M)
    pub m: usize,
    /// Size of dynamic candidate list during construction (efConstruction)
    pub ef_construction: usize,
    /// Size of dynamic candidate list during search (efSearch)
    pub ef_search: usize,
    /// Maximum number of elements
    pub max_elements: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 100,
            ef_search: 100,
            // 1M is a reasonable default that avoids excessive upfront memory
            // allocation while still being suitable for production workloads.
            // Callers building large indexes should set this explicitly.
            max_elements: 1_000_000,
        }
    }
}

/// Quantization configuration.
///
/// NOTE (issue #563): these variants are accepted, persisted, and restored, but
/// quantization is **not yet applied** to the index/storage — vectors are stored
/// in full precision regardless. The compression figures below describe the
/// intended behavior once quantization is wired into the index; today they are a
/// target, not a guarantee. `VectorDB::new` logs a warning when a non-`None`
/// value is set so it is not silently ignored.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationConfig {
    /// No quantization (full precision)
    None,
    /// Scalar quantization to int8 (target: 4x compression — not yet applied)
    Scalar,
    /// Product quantization (not yet applied)
    Product {
        /// Number of subspaces
        subspaces: usize,
        /// Codebook size (typically 256)
        k: usize,
    },
    /// Binary quantization (target: 32x compression — not yet applied)
    Binary,
}

impl Default for DbOptions {
    fn default() -> Self {
        Self {
            dimensions: 384,
            distance_metric: DistanceMetric::Cosine,
            storage_path: "./ruvector.db".to_string(),
            hnsw_config: Some(HnswConfig::default()),
            // Quantization is not yet applied to the index/storage (issue #563),
            // so the default is `None` to avoid advertising a compression that
            // does not happen. Set this explicitly once quantization is wired in.
            quantization: None,
        }
    }
}
