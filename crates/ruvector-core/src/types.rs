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
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SearchQuery {
    /// Query vector
    pub vector: Vec<f32>,
    /// Number of results to return (top-k)
    pub k: usize,
    /// Optional metadata filters
    pub filter: Option<HashMap<String, serde_json::Value>>,
    /// Optional ef_search parameter for HNSW (overrides default)
    pub ef_search: Option<usize>,
    /// Controls whether search results are enriched with vector data and metadata
    /// from storage. This requires a REDB read transaction per result.
    ///
    /// - `None` (default): Auto — enrich only when a metadata filter is present,
    ///   since the filter requires metadata to evaluate. When no filter is set,
    ///   skip enrichment for faster searches (IDs + scores only).
    /// - `Some(true)`: Always enrich — results will include vector data and metadata
    ///   regardless of whether a filter is present.
    /// - `Some(false)`: Never enrich — results will only contain IDs and scores.
    ///   Note: metadata filters will NOT work when enrichment is disabled.
    #[serde(default)]
    pub enrich: Option<bool>,
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
            m: 32,
            ef_construction: 200,
            ef_search: 100,
            max_elements: 10_000_000,
        }
    }
}

/// Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationConfig {
    /// No quantization (full precision)
    None,
    /// Scalar quantization to int8 (4x compression)
    Scalar,
    /// Product quantization
    Product {
        /// Number of subspaces
        subspaces: usize,
        /// Codebook size (typically 256)
        k: usize,
    },
    /// Binary quantization (32x compression)
    Binary,
}

impl Default for DbOptions {
    fn default() -> Self {
        Self {
            dimensions: 384,
            distance_metric: DistanceMetric::Cosine,
            storage_path: "./ruvector.db".to_string(),
            hnsw_config: Some(HnswConfig::default()),
            quantization: Some(QuantizationConfig::Scalar),
        }
    }
}
