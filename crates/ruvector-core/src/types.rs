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
    /// Turbo4 4-bit Lloyd-Max quantization (ADR-296). Unlike the legacy
    /// variants above, this one **is applied**: with an HNSW config the index
    /// stores only packed 4-bit codes (`D/2 + 8` bytes per vector, ~7.9x
    /// payload compression at 1536-D) and scores directly on them —
    /// symmetric during graph construction, asymmetric int8 during
    /// traversal, exact f32 rescoring of the top `k * rescore_multiplier`
    /// candidates. Requires an even dimension and Euclidean/Cosine/DotProduct
    /// metric (Manhattan is rejected).
    Turbo4 {
        /// Seed of the deterministic randomized rotation. Part of the
        /// persisted index contract: codes encoded under one seed are
        /// meaningless under another.
        #[serde(default = "default_turbo4_rotation_seed")]
        rotation_seed: u64,
        /// How many candidates (times `k`) the traversal fetches for the
        /// exact rescoring pass. The recall/latency dial; default 4.
        #[serde(default = "default_turbo4_rescore_multiplier")]
        rescore_multiplier: usize,
        /// Outcome-level policy governing adaptive escalation (ADR-297 §3/§9).
        #[serde(default)]
        policy: SearchPolicy,
        /// Which representation drives graph traversal (ADR-297 §2).
        #[serde(default)]
        search_quantization: SearchQuantization,
    },
}

/// Traversal representation for a Turbo4 index (ADR-297 §2): what the HNSW
/// walk scores against before Turbo4 rescoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SearchQuantization {
    /// Score traversal directly on Turbo4 codes (int8 query × nibbles).
    #[default]
    Turbo4Direct,
    /// Score traversal on 1-bit sign codes (RaBitQ-style, pure popcount —
    /// ~4× less memory traffic), rescore candidates with Turbo4. Stores
    /// both planes (~5 bits/dim total).
    RaBitQ1,
}

/// Outcome-level search policy (ADR-297 §9). Users pick a goal; the engine
/// maps it to escalation thresholds, rescore pools, and verification tiers —
/// no quantization knowledge required.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SearchPolicy {
    /// Maximum retrieval quality: aggressive escalation on uncertain queries.
    Quality,
    /// Default trade-off: escalate only when the result margin is unstable.
    #[default]
    Balanced,
    /// Minimum memory/latency: never escalate beyond the base quantized pass.
    MaxCompression,
}

/// Default rotation seed for [`QuantizationConfig::Turbo4`].
pub fn default_turbo4_rotation_seed() -> u64 {
    42
}

/// Default rescore multiplier for [`QuantizationConfig::Turbo4`].
pub fn default_turbo4_rescore_multiplier() -> usize {
    4
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
