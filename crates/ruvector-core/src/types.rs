//! Core types and data structures

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique identifier for vectors
pub type VectorId = String;

/// Distance metric for similarity calculation
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, bincode::Encode, bincode::Decode,
)]
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

/// Unified Quantum Vector type to replace raw f32 vectors
#[derive(Debug, Clone, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
pub enum QuantumVector {
    /// Full precision (only for in-flight/transfer, will be purged in storage)
    F32(Vec<f32>),
    /// 8-bit Quantized (Q8_0)
    Q8(Vec<i8>, f32), // data, scale
    /// 4-bit Normal Float (NF4)
    NF4 {
        data: Vec<u8>,
        scale: f32,
        orig_len: usize,
    },
    /// Binary (1-bit)
    Binary(Vec<u8>),
}

impl Default for QuantumVector {
    fn default() -> Self {
        QuantumVector::F32(Vec::new())
    }
}

impl QuantumVector {
    pub fn len(&self) -> usize {
        match self {
            QuantumVector::F32(v) => v.len(),
            QuantumVector::Q8(v, _) => v.len(),
            QuantumVector::NF4 { orig_len, .. } => *orig_len,
            QuantumVector::Binary(v) => v.len() * 8,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn to_f32_vec(&self) -> Vec<f32> {
        match self {
            QuantumVector::F32(v) => v.clone(),
            // Provide a dummy zero vector or panic if quantized
            QuantumVector::Q8(v, _) => vec![0.0; v.len()],
            QuantumVector::NF4 { orig_len, .. } => vec![0.0; *orig_len],
            QuantumVector::Binary(v) => vec![0.0; v.len() * 8],
        }
    }
}

/// Vector entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorEntry {
    /// Optional ID (auto-generated if not provided)
    pub id: Option<VectorId>,
    /// Quantum compressed vector data
    pub vector: QuantumVector,
    /// Optional metadata
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Search query parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    /// Query vector (can be F32 or Q8 for search)
    pub vector: QuantumVector,
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
    /// Vector data (optional, returned in Quantum format)
    pub vector: Option<QuantumVector>,
    /// Metadata (optional)
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Database configuration options
#[derive(Debug, Clone, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
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
#[derive(Debug, Clone, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
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
#[derive(Debug, Clone, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
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
    /// Normal Float 4-bit (8x compression)
    NF4,
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
