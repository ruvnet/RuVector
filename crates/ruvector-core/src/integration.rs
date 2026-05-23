//! Cross-integration helpers for ruvnet crate ecosystem.
//!
//! This module provides ergonomic adapters that make it straightforward to use
//! `ruvector-core` as a dependency from other ruvnet crates:
//!
//! - **ruv-FANN**: neural-network weights can be stored and retrieved via
//!   [`FannAdapter`] using cosine similarity search across layer embeddings.
//! - **sparc / semantic file search**: [`SemanticSearchAdapter`] wraps
//!   [`VectorDB`] with file-path metadata so sparc can locate relevant source
//!   files by embedding query strings.
//!
//! Both adapters are thin, zero-overhead wrappers — they own no additional
//! memory beyond what the underlying [`VectorDB`] already holds.

use crate::error::{Result, RuvectorError};
use crate::types::{DbOptions, DistanceMetric, HnswConfig, SearchQuery, SearchResult, VectorEntry};
use crate::vector_db::VectorDB;
use std::collections::HashMap;

// ── ruv-FANN integration ────────────────────────────────────────────────────

/// Adapter that lets ruv-FANN store and retrieve layer-weight embeddings.
///
/// Each neural-network layer can be fingerprinted as a flat `f32` embedding
/// (e.g. the flattened weight matrix or its PCA projection).  Storing these
/// fingerprints in RuVector enables fast recall of "similar layers" across
/// model checkpoints.
///
/// # Example
/// ```no_run
/// use ruvector_core::integration::FannAdapter;
///
/// let mut adapter = FannAdapter::new(128, "./fann_index.db").unwrap();
/// adapter.store_layer("model_v1/layer_0", &[0.1f32; 128], None).unwrap();
/// let similar = adapter.find_similar_layers(&[0.1f32; 128], 5).unwrap();
/// ```
pub struct FannAdapter {
    db: VectorDB,
}

impl FannAdapter {
    /// Create a new adapter backed by a RuVector database.
    ///
    /// `dimensions` must match the size of the layer embeddings you intend
    /// to store.  Cosine distance is used because weight embeddings are
    /// typically meaningful up to scale.
    pub fn new(dimensions: usize, storage_path: impl Into<String>) -> Result<Self> {
        let options = DbOptions {
            dimensions,
            distance_metric: DistanceMetric::Cosine,
            storage_path: storage_path.into(),
            hnsw_config: Some(HnswConfig {
                m: 16,
                ef_construction: 100,
                ef_search: 100,
                max_elements: 100_000,
            }),
            quantization: None,
        };
        Ok(Self {
            db: VectorDB::new(options)?,
        })
    }

    /// Store a layer embedding identified by `layer_id`.
    ///
    /// `metadata` can carry arbitrary JSON-serialisable key-value pairs
    /// (e.g. model name, checkpoint step, layer type).
    pub fn store_layer(
        &self,
        layer_id: impl Into<String>,
        embedding: &[f32],
        metadata: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<String> {
        let id = layer_id.into();
        self.db.insert(VectorEntry {
            id: Some(id),
            vector: embedding.to_vec(),
            metadata,
        })
    }

    /// Find the `k` most similar layer embeddings to `query`.
    ///
    /// Returns results sorted by ascending cosine distance.
    pub fn find_similar_layers(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.db.search(SearchQuery {
            vector: query.to_vec(),
            k,
            filter: None,
            ef_search: None,
        })
    }

    /// Find similar layers with a filter on metadata fields.
    ///
    /// Only results where every `(key, value)` in `filter` matches are returned.
    pub fn find_similar_layers_filtered(
        &self,
        query: &[f32],
        k: usize,
        filter: HashMap<String, serde_json::Value>,
    ) -> Result<Vec<SearchResult>> {
        self.db.search(SearchQuery {
            vector: query.to_vec(),
            k,
            filter: Some(filter),
            ef_search: None,
        })
    }

    /// Delete a layer embedding by ID.
    pub fn delete_layer(&self, layer_id: &str) -> Result<bool> {
        self.db.delete(layer_id)
    }

    /// Total number of stored layer embeddings.
    pub fn len(&self) -> Result<usize> {
        self.db.len()
    }

    /// Returns `true` if no embeddings have been stored yet.
    pub fn is_empty(&self) -> Result<bool> {
        self.db.is_empty()
    }
}

// ── sparc / semantic file search integration ────────────────────────────────

/// A file-path entry as indexed by [`SemanticSearchAdapter`].
#[derive(Debug, Clone)]
pub struct FileEntry {
    /// Absolute or relative path to the source file.
    pub path: String,
    /// Brief human-readable description of the file's contents.
    pub description: String,
    /// The embedding dimension used to index this file.
    pub dimensions: usize,
}

/// Adapter for sparc-style semantic file search.
///
/// sparc needs to locate relevant source files given a natural-language query
/// string.  This adapter stores one embedding per file (derived externally,
/// e.g. from an ONNX all-MiniLM model) and retrieves the closest matches
/// using HNSW approximate nearest-neighbour search.
///
/// # Example
/// ```no_run
/// use ruvector_core::integration::SemanticSearchAdapter;
///
/// let mut adapter = SemanticSearchAdapter::new(384, "./sparc_index.db").unwrap();
///
/// // Index source files (embeddings produced by your embedding pipeline)
/// adapter.index_file("src/auth/service.rs", "authentication service", &[0.0f32; 384]).unwrap();
/// adapter.index_file("src/user/model.rs", "user data model", &[0.1f32; 384]).unwrap();
///
/// // Query with a natural-language description
/// let results = adapter.search("jwt token validation", &[0.05f32; 384], 5).unwrap();
/// for r in results {
///     println!("  {} (score={:.4})", r.id, r.score);
/// }
/// ```
pub struct SemanticSearchAdapter {
    db: VectorDB,
    dimensions: usize,
}

impl SemanticSearchAdapter {
    /// Create a new adapter.
    ///
    /// `dimensions` is the embedding dimension of your model (e.g. 384 for
    /// all-MiniLM-L6-v2, 768 for BERT-base).
    pub fn new(dimensions: usize, storage_path: impl Into<String>) -> Result<Self> {
        let options = DbOptions {
            dimensions,
            distance_metric: DistanceMetric::Cosine,
            storage_path: storage_path.into(),
            hnsw_config: Some(HnswConfig {
                m: 16,
                ef_construction: 100,
                ef_search: 100,
                max_elements: 500_000,
            }),
            quantization: None,
        };
        Ok(Self {
            db: VectorDB::new(options)?,
            dimensions,
        })
    }

    /// Index a source file.
    ///
    /// The file `path` is used as the vector ID so look-ups are O(1).
    /// `description` is stored in metadata for debugging / display.
    /// `embedding` must have the same length as the adapter's `dimensions`.
    pub fn index_file(
        &self,
        path: impl Into<String>,
        description: impl Into<String>,
        embedding: &[f32],
    ) -> Result<String> {
        let path_str = path.into();
        if embedding.len() != self.dimensions {
            return Err(RuvectorError::DimensionMismatch {
                expected: self.dimensions,
                actual: embedding.len(),
            });
        }

        let mut metadata = HashMap::new();
        metadata.insert(
            "description".to_string(),
            serde_json::Value::String(description.into()),
        );
        metadata.insert(
            "path".to_string(),
            serde_json::Value::String(path_str.clone()),
        );

        self.db.insert(VectorEntry {
            id: Some(path_str),
            vector: embedding.to_vec(),
            metadata: Some(metadata),
        })
    }

    /// Remove a previously indexed file.
    pub fn remove_file(&self, path: &str) -> Result<bool> {
        self.db.delete(path)
    }

    /// Search for source files semantically related to `query_embedding`.
    ///
    /// Returns up to `k` results sorted by ascending cosine distance
    /// (most relevant first).  Each [`SearchResult`] has `.id` set to the
    /// file path and `.metadata` containing the description.
    pub fn search(
        &self,
        _query_text: &str,
        query_embedding: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        if query_embedding.len() != self.dimensions {
            return Err(RuvectorError::DimensionMismatch {
                expected: self.dimensions,
                actual: query_embedding.len(),
            });
        }
        self.db.search(SearchQuery {
            vector: query_embedding.to_vec(),
            k,
            filter: None,
            ef_search: None,
        })
    }

    /// Total number of indexed files.
    pub fn len(&self) -> Result<usize> {
        self.db.len()
    }

    /// Returns `true` if no files have been indexed yet.
    pub fn is_empty(&self) -> Result<bool> {
        self.db.is_empty()
    }

    /// List all indexed file paths.
    pub fn list_files(&self) -> Result<Vec<String>> {
        self.db.keys()
    }
}

// ── Shared utility ──────────────────────────────────────────────────────────

/// Normalise a vector to unit length for cosine-distance workloads.
///
/// Returns the original vector unchanged if its norm is effectively zero
/// (to avoid division by zero on zero vectors).
#[inline]
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    if norm_sq < f32::EPSILON {
        return v.to_vec();
    }
    let norm = norm_sq.sqrt();
    v.iter().map(|x| x / norm).collect()
}

/// Compute the cosine similarity in [−1, 1] between two vectors.
///
/// Both inputs are treated as raw (un-normalised) vectors.
/// Returns `0.0` if either vector is zero-length.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "cosine_similarity: length mismatch");
    let (mut dot, mut norm_a, mut norm_b) = (0.0f32, 0.0f32, 0.0f32);
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom > f32::EPSILON {
        dot / denom
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_normalize_unit_vector() {
        let v = vec![3.0f32, 4.0];
        let n = normalize(&v);
        let norm: f32 = n.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "Expected unit norm, got {}",
            norm
        );
    }

    #[test]
    fn test_normalize_zero_vector() {
        let v = vec![0.0f32, 0.0, 0.0];
        let n = normalize(&v);
        assert_eq!(n, v, "Zero vector should be returned unchanged");
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0f32, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Identical vectors: expected 1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            sim.abs() < 1e-5,
            "Orthogonal vectors: expected 0.0, got {}",
            sim
        );
    }

    #[test]
    fn test_semantic_search_adapter_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sparc.db").to_string_lossy().to_string();
        let adapter = SemanticSearchAdapter::new(4, path).unwrap();

        let emb_a = normalize(&[1.0, 0.0, 0.0, 0.0]);
        let emb_b = normalize(&[0.0, 1.0, 0.0, 0.0]);
        let emb_c = normalize(&[0.0, 0.0, 1.0, 0.0]);

        // hnsw_rs requires at least 2 elements before searching.
        adapter
            .index_file("src/auth.rs", "authentication", &emb_a)
            .unwrap();
        adapter
            .index_file("src/user.rs", "user model", &emb_b)
            .unwrap();
        adapter
            .index_file("src/storage.rs", "storage layer", &emb_c)
            .unwrap();

        assert_eq!(adapter.len().unwrap(), 3);

        // Query close to emb_a — should return src/auth.rs first
        let results = adapter.search("auth", &emb_a, 2).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "src/auth.rs");
    }

    #[test]
    fn test_fann_adapter_store_and_retrieve() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("fann.db").to_string_lossy().to_string();
        let adapter = FannAdapter::new(4, path).unwrap();

        let layer_emb_0 = normalize(&[1.0, 1.0, 0.0, 0.0]);
        let layer_emb_1 = normalize(&[0.0, 0.0, 1.0, 1.0]);
        let layer_emb_2 = normalize(&[1.0, 0.0, 1.0, 0.0]);

        // hnsw_rs requires at least 2 elements before searching.
        adapter
            .store_layer("model_v1/layer_0", &layer_emb_0, None)
            .unwrap();
        adapter
            .store_layer("model_v1/layer_1", &layer_emb_1, None)
            .unwrap();
        adapter
            .store_layer("model_v1/layer_2", &layer_emb_2, None)
            .unwrap();

        let results = adapter.find_similar_layers(&layer_emb_0, 1).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "model_v1/layer_0");
    }
}
