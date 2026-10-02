//! # streamcloud-memory
//!
//! Streaming **trajectory memory** for StreamCloud.
//!
//! ## Why this crate exists
//!
//! The original LingBot-Map (PyTorch + FlashInfer) manages temporal frame
//! representations in a **paged KV cache** living in GPU VRAM. That cache grows
//! linearly with the number of frames; past a few thousand frames the model
//! either exhausts VRAM or falls back to a *sliding window* that permanently
//! forgets old geometry, causing long-range drift.
//!
//! This crate replaces that mechanism with a lock-free **HNSW** index backed by
//! [`ruvector_core`]. Keyframe feature vectors are inserted as they stream in;
//! for each new frame the transformer queries the index for the top-K most
//! structurally similar past keyframes — whether they occurred 10 frames ago or
//! 10,000 frames ago — giving O(log N) drift correction with no VRAM ceiling.
//!
//! ## Relationship to the `ruvector-core` API
//!
//! Earlier design sketches referenced a `VectorDb` / `IndexConfig` /
//! `MetricType` surface. The *real* `ruvector-core` API is [`VectorDB`] driven
//! by [`DbOptions`] / [`VectorEntry`] / [`SearchQuery`] / [`SearchResult`].
//! [`StreamingMemory`] is the thin, intent-revealing wrapper over that real API
//! — `insert_keyframe` / `retrieve_context` — so the rest of the port speaks in
//! frame-domain terms (`u64` frame ids, `&[f32]` features) instead of database
//! terms.
//!
//! The wrapper is deliberately **candle-free**: it operates on `&[f32]` slices
//! so it builds fast and stays testable without a tensor backend. The
//! `streamcloud-pipeline` crate owns the `candle_core::Tensor` <-> `&[f32]` bridge.

use ruvector_core::types::{DbOptions, DistanceMetric, HnswConfig, SearchQuery, VectorEntry};
use ruvector_core::vector_db::VectorDB;

/// Errors surfaced by the streaming memory layer.
#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    /// A feature vector did not match the configured dimensionality.
    #[error("feature dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// Configured dimensionality.
        expected: usize,
        /// Dimensionality of the supplied vector.
        got: usize,
    },

    /// The underlying `ruvector-core` database returned an error.
    #[error("ruvector-core error: {0}")]
    Backend(String),
}

/// Distance metric used to compare keyframe features.
///
/// Geometric embeddings are normally L2-normalized, so [`Metric::Cosine`] is the
/// sensible default for structural similarity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Metric {
    /// Cosine similarity (recommended for normalized geometric embeddings).
    Cosine,
    /// Euclidean (L2) distance.
    Euclidean,
    /// Dot product.
    DotProduct,
}

impl From<Metric> for DistanceMetric {
    fn from(m: Metric) -> Self {
        match m {
            Metric::Cosine => DistanceMetric::Cosine,
            Metric::Euclidean => DistanceMetric::Euclidean,
            Metric::DotProduct => DistanceMetric::DotProduct,
        }
    }
}

/// Configuration for the streaming HNSW memory pool.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StreamingMemoryConfig {
    /// Dimensionality of keyframe feature vectors.
    pub dimension: usize,
    /// Distance metric.
    pub metric: Metric,
    /// HNSW connections per layer (M). Higher = better recall, more memory.
    pub hnsw_m: usize,
    /// HNSW build-time candidate list size (efConstruction).
    pub ef_construction: usize,
    /// HNSW query-time candidate list size (efSearch).
    pub ef_search: usize,
    /// Upper bound on stored keyframes (pre-allocation hint, not a hard cap on
    /// streaming length — bump it for very long sessions).
    pub max_elements: usize,
}

impl StreamingMemoryConfig {
    /// Config tuned for dense geometric keyframe lookups at the given dimension.
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            metric: Metric::Cosine,
            hnsw_m: 16,
            ef_construction: 128,
            ef_search: 64,
            max_elements: 1_000_000,
        }
    }
}

/// A retrieved keyframe and its similarity score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeyframeMatch {
    /// Frame id of the matched keyframe.
    pub frame_id: u64,
    /// Distance/similarity score (lower is closer for distance metrics).
    pub score: f32,
}

/// Streaming trajectory memory: insert keyframes, retrieve drift-correction
/// anchors. Replaces the paged KV cache of the Python original.
pub struct StreamingMemory {
    db: VectorDB,
    config: StreamingMemoryConfig,
}

impl StreamingMemory {
    /// Build an in-memory, lock-free HNSW keyframe index.
    pub fn new(config: StreamingMemoryConfig) -> Result<Self, MemoryError> {
        let options = DbOptions {
            dimensions: config.dimension,
            distance_metric: config.metric.into(),
            // In-memory backend (no `storage` feature): the storage path is
            // unused but the field is required.
            storage_path: String::from(":memory:"),
            hnsw_config: Some(HnswConfig {
                m: config.hnsw_m,
                ef_construction: config.ef_construction,
                ef_search: config.ef_search,
                max_elements: config.max_elements,
            }),
            quantization: None,
        };

        let db = VectorDB::new(options).map_err(|e| MemoryError::Backend(e.to_string()))?;
        Ok(Self { db, config })
    }

    /// Convenience constructor with defaults for the given dimension.
    pub fn with_dimension(dimension: usize) -> Result<Self, MemoryError> {
        Self::new(StreamingMemoryConfig::new(dimension))
    }

    /// The active configuration.
    pub fn config(&self) -> &StreamingMemoryConfig {
        &self.config
    }

    /// Ingest a keyframe's flattened feature vector.
    ///
    /// The `frame_id` is stored as the entry id so retrieval can map straight
    /// back to the streaming timeline. `ruvector-core` is lock-free, so this may
    /// run on a background ingestion thread while inference proceeds.
    pub fn insert_keyframe(&self, frame_id: u64, features: &[f32]) -> Result<(), MemoryError> {
        if features.len() != self.config.dimension {
            return Err(MemoryError::DimensionMismatch {
                expected: self.config.dimension,
                got: features.len(),
            });
        }

        let entry = VectorEntry {
            id: Some(frame_id.to_string()),
            vector: features.to_vec(),
            metadata: None,
        };

        self.db
            .insert(entry)
            .map(|_| ())
            .map_err(|e| MemoryError::Backend(e.to_string()))
    }

    /// Retrieve the top-K structurally most similar past keyframes for drift
    /// correction / pose-reference windowing.
    pub fn retrieve_context(
        &self,
        query_features: &[f32],
        top_k: usize,
    ) -> Result<Vec<KeyframeMatch>, MemoryError> {
        if query_features.len() != self.config.dimension {
            return Err(MemoryError::DimensionMismatch {
                expected: self.config.dimension,
                got: query_features.len(),
            });
        }

        let query = SearchQuery {
            vector: query_features.to_vec(),
            k: top_k,
            filter: None,
            ef_search: Some(self.config.ef_search.max(top_k)),
        };

        let results = self
            .db
            .search(query)
            .map_err(|e| MemoryError::Backend(e.to_string()))?;

        Ok(results
            .into_iter()
            .filter_map(|r| {
                r.id.parse::<u64>().ok().map(|frame_id| KeyframeMatch {
                    frame_id,
                    score: r.score,
                })
            })
            .collect())
    }

    /// Number of keyframes currently held in memory.
    pub fn len(&self) -> usize {
        self.db.len().unwrap_or(0)
    }

    /// Whether the memory pool is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic, low-collision synthetic feature. Uses a per-(seed, index)
    /// hash so distinct frames get distinct vectors with negligible aliasing,
    /// even across thousands of frames (a smooth `sin(seed)` ramp aliases).
    fn synth(dim: usize, seed: u64) -> Vec<f32> {
        (0..dim)
            .map(|i| {
                let h = (seed.wrapping_mul(0x9E3779B97F4A7C15))
                    ^ (i as u64).wrapping_mul(0xC2B2AE3D27D4EB4F);
                // map to [-1, 1]
                ((h >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn metric_maps_to_core() {
        assert_eq!(DistanceMetric::from(Metric::Cosine), DistanceMetric::Cosine);
        assert_eq!(
            DistanceMetric::from(Metric::Euclidean),
            DistanceMetric::Euclidean
        );
    }

    #[test]
    fn insert_and_count() {
        let mem = StreamingMemory::with_dimension(64).unwrap();
        assert!(mem.is_empty());
        for f in 0..10u64 {
            mem.insert_keyframe(f, &synth(64, f)).unwrap();
        }
        assert_eq!(mem.len(), 10);
    }

    #[test]
    fn rejects_wrong_dimension() {
        let mem = StreamingMemory::with_dimension(32).unwrap();
        let err = mem.insert_keyframe(0, &synth(16, 0)).unwrap_err();
        assert!(matches!(err, MemoryError::DimensionMismatch { .. }));
    }

    #[test]
    fn retrieves_self_as_nearest() {
        let dim = 128;
        let mem = StreamingMemory::with_dimension(dim).unwrap();
        for f in 0..200u64 {
            mem.insert_keyframe(f, &synth(dim, f)).unwrap();
        }
        // Query with the exact features of frame 137; it should be the top hit.
        let matches = mem.retrieve_context(&synth(dim, 137), 5).unwrap();
        assert!(!matches.is_empty());
        assert_eq!(matches[0].frame_id, 137);
    }

    #[test]
    fn long_range_recall_beyond_sliding_window() {
        // A frame 5000 steps in the past is still retrievable — the property a
        // fixed sliding-window KV cache cannot provide.
        let dim = 96;
        let mem = StreamingMemory::with_dimension(dim).unwrap();
        for f in 0..5_001u64 {
            mem.insert_keyframe(f, &synth(dim, f)).unwrap();
        }
        let matches = mem.retrieve_context(&synth(dim, 3), 3).unwrap();
        assert_eq!(matches[0].frame_id, 3);
    }
}
