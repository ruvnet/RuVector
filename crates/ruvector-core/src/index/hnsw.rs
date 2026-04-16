//! HNSW (Hierarchical Navigable Small World) index implementation

use crate::distance::distance;
use crate::error::{Result, RuvectorError};
use crate::index::hnsw_selected::{SelectedDimsBackend, SelectedDimsSelector};
use crate::index::VectorIndex;
use crate::types::{DistanceMetric, HnswConfig, SearchResult, VectorId};
use bincode::{Decode, Encode};
use dashmap::DashMap;
use hnsw_rs::prelude::*;
use parking_lot::RwLock;
use std::sync::Arc;

/// Distance function wrapper for hnsw_rs
struct DistanceFn {
    metric: DistanceMetric,
}

impl DistanceFn {
    fn new(metric: DistanceMetric) -> Self {
        Self { metric }
    }
}

impl Distance<f32> for DistanceFn {
    fn eval(&self, a: &[f32], b: &[f32]) -> f32 {
        distance(a, b, self.metric).unwrap_or(f32::MAX)
    }
}

/// HNSW index wrapper.
///
/// Two backends are selectable at construction time:
///
/// - [`HnswIndex::new`] — standard full-dim HNSW with the configured metric.
///   This is the legacy path; existing callers see zero behavioural or API
///   change.
/// - [`HnswIndex::new_with_selected_dims`] — project vectors onto a learned
///   subset of dimensions, build a reduced-dim HNSW, then expose exact
///   re-rank via [`HnswIndex::search_with_rerank`]. See ADR-151 for the
///   acceptance bar and measured SIFT1M evidence.
pub struct HnswIndex {
    inner: Arc<RwLock<HnswInner>>,
    config: HnswConfig,
    metric: DistanceMetric,
    dimensions: usize,
}

/// Internal backend — either a plain HNSW (legacy default) or a learned
/// selected-dims pipeline. Callers never see this; dispatch is by
/// [`HnswIndex`] methods.
enum Backend {
    /// Full-dim HNSW. This is what every existing caller gets today.
    Standard(Hnsw<'static, f32, DistanceFn>),
    /// Selected-dims HNSW: trained selector + reduced-dim graph + full-dim
    /// store for exact rerank.
    SelectedDims(Box<SelectedDimsBackend>),
}

struct HnswInner {
    backend: Backend,
    vectors: DashMap<VectorId, Vec<f32>>,
    id_to_idx: DashMap<VectorId, usize>,
    idx_to_id: DashMap<usize, VectorId>,
    next_idx: usize,
}

/// Serializable HNSW index state
#[derive(Encode, Decode, Clone)]
pub struct HnswState {
    vectors: Vec<(String, Vec<f32>)>,
    id_to_idx: Vec<(String, usize)>,
    idx_to_id: Vec<(usize, String)>,
    next_idx: usize,
    config: SerializableHnswConfig,
    dimensions: usize,
    metric: SerializableDistanceMetric,
}

#[derive(Encode, Decode, Clone)]
struct SerializableHnswConfig {
    m: usize,
    ef_construction: usize,
    ef_search: usize,
    max_elements: usize,
}

#[derive(Encode, Decode, Clone, Copy)]
enum SerializableDistanceMetric {
    Euclidean,
    Cosine,
    DotProduct,
    Manhattan,
}

impl From<DistanceMetric> for SerializableDistanceMetric {
    fn from(metric: DistanceMetric) -> Self {
        match metric {
            DistanceMetric::Euclidean => SerializableDistanceMetric::Euclidean,
            DistanceMetric::Cosine => SerializableDistanceMetric::Cosine,
            DistanceMetric::DotProduct => SerializableDistanceMetric::DotProduct,
            DistanceMetric::Manhattan => SerializableDistanceMetric::Manhattan,
        }
    }
}

impl From<SerializableDistanceMetric> for DistanceMetric {
    fn from(metric: SerializableDistanceMetric) -> Self {
        match metric {
            SerializableDistanceMetric::Euclidean => DistanceMetric::Euclidean,
            SerializableDistanceMetric::Cosine => DistanceMetric::Cosine,
            SerializableDistanceMetric::DotProduct => DistanceMetric::DotProduct,
            SerializableDistanceMetric::Manhattan => DistanceMetric::Manhattan,
        }
    }
}

impl HnswIndex {
    /// Create a new full-dim HNSW index.
    ///
    /// This is the default path: vectors are indexed and searched at their
    /// full dimensionality using the requested `metric`. Existing callers
    /// see no change.
    pub fn new(dimensions: usize, metric: DistanceMetric, config: HnswConfig) -> Result<Self> {
        let distance_fn = DistanceFn::new(metric);

        // Create HNSW with configured parameters
        let hnsw = Hnsw::<f32, DistanceFn>::new(
            config.m,
            config.max_elements,
            dimensions,
            config.ef_construction,
            distance_fn,
        );

        Ok(Self {
            inner: Arc::new(RwLock::new(HnswInner {
                backend: Backend::Standard(hnsw),
                vectors: DashMap::new(),
                id_to_idx: DashMap::new(),
                idx_to_id: DashMap::new(),
                next_idx: 0,
            })),
            config,
            metric,
            dimensions,
        })
    }

    /// Construct an HNSW index that projects vectors onto a learned subset
    /// of dimensions on insert/search, then exposes exact re-rank via
    /// [`HnswIndex::search_with_rerank`]. See ADR-151 for the acceptance bar
    /// and measured SIFT1M evidence.
    ///
    /// # Arguments
    ///
    /// - `full_dim`: dimensionality of the input vectors (e.g. 128 for SIFT).
    /// - `representative_sample`: a set of vectors drawn from the same
    ///   distribution as the production corpus — used to train the
    ///   dimension selector. 500–2000 vectors is the typical range.
    /// - `selected_k`: reduced dimensionality the graph will be built on.
    ///   On SIFT1M, `selected_k ∈ [32, 48]` is the empirically validated
    ///   band.
    /// - `metric`: the exact metric used for re-rank. The underlying
    ///   reduced-dim graph always uses cosine on the projection (the cheapest
    ///   monotone proxy); `metric` only affects what
    ///   [`HnswIndex::search_with_rerank`] computes at full dim.
    /// - `config`: standard HNSW parameters (`m`, `ef_construction`,
    ///   `ef_search`, `max_elements`).
    ///
    /// # Acceptance bar (SIFT1M, from ADR-151)
    ///
    /// - `selected_k ∈ [32, 48]`
    /// - `fetch_k ≥ 500` at search time
    /// - recall\@10 with rerank ≥ 0.80
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ruvector_core::index::hnsw::HnswIndex;
    /// use ruvector_core::types::{DistanceMetric, HnswConfig};
    /// use ruvector_core::index::VectorIndex;
    ///
    /// // 1) representative sample drawn from the production distribution
    /// let sample: Vec<Vec<f32>> = (0..1000).map(|i| vec![i as f32 / 1000.0; 128]).collect();
    ///
    /// // 2) build a selected-dims index (selected_k = 32, cosine rerank)
    /// let mut index = HnswIndex::new_with_selected_dims(
    ///     128,
    ///     &sample,
    ///     32,
    ///     DistanceMetric::Cosine,
    ///     HnswConfig::default(),
    /// ).expect("selector trains on the sample");
    ///
    /// // 3) insert full-dim vectors as usual
    /// for (i, v) in sample.iter().enumerate() {
    ///     index.add(format!("v_{i}"), v.clone()).unwrap();
    /// }
    ///
    /// // 4) approximate-then-exact search: pull fetch_k=500 candidates at
    /// //    reduced dim, rerank with full-dim cosine, return top-10.
    /// let hits = index.search_with_rerank(&sample[0], 10, 500, 100).unwrap();
    /// assert!(!hits.is_empty());
    /// ```
    pub fn new_with_selected_dims(
        full_dim: usize,
        representative_sample: &[Vec<f32>],
        selected_k: usize,
        metric: DistanceMetric,
        config: HnswConfig,
    ) -> Result<Self> {
        if representative_sample.is_empty() {
            return Err(RuvectorError::InvalidInput(
                "new_with_selected_dims requires a non-empty representative_sample".into(),
            ));
        }
        if representative_sample[0].len() != full_dim {
            return Err(RuvectorError::DimensionMismatch {
                expected: full_dim,
                actual: representative_sample[0].len(),
            });
        }
        if selected_k == 0 || selected_k > full_dim {
            return Err(RuvectorError::InvalidInput(format!(
                "selected_k must be in 1..={full_dim}, got {selected_k}"
            )));
        }

        let selector = SelectedDimsSelector::train(representative_sample, selected_k)?;
        let backend = SelectedDimsBackend::new(selector, &config);

        Ok(Self {
            inner: Arc::new(RwLock::new(HnswInner {
                backend: Backend::SelectedDims(Box::new(backend)),
                vectors: DashMap::new(),
                id_to_idx: DashMap::new(),
                idx_to_id: DashMap::new(),
                next_idx: 0,
            })),
            config,
            metric,
            dimensions: full_dim,
        })
    }

    /// Get configuration
    pub fn config(&self) -> &HnswConfig {
        &self.config
    }

    /// Set efSearch parameter for query-time accuracy tuning
    pub fn set_ef_search(&mut self, _ef_search: usize) {
        // Note: hnsw_rs controls ef_search via the search method's knbn parameter
        // We store it in config and use it in search_with_ef
    }

    /// Serialize the index to bytes using bincode.
    ///
    /// Only the `Standard` backend is currently serialisable; attempting to
    /// serialise a `SelectedDims` index returns
    /// [`RuvectorError::SerializationError`]. Persisting the learned
    /// selector + reduced-dim graph is tracked as a follow-up.
    pub fn serialize(&self) -> Result<Vec<u8>> {
        let inner = self.inner.read();
        if let Backend::SelectedDims(_) = inner.backend {
            return Err(RuvectorError::SerializationError(
                "HnswIndex serialization is not yet supported for the selected-dims backend"
                    .into(),
            ));
        }

        let state = HnswState {
            vectors: inner
                .vectors
                .iter()
                .map(|entry| (entry.key().clone(), entry.value().clone()))
                .collect(),
            id_to_idx: inner
                .id_to_idx
                .iter()
                .map(|entry| (entry.key().clone(), *entry.value()))
                .collect(),
            idx_to_id: inner
                .idx_to_id
                .iter()
                .map(|entry| (*entry.key(), entry.value().clone()))
                .collect(),
            next_idx: inner.next_idx,
            config: SerializableHnswConfig {
                m: self.config.m,
                ef_construction: self.config.ef_construction,
                ef_search: self.config.ef_search,
                max_elements: self.config.max_elements,
            },
            dimensions: self.dimensions,
            metric: self.metric.into(),
        };

        bincode::encode_to_vec(&state, bincode::config::standard()).map_err(|e| {
            RuvectorError::SerializationError(format!("Failed to serialize HNSW index: {}", e))
        })
    }

    /// Deserialize the index from bytes using bincode
    pub fn deserialize(bytes: &[u8]) -> Result<Self> {
        let (state, _): (HnswState, usize) =
            bincode::decode_from_slice(bytes, bincode::config::standard()).map_err(|e| {
                RuvectorError::SerializationError(format!(
                    "Failed to deserialize HNSW index: {}",
                    e
                ))
            })?;

        let config = HnswConfig {
            m: state.config.m,
            ef_construction: state.config.ef_construction,
            ef_search: state.config.ef_search,
            max_elements: state.config.max_elements,
        };

        let dimensions = state.dimensions;
        let metric: DistanceMetric = state.metric.into();

        let distance_fn = DistanceFn::new(metric);
        let mut hnsw = Hnsw::<'static, f32, DistanceFn>::new(
            config.m,
            config.max_elements,
            dimensions,
            config.ef_construction,
            distance_fn,
        );

        // Rebuild the index by inserting all vectors
        let id_to_idx: DashMap<VectorId, usize> = state.id_to_idx.into_iter().collect();
        let idx_to_id: DashMap<usize, VectorId> = state.idx_to_id.into_iter().collect();

        // Insert vectors into HNSW in order
        for entry in idx_to_id.iter() {
            let idx = *entry.key();
            let id = entry.value();
            if let Some(vector) = state.vectors.iter().find(|(vid, _)| vid == id) {
                // Use insert_data method with slice and idx
                hnsw.insert_data(&vector.1, idx);
            }
        }

        let vectors_map: DashMap<VectorId, Vec<f32>> = state.vectors.into_iter().collect();

        Ok(Self {
            inner: Arc::new(RwLock::new(HnswInner {
                backend: Backend::Standard(hnsw),
                vectors: vectors_map,
                id_to_idx,
                idx_to_id,
                next_idx: state.next_idx,
            })),
            config,
            metric,
            dimensions,
        })
    }

    /// Search with custom efSearch parameter.
    ///
    /// Dispatches on the underlying backend:
    /// - `Standard` — full-dim HNSW search.
    /// - `SelectedDims` — reduced-dim HNSW search on the projection. The
    ///   returned `score` is cosine over the projection (a monotone proxy);
    ///   if you need the exact-metric score, call
    ///   [`HnswIndex::search_with_rerank`] instead.
    pub fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<SearchResult>> {
        if query.len() != self.dimensions {
            return Err(RuvectorError::DimensionMismatch {
                expected: self.dimensions,
                actual: query.len(),
            });
        }

        let inner = self.inner.read();

        let neighbors: Vec<(usize, f32)> = match &inner.backend {
            Backend::Standard(hnsw) => hnsw
                .search(query, k, ef_search)
                .into_iter()
                .map(|n| (n.d_id, n.distance))
                .collect(),
            Backend::SelectedDims(sd) => sd.search_reduced(query, k, ef_search),
        };

        Ok(neighbors
            .into_iter()
            .filter_map(|(d_id, distance)| {
                inner.idx_to_id.get(&d_id).map(|id| SearchResult {
                    id: id.clone(),
                    score: distance,
                    vector: None,
                    metadata: None,
                })
            })
            .collect())
    }

    /// Approximate-then-exact search.
    ///
    /// For the selected-dims backend: pulls `fetch_k` candidates from the
    /// reduced-dim graph, re-ranks them with the configured full-dim
    /// `metric`, and returns the top-`k`. This is the canonical "fast
    /// narrow, then exact re-order" pipeline from ADR-151.
    ///
    /// For the standard backend: falls back to a plain
    /// [`HnswIndex::search_with_ef`] call with `k`. `fetch_k` is ignored in
    /// that path because the graph is already at full dim.
    ///
    /// # Recommended operating points (SIFT1M, ADR-151)
    ///
    /// - `fetch_k` ≥ 500
    /// - `selected_k` (at construction) ∈ [32, 48]
    ///
    /// Smaller `fetch_k` trades recall for latency.
    pub fn search_with_rerank(
        &self,
        query: &[f32],
        k: usize,
        fetch_k: usize,
        ef_search: usize,
    ) -> Result<Vec<SearchResult>> {
        if query.len() != self.dimensions {
            return Err(RuvectorError::DimensionMismatch {
                expected: self.dimensions,
                actual: query.len(),
            });
        }

        let inner = self.inner.read();

        let neighbors: Vec<(usize, f32)> = match &inner.backend {
            Backend::Standard(_) => {
                drop(inner);
                return self.search_with_ef(query, k, ef_search);
            }
            Backend::SelectedDims(sd) => {
                sd.search_with_rerank(query, k, fetch_k, ef_search, self.metric)
            }
        };

        Ok(neighbors
            .into_iter()
            .filter_map(|(d_id, distance)| {
                inner.idx_to_id.get(&d_id).map(|id| SearchResult {
                    id: id.clone(),
                    score: distance,
                    vector: None,
                    metadata: None,
                })
            })
            .collect())
    }
}

impl VectorIndex for HnswIndex {
    fn add(&mut self, id: VectorId, vector: Vec<f32>) -> Result<()> {
        if vector.len() != self.dimensions {
            return Err(RuvectorError::DimensionMismatch {
                expected: self.dimensions,
                actual: vector.len(),
            });
        }

        let mut inner = self.inner.write();
        // Destructure to sidestep the "simultaneous mutable borrows of the
        // same struct" borrow-check trap — `&mut inner.backend` and
        // `inner.next_idx` alias `inner` otherwise.
        let HnswInner {
            backend,
            vectors,
            id_to_idx,
            idx_to_id,
            next_idx,
        } = &mut *inner;

        // For the SelectedDims backend we let the backend own the running
        // index so the internal id matches `full_store` position; for
        // Standard we keep using the existing next_idx scheme.
        let idx = match backend {
            Backend::Standard(hnsw) => {
                let i = *next_idx;
                *next_idx += 1;
                hnsw.insert_data(&vector, i);
                i
            }
            Backend::SelectedDims(sd) => {
                let i = sd.insert(&vector);
                *next_idx = i + 1;
                i
            }
        };

        vectors.insert(id.clone(), vector);
        id_to_idx.insert(id.clone(), idx);
        idx_to_id.insert(idx, id);

        Ok(())
    }

    fn add_batch(&mut self, entries: Vec<(VectorId, Vec<f32>)>) -> Result<()> {
        // Validate all dimensions first
        for (_, vector) in &entries {
            if vector.len() != self.dimensions {
                return Err(RuvectorError::DimensionMismatch {
                    expected: self.dimensions,
                    actual: vector.len(),
                });
            }
        }

        let mut inner = self.inner.write();
        let HnswInner {
            backend,
            vectors,
            id_to_idx,
            idx_to_id,
            next_idx,
        } = &mut *inner;

        match backend {
            Backend::Standard(hnsw) => {
                let base = *next_idx;
                let data_with_ids: Vec<_> = entries
                    .iter()
                    .enumerate()
                    .map(|(i, (id, vector))| (id.clone(), base + i, vector.clone()))
                    .collect();

                *next_idx += entries.len();

                // Sequential insertion avoids Send requirements on the RwLock
                // guard. hnsw_rs parallel_insert could be reintroduced once
                // we refactor for Send-safe batch access.
                for (_id, idx, vector) in &data_with_ids {
                    hnsw.insert_data(vector, *idx);
                }

                for (id, idx, vector) in data_with_ids {
                    vectors.insert(id.clone(), vector);
                    id_to_idx.insert(id.clone(), idx);
                    idx_to_id.insert(idx, id);
                }
            }
            Backend::SelectedDims(sd) => {
                // The backend assigns indices itself (next_idx = full_store
                // position). Mirror them into our id maps.
                let mut assigned: Vec<(VectorId, usize, Vec<f32>)> =
                    Vec::with_capacity(entries.len());
                for (id, vector) in entries.into_iter() {
                    let idx = sd.insert(&vector);
                    assigned.push((id, idx, vector));
                }
                if let Some((_, last_idx, _)) = assigned.last() {
                    *next_idx = last_idx + 1;
                }
                for (id, idx, vector) in assigned {
                    vectors.insert(id.clone(), vector);
                    id_to_idx.insert(id.clone(), idx);
                    idx_to_id.insert(idx, id);
                }
            }
        }

        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        // Use configured ef_search
        self.search_with_ef(query, k, self.config.ef_search)
    }

    fn remove(&mut self, id: &VectorId) -> Result<bool> {
        let inner = self.inner.write();

        // Note: hnsw_rs doesn't support direct deletion
        // We remove from our mappings but the graph structure remains
        // This is a known limitation of HNSW
        let removed = inner.vectors.remove(id).is_some();

        if removed {
            if let Some((_, idx)) = inner.id_to_idx.remove(id) {
                inner.idx_to_id.remove(&idx);
            }
        }

        Ok(removed)
    }

    fn len(&self) -> usize {
        self.inner.read().vectors.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_random_vectors(count: usize, dimensions: usize) -> Vec<Vec<f32>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        (0..count)
            .map(|_| (0..dimensions).map(|_| rng.gen::<f32>()).collect())
            .collect()
    }

    fn normalize_vector(v: &[f32]) -> Vec<f32> {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter().map(|x| x / norm).collect()
        } else {
            v.to_vec()
        }
    }

    #[test]
    fn test_hnsw_index_creation() -> Result<()> {
        let config = HnswConfig::default();
        let index = HnswIndex::new(128, DistanceMetric::Cosine, config)?;
        assert_eq!(index.len(), 0);
        Ok(())
    }

    #[test]
    fn test_hnsw_insert_and_search() -> Result<()> {
        let config = HnswConfig {
            m: 16,
            ef_construction: 100,
            ef_search: 50,
            max_elements: 1000,
        };

        let mut index = HnswIndex::new(128, DistanceMetric::Cosine, config)?;

        // Insert a few vectors
        let vectors = generate_random_vectors(100, 128);
        for (i, vector) in vectors.iter().enumerate() {
            let normalized = normalize_vector(vector);
            index.add(format!("vec_{}", i), normalized)?;
        }

        assert_eq!(index.len(), 100);

        // Search for the first vector
        let query = normalize_vector(&vectors[0]);
        let results = index.search(&query, 10)?;

        assert!(!results.is_empty());
        assert_eq!(results[0].id, "vec_0");

        Ok(())
    }

    #[test]
    fn test_hnsw_batch_insert() -> Result<()> {
        let config = HnswConfig::default();
        let mut index = HnswIndex::new(128, DistanceMetric::Cosine, config)?;

        let vectors = generate_random_vectors(100, 128);
        let entries: Vec<_> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (format!("vec_{}", i), normalize_vector(v)))
            .collect();

        index.add_batch(entries)?;
        assert_eq!(index.len(), 100);

        Ok(())
    }

    #[test]
    fn test_hnsw_serialization() -> Result<()> {
        let config = HnswConfig {
            m: 16,
            ef_construction: 100,
            ef_search: 50,
            max_elements: 1000,
        };

        let mut index = HnswIndex::new(128, DistanceMetric::Cosine, config)?;

        // Insert vectors
        let vectors = generate_random_vectors(50, 128);
        for (i, vector) in vectors.iter().enumerate() {
            let normalized = normalize_vector(vector);
            index.add(format!("vec_{}", i), normalized)?;
        }

        // Serialize
        let bytes = index.serialize()?;

        // Deserialize
        let restored_index = HnswIndex::deserialize(&bytes)?;

        assert_eq!(restored_index.len(), 50);

        // Test search on restored index
        let query = normalize_vector(&vectors[0]);
        let results = restored_index.search(&query, 5)?;

        assert!(!results.is_empty());

        Ok(())
    }

    #[test]
    fn test_dimension_mismatch() -> Result<()> {
        let config = HnswConfig::default();
        let mut index = HnswIndex::new(128, DistanceMetric::Cosine, config)?;

        let result = index.add("test".to_string(), vec![1.0; 64]);
        assert!(result.is_err());

        Ok(())
    }
}
