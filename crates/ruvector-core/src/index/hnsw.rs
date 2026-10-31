//! HNSW (Hierarchical Navigable Small World) index implementation

use crate::distance::distance;
use crate::error::{Result, RuvectorError};
use crate::index::flat_graph::FlatGraph;
use crate::index::VectorIndex;
use crate::types::{DistanceMetric, HnswConfig, SearchResult, VectorId};
use bincode::{Decode, Encode};
use dashmap::DashMap;
use parking_lot::RwLock;
use std::sync::Arc;

/// At or below this many live vectors, HNSW's approximation is not worth its
/// blind spots (ruvnet/RuVector#1298): a graph this small is scanned exactly,
/// which also skips tombstones left behind by deletes.
const EXACT_SEARCH_MAX_ELEMENTS: usize = 1_024;

/// HNSW index wrapper
pub struct HnswIndex {
    inner: Arc<RwLock<HnswInner>>,
    config: HnswConfig,
    metric: DistanceMetric,
    dimensions: usize,
}

struct HnswInner {
    hnsw: FlatGraph,
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
    /// Create a new HNSW index
    pub fn new(dimensions: usize, metric: DistanceMetric, config: HnswConfig) -> Result<Self> {
        let hnsw = FlatGraph::new(dimensions, metric, config.m, config.ef_construction);

        Ok(Self {
            inner: Arc::new(RwLock::new(HnswInner {
                hnsw,
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

    /// Get configuration
    /// Structural violations of the layer invariants restored by the
    /// ruvnet/RuVector#773 fix: `(edges above an endpoint's level, points
    /// unreachable over layer-0 edges)`.
    ///
    /// Both are empty on a healthy index. Exposed so the regression suite can
    /// assert the invariant EXHAUSTIVELY on any index, instead of sampling
    /// `search()` results and arguing from probability — levels come from
    /// `StdRng::from_entropy()`, so a sampled trial cannot be made
    /// deterministic, while these hold for every draw.
    #[doc(hidden)]
    #[cfg(feature = "hnsw-structural-diagnostics")]
    pub fn structural_violations(&self) -> (Vec<String>, Vec<usize>) {
        let guard = self.inner.read();
        (
            guard.hnsw.edges_above_endpoint_level(),
            guard.hnsw.points_unreachable_at_layer0(),
        )
    }

    pub fn config(&self) -> &HnswConfig {
        &self.config
    }

    /// Set efSearch parameter for query-time accuracy tuning.
    ///
    /// Higher values increase recall at the cost of search latency.
    /// Typical range: 50–500. Must be >= k for meaningful results.
    pub fn set_ef_search(&mut self, ef_search: usize) {
        self.config.ef_search = ef_search;
    }

    /// Serialize the index to bytes using bincode
    pub fn serialize(&self) -> Result<Vec<u8>> {
        let inner = self.inner.read();

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

        let mut hnsw = FlatGraph::new(dimensions, metric, config.m, config.ef_construction);

        // Rebuild the index by inserting all vectors.
        // Build a HashMap first to avoid O(n^2) linear search in the loop below.
        let vectors_lookup: std::collections::HashMap<&str, &Vec<f32>> = state
            .vectors
            .iter()
            .map(|(id, v)| (id.as_str(), v))
            .collect();

        let id_to_idx: DashMap<VectorId, usize> = state.id_to_idx.into_iter().collect();
        let idx_to_id: DashMap<usize, VectorId> = state.idx_to_id.into_iter().collect();

        // Insert vectors into HNSW in index order for deterministic reconstruction.
        let mut sorted_entries: Vec<_> = idx_to_id
            .iter()
            .map(|e| (*e.key(), e.value().clone()))
            .collect();
        sorted_entries.sort_unstable_by_key(|(idx, _)| *idx);

        for (idx, id) in &sorted_entries {
            if let Some(vector) = vectors_lookup.get(id.as_str()) {
                hnsw.insert(vector, *idx);
            }
        }

        let vectors_map: DashMap<VectorId, Vec<f32>> = state.vectors.into_iter().collect();

        Ok(Self {
            inner: Arc::new(RwLock::new(HnswInner {
                hnsw,
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
    /// `ef_search` must be >= `k`; values smaller than `k` are clamped to `k`
    /// to avoid silent under-recall.  Results are returned sorted by ascending
    /// distance (closest first).
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

        if k == 0 {
            return Ok(vec![]);
        }

        let inner = self.inner.read();

        // hnsw_rs panics in its BinaryHeap traversal when the index is empty
        // or contains only a single element (the candidate/return-point loop
        // calls .peek().unwrap() without an emptiness guard).  Return early
        // to surface a clean error instead of an assertion panic.
        if inner.vectors.is_empty() {
            return Ok(vec![]);
        }

        // On tiny indexes, scan the live vectors exactly. HNSW can return
        // fewer than the stored rows (ruvnet/RuVector#1298), and a deleted
        // node still occupies a graph slot, so a top-k cut made before
        // filtering tombstones also drops live neighbours.
        if inner.vectors.len() <= EXACT_SEARCH_MAX_ELEMENTS {
            let mut results: Vec<SearchResult> = inner
                .vectors
                .iter()
                .map(|entry| SearchResult {
                    id: entry.key().clone(),
                    score: distance(query, entry.value(), self.metric).unwrap_or(f32::MAX),
                    vector: None,
                    metadata: None,
                })
                .collect();
            results.sort_unstable_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            results.truncate(k);
            return Ok(results);
        }

        // ef_search < k causes the graph to return fewer than k candidates;
        // clamp. Tombstoned nodes still have graph slots, so over-fetch enough
        // candidates to fill k after they are filtered out.
        let tombstones = inner.hnsw.len().saturating_sub(inner.vectors.len());
        let wanted = k.saturating_add(tombstones);
        let effective_ef = ef_search.max(wanted);

        let neighbors = inner.hnsw.search(query, wanted, effective_ef);

        let mut results: Vec<SearchResult> = neighbors
            .into_iter()
            .filter_map(|(idx, distance)| {
                inner.idx_to_id.get(&idx).map(|id| SearchResult {
                    id: id.clone(),
                    score: distance,
                    vector: None,
                    metadata: None,
                })
            })
            .collect();

        // The graph does not guarantee sort order — ensure ascending distance.
        results.sort_unstable_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);

        Ok(results)
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
        let idx = inner.next_idx;
        inner.next_idx += 1;

        inner.hnsw.insert(&vector, idx);

        // Store mappings
        inner.vectors.insert(id.clone(), vector);
        inner.id_to_idx.insert(id.clone(), idx);
        inner.idx_to_id.insert(idx, id);

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

        // Assign indices, then insert in order. (The graph has one writer; batches are not split across threads.)
        let data_with_ids: Vec<_> = entries
            .into_iter()
            .enumerate()
            .map(|(i, (id, vector))| (id, inner.next_idx + i, vector))
            .collect();
        inner.next_idx += data_with_ids.len();
        let items: Vec<(&[f32], usize)> = data_with_ids
            .iter()
            .map(|(_, idx, v)| (v.as_slice(), *idx))
            .collect();
        inner.hnsw.insert_batch(&items);

        // Store mappings
        for (id, idx, vector) in data_with_ids {
            inner.vectors.insert(id.clone(), vector);
            inner.id_to_idx.insert(id.clone(), idx);
            inner.idx_to_id.insert(idx, id);
        }

        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        // Use configured ef_search
        self.search_with_ef(query, k, self.config.ef_search)
    }

    fn search_ef(
        &self,
        query: &[f32],
        k: usize,
        ef_search: Option<usize>,
    ) -> Result<Vec<SearchResult>> {
        self.search_with_ef(query, k, ef_search.unwrap_or(self.config.ef_search))
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
    fn test_search_ef_overrides_the_configured_value() -> Result<()> {
        let (n, dim, k) = (1500usize, 16usize, 10usize);
        let base = generate_random_vectors(n, dim);
        let queries = generate_random_vectors(60, dim);
        let config = HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: k,
            max_elements: n + 10,
        };
        let mut index = HnswIndex::new(dim, DistanceMetric::Euclidean, config)?;
        for (i, v) in base.iter().enumerate() {
            index.add(i.to_string(), v.clone())?;
        }
        let truth = |q: &[f32]| -> Vec<String> {
            let mut d: Vec<(f32, usize)> = base
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    (
                        v.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum::<f32>(),
                        i,
                    )
                })
                .collect();
            d.sort_by(|a, b| a.0.total_cmp(&b.0));
            d.into_iter().take(k).map(|x| x.1.to_string()).collect()
        };
        let (mut low, mut high) = (0usize, 0usize);
        for q in &queries {
            let t = truth(q);
            let count = |r: Vec<SearchResult>| r.iter().filter(|x| t.contains(&x.id)).count();
            assert_eq!(
                index
                    .search_ef(q, k, None)?
                    .iter()
                    .map(|r| r.id.clone())
                    .collect::<Vec<_>>(),
                index
                    .search(q, k)?
                    .iter()
                    .map(|r| r.id.clone())
                    .collect::<Vec<_>>(),
                "None means the configured value"
            );
            low += count(index.search_ef(q, k, Some(k))?);
            high += count(index.search_ef(q, k, Some(300))?);
        }
        let (low, high) = (
            low as f64 / (queries.len() * k) as f64,
            high as f64 / (queries.len() * k) as f64,
        );
        assert!(
            high >= low,
            "a larger ef must not lower recall: {low} -> {high}"
        );
        assert!(high > 0.97, "ef=300 recall {high}");
        Ok(())
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
