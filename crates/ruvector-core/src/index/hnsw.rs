//! HNSW (Hierarchical Navigable Small World) index implementation

use crate::distance::distance;
use crate::error::{Result, RuvectorError};
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
    #[inline(always)]
    fn eval(&self, a: &[f32], b: &[f32]) -> f32 {
        // Bypass the simsimd/Result-overhead path and call our hand-written
        // SIMD kernels directly.  hnsw_rs asserts dist >= 0 in its search
        // loop, so clamp any floating-point rounding below zero.
        use crate::simd_intrinsics;
        match self.metric {
            DistanceMetric::Euclidean => simd_intrinsics::euclidean_distance_simd(a, b),
            DistanceMetric::Cosine => {
                // cosine_similarity_simd returns dot/(|a||b|); HNSW needs
                // cosine DISTANCE = 1 - sim, clamped to 0.
                (1.0_f32 - simd_intrinsics::cosine_similarity_simd(a, b)).max(0.0)
            }
            DistanceMetric::DotProduct => {
                // Negate for minimization; clamp per hnsw_rs assertion.
                (-simd_intrinsics::dot_product_simd(a, b)).max(0.0)
            }
            DistanceMetric::Manhattan => simd_intrinsics::manhattan_distance_simd(a, b),
        }
    }
}

/// HNSW index wrapper
pub struct HnswIndex {
    inner: Arc<RwLock<HnswInner>>,
    config: HnswConfig,
    metric: DistanceMetric,
    dimensions: usize,
}

struct HnswInner {
    hnsw: Hnsw<'static, f32, DistanceFn>,
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

        let distance_fn = DistanceFn::new(metric);
        let mut hnsw = Hnsw::<'static, f32, DistanceFn>::new(
            config.m,
            config.max_elements,
            dimensions,
            config.ef_construction,
            distance_fn,
        );

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
                hnsw.insert_data(vector, *idx);
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

        // ef_search < k causes hnsw_rs to return fewer than k candidates; clamp.
        let effective_ef = ef_search.max(k);

        // Use HNSW search with custom ef parameter (knbn)
        let neighbors = inner.hnsw.search(query, k, effective_ef);

        let mut results: Vec<SearchResult> = neighbors
            .into_iter()
            .filter_map(|neighbor| {
                inner.idx_to_id.get(&neighbor.d_id).map(|id| SearchResult {
                    id: id.clone(),
                    score: neighbor.distance,
                    vector: None,
                    metadata: None,
                })
            })
            .collect();

        // hnsw_rs does not guarantee sort order — ensure ascending distance.
        results.sort_unstable_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

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

        // Insert into HNSW graph using insert_data
        inner.hnsw.insert_data(&vector, idx);

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

        // Prepare batch data for insertion
        // First, assign indices and collect vector data
        let data_with_ids: Vec<_> = entries
            .iter()
            .enumerate()
            .map(|(i, (id, vector))| {
                let idx = inner.next_idx + i;
                (id.clone(), idx, vector.clone())
            })
            .collect();

        // Update next_idx
        inner.next_idx += entries.len();

        // For large batches (>=PARALLEL_THRESHOLD), use hnsw_rs parallel
        // insert (rayon-based) to cut build time.  Below this threshold,
        // sequential insert maintains better graph connectivity — parallel
        // workers can miss each other's in-flight inserts, producing fewer
        // optimal neighbors and increasing search latency on small indexes.
        //
        // Rule of thumb from hnsw_rs: parallel is efficient only when
        // n_inserts >= 1000 * num_threads.  We conservatively gate at 10 K.
        const PARALLEL_THRESHOLD: usize = 10_000;
        if data_with_ids.len() >= PARALLEL_THRESHOLD {
            let datas: Vec<(&[f32], usize)> = data_with_ids
                .iter()
                .map(|(_id, idx, vector)| (vector.as_slice(), *idx))
                .collect();
            inner.hnsw.parallel_insert_slice(&datas);
        } else {
            for (_id, idx, vector) in &data_with_ids {
                inner.hnsw.insert_data(vector, *idx);
            }
        }

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
        HnswIndex::search_with_ef(self, query, k, self.config.ef_search)
    }

    fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<SearchResult>> {
        HnswIndex::search_with_ef(self, query, k, ef_search)
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

    /// Deterministic unit vector, mirroring the generator used by the
    /// `hnsw_completeness_test` integration tests.
    fn seeded_unit_vector(seed: u64, dims: usize) -> Vec<f32> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut v = Vec::with_capacity(dims);
        for _ in 0..dims {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            let bits = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
            v.push((bits >> 40) as f32 / 8_388_608.0 - 1.0);
        }
        normalize_vector(&v)
    }

    /// Walk the layer-0 graph from the entry point and return the set of
    /// external ids reachable, plus the total number of points in the graph.
    ///
    /// The entry point is the first point stored in the highest occupied
    /// layer: `check_entry_point` only replaces the entry point on a strictly
    /// greater level, and `points_by_layer` preserves insertion order.
    fn layer0_reachable(index: &HnswIndex) -> (std::collections::HashSet<usize>, usize) {
        use std::collections::HashSet;

        let inner = index.inner.read();
        let indexation = inner.hnsw.get_point_indexation();
        let max_level = indexation.get_max_level_observed() as usize;
        let total = indexation.get_nb_point();

        let entry = indexation
            .get_layer_iterator(max_level)
            .next()
            .expect("entry point");

        // Map point id -> (origin id, layer-0 neighbour point ids) for the
        // whole graph so the walk can follow edges without re-locking.
        let mut adjacency = std::collections::HashMap::new();
        for level in 0..=max_level {
            for point in indexation.get_layer_iterator(level) {
                let neighbours = point.get_neighborhood_id();
                let layer0: Vec<_> = neighbours
                    .first()
                    .map(|n| n.iter().map(|nb| nb.p_id).collect())
                    .unwrap_or_default();
                adjacency.insert(point.get_point_id(), (point.get_origin_id(), layer0));
            }
        }

        let mut reachable = HashSet::new();
        let mut stack = vec![entry.get_point_id()];
        let mut visited = HashSet::new();
        while let Some(p_id) = stack.pop() {
            if !visited.insert(p_id) {
                continue;
            }
            if let Some((origin, neighbours)) = adjacency.get(&p_id) {
                reachable.insert(*origin);
                stack.extend(neighbours.iter().copied());
            }
        }

        (reachable, total)
    }

    /// Every inserted point must be reachable from the entry point by
    /// following layer-0 edges. A point with no layer-0 in-edge is an orphan:
    /// no `efSearch` can recover it, because search widens the frontier but
    /// never reaches a node nothing points at (issue #773).
    #[test]
    fn test_hnsw_layer0_reachability_invariant() -> Result<()> {
        let config = HnswConfig {
            m: 16,
            ef_construction: 100,
            ef_search: 100,
            max_elements: 1_000,
        };

        // The failure is level-assignment dependent and hits a fraction of a
        // percent of small graphs, so the sweep has to be wide enough that a
        // regression cannot slip through on luck.
        for trial in 0..1_500u64 {
            for &rows in &[2usize, 3, 5, 9] {
                let mut index = HnswIndex::new(64, DistanceMetric::Cosine, config.clone())?;
                for i in 0..rows {
                    index.add(
                        format!("m{i}"),
                        seeded_unit_vector(trial * 1_000 + i as u64, 64),
                    )?;
                }

                let (reachable, total) = layer0_reachable(&index);
                assert_eq!(total, rows, "trial {trial}: point count drifted");
                assert_eq!(
                    reachable.len(),
                    rows,
                    "trial {trial} rows={rows}: only {} of {rows} points are reachable \
                     from the entry point on layer 0 (orphaned origin ids: {:?})",
                    reachable.len(),
                    (0..rows)
                        .filter(|i| !reachable.contains(i))
                        .collect::<Vec<_>>()
                );
            }
        }

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
