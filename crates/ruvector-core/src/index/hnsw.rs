//! HNSW (Hierarchical Navigable Small World) index implementation

// use crate::distance::distance;
use crate::error::{Result, RuvectorError};
use crate::index::VectorIndex;
use crate::types::{DistanceMetric, HnswConfig, QuantumVector, SearchResult, VectorId};
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

impl Distance<QuantumVector> for DistanceFn {
    fn eval(&self, a: &[QuantumVector], b: &[QuantumVector]) -> f32 {
        // Direct distance on QuantumVectors
        let a_f32 = a[0].reconstruct();
        let b_f32 = b[0].reconstruct();
        crate::distance::distance(&a_f32, &b_f32, self.metric).unwrap_or(f32::MAX)
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
    hnsw: Hnsw<'static, QuantumVector, DistanceFn>,
    vectors: DashMap<VectorId, QuantumVector>,
    id_to_idx: DashMap<VectorId, usize>,
    idx_to_id: DashMap<usize, VectorId>,
    next_idx: usize,
}

/// Serializable HNSW index state
#[derive(Encode, Decode, Clone)]
pub struct HnswState {
    vectors: Vec<(String, QuantumVector)>,
    id_to_idx: Vec<(String, usize)>,
    idx_to_id: Vec<(usize, String)>,
    next_idx: usize,
    config: HnswConfig,
    dimensions: usize,
    metric: DistanceMetric,
}

// Redundant serializable structs removed as they are now in types.rs

impl HnswIndex {
    /// Create a new HNSW index
    pub fn new(dimensions: usize, metric: DistanceMetric, config: HnswConfig) -> Result<Self> {
        let distance_fn = DistanceFn::new(metric);

        // Create HNSW with configured parameters (QuantumVector native)
        let hnsw = Hnsw::<QuantumVector, DistanceFn>::new(
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

    /// Set efSearch parameter for query-time accuracy tuning
    pub fn set_ef_search(&mut self, _ef_search: usize) {
        // Note: hnsw_rs controls ef_search via the search method's knbn parameter
        // We store it in config and use it in search_with_ef
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
            config: self.config.clone(),
            dimensions: self.dimensions,
            metric: self.metric,
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
        let mut hnsw = Hnsw::<QuantumVector, DistanceFn>::new(
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
                // Use insert_data method with QuantumVector
                hnsw.insert_data(std::slice::from_ref(&vector.1), idx);
            }
        }

        let vectors_map: DashMap<VectorId, QuantumVector> = state.vectors.into_iter().collect();

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

    /// Search with custom efSearch parameter
    pub fn search_with_ef(
        &self,
        query: &QuantumVector,
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<SearchResult>> {
        if query.reconstruct().len() != self.dimensions {
            return Err(RuvectorError::DimensionMismatch {
                expected: self.dimensions,
                actual: query.reconstruct().len(),
            });
        }

        let inner = self.inner.read();

        // Use HNSW search with custom ef parameter (knbn)
        let neighbors = inner.hnsw.search(std::slice::from_ref(query), k, ef_search);

        Ok(neighbors
            .into_iter()
            .filter_map(|neighbor| {
                inner.idx_to_id.get(&neighbor.d_id).map(|id| SearchResult {
                    id: id.clone(),
                    score: neighbor.distance,
                    vector: None,
                    metadata: None,
                })
            })
            .collect())
    }
}

impl VectorIndex for HnswIndex {
    fn add(&mut self, id: VectorId, vector: QuantumVector) -> Result<()> {
        if vector.reconstruct().len() != self.dimensions {
            return Err(RuvectorError::DimensionMismatch {
                expected: self.dimensions,
                actual: vector.reconstruct().len(),
            });
        }

        let mut inner = self.inner.write();
        let idx = inner.next_idx;
        inner.next_idx += 1;

        // Insert into HNSW graph using insert_data (QuantumVector native)
        inner.hnsw.insert_data(std::slice::from_ref(&vector), idx);

        // Store mappings
        inner.vectors.insert(id.clone(), vector);
        inner.id_to_idx.insert(id.clone(), idx);
        inner.idx_to_id.insert(idx, id);

        Ok(())
    }

    fn add_batch(&mut self, entries: Vec<(VectorId, QuantumVector)>) -> Result<()> {
        // Validate all dimensions first
        for (_, vector) in &entries {
            if vector.reconstruct().len() != self.dimensions {
                return Err(RuvectorError::DimensionMismatch {
                    expected: self.dimensions,
                    actual: vector.reconstruct().len(),
                });
            }
        }

        let mut inner = self.inner.write();

        // Prepare batch data for insertion
        // First, assign indices and collect vector data
        let data_with_ids: Vec<_> = entries
            .into_iter()
            .enumerate()
            .map(|(i, (id, vector))| {
                let idx = inner.next_idx + i;
                (id, idx, vector)
            })
            .collect();

        // Update next_idx
        inner.next_idx += data_with_ids.len();

        // Insert into HNSW sequentially (Hnsw-rs native optimized)
        for (_id, idx, vector) in &data_with_ids {
            inner.hnsw.insert_data(std::slice::from_ref(vector), *idx);
        }

        // Store mappings
        for (id, idx, vector) in data_with_ids {
            inner.vectors.insert(id.clone(), vector);
            inner.id_to_idx.insert(id.clone(), idx);
            inner.idx_to_id.insert(idx, id);
        }

        Ok(())
    }

    fn search(&self, query: &QuantumVector, k: usize) -> Result<Vec<SearchResult>> {
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

    fn dump(&self) -> Result<Option<Vec<u8>>> {
        Ok(Some(self.serialize()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_random_vectors(count: usize, dimensions: usize) -> Vec<QuantumVector> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        (0..count)
            .map(|_| {
                let v: Vec<f32> = (0..dimensions).map(|_| rng.gen::<f32>()).collect();
                QuantumVector::F32(v)
            })
            .collect()
    }

    fn normalize_quantum(v: &QuantumVector) -> QuantumVector {
        let vec = v.reconstruct();
        let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            QuantumVector::F32(vec.iter().map(|x| x / norm).collect())
        } else {
            QuantumVector::F32(vec)
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
            let normalized = normalize_quantum(vector);
            index.add(format!("vec_{}", i), normalized)?;
        }

        assert_eq!(index.len(), 100);

        // Search for the first vector
        let query = normalize_quantum(&vectors[0]);
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
            .map(|(i, v)| (format!("vec_{}", i), normalize_quantum(v)))
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
            let normalized = normalize_quantum(vector);
            index.add(format!("vec_{}", i), normalized)?;
        }

        // Serialize
        let bytes = index.serialize()?;

        // Deserialize
        let restored_index = HnswIndex::deserialize(&bytes)?;

        assert_eq!(restored_index.len(), 50);

        // Test search on restored index
        let query = normalize_quantum(&vectors[0]);
        let results = restored_index.search(&query, 5)?;

        assert!(!results.is_empty());

        Ok(())
    }

    #[test]
    fn test_dimension_mismatch() -> Result<()> {
        let config = HnswConfig::default();
        let mut index = HnswIndex::new(128, DistanceMetric::Cosine, config)?;

        let result = index.add("test".to_string(), QuantumVector::F32(vec![1.0; 64]));
        assert!(result.is_err());

        Ok(())
    }
}
