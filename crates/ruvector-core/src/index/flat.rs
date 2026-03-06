//! Flat (brute-force) index for baseline and small datasets

use crate::error::Result;
use crate::index::VectorIndex;
use crate::types::{DistanceMetric, QuantumVector, SearchResult, VectorId};
use dashmap::DashMap;

#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
use rayon::prelude::*;

/// Flat index using brute-force search
pub struct FlatIndex {
    vectors: DashMap<VectorId, QuantumVector>,
    metric: DistanceMetric,
    _dimensions: usize,
}

impl FlatIndex {
    /// Create a new flat index
    pub fn new(dimensions: usize, metric: DistanceMetric) -> Self {
        Self {
            vectors: DashMap::new(),
            metric,
            _dimensions: dimensions,
        }
    }
}

impl VectorIndex for FlatIndex {
    fn add(&mut self, id: VectorId, vector: QuantumVector) -> Result<()> {
        self.vectors.insert(id, vector);
        Ok(())
    }

    fn search(&self, query: &QuantumVector, k: usize) -> Result<Vec<SearchResult>> {
        let query_f32 = query.reconstruct();

        // Distance calculation - parallel on native, sequential on WASM
        #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
        let mut results: Vec<_> = self
            .vectors
            .iter()
            .par_bridge()
            .map(|entry| {
                let id = entry.key().clone();
                let vector_f32 = entry.value().reconstruct();
                let dist = crate::distance::distance(&query_f32, &vector_f32, self.metric)?;
                Ok((id, dist))
            })
            .collect::<Result<Vec<_>>>()?;

        #[cfg(any(not(feature = "parallel"), target_arch = "wasm32"))]
        let mut results: Vec<_> = self
            .vectors
            .iter()
            .map(|entry| {
                let id = entry.key().clone();
                let vector_f32 = entry.value().reconstruct();
                let dist = crate::distance::distance(&query_f32, &vector_f32, self.metric)?;
                Ok((id, dist))
            })
            .collect::<Result<Vec<_>>>()?;

        // Sort by distance and take top k
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);

        Ok(results
            .into_iter()
            .map(|(id, score)| SearchResult {
                id,
                score,
                vector: None,
                metadata: None,
            })
            .collect())
    }

    fn remove(&mut self, id: &VectorId) -> Result<bool> {
        Ok(self.vectors.remove(id).is_some())
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_index() -> Result<()> {
        let mut index = FlatIndex::new(3, DistanceMetric::Euclidean);

        index.add("v1".to_string(), QuantumVector::F32(vec![1.0, 0.0, 0.0]))?;
        index.add("v2".to_string(), QuantumVector::F32(vec![0.0, 1.0, 0.0]))?;
        index.add("v3".to_string(), QuantumVector::F32(vec![0.0, 0.0, 1.0]))?;

        let query = QuantumVector::F32(vec![1.0, 0.0, 0.0]);
        let results = index.search(&query, 2)?;

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "v1");
        assert!(results[0].score < 0.01);

        Ok(())
    }
}
