//! Variant 1 — FlatIsolated: strict per-namespace linear scan.
//!
//! Each namespace owns an independent `Vec`. Search is a full linear scan over
//! only that namespace's vectors. There is zero cross-namespace visibility by
//! design; no centroid index, no coherence computation.

use std::collections::HashMap;

use crate::{l2sq, NamespaceId, NamespaceIndex, SearchResult, VectorId};

/// Entry stored per namespace.
struct Entry {
    id: VectorId,
    vector: Vec<f32>,
}

/// Variant 1: isolated per-namespace flat index.
pub struct FlatIsolated {
    namespaces: HashMap<NamespaceId, Vec<Entry>>,
    dim: usize,
}

impl FlatIsolated {
    pub fn new(dim: usize) -> Self {
        Self {
            namespaces: HashMap::new(),
            dim,
        }
    }
}

impl NamespaceIndex for FlatIsolated {
    fn insert(&mut self, ns: NamespaceId, id: VectorId, vector: Vec<f32>) -> Result<(), String> {
        if vector.len() != self.dim {
            return Err(format!("dim mismatch: {} vs {}", vector.len(), self.dim));
        }
        self.namespaces
            .entry(ns)
            .or_default()
            .push(Entry { id, vector });
        Ok(())
    }

    fn search(&self, ns: NamespaceId, query: &[f32], k: usize) -> Vec<SearchResult> {
        let Some(entries) = self.namespaces.get(&ns) else {
            return Vec::new();
        };
        let mut dists: Vec<(VectorId, f32)> = entries
            .iter()
            .map(|e| (e.id, l2sq(query, &e.vector)))
            .collect();
        dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        dists
            .into_iter()
            .take(k)
            .map(|(id, dist)| SearchResult {
                id,
                namespace: ns,
                distance: dist,
            })
            .collect()
    }

    fn namespace_count(&self) -> usize {
        self.namespaces.len()
    }

    fn total_vectors(&self) -> usize {
        self.namespaces.values().map(|v| v.len()).sum()
    }

    fn memory_bytes(&self) -> usize {
        self.namespaces
            .values()
            .map(|v| v.len() * (8 + self.dim * 4))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vec(dim: usize, seed: f32) -> Vec<f32> {
        (0..dim).map(|i| seed + i as f32 * 0.01).collect()
    }

    #[test]
    fn insert_and_search_returns_nearest() {
        let mut idx = FlatIsolated::new(4);
        idx.insert(0, 1, vec![0.0, 0.0, 0.0, 0.0]).unwrap();
        idx.insert(0, 2, vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        idx.insert(0, 3, vec![10.0, 10.0, 10.0, 10.0]).unwrap();
        let results = idx.search(0, &[0.1, 0.1, 0.1, 0.1], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 1, "nearest should be id=1");
    }

    #[test]
    fn namespace_isolation() {
        let mut idx = FlatIsolated::new(4);
        idx.insert(0, 10, vec![0.0; 4]).unwrap();
        idx.insert(1, 20, vec![100.0; 4]).unwrap();
        // query ns=1; must not return vectors from ns=0
        let results = idx.search(1, &[0.0; 4], 5);
        assert!(results.iter().all(|r| r.namespace == 1));
        assert_eq!(results[0].id, 20);
    }

    #[test]
    fn empty_namespace_returns_empty() {
        let idx = FlatIsolated::new(4);
        assert!(idx.search(99, &[0.0; 4], 5).is_empty());
    }

    #[test]
    fn dim_mismatch_is_error() {
        let mut idx = FlatIsolated::new(4);
        assert!(idx.insert(0, 1, vec![1.0; 5]).is_err());
    }

    #[test]
    fn memory_bytes_scales_linearly() {
        let mut idx = FlatIsolated::new(8);
        for i in 0..100u64 {
            idx.insert(0, i, make_vec(8, i as f32)).unwrap();
        }
        let expected = 100 * (8 + 8 * 4); // 100 × 40 bytes
        assert_eq!(idx.memory_bytes(), expected);
    }
}
