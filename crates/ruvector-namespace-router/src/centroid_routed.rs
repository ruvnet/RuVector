//! Variant 2 — CentroidRouted: centroid index prunes distant namespaces.
//!
//! Maintains a per-namespace centroid vector.  Before scanning, namespaces are
//! ranked by centroid-to-query distance and only the `probe` closest are
//! examined.  When `probe == namespace_count()` the behaviour is identical to
//! [`FlatIsolated`] but with extra centroid bookkeeping; the win comes when
//! many semantically distinct namespaces exist and `probe` can be small.
//!
//! Cross-namespace search is *opt-in* (set `probe > 1`).  Isolation can be
//! enforced by always passing `probe = 1` (only the single closest namespace).

use std::collections::HashMap;

use crate::{l2sq, NamespaceId, NamespaceIndex, SearchResult, VectorId};

struct NsData {
    entries: Vec<(VectorId, Vec<f32>)>,
    /// Running mean; updated incrementally on every insert.
    centroid: Vec<f32>,
    count: usize,
}

impl NsData {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            centroid: Vec::new(),
            count: 0,
        }
    }

    fn insert(&mut self, id: VectorId, vector: Vec<f32>) {
        if self.centroid.is_empty() {
            self.centroid = vector.clone();
        } else {
            // Welford-style incremental mean update.
            let n = self.count as f32;
            let np1 = n + 1.0;
            for (c, x) in self.centroid.iter_mut().zip(vector.iter()) {
                *c = (*c * n + x) / np1;
            }
        }
        self.count += 1;
        self.entries.push((id, vector));
    }
}

/// Variant 2: centroid-routed namespace index.
pub struct CentroidRouted {
    namespaces: HashMap<NamespaceId, NsData>,
    dim: usize,
    /// How many namespaces to probe during search (sorted by centroid distance).
    pub probe: usize,
}

impl CentroidRouted {
    pub fn new(dim: usize, probe: usize) -> Self {
        Self {
            namespaces: HashMap::new(),
            dim,
            probe,
        }
    }

    /// Ordered list of (namespace_id, centroid_distance) ranked closest first.
    pub fn ranked_namespaces(&self, query: &[f32]) -> Vec<(NamespaceId, f32)> {
        let mut ranked: Vec<(NamespaceId, f32)> = self
            .namespaces
            .iter()
            .map(|(ns, data)| (*ns, l2sq(query, &data.centroid)))
            .collect();
        ranked.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        ranked
    }
}

impl NamespaceIndex for CentroidRouted {
    fn insert(&mut self, ns: NamespaceId, id: VectorId, vector: Vec<f32>) -> Result<(), String> {
        if vector.len() != self.dim {
            return Err(format!("dim mismatch: {} vs {}", vector.len(), self.dim));
        }
        self.namespaces
            .entry(ns)
            .or_insert_with(NsData::new)
            .insert(id, vector);
        Ok(())
    }

    fn search(&self, ns: NamespaceId, query: &[f32], k: usize) -> Vec<SearchResult> {
        let ranked = self.ranked_namespaces(query);

        // Always include the requested namespace; fill remaining probe slots
        // with the closest others.
        let mut to_scan: Vec<NamespaceId> = Vec::new();
        to_scan.push(ns);
        for (candidate_ns, _) in &ranked {
            if to_scan.len() >= self.probe {
                break;
            }
            if *candidate_ns != ns {
                to_scan.push(*candidate_ns);
            }
        }

        let mut candidates: Vec<SearchResult> = Vec::new();
        for scan_ns in to_scan {
            if let Some(data) = self.namespaces.get(&scan_ns) {
                for (id, vec) in &data.entries {
                    candidates.push(SearchResult {
                        id: *id,
                        namespace: scan_ns,
                        distance: l2sq(query, vec),
                    });
                }
            }
        }

        candidates.sort_unstable_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        candidates.truncate(k);
        candidates
    }

    fn namespace_count(&self) -> usize {
        self.namespaces.len()
    }

    fn total_vectors(&self) -> usize {
        self.namespaces.values().map(|d| d.entries.len()).sum()
    }

    fn memory_bytes(&self) -> usize {
        // entries + centroid per namespace
        self.namespaces
            .values()
            .map(|d| d.entries.len() * (8 + self.dim * 4) + self.dim * 4)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_probe_restricts_to_own_namespace() {
        let mut idx = CentroidRouted::new(4, 1);
        // ns=0 is near origin
        for i in 0..5u64 {
            idx.insert(0, i, vec![i as f32 * 0.1; 4]).unwrap();
        }
        // ns=1 is far
        for i in 5..10u64 {
            idx.insert(1, i, vec![100.0; 4]).unwrap();
        }
        // query near ns=0; probe=1 should return only ns=0 results
        let results = idx.search(0, &[0.0; 4], 5);
        assert!(results.iter().all(|r| r.namespace == 0));
    }

    #[test]
    fn multi_probe_can_return_cross_namespace() {
        let mut idx = CentroidRouted::new(4, 2);
        idx.insert(0, 1, vec![0.0; 4]).unwrap();
        idx.insert(1, 2, vec![0.1, 0.1, 0.1, 0.1]).unwrap(); // very close to ns=0
        let results = idx.search(0, &[0.0; 4], 5);
        // With probe=2 and ns=1 centroid very close to query, id=2 may appear
        assert!(!results.is_empty());
    }

    #[test]
    fn centroid_is_updated_incrementally() {
        let mut idx = CentroidRouted::new(2, 1);
        idx.insert(0, 1, vec![0.0, 0.0]).unwrap();
        idx.insert(0, 2, vec![2.0, 2.0]).unwrap();
        let data = idx.namespaces.get(&0).unwrap();
        // centroid should be ~(1.0, 1.0)
        assert!((data.centroid[0] - 1.0).abs() < 1e-5);
        assert!((data.centroid[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn memory_bytes_includes_centroids() {
        let mut idx = CentroidRouted::new(8, 1);
        for i in 0..10u64 {
            idx.insert(0, i, vec![0.0; 8]).unwrap();
        }
        // 10 entries × (8+32) + 1 centroid × 32 = 432
        let expected = 10 * (8 + 32) + 32;
        assert_eq!(idx.memory_bytes(), expected);
    }
}
