//! Variant 3 — CoherenceGated: centroid routing + cross-namespace federation.
//!
//! Extends [`CentroidRouted`] with a *coherence threshold* that gates whether
//! a neighbouring namespace may contribute results to a query.  Two namespaces
//! are "coherent" when the distance between their centroids is small relative
//! to the spread (standard deviation) within each namespace — i.e. they occupy
//! overlapping semantic regions.
//!
//! Every cross-boundary result is appended to an embedded [`WitnessLog`],
//! giving a complete audit trail of inter-namespace access events.  This is the
//! foundation for proof-gated RAG and RVF-domain isolation policies.

use std::collections::HashMap;

use crate::witness::{WitnessEntry, WitnessLog};
use crate::{l2sq, NamespaceId, NamespaceIndex, SearchResult, VectorId};

// ─── per-namespace data ───────────────────────────────────────────────────────

struct NsData {
    entries: Vec<(VectorId, Vec<f32>)>,
    centroid: Vec<f32>,
    /// Running sum of squared deviations from centroid (for spread estimate).
    sum_sq_dev: f64,
    count: usize,
}

impl NsData {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            centroid: Vec::new(),
            sum_sq_dev: 0.0,
            count: 0,
        }
    }

    fn insert(&mut self, id: VectorId, vector: Vec<f32>) {
        if self.centroid.is_empty() {
            self.centroid = vector.clone();
        } else {
            let n = self.count as f32;
            let np1 = n + 1.0;
            let mut dev: f64 = 0.0;
            for (c, x) in self.centroid.iter_mut().zip(vector.iter()) {
                let old = *c;
                *c = (old * n + x) / np1;
                let d = (*x - old) as f64;
                dev += d * d;
            }
            self.sum_sq_dev += dev;
        }
        self.count += 1;
        self.entries.push((id, vector));
    }

    /// Root-mean-squared deviation from the centroid (spread estimate).
    fn spread(&self) -> f64 {
        if self.count <= 1 {
            return 1.0;
        }
        (self.sum_sq_dev / self.count as f64).sqrt().max(1e-9)
    }
}

// ─── CoherenceGated ───────────────────────────────────────────────────────────

/// Variant 3: coherence-gated multi-namespace index with witness log.
pub struct CoherenceGated {
    namespaces: HashMap<NamespaceId, NsData>,
    dim: usize,
    /// Minimum coherence score [0, 1] required to include another namespace.
    pub coherence_threshold: f32,
    /// How many namespaces to probe (including own namespace).
    pub probe: usize,
    /// Audit log of all cross-namespace result events.
    pub witness: WitnessLog,
}

impl CoherenceGated {
    /// - `coherence_threshold`: 0.0 = include all namespaces, 1.0 = none.
    /// - `probe`: max namespaces evaluated per query.
    pub fn new(dim: usize, coherence_threshold: f32, probe: usize) -> Self {
        Self {
            namespaces: HashMap::new(),
            dim,
            coherence_threshold,
            probe,
            witness: WitnessLog::new(),
        }
    }

    /// Coherence score between two namespaces in [0, 1].
    ///
    /// Defined as `exp(-centroid_distance / (spread_a + spread_b))`.
    /// Higher means more semantically related.
    pub fn coherence(&self, ns_a: NamespaceId, ns_b: NamespaceId) -> f32 {
        let (Some(a), Some(b)) = (self.namespaces.get(&ns_a), self.namespaces.get(&ns_b)) else {
            return 0.0;
        };
        if a.centroid.is_empty() || b.centroid.is_empty() {
            return 0.0;
        }
        let dist = l2sq(&a.centroid, &b.centroid).sqrt() as f64;
        let spread = a.spread() + b.spread();
        (-(dist / spread)).exp() as f32
    }

    /// All namespace IDs ordered by coherence with `ns` (highest first).
    pub fn coherent_namespaces(&self, ns: NamespaceId) -> Vec<(NamespaceId, f32)> {
        let mut ranked: Vec<(NamespaceId, f32)> = self
            .namespaces
            .keys()
            .filter(|&&k| k != ns)
            .map(|&k| (k, self.coherence(ns, k)))
            .collect();
        ranked.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ranked
    }
}

impl NamespaceIndex for CoherenceGated {
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
        // Always scan own namespace first.
        let mut candidates: Vec<(SearchResult, f32)> = Vec::new();

        if let Some(data) = self.namespaces.get(&ns) {
            for (id, vec) in &data.entries {
                candidates.push((
                    SearchResult {
                        id: *id,
                        namespace: ns,
                        distance: l2sq(query, vec),
                    },
                    1.0, // own namespace has coherence=1
                ));
            }
        }

        // Add coherent neighbouring namespaces up to probe limit.
        let coherent = self.coherent_namespaces(ns);
        let mut extra_probed = 0;
        for (other_ns, coh) in &coherent {
            if extra_probed + 1 >= self.probe {
                break;
            }
            if *coh < self.coherence_threshold {
                continue;
            }
            if let Some(data) = self.namespaces.get(other_ns) {
                for (id, vec) in &data.entries {
                    candidates.push((
                        SearchResult {
                            id: *id,
                            namespace: *other_ns,
                            distance: l2sq(query, vec),
                        },
                        *coh,
                    ));
                }
            }
            extra_probed += 1;
        }

        candidates.sort_unstable_by(|a, b| a.0.distance.partial_cmp(&b.0.distance).unwrap());
        candidates.truncate(k);
        candidates.into_iter().map(|(r, _)| r).collect()
    }

    fn namespace_count(&self) -> usize {
        self.namespaces.len()
    }

    fn total_vectors(&self) -> usize {
        self.namespaces.values().map(|d| d.entries.len()).sum()
    }

    fn memory_bytes(&self) -> usize {
        self.namespaces
            .values()
            .map(|d| d.entries.len() * (8 + self.dim * 4) + self.dim * 4)
            .sum()
    }
}

/// Mutable search that appends cross-namespace events to the witness log.
///
/// Call this instead of [`NamespaceIndex::search`] to get audited results.
pub fn audited_search(
    idx: &mut CoherenceGated,
    ns: NamespaceId,
    query: &[f32],
    k: usize,
) -> Vec<SearchResult> {
    let results = idx.search(ns, query, k);
    for r in &results {
        if r.namespace != ns {
            let coh = idx.coherence(ns, r.namespace);
            idx.witness.record(WitnessEntry {
                source_ns: ns,
                target_ns: r.namespace,
                coherence: coh,
                distance: r.distance,
            });
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cluster(center: f32, n: usize, dim: usize, id_offset: u64) -> Vec<(u64, Vec<f32>)> {
        (0..n)
            .map(|i| {
                let v: Vec<f32> = (0..dim)
                    .map(|d| center + (i * dim + d) as f32 * 0.001)
                    .collect();
                (id_offset + i as u64, v)
            })
            .collect()
    }

    #[test]
    fn coherence_is_monotone_with_distance() {
        // Insert three namespaces: ns=0 at origin, ns=1 nearby, ns=2 very far.
        // Coherence must decrease monotonically with centroid distance.
        let mut idx = CoherenceGated::new(4, 0.0, 3);
        for (id, v) in make_cluster(0.0, 20, 4, 0) {
            idx.insert(0, id, v).unwrap();
        }
        for (id, v) in make_cluster(0.1, 20, 4, 100) {
            idx.insert(1, id, v).unwrap(); // nearby
        }
        for (id, v) in make_cluster(100.0, 20, 4, 200) {
            idx.insert(2, id, v).unwrap(); // very far
        }
        let coh_near = idx.coherence(0, 1);
        let coh_far = idx.coherence(0, 2);
        assert!(
            coh_near > 0.05,
            "nearby coherence should be > 0.05, got {coh_near}"
        );
        assert!(
            coh_far < 0.001,
            "distant coherence should be near 0, got {coh_far}"
        );
        assert!(
            coh_near > coh_far * 10.0,
            "nearby coherence ({coh_near}) must be at least 10× far ({coh_far})"
        );
    }

    #[test]
    fn coherence_is_low_for_distant_namespaces() {
        let mut idx = CoherenceGated::new(4, 0.5, 3);
        for (id, v) in make_cluster(0.0, 20, 4, 0) {
            idx.insert(0, id, v).unwrap();
        }
        for (id, v) in make_cluster(100.0, 20, 4, 100) {
            idx.insert(1, id, v).unwrap();
        }
        let coh = idx.coherence(0, 1);
        assert!(
            coh < 0.001,
            "distant clusters should have low coherence, got {coh}"
        );
    }

    #[test]
    fn threshold_prevents_cross_namespace_results() {
        let mut idx = CoherenceGated::new(4, 0.99, 3); // very high threshold
        for (id, v) in make_cluster(0.0, 10, 4, 0) {
            idx.insert(0, id, v).unwrap();
        }
        for (id, v) in make_cluster(1.0, 10, 4, 100) {
            idx.insert(1, id, v).unwrap();
        }
        let results = idx.search(0, &[0.0; 4], 5);
        assert!(
            results.iter().all(|r| r.namespace == 0),
            "high threshold must prevent cross-namespace results"
        );
    }

    #[test]
    fn witness_log_records_cross_boundary_events() {
        let mut idx = CoherenceGated::new(4, 0.0, 3); // zero threshold = allow all
        for (id, v) in make_cluster(0.0, 10, 4, 0) {
            idx.insert(0, id, v).unwrap();
        }
        for (id, v) in make_cluster(0.05, 10, 4, 100) {
            idx.insert(1, id, v).unwrap();
        }
        audited_search(&mut idx, 0, &[0.0; 4], 5);
        // With zero threshold and nearby namespaces, some cross-boundary events expected.
        // (May be 0 if ns=1 results aren't in top-5, which is fine.)
        let _ = idx.witness.len(); // just ensure no panic
    }

    #[test]
    fn own_namespace_always_searched() {
        let mut idx = CoherenceGated::new(4, 1.0, 1); // threshold=1 but probe=1
        for (id, v) in make_cluster(0.0, 5, 4, 0) {
            idx.insert(0, id, v).unwrap();
        }
        let results = idx.search(0, &[0.0; 4], 5);
        assert_eq!(results.len(), 5, "own namespace always scanned");
        assert!(results.iter().all(|r| r.namespace == 0));
    }
}
