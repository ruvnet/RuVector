/// Namespace-partitioned multi-agent HNSW memory for RuVector.
///
/// Agents in multi-agent systems each need an isolated memory space, yet they
/// also need to retrieve knowledge across namespace boundaries.  Three variants
/// benchmark the tradeoff:
///
/// | Variant          | Structure                             | Single-NS cost | Cross-NS cost |
/// |------------------|---------------------------------------|----------------|---------------|
/// | GlobalFlat       | one HNSW, namespace as metadata       | O(n_total)     | O(n_total)    |
/// | Partitioned      | one HNSW per namespace                | O(n_ns)        | O(n_ns × |NS|)|
/// | HierarchicalNS   | routing index + per-namespace HNSW    | O(n_ns)        | O(n_ns × k)  |
pub mod hnsw;

use hnsw::MiniHnsw;
use std::collections::HashMap;

// ── Common types ──────────────────────────────────────────────────────────────

/// A retrieved result from any namespace-aware index.
#[derive(Clone, Debug)]
pub struct NsResult {
    pub id: u64,
    pub namespace: String,
    pub dist_sq: f32,
}

/// Unified trait for all three namespace-partitioned backends.
pub trait NamespacedIndex {
    /// Insert a vector under the given namespace.
    fn insert(&mut self, ns: &str, id: u64, vector: Vec<f32>);

    /// Search within a single namespace.
    fn search_single(&self, ns: &str, query: &[f32], k: usize, ef: usize) -> Vec<NsResult>;

    /// Search across all registered namespaces.
    fn search_cross(&self, query: &[f32], k: usize, ef: usize) -> Vec<NsResult>;

    /// Human-readable variant name.
    fn name(&self) -> &'static str;

    /// Approximate memory footprint in bytes.
    fn memory_bytes(&self) -> usize;
}

/// Compute recall@k: fraction of oracle IDs present in candidate set.
pub fn recall_at_k(oracle: &[NsResult], candidates: &[NsResult], k: usize) -> f32 {
    let oracle_ids: std::collections::HashSet<u64> = oracle.iter().take(k).map(|r| r.id).collect();
    let hits = candidates
        .iter()
        .take(k)
        .filter(|r| oracle_ids.contains(&r.id))
        .count();
    let denom = oracle.len().min(k).max(1);
    hits as f32 / denom as f32
}

// ── Variant 1: GlobalFlat ────────────────────────────────────────────────────

/// All vectors stored in one HNSW.  Namespace filtering happens post-search.
///
/// Simple to implement, but wastes ef budget on foreign-namespace vectors.
/// Cross-namespace search is free (no merge needed).
pub struct GlobalFlat {
    index: MiniHnsw,
    ns_labels: Vec<String>, // parallel to index.nodes
}

impl GlobalFlat {
    pub fn new(m: usize, ef_construction: usize) -> Self {
        Self {
            index: MiniHnsw::new(m, ef_construction, 0xdead_beef),
            ns_labels: Vec::new(),
        }
    }
}

impl NamespacedIndex for GlobalFlat {
    fn insert(&mut self, ns: &str, id: u64, vector: Vec<f32>) {
        self.index.insert(id, vector);
        self.ns_labels.push(ns.to_owned());
    }

    fn search_single(&self, ns: &str, query: &[f32], k: usize, ef: usize) -> Vec<NsResult> {
        // Search broad ef to compensate for post-filter discard
        let broad_ef = (ef * 4).min(self.index.len().max(1));
        let raw = self.index.search(query, self.index.len(), broad_ef);
        raw.into_iter()
            .filter(|&(_, id)| {
                // Recover namespace via insertion order
                self.ns_labels
                    .get(id as usize)
                    .map(|n| n == ns)
                    .unwrap_or(false)
            })
            .take(k)
            .map(|(d, id)| NsResult {
                id,
                namespace: ns.to_owned(),
                dist_sq: d,
            })
            .collect()
    }

    fn search_cross(&self, query: &[f32], k: usize, ef: usize) -> Vec<NsResult> {
        let raw = self.index.search(query, k, ef);
        raw.into_iter()
            .map(|(d, id)| NsResult {
                id,
                namespace: self.ns_labels.get(id as usize).cloned().unwrap_or_default(),
                dist_sq: d,
            })
            .collect()
    }

    fn name(&self) -> &'static str {
        "GlobalFlat"
    }

    fn memory_bytes(&self) -> usize {
        self.index.memory_bytes() + self.ns_labels.iter().map(|s| s.len() + 24).sum::<usize>()
    }
}

// ── Variant 2: Partitioned ───────────────────────────────────────────────────

/// One HNSW per namespace.  Single-namespace search is focused; cross-namespace
/// requires sequential search + merge.
pub struct Partitioned {
    indexes: HashMap<String, MiniHnsw>,
    m: usize,
    ef_construction: usize,
    seed_counter: u64,
}

impl Partitioned {
    pub fn new(m: usize, ef_construction: usize) -> Self {
        Self {
            indexes: HashMap::new(),
            m,
            ef_construction,
            seed_counter: 0,
        }
    }

    fn get_or_create(&mut self, ns: &str) -> &mut MiniHnsw {
        let m = self.m;
        let ef = self.ef_construction;
        let seed = self.seed_counter;
        self.seed_counter = self.seed_counter.wrapping_add(0x9e37_79b9);
        self.indexes
            .entry(ns.to_owned())
            .or_insert_with(|| MiniHnsw::new(m, ef, seed))
    }
}

impl NamespacedIndex for Partitioned {
    fn insert(&mut self, ns: &str, id: u64, vector: Vec<f32>) {
        self.get_or_create(ns).insert(id, vector);
    }

    fn search_single(&self, ns: &str, query: &[f32], k: usize, ef: usize) -> Vec<NsResult> {
        let Some(idx) = self.indexes.get(ns) else {
            return Vec::new();
        };
        idx.search(query, k, ef)
            .into_iter()
            .map(|(d, id)| NsResult {
                id,
                namespace: ns.to_owned(),
                dist_sq: d,
            })
            .collect()
    }

    fn search_cross(&self, query: &[f32], k: usize, ef: usize) -> Vec<NsResult> {
        // Sequential search + merge-sort
        let mut all: Vec<NsResult> = self
            .indexes
            .iter()
            .flat_map(|(ns, idx)| {
                idx.search(query, k, ef)
                    .into_iter()
                    .map(|(d, id)| NsResult {
                        id,
                        namespace: ns.clone(),
                        dist_sq: d,
                    })
            })
            .collect();
        all.sort_by(|a, b| a.dist_sq.partial_cmp(&b.dist_sq).unwrap());
        all.truncate(k);
        all
    }

    fn name(&self) -> &'static str {
        "Partitioned"
    }

    fn memory_bytes(&self) -> usize {
        self.indexes
            .values()
            .map(|i| i.memory_bytes())
            .sum::<usize>()
            + self.indexes.keys().map(|k| k.len() + 24).sum::<usize>()
    }
}

// ── Variant 3: HierarchicalNS ────────────────────────────────────────────────

/// Per-namespace HNSW + a small routing index of namespace centroid vectors.
///
/// For cross-namespace search, the router finds the top-R namespaces whose
/// centroids are nearest to the query, then only those namespaces are probed.
/// This avoids scanning every namespace when only a subset is relevant.
pub struct HierarchicalNS {
    indexes: HashMap<String, MiniHnsw>,
    /// Each namespace centroid is stored here; node id = namespace slot index.
    router: MiniHnsw,
    /// Map from router node id back to namespace name.
    router_ns: Vec<String>,
    /// Running centroid sums per namespace for online update.
    centroid_sums: HashMap<String, Vec<f64>>,
    centroid_counts: HashMap<String, u64>,
    dims: usize,
    m: usize,
    ef_construction: usize,
    seed_counter: u64,
    /// How many namespaces to probe during cross-namespace search.
    route_k: usize,
}

impl HierarchicalNS {
    pub fn new(m: usize, ef_construction: usize, dims: usize, route_k: usize) -> Self {
        Self {
            indexes: HashMap::new(),
            router: MiniHnsw::new(m, ef_construction, 0xcafe_babe),
            router_ns: Vec::new(),
            centroid_sums: HashMap::new(),
            centroid_counts: HashMap::new(),
            dims,
            m,
            ef_construction,
            seed_counter: 0x1234_5678,
            route_k,
        }
    }

    /// Rebuild or update the router centroid for a namespace.
    fn update_router(&mut self, ns: &str) {
        let count = *self.centroid_counts.get(ns).unwrap_or(&0);
        if count == 0 || self.dims == 0 {
            return;
        }
        let sums = self.centroid_sums.get(ns).unwrap();
        let centroid: Vec<f32> = sums.iter().map(|&s| (s / count as f64) as f32).collect();

        // Find if ns is already in router
        if let Some(slot) = self.router_ns.iter().position(|n| n == ns) {
            // Update in-place: rebuild router node's vector.
            // For simplicity we replace the vector directly (no re-index needed
            // since centroid drift is small and we only use this for routing).
            self.router.nodes[slot].vector = centroid;
        } else {
            let slot = self.router_ns.len() as u64;
            self.router.insert(slot, centroid);
            self.router_ns.push(ns.to_owned());
        }
    }

    fn get_or_create_ns(&mut self, ns: &str) -> &mut MiniHnsw {
        let m = self.m;
        let ef = self.ef_construction;
        let seed = self.seed_counter;
        self.seed_counter = self.seed_counter.wrapping_add(0x9e37_79b9);
        self.indexes
            .entry(ns.to_owned())
            .or_insert_with(|| MiniHnsw::new(m, ef, seed))
    }
}

impl NamespacedIndex for HierarchicalNS {
    fn insert(&mut self, ns: &str, id: u64, vector: Vec<f32>) {
        // Update centroid tracking
        let sums = self
            .centroid_sums
            .entry(ns.to_owned())
            .or_insert_with(|| vec![0.0f64; self.dims]);
        let count = self.centroid_counts.entry(ns.to_owned()).or_insert(0);
        for (s, v) in sums.iter_mut().zip(vector.iter()) {
            *s += *v as f64;
        }
        *count += 1;

        // Rebuild router centroid periodically (every 50 inserts)
        if *count % 50 == 0 || *count == 1 {
            let ns_owned = ns.to_owned();
            self.update_router(&ns_owned);
        }

        self.get_or_create_ns(ns).insert(id, vector);
    }

    fn search_single(&self, ns: &str, query: &[f32], k: usize, ef: usize) -> Vec<NsResult> {
        let Some(idx) = self.indexes.get(ns) else {
            return Vec::new();
        };
        idx.search(query, k, ef)
            .into_iter()
            .map(|(d, id)| NsResult {
                id,
                namespace: ns.to_owned(),
                dist_sq: d,
            })
            .collect()
    }

    fn search_cross(&self, query: &[f32], k: usize, ef: usize) -> Vec<NsResult> {
        // Step 1: find top-R nearest namespace centroids via router
        let r = self.route_k.min(self.router_ns.len()).max(1);
        let routed = self.router.search(query, r, r * 2);

        // Step 2: probe only routed namespaces
        let mut all: Vec<NsResult> = routed
            .into_iter()
            .filter_map(|(_, ns_id)| self.router_ns.get(ns_id as usize))
            .flat_map(|ns| {
                let idx = self.indexes.get(ns)?;
                Some(
                    idx.search(query, k, ef)
                        .into_iter()
                        .map(|(d, id)| NsResult {
                            id,
                            namespace: ns.clone(),
                            dist_sq: d,
                        }),
                )
            })
            .flatten()
            .collect();
        all.sort_by(|a, b| a.dist_sq.partial_cmp(&b.dist_sq).unwrap());
        all.truncate(k);
        all
    }

    fn name(&self) -> &'static str {
        "HierarchicalNS"
    }

    fn memory_bytes(&self) -> usize {
        self.indexes
            .values()
            .map(|i| i.memory_bytes())
            .sum::<usize>()
            + self.router.memory_bytes()
            + self
                .centroid_sums
                .values()
                .map(|v| v.len() * 8)
                .sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gen_vecs(n: usize, dims: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                (0..dims)
                    .map(|_| {
                        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                        ((s >> 33) as f32) / (u32::MAX as f32) - 0.5
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn global_flat_single_namespace() {
        let mut idx = GlobalFlat::new(8, 50);
        let vecs = gen_vecs(200, 32, 42);
        for (i, v) in vecs.iter().enumerate() {
            idx.insert("agent-0", i as u64, v.clone());
        }
        let results = idx.search_single("agent-0", &vecs[0], 5, 20);
        assert!(!results.is_empty(), "should return results");
        assert_eq!(results[0].id, 0, "nearest to itself is id=0");
    }

    #[test]
    fn partitioned_namespace_isolation() {
        let mut idx = Partitioned::new(8, 50);
        let vecs_a = gen_vecs(100, 32, 1);
        let vecs_b = gen_vecs(100, 32, 2);
        for (i, v) in vecs_a.iter().enumerate() {
            idx.insert("ns-a", i as u64, v.clone());
        }
        for (i, v) in vecs_b.iter().enumerate() {
            idx.insert("ns-b", 1000 + i as u64, v.clone());
        }
        let single = idx.search_single("ns-a", &vecs_a[0], 5, 20);
        assert!(
            single.iter().all(|r| r.namespace == "ns-a"),
            "no ns-b leakage"
        );
    }

    #[test]
    fn hierarchical_cross_search_finds_results() {
        let mut idx = HierarchicalNS::new(8, 50, 32, 3);
        let vecs = gen_vecs(300, 32, 99);
        for (i, v) in vecs.iter().enumerate() {
            let ns = format!("agent-{}", i % 3);
            idx.insert(&ns, i as u64, v.clone());
        }
        let results = idx.search_cross(&vecs[0], 10, 30);
        assert!(!results.is_empty(), "cross search must return something");
    }

    #[test]
    fn recall_at_k_perfect() {
        let results: Vec<NsResult> = (0..5)
            .map(|i| NsResult {
                id: i,
                namespace: "x".into(),
                dist_sq: i as f32,
            })
            .collect();
        assert!((recall_at_k(&results, &results, 5) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn recall_at_k_zero() {
        let oracle = vec![NsResult {
            id: 1,
            namespace: "x".into(),
            dist_sq: 0.1,
        }];
        let cands = vec![NsResult {
            id: 99,
            namespace: "x".into(),
            dist_sq: 0.5,
        }];
        assert_eq!(recall_at_k(&oracle, &cands, 1), 0.0);
    }

    #[test]
    fn namespace_count_scales_partitioned_memory() {
        let mut idx = Partitioned::new(8, 50);
        let vecs = gen_vecs(500, 64, 7);
        for (i, v) in vecs.iter().enumerate() {
            let ns = format!("agent-{}", i % 5);
            idx.insert(&ns, i as u64, v.clone());
        }
        // Memory should be non-trivial but bounded
        let mem = idx.memory_bytes();
        assert!(mem > 500 * 64 * 4, "must store at least raw vectors");
        assert!(mem < 500 * 64 * 4 * 20, "should not wildly exceed raw size");
    }

    #[test]
    fn single_ns_recall_acceptable() {
        let dims = 32;
        let n_per_ns = 300;
        let mut idx = Partitioned::new(12, 100);
        let vecs = gen_vecs(n_per_ns, dims, 555);
        for (i, v) in vecs.iter().enumerate() {
            idx.insert("solo", i as u64, v.clone());
        }

        // Brute-force oracle for query vecs[0]
        let query = &vecs[0];
        let mut dists: Vec<(f32, u64)> = vecs
            .iter()
            .enumerate()
            .map(|(i, v)| (hnsw::l2_sq(query, v), i as u64))
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let oracle: Vec<NsResult> = dists[..10]
            .iter()
            .map(|&(d, id)| NsResult {
                id,
                namespace: "solo".into(),
                dist_sq: d,
            })
            .collect();

        let results = idx.search_single("solo", query, 10, 50);
        let recall = recall_at_k(&oracle, &results, 10);
        assert!(recall >= 0.7, "recall@10 must be ≥ 70%, got {:.2}", recall);
    }
}
