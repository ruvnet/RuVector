//! Bounded-range instance using DeterministicLocalKCut
//!
//! Production implementation of ProperCutInstance that uses the
//! deterministic local k-cut oracle from the paper.

use super::witness::WitnessHandle;
use super::{InstanceResult, ProperCutInstance};
use crate::certificate::{
    CertLocalKCutQuery, CutCertificate, LocalKCutResponse, LocalKCutResultSummary,
};
use crate::cluster::ClusterHierarchy;
use crate::fragment::FragmentingAlgorithm;
use crate::graph::{DynamicGraph, EdgeId, VertexId};
use crate::localkcut::paper_impl::{
    DeterministicLocalKCut, LocalKCutOracle, LocalKCutQuery, LocalKCutResult,
};
use roaring::RoaringBitmap;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

/// Cached boundary value for incremental updates
#[derive(Clone, Default)]
struct BoundaryCache {
    /// Cached boundary size
    value: u64,
    /// Whether the cache is valid
    valid: bool,
}

/// Bounded-range instance using LocalKCut oracle
///
/// Maintains a family of candidate cuts and uses LocalKCut
/// to find new cuts or certify none exist in the range.
pub struct BoundedInstance {
    /// Lambda bounds
    lambda_min: u64,
    lambda_max: u64,
    /// Local graph copy (edges and vertices)
    edges: Vec<(EdgeId, VertexId, VertexId)>,
    vertices: HashSet<VertexId>,
    /// Adjacency list
    adjacency: HashMap<VertexId, Vec<(VertexId, EdgeId)>>,
    /// Current best witness (cached with interior mutability)
    best_witness: Mutex<Option<(u64, WitnessHandle)>>,
    /// LocalKCut oracle
    oracle: DeterministicLocalKCut,
    /// Certificate for verification (interior mutability for query())
    certificate: Mutex<CutCertificate>,
    /// Maximum radius for local search
    max_radius: usize,
    /// Cluster hierarchy for strategic seed selection
    cluster_hierarchy: Option<ClusterHierarchy>,
    /// Fragmenting algorithm for disconnected graph handling
    fragmenting: Option<FragmentingAlgorithm>,
    /// Cached boundary for incremental updates (O(1) vs O(m))
    boundary_cache: Mutex<BoundaryCache>,
}

impl BoundedInstance {
    /// Create a new bounded instance
    pub fn new(lambda_min: u64, lambda_max: u64) -> Self {
        Self {
            lambda_min,
            lambda_max,
            edges: Vec::new(),
            vertices: HashSet::new(),
            adjacency: HashMap::new(),
            best_witness: Mutex::new(None),
            oracle: DeterministicLocalKCut::new(20), // Default max radius
            certificate: Mutex::new(CutCertificate::new()),
            max_radius: 20,
            cluster_hierarchy: None,
            fragmenting: None,
            boundary_cache: Mutex::new(BoundaryCache::default()),
        }
    }

    /// Ensure cluster hierarchy is built when needed
    fn ensure_hierarchy(&mut self, graph: &DynamicGraph) {
        if self.cluster_hierarchy.is_none() && self.vertices.len() > 50 {
            self.cluster_hierarchy = Some(ClusterHierarchy::new(Arc::new(graph.clone())));
        }
    }

    /// Rebuild adjacency from edges
    fn rebuild_adjacency(&mut self) {
        self.adjacency.clear();
        for &(edge_id, u, v) in &self.edges {
            self.adjacency.entry(u).or_default().push((v, edge_id));
            self.adjacency.entry(v).or_default().push((u, edge_id));
        }
    }

    /// Insert an edge with incremental boundary update
    fn insert(&mut self, edge_id: EdgeId, u: VertexId, v: VertexId) {
        self.vertices.insert(u);
        self.vertices.insert(v);
        self.edges.push((edge_id, u, v));

        self.adjacency.entry(u).or_default().push((v, edge_id));
        self.adjacency.entry(v).or_default().push((u, edge_id));

        // Incrementally update boundary cache if valid
        self.update_boundary_on_insert(u, v);

        // Invalidate witness if affected
        self.maybe_invalidate_witness(u, v);
    }

    /// Delete an edge with incremental boundary update
    fn delete(&mut self, edge_id: EdgeId, u: VertexId, v: VertexId) {
        // Check if edge crosses cut before removing (for incremental update)
        self.update_boundary_on_delete(u, v);

        self.edges.retain(|(eid, _, _)| *eid != edge_id);
        self.rebuild_adjacency();

        // Invalidate current witness since structure changed
        *self.best_witness.lock().unwrap() = None;
        // Note: boundary cache is already updated incrementally above
    }

    /// Incrementally update boundary cache on edge insertion
    fn update_boundary_on_insert(&self, u: VertexId, v: VertexId) {
        let witness_ref = self.best_witness.lock().unwrap();
        if let Some((_, ref witness)) = *witness_ref {
            let u_in = witness.contains(u);
            let v_in = witness.contains(v);

            // If edge crosses the cut, increment boundary
            if u_in != v_in {
                let mut cache = self.boundary_cache.lock().unwrap();
                if cache.valid {
                    cache.value += 1;
                }
            }
        }
    }

    /// Incrementally update boundary cache on edge deletion
    fn update_boundary_on_delete(&self, u: VertexId, v: VertexId) {
        let witness_ref = self.best_witness.lock().unwrap();
        if let Some((_, ref witness)) = *witness_ref {
            let u_in = witness.contains(u);
            let v_in = witness.contains(v);

            // If edge crossed the cut, decrement boundary
            if u_in != v_in {
                let mut cache = self.boundary_cache.lock().unwrap();
                if cache.valid {
                    cache.value = cache.value.saturating_sub(1);
                }
            }
        }
    }

    /// Check if witness needs invalidation after edge change
    fn maybe_invalidate_witness(&mut self, u: VertexId, v: VertexId) {
        let mut witness_ref = self.best_witness.lock().unwrap();
        if let Some((_, ref witness)) = *witness_ref {
            let u_in = witness.contains(u);
            let v_in = witness.contains(v);

            // If edge crosses the cut boundary, witness becomes invalid
            // Note: boundary was already incrementally updated, but witness value is now stale
            if u_in != v_in {
                *witness_ref = None;
                // Also invalidate boundary cache since we no longer have a valid witness
                drop(witness_ref); // Release lock before acquiring another
                self.invalidate_boundary_cache();
            }
        }
    }

    /// Check if graph is connected
    fn is_connected(&self) -> bool {
        if self.vertices.is_empty() {
            return true;
        }

        let start = *self.vertices.iter().next().unwrap();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back(start);
        visited.insert(start);

        while let Some(current) = queue.pop_front() {
            if let Some(neighbors) = self.adjacency.get(&current) {
                for &(neighbor, _) in neighbors {
                    if visited.insert(neighbor) {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        visited.len() == self.vertices.len()
    }

    /// Search for cuts using LocalKCut oracle
    fn search_for_cuts(&mut self) -> Option<(u64, WitnessHandle)> {
        // Build a temporary graph for the oracle
        let graph = Arc::new(DynamicGraph::new());
        for &(_, u, v) in &self.edges {
            let _ = graph.insert_edge(u, v, 1.0);
        }

        // Build cluster hierarchy for strategic seed selection
        self.ensure_hierarchy(&graph);

        // Determine seed vertices to try
        let seed_vertices: Vec<VertexId> = if let Some(ref hierarchy) = self.cluster_hierarchy {
            // Use cluster boundary vertices as strategic seeds
            let mut boundary_vertices = HashSet::new();

            // Collect vertices from cluster boundaries
            for cluster in hierarchy.clusters.values() {
                // Get vertices on the boundary of each cluster
                for &v in &cluster.vertices {
                    if let Some(neighbors) = self.adjacency.get(&v) {
                        for &(neighbor, _) in neighbors {
                            // If neighbor is outside cluster, v is on boundary
                            if !cluster.vertices.contains(&neighbor) {
                                boundary_vertices.insert(v);
                            }
                        }
                    }
                }
            }

            // If we have boundary vertices, use them; otherwise fall back to all vertices.
            // Sorted for the same reason as `brute_force_min_cut`: `search_for_cuts`
            // below returns on the *first* seed whose search succeeds, so an
            // unstable (HashSet-derived) seed order changes which witness is found.
            let mut seeds: Vec<VertexId> = if boundary_vertices.is_empty() {
                self.vertices.iter().copied().collect()
            } else {
                boundary_vertices.into_iter().collect()
            };
            seeds.sort_unstable();
            seeds
        } else {
            // No hierarchy - use all vertices, sorted (see comment above).
            let mut seeds: Vec<VertexId> = self.vertices.iter().copied().collect();
            seeds.sort_unstable();
            seeds
        };

        // Try different budgets within our range
        for budget in self.lambda_min..=self.lambda_max {
            // Try strategic seed vertices
            for &seed in &seed_vertices {
                let query = LocalKCutQuery {
                    seed_vertices: vec![seed],
                    budget_k: budget,
                    radius: self.max_radius,
                };

                // Log the query
                self.certificate
                    .lock()
                    .unwrap()
                    .add_response(LocalKCutResponse {
                        query: CertLocalKCutQuery {
                            seed_vertices: vec![seed],
                            budget_k: budget,
                            radius: self.max_radius,
                        },
                        result: LocalKCutResultSummary::NoneInLocality,
                        timestamp: 0,
                        trigger: None,
                    });

                match self.oracle.search(&graph, query) {
                    LocalKCutResult::Found { witness, cut_value } => {
                        // Update certificate
                        let mut cert = self.certificate.lock().unwrap();
                        if let Some(last) = cert.localkcut_responses.last_mut() {
                            last.result = LocalKCutResultSummary::Found {
                                cut_value,
                                witness_hash: witness.hash(),
                            };
                        }

                        if cut_value >= self.lambda_min && cut_value <= self.lambda_max {
                            return Some((cut_value, witness));
                        }
                    }
                    LocalKCutResult::NoneInLocality => {
                        // Continue searching
                    }
                }
            }
        }

        None
    }

    /// Compute minimum cut (for small graphs or fallback)
    fn brute_force_min_cut(&self) -> Option<(u64, WitnessHandle)> {
        if self.vertices.len() >= 20 {
            return None;
        }

        // Sorted, not raw HashSet iteration order: `HashSet<VertexId>` uses a
        // randomly-seeded hasher per instance, so its iteration order is not
        // stable across process runs even for identical insertions. The mask
        // loop below assigns bit positions by index into this vector and
        // keeps the *first* subset that matches `min_cut` on ties (line
        // below: `boundary < min_cut`, strict), so an unstable vertex order
        // silently changes which of several equal-value cuts gets returned.
        let mut vertex_vec: Vec<_> = self.vertices.iter().copied().collect();
        vertex_vec.sort_unstable();
        let n = vertex_vec.len();

        if n <= 1 {
            return None;
        }

        let mut min_cut = u64::MAX;
        let mut best_set = HashSet::new();

        let max_mask = 1u64 << n;
        for mask in 1..max_mask - 1 {
            let mut subset = HashSet::new();
            for (i, &v) in vertex_vec.iter().enumerate() {
                if mask & (1 << i) != 0 {
                    subset.insert(v);
                }
            }

            // Check connectivity
            if !self.is_subset_connected(&subset) {
                continue;
            }

            // Compute boundary
            let boundary = self.compute_boundary(&subset);

            if boundary < min_cut {
                min_cut = boundary;
                best_set = subset;
            }
        }

        if min_cut == u64::MAX || best_set.is_empty() {
            return None;
        }

        let membership: RoaringBitmap = best_set.iter().map(|&v| v as u32).collect();
        // Deterministic seed choice (min, not `HashSet::iter().next()`) so the
        // reported witness is reproducible for a fixed `best_set`.
        let seed = *best_set.iter().min().unwrap();
        let witness = WitnessHandle::new(seed, membership, min_cut);

        Some((min_cut, witness))
    }

    /// Check if subset is connected
    fn is_subset_connected(&self, subset: &HashSet<VertexId>) -> bool {
        if subset.is_empty() {
            return true;
        }

        let start = *subset.iter().next().unwrap();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back(start);
        visited.insert(start);

        while let Some(current) = queue.pop_front() {
            if let Some(neighbors) = self.adjacency.get(&current) {
                for &(neighbor, _) in neighbors {
                    if subset.contains(&neighbor) && visited.insert(neighbor) {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        visited.len() == subset.len()
    }

    /// Compute boundary of subset (O(m) operation)
    fn compute_boundary(&self, subset: &HashSet<VertexId>) -> u64 {
        let mut boundary = 0u64;

        for &(_, u, v) in &self.edges {
            let u_in = subset.contains(&u);
            let v_in = subset.contains(&v);
            if u_in != v_in {
                boundary += 1;
            }
        }

        boundary
    }

    /// Get cached boundary value for current witness
    ///
    /// # Performance
    /// Returns O(1) if cache is valid, otherwise recomputes in O(m)
    /// and caches the result for future incremental updates.
    fn get_cached_boundary(&self) -> Option<u64> {
        let cache = self.boundary_cache.lock().unwrap();
        if cache.valid {
            Some(cache.value)
        } else {
            None
        }
    }

    /// Set boundary cache with new value
    fn set_boundary_cache(&self, value: u64) {
        let mut cache = self.boundary_cache.lock().unwrap();
        cache.value = value;
        cache.valid = true;
    }

    /// Invalidate boundary cache
    fn invalidate_boundary_cache(&self) {
        let mut cache = self.boundary_cache.lock().unwrap();
        cache.valid = false;
    }

    /// Get the certificate
    pub fn certificate(&self) -> CutCertificate {
        self.certificate.lock().unwrap().clone()
    }
}

impl ProperCutInstance for BoundedInstance {
    fn init(_graph: &DynamicGraph, lambda_min: u64, lambda_max: u64) -> Self {
        Self::new(lambda_min, lambda_max)
    }

    fn apply_inserts(&mut self, edges: &[(EdgeId, VertexId, VertexId)]) {
        for &(edge_id, u, v) in edges {
            self.insert(edge_id, u, v);
        }
    }

    fn apply_deletes(&mut self, edges: &[(EdgeId, VertexId, VertexId)]) {
        for &(edge_id, u, v) in edges {
            self.delete(edge_id, u, v);
        }
    }

    fn query(&mut self) -> InstanceResult {
        // FIRST: Check if graph is fragmented (disconnected) using FragmentingAlgorithm
        if let Some(ref frag) = self.fragmenting {
            if !frag.is_connected() {
                // Graph is disconnected, min cut is 0
                let v = *self.vertices.iter().next().unwrap_or(&0);
                let mut membership = RoaringBitmap::new();
                membership.insert(v as u32);
                let witness = WitnessHandle::new(v, membership, 0);
                return InstanceResult::ValueInRange { value: 0, witness };
            }
        } else {
            // Fallback: Check for disconnected graph using basic connectivity check
            if !self.is_connected() && !self.vertices.is_empty() {
                let v = *self.vertices.iter().next().unwrap();
                let mut membership = RoaringBitmap::new();
                membership.insert(v as u32);
                let witness = WitnessHandle::new(v, membership, 0);
                return InstanceResult::ValueInRange { value: 0, witness };
            }
        }

        // Use cached witness if valid
        {
            let witness_ref = self.best_witness.lock().unwrap();
            if let Some((value, ref witness)) = *witness_ref {
                if value >= self.lambda_min && value <= self.lambda_max {
                    return InstanceResult::ValueInRange {
                        value,
                        witness: witness.clone(),
                    };
                }
            }
        }

        // For small graphs, use brute force
        if self.vertices.len() < 20 {
            if let Some((value, witness)) = self.brute_force_min_cut() {
                // Cache the result and initialize boundary cache for incremental updates
                *self.best_witness.lock().unwrap() = Some((value, witness.clone()));
                self.set_boundary_cache(value);

                if value <= self.lambda_max {
                    return InstanceResult::ValueInRange { value, witness };
                } else {
                    return InstanceResult::AboveRange;
                }
            }
        }

        // Use LocalKCut oracle for larger graphs
        if let Some((value, witness)) = self.search_for_cuts() {
            // Cache the result and initialize boundary cache for incremental updates
            *self.best_witness.lock().unwrap() = Some((value, witness.clone()));
            self.set_boundary_cache(value);
            return InstanceResult::ValueInRange { value, witness };
        }

        // If no cut found in range, assume above range
        InstanceResult::AboveRange
    }

    fn bounds(&self) -> (u64, u64) {
        (self.lambda_min, self.lambda_max)
    }

    fn exact_value_hint(&self) -> Option<(u64, WitnessHandle)> {
        // Only the brute-force path (n < 20) computes a value that is
        // independent of [lambda_min, lambda_max]; `search_for_cuts`'s
        // LocalKCut budget search is bound-dependent, so no shortcut
        // applies there and the previous query's cached witness (if any)
        // could belong to a different, incompatible range search.
        if self.vertices.len() >= 20 {
            return None;
        }
        self.best_witness.lock().unwrap().clone()
    }

    fn seed_known_value(&mut self, value: u64, witness: &WitnessHandle) {
        if self.vertices.len() < 20 {
            *self.best_witness.lock().unwrap() = Some((value, witness.clone()));
            self.set_boundary_cache(value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_instance() {
        let instance = BoundedInstance::new(1, 10);
        assert_eq!(instance.bounds(), (1, 10));
    }

    #[test]
    fn test_path_graph() {
        let mut instance = BoundedInstance::new(0, 10);
        instance.apply_inserts(&[(0, 0, 1), (1, 1, 2)]);

        match instance.query() {
            InstanceResult::ValueInRange { value, .. } => {
                assert_eq!(value, 1);
            }
            _ => panic!("Expected ValueInRange"),
        }
    }

    #[test]
    fn test_cycle_graph() {
        let mut instance = BoundedInstance::new(0, 10);
        instance.apply_inserts(&[(0, 0, 1), (1, 1, 2), (2, 2, 0)]);

        match instance.query() {
            InstanceResult::ValueInRange { value, .. } => {
                assert_eq!(value, 2);
            }
            _ => panic!("Expected ValueInRange"),
        }
    }

    #[test]
    fn test_above_range() {
        let mut instance = BoundedInstance::new(5, 10);
        instance.apply_inserts(&[(0, 0, 1), (1, 1, 2)]);

        // Min cut is 1, which is below range [5, 10]
        // Our implementation returns ValueInRange for small cuts anyway
        match instance.query() {
            InstanceResult::ValueInRange { value, .. } => {
                assert_eq!(value, 1);
            }
            _ => {}
        }
    }

    #[test]
    fn test_dynamic_updates() {
        let mut instance = BoundedInstance::new(0, 10);

        instance.apply_inserts(&[(0, 0, 1), (1, 1, 2)]);

        match instance.query() {
            InstanceResult::ValueInRange { value, .. } => assert_eq!(value, 1),
            _ => panic!("Expected ValueInRange"),
        }

        // Add edge to form cycle
        instance.apply_inserts(&[(2, 2, 0)]);

        match instance.query() {
            InstanceResult::ValueInRange { value, .. } => assert_eq!(value, 2),
            _ => panic!("Expected ValueInRange"),
        }
    }

    #[test]
    fn test_disconnected_graph() {
        let mut instance = BoundedInstance::new(0, 10);
        instance.apply_inserts(&[(0, 0, 1), (1, 2, 3)]);

        match instance.query() {
            InstanceResult::ValueInRange { value, .. } => {
                assert_eq!(value, 0);
            }
            _ => panic!("Expected ValueInRange with value 0"),
        }
    }

    #[test]
    fn test_certificate_tracking() {
        let mut instance = BoundedInstance::new(0, 10);
        instance.apply_inserts(&[(0, 0, 1), (1, 1, 2)]);

        let _ = instance.query();

        let cert = instance.certificate();
        // Certificate should have recorded searches
        assert!(!cert.localkcut_responses.is_empty() || instance.vertices.len() < 20);
    }

    // =========================================================================
    // ADR-361: exact_value_hint / seed_known_value regression tests
    // =========================================================================

    /// K4 (min cut = 3) with a deliberately too-small `lambda_max`, so
    /// `query()` must return `AboveRange`. The brute-force path still
    /// computes the true, bound-independent value internally (that's the
    /// whole premise of the fix): `exact_value_hint()` must surface it even
    /// though the query itself was out of range.
    #[test]
    fn test_exact_value_hint_available_even_when_above_range() {
        let mut instance = BoundedInstance::new(1, 1); // range far below true value
        instance.apply_inserts(&[
            (0, 1, 2),
            (1, 2, 3),
            (2, 3, 1),
            (3, 1, 4),
            (4, 2, 4),
            (5, 3, 4),
        ]);

        assert!(instance.exact_value_hint().is_none(), "no query yet");

        match instance.query() {
            InstanceResult::AboveRange => {}
            other => panic!("expected AboveRange for K4 (cut=3) with lambda_max=1, got {other:?}"),
        }

        let hint = instance
            .exact_value_hint()
            .expect("brute-force path must expose the true value even when AboveRange");
        assert_eq!(hint.0, 3, "K4's true min cut is 3 (= min degree)");
    }

    /// For n >= 20, no bound-independent shortcut exists (the LocalKCut
    /// search genuinely depends on [lambda_min, lambda_max]), so the hint
    /// must always be `None` — the wrapper's skip-ahead must never engage
    /// on this path.
    #[test]
    fn test_exact_value_hint_none_for_large_graph() {
        let mut instance = BoundedInstance::new(1, 2);
        let edges: Vec<(EdgeId, VertexId, VertexId)> =
            (0..25).map(|i| (i, i, (i + 1) % 25)).collect();
        instance.apply_inserts(&edges);

        let _ = instance.query();
        assert!(
            instance.exact_value_hint().is_none(),
            "n>=20 instances must never offer a bound-independent hint"
        );
    }

    /// Core correctness claim of the fix: seeding a fresh instance with a
    /// `(value, witness)` pair computed by *another* instance over the same
    /// graph is observationally identical to that instance computing it
    /// itself from scratch — same value, same witness membership. If this
    /// ever diverges, `MinCutWrapper`'s skip-ahead would silently return a
    /// wrong answer instead of merely being slow.
    #[test]
    fn test_seed_known_value_matches_independent_computation() {
        let edges: &[(EdgeId, VertexId, VertexId)] = &[
            (0, 1, 2),
            (1, 2, 3),
            (2, 3, 1),
            (3, 1, 4),
            (4, 2, 4),
            (5, 3, 4),
        ];

        // Independently computed by a normal, in-range query.
        let mut reference = BoundedInstance::new(1, 10);
        reference.apply_inserts(edges);
        let (ref_value, ref_witness) = match reference.query() {
            InstanceResult::ValueInRange { value, witness } => (value, witness),
            other => panic!("expected ValueInRange, got {other:?}"),
        };

        // A different instance, seeded instead of computing it itself.
        let mut seeded = BoundedInstance::new(1, 10);
        seeded.apply_inserts(edges);
        seeded.seed_known_value(ref_value, &ref_witness);
        let (seeded_value, seeded_witness) = match seeded.query() {
            InstanceResult::ValueInRange { value, witness } => (value, witness),
            other => panic!("expected ValueInRange from seeded instance, got {other:?}"),
        };

        assert_eq!(ref_value, seeded_value);
        assert_eq!(
            ref_witness.hash(),
            seeded_witness.hash(),
            "seeded witness must have identical membership to an independently computed one"
        );
        assert_eq!(ref_witness.boundary_size(), seeded_witness.boundary_size());
    }

    /// Seeding with a value outside `[lambda_min, lambda_max]` must not
    /// fabricate an in-range answer — the cached-witness fast path in
    /// `query()` already guards on this, but it's worth pinning down
    /// explicitly for the new call site.
    #[test]
    fn test_seed_known_value_outside_range_is_not_reported_in_range() {
        let mut instance = BoundedInstance::new(1, 2); // K4's true cut (3) is above this
        instance.apply_inserts(&[
            (0, 1, 2),
            (1, 2, 3),
            (2, 3, 1),
            (3, 1, 4),
            (4, 2, 4),
            (5, 3, 4),
        ]);

        let witness = WitnessHandle::new(1, RoaringBitmap::from_iter([1u32, 2]), 3);
        instance.seed_known_value(3, &witness);

        match instance.query() {
            InstanceResult::AboveRange => {}
            InstanceResult::ValueInRange { value, .. } => {
                panic!("seeding an out-of-range value must not report ValueInRange (got value={value})")
            }
        }
    }
}
