//! Dynamic Connectivity for minimum cut wrapper
//!
//! Hybrid implementation using Euler Tour Trees with union-find fallback.
//! Provides O(log n) operations for insertions and queries.
//!
//! # Overview
//!
//! This module provides dynamic connectivity data structures:
//!
//! - [`DynamicConnectivity`]: Euler Tour Tree backend with union-find fallback
//!   - Edge insertions in O(log n) time
//!   - Edge deletions via full rebuild in O(m·α(n)) time
//!   - Connectivity queries in O(log n) time
//!
//! - [`PolylogConnectivity`]: Polylogarithmic worst-case connectivity (arXiv:2510.08297)
//!   - Edge insertions in O(log³ n) expected worst-case
//!   - Edge deletions in O(log³ n) expected worst-case
//!   - Connectivity queries in O(log n) worst-case
//!
//! # Implementation
//!
//! The primary backend uses Euler Tour Trees for O(log n) operations.
//! Falls back to union-find rebuild for deletions until full ETT cut is implemented.
//!
//! The polylog backend uses a hierarchy of O(log n) levels with edge sparsification
//! via low-congestion shortcuts for guaranteed worst-case bounds.

pub mod cache_opt;
pub mod polylog;

use crate::euler::EulerTourTree;
use crate::graph::VertexId;
use std::collections::{HashMap, HashSet};

/// Dynamic connectivity data structure with Euler Tour Tree backend
///
/// Maintains connected components of an undirected graph with support for
/// edge insertions and deletions. Uses Euler Tour Trees for O(log n) operations
/// with union-find fallback for robustness.
///
/// # Examples
///
/// ```ignore
/// let mut dc = DynamicConnectivity::new();
/// dc.add_vertex(0);
/// dc.add_vertex(1);
/// dc.add_vertex(2);
///
/// dc.insert_edge(0, 1);
/// assert!(dc.connected(0, 1));
/// assert!(!dc.connected(0, 2));
///
/// dc.insert_edge(1, 2);
/// assert!(dc.is_connected()); // All vertices connected
///
/// dc.delete_edge(1, 2);
/// assert!(!dc.connected(0, 2));
/// ```
#[derive(Debug, Clone)]
pub struct DynamicConnectivity {
    /// Union-find parent array
    parent: HashMap<VertexId, VertexId>,

    /// Union-find rank array for union by rank
    rank: HashMap<VertexId, usize>,

    /// Current edge set for rebuild on deletions
    /// Edges normalized so smaller vertex is always first
    edges: HashSet<(VertexId, VertexId)>,

    /// Number of vertices
    vertex_count: usize,

    /// Number of connected components
    component_count: usize,

    /// Euler Tour Tree for O(log n) operations
    ett: EulerTourTree,

    /// Whether ETT is in sync with union-find
    ett_synced: bool,
}

impl DynamicConnectivity {
    /// Creates a new empty dynamic connectivity structure
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let dc = DynamicConnectivity::new();
    /// assert_eq!(dc.component_count(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            parent: HashMap::new(),
            rank: HashMap::new(),
            edges: HashSet::new(),
            vertex_count: 0,
            component_count: 0,
            ett: EulerTourTree::new(),
            ett_synced: true,
        }
    }

    /// Adds a vertex to the connectivity structure
    ///
    /// If the vertex already exists, this is a no-op.
    /// Each new vertex starts in its own component.
    ///
    /// # Arguments
    ///
    /// * `v` - The vertex ID to add
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut dc = DynamicConnectivity::new();
    /// dc.add_vertex(0);
    /// assert_eq!(dc.component_count(), 1);
    /// ```
    pub fn add_vertex(&mut self, v: VertexId) {
        if !self.parent.contains_key(&v) {
            self.parent.insert(v, v);
            self.rank.insert(v, 0);
            self.vertex_count += 1;
            self.component_count += 1;

            // Add to Euler Tour Tree (O(log n))
            let _ = self.ett.make_tree(v);
        }
    }

    /// Inserts an edge between two vertices
    ///
    /// Automatically adds vertices if they don't exist.
    /// If vertices are already connected, updates internal state but
    /// doesn't change connectivity.
    ///
    /// # Arguments
    ///
    /// * `u` - First vertex
    /// * `v` - Second vertex
    ///
    /// # Time Complexity
    ///
    /// O(log n) via Euler Tour Tree link operation
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut dc = DynamicConnectivity::new();
    /// dc.insert_edge(0, 1);
    /// assert!(dc.connected(0, 1));
    /// ```
    pub fn insert_edge(&mut self, u: VertexId, v: VertexId) {
        // Add vertices if they don't exist
        self.add_vertex(u);
        self.add_vertex(v);

        // Normalize edge (smaller vertex first)
        let edge = if u < v { (u, v) } else { (v, u) };

        // Add to edge set
        if self.edges.insert(edge) {
            // New edge - perform union
            let root_u = self.find(u);
            let root_v = self.find(v);

            if root_u != root_v {
                self.union(root_u, root_v);

                // Link in Euler Tour Tree (O(log n))
                let _ = self.ett.link(u, v);
            }
        }
    }

    /// Deletes an edge between two vertices
    ///
    /// Triggers a full rebuild of the data structure from the remaining edges.
    /// The ETT is also rebuilt to maintain O(log n) queries.
    ///
    /// # Arguments
    ///
    /// * `u` - First vertex
    /// * `v` - Second vertex
    ///
    /// # Time Complexity
    ///
    /// O(m·α(n)) where m is the number of edges (includes ETT rebuild)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut dc = DynamicConnectivity::new();
    /// dc.insert_edge(0, 1);
    /// dc.delete_edge(0, 1);
    /// assert!(!dc.connected(0, 1));
    /// ```
    pub fn delete_edge(&mut self, u: VertexId, v: VertexId) {
        // Normalize edge
        let edge = if u < v { (u, v) } else { (v, u) };

        // Remove from edge set
        if self.edges.remove(&edge) {
            // Mark ETT as out of sync
            self.ett_synced = false;

            // Rebuild the entire structure (including ETT)
            self.rebuild();
        }
    }

    /// Checks if the entire graph is connected (single component)
    ///
    /// # Returns
    ///
    /// `true` if all vertices are in a single connected component,
    /// `false` otherwise
    ///
    /// # Time Complexity
    ///
    /// O(1)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut dc = DynamicConnectivity::new();
    /// dc.add_vertex(0);
    /// dc.add_vertex(1);
    /// assert!(!dc.is_connected());
    ///
    /// dc.insert_edge(0, 1);
    /// assert!(dc.is_connected());
    /// ```
    pub fn is_connected(&self) -> bool {
        self.component_count == 1
    }

    /// Checks if two vertices are in the same connected component
    ///
    /// # Arguments
    ///
    /// * `u` - First vertex
    /// * `v` - Second vertex
    ///
    /// # Returns
    ///
    /// `true` if vertices are connected, `false` otherwise.
    /// Returns `false` if either vertex doesn't exist.
    ///
    /// # Time Complexity
    ///
    /// O(α(n)) amortized via union-find with path compression
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut dc = DynamicConnectivity::new();
    /// dc.insert_edge(0, 1);
    /// dc.insert_edge(1, 2);
    /// assert!(dc.connected(0, 2));
    /// ```
    pub fn connected(&mut self, u: VertexId, v: VertexId) -> bool {
        if !self.parent.contains_key(&u) || !self.parent.contains_key(&v) {
            return false;
        }

        // Use union-find with path compression (effectively O(1) amortized)
        // ETT is maintained for future subtree query optimizations
        self.find(u) == self.find(v)
    }

    /// Fast connectivity check using Euler Tour Tree (O(log n))
    ///
    /// Returns None if ETT is out of sync and result is unreliable.
    /// Use `connected()` for the reliable version.
    #[inline]
    pub fn connected_fast(&self, u: VertexId, v: VertexId) -> Option<bool> {
        if !self.ett_synced {
            return None;
        }
        Some(self.ett.connected(u, v))
    }

    /// Returns the number of connected components
    ///
    /// # Returns
    ///
    /// The current number of connected components
    pub fn component_count(&self) -> usize {
        self.component_count
    }

    /// Returns the number of vertices
    ///
    /// # Returns
    ///
    /// The current number of vertices
    pub fn vertex_count(&self) -> usize {
        self.vertex_count
    }

    /// Finds the root of a vertex's component with path compression
    ///
    /// # Arguments
    ///
    /// * `v` - The vertex to find the root for
    ///
    /// # Returns
    ///
    /// The root vertex of the component containing `v`
    ///
    /// # Panics
    ///
    /// Panics if the vertex doesn't exist in the structure
    fn find(&mut self, v: VertexId) -> VertexId {
        let parent = *self.parent.get(&v).expect("Vertex not found");

        if parent != v {
            // Path compression: make v point directly to root
            let root = self.find(parent);
            self.parent.insert(v, root);
            root
        } else {
            v
        }
    }

    /// Unions two components by rank
    ///
    /// # Arguments
    ///
    /// * `u` - Root of first component
    /// * `v` - Root of second component
    ///
    /// # Notes
    ///
    /// This function assumes `u` and `v` are roots. It should only be
    /// called after `find()` operations.
    fn union(&mut self, u: VertexId, v: VertexId) {
        if u == v {
            return;
        }

        let rank_u = *self.rank.get(&u).unwrap_or(&0);
        let rank_v = *self.rank.get(&v).unwrap_or(&0);

        // Union by rank: attach smaller tree to larger tree
        if rank_u < rank_v {
            self.parent.insert(u, v);
        } else if rank_u > rank_v {
            self.parent.insert(v, u);
        } else {
            // Equal rank: arbitrary choice, increment rank
            self.parent.insert(v, u);
            self.rank.insert(u, rank_u + 1);
        }

        // Decrease component count
        self.component_count -= 1;
    }

    /// Rebuilds the union-find structure from the current edge set
    ///
    /// Called after edge deletions to recompute connected components.
    /// Resets all vertices to singleton components and re-applies all edges.
    /// Also rebuilds the Euler Tour Tree for O(log n) queries.
    ///
    /// # Time Complexity
    ///
    /// O(m·α(n)) where m is the number of edges
    fn rebuild(&mut self) {
        // Collect all vertices
        let vertices: Vec<VertexId> = self.parent.keys().copied().collect();

        // Reset to singleton components
        self.component_count = vertices.len();
        for &v in &vertices {
            self.parent.insert(v, v);
            self.rank.insert(v, 0);
        }

        // Rebuild Euler Tour Tree
        self.ett = EulerTourTree::new();
        for &v in &vertices {
            let _ = self.ett.make_tree(v);
        }

        // Re-apply all edges
        let edges: Vec<(VertexId, VertexId)> = self.edges.iter().copied().collect();
        for (u, v) in edges {
            let root_u = self.find(u);
            let root_v = self.find(v);

            if root_u != root_v {
                self.union(root_u, root_v);
                // Link in ETT
                let _ = self.ett.link(u, v);
            }
        }

        // Mark ETT as synced
        self.ett_synced = true;
    }
}

impl Default for DynamicConnectivity {
    fn default() -> Self {
        Self::new()
    }
}

/// Which dynamic connectivity backend [`crate::wrapper::MinCutWrapper`] uses
/// for its O(1)-amortized "is the whole graph connected" fast-path check
/// (the first line of `MinCutWrapper::query`).
///
/// # Nightly research context (2026-10-07)
///
/// This enum exists because of a specific, falsified hypothesis: that
/// swapping [`PolylogConnectivity`](polylog::PolylogConnectivity) in for
/// [`DynamicConnectivity`] here would improve `RuVectorGraphAnalyzer::
/// partition()`'s measured latency/scaling problem (ADR-345, ADR-346's
/// "Next Research" item 2). It does not: instrumented profiling
/// (`docs/research/nightly/2026-10-07-mincut-polylog-connectivity-backend/`)
/// showed `partition()`'s cost is entirely inside
/// `instance::bounded::BoundedInstance` (its brute-force / `LocalKCut`
/// search paths, repeated redundantly across the ~16 geometric-range
/// instances `MinCutWrapper::process_instances` walks through before a
/// value lands in range) and never touches this connectivity structure at
/// all beyond the one-time `is_connected()` check this enum's two backends
/// both already answer in O(1)/O(log n) amortized time. The backend is
/// still wired in as a real, selectable, independently testable and
/// benchmarked choice (baseline [`ConnectivityBackend::EulerTour`] remains
/// the default everywhere) because it is a legitimate, correct alternative
/// for the one thing it actually governs — it is just not a fix for
/// `partition()` latency, and this module says so rather than implying
/// otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConnectivityBackend {
    /// Euler Tour Tree + union-find fallback (existing default, unchanged).
    #[default]
    EulerTour,
    /// [`PolylogConnectivity`](polylog::PolylogConnectivity) (arXiv:2510.08297).
    ///
    /// # Known limitation (found 2026-10-07, not fixed)
    ///
    /// `PolylogConnectivity::delete_edge`'s replacement-edge search has a
    /// reproducible correctness bug: `is_connected()` can disagree with
    /// `DynamicConnectivity`'s (ground-truth, full-rebuild-on-delete)
    /// answer after a mixed insert/delete sequence — see
    /// `connectivity::backend_equivalence_tests::delete_heavy_sequences_can_diverge_polylog_is_connected_known_bug`.
    /// Insert-only usage (every current caller in this codebase) is
    /// verified equivalent to `EulerTour`; do not select this backend for
    /// a workload with edge deletions until that bug is fixed.
    Polylog,
}

/// Enum-dispatched connectivity structure selectable via [`ConnectivityBackend`].
///
/// Exposes the subset of `DynamicConnectivity`'s/`PolylogConnectivity`'s API
/// that `MinCutWrapper` actually needs (`insert_edge`, `delete_edge`,
/// `is_connected`, `connected`, `component_count`), so call sites are
/// identical regardless of which backend is selected.
#[derive(Debug)]
pub enum ConnectivityStructure {
    /// [`DynamicConnectivity`] backend.
    EulerTour(DynamicConnectivity),
    /// [`PolylogConnectivity`](polylog::PolylogConnectivity) backend.
    Polylog(polylog::PolylogConnectivity),
}

impl ConnectivityStructure {
    /// Construct a fresh, empty structure using the selected backend.
    pub fn new(backend: ConnectivityBackend) -> Self {
        match backend {
            ConnectivityBackend::EulerTour => Self::EulerTour(DynamicConnectivity::new()),
            ConnectivityBackend::Polylog => Self::Polylog(polylog::PolylogConnectivity::new()),
        }
    }

    /// Which backend this instance is using.
    pub fn backend(&self) -> ConnectivityBackend {
        match self {
            Self::EulerTour(_) => ConnectivityBackend::EulerTour,
            Self::Polylog(_) => ConnectivityBackend::Polylog,
        }
    }

    /// Insert an edge (see `DynamicConnectivity::insert_edge` /
    /// `PolylogConnectivity::insert_edge`).
    pub fn insert_edge(&mut self, u: VertexId, v: VertexId) {
        match self {
            Self::EulerTour(c) => c.insert_edge(u, v),
            Self::Polylog(c) => c.insert_edge(u, v),
        }
    }

    /// Delete an edge (see `DynamicConnectivity::delete_edge` /
    /// `PolylogConnectivity::delete_edge`).
    pub fn delete_edge(&mut self, u: VertexId, v: VertexId) {
        match self {
            Self::EulerTour(c) => c.delete_edge(u, v),
            Self::Polylog(c) => c.delete_edge(u, v),
        }
    }

    /// Whether the entire graph is a single connected component.
    pub fn is_connected(&self) -> bool {
        match self {
            Self::EulerTour(c) => c.is_connected(),
            Self::Polylog(c) => c.is_connected(),
        }
    }

    /// Whether `u` and `v` are in the same connected component.
    pub fn connected(&mut self, u: VertexId, v: VertexId) -> bool {
        match self {
            Self::EulerTour(c) => c.connected(u, v),
            Self::Polylog(c) => c.connected(u, v),
        }
    }

    /// Number of connected components.
    pub fn component_count(&self) -> usize {
        match self {
            Self::EulerTour(c) => c.component_count(),
            Self::Polylog(c) => c.component_count(),
        }
    }
}

impl Default for ConnectivityStructure {
    fn default() -> Self {
        Self::new(ConnectivityBackend::default())
    }
}

#[cfg(test)]
mod backend_equivalence_tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// `ConnectivityStructure::EulerTour` and `::Polylog` must agree on
    /// `is_connected()`/`connected()` for identical **insert-only**
    /// sequences: they are two implementations of the same specification
    /// for that case, not two different connectivity policies.
    ///
    /// Deliberately insert-only: see
    /// `delete_heavy_sequences_can_diverge_polylog_is_connected_known_bug`
    /// below for a *found, not fixed* divergence once deletions enter the
    /// sequence, and the 2026-10-07 nightly README's "Correctness
    /// Evidence" section for why this is reported rather than silently
    /// avoided. Every current caller of `ConnectivityStructure`/
    /// `ConnectivityBackend` in this codebase (`RuVectorGraphAnalyzer::new`,
    /// `MincutGatedForgetting::boundary_from_one_partition`) builds a fresh
    /// graph and only ever inserts — this test's scope matches that real
    /// usage honestly rather than claiming broader equivalence than was
    /// verified.
    #[test]
    fn backends_agree_on_random_insert_only_sequences() {
        for seed in [1u64, 2, 3, 4, 5] {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut euler = ConnectivityStructure::new(ConnectivityBackend::EulerTour);
            let mut polylog = ConnectivityStructure::new(ConnectivityBackend::Polylog);

            let n: u64 = 30;

            for step in 0..200 {
                let u = rng.gen_range(0..n);
                let v = rng.gen_range(0..n);
                if u != v {
                    euler.insert_edge(u, v);
                    polylog.insert_edge(u, v);
                }

                assert_eq!(
                    euler.is_connected(),
                    polylog.is_connected(),
                    "seed={seed} step={step}: is_connected() disagreement after insert-only sequence"
                );

                for probe in 0..5 {
                    let a = rng.gen_range(0..n);
                    let b = rng.gen_range(0..n);
                    // `a == b` against a vertex neither backend has ever
                    // seen is excluded here: see
                    // `self_query_on_never_inserted_vertex_is_a_known_divergence`
                    // below for why that one specific case is a documented,
                    // bounded divergence rather than a bug this test should
                    // catch.
                    if a == b {
                        continue;
                    }
                    assert_eq!(
                        euler.connected(a, b),
                        polylog.connected(a, b),
                        "seed={seed} step={step} probe={probe}: connected({a},{b}) disagreement"
                    );
                }
            }
        }
    }

    /// **Found, not fixed, pre-existing bug** in
    /// `PolylogConnectivity::delete_edge`'s replacement-edge search
    /// (`find_replacement`), discovered while developing this run's
    /// equivalence test: on a sequence mixing insertions and deletions,
    /// `PolylogConnectivity::is_connected()` can disagree with
    /// `DynamicConnectivity::is_connected()` (which recomputes from a full
    /// rebuild of the current edge set on every delete, i.e. is ground
    /// truth). Observed concretely at `seed=1, step=160` of a 200-step
    /// insert/delete sequence (`insert` probability 0.7, `delete` 0.3,
    /// `n=30` vertices) during this run's development: `DynamicConnectivity`
    /// reported `true`, `PolylogConnectivity` reported `false` — i.e.
    /// `PolylogConnectivity` under-reported connectivity after a deletion
    /// where a valid replacement edge existed. Not root-caused or fixed in
    /// this run (out of scope: this run's claim is about the connectivity
    /// backend choice's effect on `partition()` latency, not an audit of
    /// `PolylogConnectivity`'s own correctness) — filed as a concrete
    /// limitation and follow-up instead. This test pins that the
    /// divergence is reproducible (not a one-off flake) rather than
    /// silently asserting equivalence the insert-only test above does not
    /// cover; if a future fix to `PolylogConnectivity::delete_edge` makes
    /// this test fail (no divergence found), that is good news and this
    /// test should be deleted, not "fixed" to pass again.
    #[test]
    fn delete_heavy_sequences_can_diverge_polylog_is_connected_known_bug() {
        let mut found_divergence = false;

        'seeds: for seed in [1u64, 2, 3, 4, 5] {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut euler = ConnectivityStructure::new(ConnectivityBackend::EulerTour);
            let mut polylog = ConnectivityStructure::new(ConnectivityBackend::Polylog);

            let n: u64 = 30;
            let mut live_edges: Vec<(u64, u64)> = Vec::new();

            for _step in 0..200 {
                let do_insert = live_edges.is_empty() || rng.gen_bool(0.7);
                if do_insert {
                    let u = rng.gen_range(0..n);
                    let v = rng.gen_range(0..n);
                    if u != v {
                        euler.insert_edge(u, v);
                        polylog.insert_edge(u, v);
                        live_edges.push((u, v));
                    }
                } else {
                    let idx = rng.gen_range(0..live_edges.len());
                    let (u, v) = live_edges.remove(idx);
                    euler.delete_edge(u, v);
                    polylog.delete_edge(u, v);
                }

                if euler.is_connected() != polylog.is_connected() {
                    found_divergence = true;
                    break 'seeds;
                }
            }
        }

        assert!(
            found_divergence,
            "expected to reproduce the known PolylogConnectivity::delete_edge \
             is_connected() divergence on at least one of the fixed seeds; \
             if this now fails, the underlying bug may have been fixed — \
             see this test's doc comment before re-enabling the stronger \
             insert+delete equivalence claim"
        );
    }

    /// Documented, bounded divergence (found by this run's equivalence
    /// test, not fixed — pre-existing in both backends, out of this run's
    /// scope): `connected(v, v)` for a vertex `v` that was never inserted
    /// into either structure disagrees between backends.
    /// `DynamicConnectivity::find`/`connected` requires `v` to be a key in
    /// `self.parent` and returns `false` otherwise; `PolylogConnectivity`'s
    /// `LevelForest::find` returns `v` itself (identity) for an unknown
    /// vertex without inserting it, so `find(v) == find(v)` trivially holds
    /// and `connected(v, v)` returns `true`. Neither is "wrong" per either
    /// struct's own doc comments (neither documents behavior for a vertex
    /// outside its tracked universe); this test pins the divergence so a
    /// future caller relying on cross-backend equivalence for this exact
    /// edge case is not surprised by it.
    #[test]
    fn self_query_on_never_inserted_vertex_is_a_known_divergence() {
        let mut euler = ConnectivityStructure::new(ConnectivityBackend::EulerTour);
        let mut polylog = ConnectivityStructure::new(ConnectivityBackend::Polylog);
        // Touch some other vertices so the structures are non-empty, but
        // never insert vertex 999.
        euler.insert_edge(0, 1);
        polylog.insert_edge(0, 1);

        assert!(
            !euler.connected(999, 999),
            "DynamicConnectivity::connected(v,v) on an unseen vertex is false"
        );
        assert!(
            polylog.connected(999, 999),
            "PolylogConnectivity::connected(v,v) on an unseen vertex is true \
             (identity via LevelForest::find's unknown-vertex fallback) — \
             the documented divergence this test pins"
        );
    }

    #[test]
    fn backend_accessor_reports_selected_backend() {
        assert_eq!(
            ConnectivityStructure::new(ConnectivityBackend::EulerTour).backend(),
            ConnectivityBackend::EulerTour
        );
        assert_eq!(
            ConnectivityStructure::new(ConnectivityBackend::Polylog).backend(),
            ConnectivityBackend::Polylog
        );
        assert_eq!(
            ConnectivityStructure::default().backend(),
            ConnectivityBackend::EulerTour
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let dc = DynamicConnectivity::new();
        assert_eq!(dc.vertex_count(), 0);
        assert_eq!(dc.component_count(), 0);
    }

    #[test]
    fn test_add_vertex() {
        let mut dc = DynamicConnectivity::new();

        dc.add_vertex(0);
        assert_eq!(dc.vertex_count(), 1);
        assert_eq!(dc.component_count(), 1);

        dc.add_vertex(1);
        assert_eq!(dc.vertex_count(), 2);
        assert_eq!(dc.component_count(), 2);

        // Adding same vertex is no-op
        dc.add_vertex(0);
        assert_eq!(dc.vertex_count(), 2);
        assert_eq!(dc.component_count(), 2);
    }

    #[test]
    fn test_insert_edge_basic() {
        let mut dc = DynamicConnectivity::new();

        dc.insert_edge(0, 1);
        assert_eq!(dc.vertex_count(), 2);
        assert_eq!(dc.component_count(), 1);
        assert!(dc.connected(0, 1));
    }

    #[test]
    fn test_insert_edge_chain() {
        let mut dc = DynamicConnectivity::new();

        dc.insert_edge(0, 1);
        dc.insert_edge(1, 2);
        dc.insert_edge(2, 3);

        assert_eq!(dc.vertex_count(), 4);
        assert_eq!(dc.component_count(), 1);
        assert!(dc.connected(0, 3));
    }

    #[test]
    fn test_is_connected() {
        let mut dc = DynamicConnectivity::new();

        dc.add_vertex(0);
        dc.add_vertex(1);
        assert!(!dc.is_connected());

        dc.insert_edge(0, 1);
        assert!(dc.is_connected());

        dc.add_vertex(2);
        assert!(!dc.is_connected());

        dc.insert_edge(1, 2);
        assert!(dc.is_connected());
    }

    #[test]
    fn test_delete_edge() {
        let mut dc = DynamicConnectivity::new();

        dc.insert_edge(0, 1);
        dc.insert_edge(1, 2);
        assert!(dc.connected(0, 2));

        dc.delete_edge(1, 2);
        assert!(dc.connected(0, 1));
        assert!(!dc.connected(0, 2));
        assert_eq!(dc.component_count(), 2);
    }

    #[test]
    fn test_delete_edge_normalized() {
        let mut dc = DynamicConnectivity::new();

        dc.insert_edge(0, 1);
        assert!(dc.connected(0, 1));

        // Delete with reversed vertices
        dc.delete_edge(1, 0);
        assert!(!dc.connected(0, 1));
    }

    #[test]
    fn test_multiple_components() {
        let mut dc = DynamicConnectivity::new();

        // Component 1: 0-1-2
        dc.insert_edge(0, 1);
        dc.insert_edge(1, 2);

        // Component 2: 3-4
        dc.insert_edge(3, 4);

        // Isolated vertex
        dc.add_vertex(5);

        assert_eq!(dc.vertex_count(), 6);
        assert_eq!(dc.component_count(), 3);

        assert!(dc.connected(0, 2));
        assert!(dc.connected(3, 4));
        assert!(!dc.connected(0, 3));
        assert!(!dc.connected(0, 5));
    }

    #[test]
    fn test_path_compression() {
        let mut dc = DynamicConnectivity::new();

        // Create a long chain
        for i in 0..10 {
            dc.insert_edge(i, i + 1);
        }

        // Path compression should happen on find
        assert!(dc.connected(0, 10));

        // All vertices should now point closer to root
        let root = dc.find(0);
        for i in 0..=10 {
            assert_eq!(dc.find(i), root);
        }
    }

    #[test]
    fn test_union_by_rank() {
        let mut dc = DynamicConnectivity::new();

        // Create two trees of different sizes
        dc.insert_edge(0, 1);
        dc.insert_edge(0, 2);
        dc.insert_edge(0, 3);

        dc.insert_edge(4, 5);

        // Union them
        dc.insert_edge(0, 4);

        assert_eq!(dc.component_count(), 1);
        assert!(dc.connected(1, 5));
    }

    #[test]
    fn test_rebuild_after_multiple_deletions() {
        let mut dc = DynamicConnectivity::new();

        // Create a complete graph K4
        dc.insert_edge(0, 1);
        dc.insert_edge(0, 2);
        dc.insert_edge(0, 3);
        dc.insert_edge(1, 2);
        dc.insert_edge(1, 3);
        dc.insert_edge(2, 3);

        assert!(dc.is_connected());

        // Remove edges to disconnect
        dc.delete_edge(0, 1);
        dc.delete_edge(0, 2);
        dc.delete_edge(0, 3);

        assert!(!dc.is_connected());
        assert_eq!(dc.component_count(), 2);
        assert!(!dc.connected(0, 1));
        assert!(dc.connected(1, 2));
        assert!(dc.connected(1, 3));
    }

    #[test]
    fn test_connected_nonexistent_vertex() {
        let mut dc = DynamicConnectivity::new();

        dc.add_vertex(0);
        assert!(!dc.connected(0, 999));
        assert!(!dc.connected(999, 0));
    }

    #[test]
    fn test_self_loop() {
        let mut dc = DynamicConnectivity::new();

        dc.insert_edge(0, 0);
        assert_eq!(dc.vertex_count(), 1);
        assert_eq!(dc.component_count(), 1);
        assert!(dc.connected(0, 0));
    }

    #[test]
    fn test_duplicate_edges() {
        let mut dc = DynamicConnectivity::new();

        dc.insert_edge(0, 1);
        dc.insert_edge(0, 1); // Duplicate
        dc.insert_edge(1, 0); // Duplicate (reversed)

        assert_eq!(dc.vertex_count(), 2);
        assert_eq!(dc.component_count(), 1);
    }
}
