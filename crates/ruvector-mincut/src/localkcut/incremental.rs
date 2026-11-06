//! Incremental-boundary variant of the deterministic local k-cut search.
//!
//! # Root cause this closes
//!
//! `DeterministicLocalKCut::deterministic_bfs` (`paper_impl.rs`) recomputes
//! the boundary of the growing BFS-visited set **from scratch** — a full
//! scan of every visited vertex's incident edges, deduplicated into a fresh
//! `HashSet<EdgeId>` — at *every* BFS layer (`calculate_boundary`, called
//! once per `depth` in `0..=radius`). For a visited set that reaches size
//! `v` over `d` layers, that is `O(d * v * avg_degree)` work for a
//! traversal whose boundary could be tracked in `O(v * avg_degree)` total:
//! each edge needs to be classified (crossing / internal) exactly once,
//! right when either of its endpoints joins `visited`, not re-derived from
//! nothing at every subsequent layer.
//!
//! This is the mechanism named as the remaining blocker for
//! `MincutGatedForgetting` (ADR-345) in the 2026-09-11 nightly's Next
//! Research item 2 ("root-cause the `partition()` latency scaling
//! problem... likely by evaluating `connectivity::polylog::PolylogConnectivity`
//! ... as a replacement backend"). `PolylogConnectivity` turned out not to
//! apply directly: it answers *pairwise connectivity* queries, not minimum
//! cut boundary size, so it cannot replace `BoundedInstance`'s oracle. This
//! module instead fixes the actual bottleneck the profiling in this
//! nightly's README located: the per-layer boundary recompute.
//!
//! # Determinism and equivalence
//!
//! `IncrementalLocalKCut` performs the *exact same* deterministic BFS as
//! `DeterministicLocalKCut` — same seed handling (sorted), same per-layer
//! vertex-sort-then-expand order, same accept/early-exit rules — so it is
//! validated to produce byte-identical `(cut_value, witness membership)`
//! output on every graph in
//! `crates/ruvector-mincut/tests/incremental_oracle_equivalence.rs`. It does
//! not touch `DeterministicLocalKCut`, is not used by any existing caller,
//! and only takes effect when `BoundedInstance::with_strategy` opts a
//! *new* instance into it (see `instance::bounded::CutSearchStrategy`).
//!
//! It also decouples the oracle from `DynamicGraph`: it searches a plain
//! `GraphView` borrow of the adjacency lists `BoundedInstance` already
//! maintains, avoiding `search_for_cuts`'s legacy behavior of rebuilding a
//! fresh `DynamicGraph` (with its own locking and allocation) on every
//! single query.

use crate::graph::{EdgeId, VertexId};
use crate::instance::WitnessHandle;
use roaring::RoaringBitmap;
use std::collections::{HashMap, HashSet};

/// Borrowed view over a graph's vertex set and adjacency list.
///
/// Lets the incremental oracle search `BoundedInstance`'s own
/// `vertices`/`adjacency` maps directly, with no `DynamicGraph` rebuild.
#[derive(Debug, Clone, Copy)]
pub struct GraphView<'a> {
    vertices: &'a HashSet<VertexId>,
    adjacency: &'a HashMap<VertexId, Vec<(VertexId, EdgeId)>>,
}

impl<'a> GraphView<'a> {
    /// Borrow a vertex set and adjacency list as a `GraphView`.
    pub fn new(
        vertices: &'a HashSet<VertexId>,
        adjacency: &'a HashMap<VertexId, Vec<(VertexId, EdgeId)>>,
    ) -> Self {
        Self {
            vertices,
            adjacency,
        }
    }

    fn has_vertex(&self, v: VertexId) -> bool {
        self.vertices.contains(&v)
    }

    fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    fn neighbors(&self, v: VertexId) -> &[(VertexId, EdgeId)] {
        self.adjacency.get(&v).map_or(&[], |n| n.as_slice())
    }
}

/// Outcome of one seed's incremental BFS: the best (smallest) boundary
/// found within `radius` layers, admissible under `budget`, plus the
/// vertex set that achieved it.
type BfsOutcome = Option<(HashSet<VertexId>, u64)>;

/// Deterministic local k-cut search with incremental boundary tracking.
///
/// See the module doc for the root cause this closes and the equivalence
/// guarantee it is held to.
#[derive(Debug, Clone)]
pub struct IncrementalLocalKCut {
    max_radius: usize,
}

impl IncrementalLocalKCut {
    /// Create a new incremental-boundary oracle with the given max radius.
    pub fn new(max_radius: usize) -> Self {
        Self { max_radius }
    }

    /// When a vertex newly joins `visited`, classify each of its incident
    /// edges in O(1) amortized time: an edge to an already-visited
    /// neighbor just became internal (drop it from `boundary`); an edge to
    /// a not-yet-visited neighbor just became a crossing edge (add it).
    fn absorb_vertex(
        view: &GraphView,
        v: VertexId,
        visited: &HashSet<VertexId>,
        boundary: &mut HashSet<EdgeId>,
    ) {
        for &(neighbor, edge_id) in view.neighbors(v) {
            if visited.contains(&neighbor) {
                boundary.remove(&edge_id);
            } else {
                boundary.insert(edge_id);
            }
        }
    }

    /// Deterministic BFS from `seeds`, bounded by `budget` and `radius`,
    /// tracking the crossing-edge set incrementally instead of
    /// recomputing it from scratch at every layer.
    ///
    /// Produces the identical `(visited_set, boundary_size)` as
    /// `DeterministicLocalKCut::deterministic_bfs` would for the same
    /// inputs — see the differential equivalence test suite.
    fn deterministic_bfs_incremental(
        &self,
        view: &GraphView,
        seeds: &[VertexId],
        budget: u64,
        radius: usize,
    ) -> BfsOutcome {
        if seeds.is_empty() {
            return None;
        }

        let mut visited: HashSet<VertexId> = HashSet::new();
        let mut boundary: HashSet<EdgeId> = HashSet::new();
        let mut best_cut: BfsOutcome = None;

        // Seed the frontier. Sorted for the same determinism reason as the
        // baseline: insertion order must not depend on an unstable hash
        // iteration order.
        let mut sorted_seeds: Vec<VertexId> = seeds
            .iter()
            .copied()
            .filter(|&s| view.has_vertex(s))
            .collect();
        sorted_seeds.sort_unstable();
        sorted_seeds.dedup();

        if sorted_seeds.is_empty() {
            return None;
        }

        let mut current_layer: Vec<VertexId> = Vec::with_capacity(sorted_seeds.len());
        for &s in &sorted_seeds {
            if visited.insert(s) {
                Self::absorb_vertex(view, s, &visited, &mut boundary);
                current_layer.push(s);
            }
        }

        for depth in 0..=radius {
            let boundary_size = boundary.len() as u64;

            if boundary_size <= budget && !visited.is_empty() && visited.len() < view.num_vertices()
            {
                let should_update = match &best_cut {
                    None => true,
                    Some((_, prev_boundary)) => boundary_size < *prev_boundary,
                };
                if should_update {
                    best_cut = Some((visited.clone(), boundary_size));
                }
            }

            if let Some((_, b)) = &best_cut {
                if *b == 0 {
                    break;
                }
            }

            if depth >= radius {
                break;
            }

            let mut layer_vertices = current_layer.clone();
            layer_vertices.sort_unstable();
            let mut next_layer = Vec::new();

            for v in layer_vertices {
                let mut neighbors: Vec<VertexId> = view
                    .neighbors(v)
                    .iter()
                    .map(|&(n, _)| n)
                    .filter(|n| !visited.contains(n))
                    .collect();
                neighbors.sort_unstable();

                for n in neighbors {
                    if visited.insert(n) {
                        Self::absorb_vertex(view, n, &visited, &mut boundary);
                        next_layer.push(n);
                    }
                }
            }

            current_layer = next_layer;
            if current_layer.is_empty() {
                break;
            }
        }

        best_cut
    }

    fn create_witness(
        seed: VertexId,
        vertices: &HashSet<VertexId>,
        boundary_size: u64,
    ) -> WitnessHandle {
        let mut membership = RoaringBitmap::new();
        for &v in vertices {
            if v <= u32::MAX as u64 {
                membership.insert(v as u32);
            }
        }
        WitnessHandle::new(seed, membership, boundary_size)
    }

    /// Search for a local minimum cut at a single `(seed, budget)` point,
    /// mirroring `DeterministicLocalKCut::search`'s single-query contract.
    pub fn search_single(
        &self,
        view: &GraphView,
        seed: VertexId,
        budget_k: u64,
        radius: usize,
    ) -> Option<(u64, WitnessHandle)> {
        let radius = radius.min(self.max_radius);
        let (vertices, boundary_size) =
            self.deterministic_bfs_incremental(view, &[seed], budget_k, radius)?;
        let witness = Self::create_witness(seed, &vertices, boundary_size);
        Some((boundary_size, witness))
    }

    /// Search for the seed's **global** best boundary within `radius`
    /// layers — equivalent to calling `search_single` with
    /// `budget_k = u64::MAX` (every layer admissible), which is what
    /// collapsing the legacy per-integer-budget loop relies on: budget
    /// only ever gates *acceptance* of a layer, never which layers get
    /// visited, so the global-budget result already contains the answer
    /// for every smaller budget (see module doc and
    /// `instance::bounded::CutSearchStrategy::IncrementalOracleCollapsedBudget`).
    pub fn search_global(
        &self,
        view: &GraphView,
        seed: VertexId,
        radius: usize,
    ) -> Option<(u64, WitnessHandle)> {
        self.search_single(view, seed, u64::MAX, radius)
    }
}

impl Default for IncrementalLocalKCut {
    fn default() -> Self {
        Self::new(10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view_of<'a>(
        vertices: &'a HashSet<VertexId>,
        adjacency: &'a HashMap<VertexId, Vec<(VertexId, EdgeId)>>,
    ) -> GraphView<'a> {
        GraphView::new(vertices, adjacency)
    }

    fn build(
        edges: &[(VertexId, VertexId, EdgeId)],
    ) -> (
        HashSet<VertexId>,
        HashMap<VertexId, Vec<(VertexId, EdgeId)>>,
    ) {
        let mut vertices = HashSet::new();
        let mut adjacency: HashMap<VertexId, Vec<(VertexId, EdgeId)>> = HashMap::new();
        for &(u, v, eid) in edges {
            vertices.insert(u);
            vertices.insert(v);
            adjacency.entry(u).or_default().push((v, eid));
            adjacency.entry(v).or_default().push((u, eid));
        }
        (vertices, adjacency)
    }

    #[test]
    fn dumbbell_bridge_cut() {
        // Two triangles {1,2,3} and {4,5,6} joined by bridge 3-4.
        let (vertices, adjacency) = build(&[
            (1, 2, 0),
            (2, 3, 1),
            (3, 1, 2),
            (3, 4, 3),
            (4, 5, 4),
            (5, 6, 5),
            (6, 4, 6),
        ]);
        let view = view_of(&vertices, &adjacency);
        let oracle = IncrementalLocalKCut::new(10);

        let (cut_value, witness) = oracle
            .search_single(&view, 1, 3, 10)
            .expect("should find bridge");
        assert_eq!(cut_value, 1);
        assert!(witness.contains(1));
    }

    #[test]
    fn triangle_no_cut_below_two() {
        let (vertices, adjacency) = build(&[(1, 2, 0), (2, 3, 1), (3, 1, 2)]);
        let view = view_of(&vertices, &adjacency);
        let oracle = IncrementalLocalKCut::new(5);

        assert!(oracle.search_single(&view, 1, 1, 3).is_none());
        let (cut_value, _) = oracle
            .search_single(&view, 1, 2, 3)
            .expect("triangle min cut is 2");
        assert_eq!(cut_value, 2);
    }

    #[test]
    fn global_budget_matches_sufficiently_large_explicit_budget() {
        let (vertices, adjacency) = build(&[
            (1, 2, 0),
            (2, 3, 1),
            (3, 1, 2),
            (3, 4, 3),
            (4, 5, 4),
            (5, 6, 5),
            (6, 4, 6),
        ]);
        let view = view_of(&vertices, &adjacency);
        let oracle = IncrementalLocalKCut::new(10);

        let global = oracle.search_global(&view, 1, 10);
        let explicit = oracle.search_single(&view, 1, 1_000_000, 10);
        assert_eq!(global.map(|(v, _)| v), explicit.map(|(v, _)| v));
    }

    #[test]
    fn empty_seed_set_is_none() {
        let (vertices, adjacency) = build(&[(1, 2, 0)]);
        let view = view_of(&vertices, &adjacency);
        let oracle = IncrementalLocalKCut::new(5);
        assert!(oracle
            .deterministic_bfs_incremental(&view, &[], 5, 3)
            .is_none());
    }

    #[test]
    fn invalid_seed_is_none() {
        let (vertices, adjacency) = build(&[(1, 2, 0)]);
        let view = view_of(&vertices, &adjacency);
        let oracle = IncrementalLocalKCut::new(5);
        assert!(oracle.search_single(&view, 999, 10, 5).is_none());
    }
}
