/// Variant 3: CapGraph — capability-pruned proximity graph walk.
///
/// Build a k-NN proximity graph over all stored vectors.  During search,
/// traverse the graph greedily from multiple entry points.  Crucially:
///
///   * A node is ADDED to the result set only if the querier's capability
///     mask satisfies the node's required mask.
///   * A node's neighbours are expanded regardless of the node's capability
///     (to preserve graph connectivity and maintain recall).
///
/// This "transparent traversal" model leaks the traversal pattern but
/// preserves recall.  A "strict isolation" model (expand only authorised
/// nodes) would be stronger security but shows measurable recall reduction
/// — documented in the research notes.
///
/// # Complexity
///
/// Build: O(n² · d) brute-force k-NN — fine for PoC, replace with HNSW for
/// production.
/// Search: O(ef · degree · d) where ef is the exploration frontier.
use crate::{dist_sq, CapGatedIndex, CapMask, SearchResult};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

// Float wrapper that is Ord (assumes no NaN in distance values).
#[derive(Clone, Copy, PartialEq)]
struct OrdF32(f32);
impl Eq for OrdF32 {}
impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

pub struct CapGraphIndex {
    vectors: Vec<Vec<f32>>,
    required: Vec<CapMask>,
    ids: Vec<usize>,
    graph: Vec<Vec<usize>>, // adjacency list: graph[i] = indices of neighbours
    degree: usize,          // max neighbours per node
    dims: usize,
    n_entry_points: usize,
    /// Exploration factor: max nodes to visit during search.
    /// Higher ef → better recall, higher latency.  Set to k * ef_multiplier.
    ef_multiplier: usize,
}

impl CapGraphIndex {
    /// `degree`: number of neighbours per node (higher = better recall, more memory).
    /// `n_entry_points`: starting nodes for greedy search (higher = better recall).
    /// `ef_multiplier`: visit up to k * ef_multiplier nodes (default 30 → ef=300 for k=10).
    pub fn new(dims: usize, degree: usize, n_entry_points: usize) -> Self {
        CapGraphIndex {
            vectors: Vec::new(),
            required: Vec::new(),
            ids: Vec::new(),
            graph: Vec::new(),
            degree,
            dims,
            n_entry_points,
            ef_multiplier: 30,
        }
    }

    pub fn with_ef_multiplier(mut self, ef_multiplier: usize) -> Self {
        self.ef_multiplier = ef_multiplier;
        self
    }

    /// Rebuild the proximity graph after insert.
    ///
    /// For n ≤ 10k this O(n²) rebuild is acceptable.  Production would use
    /// incremental HNSW-style graph maintenance.
    fn rebuild_graph(&mut self) {
        let n = self.vectors.len();
        self.graph = vec![Vec::new(); n];
        for i in 0..n {
            // Find the `degree` nearest neighbours of i (excluding self)
            let mut dists: Vec<(OrdF32, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (OrdF32(dist_sq(&self.vectors[i], &self.vectors[j])), j))
                .collect();
            dists.sort_unstable();
            self.graph[i] = dists
                .into_iter()
                .take(self.degree)
                .map(|(_, j)| j)
                .collect();
        }
    }
}

impl CapGraphIndex {
    /// Batch-insert all entries and build the graph once.
    ///
    /// Always prefer this over calling `insert` in a loop.
    pub fn batch_build(&mut self, entries: impl IntoIterator<Item = (usize, Vec<f32>, CapMask)>) {
        for (id, vector, required) in entries {
            assert_eq!(vector.len(), self.dims);
            self.vectors.push(vector);
            self.required.push(required);
            self.ids.push(id);
        }
        if !self.vectors.is_empty() {
            self.rebuild_graph();
        }
    }
}

impl CapGatedIndex for CapGraphIndex {
    /// Single-vector insert: defers graph rebuild.
    ///
    /// After inserting all vectors call `batch_build` or use `rebuild_graph`
    /// directly.  For PoC benchmarks use `batch_build` to avoid O(n³) cost.
    fn insert(&mut self, id: usize, vector: Vec<f32>, required: CapMask) {
        assert_eq!(vector.len(), self.dims);
        self.vectors.push(vector);
        self.required.push(required);
        self.ids.push(id);
        // Rebuild graph on every insert — O(n³) total.  Production: incremental.
        self.rebuild_graph();
    }

    fn search(&self, query: &[f32], k: usize, holder: CapMask) -> Vec<SearchResult> {
        assert_eq!(query.len(), self.dims);
        let n = self.vectors.len();
        if n == 0 {
            return Vec::new();
        }

        // Greedy best-first search with an exploration frontier.
        // min-heap by distance for candidates, max-heap for result set.
        let mut visited: HashSet<usize> = HashSet::new();
        // (dist, idx) — min-heap
        let mut frontier: BinaryHeap<Reverse<(OrdF32, usize)>> = BinaryHeap::new();
        // result set: max-heap so we can evict the worst easily
        let mut results: BinaryHeap<(OrdF32, usize)> = BinaryHeap::new();

        // Seed entry points: evenly spaced by index
        let step = (n / self.n_entry_points).max(1);
        for ep in (0..n).step_by(step).take(self.n_entry_points) {
            if visited.insert(ep) {
                let d = OrdF32(dist_sq(query, &self.vectors[ep]));
                frontier.push(Reverse((d, ep)));
            }
        }

        // ef (exploration factor): explore at least k * ef_multiplier nodes.
        // This is critical for capability-gated search: when access ratio is
        // low, many nearby nodes are unauthorised and the standard "stop when
        // frontier's best > worst result" heuristic terminates too early.
        let ef = (k * self.ef_multiplier).min(n);
        let mut n_visited = 0usize;

        while let Some(Reverse((d, idx))) = frontier.pop() {
            // Hard ef cap: stop after visiting ef nodes
            if n_visited >= ef {
                break;
            }
            n_visited += 1;

            // Add to results if authorised
            if holder.satisfies(self.required[idx]) {
                results.push((d, idx));
                if results.len() > k {
                    results.pop();
                }
            }

            // Expand neighbours (always, regardless of authorisation)
            for &nb in &self.graph[idx] {
                if visited.insert(nb) {
                    let nd = OrdF32(dist_sq(query, &self.vectors[nb]));
                    frontier.push(Reverse((nd, nb)));
                }
            }
        }

        // Convert to SearchResult, sorted by ascending distance
        let mut out: Vec<SearchResult> = results
            .into_iter()
            .map(|(OrdF32(d), idx)| SearchResult {
                id: self.ids[idx],
                dist_sq: d,
            })
            .collect();
        out.sort_by(|a, b| a.dist_sq.partial_cmp(&b.dist_sq).unwrap());
        out
    }

    fn name(&self) -> &'static str {
        "CapGraph"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CapMask;

    fn build_small_graph() -> CapGraphIndex {
        let mut idx = CapGraphIndex::new(2, 4, 2);
        let vecs = vec![
            (0usize, vec![0.0f32, 0.0], CapMask::NONE),
            (1, vec![0.1, 0.0], CapMask::NONE),
            (2, vec![0.9, 0.0], CapMask::single(5)), // restricted
            (3, vec![0.5, 0.0], CapMask::NONE),
            (4, vec![0.2, 0.1], CapMask::NONE),
        ];
        for (id, v, c) in vecs {
            idx.insert(id, v, c);
        }
        idx
    }

    #[test]
    fn cap_graph_only_returns_authorised() {
        let idx = build_small_graph();
        let holder = CapMask::NONE; // doesn't hold bit 5
        let results = idx.search(&[0.0, 0.0], 10, holder);
        for r in &results {
            assert_ne!(r.id, 2, "id=2 requires bit 5 and should not be returned");
        }
    }

    #[test]
    fn cap_graph_returns_up_to_k() {
        let idx = build_small_graph();
        let results = idx.search(&[0.0, 0.0], 3, CapMask::ALL);
        assert!(results.len() <= 3);
    }

    #[test]
    fn cap_graph_empty_index() {
        let idx = CapGraphIndex::new(4, 5, 3);
        let results = idx.search(&[0.0, 0.0, 0.0, 0.0], 5, CapMask::ALL);
        assert!(results.is_empty());
    }

    #[test]
    fn cap_graph_full_access_finds_something() {
        let mut idx = CapGraphIndex::new(2, 4, 2);
        for i in 0..10usize {
            idx.insert(i, vec![i as f32, 0.0], CapMask::NONE);
        }
        let results = idx.search(&[0.0, 0.0], 3, CapMask::NONE);
        assert!(!results.is_empty());
        // Closest should be id=0 (at [0.0, 0.0])
        assert_eq!(results[0].id, 0);
    }
}
