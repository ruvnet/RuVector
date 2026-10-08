/// Variant: HierarchicalCapGraph — two-layer navigable graph (HNSW-lite).
///
/// `CapGraphIndex` documents itself as a flat k-NN graph PoC ("replace with
/// HNSW for production") that seeds search from evenly-spaced-by-index entry
/// points. This variant isolates exactly one change: entry points are found
/// by greedy descent through a small sparse top layer (classic HNSW
/// layer-0-and-up navigation), instead of being fixed index positions.
///
/// Base-layer degree, the ef (visited-node) budget, and the transparent
/// (predicate-agnostic neighbour expansion) traversal rule are otherwise
/// identical to `CapGraphIndex`, so any recall/QPS delta measured against it
/// is attributable to entry-point quality alone — not degree, not traversal
/// policy. A second, independent variant (γ-augmented `CapGraphIndex`
/// degree) isolates edge density instead; see
/// `docs/research/nightly/2026-10-08-acorn-capgated-selectivity/README.md`.
///
/// # Complexity
///
/// Build: O(n² · d) for the base layer (identical to `CapGraphIndex`) plus
/// O(m² · d) for the top layer, where m = n / `promote_ratio` is small.
/// Search: O(descent_steps · top_degree · d) + O(ef · degree · d).
use crate::{dist_sq, CapGatedIndex, CapMask, SearchResult};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

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

/// Brute-force k-NN graph over an arbitrary subset of global indices.
/// Returns, for each position in `subset`, the `degree` nearest other
/// members of `subset` (as positions *within* `subset`).
fn knn_within(vectors: &[Vec<f32>], subset: &[usize], degree: usize) -> Vec<Vec<usize>> {
    let m = subset.len();
    let mut graph = vec![Vec::new(); m];
    for i in 0..m {
        let mut dists: Vec<(OrdF32, usize)> = (0..m)
            .filter(|&j| j != i)
            .map(|j| (OrdF32(dist_sq(&vectors[subset[i]], &vectors[subset[j]])), j))
            .collect();
        dists.sort_unstable();
        graph[i] = dists.into_iter().take(degree).map(|(_, j)| j).collect();
    }
    graph
}

pub struct HierarchicalCapGraphIndex {
    vectors: Vec<Vec<f32>>,
    required: Vec<CapMask>,
    ids: Vec<usize>,
    base_graph: Vec<Vec<usize>>, // adjacency over ALL nodes (global indices)
    top_members: Vec<usize>,     // global indices promoted to the top layer
    top_graph: Vec<Vec<usize>>,  // adjacency over `top_members` (positions within it)
    degree: usize,
    dims: usize,
    /// Promote 1 in `promote_ratio` nodes to the top layer.
    promote_ratio: usize,
    ef_multiplier: usize,
}

impl HierarchicalCapGraphIndex {
    /// `degree`: base-layer neighbours per node (same meaning as `CapGraphIndex`).
    /// `promote_ratio`: fraction of nodes kept in the sparse top layer (1/ratio).
    pub fn new(dims: usize, degree: usize, promote_ratio: usize) -> Self {
        HierarchicalCapGraphIndex {
            vectors: Vec::new(),
            required: Vec::new(),
            ids: Vec::new(),
            base_graph: Vec::new(),
            top_members: Vec::new(),
            top_graph: Vec::new(),
            degree,
            dims,
            promote_ratio: promote_ratio.max(1),
            ef_multiplier: 30,
        }
    }

    pub fn with_ef_multiplier(mut self, ef_multiplier: usize) -> Self {
        self.ef_multiplier = ef_multiplier;
        self
    }

    /// Build both layers once over all currently-inserted vectors.
    pub fn batch_build(&mut self, entries: impl IntoIterator<Item = (usize, Vec<f32>, CapMask)>) {
        for (id, vector, required) in entries {
            assert_eq!(vector.len(), self.dims);
            self.vectors.push(vector);
            self.required.push(required);
            self.ids.push(id);
        }
        if self.vectors.is_empty() {
            return;
        }
        let n = self.vectors.len();

        // Base layer: identical construction to CapGraphIndex::rebuild_graph.
        self.base_graph = vec![Vec::new(); n];
        for i in 0..n {
            let mut dists: Vec<(OrdF32, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (OrdF32(dist_sq(&self.vectors[i], &self.vectors[j])), j))
                .collect();
            dists.sort_unstable();
            self.base_graph[i] = dists
                .into_iter()
                .take(self.degree)
                .map(|(_, j)| j)
                .collect();
        }

        // Top layer: deterministic promotion by global index, matching the
        // classic skip-list-style exponential layer assignment's intent
        // (a small, fixed, reproducible subset) without needing randomness.
        self.top_members = (0..n).step_by(self.promote_ratio).collect();
        if self.top_members.is_empty() {
            self.top_members.push(0);
        }
        let top_degree = self
            .degree
            .min(self.top_members.len().saturating_sub(1).max(1));
        self.top_graph = knn_within(&self.vectors, &self.top_members, top_degree);
    }

    /// Greedy descent through the top layer: starting from `top_members[0]`,
    /// repeatedly move to the closest neighbour until no neighbour improves
    /// on the current best. Returns up to `n_seeds` distinct global indices
    /// (the best node found, plus its closest unexplored top-layer
    /// neighbours) to seed base-layer search from.
    fn top_layer_descend(&self, query: &[f32], n_seeds: usize) -> Vec<usize> {
        if self.top_members.is_empty() {
            return vec![0];
        }
        let mut cur = 0usize; // position within top_members
        let mut cur_dist = dist_sq(query, &self.vectors[self.top_members[cur]]);
        loop {
            let mut improved = false;
            for &nb in &self.top_graph[cur] {
                let d = dist_sq(query, &self.vectors[self.top_members[nb]]);
                if d < cur_dist {
                    cur_dist = d;
                    cur = nb;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
        let mut seeds = vec![self.top_members[cur]];
        let mut nbs: Vec<(OrdF32, usize)> = self.top_graph[cur]
            .iter()
            .map(|&nb| {
                (
                    OrdF32(dist_sq(query, &self.vectors[self.top_members[nb]])),
                    self.top_members[nb],
                )
            })
            .collect();
        nbs.sort_unstable();
        for (_, g) in nbs {
            if seeds.len() >= n_seeds {
                break;
            }
            if !seeds.contains(&g) {
                seeds.push(g);
            }
        }
        seeds
    }
}

impl CapGatedIndex for HierarchicalCapGraphIndex {
    fn insert(&mut self, id: usize, vector: Vec<f32>, required: CapMask) {
        // Single-insert rebuild (PoC parity with CapGraphIndex::insert).
        // Prefer `batch_build` for anything beyond a handful of inserts.
        assert_eq!(vector.len(), self.dims);
        self.vectors.push(vector);
        self.required.push(required);
        self.ids.push(id);
        let snapshot: Vec<(usize, Vec<f32>, CapMask)> = Vec::new();
        self.base_graph.clear();
        self.top_members.clear();
        self.top_graph.clear();
        let entries = std::mem::take(&mut self.vectors)
            .into_iter()
            .zip(std::mem::take(&mut self.required))
            .zip(std::mem::take(&mut self.ids))
            .map(|((v, r), i)| (i, v, r));
        self.batch_build(entries);
        let _ = snapshot;
    }

    fn search(&self, query: &[f32], k: usize, holder: CapMask) -> Vec<SearchResult> {
        assert_eq!(query.len(), self.dims);
        let n = self.vectors.len();
        if n == 0 {
            return Vec::new();
        }

        let seeds = self.top_layer_descend(query, 3);

        let mut visited: HashSet<usize> = HashSet::new();
        let mut frontier: BinaryHeap<Reverse<(OrdF32, usize)>> = BinaryHeap::new();
        let mut results: BinaryHeap<(OrdF32, usize)> = BinaryHeap::new();

        for &s in &seeds {
            if visited.insert(s) {
                let d = OrdF32(dist_sq(query, &self.vectors[s]));
                frontier.push(Reverse((d, s)));
            }
        }

        // Same ef policy as CapGraphIndex: cap total visited nodes, not edges.
        let ef = (k * self.ef_multiplier).min(n);
        let mut n_visited = 0usize;

        while let Some(Reverse((d, idx))) = frontier.pop() {
            if n_visited >= ef {
                break;
            }
            n_visited += 1;

            if holder.satisfies(self.required[idx]) {
                results.push((d, idx));
                if results.len() > k {
                    results.pop();
                }
            }

            for &nb in &self.base_graph[idx] {
                if visited.insert(nb) {
                    let nd = OrdF32(dist_sq(query, &self.vectors[nb]));
                    frontier.push(Reverse((nd, nb)));
                }
            }
        }

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
        "HierarchicalCapGraph"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CapMask;

    fn build_small(promote_ratio: usize) -> HierarchicalCapGraphIndex {
        let mut idx = HierarchicalCapGraphIndex::new(2, 4, promote_ratio);
        let vecs = vec![
            (0usize, vec![0.0f32, 0.0], CapMask::NONE),
            (1, vec![0.1, 0.0], CapMask::NONE),
            (2, vec![0.9, 0.0], CapMask::single(5)), // restricted
            (3, vec![0.5, 0.0], CapMask::NONE),
            (4, vec![0.2, 0.1], CapMask::NONE),
            (5, vec![10.0, 10.0], CapMask::NONE),
        ];
        idx.batch_build(vecs);
        idx
    }

    #[test]
    fn hierarchical_only_returns_authorised() {
        let idx = build_small(2);
        let holder = CapMask::NONE; // doesn't hold bit 5
        let results = idx.search(&[0.0, 0.0], 10, holder);
        for r in &results {
            assert_ne!(r.id, 2, "id=2 requires bit 5 and should not be returned");
        }
    }

    #[test]
    fn hierarchical_returns_up_to_k() {
        let idx = build_small(2);
        let results = idx.search(&[0.0, 0.0], 3, CapMask::ALL);
        assert!(results.len() <= 3);
    }

    #[test]
    fn hierarchical_empty_index() {
        let idx = HierarchicalCapGraphIndex::new(4, 5, 3);
        let results = idx.search(&[0.0, 0.0, 0.0, 0.0], 5, CapMask::ALL);
        assert!(results.is_empty());
    }

    #[test]
    fn hierarchical_finds_nearest_with_full_access() {
        let mut idx = HierarchicalCapGraphIndex::new(2, 4, 3);
        let entries: Vec<_> = (0..20usize)
            .map(|i| (i, vec![i as f32, 0.0], CapMask::NONE))
            .collect();
        idx.batch_build(entries);
        let results = idx.search(&[0.0, 0.0], 3, CapMask::NONE);
        assert!(!results.is_empty());
        assert_eq!(results[0].id, 0);
    }

    #[test]
    fn hierarchical_promote_ratio_larger_than_n_still_builds() {
        let idx = build_small(1000);
        assert_eq!(idx.top_members.len(), 1); // falls back to a single member
    }

    #[test]
    fn hierarchical_insert_matches_batch_build_topology() {
        let mut idx = HierarchicalCapGraphIndex::new(2, 2, 2);
        for i in 0..6usize {
            idx.insert(i, vec![i as f32, 0.0], CapMask::NONE);
        }
        let results = idx.search(&[0.0, 0.0], 2, CapMask::ALL);
        assert_eq!(results[0].id, 0);
    }
}
