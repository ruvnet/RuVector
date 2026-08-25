//! Flat k-NN proximity graph — equivalent to HNSW layer-0.
//!
//! Same core construction as `ruvector-entropy-ann::graph::FlatGraph`
//! (ADR-303): an exact per-node k-NN graph, O(n^2 * dim) to build, suitable
//! for PoC datasets up to ~50k vectors.
//!
//! **Entry routing differs from ADR-303 on purpose.** entropy-ann finds each
//! query's entry point by an O(n) brute-force scan; that is deliberately
//! excluded here because it would swamp the metric this crate measures
//! (distance computations spent on beam *traversal*): at N=2000 the O(n)
//! entry scan alone would be an order of magnitude larger than any
//! traversal-stopping-rule difference under test.
//!
//! A single *fixed* entry point (the earlier design of this crate) is wrong
//! for a different reason, discovered by this crate's own test suite: an
//! exact k-NN graph over well-separated clusters has few or no edges
//! *between* clusters, so a single fixed entry point cannot reach clusters
//! other than its own — recall on this test's clustered dataset was ~19%,
//! not because of the stopping rule, but because most queries were
//! structurally unreachable from the one entry point. See the research
//! README's attack pass for the full account.
//!
//! The fix used here — `entry_seeds`, a small deterministic sample of nodes
//! probed at query time (O(seeds), not O(n)) to pick the nearest as the
//! traversal entry point — approximates a coarse HNSW upper-layer routing
//! step without building a real multi-layer index.

use crate::dataset::l2sq;

/// Configuration for graph construction.
#[derive(Clone, Debug)]
pub struct GraphConfig {
    /// Number of neighbours per node in the built graph.
    pub k_neighbours: usize,
    /// Number of deterministic entry-point candidates probed per query.
    pub num_entry_seeds: usize,
}

impl Default for GraphConfig {
    fn default() -> Self {
        GraphConfig {
            k_neighbours: 16,
            num_entry_seeds: 32,
        }
    }
}

/// A single-layer k-NN proximity graph over f32 vectors, with a small
/// deterministic set of entry-routing seed nodes computed at build time.
pub struct FlatGraph {
    pub vectors: Vec<Vec<f32>>,
    /// adjacency[i] = sorted list of (dist^2, neighbour_id) for node i
    pub adjacency: Vec<Vec<(f32, usize)>>,
    /// Deterministic sample of node ids probed at query time to pick a
    /// traversal entry point (nearest of these to the query).
    pub entry_seeds: Vec<usize>,
    pub config: GraphConfig,
}

impl FlatGraph {
    /// Build the graph from a vector corpus.
    pub fn build(vectors: Vec<Vec<f32>>, config: GraphConfig) -> Self {
        let n = vectors.len();
        let k = config.k_neighbours.min(n.saturating_sub(1));

        let mut adjacency: Vec<Vec<(f32, usize)>> = vec![Vec::with_capacity(k); n];

        for i in 0..n {
            let mut dists: Vec<(f32, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (l2sq(&vectors[i], &vectors[j]), j))
                .collect();
            dists.sort_unstable_by(|a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            dists.truncate(k);
            adjacency[i] = dists;
        }

        let entry_seeds = Self::compute_entry_seeds(n, config.num_entry_seeds);

        FlatGraph {
            vectors,
            adjacency,
            entry_seeds,
            config,
        }
    }

    /// Deterministic (seeded, not corpus-content-dependent) sample of up to
    /// `count` distinct node indices in `[0, n)`, used as entry-routing
    /// candidates. Fixed seed so results are reproducible across runs.
    fn compute_entry_seeds(n: usize, count: usize) -> Vec<usize> {
        if n == 0 {
            return Vec::new();
        }
        let count = count.min(n);
        let mut state: u64 = 0x5eed_1234_dab5_eed1;
        let mut seen = std::collections::HashSet::with_capacity(count);
        let mut seeds = Vec::with_capacity(count);
        let mut attempts = 0usize;
        // Rejection sampling; bounded attempts guarantees termination even
        // if count == n (every index eventually gets sampled).
        while seeds.len() < count && attempts < count * 50 + n {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let idx = ((state >> 33) as usize) % n;
            if seen.insert(idx) {
                seeds.push(idx);
            }
            attempts += 1;
        }
        if seeds.is_empty() {
            seeds.push(0);
        }
        seeds
    }

    /// Probe the entry-seed set and return the nearest as the traversal
    /// entry point, along with the number of distance computations spent
    /// (O(len(entry_seeds)), not O(n)).
    pub fn route_entry(&self, query: &[f32]) -> (usize, f32, u64) {
        let mut best = self.entry_seeds[0];
        let mut best_d = l2sq(query, &self.vectors[best]);
        let mut count: u64 = 1;
        for &s in &self.entry_seeds[1..] {
            let d = l2sq(query, &self.vectors[s]);
            count += 1;
            if d < best_d {
                best_d = d;
                best = s;
            }
        }
        (best, best_d, count)
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    pub fn dim(&self) -> usize {
        self.vectors.first().map(|v| v.len()).unwrap_or(0)
    }

    pub fn memory_bytes(&self) -> usize {
        let vecs: usize = self.vectors.iter().map(|v| v.len() * 4).sum();
        let adj: usize = self.adjacency.iter().map(|a| a.len() * 8).sum();
        vecs + adj
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::random_unit_vectors;

    #[test]
    fn graph_has_k_neighbours() {
        let vecs = random_unit_vectors(30, 8, 7);
        let cfg = GraphConfig {
            k_neighbours: 6,
            num_entry_seeds: 8,
        };
        let g = FlatGraph::build(vecs, cfg);
        for adj in &g.adjacency {
            assert_eq!(adj.len(), 6);
        }
    }

    #[test]
    fn entry_seeds_are_valid_indices_and_distinct() {
        let vecs = random_unit_vectors(50, 8, 3);
        let g = FlatGraph::build(vecs, GraphConfig::default());
        assert!(!g.entry_seeds.is_empty());
        for &s in &g.entry_seeds {
            assert!(s < g.len());
        }
        let unique: std::collections::HashSet<_> = g.entry_seeds.iter().collect();
        assert_eq!(unique.len(), g.entry_seeds.len());
    }

    #[test]
    fn entry_seeds_deterministic() {
        let vecs_a = random_unit_vectors(40, 8, 5);
        let vecs_b = random_unit_vectors(40, 8, 5);
        let ga = FlatGraph::build(vecs_a, GraphConfig::default());
        let gb = FlatGraph::build(vecs_b, GraphConfig::default());
        assert_eq!(ga.entry_seeds, gb.entry_seeds);
    }

    #[test]
    fn route_entry_picks_nearest_seed() {
        let vecs = random_unit_vectors(60, 8, 11);
        let g = FlatGraph::build(vecs, GraphConfig::default());
        let query = g.vectors[g.entry_seeds[0]].clone();
        let (best, best_d, count) = g.route_entry(&query);
        assert_eq!(count as usize, g.entry_seeds.len());
        // Querying exactly at a seed's own location must return that seed
        // with distance 0 (it is at least as close as any other seed).
        assert_eq!(best, g.entry_seeds[0]);
        assert!(best_d < 1e-6);
    }
}
