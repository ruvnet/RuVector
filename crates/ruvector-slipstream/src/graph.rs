//! Flat proximity graph with warm-start insertion support.
//!
//! A single-layer graph where each node stores its M nearest neighbours found
//! at insertion time.  Insertions accept an optional warm-start candidate list;
//! the returned candidate set can be cached for the next insertion.
//!
//! This is HNSW layer-0 extracted as a self-contained building block.  The
//! warm-start mechanism is the core Slipstream innovation; the graph structure
//! itself is unchanged from standard small-world construction.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct GraphConfig {
    /// Max neighbours per node (HNSW M parameter).
    pub m: usize,
    /// Random long-jump links per node for small-world navigability.
    pub m_longjump: usize,
    /// Beam width for insertion search (HNSW ef_construction).
    pub ef: usize,
    /// Vector dimensionality.
    pub dims: usize,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m_longjump: 6,
            ef: 64,
            dims: 128,
        }
    }
}

/// Flat proximity graph: all vectors + adjacency lists in DRAM.
pub struct FlatGraph {
    pub data: Vec<Vec<f32>>,
    pub neighbors: Vec<Vec<u32>>,
    pub config: GraphConfig,
    /// PRNG for random long-jump edge assignment.
    rng: StdRng,
}

impl FlatGraph {
    pub fn new_empty(config: GraphConfig) -> Self {
        Self {
            data: Vec::new(),
            neighbors: Vec::new(),
            config,
            rng: StdRng::seed_from_u64(0xFEED_DEAD_BEEF_CA11),
        }
    }

    /// Pre-build from a static dataset (brute-force k-NN, O(N²·D)).
    pub fn build_static(dataset: Vec<Vec<f32>>, config: GraphConfig) -> Self {
        let n = dataset.len();
        let m = config.m;
        let mut graph = Self::new_empty(config);
        graph.data = dataset;
        graph.neighbors = vec![Vec::new(); n];

        for i in 0..n {
            let mut dists: Vec<(f32, u32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (l2_sq(&graph.data[i], &graph.data[j]), j as u32))
                .collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
            graph.neighbors[i] = dists.iter().take(m).map(|&(_, id)| id).collect();
        }
        graph
    }

    /// Insert a single vector using the provided warm-start candidates.
    ///
    /// Returns the full discovered candidate set (suitable for caching as
    /// the next insertion's warm-start).  If `warm_candidates` is empty,
    /// traversal begins from node 0 (or the sole existing node).
    pub fn insert_warm(&mut self, vec: Vec<f32>, warm_candidates: &[u32]) -> Vec<u32> {
        let new_id = self.data.len() as u32;

        if self.data.is_empty() {
            self.data.push(vec);
            self.neighbors.push(Vec::new());
            return vec![0];
        }

        // Seed the beam search.
        let seeds: Vec<u32> = if warm_candidates.is_empty() {
            vec![0]
        } else {
            // Filter out any stale IDs (shouldn't happen, but guard anyway).
            warm_candidates
                .iter()
                .copied()
                .filter(|&id| (id as usize) < self.data.len())
                .collect()
        };

        let ef = self.config.ef;
        let m = self.config.m;

        // Beam search: find ef nearest existing nodes.
        let (candidates, discovered) = self.beam_search_insert(&vec, ef, &seeds);

        // Select top-M for links.
        let link_targets: Vec<u32> = candidates.iter().take(m).map(|&(_, id)| id).collect();

        // Insert new vector.
        self.data.push(vec);
        self.neighbors.push(link_targets.clone());

        // Add reverse links with degree pruning.
        for &target in &link_targets {
            let t = target as usize;
            self.neighbors[t].push(new_id);
            if self.neighbors[t].len() > m {
                // Prune: keep the M nearest to target.
                let tv = &self.data[t];
                self.neighbors[t].sort_by(|&a, &b| {
                    let da = l2_sq(tv, &self.data[a as usize]);
                    let db = l2_sq(tv, &self.data[b as usize]);
                    da.partial_cmp(&db).unwrap_or(Ordering::Equal)
                });
                self.neighbors[t].truncate(m);
            }
        }

        // Add random long-jump edges for small-world navigability.
        // These are bidirectional and bypass cluster boundaries, enabling
        // beam search to traverse the full graph from any entry point.
        let n_now = self.data.len(); // includes the new node
        let ml = self.config.m_longjump;
        if n_now > 1 {
            let jumps = ml.min(n_now - 1);
            for _ in 0..jumps {
                let j = self.rng.gen_range(0..n_now - 1) as u32; // pick any existing node
                if !self.neighbors[new_id as usize].contains(&j) {
                    self.neighbors[new_id as usize].push(j);
                }
                if !self.neighbors[j as usize].contains(&new_id) {
                    self.neighbors[j as usize].push(new_id);
                }
            }
        }

        discovered
    }

    /// Beam search for **search** (read path, does not modify graph).
    ///
    /// Returns up to `k` nearest nodes sorted by ascending distance.
    pub fn search_from(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        seeds: &[u32],
    ) -> Vec<(f32, usize)> {
        if self.data.is_empty() {
            return Vec::new();
        }
        let valid_seeds: Vec<u32> = seeds
            .iter()
            .copied()
            .filter(|&id| (id as usize) < self.data.len())
            .collect();
        let effective_seeds = if valid_seeds.is_empty() {
            vec![0]
        } else {
            valid_seeds
        };

        let (mut results, _) = self.beam_search_insert(query, ef, &effective_seeds);
        results.truncate(k);
        results
            .into_iter()
            .map(|(d, id)| (d, id as usize))
            .collect()
    }

    /// Core beam search.  Returns (sorted_results, all_discovered_ids).
    fn beam_search_insert(
        &self,
        query: &[f32],
        ef: usize,
        seeds: &[u32],
    ) -> (Vec<(f32, u32)>, Vec<u32>) {
        // visited bitmap using a simple Vec<bool>.
        let n = self.data.len();
        let mut visited = vec![false; n];

        // candidates: min-heap by distance (we want to expand closest first).
        // We negate the key because BinaryHeap is a max-heap.
        let mut candidates: BinaryHeap<OrdF32Pair> = BinaryHeap::new();
        // results: max-heap (tracks ef best found so far, we evict worst).
        let mut results: BinaryHeap<OrdF32Pair> = BinaryHeap::new();

        let mut discovered: Vec<u32> = Vec::new();

        for &seed in seeds {
            if (seed as usize) < n && !visited[seed as usize] {
                visited[seed as usize] = true;
                let d = l2_sq(query, &self.data[seed as usize]);
                candidates.push(OrdF32Pair {
                    neg_dist: -d,
                    id: seed,
                });
                results.push(OrdF32Pair {
                    neg_dist: d,
                    id: seed,
                }); // max-heap: store positive
                discovered.push(seed);
            }
        }

        while let Some(OrdF32Pair { neg_dist, id: curr }) = candidates.pop() {
            let d_curr = -neg_dist;

            // If the current candidate is worse than the ef-th best result, stop.
            if results.len() >= ef {
                if let Some(worst) = results.peek() {
                    if d_curr > worst.neg_dist {
                        break;
                    }
                }
            }

            // Expand neighbours.
            let nbrs = &self.neighbors[curr as usize];
            for &nb in nbrs {
                if (nb as usize) < n && !visited[nb as usize] {
                    visited[nb as usize] = true;
                    discovered.push(nb);
                    let d_nb = l2_sq(query, &self.data[nb as usize]);

                    let add = if results.len() < ef {
                        true
                    } else if let Some(worst) = results.peek() {
                        d_nb < worst.neg_dist
                    } else {
                        false
                    };

                    if add {
                        candidates.push(OrdF32Pair {
                            neg_dist: -d_nb,
                            id: nb,
                        });
                        results.push(OrdF32Pair {
                            neg_dist: d_nb,
                            id: nb,
                        });
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        // Drain results into sorted order (ascending distance).
        let mut sorted: Vec<(f32, u32)> = results.into_iter().map(|p| (p.neg_dist, p.id)).collect();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        (sorted, discovered)
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Float pair for BinaryHeap.  Ordering is by `neg_dist` (larger = higher priority).
/// For the candidates heap we store negative distance, so max-heap = min-distance.
/// For the results heap we store positive distance, so max-heap = max-distance (evict worst).
#[derive(Clone, PartialEq)]
struct OrdF32Pair {
    neg_dist: f32,
    id: u32,
}

impl Eq for OrdF32Pair {}
impl PartialOrd for OrdF32Pair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrdF32Pair {
    fn cmp(&self, other: &Self) -> Ordering {
        self.neg_dist
            .partial_cmp(&other.neg_dist)
            .unwrap_or(Ordering::Equal)
    }
}

/// Squared L2 distance.
#[inline(always)]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Cosine similarity.
#[inline(always)]
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_sq_same_vector_is_zero() {
        let v = vec![1.0f32, 2.0, 3.0];
        assert_eq!(l2_sq(&v, &v), 0.0);
    }

    #[test]
    fn cosine_sim_identical_is_one() {
        let v = vec![1.0f32, 0.0, 0.0];
        assert!((cosine_sim(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn insert_warm_builds_graph() {
        let config = GraphConfig {
            m: 4,
            m_longjump: 2,
            ef: 16,
            dims: 8,
        };
        let mut graph = FlatGraph::new_empty(config);
        for i in 0..20u32 {
            let vec: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32).collect();
            graph.insert_warm(vec, &[]);
        }
        assert_eq!(graph.len(), 20);
        // Each node (except possibly the first) should have some neighbours.
        assert!(!graph.neighbors[10].is_empty());
    }

    #[test]
    fn search_from_returns_k_results() {
        let config = GraphConfig {
            m: 4,
            m_longjump: 2,
            ef: 16,
            dims: 4,
        };
        let mut graph = FlatGraph::new_empty(config);
        for i in 0..50u32 {
            let vec: Vec<f32> = (0..4).map(|d| (i as f32) + d as f32 * 0.01).collect();
            graph.insert_warm(vec, &[]);
        }
        let query = vec![25.0f32, 25.01, 25.02, 25.03];
        let results = graph.search_from(&query, 5, 32, &[0]);
        assert_eq!(results.len(), 5);
        // Nearest neighbour should be close to index 25.
        assert!(results[0].0 < 1.0);
    }
}
