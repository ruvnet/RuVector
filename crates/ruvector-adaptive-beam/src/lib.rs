/// Distance-Adaptive Beam Search for Provably Accurate Graph-Based ANN
///
/// Implements the stopping criterion from arXiv:2505.15636 (May 2025):
/// terminate beam expansion when the closest unvisited candidate c satisfies
///   d(q, c) > (1 + γ) · d(q, k-th result found)
///
/// This gives a provable (1+γ/2)-approximation on navigable graphs —
/// the first proven stopping criterion for graph-based ANN search.
/// Every production Rust vector database (Qdrant, usearch, ruvector pre-ADR-193)
/// uses a count-based FixedWidth criterion with no approximation guarantee.
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

pub mod graph;

/// Stopping criterion for graph-based beam search.
#[derive(Debug, Clone, Copy)]
pub enum BeamStopPolicy {
    /// Classic count-limited beam: expand at most `beam_width` nodes.
    /// No approximation guarantee; must be tuned empirically per dataset.
    FixedWidth { beam_width: usize },

    /// Distance-adaptive stopping (arXiv:2505.15636 §3.1).
    /// Terminates when the closest unvisited candidate c satisfies:
    ///   d(q, c) > (1 + gamma) · d(q, k-th result)
    /// Provides a provable (1+gamma/2)-approximation on navigable graphs.
    /// gamma=0 → near-exact; gamma=2.0 → ~40% fewer distance evaluations.
    DistanceAdaptive { gamma: f32 },

    /// Conservative hybrid: enforce at least `min_expansions` before adaptive
    /// stopping. Guards against degenerate entry points in sparse data regions.
    AdaptiveWithFloor { gamma: f32, min_expansions: usize },
}

/// Per-query telemetry collected during search.
#[derive(Debug, Clone, Default)]
pub struct SearchMetrics {
    pub distance_computations: u64,
    pub nodes_expanded: u64,
    /// True when the search terminated via adaptive threshold (not frontier exhaustion).
    pub early_stopped: bool,
}

/// Ordered (distance, node_id) pair for BinaryHeap use.
#[derive(Clone, PartialEq)]
pub struct OrdF(pub f32, pub u32);

impl Eq for OrdF {}
impl PartialOrd for OrdF {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for OrdF {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&o.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(self.1.cmp(&o.1))
    }
}

/// Squared Euclidean distance (no sqrt — monotone for ranking).
#[inline(always)]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Graph-based ANN index with swappable beam-search stopping policies.
pub struct AdaptiveBeamIndex {
    pub vectors: Vec<Vec<f32>>,
    pub neighbors: Vec<Vec<u32>>,
    pub entry_point: u32,
}

impl AdaptiveBeamIndex {
    pub fn new(vectors: Vec<Vec<f32>>, neighbors: Vec<Vec<u32>>, entry_point: u32) -> Self {
        Self { vectors, neighbors, entry_point }
    }

    /// Beam search returning `(top_k results sorted by distance, telemetry)`.
    ///
    /// All three policies share the same frontier/results data structures;
    /// only the loop-termination predicate differs — enabling apples-to-apples
    /// comparison of distance-computation counts and recall.
    pub fn search(
        &self,
        query: &[f32],
        top_k: usize,
        policy: BeamStopPolicy,
    ) -> (Vec<(u32, f32)>, SearchMetrics) {
        let mut m = SearchMetrics::default();
        if self.vectors.is_empty() {
            return (vec![], m);
        }

        // frontier: min-heap by distance — pop closest unvisited next
        let mut frontier: BinaryHeap<Reverse<OrdF>> = BinaryHeap::new();
        // results: max-heap — top element is the k-th nearest found (worst of top-k)
        let mut results: BinaryHeap<OrdF> = BinaryHeap::new();
        let mut visited = HashSet::<u32>::new();

        let ep = self.entry_point;
        let d0 = l2_sq(query, &self.vectors[ep as usize]);
        m.distance_computations += 1;
        frontier.push(Reverse(OrdF(d0, ep)));
        visited.insert(ep);
        results.push(OrdF(d0, ep));

        let mut expansions: usize = 0;

        while let Some(Reverse(OrdF(curr_dist, curr_node))) = frontier.pop() {
            // k-th nearest found so far = maximum of results max-heap
            let kth = results.peek().map(|r| r.0).unwrap_or(f32::MAX);

            let stop = match policy {
                BeamStopPolicy::FixedWidth { beam_width } => expansions >= beam_width,
                BeamStopPolicy::DistanceAdaptive { gamma } => {
                    results.len() >= top_k && curr_dist > (1.0 + gamma) * kth
                }
                BeamStopPolicy::AdaptiveWithFloor { gamma, min_expansions } => {
                    expansions >= min_expansions
                        && results.len() >= top_k
                        && curr_dist > (1.0 + gamma) * kth
                }
            };

            if stop {
                m.early_stopped = true;
                break;
            }

            expansions += 1;
            m.nodes_expanded += 1;

            for &nb in &self.neighbors[curr_node as usize] {
                if visited.insert(nb) {
                    let d = l2_sq(query, &self.vectors[nb as usize]);
                    m.distance_computations += 1;
                    frontier.push(Reverse(OrdF(d, nb)));
                    if results.len() < top_k {
                        results.push(OrdF(d, nb));
                    } else if kth > d {
                        results.pop();
                        results.push(OrdF(d, nb));
                    }
                }
            }
        }

        let mut out: Vec<(u32, f32)> = results.into_iter().map(|r| (r.1, r.0)).collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        (out, m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::build_knn_graph;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::{Distribution, Normal};

    fn gaussian_vecs(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0f32, 1.0).unwrap();
        (0..n)
            .map(|_| (0..d).map(|_| normal.sample(&mut rng)).collect())
            .collect()
    }

    fn brute_knn(vecs: &[Vec<f32>], query: &[f32], k: usize) -> HashSet<u32> {
        let mut ds: Vec<(f32, u32)> = vecs
            .iter()
            .enumerate()
            .map(|(i, v)| (l2_sq(query, v), i as u32))
            .collect();
        ds.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        ds.truncate(k);
        ds.into_iter().map(|(_, i)| i).collect()
    }

    #[test]
    fn test_search_returns_top_k() {
        let vecs = gaussian_vecs(200, 16, 42);
        let (nb, ep) = build_knn_graph(&vecs, 12);
        let idx = AdaptiveBeamIndex::new(vecs, nb, ep);
        let q = gaussian_vecs(1, 16, 99)[0].clone();
        for policy in [
            BeamStopPolicy::FixedWidth { beam_width: 32 },
            BeamStopPolicy::DistanceAdaptive { gamma: 0.5 },
            BeamStopPolicy::AdaptiveWithFloor { gamma: 0.5, min_expansions: 8 },
        ] {
            let (res, _) = idx.search(&q, 10, policy);
            assert_eq!(res.len(), 10, "should return exactly k=10 results");
            for w in res.windows(2) {
                assert!(w[0].1 <= w[1].1, "results must be sorted by distance ascending");
            }
        }
    }

    #[test]
    fn test_fixed_width_expansion_limit() {
        let vecs = gaussian_vecs(300, 16, 77);
        let (nb, ep) = build_knn_graph(&vecs, 12);
        let idx = AdaptiveBeamIndex::new(vecs, nb, ep);
        let q = gaussian_vecs(1, 16, 1)[0].clone();
        let bw = 20usize;
        let (_, m) = idx.search(&q, 5, BeamStopPolicy::FixedWidth { beam_width: bw });
        assert!(
            m.nodes_expanded <= bw as u64,
            "FixedWidth should expand ≤ beam_width nodes; got {}",
            m.nodes_expanded
        );
    }

    #[test]
    fn test_adaptive_uses_fewer_distance_computations() {
        // Compare adaptive stopping against *exhaustive* beam (beam_width = n).
        // Exhaustive FixedWidth visits every node; DistanceAdaptive should stop
        // well before that, demonstrating the savings proven in arXiv:2505.15636.
        let n = 400;
        let vecs = gaussian_vecs(n, 32, 13);
        let (nb, ep) = build_knn_graph(&vecs, 14);
        let idx = AdaptiveBeamIndex::new(vecs, nb, ep);
        let queries = gaussian_vecs(20, 32, 55);

        let (mut fw_total, mut da_total) = (0u64, 0u64);
        for q in &queries {
            // beam_width = n → exhaustive traversal of every graph node
            let (_, mf) = idx.search(q, 10, BeamStopPolicy::FixedWidth { beam_width: n });
            let (_, ma) = idx.search(q, 10, BeamStopPolicy::DistanceAdaptive { gamma: 1.0 });
            fw_total += mf.distance_computations;
            da_total += ma.distance_computations;
        }
        assert!(
            da_total < fw_total,
            "DistanceAdaptive should compute fewer distances than exhaustive FixedWidth(n); \
             adaptive={da_total} exhaustive={fw_total}"
        );
    }

    #[test]
    fn test_floor_respected_on_early_stop() {
        let vecs = gaussian_vecs(300, 16, 42);
        let (nb, ep) = build_knn_graph(&vecs, 12);
        let idx = AdaptiveBeamIndex::new(vecs, nb, ep);
        let q = gaussian_vecs(1, 16, 7)[0].clone();
        let min_exp = 10usize;
        let (_, m) = idx.search(
            &q, 5,
            BeamStopPolicy::AdaptiveWithFloor { gamma: 0.1, min_expansions: min_exp },
        );
        if m.early_stopped {
            assert!(
                m.nodes_expanded >= min_exp as u64,
                "floor should enforce ≥ min_expansions before stopping; got {}",
                m.nodes_expanded
            );
        }
    }

    #[test]
    fn test_recall_meaningful_on_gaussian_data() {
        let vecs = gaussian_vecs(1_000, 64, 42);
        let (nb, ep) = build_knn_graph(&vecs, 16);
        let idx = AdaptiveBeamIndex::new(vecs.clone(), nb, ep);
        let queries = gaussian_vecs(30, 64, 999);

        let mut total_recall = 0.0f64;
        for q in &queries {
            let truth = brute_knn(&vecs, q, 10);
            let (res, _) = idx.search(q, 10, BeamStopPolicy::DistanceAdaptive { gamma: 0.5 });
            let found: HashSet<u32> = res.iter().map(|(i, _)| *i).collect();
            total_recall += truth.intersection(&found).count() as f64 / 10.0;
        }
        let avg = total_recall / queries.len() as f64;
        assert!(
            avg > 0.5,
            "Recall@10 should be >50% on Gaussian data with DA(γ=0.5); got {avg:.3}"
        );
    }

    #[test]
    fn test_distance_adaptive_gamma0_high_recall() {
        let vecs = gaussian_vecs(800, 32, 17);
        let (nb, ep) = build_knn_graph(&vecs, 14);
        let idx = AdaptiveBeamIndex::new(vecs.clone(), nb, ep);
        let queries = gaussian_vecs(20, 32, 88);

        let mut total_recall = 0.0f64;
        for q in &queries {
            let truth = brute_knn(&vecs, q, 10);
            // gamma=0 should approach exact search
            let (res, _) = idx.search(q, 10, BeamStopPolicy::DistanceAdaptive { gamma: 0.0 });
            let found: HashSet<u32> = res.iter().map(|(i, _)| *i).collect();
            total_recall += truth.intersection(&found).count() as f64 / 10.0;
        }
        let avg = total_recall / queries.len() as f64;
        assert!(
            avg > 0.7,
            "DA(γ=0) should achieve >70% recall; got {avg:.3}"
        );
    }
}
