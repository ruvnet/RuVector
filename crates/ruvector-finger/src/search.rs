use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::HashSet;

use crate::basis::NodeBasis;
use crate::dist::{dot, l2_sq, sub_into};
use crate::error::FingerError;
use crate::graph::GraphWalk;

/// Statistics collected during a single search call.
#[derive(Debug, Default, Clone)]
pub struct SearchStats {
    /// Exact L2 distance computations performed.
    pub exact_dists: usize,
    /// Neighbors skipped by the FINGER approximation.
    pub finger_pruned: usize,
    /// Total neighbor edges evaluated.
    pub edges_visited: usize,
}

impl SearchStats {
    pub fn prune_rate(&self) -> f64 {
        if self.edges_visited == 0 {
            0.0
        } else {
            self.finger_pruned as f64 / self.edges_visited as f64
        }
    }
}

/// Ordered f32 wrapper with total ordering via `total_cmp`.
#[derive(Clone, Copy, PartialEq)]
struct OrdF32(f32);
impl Eq for OrdF32 {}
impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for OrdF32 {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&o.0)
    }
}

/// Exact greedy beam search (no FINGER approximation).
///
/// Standard algorithm: maintain a min-heap of candidates (by distance) and
/// a max-heap of top-k results. Process candidates in distance order,
/// expanding their neighbors until the frontier is exhausted.
pub fn exact_beam_search<G: GraphWalk>(
    graph: &G,
    query: &[f32],
    k: usize,
    ef: usize,
) -> Result<(Vec<(u32, f32)>, SearchStats), FingerError> {
    let n = graph.n_nodes();
    if n == 0 {
        return Err(FingerError::EmptyDataset);
    }
    if k > n {
        return Err(FingerError::KTooLarge { k, n });
    }
    if ef < k {
        return Err(FingerError::EfTooSmall { ef, k });
    }

    let entry = graph.entry_point() as usize;
    let d_entry = l2_sq(query, graph.vector(entry)).sqrt();

    let mut visited: HashSet<u32> = HashSet::new();
    // candidates: min-heap (smallest distance = most promising)
    let mut candidates: BinaryHeap<Reverse<(OrdF32, u32)>> = BinaryHeap::new();
    // results: max-heap (largest distance = worst current result, easy to discard)
    let mut results: BinaryHeap<(OrdF32, u32)> = BinaryHeap::new();

    visited.insert(entry as u32);
    candidates.push(Reverse((OrdF32(d_entry), entry as u32)));
    results.push((OrdF32(d_entry), entry as u32));

    let mut stats = SearchStats::default();

    while let Some(Reverse((OrdF32(d_curr), curr_id))) = candidates.pop() {
        // Early termination: the closest unvisited candidate is farther than our
        // current worst result and we already have ef results.
        if results.len() >= ef {
            if let Some(&(OrdF32(worst), _)) = results.peek() {
                if d_curr > worst {
                    break;
                }
            }
        }

        for &nb_id in graph.neighbors(curr_id as usize) {
            if visited.contains(&nb_id) {
                continue;
            }
            visited.insert(nb_id);
            stats.edges_visited += 1;

            let d_nb = l2_sq(query, graph.vector(nb_id as usize)).sqrt();
            stats.exact_dists += 1;

            let worst = results.peek().map(|&(OrdF32(w), _)| w).unwrap_or(f32::MAX);
            if d_nb < worst || results.len() < ef {
                candidates.push(Reverse((OrdF32(d_nb), nb_id)));
                results.push((OrdF32(d_nb), nb_id));
                if results.len() > ef {
                    results.pop();
                }
            }
        }
    }

    let mut out: Vec<(u32, f32)> = results.iter().map(|&(OrdF32(d), id)| (id, d)).collect();
    out.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    out.truncate(k);
    Ok((out, stats))
}

/// FINGER beam search: approximate distance skipping using precomputed residual bases.
///
/// For each node `u` popped from the candidate heap, we:
///   1. Compute the query residual `w = query - u` once (O(D)).
///   2. Project `w` onto the K-dimensional basis stored at `u` (O(K × D)).
///   3. For each neighbor `v` of `u`: compute an approximate distance in O(K)
///      using the precomputed edge projections.
///   4. Skip neighbors whose approximate distance exceeds `slack × worst_result`.
///   5. Only for non-skipped neighbors: compute the exact distance (O(D)).
///
/// Total per-node cost: O(D + K×D + M×K + P×D) where P is non-pruned fraction.
/// vs exact: O(M×D). Speedup when K << D and P < 1 − K/M.
pub fn finger_beam_search<G: GraphWalk>(
    graph: &G,
    bases: &[NodeBasis],
    query: &[f32],
    k: usize,
    ef: usize,
    slack: f32,
) -> Result<(Vec<(u32, f32)>, SearchStats), FingerError> {
    let n = graph.n_nodes();
    if n == 0 {
        return Err(FingerError::EmptyDataset);
    }
    if k > n {
        return Err(FingerError::KTooLarge { k, n });
    }
    if ef < k {
        return Err(FingerError::EfTooSmall { ef, k });
    }

    let dim = graph.dim();
    let entry = graph.entry_point() as usize;
    let d_entry = l2_sq(query, graph.vector(entry)).sqrt();

    let mut visited: HashSet<u32> = HashSet::new();
    let mut candidates: BinaryHeap<Reverse<(OrdF32, u32)>> = BinaryHeap::new();
    let mut results: BinaryHeap<(OrdF32, u32)> = BinaryHeap::new();

    visited.insert(entry as u32);
    candidates.push(Reverse((OrdF32(d_entry), entry as u32)));
    results.push((OrdF32(d_entry), entry as u32));

    let mut stats = SearchStats::default();
    // Scratch buffer for query residual — allocated once, reused per node.
    let mut residual = vec![0.0f32; dim];

    while let Some(Reverse((OrdF32(d_curr), curr_id))) = candidates.pop() {
        if results.len() >= ef {
            if let Some(&(OrdF32(worst), _)) = results.peek() {
                if d_curr > worst {
                    break;
                }
            }
        }

        let curr_vec = graph.vector(curr_id as usize);
        let d_curr_sq = d_curr * d_curr;

        // FINGER step 1: compute query residual w = query - curr_vec (O(D)).
        sub_into(query, curr_vec, &mut residual);

        // FINGER step 2: project w onto the node's basis (O(K×D)).
        let basis = &bases[curr_id as usize];
        let query_proj: Vec<f32> = if basis.k > 0 {
            basis.project(&residual)
        } else {
            Vec::new()
        };

        let worst = results.peek().map(|&(OrdF32(w), _)| w).unwrap_or(f32::MAX);

        for (mi, &nb_id) in graph.neighbors(curr_id as usize).iter().enumerate() {
            if visited.contains(&nb_id) {
                continue;
            }
            stats.edges_visited += 1;

            // FINGER step 3: approximate distance (O(K)).
            // Per the original paper (Algorithm 1): skip the exact computation but
            // do NOT mark the node as visited — it can still be reached via a
            // different parent where the approximation error is smaller.
            // Marking pruned nodes visited (common mistake) causes hard recall loss
            // on unstructured data where edge directions don't align with queries.
            if basis.k > 0 && results.len() >= k {
                let approx = basis.approx_dist(&query_proj, d_curr_sq, mi);
                if approx > worst * slack {
                    stats.finger_pruned += 1;
                    continue;  // skip exact dist, but leave node unvisited
                }
            }

            visited.insert(nb_id);

            // FINGER step 4: exact distance only for non-pruned neighbors (O(D)).
            let d_nb = l2_sq(query, graph.vector(nb_id as usize)).sqrt();
            stats.exact_dists += 1;

            let worst_now = results.peek().map(|&(OrdF32(w), _)| w).unwrap_or(f32::MAX);
            if d_nb < worst_now || results.len() < ef {
                candidates.push(Reverse((OrdF32(d_nb), nb_id)));
                results.push((OrdF32(d_nb), nb_id));
                if results.len() > ef {
                    results.pop();
                }
            }
        }
    }

    let mut out: Vec<(u32, f32)> = results.iter().map(|&(OrdF32(d), id)| (id, d)).collect();
    out.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    out.truncate(k);
    Ok((out, stats))
}

/// Dot product used only in the projection fallback path.
#[allow(dead_code)]
fn _dot_check(a: &[f32], b: &[f32]) -> f32 {
    dot(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::FlatGraph;

    fn gaussian_data(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut state = seed | 1;
        let xorshift = |s: &mut u64| -> f32 {
            let mut x = *s;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *s = x;
            // Box-Muller transform for roughly Gaussian samples
            let u = (x as f32) / (u64::MAX as f32);
            (2.0 * u - 1.0) * 2.0
        };
        (0..n)
            .map(|_| (0..dim).map(|_| xorshift(&mut state)).collect())
            .collect()
    }

    #[test]
    fn exact_search_finds_true_neighbor() {
        let data = gaussian_data(200, 16, 42);
        let g = FlatGraph::build(&data, 12).unwrap();
        let query = &data[0];
        let (results, _) = exact_beam_search(&g, query, 5, 50).unwrap();
        // The query is node 0 itself — should be first result
        assert_eq!(results[0].0, 0);
        assert!(results[0].1 < 1e-5);
    }

    #[test]
    fn finger_search_high_recall() {
        let data = gaussian_data(500, 16, 7);
        let g = FlatGraph::build(&data, 16).unwrap();

        // Build bases for all nodes (k_basis=4)
        let k_basis = 4;
        let bases: Vec<NodeBasis> = (0..g.n_nodes())
            .map(|i| {
                let node_vec = g.vector(i);
                let nb_vecs: Vec<&[f32]> =
                    g.neighbors(i).iter().map(|&j| g.vector(j as usize)).collect();
                NodeBasis::build(node_vec, &nb_vecs, k_basis)
            })
            .collect();

        let query = &data[7];
        let (exact_res, _) = exact_beam_search(&g, query, 10, 50).unwrap();
        let (finger_res, stats) = finger_beam_search(&g, &bases, query, 10, 50, 1.0).unwrap();

        let exact_ids: Vec<u32> = exact_res.iter().map(|(id, _)| *id).collect();
        let finger_ids: Vec<u32> = finger_res.iter().map(|(id, _)| *id).collect();
        let recall = crate::graph::recall_at_k(&finger_ids, &exact_ids, 10);
        // FINGER should achieve at least 70% recall vs exact beam search at this scale
        assert!(
            recall >= 0.70,
            "recall vs exact beam search too low: {recall:.2} (pruned {}%)",
            stats.prune_rate() * 100.0
        );
    }
}
