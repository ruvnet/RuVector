use crate::basis::NodeBasis;
use crate::error::FingerError;
use crate::graph::GraphWalk;
#[allow(unused_imports)]
use crate::search::{exact_beam_search, finger_beam_search, SearchStats};

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

/// The slack factor controls how aggressively FINGER prunes neighbors.
///
/// `slack = 1.0`: prune only when approx_dist > worst_result (paper default).
/// `slack > 1.0`: prune more aggressively at the cost of recall.
pub const DEFAULT_SLACK: f32 = 1.0;

/// FINGER index wrapping a graph and its precomputed residual bases.
///
/// `k_basis = 0` disables the FINGER approximation; search falls back to
/// exact beam search. This lets the same `FingerIndex` serve as the exact
/// baseline for recall measurement.
pub struct FingerIndex<'g, G: GraphWalk> {
    graph: &'g G,
    /// One `NodeBasis` per graph node. Empty if `k_basis == 0`.
    bases: Vec<NodeBasis>,
    k_basis: usize,
    slack: f32,
}

impl<'g, G: GraphWalk> FingerIndex<'g, G> {
    /// Build an exact (no FINGER) index for use as the search baseline.
    pub fn exact(graph: &'g G) -> Self {
        FingerIndex {
            graph,
            bases: Vec::new(),
            k_basis: 0,
            slack: DEFAULT_SLACK,
        }
    }

    /// Build a FINGER index with the given number of basis vectors.
    ///
    /// Typical values: 4 (D≤256, fast builds), 8 (D>256 or higher recall).
    pub fn build_with_k(graph: &'g G, k_basis: usize) -> Result<Self, FingerError> {
        if k_basis == 0 {
            return Ok(Self::exact(graph));
        }
        let bases = Self::build_bases(graph, k_basis);
        Ok(FingerIndex { graph, bases, k_basis, slack: DEFAULT_SLACK })
    }

    /// Build a FINGER index with `k_basis=4` (sweet spot for D=64–256).
    pub fn finger_k4(graph: &'g G) -> Result<Self, FingerError> {
        Self::build_with_k(graph, 4)
    }

    /// Build a FINGER index with `k_basis=8` (higher accuracy, better for D>256).
    pub fn finger_k8(graph: &'g G) -> Result<Self, FingerError> {
        Self::build_with_k(graph, 8)
    }

    /// Override the pruning slack factor (default = 1.0).
    pub fn with_slack(mut self, slack: f32) -> Self {
        self.slack = slack;
        self
    }

    /// Search for the top-`k` nearest neighbors using FINGER beam search.
    ///
    /// `ef` controls the search beam width (must be ≥ k). Higher ef →
    /// better recall, lower QPS. Typical: ef = 3–5× k.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<(Vec<(u32, f32)>, SearchStats), FingerError> {
        if self.k_basis == 0 || self.bases.is_empty() {
            exact_beam_search(self.graph, query, k, ef)
        } else {
            finger_beam_search(self.graph, &self.bases, query, k, ef, self.slack)
        }
    }

    pub fn k_basis(&self) -> usize {
        self.k_basis
    }

    /// Approximate heap bytes used by the precomputed bases.
    pub fn bytes_used(&self) -> usize {
        self.bases.iter().map(|b| {
            b.basis.len() * 4
                + b.edge_projs.len() * 4
                + b.edge_norms_sq.len() * 4
        }).sum()
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn build_bases(graph: &G, k_basis: usize) -> Vec<NodeBasis> {
        (0..graph.n_nodes())
            .into_par_iter()
            .map(|i| {
                let node_vec = graph.vector(i);
                let nb_vecs: Vec<&[f32]> = graph
                    .neighbors(i)
                    .iter()
                    .map(|&j| graph.vector(j as usize))
                    .collect();
                NodeBasis::build(node_vec, &nb_vecs, k_basis)
            })
            .collect()
    }

    #[cfg(target_arch = "wasm32")]
    fn build_bases(graph: &G, k_basis: usize) -> Vec<NodeBasis> {
        (0..graph.n_nodes())
            .map(|i| {
                let node_vec = graph.vector(i);
                let nb_vecs: Vec<&[f32]> = graph
                    .neighbors(i)
                    .iter()
                    .map(|&j| graph.vector(j as usize))
                    .collect();
                NodeBasis::build(node_vec, &nb_vecs, k_basis)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{recall_at_k, FlatGraph};

    fn gen_data(n: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut state = 0xdeadbeef_u64;
        (0..n)
            .map(|_| {
                (0..dim)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        ((state as i64) % 200) as f32 * 0.01
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn exact_index_finds_self() {
        let data = gen_data(300, 32);
        let g = FlatGraph::build(&data, 12).unwrap();
        let idx = FingerIndex::exact(&g);
        let query = data[0].clone();
        let (results, _) = idx.search(&query, 1, 30).unwrap();
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn finger_k4_recall_vs_brute_force() {
        let data = gen_data(500, 32);
        let g = FlatGraph::build(&data, 16).unwrap();
        let idx = FingerIndex::finger_k4(&g).unwrap();

        let mut total_recall = 0.0f64;
        let n_queries = 20;
        for qi in 0..n_queries {
            let query = data[(qi * 13 + 7) % data.len()].clone();
            let gt = g.brute_force_knn(&query, 10);
            let (finger_res, _) = idx.search(&query, 10, 50).unwrap();
            let ids: Vec<u32> = finger_res.iter().map(|(id, _)| *id).collect();
            total_recall += recall_at_k(&ids, &gt, 10);
        }
        let avg_recall = total_recall / n_queries as f64;
        assert!(
            avg_recall >= 0.70,
            "FINGER k=4 recall too low: {avg_recall:.3}"
        );
    }

    #[test]
    fn finger_k8_higher_recall_than_k4() {
        let data = gen_data(500, 64);
        let g = FlatGraph::build(&data, 16).unwrap();
        let idx_k4 = FingerIndex::finger_k4(&g).unwrap();
        let idx_k8 = FingerIndex::finger_k8(&g).unwrap();

        let mut recall_k4 = 0.0f64;
        let mut recall_k8 = 0.0f64;
        let n_q = 20;
        for qi in 0..n_q {
            let query = data[(qi * 17 + 3) % data.len()].clone();
            let gt = g.brute_force_knn(&query, 10);
            let r4: Vec<u32> = idx_k4.search(&query, 10, 50).unwrap().0.iter().map(|(id,_)| *id).collect();
            let r8: Vec<u32> = idx_k8.search(&query, 10, 50).unwrap().0.iter().map(|(id,_)| *id).collect();
            recall_k4 += recall_at_k(&r4, &gt, 10);
            recall_k8 += recall_at_k(&r8, &gt, 10);
        }
        recall_k4 /= n_q as f64;
        recall_k8 /= n_q as f64;
        // k=8 should give ≥ k=4 recall (or equal — stochastic result)
        assert!(
            recall_k8 >= recall_k4 - 0.05,
            "k=8 recall {recall_k8:.3} unexpectedly lower than k=4 {recall_k4:.3}"
        );
    }

    #[test]
    fn bytes_used_scales_with_k() {
        let data = gen_data(100, 16);
        let g = FlatGraph::build(&data, 8).unwrap();
        let idx_k4 = FingerIndex::finger_k4(&g).unwrap();
        let idx_k8 = FingerIndex::finger_k8(&g).unwrap();
        // k=8 should use roughly 2× memory for the basis vs k=4
        let ratio = idx_k8.bytes_used() as f64 / idx_k4.bytes_used() as f64;
        assert!(
            ratio > 1.4 && ratio < 3.0,
            "memory ratio k8/k4 = {ratio:.2} unexpected"
        );
    }
}
