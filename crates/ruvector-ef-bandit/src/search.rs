//! Three adaptive ef-search variants wrapping the NSW graph.
//!
//! All variants share the same underlying graph (passed as a reference)
//! so comparisons are apples-to-apples: the only variable is the ef
//! selection policy.

use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::time::Instant;

use crate::bandit::{EpsilonGreedyBandit, Ucb1Bandit};
use crate::graph::NswGraph;
use crate::metrics::recall_at_k;
use crate::{AdaptiveSearch, QueryResult, SearchConfig};

// ── Baseline ──────────────────────────────────────────────────────────────────

/// Baseline: always uses the median ef value from the candidate list.
///
/// Represents current practice in most deployed vector databases:
/// a fixed, hand-tuned beam width chosen at deploy time and never changed.
pub struct BaselineSearch<'g> {
    graph: &'g NswGraph,
    fixed_ef: usize,
    k: usize,
    query_count: usize,
}

impl<'g> BaselineSearch<'g> {
    pub fn new(graph: &'g NswGraph, config: &SearchConfig) -> Self {
        // Pick the middle ef value as the "reasonable default".
        let fixed_ef = config.ef_candidates[config.ef_candidates.len() / 2];
        Self {
            graph,
            fixed_ef,
            k: config.k,
            query_count: 0,
        }
    }
}

impl<'g> AdaptiveSearch for BaselineSearch<'g> {
    fn name(&self) -> &str {
        "Baseline (fixed-ef)"
    }

    fn query(&mut self, q: &[f32], _ground_truth: &[usize]) -> QueryResult {
        let t0 = Instant::now();
        let raw = self.graph.search(q, self.k, self.fixed_ef);
        let latency_ns = t0.elapsed().as_nanos() as u64;
        self.query_count += 1;
        QueryResult {
            indices: raw.into_iter().map(|(i, _)| i).collect(),
            ef_used: self.fixed_ef,
            latency_ns,
        }
    }

    fn current_best_ef(&self) -> usize {
        self.fixed_ef
    }

    fn query_count(&self) -> usize {
        self.query_count
    }

    fn bandit_memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

// ── UCB1 ──────────────────────────────────────────────────────────────────────

/// UCB1 bandit adaptive ef selection.
///
/// After each query the arm that was used receives a reward equal to
/// recall@k measured against the provided ground truth. Over time UCB1
/// converges to the minimum-ef arm that still achieves acceptable recall
/// — giving the best latency/recall operating point the workload allows.
pub struct Ucb1Search<'g> {
    graph: &'g NswGraph,
    bandit: Ucb1Bandit,
    k: usize,
    query_count: usize,
}

impl<'g> Ucb1Search<'g> {
    pub fn new(graph: &'g NswGraph, config: &SearchConfig) -> Self {
        Self {
            graph,
            bandit: Ucb1Bandit::new(&config.ef_candidates, config.exploration_c),
            k: config.k,
            query_count: 0,
        }
    }

    pub fn arm_stats(&self) -> Vec<(usize, u64, f64)> {
        self.bandit
            .arms()
            .iter()
            .map(|a| (a.ef, a.n_pulls, a.mean_reward()))
            .collect()
    }
}

impl<'g> AdaptiveSearch for Ucb1Search<'g> {
    fn name(&self) -> &str {
        "UCB1 Bandit"
    }

    fn query(&mut self, q: &[f32], ground_truth: &[usize]) -> QueryResult {
        let (arm_idx, ef) = self.bandit.select();

        let t0 = Instant::now();
        let raw = self.graph.search(q, self.k, ef);
        let latency_ns = t0.elapsed().as_nanos() as u64;

        let indices: Vec<usize> = raw.into_iter().map(|(i, _)| i).collect();
        let reward = recall_at_k(&indices, ground_truth, self.k);
        self.bandit.update(arm_idx, reward);
        self.query_count += 1;

        QueryResult {
            indices,
            ef_used: ef,
            latency_ns,
        }
    }

    fn current_best_ef(&self) -> usize {
        self.bandit.best_arm().1
    }

    fn query_count(&self) -> usize {
        self.query_count
    }

    fn bandit_memory_bytes(&self) -> usize {
        self.bandit.memory_bytes()
    }
}

// ── ε-Greedy ─────────────────────────────────────────────────────────────────

/// ε-Greedy adaptive ef selection with exponential epsilon decay.
///
/// Starts with high exploration (ε = 0.30), gradually shifting toward
/// pure exploitation as confidence in the best arm grows.
pub struct EpsilonGreedySearch<'g> {
    graph: &'g NswGraph,
    bandit: EpsilonGreedyBandit,
    k: usize,
    query_count: usize,
    rng: SmallRng,
}

impl<'g> EpsilonGreedySearch<'g> {
    pub fn new(graph: &'g NswGraph, config: &SearchConfig, seed: u64) -> Self {
        Self {
            graph,
            bandit: EpsilonGreedyBandit::new(
                &config.ef_candidates,
                config.epsilon_init,
                config.epsilon_decay,
            ),
            k: config.k,
            query_count: 0,
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    pub fn arm_stats(&self) -> Vec<(usize, u64, f64)> {
        self.bandit
            .arms()
            .iter()
            .map(|a| (a.ef, a.n_pulls, a.mean_reward()))
            .collect()
    }

    pub fn current_epsilon(&self) -> f64 {
        self.bandit.current_epsilon()
    }
}

impl<'g> AdaptiveSearch for EpsilonGreedySearch<'g> {
    fn name(&self) -> &str {
        "ε-Greedy Decay"
    }

    fn query(&mut self, q: &[f32], ground_truth: &[usize]) -> QueryResult {
        let (arm_idx, ef) = self.bandit.select(&mut self.rng);

        let t0 = Instant::now();
        let raw = self.graph.search(q, self.k, ef);
        let latency_ns = t0.elapsed().as_nanos() as u64;

        let indices: Vec<usize> = raw.into_iter().map(|(i, _)| i).collect();
        let reward = recall_at_k(&indices, ground_truth, self.k);
        self.bandit.update(arm_idx, reward);
        self.query_count += 1;

        QueryResult {
            indices,
            ef_used: ef,
            latency_ns,
        }
    }

    fn current_best_ef(&self) -> usize {
        self.bandit.best_arm().1
    }

    fn query_count(&self) -> usize {
        self.query_count
    }

    fn bandit_memory_bytes(&self) -> usize {
        self.bandit.memory_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{brute_force_knn, generate_clustered, generate_queries};

    fn build_test_graph() -> NswGraph {
        let data = generate_clustered(500, 32, 10, 1);
        let mut g = NswGraph::new(8, 40);
        for v in data {
            g.insert(v);
        }
        g
    }

    #[test]
    fn baseline_returns_k_results() {
        let g = build_test_graph();
        let cfg = SearchConfig::default();
        let mut s = BaselineSearch::new(&g, &cfg);
        let queries = generate_queries(10, 32, 2);
        let gt = brute_force_knn(&generate_clustered(500, 32, 10, 1), &queries, cfg.k);
        for (q, gt_row) in queries.iter().zip(gt.iter()) {
            let res = s.query(q, gt_row);
            assert!(res.indices.len() <= cfg.k);
            assert_eq!(res.ef_used, cfg.ef_candidates[cfg.ef_candidates.len() / 2]);
        }
    }

    #[test]
    fn ucb1_achieves_positive_recall_after_warmup() {
        let data = generate_clustered(500, 32, 10, 1);
        let mut g = NswGraph::new(8, 40);
        for v in data.iter().cloned() {
            g.insert(v);
        }
        let cfg = SearchConfig::default();
        let mut s = Ucb1Search::new(&g, &cfg);
        let queries = generate_queries(200, 32, 3);
        let gt = brute_force_knn(&data, &queries, cfg.k);

        let mut total_recall = 0.0f64;
        for (q, gt_row) in queries[..100].iter().zip(gt.iter()) {
            let res = s.query(q, gt_row);
            total_recall += crate::metrics::recall_at_k(&res.indices, gt_row, cfg.k);
        }
        let avg_recall = total_recall / 100.0;
        assert!(
            avg_recall > 0.5,
            "UCB1 recall after warm-up should exceed 0.5, got {avg_recall:.3}"
        );
    }

    #[test]
    fn epsilon_greedy_epsilon_decreases() {
        let g = build_test_graph();
        let cfg = SearchConfig::default();
        let mut s = EpsilonGreedySearch::new(&g, &cfg, 42);
        let eps_before = s.current_epsilon();
        let q = generate_queries(1, 32, 5);
        let gt = brute_force_knn(&generate_clustered(500, 32, 10, 1), &q, cfg.k);
        s.query(&q[0], &gt[0]);
        let eps_after = s.current_epsilon();
        assert!(eps_after < eps_before, "epsilon must decay");
    }
}
