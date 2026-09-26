//! # ruvector-anytime-ann
//!
//! Anytime ANN search with budget-aware early termination.
//!
//! Standard HNSW beam search uses a single stopping rule: expand candidates
//! until all remaining are farther than the current kth result. This is
//! optimal for recall but ignores latency budgets and compute limits.
//!
//! This crate implements three search variants on a flat navigable small-world
//! graph (HNSW layer-0 equivalent):
//!
//! | Variant | Stopping criterion | Best for |
//! |---------|-------------------|---------|
//! | [`FixedEfSearch`] | All candidates exhausted | Maximum recall |
//! | [`BudgetedEvalsSearch`] | Distance evals ≥ budget | Edge / WASM deployments |
//! | [`EarlyConvergenceSearch`] | kth result stalled for P steps | Anytime quality |
//!
//! ## The Anytime Property
//!
//! An anytime algorithm produces a valid answer at any interrupt point.
//! All three variants maintain a result heap throughout the graph walk;
//! stopping early returns the best k candidates seen so far.
//!
//! ## Edge and WASM relevance
//!
//! On Cognitum Seed or WASM targets, compute budgets are hard constraints.
//! `BudgetedEvalsSearch` lets the caller say "use at most 1000 distance
//! evaluations" rather than guessing an ef that happens to fit the budget.
//!
//! ## ruFlo relevance
//!
//! A ruFlo workflow can observe `compute_per_query` and tune `max_evals` or
//! `patience` to hit a target recall–latency operating point over time.

pub mod dataset;
pub mod graph;
pub mod search;

pub use graph::{FlatGraph, GraphConfig};
pub use search::{
    BudgetedEvalsSearch, EarlyConvergenceSearch, FixedEfSearch, SearchResult, Searcher,
};

/// Compute recall@k for a searcher (uses FixedEf with given ef as ground truth).
///
/// Ground truth is computed by brute-force L2 scan.
pub fn recall_at_k(
    graph: &FlatGraph,
    queries: &[Vec<f32>],
    k: usize,
    ef: usize,
    searcher: &dyn Searcher,
    entry: usize,
) -> f32 {
    let gt = brute_force_knn(graph, queries, k);
    let mut total = 0.0f32;
    for (i, query) in queries.iter().enumerate() {
        let result = searcher.search(graph, query, k, ef, entry);
        let gt_set: std::collections::HashSet<u32> = gt[i].iter().cloned().collect();
        let hits = result.neighbors.iter().filter(|(id, _)| gt_set.contains(id)).count();
        total += hits as f32 / k as f32;
    }
    total / queries.len() as f32
}

/// Brute-force kNN for ground truth.
pub fn brute_force_knn(graph: &FlatGraph, queries: &[Vec<f32>], k: usize) -> Vec<Vec<u32>> {
    let dims = graph.config.dims;
    queries
        .iter()
        .map(|q| {
            let mut dists: Vec<(f32, u32)> = (0..graph.n)
                .map(|i| {
                    let v = &graph.vectors[i * dims..(i + 1) * dims];
                    (graph::l2_sq(q, v), i as u32)
                })
                .collect();
            dists.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
            dists.truncate(k);
            dists.into_iter().map(|(_, id)| id).collect()
        })
        .collect()
}

/// Estimate memory usage for the graph in bytes.
pub fn memory_estimate_bytes(graph: &FlatGraph) -> usize {
    let vector_bytes = graph.vectors.len() * core::mem::size_of::<f32>();
    let neighbor_bytes: usize = graph.neighbors.iter().map(|nl| nl.len() * 4 + 24).sum();
    vector_bytes + neighbor_bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{clustered_queries, clustered_vectors};

    const SMALL_N_CLUSTERS: usize = 8;
    const SMALL_N_PER: usize = 50;
    const SMALL_DIMS: usize = 32;
    const SMALL_STD: f32 = 0.2;
    const SMALL_DATA_SEED: u64 = 42;
    const SMALL_QUERY_SEED: u64 = 99;

    fn small_graph() -> (FlatGraph, Vec<Vec<f32>>) {
        let (data, _) =
            clustered_vectors(SMALL_N_CLUSTERS, SMALL_N_PER, SMALL_DIMS, SMALL_STD, SMALL_DATA_SEED);
        let config = GraphConfig { m: 12, m_longjump: 4, dims: SMALL_DIMS };
        let graph = FlatGraph::build(data, config);
        let (queries, _) =
            clustered_queries(20, SMALL_DIMS, SMALL_STD, SMALL_DATA_SEED, SMALL_QUERY_SEED);
        (graph, queries)
    }

    #[test]
    fn fixed_ef_recall_above_threshold() {
        let (graph, queries) = small_graph();
        let searcher = FixedEfSearch;
        let r = recall_at_k(&graph, &queries, 5, 40, &searcher, 0);
        assert!(r >= 0.70, "FixedEf recall@5 = {r:.3} < 0.70");
    }

    #[test]
    fn budgeted_evals_returns_at_most_k_results() {
        let (graph, _) = small_graph();
        let query = vec![0.1f32; 32];
        let searcher = BudgetedEvalsSearch { max_evals: 300 };
        let result = searcher.search(&graph, &query, 10, 40, 0);
        assert!(result.neighbors.len() <= 10, "Returned > k results");
    }

    #[test]
    fn budgeted_evals_respects_budget() {
        let (graph, _) = small_graph();
        let query = vec![0.0f32; 32];
        let budget = 200usize;
        let searcher = BudgetedEvalsSearch { max_evals: budget };
        let result = searcher.search(&graph, &query, 5, 40, 0);
        // Allow slight overshoot: the inner loop checks the counter at the
        // start of each expansion, so it may consume up to m_neighbors more
        // evaluations after the budget is first exceeded.
        assert!(
            result.evaluations <= budget + 30,
            "evaluations {} exceeded budget {} + 30",
            result.evaluations,
            budget
        );
    }

    #[test]
    fn early_convergence_uses_fewer_or_equal_evals_than_fixed() {
        let (data, _) = clustered_vectors(SMALL_N_CLUSTERS, 80, SMALL_DIMS, 0.15, SMALL_DATA_SEED);
        let config = GraphConfig { m: 12, m_longjump: 4, dims: 32 };
        let graph = FlatGraph::build(data, config);
        let query = vec![0.05f32; 32];

        let r_fixed = FixedEfSearch.search(&graph, &query, 10, 50, 0);
        let r_early = EarlyConvergenceSearch { patience: 4, min_improvement: 1e-4 }
            .search(&graph, &query, 10, 50, 0);

        let ratio = r_early.evaluations as f32 / (r_fixed.evaluations as f32 + 1.0);
        assert!(
            ratio <= 1.05,
            "EarlyConverge used {:.0}% of FixedEf evals ({}% > 105%)",
            ratio * 100.0,
            (ratio * 100.0) as u32
        );
    }

    #[test]
    fn results_sorted_nearest_first() {
        let (data, _) = clustered_vectors(4, 30, 16, 0.3, 7);
        let config = GraphConfig { m: 8, m_longjump: 2, dims: 16 };
        let graph = FlatGraph::build(data, config);
        let query = vec![0.0f32; 16];
        let result = FixedEfSearch.search(&graph, &query, 5, 20, 0);
        let dists: Vec<f32> = result.neighbors.iter().map(|(_, d)| *d).collect();
        for w in dists.windows(2) {
            assert!(w[0] <= w[1], "Not sorted: {} > {}", w[0], w[1]);
        }
    }
}
