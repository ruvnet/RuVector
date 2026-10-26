//! Shared types and distance functions for the Fisher-weighted ANN experiment.
//!
//! `TaskRecord` mirrors the shape ADR-002's `PolicyEntry.fisher_diagonal`
//! anticipates: a stored task carries both its optimal weights and its own
//! Fisher diagonal, matching `ruvector_sona::ewc::TaskFisher`.

/// A single stored "task" snapshot: optimal weights plus the Fisher
/// diagonal and importance that `EwcPlusPlus::regularization_loss` and
/// `apply_anchor` use to decide how strongly to protect it.
#[derive(Clone, Debug)]
pub struct TaskRecord {
    pub id: usize,
    pub weights: Vec<f32>,
    pub fisher: Vec<f32>,
    pub importance: f32,
}

/// Plain squared Euclidean distance between two equal-length slices.
pub fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Fisher-weighted squared distance: `sum_d fisher[d] * (a[d] - b[d])^2`.
///
/// This is the same per-dimension form as the per-task term inside
/// `EwcPlusPlus::regularization_loss` (`fi * diff * diff`), with `fisher`
/// taken from the *candidate* task being scored, not the query — the
/// question being asked is "how much would anchoring to this task cost,
/// in that task's own units of importance".
pub fn fisher_weighted_sq(query: &[f32], task_weights: &[f32], fisher: &[f32]) -> f32 {
    debug_assert_eq!(query.len(), task_weights.len());
    debug_assert_eq!(query.len(), fisher.len());
    query
        .iter()
        .zip(task_weights.iter())
        .zip(fisher.iter())
        .map(|((q, w), f)| f * (q - w) * (q - w))
        .sum()
}

/// A scoring metric used to rank candidate `TaskRecord`s against a query.
/// Lower is "closer". Implementations are not required to be symmetric
/// (the Fisher-weighted metric below is not: it depends on which side's
/// Fisher diagonal is used) — this experiment treats it as a one-sided
/// nearest-neighbor score, same as the production linear scan in
/// `EwcPlusPlus::apply_constraints`/`apply_anchor` already does.
pub trait TaskMetric: Copy + Send + Sync + 'static {
    fn distance(&self, query: &[f32], task: &TaskRecord) -> f32;
    fn name(&self) -> &'static str;
}

/// Candidate metric: Fisher-weighted squared distance, scaled by the
/// task's `importance` (matching `regularization_loss`'s `* task.importance`
/// factor).
#[derive(Copy, Clone)]
pub struct FisherWeighted;

impl TaskMetric for FisherWeighted {
    fn distance(&self, query: &[f32], task: &TaskRecord) -> f32 {
        fisher_weighted_sq(query, &task.weights, &task.fisher) * task.importance
    }
    fn name(&self) -> &'static str {
        "fisher_weighted"
    }
}

/// Baseline/ablation metric: plain squared L2, ignoring Fisher entirely.
#[derive(Copy, Clone)]
pub struct PlainL2;

impl TaskMetric for PlainL2 {
    fn distance(&self, query: &[f32], task: &TaskRecord) -> f32 {
        squared_l2(query, &task.weights)
    }
    fn name(&self) -> &'static str {
        "plain_l2"
    }
}

/// The closed-form EWC regularization loss for anchoring `query` to a
/// *single* task, in the same units as
/// `ruvector_sona::EwcPlusPlus::regularization_loss`:
/// `lambda/2 * sum_d fisher[d] * importance * (query[d] - weights[d])^2`.
///
/// This is intentionally the identical formula to the production function
/// (see `tests::matches_production_ewc_regularization_loss` below, which
/// constructs a real `EwcPlusPlus` via its public API and checks the two
/// agree bit-for-bit within floating point tolerance) rather than a
/// reimplementation that might silently diverge from it. It exists only
/// because `EwcPlusPlus::task_memory` has no public constructor that
/// accepts an externally-synthesized `TaskFisher`, so the benchmark cannot
/// hand a single chosen candidate to the real struct directly.
pub fn single_task_regularization_loss(query: &[f32], task: &TaskRecord, lambda: f32) -> f32 {
    lambda * fisher_weighted_sq(query, &task.weights, &task.fisher) * task.importance / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvector_sona::{EwcConfig, EwcPlusPlus};

    #[test]
    fn squared_l2_known_values() {
        assert!((squared_l2(&[0.0, 0.0], &[3.0, 4.0]) - 25.0).abs() < 1e-6);
        assert!((squared_l2(&[1.0, 1.0], &[1.0, 1.0])).abs() < 1e-6);
    }

    #[test]
    fn fisher_weighted_zero_fisher_is_zero_distance() {
        let d = fisher_weighted_sq(&[5.0, 5.0], &[0.0, 0.0], &[0.0, 0.0]);
        assert_eq!(d, 0.0);
    }

    #[test]
    fn fisher_weighted_matches_manual_sum() {
        let q = [1.0, 2.0, 3.0];
        let w = [0.0, 0.0, 0.0];
        let f = [1.0, 2.0, 0.5];
        // 1*1 + 2*4 + 0.5*9 = 1 + 8 + 4.5 = 13.5
        let d = fisher_weighted_sq(&q, &w, &f);
        assert!((d - 13.5).abs() < 1e-5);
    }

    /// Cross-check against the real production implementation: build a
    /// genuine `EwcPlusPlus`, feed it one task through its public API, and
    /// confirm `single_task_regularization_loss` reproduces exactly what
    /// `EwcPlusPlus::regularization_loss` reports for that same task.
    #[test]
    fn matches_production_ewc_regularization_loss() {
        let dim = 6usize;
        let mut ewc = EwcPlusPlus::new(EwcConfig {
            param_count: dim,
            initial_lambda: 1234.0,
            ..Default::default()
        });

        let optimal_weights = vec![0.3, -0.1, 0.9, 0.0, -0.4, 0.2];
        ewc.set_optimal_weights(&optimal_weights);
        // Build up a non-trivial Fisher diagonal via the real online update.
        let grads = vec![0.9, 0.1, 0.4, 0.0, 0.7, 0.2];
        for _ in 0..200 {
            ewc.update_fisher(&grads);
        }
        ewc.start_new_task();
        let lambda = ewc.lambda();

        let query = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let production_loss = ewc.regularization_loss(&query);

        let task = TaskRecord {
            id: 0,
            weights: optimal_weights,
            fisher: ewc.importance_scores(),
            importance: 1.0,
        };
        let local_loss = single_task_regularization_loss(&query, &task, lambda);

        assert!(
            (production_loss - local_loss).abs() < 1e-3 * production_loss.max(1.0),
            "production={production_loss} local={local_loss}"
        );
    }

    #[test]
    fn metrics_report_distinct_names() {
        assert_ne!(FisherWeighted.name(), PlainL2.name());
    }

    #[test]
    fn plain_l2_ignores_fisher() {
        let t = TaskRecord {
            id: 0,
            weights: vec![1.0, 1.0],
            fisher: vec![1000.0, 0.0],
            importance: 1.0,
        };
        let q = [2.0, 1.0];
        assert!((PlainL2.distance(&q, &t) - 1.0).abs() < 1e-6);
        // FisherWeighted scales the same raw difference by fisher[0] = 1000.
        assert!((FisherWeighted.distance(&q, &t) - 1000.0).abs() < 1e-3);
    }
}
