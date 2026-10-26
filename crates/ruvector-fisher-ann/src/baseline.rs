//! Brute-force linear scan: the "baseline" variant.
//!
//! This is a direct scale-up of what `EwcPlusPlus::apply_constraints` and
//! `apply_anchor` already do in production today — loop over every stored
//! task and score it — just generalized from "at most `max_tasks` (10)"
//! to arbitrary `N`. It is also the ground truth recall is measured
//! against: by construction it returns the exact top-k under whichever
//! metric it is given.

use crate::types::{TaskMetric, TaskRecord};

/// Return the exact `k` nearest `TaskRecord`s to `query` under `metric`,
/// sorted ascending by distance, as `(distance, index_into_records)`.
pub fn brute_force_topk<M: TaskMetric>(
    metric: M,
    records: &[TaskRecord],
    query: &[f32],
    k: usize,
) -> Vec<(f32, usize)> {
    let mut scored: Vec<(f32, usize)> = records
        .iter()
        .enumerate()
        .map(|(i, t)| (metric.distance(query, t), i))
        .collect();
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::PlainL2;

    #[test]
    fn returns_exact_nearest_first() {
        let records = vec![
            TaskRecord {
                id: 0,
                weights: vec![0.0, 0.0],
                fisher: vec![1.0, 1.0],
                importance: 1.0,
            },
            TaskRecord {
                id: 1,
                weights: vec![1.0, 0.0],
                fisher: vec![1.0, 1.0],
                importance: 1.0,
            },
            TaskRecord {
                id: 2,
                weights: vec![5.0, 5.0],
                fisher: vec![1.0, 1.0],
                importance: 1.0,
            },
        ];
        let top = brute_force_topk(PlainL2, &records, &[0.1, 0.0], 2);
        assert_eq!(top[0].1, 0);
        assert_eq!(top[1].1, 1);
    }

    #[test]
    fn k_larger_than_n_returns_all() {
        let records = vec![TaskRecord {
            id: 0,
            weights: vec![0.0],
            fisher: vec![1.0],
            importance: 1.0,
        }];
        let top = brute_force_topk(PlainL2, &records, &[0.0], 10);
        assert_eq!(top.len(), 1);
    }
}
