//! Coherence-drift signal: an O(dims)-per-event running centroid, compared
//! against the centroid captured at the last snapshot via cosine distance.
//!
//! This mirrors the drift concept `ruvector-temporal-coherence` uses for
//! retrieval gating, but applied to whole-store state instead of a single
//! query: "how far has the memory store's semantic center moved since the
//! last durable checkpoint?"

use ruvector_agent_memory::scoring::cosine_sim;

/// Incrementally maintained mean vector over all inserted entries.
///
/// Updating is O(dims) per insert regardless of store size, so drift can be
/// tracked on every event of a long-running stream without the O(n) cost of
/// recomputing a centroid from scratch each time.
#[derive(Debug, Clone)]
pub struct RunningCentroid {
    sum: Vec<f32>,
    count: u64,
}

impl RunningCentroid {
    pub fn new(dims: usize) -> Self {
        Self {
            sum: vec![0.0; dims],
            count: 0,
        }
    }

    pub fn update(&mut self, vector: &[f32]) {
        debug_assert_eq!(vector.len(), self.sum.len());
        for (s, v) in self.sum.iter_mut().zip(vector.iter()) {
            *s += v;
        }
        self.count += 1;
    }

    /// Current mean vector. Zero vector if no updates yet.
    pub fn current(&self) -> Vec<f32> {
        if self.count == 0 {
            return self.sum.clone();
        }
        let n = self.count as f32;
        self.sum.iter().map(|s| s / n).collect()
    }
}

/// Cosine distance (1 - cosine similarity) between two centroids.
///
/// 0.0 means no drift; 2.0 is the maximum (exactly opposite directions).
/// Returns 0.0 if either vector has zero norm (nothing to compare against).
pub fn drift(previous_centroid: &[f32], current_centroid: &[f32]) -> f32 {
    1.0 - cosine_sim(previous_centroid, current_centroid)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn running_centroid_matches_manual_mean() {
        let mut rc = RunningCentroid::new(2);
        rc.update(&[1.0, 0.0]);
        rc.update(&[0.0, 1.0]);
        rc.update(&[2.0, 2.0]);
        let c = rc.current();
        assert!((c[0] - 1.0).abs() < 1e-6);
        assert!((c[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn drift_is_zero_for_identical_direction() {
        let a = [1.0, 0.0, 0.0];
        let b = [2.0, 0.0, 0.0];
        assert!(drift(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn drift_is_max_for_opposite_direction() {
        let a = [1.0, 0.0];
        let b = [-1.0, 0.0];
        assert!((drift(&a, &b) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn drift_grows_with_angle() {
        let a = [1.0, 0.0];
        let small_shift = [0.95, 0.05];
        let large_shift = [0.1, 0.9];
        assert!(drift(&a, &small_shift) < drift(&a, &large_shift));
    }
}
