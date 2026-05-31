//! Graph-coherence drift detector.
//!
//! Maintains a fixed-capacity ring buffer of recent vectors. After each
//! insertion it recomputes the k-NN graph coherence: mean cosine similarity
//! across all k-nearest-neighbour edges. When coherence drops below a
//! baseline learned during warmup, semantic drift is signalled.
//!
//! Also provides [`GraphCoherenceDriftDetector::compaction_hint`], which
//! scores every buffered vector by its local coherence and flags those
//! below threshold for removal — giving the agent memory store an explicit
//! pruning target.
//!
//! Complexity: O(capacity² · dim) per observation (dominated by full k-NN
//! rebuild). Keep `capacity` ≤ 512 for interactive write latency.

use crate::{
    math::cosine_sim,
    CompactionHint, DriftDetector, DriftScore, DriftSummary,
};

pub struct GraphCoherenceDriftDetector {
    memory: Vec<Vec<f32>>,
    capacity: usize,
    /// Retained for API compatibility; pairwise metric uses all pairs.
    #[allow(dead_code)]
    k: usize,
    baseline_coherence: f32,
    baseline_set: bool,
    n_warmup: usize,
    threshold_drop: f32,
    n_observed: usize,
    n_alerts: usize,
    current_score: f32,
    current_coherence: f32,
    peak_score: f32,
}

impl GraphCoherenceDriftDetector {
    /// * `capacity`        — max vectors kept in the sliding window
    /// * `k`               — k for k-NN graph coherence
    /// * `n_warmup`        — observations to establish baseline coherence
    /// * `threshold_drop`  — alert when coherence drops by this much (e.g. 0.10)
    pub fn new(capacity: usize, k: usize, n_warmup: usize, threshold_drop: f32) -> Self {
        Self {
            memory: Vec::with_capacity(capacity),
            capacity,
            k,
            baseline_coherence: 0.0,
            baseline_set: false,
            n_warmup,
            threshold_drop,
            n_observed: 0,
            n_alerts: 0,
            current_score: 0.0,
            current_coherence: 1.0,
            peak_score: 0.0,
        }
    }

    /// Mean pairwise cosine similarity across all pairs in `memory`.
    ///
    /// Unlike k-NN coherence, pairwise mean is sensitive to mixed-cluster
    /// windows: when half the vectors are from cluster A (high intra-sim)
    /// and half from cluster B (high intra-sim, near-zero cross-sim), the
    /// pairwise mean drops proportionally to the mixing fraction.
    ///
    /// For a 50/50 A+B mix with intra-sim ≈ 0.9 and cross-sim ≈ 0:
    ///   coherence ≈ 0.9 × (0.5² + 0.5²) = 0.45
    fn graph_coherence(&self) -> f32 {
        let n = self.memory.len();
        if n < 2 {
            return 1.0;
        }
        let mut total = 0.0f32;
        let mut count = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                total += cosine_sim(&self.memory[i], &self.memory[j]);
                count += 1;
            }
        }
        if count == 0 {
            1.0
        } else {
            total / count as f32
        }
    }

    /// Per-vector coherence: mean cosine-sim to all other vectors in memory.
    fn per_vector_coherence(&self) -> Vec<f32> {
        let n = self.memory.len();
        if n < 2 {
            return vec![1.0; n];
        }
        let mut out = vec![0.0f32; n];
        for i in 0..n {
            let sum: f32 = (0..n)
                .filter(|&j| j != i)
                .map(|j| cosine_sim(&self.memory[i], &self.memory[j]))
                .sum();
            out[i] = sum / (n - 1) as f32;
        }
        out
    }
}

impl DriftDetector for GraphCoherenceDriftDetector {
    fn observe(&mut self, vector: &[f32]) -> DriftScore {
        self.n_observed += 1;

        // Ring-buffer insert.
        if self.memory.len() >= self.capacity {
            self.memory.remove(0);
        }
        self.memory.push(vector.to_vec());

        let coh = self.graph_coherence();
        self.current_coherence = coh;

        // Warmup: learn baseline coherence.
        if self.n_observed == self.n_warmup && !self.baseline_set {
            self.baseline_coherence = coh;
            self.baseline_set = true;
        }

        if !self.baseline_set {
            return DriftScore {
                score: 0.0,
                alert: false,
                epoch: self.n_observed as u64,
                detector: self.name(),
            };
        }

        let drop = (self.baseline_coherence - coh).max(0.0);
        // Normalise to [0, 1] relative to threshold.
        let score = (drop / self.threshold_drop.max(1e-8)).min(1.0);
        self.current_score = score;
        self.peak_score = self.peak_score.max(score);

        let alert = drop > self.threshold_drop;
        if alert {
            self.n_alerts += 1;
        }

        DriftScore {
            score,
            alert,
            epoch: self.n_observed as u64,
            detector: self.name(),
        }
    }

    fn is_drifted(&self) -> bool {
        self.baseline_set && self.current_score >= 1.0
    }

    fn reset(&mut self) {
        self.memory.clear();
        self.baseline_coherence = 0.0;
        self.baseline_set = false;
        self.n_observed = 0;
        self.n_alerts = 0;
        self.current_score = 0.0;
        self.current_coherence = 1.0;
        self.peak_score = 0.0;
    }

    fn name(&self) -> &'static str {
        "GraphCoherence"
    }

    fn summary(&self) -> DriftSummary {
        DriftSummary {
            detector: self.name(),
            n_observed: self.n_observed,
            n_alerts: self.n_alerts,
            current_score: self.current_score,
            peak_score: self.peak_score,
        }
    }

    fn compaction_hint(&self) -> Option<CompactionHint> {
        if self.memory.is_empty() || !self.baseline_set {
            return None;
        }
        let threshold = (self.current_coherence + self.baseline_coherence) / 2.0;
        let scores = self.per_vector_coherence();
        let flagged_ids: Vec<usize> = scores
            .iter()
            .enumerate()
            .filter(|(_, &s)| s < threshold)
            .map(|(i, _)| i)
            .collect();
        Some(CompactionHint {
            n_vectors: self.memory.len(),
            n_flagged: flagged_ids.len(),
            flagged_ids,
            coherence_scores: scores,
            drift_threshold: threshold,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn axis_vec(dim: usize, idx: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        v[idx] = 1.0;
        v
    }

    #[test]
    fn coherent_memory_no_alert() {
        let mut det = GraphCoherenceDriftDetector::new(64, 5, 20, 0.15);
        let v = axis_vec(32, 0);
        for i in 0..60 {
            let s = det.observe(&v);
            assert!(!s.alert, "false positive at epoch {}", i + 1);
        }
    }

    #[test]
    fn orthogonal_drift_detected() {
        let dim = 32;
        let mut det = GraphCoherenceDriftDetector::new(64, 5, 30, 0.05);
        for _ in 0..30 {
            det.observe(&axis_vec(dim, 0));
        }
        let mut alerted = false;
        for _ in 0..50 {
            let s = det.observe(&axis_vec(dim, 1));
            if s.alert {
                alerted = true;
            }
        }
        assert!(alerted, "GraphCoherence failed to detect orthogonal drift");
    }

    #[test]
    fn compaction_hint_returns_some_after_warmup() {
        let dim = 16;
        let mut det = GraphCoherenceDriftDetector::new(32, 4, 10, 0.10);
        for _ in 0..20 {
            det.observe(&axis_vec(dim, 0));
        }
        // Inject some drifted vectors to create low-coherence entries.
        for _ in 0..5 {
            det.observe(&axis_vec(dim, 1));
        }
        let hint = det.compaction_hint().expect("should return a hint");
        assert!(hint.n_vectors > 0);
        assert!(hint.coherence_scores.len() == hint.n_vectors);
    }

    #[test]
    fn reset_clears_all_state() {
        let mut det = GraphCoherenceDriftDetector::new(32, 4, 10, 0.10);
        for _ in 0..20 {
            det.observe(&axis_vec(8, 0));
        }
        det.reset();
        assert_eq!(det.n_observed, 0);
        assert!(det.memory.is_empty());
        assert!(!det.baseline_set);
    }

    #[test]
    fn graph_coherence_perfect_cluster() {
        let mut det = GraphCoherenceDriftDetector::new(32, 3, 5, 0.10);
        let v = vec![1.0f32, 0.0, 0.0, 0.0];
        for _ in 0..10 {
            det.observe(&v);
        }
        // All vectors identical → coherence ≈ 1.0.
        assert!(
            det.current_coherence > 0.95,
            "coherence={:.3}",
            det.current_coherence
        );
    }
}
