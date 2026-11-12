//! Calibration: build an ef → recall monotone table from sample queries.
//!
//! ## How it works
//!
//! 1. Run beam search at a set of candidate `ef` values on a small random
//!    sample of queries drawn from the data distribution.
//! 2. For each `ef`, compute mean recall@k against brute-force ground truth.
//! 3. Store the resulting `(ef, mean_recall)` pairs as a sorted table.
//! 4. At query time, binary-search the table to find the smallest `ef` ≥
//!    the caller's recall target.
//!
//! ## Monotonicity
//!
//! Recall is weakly monotone in `ef` for any fixed graph: larger beam width
//! can only find more neighbours. The table enforces this by sweeping in
//! ascending `ef` order and keeping a running maximum.
//!
//! ## Sample size
//!
//! Using ~5–10% of the dataset as calibration queries gives accurate estimates
//! without significant overhead. The calibration is one-time (amortised over
//! all subsequent queries).

use crate::dataset::ground_truth;
use crate::graph::FlatGraph;
use crate::metrics::recall_at_k;
use crate::search::beam_search;

/// Monotone table mapping ef → achieved mean recall@k.
#[derive(Debug, Clone)]
pub struct CalibrationTable {
    /// Sorted ascending by ef. Each entry is `(ef, mean_recall)`.
    entries: Vec<(usize, f32)>,
    /// Fallback ef if the table is empty or target exceeds all calibrated recalls.
    pub max_ef: usize,
    /// Result count used when measuring the table.
    pub k: usize,
}

impl CalibrationTable {
    /// Return the minimum ef that achieves `recall_target` for searches of size `k`.
    ///
    /// Falls back to `max_ef` if no calibrated ef reaches the target.
    pub fn min_ef_for_target(&self, recall_target: f32, k: usize) -> usize {
        assert_eq!(
            k, self.k,
            "calibration table was built for k={} but queried with k={k}",
            self.k
        );
        let target = recall_target.clamp(0.0, 1.0);
        // Ensure ef ≥ k.
        for &(ef, recall) in &self.entries {
            if recall >= target && ef >= k {
                return ef;
            }
        }
        self.max_ef.max(k)
    }

    /// All calibration entries (ef, mean_recall), sorted by ef.
    pub fn entries(&self) -> &[(usize, f32)] {
        &self.entries
    }
}

/// Builds a [`CalibrationTable`] from sample queries.
pub struct Calibrator<'g> {
    pub graph: &'g FlatGraph,
    pub entry: usize,
    /// Candidate ef values to probe. Will be sorted ascending.
    pub ef_candidates: Vec<usize>,
    /// Number of sample queries to use for calibration.
    pub n_sample: usize,
    /// Seed for selecting sample queries from the dataset.
    pub seed: u64,
}

impl<'g> Calibrator<'g> {
    /// Run calibration: measure mean recall@k at each ef candidate.
    ///
    /// `k` is the search result count to calibrate for.
    /// `data_queries` are queries drawn from the same distribution as production
    /// queries; if `None`, random offsets from the stored vectors are used.
    pub fn calibrate(
        &mut self,
        k: usize,
        sample_queries: &[f32],
        sample_gt: &[Vec<u32>],
    ) -> CalibrationTable {
        assert!(k > 0, "calibration k must be greater than zero");
        assert!(
            (sample_queries.len())
                .checked_rem(self.graph.config.dims)
                .expect("attempt to calculate the remainder with a divisor of zero")
                == 0,
            "sample query buffer must contain whole vectors"
        );
        self.ef_candidates.sort_unstable();
        self.ef_candidates.dedup();

        let n_sample = self
            .n_sample
            .min(sample_queries.len() / self.graph.config.dims)
            .min(sample_gt.len());
        let dims = self.graph.config.dims;

        let mut entries: Vec<(usize, f32)> = Vec::with_capacity(self.ef_candidates.len());
        let mut running_max_recall = 0.0f32;

        for &ef in &self.ef_candidates {
            let ef_clamped = ef.max(k);
            let mut total_recall = 0.0f32;

            for qi in 0..n_sample {
                let q = &sample_queries[qi * dims..(qi + 1) * dims];
                let results = beam_search(self.graph, q, k, ef_clamped, self.entry);
                let ids: Vec<u32> = results.iter().map(|r| r.id).collect();
                total_recall += recall_at_k(&ids, &sample_gt[qi]);
            }

            let mean_recall = if n_sample > 0 {
                total_recall / n_sample as f32
            } else {
                0.0
            };

            // Enforce monotonicity: recall at larger ef is never lower.
            running_max_recall = running_max_recall.max(mean_recall);
            entries.push((ef_clamped, running_max_recall));
        }

        let max_ef = entries.last().map(|&(ef, _)| ef).unwrap_or(64);
        CalibrationTable { entries, max_ef, k }
    }

    /// Convenience: generate calibration ground truth from graph vectors.
    ///
    /// Uses the first `n_sample` vectors as calibration queries (they are
    /// in the graph, providing realistic distance distributions).
    pub fn build_ground_truth(&self, k: usize, n_sample: usize) -> (Vec<f32>, Vec<Vec<u32>>) {
        let dims = self.graph.config.dims;
        let n = n_sample.min(self.graph.n);
        let queries: Vec<f32> = self.graph.vectors[..n * dims].to_vec();
        let gt = ground_truth(&self.graph.vectors, &queries, dims, k);
        (queries, gt)
    }
}
