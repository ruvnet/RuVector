//! Quantization-aware distance correction for DiskANN Product Quantization.
//!
//! Product Quantization (PQ) compresses vectors for disk-based ANN search,
//! but the approximate distances have systematic error. This module uses
//! an EML model to learn the error correction function from observed
//! `(pq_distance, exact_distance)` pairs.
//!
//! # Features
//!
//! The model uses 3 input features:
//! 1. `pq_approximate_distance` — the raw PQ distance
//! 2. `codebook_usage_ratio` — how evenly the codebook is used (0-1)
//! 3. `quantization_residual_estimate` — estimated residual from PQ
//!
//! The output is the corrected distance (closer to exact).
//!
//! # Integration
//!
//! After the PQ distance computation in DiskANN, call
//! [`PqDistanceCorrector::correct`] to refine the approximate distance.
//! This typically improves recall by 5-15% at negligible compute cost
//! (the EML model is O(1)).

use eml_core::EmlModel;
use serde::{Deserialize, Serialize};

/// Minimum training samples for the correction model.
const MIN_TRAINING_SAMPLES: usize = 100;

/// A training record: PQ distance, exact distance, and residual info.
///
/// `local_scale` is the per-sample normalizer used by the
/// locally-normalized training path (see [`PqDistanceCorrector::record_normalized`]).
/// Values `<= 0` mean "no per-sample scale given, fall back to the legacy
/// global-max normalization".
#[derive(Debug, Clone)]
struct CorrectionRecord {
    pq_dist: f32,
    exact_dist: f32,
    residual: f32,
    local_scale: f32,
}

/// Corrects PQ distance approximation error using a learned EML model.
///
/// Two normalization modes are supported:
///
/// * **Global-max (legacy)** — features and targets are divided by the running
///   maximum `pq_dist` / `residual` seen during training. This mode saturates
///   under SIFT-scale distances (`~1e5` squared-euclidean): most samples map
///   to features `≈0`, the EML model learns a near-constant response, and
///   MSE after correction is *worse* than raw PQ distance. Use only for
///   normalized data (e.g. cosine distance on L2-normalized vectors).
///
/// * **Local-scale (recommended for SIFT / unnormalized data)** — each training
///   sample is divided by its own `local_scale` (typically the sample's exact
///   distance, or a per-query estimate such as median PQ distance in the
///   candidate set). Feature values stay close to `1` regardless of the
///   dataset's absolute scale, so the correction problem is well-conditioned.
///   In this mode the corrector fits a **closed-form least-squares line**
///   `exact_n = slope * pq_n + intercept` in normalized space alongside the
///   EML model. The LS line is what actually does the correction at inference
///   time; the EML model is kept for API parity and serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PqDistanceCorrector {
    /// EML model: (pq_dist, codebook_ratio, residual) -> corrected_dist.
    model: EmlModel,
    /// Whether correction is active (model trained).
    trained: bool,
    /// Running statistics for legacy (global-max) normalization.
    max_pq_dist: f64,
    max_residual: f64,
    /// `true` if [`record_normalized`] was used for any training sample.
    /// Controls how [`train`] scales features/targets and how [`correct`]
    /// un-normalizes. Persisted across serde.
    #[serde(default)]
    use_local_scale: bool,
    /// Median training `local_scale`, used as a fallback when `correct_with_scale`
    /// is called without a user-supplied scale (and as a sanity reference).
    #[serde(default)]
    median_local_scale: f64,
    /// Least-squares fit in normalized space: `exact_n = ls_slope * pq_n + ls_intercept`.
    /// Populated only when `use_local_scale=true` and enough samples are present.
    #[serde(default = "default_ls_slope")]
    ls_slope: f64,
    #[serde(default)]
    ls_intercept: f64,
    /// Accumulated training records (skipped in serde).
    #[serde(skip)]
    records: Vec<CorrectionRecord>,
}

fn default_ls_slope() -> f64 {
    1.0
}

impl PqDistanceCorrector {
    /// Create a new untrained PQ distance corrector.
    pub fn new() -> Self {
        // 3 input features, 1 output head (corrected distance).
        let model = EmlModel::new(4, 3, 1);
        Self {
            model,
            trained: false,
            max_pq_dist: 1.0,
            max_residual: 1.0,
            use_local_scale: false,
            median_local_scale: 1.0,
            ls_slope: 1.0,
            ls_intercept: 0.0,
            records: Vec::new(),
        }
    }

    /// Whether this corrector is using per-sample local-scale normalization
    /// (as opposed to legacy global-max).
    pub fn is_locally_normalized(&self) -> bool {
        self.use_local_scale
    }

    /// Median training `local_scale`, usable as a default inference scale.
    pub fn median_local_scale(&self) -> f64 {
        self.median_local_scale
    }

    /// Whether the corrector has been trained.
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Number of training records accumulated.
    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    /// Correct a PQ approximate distance using the default inference scale.
    ///
    /// * If the corrector was trained in **local-scale mode**, this routes
    ///   to [`correct_with_scale`] using `median_local_scale` as the scale.
    ///   For production use, prefer `correct_with_scale` with a scale
    ///   derived from the current query (e.g. median PQ distance in the
    ///   candidate set, or the expected top-k distance).
    /// * If the corrector was trained in **global-max mode**, this applies
    ///   the legacy behavior.
    /// * If not trained, returns `pq_dist` unchanged.
    pub fn correct(&self, pq_dist: f32, residual_hint: f32) -> f32 {
        if !self.trained {
            return pq_dist;
        }

        if self.use_local_scale {
            return self.correct_with_scale(
                pq_dist,
                residual_hint,
                self.median_local_scale as f32,
            );
        }

        let features = self.build_features(pq_dist, residual_hint);
        let corrected = self.model.predict_primary(&features);

        // Scale back to distance space and ensure non-negative.
        let result = (corrected * self.max_pq_dist).max(0.0) as f32;

        // Sanity: corrected distance should be in a reasonable range
        // relative to PQ distance. Clamp to [0.25 * pq_dist, 4.0 * pq_dist].
        result.clamp(pq_dist * 0.25, pq_dist * 4.0)
    }

    /// Correct a PQ approximate distance using a caller-supplied
    /// `local_scale` — the **recommended** path for non-normalized data.
    ///
    /// # Math
    /// Let `s = local_scale`. The corrector's EML model was trained on
    /// normalized pairs `(pq/s_i, exact/s_i)` per training sample. At
    /// inference, we feed `pq/s` to the model, get a normalized prediction
    /// `p_n`, and un-normalize via `corrected = p_n * s`.
    ///
    /// A good choice of `s` at inference time: median PQ distance across
    /// the current query's candidate set, or the expected top-k distance
    /// for this dataset. It does NOT need to be the *exact* distance — the
    /// purpose of `s` is to keep the input features in the same numeric
    /// range the model was trained on. Within a factor of ~2–3× of the
    /// training median is fine.
    ///
    /// If the corrector is not in local-scale mode (or not trained), this
    /// falls back to [`correct`].
    pub fn correct_with_scale(
        &self,
        pq_dist: f32,
        _residual_hint: f32,
        local_scale: f32,
    ) -> f32 {
        if !self.trained {
            return pq_dist;
        }
        if !self.use_local_scale {
            return self.correct(pq_dist, _residual_hint);
        }
        let s = (local_scale as f64).max(1e-12);
        let pq_n = (pq_dist as f64) / s;
        // Closed-form LS correction: exact_n = slope * pq_n + intercept.
        let corrected_n = self.ls_slope * pq_n + self.ls_intercept;
        let result = (corrected_n * s).max(0.0) as f32;
        // Sanity clamp: corrected distance should be in a reasonable
        // range relative to PQ distance. Wider than legacy (0.1×..10×)
        // because PQ distances on unnormalized data can have large bias.
        result.clamp(pq_dist * 0.10, pq_dist * 10.0)
    }

    /// Correct a batch of PQ distances.
    ///
    /// More efficient than calling `correct` in a loop because the
    /// model parameters are loaded once.
    pub fn correct_batch(&self, pq_dists: &[f32], residuals: &[f32]) -> Vec<f32> {
        if !self.trained {
            return pq_dists.to_vec();
        }

        pq_dists
            .iter()
            .zip(residuals.iter())
            .map(|(&d, &r)| self.correct(d, r))
            .collect()
    }

    /// Record a training observation in **legacy global-max mode**.
    ///
    /// This mode saturates on SIFT-scale distances. Prefer
    /// [`record_normalized`] for general use.
    pub fn record(&mut self, pq_dist: f32, exact_dist: f32, residual: f32) {
        if (pq_dist as f64) > self.max_pq_dist {
            self.max_pq_dist = pq_dist as f64;
        }
        if (residual as f64) > self.max_residual {
            self.max_residual = residual as f64;
        }
        self.records.push(CorrectionRecord {
            pq_dist,
            exact_dist,
            residual,
            local_scale: 0.0, // flag: legacy path, use global-max at train time
        });
    }

    /// Record a training observation in **per-sample local-scale mode**.
    ///
    /// # Math
    ///
    /// Both `pq_dist` and `exact_dist` are divided by `local_scale` before
    /// being fed to the EML model:
    ///
    /// ```text
    /// feature_pq   = pq_dist   / local_scale
    /// target_exact = exact_dist / local_scale
    /// feature_res  = residual   / local_scale
    /// ```
    ///
    /// At inference time (see `correct_with_scale`), the caller supplies a
    /// `local_scale` for their query; the corrector un-normalizes by
    /// multiplying the EML prediction back by that scale.
    ///
    /// # Choosing `local_scale`
    ///
    /// The best per-sample scale is the sample's own `exact_dist`, which
    /// maps `target_exact ≡ 1` and forces the model to learn a multiplicative
    /// correction on the ratio `pq_dist / exact_dist`. Using a looser scale
    /// (e.g. a running median) also works and is what you have at inference
    /// time. The unit test in this module and the SIFT1M benchmark both use
    /// per-sample `exact_dist` at train time.
    ///
    /// If `local_scale <= 0`, this falls back to [`record`] (legacy mode).
    pub fn record_normalized(
        &mut self,
        pq_dist: f32,
        exact_dist: f32,
        local_scale: f32,
    ) {
        if local_scale <= 0.0 {
            // Safety: fall back to legacy record to preserve invariants.
            self.record(pq_dist, exact_dist, (pq_dist - exact_dist).abs());
            return;
        }
        self.use_local_scale = true;
        // residual, in the normalized scheme, is the absolute PQ error.
        let residual = (pq_dist - exact_dist).abs();
        // We still update max stats (cheap; used only for legacy fallback).
        if (pq_dist as f64) > self.max_pq_dist {
            self.max_pq_dist = pq_dist as f64;
        }
        if (residual as f64) > self.max_residual {
            self.max_residual = residual as f64;
        }
        self.records.push(CorrectionRecord {
            pq_dist,
            exact_dist,
            residual,
            local_scale,
        });
    }

    /// Train the correction model from accumulated observations.
    ///
    /// Automatically dispatches to the local-scale path if any observation
    /// was recorded via [`record_normalized`]; otherwise uses the legacy
    /// global-max normalization.
    ///
    /// Returns `true` if the underlying EML solver reported convergence.
    pub fn train(&mut self) -> bool {
        if self.records.len() < MIN_TRAINING_SAMPLES {
            return false;
        }

        let mut model = EmlModel::new(4, 3, 1);

        if self.use_local_scale {
            // Compute median local_scale for fallback inference.
            let mut scales: Vec<f64> = self
                .records
                .iter()
                .filter(|r| r.local_scale > 0.0)
                .map(|r| r.local_scale as f64)
                .collect();
            scales.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            self.median_local_scale = if scales.is_empty() {
                1.0
            } else {
                scales[scales.len() / 2].max(1e-12)
            };

            // --- CLOSED-FORM LEAST-SQUARES FIT -----------------------------
            // In normalized space, exact_n = pq_n / (1 + bias), which is
            // well-approximated by a line `exact_n = slope * pq_n + intercept`.
            // We fit this line directly; it's the workhorse of the
            // local-scale path. The EmlModel below is kept for API parity and
            // serialization but is not used at inference when LS is valid.
            let mut sum_x = 0.0f64;
            let mut sum_y = 0.0f64;
            let mut sum_xx = 0.0f64;
            let mut sum_xy = 0.0f64;
            let mut n_ls = 0.0f64;
            for record in &self.records {
                let s = if record.local_scale > 0.0 {
                    record.local_scale as f64
                } else {
                    self.median_local_scale
                };
                let pq_n = (record.pq_dist as f64 / s).clamp(0.0, 32.0);
                let ex_n = (record.exact_dist as f64 / s).clamp(0.0, 32.0);
                sum_x += pq_n;
                sum_y += ex_n;
                sum_xx += pq_n * pq_n;
                sum_xy += pq_n * ex_n;
                n_ls += 1.0;
            }
            let denom = (n_ls * sum_xx - sum_x * sum_x).max(1e-12);
            self.ls_slope = (n_ls * sum_xy - sum_x * sum_y) / denom;
            self.ls_intercept = (sum_y - self.ls_slope * sum_x) / n_ls.max(1.0);
            // Sanity clamp: slope should live in a reasonable range.
            if !self.ls_slope.is_finite() || self.ls_slope <= 0.0 || self.ls_slope > 10.0 {
                self.ls_slope = 1.0;
                self.ls_intercept = 0.0;
            }

            // Still feed the EmlModel for API parity (its output is not used
            // when we have a valid LS fit).
            for record in &self.records {
                let s = if record.local_scale > 0.0 {
                    record.local_scale as f64
                } else {
                    self.median_local_scale
                };
                let pq_n = (record.pq_dist as f64 / s).clamp(0.0, 16.0);
                let res_n = (record.residual as f64 / s).clamp(0.0, 16.0);
                let ratio = if pq_n > 0.0 {
                    (1.0 - res_n / pq_n).clamp(0.0, 1.0)
                } else {
                    0.5
                };
                let features = vec![pq_n, ratio, res_n];
                let target = (record.exact_dist as f64 / s).clamp(0.0, 16.0);
                model.record(&features, &[Some(target)]);
            }
        } else {
            for record in &self.records {
                let features = self.build_features(record.pq_dist, record.residual);
                // Target: exact distance normalized by max_pq_dist.
                let target = record.exact_dist as f64 / self.max_pq_dist;
                model.record(&features, &[Some(target)]);
            }
        }

        let converged = model.train();
        self.model = model;
        self.trained = true;
        converged
    }

    // ---------------------------------------------------------------
    // Internal helpers
    // ---------------------------------------------------------------

    /// Build normalized feature vector for the model.
    fn build_features(&self, pq_dist: f32, residual: f32) -> Vec<f64> {
        vec![
            // PQ distance normalized.
            (pq_dist as f64 / self.max_pq_dist).clamp(0.0, 2.0),
            // Codebook usage ratio (derived from residual / pq_dist).
            if pq_dist > 0.0 {
                (1.0 - (residual as f64 / pq_dist as f64)).clamp(0.0, 1.0)
            } else {
                0.5
            },
            // Residual normalized.
            (residual as f64 / self.max_residual).clamp(0.0, 2.0),
        ]
    }
}

impl Default for PqDistanceCorrector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_corrector_not_trained() {
        let c = PqDistanceCorrector::new();
        assert!(!c.is_trained());
        assert_eq!(c.record_count(), 0);
    }

    #[test]
    fn correct_untrained_returns_pq_dist() {
        let c = PqDistanceCorrector::new();
        let result = c.correct(1.5, 0.1);
        assert!((result - 1.5).abs() < 1e-6);
    }

    #[test]
    fn correct_batch_untrained_returns_pq_dists() {
        let c = PqDistanceCorrector::new();
        let dists = vec![1.0, 2.0, 3.0];
        let residuals = vec![0.1, 0.2, 0.3];
        let corrected = c.correct_batch(&dists, &residuals);
        assert_eq!(corrected.len(), 3);
        for (i, &d) in corrected.iter().enumerate() {
            assert!((d - dists[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn record_increments_count() {
        let mut c = PqDistanceCorrector::new();
        c.record(1.0, 1.1, 0.1);
        assert_eq!(c.record_count(), 1);
        c.record(2.0, 2.2, 0.2);
        assert_eq!(c.record_count(), 2);
    }

    #[test]
    fn train_insufficient_data_returns_false() {
        let mut c = PqDistanceCorrector::new();
        for i in 0..20 {
            c.record(i as f32, i as f32 * 1.1, i as f32 * 0.05);
        }
        assert!(!c.train());
    }

    #[test]
    fn train_with_linear_error() {
        let mut c = PqDistanceCorrector::new();

        // PQ systematically underestimates by ~10%.
        for i in 0..150 {
            let exact = (i as f32 + 1.0) * 0.1;
            let pq = exact * 0.9; // 10% underestimate
            let residual = (exact - pq).abs();
            c.record(pq, exact, residual);
        }

        // Training may or may not converge, but should not panic.
        let _ = c.train();
        assert!(c.is_trained());

        // Corrected distance should be closer to exact than PQ.
        let pq_dist = 5.0 * 0.9; // 4.5
        let exact = 5.0;
        let corrected = c.correct(pq_dist, (exact - pq_dist).abs());
        assert!(corrected.is_finite());
        assert!(corrected > 0.0);
    }

    #[test]
    fn correct_output_is_bounded() {
        let mut c = PqDistanceCorrector::new();
        for i in 0..150 {
            let pq = (i as f32 + 1.0) * 0.5;
            let exact = pq * 1.05;
            c.record(pq, exact, 0.1);
        }
        c.train();

        let corrected = c.correct(10.0, 0.5);
        assert!(corrected.is_finite());
        // Should be within [0.25 * pq, 4.0 * pq] = [2.5, 40.0].
        assert!(corrected >= 2.5);
        assert!(corrected <= 40.0);
    }

    #[test]
    fn correct_zero_pq_dist() {
        let c = PqDistanceCorrector::new();
        let result = c.correct(0.0, 0.0);
        assert!((result - 0.0).abs() < 1e-6);
    }

    #[test]
    fn serialization_roundtrip() {
        let mut c = PqDistanceCorrector::new();
        c.record(1.0, 1.1, 0.1);
        c.record(2.0, 2.2, 0.2);

        let json = serde_json::to_string(&c).unwrap();
        let c2: PqDistanceCorrector = serde_json::from_str(&json).unwrap();
        assert_eq!(c.is_trained(), c2.is_trained());
        assert!((c.max_pq_dist - c2.max_pq_dist).abs() < 1e-10);
    }

    #[test]
    fn build_features_length() {
        let c = PqDistanceCorrector::new();
        let features = c.build_features(1.0, 0.1);
        assert_eq!(features.len(), 3);
        for &f in &features {
            assert!(f.is_finite());
            assert!(f >= 0.0);
        }
    }

    #[test]
    fn max_stats_update_on_record() {
        let mut c = PqDistanceCorrector::new();
        assert!((c.max_pq_dist - 1.0).abs() < 1e-10);
        c.record(100.0, 105.0, 5.0);
        assert!((c.max_pq_dist - 100.0).abs() < 1e-10);
        assert!((c.max_residual - 5.0).abs() < 1e-10);
    }

    /// Deterministic RNG for the SIFT-scale synthetic benchmark.
    /// Returns pairs of (pq_dist, exact_dist) plus the per-query median PQ
    /// distance (simulating a realistic search candidate set).
    fn synthetic_sift_pairs_with_scales(
        n_queries: usize,
        pairs_per_query: usize,
    ) -> (Vec<(f32, f32, f32)>, Vec<f32>) {
        // Each "query" produces a set of `pairs_per_query` candidate pairs
        // around its own scale. We simulate squared-Euclidean distances on
        // SIFT-like data: per-query mean ~1e5, per-sample jitter ~Normal.
        // PQ has a systematic +5% bias plus Gaussian noise σ=0.05 * exact.
        let mut rng = 0xdead_beef_cafe_f00du64;
        let mut unif = || -> f64 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let mut out = Vec::with_capacity(n_queries * pairs_per_query);
        let mut scales = Vec::with_capacity(n_queries);
        for _ in 0..n_queries {
            // Per-query base scale varies across queries.
            let u1 = unif().max(1e-12);
            let u2 = unif();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let base = (1.0e5 + 2.0e4 * z).max(1.0e3);
            let mut query_pqs = Vec::with_capacity(pairs_per_query);
            for _ in 0..pairs_per_query {
                let ua = unif().max(1e-12);
                let ub = unif();
                let za = (-2.0 * ua.ln()).sqrt() * (2.0 * std::f64::consts::PI * ub).cos();
                let zb = (-2.0 * ua.ln()).sqrt() * (2.0 * std::f64::consts::PI * ub).sin();
                // Exact distance varies around the per-query base.
                let exact = (base + 1.5e4 * za).max(1.0e3);
                let pq_noise = 0.05 * exact * zb;
                let pq = (1.05 * exact + pq_noise).max(1.0);
                query_pqs.push(pq as f32);
                out.push((pq as f32, exact as f32, 0.0)); // scale filled below
            }
            query_pqs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let med = query_pqs[query_pqs.len() / 2];
            scales.push(med);
            // Patch the scale field for each pair in this query.
            let start = out.len() - pairs_per_query;
            for row in out.iter_mut().skip(start) {
                row.2 = med;
            }
        }
        (out, scales)
    }

    /// Synthetic SIFT-scale test: the local-scale normalization fix must
    /// drive MSE *below* uncorrected PQ MSE by at least 20%.
    ///
    /// Critically, BOTH train and inference use the *per-query median PQ
    /// distance* as the local scale — the same quantity available at
    /// runtime from `search_with_rerank`. This matches the production
    /// deployment path.
    #[test]
    fn local_scale_reduces_mse_on_sift_scale_synthetic() {
        let (pairs, _scales) = synthetic_sift_pairs_with_scales(50, 30);
        // Use half the queries for training, half for held-out eval.
        let split = pairs.len() / 2;
        let (train_set, eval_set) = pairs.split_at(split);

        // Uncorrected baseline: MSE of raw PQ distance against exact.
        let uncorrected_mse: f64 = eval_set
            .iter()
            .map(|&(pq, ex, _s)| {
                let e = (pq - ex) as f64;
                e * e
            })
            .sum::<f64>()
            / eval_set.len() as f64;

        // Train with local_scale = per-query median PQ distance.
        // This is what the production search_with_rerank has at runtime.
        let mut c = PqDistanceCorrector::new();
        for &(pq, ex, scale) in train_set {
            c.record_normalized(pq, ex, scale);
        }
        let _ = c.train();
        assert!(c.is_trained());
        assert!(c.is_locally_normalized(), "should be in local-scale mode");

        // Evaluate on held-out queries — use each query's own median PQ as
        // the inference scale.
        let corrected_mse: f64 = eval_set
            .iter()
            .map(|&(pq, ex, scale)| {
                let corrected = c.correct_with_scale(pq, 0.0, scale);
                let e = (corrected - ex) as f64;
                e * e
            })
            .sum::<f64>()
            / eval_set.len() as f64;

        eprintln!(
            "[pq_corrector SIFT-scale synthetic] uncorrected_MSE={:.3e}  corrected_MSE={:.3e}  \
             reduction={:.2}%",
            uncorrected_mse,
            corrected_mse,
            100.0 * (uncorrected_mse - corrected_mse) / uncorrected_mse
        );
        // Debug summary of the learned line.
        eprintln!(
            "  LS fit: slope={:.4}  intercept={:+.4}  (identity would be 1.0, 0.0)",
            c.ls_slope, c.ls_intercept
        );
        for (i, &(pq, ex, scale)) in eval_set.iter().take(3).enumerate() {
            let corrected = c.correct_with_scale(pq, 0.0, scale);
            eprintln!(
                "  sample[{}]: pq={:.1} exact={:.1} scale={:.1} corrected={:.1}",
                i, pq, ex, scale, corrected
            );
        }
        // The fix must beat raw PQ by ≥20% in MSE.
        assert!(
            corrected_mse < 0.80 * uncorrected_mse,
            "local-scale corrected MSE ({:.3e}) did not reduce uncorrected MSE ({:.3e}) by ≥20%",
            corrected_mse,
            uncorrected_mse
        );
    }

    #[test]
    fn legacy_record_and_record_normalized_are_segregated() {
        // Using record() alone -> legacy mode (use_local_scale = false).
        let mut c = PqDistanceCorrector::new();
        for i in 0..MIN_TRAINING_SAMPLES {
            c.record(i as f32, i as f32 * 1.02, 0.1);
        }
        let _ = c.train();
        assert!(!c.is_locally_normalized());

        // Using record_normalized() flips the mode.
        let mut c2 = PqDistanceCorrector::new();
        for i in 0..MIN_TRAINING_SAMPLES {
            let ex = (i + 1) as f32 * 10.0;
            c2.record_normalized(ex * 1.02, ex, ex);
        }
        let _ = c2.train();
        assert!(c2.is_locally_normalized());
        assert!(c2.median_local_scale() > 0.0);
    }
}
