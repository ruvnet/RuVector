//! Pass/fail boundary analysis using spectral graph partitioning (ADR-260 §13).
//!
//! [`explain_boundary`] builds a cosine-similarity graph over the union of
//! pass and fail experiment embeddings, computes the Fiedler partition via
//! [`ruvector_coherence::spectral`], and identifies which [`OpticalConfig`]
//! variable most consistently separates the two outcome groups.

use photonlayer_core::prelude::OpticalConfig;
use ruvector_coherence::cosine_similarity;
use ruvector_coherence::spectral::{
    estimate_fiedler, estimate_largest_eigenvalue, estimate_spectral_gap, CsrMatrixView,
};
use serde::{Deserialize, Serialize};

use crate::memory::ExperimentMemory;

/// Threshold above which a cosine similarity edge is included in the graph.
const SIM_THRESHOLD: f64 = 0.5;

/// Summary of the pass/fail decision boundary.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoundaryReport {
    /// Name of the config variable that most consistently separates pass/fail.
    pub dominant_variable: String,
    /// Runner-up config variable.
    pub secondary_variable: String,
    /// Spectral gap of the similarity graph (larger = clearer boundary).
    pub spectral_gap: f64,
    /// Human-readable recommendation derived from the analysis.
    pub recommendation: String,
}

/// Compute the mean value of a config field across a slice of configs.
fn mean_f64(configs: &[&OpticalConfig], extract: impl Fn(&OpticalConfig) -> f64) -> f64 {
    if configs.is_empty() {
        return 0.0;
    }
    configs.iter().map(|c| extract(c)).sum::<f64>() / configs.len() as f64
}

/// Mean absolute deviation around `mean` for a set of values.
fn mad(values: impl Iterator<Item = f64>, mean: f64) -> f64 {
    let vals: Vec<f64> = values.collect();
    if vals.is_empty() {
        return 0.0;
    }
    vals.iter().map(|v| (v - mean).abs()).sum::<f64>() / vals.len() as f64
}

/// Score how discriminative a real-valued config field is between two groups.
///
/// Returns the absolute difference of group means, normalised by the pooled MAD.
/// A higher score means the variable separates the groups more cleanly.
fn discriminability(pass_vals: &[f64], fail_vals: &[f64]) -> f64 {
    if pass_vals.is_empty() || fail_vals.is_empty() {
        return 0.0;
    }
    let mp: f64 = pass_vals.iter().sum::<f64>() / pass_vals.len() as f64;
    let mf: f64 = fail_vals.iter().sum::<f64>() / fail_vals.len() as f64;
    let diff = (mp - mf).abs();

    let all_mean = (mp + mf) / 2.0;
    let spread = mad(pass_vals.iter().chain(fail_vals.iter()).copied(), all_mean);
    if spread < 1e-30 {
        // No within-group spread: if the means differ it is a perfect separator.
        if diff > 1e-9 {
            1e9
        } else {
            0.0
        }
    } else {
        diff / spread
    }
}

/// Analyse the pass/fail boundary in `memory`.
///
/// # Arguments
/// * `memory` — the experiment store to analyse.
/// * `pass_label` — label string used for successful experiments (e.g. `"pass"`).
/// * `fail_label` — label string used for failing experiments (e.g. `"fail"`).
///
/// Returns a [`BoundaryReport`] identifying the dominant config variable and
/// the spectral gap of the experiment similarity graph.
///
/// If either group is empty the report contains `"insufficient_data"` variables
/// and a spectral gap of `0.0`.
pub fn explain_boundary(
    memory: &ExperimentMemory,
    pass_label: &str,
    fail_label: &str,
) -> BoundaryReport {
    let pass_records = memory.by_label(pass_label);
    let fail_records = memory.by_label(fail_label);

    if pass_records.is_empty() || fail_records.is_empty() {
        return BoundaryReport {
            dominant_variable: "insufficient_data".to_owned(),
            secondary_variable: "insufficient_data".to_owned(),
            spectral_gap: 0.0,
            recommendation: "Need at least one pass and one fail experiment.".to_owned(),
        };
    }

    // ---- build embedding list in a stable order: passes first ----
    let all_records: Vec<&crate::memory::ExperimentRecord> = pass_records
        .iter()
        .chain(fail_records.iter())
        .copied()
        .collect();
    let n = all_records.len();

    // ---- build cosine-similarity graph edges ----
    let mut edges: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let sim = cosine_similarity(&all_records[i].embedding, &all_records[j].embedding);
            if sim >= SIM_THRESHOLD {
                edges.push((i, j, sim));
            }
        }
    }

    // ---- build Laplacian and estimate spectral gap ----
    let lap = CsrMatrixView::build_laplacian(n, &edges);
    let (fiedler_raw, _fiedler_vec) = estimate_fiedler(&lap, 100, 1e-6);
    let largest = estimate_largest_eigenvalue(&lap, 100);
    let spectral_gap = estimate_spectral_gap(fiedler_raw, largest);

    // ---- find which config variable best separates pass / fail ----
    let pass_cfgs: Vec<&OpticalConfig> = pass_records.iter().map(|r| &r.config).collect();
    let fail_cfgs: Vec<&OpticalConfig> = fail_records.iter().map(|r| &r.config).collect();

    let pass_prop_mm: Vec<f64> = pass_cfgs.iter().map(|c| c.propagation_mm as f64).collect();
    let fail_prop_mm: Vec<f64> = fail_cfgs.iter().map(|c| c.propagation_mm as f64).collect();

    let pass_wave: Vec<f64> = pass_cfgs.iter().map(|c| c.wavelength_nm as f64).collect();
    let fail_wave: Vec<f64> = fail_cfgs.iter().map(|c| c.wavelength_nm as f64).collect();

    let pass_binning: Vec<f64> = pass_cfgs
        .iter()
        .map(|c| c.detector.binning as f64)
        .collect();
    let fail_binning: Vec<f64> = fail_cfgs
        .iter()
        .map(|c| c.detector.binning as f64)
        .collect();

    let pass_seed: Vec<f64> = pass_cfgs.iter().map(|c| c.seed as f64).collect();
    let fail_seed: Vec<f64> = fail_cfgs.iter().map(|c| c.seed as f64).collect();

    // Propagation mode as discrete code.
    let mode_code = |c: &OpticalConfig| {
        use photonlayer_core::prelude::PropagationMode;
        match c.propagation {
            PropagationMode::Fresnel => 0.0,
            PropagationMode::Fraunhofer => 1.0,
            PropagationMode::AngularSpectrum => 2.0,
        }
    };
    let pass_mode: Vec<f64> = pass_cfgs.iter().map(|c| mode_code(c)).collect();
    let fail_mode: Vec<f64> = fail_cfgs.iter().map(|c| mode_code(c)).collect();

    let candidates: &[(&str, f64)] = &[
        (
            "propagation_mm",
            discriminability(&pass_prop_mm, &fail_prop_mm),
        ),
        ("wavelength_nm", discriminability(&pass_wave, &fail_wave)),
        (
            "detector.binning",
            discriminability(&pass_binning, &fail_binning),
        ),
        ("seed", discriminability(&pass_seed, &fail_seed)),
        ("propagation_mode", discriminability(&pass_mode, &fail_mode)),
    ];

    let mut sorted = candidates.to_vec();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let dominant = sorted[0].0.to_owned();
    let secondary = if sorted.len() > 1 {
        sorted[1].0.to_owned()
    } else {
        "none".to_owned()
    };

    let pass_mean = mean_f64(&pass_cfgs, |c| c.propagation_mm as f64);
    let fail_mean = mean_f64(&fail_cfgs, |c| c.propagation_mm as f64);

    let recommendation = format!(
        "Dominant separator: `{dominant}` (score {:.3}). \
         Secondary: `{secondary}` (score {:.3}). \
         Pass propagation_mm mean={pass_mean:.2}, fail mean={fail_mean:.2}. \
         Spectral gap={spectral_gap:.4}.",
        sorted[0].1,
        sorted[1].1.max(0.0),
    );

    BoundaryReport {
        dominant_variable: dominant,
        secondary_variable: secondary,
        spectral_gap,
        recommendation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use photonlayer_core::prelude::{
        build_receipt, DetectorConfig, InputImage, MetricReport, OpticalConfig, OpticalSimulator,
        PhaseMask, PropagationMode, Provenance, ScalarSimulator,
    };

    use crate::{embedding::experiment_embedding, memory::ExperimentRecord};

    fn make_record_with_cfg(
        id: &str,
        label: &str,
        cfg: OpticalConfig,
        seed: u64,
    ) -> ExperimentRecord {
        let n = 16;
        let pixels: Vec<f32> = (0..n * n).map(|i| (i % n) as f32 / n as f32).collect();
        let img = InputImage::from_norm_f32(n, n, pixels).unwrap();
        let mask = PhaseMask::random(n, n, seed);
        let frame = ScalarSimulator.simulate(&img, &mask, &cfg).unwrap();
        let metrics = MetricReport::default();
        let receipt = build_receipt(
            id,
            &img,
            &mask,
            &cfg,
            &frame,
            &metrics,
            &Provenance::default(),
        );
        let embedding = experiment_embedding(&mask, &frame);
        ExperimentRecord {
            id: id.to_owned(),
            label: label.to_owned(),
            config: cfg,
            mask_id: mask.mask_id,
            embedding,
            receipt,
            metrics,
        }
    }

    #[test]
    fn boundary_identifies_propagation_mm() {
        let mut mem = ExperimentMemory::new();

        // Pass group: short propagation.
        for i in 0..3u64 {
            let mut cfg = OpticalConfig::demo(16, 16);
            cfg.propagation_mm = 5.0;
            mem.remember(make_record_with_cfg(
                &format!("pass-{i}"),
                "pass",
                cfg,
                i + 1,
            ));
        }
        // Fail group: long propagation.
        for i in 0..3u64 {
            let mut cfg = OpticalConfig::demo(16, 16);
            cfg.propagation_mm = 50.0;
            mem.remember(make_record_with_cfg(
                &format!("fail-{i}"),
                "fail",
                cfg,
                i + 10,
            ));
        }

        let report = explain_boundary(&mem, "pass", "fail");
        assert_eq!(
            report.dominant_variable, "propagation_mm",
            "expected propagation_mm, got: {:?}",
            report
        );
        assert!(report.spectral_gap >= 0.0);
    }

    #[test]
    fn boundary_identifies_propagation_mode() {
        let mut mem = ExperimentMemory::new();

        for i in 0..3u64 {
            let mut cfg = OpticalConfig::demo(16, 16);
            cfg.propagation = PropagationMode::Fresnel;
            mem.remember(make_record_with_cfg(
                &format!("pass-{i}"),
                "pass",
                cfg,
                i + 1,
            ));
        }
        for i in 0..3u64 {
            let mut cfg = OpticalConfig::demo(16, 16);
            cfg.propagation = PropagationMode::Fraunhofer;
            mem.remember(make_record_with_cfg(
                &format!("fail-{i}"),
                "fail",
                cfg,
                i + 10,
            ));
        }

        let report = explain_boundary(&mem, "pass", "fail");
        // Mode is the only thing we changed, so it must be dominant.
        assert_eq!(
            report.dominant_variable, "propagation_mode",
            "got: {:?}",
            report
        );
    }

    #[test]
    fn boundary_empty_returns_gracefully() {
        let mem = ExperimentMemory::new();
        let report = explain_boundary(&mem, "pass", "fail");
        assert_eq!(report.dominant_variable, "insufficient_data");
    }

    #[test]
    fn boundary_identifies_binning() {
        let mut mem = ExperimentMemory::new();

        for i in 0..3u64 {
            let mut cfg = OpticalConfig::demo(16, 16);
            cfg.detector = DetectorConfig {
                binning: 1,
                ..DetectorConfig::default()
            };
            mem.remember(make_record_with_cfg(
                &format!("pass-{i}"),
                "pass",
                cfg,
                i + 1,
            ));
        }
        for i in 0..3u64 {
            let mut cfg = OpticalConfig::demo(16, 16);
            cfg.detector = DetectorConfig {
                binning: 4,
                ..DetectorConfig::default()
            };
            mem.remember(make_record_with_cfg(
                &format!("fail-{i}"),
                "fail",
                cfg,
                i + 10,
            ));
        }

        let report = explain_boundary(&mem, "pass", "fail");
        assert_eq!(
            report.dominant_variable, "detector.binning",
            "got: {:?}",
            report
        );
    }
}
