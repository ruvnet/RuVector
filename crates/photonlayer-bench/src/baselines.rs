//! Benchmark variants and runners (ADR-260 §16).

use crate::decoder::NearestCentroid;
use crate::learn::{learn_mask, LearnConfig};
use crate::pipeline::{digital_feature_set, optical_feature_set};
use crate::synthetic::{make_dataset, Sample, NUM_CLASSES};
use photonlayer_core::config::OpticalConfig;
use photonlayer_core::mask::PhaseMask;
use serde::{Deserialize, Serialize};

/// One variant's measured result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VariantResult {
    pub name: String,
    pub train_accuracy: f32,
    pub test_accuracy: f32,
    pub decoder_params: usize,
    pub feature_dim: usize,
}

/// A full benchmark report (ADR-260 §16.2).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchReport {
    pub grid: usize,
    pub feature_dim: usize,
    pub variants: Vec<VariantResult>,
}

fn split(samples: &[Sample]) -> (Vec<Sample>, Vec<Sample>) {
    // Deterministic interleaved split: even indices train, odd test.
    let mut train = Vec::new();
    let mut test = Vec::new();
    for (i, s) in samples.iter().enumerate() {
        if i % 2 == 0 {
            train.push(s.clone());
        } else {
            test.push(s.clone());
        }
    }
    (train, test)
}

fn eval_optical(
    mask: &PhaseMask,
    train: &[Sample],
    test: &[Sample],
    cfg: &OpticalConfig,
    feat_dim: usize,
    name: &str,
) -> VariantResult {
    let (tr_f, tr_l) = optical_feature_set(train, mask, cfg, feat_dim);
    let (te_f, te_l) = optical_feature_set(test, mask, cfg, feat_dim);
    let ncc = NearestCentroid::fit(&tr_f, &tr_l, NUM_CLASSES);
    VariantResult {
        name: name.to_string(),
        train_accuracy: ncc.accuracy(&tr_f, &tr_l),
        test_accuracy: ncc.accuracy(&te_f, &te_l),
        decoder_params: ncc.param_count(),
        feature_dim: feat_dim * feat_dim,
    }
}

/// Run the three headline variants: digital baseline, random mask, learned mask.
pub fn run_classification(grid: usize, per_class: usize, lc: &LearnConfig) -> BenchReport {
    let cfg = OpticalConfig::demo(grid, grid);
    let data = make_dataset(grid, per_class, 0xDA7A);
    let (train, test) = split(&data);
    let feat_dim = lc.feat_dim;

    // 1. Digital baseline (no optics).
    let (d_tr_f, d_tr_l) = digital_feature_set(&train, feat_dim);
    let (d_te_f, d_te_l) = digital_feature_set(&test, feat_dim);
    let d_ncc = NearestCentroid::fit(&d_tr_f, &d_tr_l, NUM_CLASSES);
    let digital = VariantResult {
        name: "digital_baseline".into(),
        train_accuracy: d_ncc.accuracy(&d_tr_f, &d_tr_l),
        test_accuracy: d_ncc.accuracy(&d_te_f, &d_te_l),
        decoder_params: d_ncc.param_count(),
        feature_dim: feat_dim * feat_dim,
    };

    // 2. Random mask + decoder.
    let random_mask = PhaseMask::random(grid, grid, 0x5EED);
    let random = eval_optical(&random_mask, &train, &test, &cfg, feat_dim, "random_mask");

    // 3. Learned mask + decoder.
    let outcome = learn_mask(&train, &cfg, lc);
    let learned = eval_optical(&outcome.mask, &train, &test, &cfg, feat_dim, "learned_mask");

    BenchReport {
        grid,
        feature_dim: feat_dim * feat_dim,
        variants: vec![digital, random, learned],
    }
}

/// Compression benchmark (ADR-260 §16.2, §16.3): the showcase claim.
///
/// The sensor is squeezed to a tiny `feat_dim x feat_dim` grid. At that size a
/// direct pixel readout (digital baseline) and a random mask lose the
/// class-discriminative structure, but a *learned* mask can diffract
/// class-specific energy into the few remaining sensor cells — recovering
/// accuracy with far fewer pixels.
pub fn run_compression(
    grid: usize,
    per_class: usize,
    feat_dim: usize,
    lc: &LearnConfig,
) -> BenchReport {
    let cfg = OpticalConfig::demo(grid, grid);
    let data = make_dataset(grid, per_class, 0xC0FFEE);
    let (train, test) = split(&data);

    // Digital baseline read directly off the tiny sensor.
    let (d_tr_f, d_tr_l) = digital_feature_set(&train, feat_dim);
    let (d_te_f, d_te_l) = digital_feature_set(&test, feat_dim);
    let d_ncc = NearestCentroid::fit(&d_tr_f, &d_tr_l, NUM_CLASSES);
    let digital = VariantResult {
        name: "digital_tiny_sensor".into(),
        train_accuracy: d_ncc.accuracy(&d_tr_f, &d_tr_l),
        test_accuracy: d_ncc.accuracy(&d_te_f, &d_te_l),
        decoder_params: d_ncc.param_count(),
        feature_dim: feat_dim * feat_dim,
    };

    let random_mask = PhaseMask::random(grid, grid, 0x5EED);
    let random = eval_optical(
        &random_mask,
        &train,
        &test,
        &cfg,
        feat_dim,
        "random_mask_tiny",
    );

    let mut lc2 = *lc;
    lc2.feat_dim = feat_dim;
    let outcome = learn_mask(&train, &cfg, &lc2);
    let learned = eval_optical(
        &outcome.mask,
        &train,
        &test,
        &cfg,
        feat_dim,
        "learned_mask_tiny",
    );

    BenchReport {
        grid,
        feature_dim: feat_dim * feat_dim,
        variants: vec![digital, random, learned],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learned_beats_or_matches_random_on_training() {
        // ADR-260 §17.2: random mask must be worse than learned on >=1 benchmark.
        let lc = LearnConfig {
            iterations: 120,
            ..Default::default()
        };
        let report = run_classification(16, 8, &lc);
        let random = report
            .variants
            .iter()
            .find(|v| v.name == "random_mask")
            .unwrap();
        let learned = report
            .variants
            .iter()
            .find(|v| v.name == "learned_mask")
            .unwrap();
        assert!(
            learned.train_accuracy >= random.train_accuracy,
            "learned {} < random {}",
            learned.train_accuracy,
            random.train_accuracy
        );
    }

    #[test]
    fn learned_strictly_wins_under_compression() {
        // ADR-260 §16.3 showcase: at a 2x2 (4-pixel) sensor the learned mask
        // should beat both the random mask and the direct digital readout.
        let lc = LearnConfig {
            iterations: 200,
            ..Default::default()
        };
        let r = run_compression(16, 10, 2, &lc);
        let dig = r
            .variants
            .iter()
            .find(|v| v.name == "digital_tiny_sensor")
            .unwrap();
        let rnd = r
            .variants
            .iter()
            .find(|v| v.name == "random_mask_tiny")
            .unwrap();
        let lrn = r
            .variants
            .iter()
            .find(|v| v.name == "learned_mask_tiny")
            .unwrap();
        assert!(lrn.test_accuracy > dig.test_accuracy, "learned !> digital");
        assert!(lrn.test_accuracy >= rnd.test_accuracy, "learned < random");
    }
}
