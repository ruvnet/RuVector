//! Sample -> feature pipelines for the three benchmark variants (ADR-260 §16.1).

use crate::decoder::{frame_features, pool_features};
use crate::synthetic::Sample;
use photonlayer_core::config::OpticalConfig;
use photonlayer_core::mask::PhaseMask;
use photonlayer_core::simulator::{OpticalSimulator, ScalarSimulator};

/// Optical feature: image -> field -> mask -> propagate -> detector -> pool.
pub fn optical_features(
    sample: &Sample,
    mask: &PhaseMask,
    config: &OpticalConfig,
    feat_dim: usize,
) -> Vec<f32> {
    let frame = ScalarSimulator
        .simulate(&sample.image, mask, config)
        .expect("simulation");
    frame_features(&frame, feat_dim)
}

/// Digital baseline feature: pooled raw image, no optics at all.
pub fn digital_features(sample: &Sample, feat_dim: usize) -> Vec<f32> {
    pool_features(
        &sample.image.pixels,
        sample.image.width,
        sample.image.height,
        feat_dim,
    )
}

/// Batch helpers.
pub fn optical_feature_set(
    samples: &[Sample],
    mask: &PhaseMask,
    config: &OpticalConfig,
    feat_dim: usize,
) -> (Vec<Vec<f32>>, Vec<usize>) {
    let feats = samples
        .iter()
        .map(|s| optical_features(s, mask, config, feat_dim))
        .collect();
    let labels = samples.iter().map(|s| s.label).collect();
    (feats, labels)
}

pub fn digital_feature_set(samples: &[Sample], feat_dim: usize) -> (Vec<Vec<f32>>, Vec<usize>) {
    let feats = samples
        .iter()
        .map(|s| digital_features(s, feat_dim))
        .collect();
    let labels = samples.iter().map(|s| s.label).collect();
    (feats, labels)
}
