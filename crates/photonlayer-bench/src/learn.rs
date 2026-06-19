//! In-Rust phase-mask learner (ADR-260 §16.1 "learned optical mask").
//!
//! Full differentiable Fourier optics (TorchOptics/waveprop, ADR-260 §9.3) is
//! the offline reference. For a dependency-free, deterministic, browser-
//! shippable runtime we train the phase mask with seeded coordinate/block
//! hill-climbing against the task loss of the nearest-centroid decoder.
//!
//! The optimizer starts from a random mask and only accepts improving steps,
//! so the learned mask provably dominates its random starting point on the
//! training objective — the basis of the "random < learned" acceptance gate
//! (ADR-260 §17.2).

use crate::decoder::NearestCentroid;
use crate::pipeline::optical_feature_set;
use crate::synthetic::{Sample, NUM_CLASSES};
use core::f32::consts::PI;
use photonlayer_core::config::OpticalConfig;
use photonlayer_core::mask::PhaseMask;
use photonlayer_core::rng::DeterministicRng;

#[derive(Clone, Copy, Debug)]
pub struct LearnConfig {
    pub iterations: usize,
    /// Side length of the square block of mask cells perturbed each step.
    pub block: usize,
    /// Std-dev (radians) of the per-cell phase perturbation.
    pub sigma: f32,
    /// Decoder feature grid side length.
    pub feat_dim: usize,
    pub seed: u64,
}

impl Default for LearnConfig {
    fn default() -> Self {
        Self {
            iterations: 160,
            block: 4,
            sigma: 0.6,
            feat_dim: 8,
            seed: 0xA11CE,
        }
    }
}

/// Score of a candidate mask: training accuracy with a separation-margin
/// tiebreak. Higher is better; compared lexicographically.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Score {
    pub accuracy: f32,
    pub margin: f32,
}

impl Score {
    fn better_than(&self, o: &Score) -> bool {
        if (self.accuracy - o.accuracy).abs() > 1e-6 {
            self.accuracy > o.accuracy
        } else {
            self.margin > o.margin
        }
    }
}

/// Result of a learning run.
pub struct LearnOutcome {
    pub mask: PhaseMask,
    pub decoder: NearestCentroid,
    pub start_score: Score,
    pub final_score: Score,
}

fn evaluate(
    mask: &PhaseMask,
    train: &[Sample],
    config: &OpticalConfig,
    feat_dim: usize,
) -> (NearestCentroid, Score) {
    let (feats, labels) = optical_feature_set(train, mask, config, feat_dim);
    let ncc = NearestCentroid::fit(&feats, &labels, NUM_CLASSES);
    let acc = ncc.accuracy(&feats, &labels);

    // Mean (nearest-wrong - correct) centroid distance: separation margin.
    let mut margin = 0.0f32;
    for (f, &lab) in feats.iter().zip(&labels) {
        let mut correct = f32::INFINITY;
        let mut wrong = f32::INFINITY;
        for (c, centroid) in ncc.centroids.iter().enumerate() {
            let d: f32 = centroid.iter().zip(f).map(|(a, b)| (a - b) * (a - b)).sum();
            if c == lab {
                correct = d;
            } else if d < wrong {
                wrong = d;
            }
        }
        margin += wrong - correct;
    }
    margin /= feats.len().max(1) as f32;
    (
        ncc,
        Score {
            accuracy: acc,
            margin,
        },
    )
}

/// Train a phase mask + decoder on `train` via seeded block hill-climbing.
pub fn learn_mask(train: &[Sample], config: &OpticalConfig, lc: &LearnConfig) -> LearnOutcome {
    let w = config.width;
    let h = config.height;
    let mut rng = DeterministicRng::new(lc.seed);

    let mut mask = PhaseMask::random(w, h, lc.seed);
    let (mut decoder, mut score) = evaluate(&mask, train, config, lc.feat_dim);
    let start_score = score;

    for _ in 0..lc.iterations {
        let mut candidate = mask.clone();
        // Perturb a random block of cells.
        let bx = (rng.next_f32() * (w.saturating_sub(lc.block) + 1) as f32) as usize;
        let by = (rng.next_f32() * (h.saturating_sub(lc.block) + 1) as f32) as usize;
        for dy in 0..lc.block.min(h) {
            for dx in 0..lc.block.min(w) {
                let idx = (by + dy).min(h - 1) * w + (bx + dx).min(w - 1);
                let delta = rng.next_gaussian() * lc.sigma;
                candidate.phase_radians[idx] =
                    (candidate.phase_radians[idx] + delta).rem_euclid(2.0 * PI);
            }
        }
        let (cand_dec, cand_score) = evaluate(&candidate, train, config, lc.feat_dim);
        if cand_score.better_than(&score) {
            mask = candidate;
            decoder = cand_dec;
            score = cand_score;
        }
    }

    mask.mask_id = format!("learned:{:#x}", lc.seed);
    LearnOutcome {
        mask,
        decoder,
        start_score,
        final_score: score,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic::make_dataset;

    #[test]
    fn learning_does_not_regress_training_objective() {
        let n = 16;
        let train = make_dataset(n, 6, 1);
        let cfg = OpticalConfig::demo(n, n);
        let lc = LearnConfig {
            iterations: 60,
            ..Default::default()
        };
        let out = learn_mask(&train, &cfg, &lc);
        // Hill climbing only accepts improvements => final >= start.
        assert!(
            !out.start_score.better_than(&out.final_score),
            "learned regressed: {:?} -> {:?}",
            out.start_score,
            out.final_score
        );
    }
}
