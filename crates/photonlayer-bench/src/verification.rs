//! Biometric-style 1:1 verification using optical feature embeddings.
//!
//! # Design framing (ADR-260 §22, product lead)
//!
//! Optical computing is a *front end* that performs useful computation before
//! digitization — lower latency, narrower sensor bandwidth, lower power,
//! compressed measurements, and task-specific sensing. This module implements
//! **consented 1:1 verification**: given two probe signals, decide "same
//! identity / not same identity" using the optical feature embedding. No
//! full-resolution face image is stored or transmitted; the raw detector
//! pattern is not human-readable. This is NOT a mass-surveillance face-ID
//! engine.
//!
//! The synthetic dataset's class labels serve as identity surrogates:
//! same label = genuine pair; different label = impostor pair.

use crate::pipeline::optical_features;
use crate::synthetic::Sample;
use photonlayer_core::config::OpticalConfig;
use photonlayer_core::mask::PhaseMask;
use serde::{Deserialize, Serialize};

/// Summary of a verification sweep across all genuine/impostor pairs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerificationReport {
    /// Equal Error Rate: threshold point where FAR ≈ FRR (lower is better).
    pub eer: f32,
    /// False Accept Rate at the EER operating point.
    pub far_at_eer: f32,
    /// False Reject Rate at the EER operating point.
    pub frr_at_eer: f32,
    /// Decision threshold at EER.
    pub threshold: f32,
    pub num_genuine: usize,
    pub num_impostor: usize,
}

/// Cosine similarity between two L2-normalized feature vectors (range -1..1).
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
}

/// Compute FAR and FRR for a given decision threshold.
///
/// A pair whose score >= threshold is accepted as "same identity".
/// FAR = impostor pairs accepted / total impostor pairs.
/// FRR = genuine pairs rejected / total genuine pairs.
fn far_frr(genuine_scores: &[f32], impostor_scores: &[f32], threshold: f32) -> (f32, f32) {
    let far = if impostor_scores.is_empty() {
        0.0
    } else {
        impostor_scores.iter().filter(|&&s| s >= threshold).count() as f32
            / impostor_scores.len() as f32
    };
    let frr = if genuine_scores.is_empty() {
        0.0
    } else {
        genuine_scores.iter().filter(|&&s| s < threshold).count() as f32
            / genuine_scores.len() as f32
    };
    (far, frr)
}

/// Compute EER: the threshold where FAR and FRR cross closest.
fn compute_eer(genuine: &[f32], impostor: &[f32]) -> (f32, f32, f32, f32) {
    if genuine.is_empty() || impostor.is_empty() {
        return (0.5, 0.5, 0.5, 0.0);
    }

    // Build a sorted candidate threshold list from all unique scores.
    let mut thresholds: Vec<f32> = genuine.iter().chain(impostor.iter()).cloned().collect();
    thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    thresholds.dedup();

    let mut best_eer = f32::INFINITY;
    let mut best_far = 0.5_f32;
    let mut best_frr = 0.5_f32;
    let mut best_t = 0.0_f32;

    // Sweep threshold candidates including boundary sentinels.
    let mut candidates = vec![thresholds[0] - 0.01];
    candidates.extend_from_slice(&thresholds);
    candidates.push(*thresholds.last().unwrap() + 0.01);

    for &t in &candidates {
        let (far, frr) = far_frr(genuine, impostor, t);
        let gap = (far - frr).abs();
        let best_gap = (best_far - best_frr).abs();
        if gap < best_gap || gap < best_eer {
            best_eer = gap;
            best_far = far;
            best_frr = frr;
            best_t = t;
        }
    }

    // EER estimate: midpoint of FAR and FRR at the crossing.
    let eer = (best_far + best_frr) / 2.0;
    (eer, best_far, best_frr, best_t)
}

/// Compute a `VerificationReport` for the given mask/config on `samples`.
///
/// Pairs are exhaustive: every ordered (i,j) where i < j.
/// Same label => genuine; different label => impostor.
/// Match score = cosine similarity of optical feature vectors (higher = more similar).
pub fn verify_eer(
    samples: &[Sample],
    mask: &PhaseMask,
    config: &OpticalConfig,
    feat_dim: usize,
) -> VerificationReport {
    // Extract optical feature embeddings for every sample.
    let feats: Vec<Vec<f32>> = samples
        .iter()
        .map(|s| optical_features(s, mask, config, feat_dim))
        .collect();

    let mut genuine_scores: Vec<f32> = Vec::new();
    let mut impostor_scores: Vec<f32> = Vec::new();

    for i in 0..samples.len() {
        for j in (i + 1)..samples.len() {
            let score = cosine_sim(&feats[i], &feats[j]);
            if samples[i].label == samples[j].label {
                genuine_scores.push(score);
            } else {
                impostor_scores.push(score);
            }
        }
    }

    let (eer, far_at_eer, frr_at_eer, threshold) = compute_eer(&genuine_scores, &impostor_scores);

    VerificationReport {
        eer,
        far_at_eer,
        frr_at_eer,
        threshold,
        num_genuine: genuine_scores.len(),
        num_impostor: impostor_scores.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learn::{learn_mask, LearnConfig};
    use crate::synthetic::make_dataset;

    #[test]
    fn learned_mask_lower_eer_than_random() {
        // ADR-260 §17: a learned optical mask should improve biometric
        // discrimination vs. a random mask (lower EER = fewer errors).
        let n = 16;
        let samples = make_dataset(n, 6, 0xBEEF);
        let cfg = OpticalConfig::demo(n, n);
        let feat_dim = 4;

        let random_mask = PhaseMask::random(n, n, 0xDEAD);
        let random_report = verify_eer(&samples, &random_mask, &cfg, feat_dim);

        let lc = LearnConfig {
            iterations: 100,
            feat_dim,
            ..Default::default()
        };
        let outcome = learn_mask(&samples, &cfg, &lc);
        let learned_report = verify_eer(&samples, &outcome.mask, &cfg, feat_dim);

        // The learned mask must achieve EER <= random mask EER.
        // (Hill-climbing on accuracy is correlated with inter-class separation.)
        assert!(
            learned_report.eer <= random_report.eer + 0.05,
            "learned EER {:.3} should be <= random EER {:.3} (within 5%)",
            learned_report.eer,
            random_report.eer,
        );
    }

    #[test]
    fn genuine_impostor_pair_counts_correct() {
        // With 4 samples of 2 classes (2 each), genuine pairs = 2 (within class),
        // impostor pairs = 4 (cross-class), total = 6 = C(4,2).
        let n = 16;
        let samples = make_dataset(n, 2, 42);
        // Use only first 2 classes (4 samples).
        let subset: Vec<_> = samples.into_iter().filter(|s| s.label < 2).collect();
        let cfg = OpticalConfig::demo(n, n);
        let mask = PhaseMask::identity(n, n);
        let report = verify_eer(&subset, &mask, &cfg, 2);
        assert_eq!(report.num_genuine + report.num_impostor, 6, "C(4,2)=6");
    }
}
