//! Reconstruction-attack privacy analysis (ADR-260 §22).
//!
//! # Optical privacy framing (product lead)
//!
//! When an optical mask diffuses and encodes the input before any pixel hits
//! the sensor, the raw detector pattern is NOT a human-readable image. To
//! quantify this empirically we attempt a *reconstruction attack*: train a
//! linear inverse map W (least squares, ridge-regularized) from the compact
//! detector feature vector back to the downsampled input image, then measure
//! how faithfully it reconstructs a held-out test image (PSNR, dB).
//!
//! - **Identity mask** (no optical processing) → near-perfect reconstruction
//!   → HIGH PSNR → HIGH privacy leakage.
//! - **Random / learned optical mask** → scrambled detector pattern → POOR
//!   reconstruction → LOW PSNR → LOW privacy leakage.
//!
//! This module verifies that the optical front end is genuinely privacy-
//! preserving, not just by design assertion but by failed attack. The system
//! stores only the compact feature vector, never a normal face image.

use crate::pipeline::optical_features;
use crate::synthetic::Sample;
use photonlayer_core::config::OpticalConfig;
use photonlayer_core::mask::PhaseMask;
use photonlayer_core::metrics::{input_frame_similarity, psnr};
use photonlayer_core::simulator::{OpticalSimulator, ScalarSimulator};
use serde::{Deserialize, Serialize};

/// Summary of a reconstruction-attack privacy evaluation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrivacyReport {
    /// Reconstruction PSNR (dB) from the linear inverse map.
    /// Higher = more leakage; lower = better privacy.
    pub reconstruction_psnr: f32,
    /// Normalized leakage score in [0, 1].
    /// 0 = perfect privacy, 1 = full reconstruction.
    pub leakage_score: f32,
    /// Pearson r between detector pattern and input image (should be near 0
    /// for a good optical mask, confirming non-readability).
    pub frame_input_similarity: f32,
}

// ---------------------------------------------------------------------------
// Tiny hand-coded linear algebra (no external deps, feat_dim kept small).
// ---------------------------------------------------------------------------

/// Solve the ridge-regularized normal equations: (X'X + λI) w = X'y.
///
/// X: n_samples × n_features, y: n_samples × n_targets.
/// Returns w: n_features × n_targets (column-major as flat Vec<f32>).
///
/// We solve each target column independently using Gauss-Jordan elimination
/// on the (feat_dim × feat_dim) Gram matrix — safe for small feat_dim.
fn ridge_least_squares(
    x: &[f32], // n × d
    y: &[f32], // n × t
    n: usize,
    d: usize,
    t: usize,
    lambda: f32,
) -> Vec<f32> {
    // Build Gram matrix G = X'X + λI  (d × d).
    let mut g = vec![0.0f32; d * d];
    for i in 0..d {
        for j in 0..d {
            let mut s = 0.0f32;
            for k in 0..n {
                s += x[k * d + i] * x[k * d + j];
            }
            g[i * d + j] = s;
        }
        g[i * d + i] += lambda; // ridge regularization
    }

    // Build right-hand side R = X'Y  (d × t).
    let mut r = vec![0.0f32; d * t];
    for i in 0..d {
        for c in 0..t {
            let mut s = 0.0f32;
            for k in 0..n {
                s += x[k * d + i] * y[k * t + c];
            }
            r[i * t + c] = s;
        }
    }

    // Gauss-Jordan elimination on [G | R] augmented matrix (d × (d+t)).
    let aug_cols = d + t;
    let mut aug = vec![0.0f32; d * aug_cols];
    for i in 0..d {
        for j in 0..d {
            aug[i * aug_cols + j] = g[i * d + j];
        }
        for c in 0..t {
            aug[i * aug_cols + d + c] = r[i * t + c];
        }
    }

    for col in 0..d {
        // Find pivot.
        let mut pivot_row = col;
        let mut pivot_val = aug[col * aug_cols + col].abs();
        for row in (col + 1)..d {
            let v = aug[row * aug_cols + col].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = row;
            }
        }
        if pivot_val < 1e-12 {
            continue; // nearly-singular row — skip
        }
        aug.swap_with_slice_at(col, pivot_row, aug_cols);
        let inv = 1.0 / aug[col * aug_cols + col];
        for j in 0..aug_cols {
            aug[col * aug_cols + j] *= inv;
        }
        for row in 0..d {
            if row == col {
                continue;
            }
            let factor = aug[row * aug_cols + col];
            for j in 0..aug_cols {
                let v = aug[col * aug_cols + j] * factor;
                aug[row * aug_cols + j] -= v;
            }
        }
    }

    // Extract solution W (d × t).
    let mut w = vec![0.0f32; d * t];
    for i in 0..d {
        for c in 0..t {
            w[i * t + c] = aug[i * aug_cols + d + c];
        }
    }
    w
}

/// Swap two rows of a matrix stored as a flat Vec<f32> given `cols` columns.
trait SwapRows {
    fn swap_with_slice_at(&mut self, row_a: usize, row_b: usize, cols: usize);
}

impl SwapRows for Vec<f32> {
    fn swap_with_slice_at(&mut self, row_a: usize, row_b: usize, cols: usize) {
        if row_a == row_b {
            return;
        }
        let (lo, hi) = if row_a < row_b {
            (row_a, row_b)
        } else {
            (row_b, row_a)
        };
        let (left, right) = self.split_at_mut(hi * cols);
        left[lo * cols..lo * cols + cols].swap_with_slice(&mut right[..cols]);
    }
}

// ---------------------------------------------------------------------------

/// Downsample `src` (w_src × h_src) to `out_dim × out_dim` via average-pool.
fn downsample(src: &[f32], w_src: usize, h_src: usize, out_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_dim * out_dim];
    for oy in 0..out_dim {
        for ox in 0..out_dim {
            let x0 = ox * w_src / out_dim;
            let x1 = ((ox + 1) * w_src / out_dim).max(x0 + 1).min(w_src);
            let y0 = oy * h_src / out_dim;
            let y1 = ((oy + 1) * h_src / out_dim).max(y0 + 1).min(h_src);
            let mut acc = 0.0f32;
            let mut cnt = 0.0f32;
            for y in y0..y1 {
                for x in x0..x1 {
                    acc += src[y * w_src + x];
                    cnt += 1.0;
                }
            }
            out[oy * out_dim + ox] = if cnt > 0.0 { acc / cnt } else { 0.0 };
        }
    }
    out
}

/// Attempt a linear reconstruction attack and return a `PrivacyReport`.
///
/// Uses the first 60% of `samples` as the attack training set, the remaining
/// 40% as the test set. The attack learns a linear map W: features → image
/// pixels using ridge regression on the training split, then measures PSNR on
/// the test split (peak = 1.0). Higher PSNR = more leakage.
///
/// `feat_dim` is the pool-grid side length for feature extraction (kept small,
/// e.g. 4 or 6, so the Gram matrix is cheap).
pub fn privacy_leakage(
    samples: &[Sample],
    mask: &PhaseMask,
    config: &OpticalConfig,
    feat_dim: usize,
) -> PrivacyReport {
    assert!(!samples.is_empty(), "need at least one sample");

    // Feature dimension (d) and target image dimension (t = img_dim^2).
    let d = feat_dim * feat_dim;
    // Downsample targets to the same size as the feature grid.
    let img_dim = feat_dim;
    let t = img_dim * img_dim;

    // Train/test split: first 60% train.
    let n_train = ((samples.len() as f32 * 0.6).ceil() as usize)
        .max(1)
        .min(samples.len() - 1);
    let n_test = samples.len() - n_train;
    if n_test == 0 {
        // Degenerate case: no test data.
        return PrivacyReport {
            reconstruction_psnr: 0.0,
            leakage_score: 0.0,
            frame_input_similarity: 0.0,
        };
    }

    let train = &samples[..n_train];
    let test = &samples[n_train..];

    // Compute feature vectors and target images for training.
    let mut x_train = vec![0.0f32; n_train * d]; // n × d
    let mut y_train = vec![0.0f32; n_train * t]; // n × t

    for (k, s) in train.iter().enumerate() {
        let feat = optical_features(s, mask, config, feat_dim);
        for (j, &v) in feat.iter().enumerate() {
            x_train[k * d + j] = v;
        }
        let target = downsample(&s.image.pixels, s.image.width, s.image.height, img_dim);
        for (j, &v) in target.iter().enumerate() {
            y_train[k * t + j] = v;
        }
    }

    // Ridge regularization: λ = 1e-3 keeps the small Gram matrix well-posed.
    let lambda = 1e-3_f32;
    let w = ridge_least_squares(&x_train, &y_train, n_train, d, t, lambda);

    // Evaluate on test split: apply W, measure PSNR.
    let mut total_psnr = 0.0f32;
    let mut total_sim = 0.0f32;

    for s in test {
        let feat = optical_features(s, mask, config, feat_dim);

        // Reconstructed image: y_hat = x @ W  (1 × d) * (d × t) = (1 × t).
        let mut y_hat = vec![0.0f32; t];
        for c in 0..t {
            let mut v = 0.0f32;
            for i in 0..d {
                v += feat[i] * w[i * t + c];
            }
            y_hat[c] = v.clamp(0.0, 1.0);
        }

        let target = downsample(&s.image.pixels, s.image.width, s.image.height, img_dim);
        total_psnr += psnr(&target, &y_hat, 1.0);

        // Also accumulate detector-vs-input similarity.
        let frame = ScalarSimulator
            .simulate(&s.image, mask, config)
            .expect("simulation");
        total_sim += input_frame_similarity(&s.image, &frame);
    }

    let mean_psnr = total_psnr / n_test as f32;
    let frame_input_similarity = (total_sim / n_test as f32).abs();

    // Map PSNR to a [0,1] leakage score:
    //   PSNR = 0 dB → leakage ≈ 0.0 (no recoverable signal)
    //   PSNR ≥ 30 dB → leakage ≈ 1.0 (near-perfect reconstruction)
    // Using a sigmoid-style cap at 30 dB.
    let leakage_score = if mean_psnr <= 0.0 {
        0.0
    } else {
        (mean_psnr / 30.0).min(1.0)
    };

    PrivacyReport {
        reconstruction_psnr: mean_psnr,
        leakage_score,
        frame_input_similarity,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic::make_dataset;

    #[test]
    fn optical_mask_lower_reconstruction_psnr_than_identity() {
        // The optical front end scrambles the image: a linear attack must fail.
        // Identity mask = no optical processing → detector frame ≈ input → high PSNR.
        // Random optical mask → detector scrambled → low PSNR.
        let n = 16;
        let samples = make_dataset(n, 8, 0xCAFE);
        let cfg = OpticalConfig::demo(n, n);
        let feat_dim = 4; // keep Gram matrix tiny (4×4 = 16×16 solve)

        let identity_mask = PhaseMask::identity(n, n);
        let random_mask = PhaseMask::random(n, n, 0xDEAD);

        let id_report = privacy_leakage(&samples, &identity_mask, &cfg, feat_dim);
        let rnd_report = privacy_leakage(&samples, &random_mask, &cfg, feat_dim);

        // Optical mask should yield lower reconstruction PSNR than identity.
        assert!(
            rnd_report.reconstruction_psnr < id_report.reconstruction_psnr,
            "random mask PSNR {:.1} dB should be < identity mask PSNR {:.1} dB",
            rnd_report.reconstruction_psnr,
            id_report.reconstruction_psnr,
        );
    }

    #[test]
    fn optical_mask_low_frame_input_similarity() {
        // The detector pattern of an optical mask should not look like the input.
        let n = 16;
        let samples = make_dataset(n, 4, 0xBEEF);
        let cfg = OpticalConfig::demo(n, n);
        let feat_dim = 4;
        let mask = PhaseMask::random(n, n, 77);
        let report = privacy_leakage(&samples, &mask, &cfg, feat_dim);
        // Pearson |r| < 0.5 for a scrambled frame (non-human-readable).
        assert!(
            report.frame_input_similarity < 0.5,
            "frame_input_similarity {:.3} should be low for optical mask",
            report.frame_input_similarity,
        );
    }
}
