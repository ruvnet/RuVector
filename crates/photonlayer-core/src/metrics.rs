//! Benchmark metrics (ADR-260 §16.2) and embedding helpers (§12).

use crate::detector::OpticalFrame;
use crate::field::InputImage;
use crate::hash::hash_f32;
use serde::{Deserialize, Serialize};

/// Mean-squared error between two equally-sized slices.
pub fn mse(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return f32::INFINITY;
    }
    let s: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    s / a.len() as f32
}

/// Peak signal-to-noise ratio in dB, assuming a peak value of `peak`.
pub fn psnr(a: &[f32], b: &[f32], peak: f32) -> f32 {
    let m = mse(a, b);
    if m <= 0.0 {
        return f32::INFINITY;
    }
    10.0 * (peak * peak / m).log10()
}

/// Sensor compression ratio: input pixels / sensor pixels (ADR-260 §16.2).
pub fn compression_ratio(input: &InputImage, frame: &OpticalFrame) -> f32 {
    let sensor = (frame.width * frame.height).max(1);
    (input.width * input.height) as f32 / sensor as f32
}

/// Classification accuracy over predicted vs. true labels.
pub fn accuracy(predicted: &[usize], truth: &[usize]) -> f32 {
    if predicted.is_empty() || predicted.len() != truth.len() {
        return 0.0;
    }
    let correct = predicted.iter().zip(truth).filter(|(p, t)| p == t).count();
    correct as f32 / predicted.len() as f32
}

/// Normalized cross-correlation between an input image and a detector frame,
/// used to demonstrate that the encoded frame is **not** human-readable
/// (ADR-260 acceptance §17.3): a low score means the sensor pattern does not
/// look like the input. Both are resampled to the frame grid by nearest pixel.
pub fn input_frame_similarity(input: &InputImage, frame: &OpticalFrame) -> f32 {
    let n = frame.width * frame.height;
    if n == 0 {
        return 0.0;
    }
    let mut a = vec![0.0f32; n];
    for fy in 0..frame.height {
        for fx in 0..frame.width {
            let ix = fx * input.width / frame.width;
            let iy = fy * input.height / frame.height;
            a[fy * frame.width + fx] =
                input.pixels[iy.min(input.height - 1) * input.width + ix.min(input.width - 1)];
        }
    }
    pearson(&a, &frame.intensity)
}

fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let ma = a.iter().sum::<f32>() / n;
    let mb = b.iter().sum::<f32>() / n;
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for (x, y) in a.iter().zip(b) {
        let dx = x - ma;
        let dy = y - mb;
        cov += dx * dy;
        va += dx * dx;
        vb += dy * dy;
    }
    let denom = (va * vb).sqrt();
    if denom <= 1e-12 {
        0.0
    } else {
        cov / denom
    }
}

/// A radial intensity-spectrum embedding of a detector frame (ADR-260 §12).
/// Produces a fixed-length vector describing how energy is distributed by
/// distance from the frame center — a compact, comparable signature.
pub fn frame_spectrum_embedding(frame: &OpticalFrame, bins: usize) -> Vec<f32> {
    let bins = bins.max(1);
    let mut acc = vec![0.0f32; bins];
    let mut cnt = vec![0.0f32; bins];
    let cx = frame.width as f32 / 2.0;
    let cy = frame.height as f32 / 2.0;
    let max_r = (cx * cx + cy * cy).sqrt().max(1e-6);
    for y in 0..frame.height {
        for x in 0..frame.width {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let r = (dx * dx + dy * dy).sqrt() / max_r;
            let mut b = (r * bins as f32) as usize;
            if b >= bins {
                b = bins - 1;
            }
            acc[b] += frame.intensity[y * frame.width + x];
            cnt[b] += 1.0;
        }
    }
    for i in 0..bins {
        if cnt[i] > 0.0 {
            acc[i] /= cnt[i];
        }
    }
    let total: f32 = acc.iter().sum();
    if total > 0.0 {
        for a in &mut acc {
            *a /= total;
        }
    }
    acc
}

/// A bundle of benchmark metrics for one experiment, hashable for receipts.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MetricReport {
    pub accuracy: f32,
    pub reconstruction_mse: f32,
    pub compression_ratio: f32,
    pub input_frame_similarity: f32,
    pub native_latency_us: f64,
}

impl MetricReport {
    /// Deterministic digest of the metric vector for the receipt (§15).
    pub fn metrics_hash(&self) -> String {
        let v = [
            self.accuracy,
            self.reconstruction_mse,
            self.compression_ratio,
            self.input_frame_similarity,
            self.native_latency_us as f32,
        ];
        hash_f32("photonlayer.metrics.v1", &[v.len()], &v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn psnr_identical_is_infinite() {
        let a = vec![0.1, 0.2, 0.3];
        assert!(psnr(&a, &a, 1.0).is_infinite());
    }

    #[test]
    fn accuracy_basic() {
        assert!((accuracy(&[1, 2, 3], &[1, 0, 3]) - 2.0 / 3.0).abs() < 1e-6);
    }
}
