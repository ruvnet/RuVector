//! Detector / sensor model (ADR-260 §9.4).
//!
//! Converts a complex field to a real intensity frame and applies optional,
//! deterministic sensor effects: spatial binning, shot noise, Gaussian read
//! noise, saturation, and quantization. All randomness is seeded so that the
//! determinism invariant (§21) holds even with noise enabled.

use crate::config::{DetectorConfig, OpticalConfig};
use crate::field::OpticalField;
use crate::hash::hash_f32;
use crate::rng::DeterministicRng;

/// An intensity-only sensor capture.
#[derive(Clone, Debug)]
pub struct OpticalFrame {
    pub width: usize,
    pub height: usize,
    /// Row-major non-negative intensities.
    pub intensity: Vec<f32>,
    /// BLAKE3 digest binding dimensions + intensity values.
    pub frame_hash: String,
}

impl OpticalFrame {
    fn finalize(width: usize, height: usize, intensity: Vec<f32>) -> Self {
        let frame_hash = hash_f32("photonlayer.frame.v1", &[width, height], &intensity);
        Self {
            width,
            height,
            intensity,
            frame_hash,
        }
    }
}

/// Average-pool `intensity` (size `w*h`) by an integer `binning` factor.
fn bin(intensity: &[f32], w: usize, h: usize, binning: usize) -> (Vec<f32>, usize, usize) {
    if binning <= 1 {
        return (intensity.to_vec(), w, h);
    }
    let ow = w / binning;
    let oh = h / binning;
    let mut out = vec![0.0f32; ow * oh];
    let area = (binning * binning) as f32;
    for oy in 0..oh {
        for ox in 0..ow {
            let mut acc = 0.0f32;
            for dy in 0..binning {
                for dx in 0..binning {
                    let sx = ox * binning + dx;
                    let sy = oy * binning + dy;
                    acc += intensity[sy * w + sx];
                }
            }
            out[oy * ow + ox] = acc / area;
        }
    }
    (out, ow, oh)
}

/// Run the sensor pipeline on a propagated field.
pub fn capture(field: &OpticalField, config: &OpticalConfig) -> OpticalFrame {
    capture_with(field, &config.detector, config.seed)
}

/// Sensor pipeline with an explicit detector config and seed.
pub fn capture_with(field: &OpticalField, det: &DetectorConfig, seed: u64) -> OpticalFrame {
    // 1. Intensity.
    let raw: Vec<f32> = field.data.iter().map(|c| c.norm_sqr()).collect();

    // 2. Spatial binning (sensor integrates over the larger pixel).
    let (mut intensity, w, h) = bin(&raw, field.width, field.height, det.binning.max(1));

    // 3 & 4. Noise (seeded -> deterministic).
    if det.shot_noise_photons > 0.0 || det.read_noise_std > 0.0 {
        let mut rng = DeterministicRng::new(seed ^ 0x5EED_0000_0000_0001);
        for v in &mut intensity {
            if det.shot_noise_photons > 0.0 {
                // Gaussian approximation to Poisson photon counting.
                let mean = (*v * det.shot_noise_photons).max(0.0);
                let std = mean.sqrt();
                let photons = mean + std * rng.next_gaussian();
                *v = (photons / det.shot_noise_photons).max(0.0);
            }
            if det.read_noise_std > 0.0 {
                *v = (*v + det.read_noise_std * rng.next_gaussian()).max(0.0);
            }
        }
    }

    // 5. Saturation clip.
    if det.saturation > 0.0 {
        for v in &mut intensity {
            if *v > det.saturation {
                *v = det.saturation;
            }
        }
    }

    // 6. Quantization.
    if det.quantization_levels > 1 {
        let max = intensity.iter().cloned().fold(0.0f32, f32::max).max(1e-12);
        let levels = det.quantization_levels as f32;
        for v in &mut intensity {
            let q = (*v / max * (levels - 1.0)).round();
            *v = q / (levels - 1.0) * max;
        }
    }

    OpticalFrame::finalize(w, h, intensity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{InputImage, OpticalField};

    fn ramp_field(n: usize) -> OpticalField {
        let px: Vec<f32> = (0..n * n).map(|i| (i % n) as f32 / n as f32).collect();
        let img = InputImage::from_norm_f32(n, n, px).unwrap();
        OpticalField::from_image(&img, n, n).unwrap()
    }

    #[test]
    fn binning_reduces_resolution() {
        let f = ramp_field(16);
        let det = DetectorConfig {
            binning: 4,
            ..Default::default()
        };
        let frame = capture_with(&f, &det, 0);
        assert_eq!(frame.width, 4);
        assert_eq!(frame.height, 4);
    }

    #[test]
    fn noise_is_deterministic() {
        let f = ramp_field(16);
        let det = DetectorConfig {
            shot_noise_photons: 100.0,
            read_noise_std: 0.01,
            ..Default::default()
        };
        let a = capture_with(&f, &det, 123);
        let b = capture_with(&f, &det, 123);
        assert_eq!(a.frame_hash, b.frame_hash);
        let c = capture_with(&f, &det, 124);
        assert_ne!(a.frame_hash, c.frame_hash);
    }
}
