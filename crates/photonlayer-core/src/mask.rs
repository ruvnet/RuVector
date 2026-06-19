//! Learned phase mask: the trainable optical surface (ADR-260 §9.2).

use crate::complex::Complex;
use crate::error::{PhotonError, Result};
use crate::field::OpticalField;
use crate::rng::DeterministicRng;
use core::f32::consts::PI;
use serde::{Deserialize, Serialize};

/// A phase-only optical element. One phase delay (radians, `[0, 2π)`) per cell.
///
/// This is the structure that training optimizes and that the mask-exchange
/// format (ADR-260 §20.5) serializes. Replay must hash-match the trained mask.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PhaseMask {
    pub width: usize,
    pub height: usize,
    pub phase_radians: Vec<f32>,
    pub mask_id: String,
}

impl PhaseMask {
    pub fn new(
        width: usize,
        height: usize,
        phase_radians: Vec<f32>,
        mask_id: impl Into<String>,
    ) -> Result<Self> {
        if phase_radians.len() != width * height {
            return Err(PhotonError::DimensionMismatch {
                expected: width * height,
                got: phase_radians.len(),
            });
        }
        Ok(Self {
            width,
            height,
            phase_radians,
            mask_id: mask_id.into(),
        })
    }

    /// An identity (flat) mask — equivalent to "optical layer off".
    pub fn identity(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            phase_radians: vec![0.0; width * height],
            mask_id: "identity".into(),
        }
    }

    /// A deterministic pseudo-random phase mask (baseline per ADR-260 §16.1).
    /// Same seed => same mask => same hash, satisfying the replay invariant.
    pub fn random(width: usize, height: usize, seed: u64) -> Self {
        let mut rng = DeterministicRng::new(seed);
        let phase_radians = (0..width * height)
            .map(|_| rng.next_f32() * 2.0 * PI)
            .collect();
        Self {
            width,
            height,
            phase_radians,
            mask_id: format!("random:{seed:#x}"),
        }
    }

    /// A hand-designed quadratic lens phase (baseline "designed lens").
    /// `focal_strength` scales the curvature; positive focuses the field.
    pub fn lens(width: usize, height: usize, focal_strength: f32) -> Self {
        let cx = width as f32 / 2.0;
        let cy = height as f32 / 2.0;
        let mut phase = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let r2 = dx * dx + dy * dy;
                // Wrap into [0, 2π) so it is a valid phase-only element.
                let p = (-focal_strength * r2).rem_euclid(2.0 * PI);
                phase[y * width + x] = p;
            }
        }
        Self {
            width,
            height,
            phase_radians: phase,
            mask_id: format!("lens:{focal_strength}"),
        }
    }

    /// Apply the mask to a field: `out = field * exp(i * phase)`.
    ///
    /// The mask is centered on the field grid when smaller (the common case:
    /// a small learned aperture inside a larger padded grid).
    pub fn apply(&self, field: &mut OpticalField) -> Result<()> {
        if self.width > field.width || self.height > field.height {
            return Err(PhotonError::InvalidMask(format!(
                "mask {}x{} larger than field {}x{}",
                self.width, self.height, field.width, field.height
            )));
        }
        let off_x = (field.width - self.width) / 2;
        let off_y = (field.height - self.height) / 2;
        for y in 0..self.height {
            for x in 0..self.width {
                let theta = self.phase_radians[y * self.width + x];
                let idx = (y + off_y) * field.width + (x + off_x);
                field.data[idx] = field.data[idx] * Complex::from_phase(theta);
            }
        }
        Ok(())
    }

    /// A normalized phase histogram (default 16 bins) used as a mask embedding
    /// for RuVector experiment memory (ADR-260 §12).
    pub fn phase_histogram(&self, bins: usize) -> Vec<f32> {
        let mut hist = vec![0.0f32; bins.max(1)];
        let two_pi = 2.0 * PI;
        for &p in &self.phase_radians {
            let norm = p.rem_euclid(two_pi) / two_pi;
            let mut b = (norm * bins as f32) as usize;
            if b >= bins {
                b = bins - 1;
            }
            hist[b] += 1.0;
        }
        let total: f32 = hist.iter().sum();
        if total > 0.0 {
            for h in &mut hist {
                *h /= total;
            }
        }
        hist
    }
}
