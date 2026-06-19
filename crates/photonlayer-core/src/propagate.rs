//! Scalar diffraction propagation (ADR-260 §9.3).
//!
//! Three modes are supported, matching the references cited in ADR-260
//! (TorchOptics / waveprop): Fresnel near-field, Fraunhofer far-field, and
//! the angular-spectrum method. All operate on a power-of-two complex grid
//! and use the in-house deterministic FFT.

use crate::complex::Complex;
use crate::config::{OpticalConfig, PropagationMode};
use crate::error::{PhotonError, Result};
use crate::fft::{checkerboard_premultiply, fft_2d, is_pow2};
use crate::field::OpticalField;
use core::f32::consts::PI;

/// Discrete FFT sample frequencies (cycles per unit length), FFT bin order.
fn fftfreq(n: usize, d: f32) -> Vec<f32> {
    let mut f = vec![0.0f32; n];
    let inv = 1.0 / (n as f32 * d);
    let half = n.div_ceil(2);
    for (i, slot) in f.iter_mut().enumerate() {
        let k = if i < half {
            i as i64
        } else {
            i as i64 - n as i64
        };
        *slot = k as f32 * inv;
    }
    f
}

/// Propagate a field by `config.propagation_mm` using the selected model.
///
/// Returns a new field at the detector plane. Power is approximately
/// conserved for Fresnel / angular-spectrum (unitary transfer functions).
pub fn propagate(field: &OpticalField, config: &OpticalConfig) -> Result<OpticalField> {
    if !is_pow2(field.width) {
        return Err(PhotonError::NotPowerOfTwo(field.width));
    }
    if !is_pow2(field.height) {
        return Err(PhotonError::NotPowerOfTwo(field.height));
    }
    match config.propagation {
        PropagationMode::Fraunhofer => fraunhofer(field),
        PropagationMode::Fresnel => transfer_fn(field, config, TransferKind::Fresnel),
        PropagationMode::AngularSpectrum => {
            transfer_fn(field, config, TransferKind::AngularSpectrum)
        }
    }
}

fn fraunhofer(field: &OpticalField) -> Result<OpticalField> {
    let (w, h) = (field.width, field.height);
    let mut data = field.data.clone();
    // fftshift(FFT(x)) == FFT((-1)^(x+y) · x): premultiply by a ±1 checkerboard
    // before the transform instead of shifting quadrants after it. Exact ±1.0
    // negation -> bit-identical to `fft_2d` + `fftshift_2d`, but no shift alloc.
    checkerboard_premultiply(&mut data, w, h);
    fft_2d(&mut data, w, h, false);
    // Normalize so total power stays in a sane range for downstream metrics.
    let norm = 1.0 / (w as f32 * h as f32).sqrt();
    for c in &mut data {
        *c = c.scale(norm);
    }
    Ok(OpticalField {
        width: w,
        height: h,
        data,
    })
}

enum TransferKind {
    Fresnel,
    AngularSpectrum,
}

/// Build the config-only transfer function H (length `w*h`, row-major). H depends
/// solely on (w, h, λ, z, d, kind) — never on the field — so it can be computed
/// once and reused across many propagations (see [`Propagator`]).
fn transfer_kernel(w: usize, h: usize, config: &OpticalConfig, kind: TransferKind) -> Vec<Complex> {
    let lambda = config.wavelength_m();
    let z = config.distance_m();
    let d = config.pixel_pitch_m();
    let fx = fftfreq(w, d);
    let fy = fftfreq(h, d);
    let k = 2.0 * PI / lambda;
    let mut hk = vec![Complex::ZERO; w * h];
    for row in 0..h {
        for col in 0..w {
            let fxx = fx[col];
            let fyy = fy[row];
            let h_val = match kind {
                TransferKind::Fresnel => {
                    // Drop constant exp(i k z); keep quadratic phase.
                    Complex::from_phase(-PI * lambda * z * (fxx * fxx + fyy * fyy))
                }
                TransferKind::AngularSpectrum => {
                    let arg = 1.0 - (lambda * fxx).powi(2) - (lambda * fyy).powi(2);
                    if arg <= 0.0 {
                        // Evanescent: does not propagate to the far detector.
                        Complex::ZERO
                    } else {
                        Complex::from_phase(k * z * arg.sqrt())
                    }
                }
            };
            hk[row * w + col] = h_val;
        }
    }
    hk
}

/// Apply a precomputed transfer kernel: forward FFT → ×H → inverse FFT.
fn apply_transfer(field: &OpticalField, w: usize, h: usize, hk: &[Complex]) -> OpticalField {
    let mut data = field.data.clone();
    fft_2d(&mut data, w, h, false);
    for (dv, hv) in data.iter_mut().zip(hk.iter()) {
        *dv = *dv * *hv;
    }
    fft_2d(&mut data, w, h, true);
    OpticalField {
        width: w,
        height: h,
        data,
    }
}

fn transfer_fn(
    field: &OpticalField,
    config: &OpticalConfig,
    kind: TransferKind,
) -> Result<OpticalField> {
    let (w, h) = (field.width, field.height);
    let hk = transfer_kernel(w, h, config, kind);
    Ok(apply_transfer(field, w, h, &hk))
}

/// A precomputed propagation operator. Build once per `(config, width, height)`
/// and reuse across many fields — the config-only transfer function is computed
/// a single time instead of on every call. This is the hot path in mask-learning
/// loops (thousands of propagations share one config). Output is bit-identical to
/// the free [`propagate`] function.
pub struct Propagator {
    width: usize,
    height: usize,
    kind: PropKind,
}

enum PropKind {
    Fraunhofer,
    /// Precomputed transfer function H (length `width*height`).
    Transfer(Vec<Complex>),
}

impl Propagator {
    /// Precompute the operator for a fixed grid + config.
    pub fn new(width: usize, height: usize, config: &OpticalConfig) -> Result<Self> {
        if !is_pow2(width) {
            return Err(PhotonError::NotPowerOfTwo(width));
        }
        if !is_pow2(height) {
            return Err(PhotonError::NotPowerOfTwo(height));
        }
        let kind = match config.propagation {
            PropagationMode::Fraunhofer => PropKind::Fraunhofer,
            PropagationMode::Fresnel => PropKind::Transfer(transfer_kernel(
                width,
                height,
                config,
                TransferKind::Fresnel,
            )),
            PropagationMode::AngularSpectrum => PropKind::Transfer(transfer_kernel(
                width,
                height,
                config,
                TransferKind::AngularSpectrum,
            )),
        };
        Ok(Self {
            width,
            height,
            kind,
        })
    }

    /// Propagate a field through the precomputed operator.
    pub fn propagate(&self, field: &OpticalField) -> Result<OpticalField> {
        if field.width != self.width || field.height != self.height {
            return Err(PhotonError::NotPowerOfTwo(field.width));
        }
        match &self.kind {
            PropKind::Fraunhofer => fraunhofer(field),
            PropKind::Transfer(hk) => Ok(apply_transfer(field, self.width, self.height, hk)),
        }
    }

    /// **In-place** propagation — forward FFT → ×H → inverse FFT, mutating `data`
    /// directly (no per-call field clone). Bit-identical to [`Propagator::propagate`];
    /// this is the batch hot path (mask-learning loops over many samples).
    pub fn propagate_into(&self, data: &mut [Complex]) -> Result<()> {
        let (w, h) = (self.width, self.height);
        if data.len() != w * h {
            return Err(PhotonError::NotPowerOfTwo(data.len()));
        }
        match &self.kind {
            PropKind::Fraunhofer => {
                // OPT-A: ±1 checkerboard premultiply folds the post-FFT fftshift
                // into the input (shift theorem) — bit-identical, no shift alloc.
                checkerboard_premultiply(data, w, h);
                fft_2d(data, w, h, false);
                let norm = 1.0 / (w as f32 * h as f32).sqrt();
                for c in data.iter_mut() {
                    *c = c.scale(norm);
                }
            }
            PropKind::Transfer(hk) => {
                fft_2d(data, w, h, false);
                for (dv, hv) in data.iter_mut().zip(hk.iter()) {
                    *dv = *dv * *hv;
                }
                fft_2d(data, w, h, true);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{InputImage, OpticalField};

    fn point_field(n: usize) -> OpticalField {
        let mut px = vec![0.0f32; n * n];
        px[(n / 2) * n + n / 2] = 1.0;
        let img = InputImage::from_norm_f32(n, n, px).unwrap();
        OpticalField::from_image(&img, n, n).unwrap()
    }

    #[test]
    fn angular_spectrum_conserves_power() {
        let f = point_field(32);
        let mut cfg = OpticalConfig::demo(32, 32);
        cfg.propagation = PropagationMode::AngularSpectrum;
        cfg.propagation_mm = 2.0;
        let out = propagate(&f, &cfg).unwrap();
        let p0 = f.power();
        let p1 = out.power();
        // Unitary transfer fn (ignoring evanescent cutoff) -> power preserved.
        assert!((p1 - p0).abs() / p0 < 0.05, "power {p0} -> {p1}");
    }

    #[test]
    fn point_spreads_under_propagation() {
        let f = point_field(32);
        let mut cfg = OpticalConfig::demo(32, 32);
        cfg.propagation = PropagationMode::Fresnel;
        cfg.propagation_mm = 5.0;
        let out = propagate(&f, &cfg).unwrap();
        // The single bright pixel should diffract into many pixels.
        let nonzero = out.data.iter().filter(|c| c.norm_sqr() > 1e-6).count();
        assert!(nonzero > 10, "point did not spread: {nonzero} nonzero");
    }

    #[test]
    fn fraunhofer_of_point_is_uniform() {
        let f = point_field(16);
        let mut cfg = OpticalConfig::demo(16, 16);
        cfg.propagation = PropagationMode::Fraunhofer;
        let out = propagate(&f, &cfg).unwrap();
        // FT of a centered delta -> uniform magnitude everywhere.
        let mags: Vec<f32> = out.data.iter().map(|c| c.abs()).collect();
        let mx = mags.iter().cloned().fold(0.0, f32::max);
        let mn = mags.iter().cloned().fold(f32::MAX, f32::min);
        assert!((mx - mn).abs() < 1e-3, "not uniform: {mn}..{mx}");
    }
}
