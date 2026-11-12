//! FFT-backed circular correlation and convolution (ADR-002 §3, ADR-001 §3).
//!
//! HolE's score is `r · (e_s ⋆ e_o)` where `⋆` is circular correlation. The
//! DFT identity `F(a ⋆ b) = conj(F(a)) ⊙ F(b)` turns the `O(d²)` sum into an
//! `O(d log d)` FFT round-trip. This module owns that primitive plus a direct
//! `O(d²)` reference that the property tests assert equality against, and that
//! the spike times FFT against.
//!
//! **Backend:** `rustfft` 6.4.1 (`MIT OR Apache-2.0`, on the `deny.toml`
//! allow-list; the exact version already resolved in the workspace lock so no
//! second copy is pulled). It compiles and runs on `wasm32-unknown-unknown` —
//! the ADR-002 hard gate — verified by `cargo check -p ruvector-kge --target
//! wasm32-unknown-unknown`.
//!
//! **Convention:** `d` must be even (HolE's Fourier index relies on a real
//! spectrum with a defined Nyquist bin); `FftPlan::new` errors otherwise.
//! Forward/inverse plans are cached in the struct and amortised across batch
//! scoring. `rustfft`'s inverse is unnormalised, so every round-trip divides by
//! `d`. `Fft::process` allocates its own scratch per call (we hold the plan
//! behind `&self` and cannot thread a reused scratch buffer through the
//! `Scorer` trait); the spike measures the primitive exactly as callers use it.
//!
//! ## Spike results (ADR-002 §3, native x86-64, 2026-09-21)
//!
//! FFT circular correlation vs the direct `O(d²)` reference, and batch matmul
//! scoring vs triple-by-triple, median ns/op. Full data + methodology in
//! `npm/packages/kge/bench/fft-spike-2026-09-21.json`.
//!
//! | d   | direct corr ns | fft corr ns | fft speedup | batch/triple speedup |
//! |-----|----------------|-------------|-------------|----------------------|
//! | 128 | 17800 | 250 | 71.2x | 10.5x |
//! | 256 | 71900 | 520 | 138.3x | 10.3x |
//! | 512 | 288271 | 970 | 297.2x | 11.4x |
//!
//! **Verdict:** PASS. FFT beats direct at every d and by 297x at d=512 (native x86-64), and rustfft compiles+runs on wasm32. The batch/per-query speedup (~10-11x) is the amortised figure: the Fourier-matrix build is one-time and reused across queries; a single query that also pays the build is roughly break-even with per-triple scoring.

use crate::{KgeError, Result};
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// Complex sample type used throughout the FFT path.
pub type C = Complex<f32>;

/// Cached forward + inverse FFT plans for a fixed, even dimension `d`.
pub struct FftPlan {
    d: usize,
    forward: Arc<dyn Fft<f32>>,
    inverse: Arc<dyn Fft<f32>>,
}

impl FftPlan {
    /// Build cached plans for dimension `d`. `d` must be even and non-zero.
    pub fn new(d: usize) -> Result<Self> {
        if d == 0 || !d.is_multiple_of(2) {
            return Err(KgeError::Invalid(format!(
                "fft dimension must be even and non-zero, got {d}"
            )));
        }
        let mut planner = FftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(d);
        let inverse = planner.plan_fft_inverse(d);
        Ok(Self {
            d,
            forward,
            inverse,
        })
    }

    /// The dimension these plans are built for.
    pub fn dims(&self) -> usize {
        self.d
    }

    /// Forward DFT of a real vector, returned as `d` complex bins
    /// (unnormalised, `rustfft` sign convention `exp(-2πi·m·i/d)`).
    pub fn forward(&self, x: &[f32]) -> Vec<C> {
        debug_assert_eq!(x.len(), self.d, "forward: length must equal dims");
        let mut buf: Vec<C> = x.iter().map(|&v| C::new(v, 0.0)).collect();
        self.forward.process(&mut buf);
        buf
    }

    /// Circular correlation `corr(a,b)[k] = Σ_i a[i]·b[(i+k) mod d]`, via
    /// `IFFT( conj(F(a)) ⊙ F(b) ) / d`.
    pub fn circular_correlation(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        debug_assert_eq!(a.len(), self.d);
        debug_assert_eq!(b.len(), self.d);
        let fa = self.forward(a);
        let fb = self.forward(b);
        let mut prod: Vec<C> = fa.iter().zip(&fb).map(|(x, y)| x.conj() * y).collect();
        self.inverse.process(&mut prod);
        let inv_d = 1.0 / self.d as f32;
        prod.iter().map(|c| c.re * inv_d).collect()
    }

    /// Circular convolution `conv(a,b)[k] = Σ_i a[i]·b[(k−i) mod d]`, via
    /// `IFFT( F(a) ⊙ F(b) ) / d`.
    pub fn circular_convolution(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        debug_assert_eq!(a.len(), self.d);
        debug_assert_eq!(b.len(), self.d);
        let fa = self.forward(a);
        let fb = self.forward(b);
        let mut prod: Vec<C> = fa.iter().zip(&fb).map(|(x, y)| x * y).collect();
        self.inverse.process(&mut prod);
        let inv_d = 1.0 / self.d as f32;
        prod.iter().map(|c| c.re * inv_d).collect()
    }
}

/// Direct `O(d²)` circular correlation — the correctness oracle and small-`d`
/// fallback (ADR-002 alternatives). `corr(a,b)[k] = Σ_i a[i]·b[(i+k) mod d]`.
pub fn corr_direct(a: &[f32], b: &[f32]) -> Vec<f32> {
    let d = a.len();
    let mut out = vec![0.0f32; d];
    for (k, slot) in out.iter_mut().enumerate() {
        let mut s = 0.0f32;
        for i in 0..d {
            s += a[i] * b[(i + k) % d];
        }
        *slot = s;
    }
    out
}

/// Direct `O(d²)` circular convolution. `conv(a,b)[k] = Σ_i a[i]·b[(k−i) mod d]`.
pub fn conv_direct(a: &[f32], b: &[f32]) -> Vec<f32> {
    let d = a.len();
    let mut out = vec![0.0f32; d];
    for (k, slot) in out.iter_mut().enumerate() {
        let mut s = 0.0f32;
        for i in 0..d {
            s += a[i] * b[(k + d - i) % d];
        }
        *slot = s;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random f32 in [-1, 1) — no `rand` dependency, so
    /// the core stays target-independent (ADR-005).
    fn rand_vec(d: usize, mut state: u64) -> Vec<f32> {
        (0..d)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let u = (state >> 11) as f32 / (1u64 << 53) as f32;
                2.0 * u - 1.0
            })
            .collect()
    }

    fn max_abs_diff(x: &[f32], y: &[f32]) -> f32 {
        x.iter()
            .zip(y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn odd_dimension_rejected() {
        assert!(FftPlan::new(7).is_err());
        assert!(FftPlan::new(0).is_err());
        assert!(FftPlan::new(8).is_ok());
    }

    #[test]
    fn fft_correlation_matches_direct() {
        for &d in &[8usize, 128, 256, 512] {
            let plan = FftPlan::new(d).unwrap();
            let a = rand_vec(d, 0x1234_5678 ^ d as u64);
            let b = rand_vec(d, 0x9e37_79b9 ^ d as u64);
            let fft = plan.circular_correlation(&a, &b);
            let direct = corr_direct(&a, &b);
            // Mixed abs+rel tolerance: f32 FFT accumulation over d terms with
            // O(1) inputs lands well under this at every d tested.
            for (f, dref) in fft.iter().zip(&direct) {
                assert!(
                    (f - dref).abs() <= 1e-4 + 1e-4 * dref.abs(),
                    "d={d}: fft {f} vs direct {dref}"
                );
            }
            assert!(max_abs_diff(&fft, &direct) < 1e-2, "d={d} gross mismatch");
        }
    }

    #[test]
    fn fft_convolution_matches_direct() {
        for &d in &[8usize, 128, 256, 512] {
            let plan = FftPlan::new(d).unwrap();
            let a = rand_vec(d, 0xabcd ^ d as u64);
            let b = rand_vec(d, 0xfeed ^ d as u64);
            let fft = plan.circular_convolution(&a, &b);
            let direct = conv_direct(&a, &b);
            for (f, dref) in fft.iter().zip(&direct) {
                assert!(
                    (f - dref).abs() <= 1e-4 + 1e-4 * dref.abs(),
                    "d={d}: fft {f} vs direct {dref}"
                );
            }
        }
    }
}
