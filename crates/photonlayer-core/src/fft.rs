//! Dependency-free, deterministic FFT (iterative radix-2 Cooley–Tukey).
//!
//! Restricted to power-of-two transform lengths. The optical engine pads
//! grids to powers of two before propagation, which keeps the transform
//! exact and bit-for-bit reproducible across platforms (no FFT library
//! threading or SIMD-order nondeterminism).

use crate::complex::Complex;
use core::f32::consts::PI;

/// Returns true if `n` is a power of two and non-zero.
#[inline]
pub fn is_pow2(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

/// Precomputed twiddle factors for a length-`n` FFT of a fixed direction.
///
/// Holds `tw[j] = exp(sign · 2π · j / n)` for `j in 0..n/2`. The stage-`len`
/// butterfly twiddle for index `k` is `tw[k * (n / len)]`, so every factor is
/// read straight from the table by index — never accumulated with repeated
/// complex multiplies. This both removes the per-butterfly `w *= wlen` cost and
/// eliminates the f32 drift that accumulation injects (a determinism *gain*:
/// the angles are computed once at full `cos/sin` precision).
#[derive(Clone)]
pub struct TwiddleTable {
    n: usize,
    inverse: bool,
    tw: Vec<Complex>,
}

impl TwiddleTable {
    /// Build the table for a length-`n` (power-of-two) transform.
    ///
    /// # Panics
    /// Panics if `n` is not a power of two.
    pub fn new(n: usize, inverse: bool) -> Self {
        assert!(is_pow2(n), "FFT length must be a power of two, got {n}");
        let sign = if inverse { 1.0 } else { -1.0 };
        let half = n / 2; // 0 when n == 1; table is unused at that size.
        let mut tw = Vec::with_capacity(half);
        let scale = sign * 2.0 * PI / n as f32;
        for j in 0..half {
            // Index the angle directly: no `w *= wlen` accumulation, no drift.
            tw.push(Complex::from_phase(j as f32 * scale));
        }
        Self { n, inverse, tw }
    }
}

/// In-place 1D FFT. `inverse = true` computes the inverse transform and
/// applies the `1/N` normalization so that `ifft(fft(x)) == x`.
///
/// Builds a one-shot [`TwiddleTable`]; callers transforming many equal-length
/// rows/columns should build the table once and use [`fft_1d_with`].
///
/// # Panics
/// Panics if `data.len()` is not a power of two.
pub fn fft_1d(data: &mut [Complex], inverse: bool) {
    let table = TwiddleTable::new(data.len().max(1), inverse);
    fft_1d_with(data, &table);
}

/// In-place 1D FFT using a precomputed [`TwiddleTable`] (must match the buffer
/// length and direction).
///
/// # Panics
/// Panics if `data.len()` is not a power of two or does not match `table.n`.
pub fn fft_1d_with(data: &mut [Complex], table: &TwiddleTable) {
    let n = data.len();
    assert!(is_pow2(n), "FFT length must be a power of two, got {n}");
    assert_eq!(n, table.n, "twiddle table length mismatch");
    if n == 1 {
        return;
    }

    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(i, j);
        }
    }

    // Danielson–Lanczos butterflies. Index the twiddle table by stride instead
    // of accumulating `w *= wlen` — same math, no per-stage drift.
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let stride = n / len; // table[k * stride] == exp(sign · 2π · k / len)
        let mut i = 0;
        while i < n {
            for k in 0..half {
                let w = table.tw[k * stride];
                let u = data[i + k];
                let v = data[i + k + half] * w;
                data[i + k] = u + v;
                data[i + k + half] = u - v;
            }
            i += len;
        }
        len <<= 1;
    }

    if table.inverse {
        let inv = 1.0 / n as f32;
        for c in data.iter_mut() {
            *c = c.scale(inv);
        }
    }
}

/// In-place 2D FFT on a row-major `width * height` buffer.
///
/// Both dimensions must be powers of two. Performs row transforms followed
/// by column transforms (separable DFT).
pub fn fft_2d(data: &mut [Complex], width: usize, height: usize, inverse: bool) {
    assert_eq!(data.len(), width * height, "buffer size mismatch");
    assert!(
        is_pow2(width) && is_pow2(height),
        "dims must be power of two"
    );

    // Build each dimension's twiddle table once and reuse it across every
    // row / column transform (OPT-B) — angles are computed a single time.
    let row_tw = TwiddleTable::new(width, inverse);
    for r in 0..height {
        let row = &mut data[r * width..(r + 1) * width];
        fft_1d_with(row, &row_tw);
    }

    // Columns (gather/scatter to keep the 1D kernel contiguous).
    let col_tw = TwiddleTable::new(height, inverse);
    let mut col = vec![Complex::ZERO; height];
    for c in 0..width {
        for r in 0..height {
            col[r] = data[r * width + c];
        }
        fft_1d_with(&mut col, &col_tw);
        for r in 0..height {
            data[r * width + c] = col[r];
        }
    }
}

/// Checkerboard premultiply: negate every sample at an odd `(row + col)`.
///
/// By the DFT shift theorem, modulating the input by `(-1)^(x+y)` shifts the
/// transform output by `(N/2, M/2)` — i.e. forward-FFT of the premultiplied
/// buffer equals `fftshift_2d` of the forward-FFT of the original. This lets a
/// Fraunhofer path do `premult → fft_2d` instead of `fft_2d → fftshift_2d`,
/// avoiding the full-buffer allocation + quadrant copy in [`fftshift_2d`].
///
/// The negation is exact (`{-re, -im}`), so the substitution is bit-identical
/// to the fft-then-fftshift sequence on every platform.
pub fn checkerboard_premultiply(data: &mut [Complex], width: usize, height: usize) {
    debug_assert_eq!(data.len(), width * height, "buffer size mismatch");
    for row in 0..height {
        // First column negated when the row index is odd; flips every column.
        let mut neg = row & 1 == 1;
        let base = row * width;
        for c in &mut data[base..base + width] {
            if neg {
                *c = Complex::ZERO - *c;
            }
            neg = !neg;
        }
    }
}

/// 2D fftshift: swaps quadrants so the zero-frequency component moves to the
/// center. `width` and `height` must be even (always true for power-of-two).
pub fn fftshift_2d(data: &mut [Complex], width: usize, height: usize) {
    let hw = width / 2;
    let hh = height / 2;
    let mut out = vec![Complex::ZERO; data.len()];
    for r in 0..height {
        let nr = (r + hh) % height;
        for c in 0..width {
            let nc = (c + hw) % width;
            out[nr * width + nc] = data[r * width + c];
        }
    }
    data.copy_from_slice(&out);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_1d() {
        let mut x: Vec<Complex> = (0..8).map(|i| Complex::new(i as f32, 0.0)).collect();
        let orig = x.clone();
        fft_1d(&mut x, false);
        fft_1d(&mut x, true);
        for (a, b) in x.iter().zip(orig.iter()) {
            assert!((a.re - b.re).abs() < 1e-4, "{a:?} vs {b:?}");
            assert!(a.im.abs() < 1e-4);
        }
    }

    #[test]
    fn dc_component_is_sum() {
        // FFT of a constant signal -> all energy in bin 0.
        let mut x = vec![Complex::new(2.0, 0.0); 16];
        fft_1d(&mut x, false);
        assert!((x[0].re - 32.0).abs() < 1e-3);
        for c in &x[1..] {
            assert!(c.abs() < 1e-3);
        }
    }

    #[test]
    fn checkerboard_premult_equals_fft_then_fftshift() {
        // OPT-A correctness gate: `premult → fft` must be ELEMENT-FOR-ELEMENT
        // identical to `fft → fftshift` (shift theorem, exact ±1 negation).
        for &(w, h) in &[(8usize, 8usize), (16, 4), (4, 16), (32, 32), (2, 2)] {
            let src: Vec<Complex> = (0..w * h)
                .map(|i| Complex::new((i % 7) as f32 - 3.0, (i % 5) as f32 - 2.0))
                .collect();

            // Old path: forward FFT, then quadrant fftshift.
            let mut old = src.clone();
            fft_2d(&mut old, w, h, false);
            fftshift_2d(&mut old, w, h);

            // New path: checkerboard premultiply, then forward FFT.
            let mut new = src.clone();
            checkerboard_premultiply(&mut new, w, h);
            fft_2d(&mut new, w, h, false);

            assert_eq!(new, old, "checkerboard path differs at {w}x{h}");
        }
    }

    #[test]
    fn checkerboard_is_exact_pm_one() {
        // Negation must be exact ±1.0 (true negate), not a multiply by -1.0f32
        // that could differ; applying it twice restores the original bits.
        let src: Vec<Complex> = (0..16)
            .map(|i| Complex::new(i as f32 * 0.123 - 1.0, i as f32 * -0.071))
            .collect();
        let mut x = src.clone();
        checkerboard_premultiply(&mut x, 4, 4);
        checkerboard_premultiply(&mut x, 4, 4);
        assert_eq!(x, src, "double checkerboard must be identity (bit-exact)");
    }

    /// Reference forward DFT in f64 (no FFT factorization, no f32 accumulation)
    /// — the ground truth OPT-B's twiddle tables are measured against.
    fn dft_1d_ref_f64(x: &[Complex]) -> Vec<(f64, f64)> {
        let n = x.len();
        let mut out = vec![(0.0f64, 0.0f64); n];
        for (k, slot) in out.iter_mut().enumerate() {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (j, c) in x.iter().enumerate() {
                let ang = -2.0 * std::f64::consts::PI * (k * j) as f64 / n as f64;
                let (s, co) = ang.sin_cos();
                re += c.re as f64 * co - c.im as f64 * s;
                im += c.re as f64 * s + c.im as f64 * co;
            }
            *slot = (re, im);
        }
        out
    }

    #[test]
    fn fft_1d_is_deterministic_bitexact() {
        // OPT-B determinism gate: identical input -> identical output bytes.
        let src: Vec<Complex> = (0..64)
            .map(|i| Complex::new((i as f32 * 0.37).sin(), (i as f32 * 0.11).cos()))
            .collect();
        let mut a = src.clone();
        let mut b = src.clone();
        fft_1d(&mut a, false);
        fft_1d(&mut b, false);
        assert_eq!(a, b, "FFT must be bit-for-bit reproducible across runs");
    }

    #[test]
    fn twiddle_table_error_does_not_increase() {
        // OPT-B accuracy gate: indexing a precomputed table must not worsen
        // max-abs error vs an f64 reference DFT — drift removal should help.
        let n = 256;
        let src: Vec<Complex> = (0..n)
            .map(|i| Complex::new((i as f32 * 0.21).sin(), (i as f32 * 0.05).cos()))
            .collect();
        let reference = dft_1d_ref_f64(&src);

        // New (table-indexed) path.
        let mut new = src.clone();
        fft_1d(&mut new, false);
        let err_new = new
            .iter()
            .zip(&reference)
            .map(|(c, &(re, im))| ((c.re as f64 - re).abs()).max((c.im as f64 - im).abs()))
            .fold(0.0f64, f64::max);

        // Old (accumulated `w *= wlen`) path, recomputed here for comparison.
        let mut old = src.clone();
        let nn = old.len();
        {
            let mut jj = 0usize;
            for i in 1..nn {
                let mut bit = nn >> 1;
                while jj & bit != 0 {
                    jj ^= bit;
                    bit >>= 1;
                }
                jj ^= bit;
                if i < jj {
                    old.swap(i, jj);
                }
            }
            let mut len = 2;
            while len <= nn {
                let wlen = Complex::from_phase(-2.0 * PI / len as f32);
                let half = len / 2;
                let mut i = 0;
                while i < nn {
                    let mut w = Complex::ONE;
                    for k in 0..half {
                        let u = old[i + k];
                        let v = old[i + k + half] * w;
                        old[i + k] = u + v;
                        old[i + k + half] = u - v;
                        w = w * wlen;
                    }
                    i += len;
                }
                len <<= 1;
            }
        }
        let err_old = old
            .iter()
            .zip(&reference)
            .map(|(c, &(re, im))| ((c.re as f64 - re).abs()).max((c.im as f64 - im).abs()))
            .fold(0.0f64, f64::max);

        assert!(
            err_new <= err_old,
            "table FFT error {err_new:e} must not exceed accumulated-twiddle error {err_old:e}"
        );
    }

    #[test]
    fn roundtrip_2d() {
        let (w, h) = (8, 4);
        let mut x: Vec<Complex> = (0..w * h)
            .map(|i| Complex::new((i % 5) as f32, 0.0))
            .collect();
        let orig = x.clone();
        fft_2d(&mut x, w, h, false);
        fft_2d(&mut x, w, h, true);
        for (a, b) in x.iter().zip(orig.iter()) {
            assert!((a.re - b.re).abs() < 1e-3);
        }
    }
}
