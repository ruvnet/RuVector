//! M1 proof: the cached + in-place `Propagator` is faster than the naive free
//! `propagate()` (which recomputes the transfer function H and clones the field
//! every call), and produces **bit-identical** output. No speedup claim without
//! this measured number.
//!
//! Run: `cargo test -p photonlayer-core --release --test propagation_speedup -- --ignored --nocapture`

use std::time::Instant;

use photonlayer_core::complex::Complex;
use photonlayer_core::config::{OpticalConfig, PropagationMode};
use photonlayer_core::fft::{fftshift_2d, is_pow2};
use photonlayer_core::field::{InputImage, OpticalField};
use photonlayer_core::propagate::{propagate, Propagator};
use std::f32::consts::PI;

const N: usize = 64; // grid (learn-loop regime where H-recompute is a large fraction)
const ITERS: usize = 3000;

fn test_field(n: usize) -> OpticalField {
    // Deterministic non-trivial pattern.
    let px: Vec<f32> = (0..n * n)
        .map(|i| {
            let (x, y) = ((i % n) as f32, (i / n) as f32);
            0.5 + 0.5 * ((x * 0.3).sin() * (y * 0.2).cos())
        })
        .collect();
    let img = InputImage::from_norm_f32(n, n, px).unwrap();
    OpticalField::from_image(&img, n, n).unwrap()
}

/// Always-on correctness gate: the cached + in-place path is bit-for-bit
/// identical to the free `propagate()`. Cheap; runs in the default suite.
#[test]
fn cached_propagator_is_bit_identical() {
    let field = test_field(N);
    let config = OpticalConfig::demo(N, N);
    let reference = propagate(&field, &config).unwrap();
    let prop = Propagator::new(N, N, &config).unwrap();
    let via_struct = prop.propagate(&field).unwrap();
    let mut buf = field.data.clone();
    prop.propagate_into(&mut buf).unwrap();
    assert_eq!(
        via_struct.data, reference.data,
        "Propagator::propagate must match free propagate"
    );
    assert_eq!(
        buf, reference.data,
        "propagate_into must be bit-identical to free propagate"
    );
}

/// Timing proof (M1). Release-only — wall-clock is meaningless in debug. Run:
/// `cargo test -p photonlayer-core --release --test propagation_speedup -- --ignored --nocapture`
#[test]
#[ignore = "timing benchmark — run with --release --ignored"]
fn cached_propagator_is_faster() {
    let field = test_field(N);
    let config = OpticalConfig::demo(N, N);

    // Warm up.
    for _ in 0..64 {
        let _ = propagate(&field, &config).unwrap();
    }

    // Naive: free propagate (recompute H + clone) every call.
    let t = Instant::now();
    let mut sink = 0.0f32;
    for _ in 0..ITERS {
        let out = propagate(&field, &config).unwrap();
        sink += out.data[0].re;
    }
    let naive = t.elapsed().as_secs_f64();

    // Optimized: build operator once; in-place propagate into a reused buffer.
    let prop = Propagator::new(N, N, &config).unwrap();
    let mut scratch = vec![photonlayer_core::complex::Complex::ZERO; N * N];
    let t = Instant::now();
    for _ in 0..ITERS {
        scratch.copy_from_slice(&field.data);
        prop.propagate_into(&mut scratch).unwrap();
        sink += scratch[0].re;
    }
    let opt = t.elapsed().as_secs_f64();
    std::hint::black_box(sink);

    let speedup = naive / opt;
    eprintln!(
        "propagation {N}x{N} x{ITERS}: naive={:.1}ms  cached+inplace={:.1}ms  speedup={speedup:.2}x",
        naive * 1e3,
        opt * 1e3
    );
    assert!(
        speedup >= 1.5,
        "cached+in-place propagator must be >= 1.5x the naive path; got {speedup:.2}x"
    );
}

// ---------------------------------------------------------------------------
// OPT-A + OPT-B benchmark: the new Fraunhofer path (±1 checkerboard premultiply
// that folds away `fftshift`, plus a table-indexed FFT that replaces the
// per-butterfly `w *= wlen` accumulation) vs a self-contained reimplementation
// of the OLD path (accumulated-twiddle 2D FFT, then `fftshift_2d`). The old
// path is rebuilt locally so the "before" number is real, not assumed.
// ---------------------------------------------------------------------------

/// Old 1D FFT: accumulates `w *= wlen` per stage (the pre-OPT-B behavior).
fn old_fft_1d(data: &mut [Complex], inverse: bool) {
    let n = data.len();
    assert!(is_pow2(n));
    if n == 1 {
        return;
    }
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
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let wlen = Complex::from_phase(sign * 2.0 * PI / len as f32);
        let half = len / 2;
        let mut i = 0;
        while i < n {
            let mut w = Complex::ONE;
            for k in 0..half {
                let u = data[i + k];
                let v = data[i + k + half] * w;
                data[i + k] = u + v;
                data[i + k + half] = u - v;
                w = w * wlen;
            }
            i += len;
        }
        len <<= 1;
    }
    if inverse {
        let inv = 1.0 / n as f32;
        for c in data.iter_mut() {
            *c = c.scale(inv);
        }
    }
}

/// Old 2D FFT: rebuilds `wlen` per row and per column (no shared table).
fn old_fft_2d(data: &mut [Complex], width: usize, height: usize, inverse: bool) {
    for r in 0..height {
        old_fft_1d(&mut data[r * width..(r + 1) * width], inverse);
    }
    let mut col = vec![Complex::ZERO; height];
    for c in 0..width {
        for r in 0..height {
            col[r] = data[r * width + c];
        }
        old_fft_1d(&mut col, inverse);
        for r in 0..height {
            data[r * width + c] = col[r];
        }
    }
}

/// Old Fraunhofer: `old_fft_2d` then `fftshift_2d` then normalize.
fn old_fraunhofer_into(data: &mut [Complex], w: usize, h: usize) {
    old_fft_2d(data, w, h, false);
    fftshift_2d(data, w, h);
    let norm = 1.0 / (w as f32 * h as f32).sqrt();
    for c in data.iter_mut() {
        *c = c.scale(norm);
    }
}

#[test]
#[ignore = "timing benchmark — run with --release --ignored"]
fn fraunhofer_optab_is_faster() {
    let field = test_field(N);
    let mut config = OpticalConfig::demo(N, N);
    config.propagation = PropagationMode::Fraunhofer;
    let prop = Propagator::new(N, N, &config).unwrap();

    // Correctness gate (always meaningful): the new in-place Fraunhofer path is
    // bit-for-bit identical to the locally-rebuilt OLD fft+fftshift path? NO —
    // OPT-B deliberately changes bits (drift removed). So assert they agree to a
    // tight f32 tolerance, and assert the new path is internally deterministic.
    let mut new_buf = field.data.clone();
    prop.propagate_into(&mut new_buf).unwrap();
    let mut new_buf2 = field.data.clone();
    prop.propagate_into(&mut new_buf2).unwrap();
    assert_eq!(
        new_buf, new_buf2,
        "new Fraunhofer path must be deterministic"
    );

    let mut old_buf = field.data.clone();
    old_fraunhofer_into(&mut old_buf, N, N);
    let max_diff = new_buf
        .iter()
        .zip(&old_buf)
        .map(|(a, b)| (a.re - b.re).abs().max((a.im - b.im).abs()))
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-3,
        "OPT-B should only shift bits within f32 noise vs old path; got {max_diff:e}"
    );

    // Warm up.
    for _ in 0..64 {
        let mut b = field.data.clone();
        prop.propagate_into(&mut b).unwrap();
    }

    // Old path timing.
    let t = Instant::now();
    let mut sink = 0.0f32;
    let mut scratch = vec![Complex::ZERO; N * N];
    for _ in 0..ITERS {
        scratch.copy_from_slice(&field.data);
        old_fraunhofer_into(&mut scratch, N, N);
        sink += scratch[0].re;
    }
    let old = t.elapsed().as_secs_f64();

    // New path timing (OPT-A checkerboard + OPT-B twiddle table, in-place).
    let t = Instant::now();
    for _ in 0..ITERS {
        scratch.copy_from_slice(&field.data);
        prop.propagate_into(&mut scratch).unwrap();
        sink += scratch[0].re;
    }
    let new = t.elapsed().as_secs_f64();
    std::hint::black_box(sink);

    let speedup = old / new;
    eprintln!(
        "fraunhofer OPT-A+B {N}x{N} x{ITERS}: old(fft+fftshift,accum-twiddle)={:.1}ms  \
         new(checkerboard+table)={:.1}ms  speedup={speedup:.2}x  max_diff_vs_old={max_diff:e}",
        old * 1e3,
        new * 1e3
    );
    assert!(
        speedup >= 1.0,
        "OPT-A+B Fraunhofer path must not be slower than the old path; got {speedup:.2}x"
    );
}
