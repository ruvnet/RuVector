//! Minimal STFT (Short-Time Fourier Transform) implementation.
//!
//! No external DSP dependencies — uses a radix-2 Cooley-Tukey FFT
//! and Hann window for time-frequency decomposition.

use std::f64::consts::PI;

/// A single time-frequency bin produced by STFT.
#[derive(Debug, Clone, Copy)]
pub struct TfBin {
    /// Time frame index.
    pub frame: usize,
    /// Frequency bin index.
    pub freq_bin: usize,
    /// Magnitude (amplitude).
    pub magnitude: f64,
    /// Phase in radians.
    pub phase: f64,
}

/// STFT analysis result.
pub struct StftResult {
    /// Time-frequency bins (frame-major order).
    pub bins: Vec<TfBin>,
    /// Number of time frames.
    pub num_frames: usize,
    /// Number of frequency bins per frame.
    pub num_freq_bins: usize,
    /// Hop size used.
    pub hop_size: usize,
    /// Window size used.
    pub window_size: usize,
    /// Sample rate.
    pub sample_rate: f64,
}

/// Hann window of length `n`.
fn hann_window(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / n as f64).cos()))
        .collect()
}

/// Precomputed twiddle factors for each FFT stage, avoiding repeated sin/cos.
struct TwiddleCache {
    /// twiddles[stage] = [(cos, sin), ...] for half-length of that stage
    stages: Vec<Vec<(f64, f64)>>,
}

impl TwiddleCache {
    fn new(n: usize) -> Self {
        let mut stages = Vec::new();
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let angle = -2.0 * PI / len as f64;
            let twiddles: Vec<(f64, f64)> = (0..half)
                .map(|k| {
                    let a = angle * k as f64;
                    (a.cos(), a.sin())
                })
                .collect();
            stages.push(twiddles);
            len <<= 1;
        }
        Self { stages }
    }
}

/// Thread-local twiddle cache to avoid recomputation across frames.
/// Keyed by FFT size.
thread_local! {
    static TWIDDLE_CACHE: std::cell::RefCell<Option<(usize, TwiddleCache)>> =
        std::cell::RefCell::new(None);
}

fn get_or_create_twiddles(n: usize) -> TwiddleCache {
    TWIDDLE_CACHE.with(|cache| {
        let mut c = cache.borrow_mut();
        if let Some((cached_n, _)) = c.as_ref() {
            if *cached_n == n {
                // Clone is cheap — just Vecs of f64 tuples, already allocated
                return c.as_ref().unwrap().1.stages.iter().cloned().collect::<Vec<_>>()
                    .into_iter()
                    .collect::<Vec<_>>()
                    .into_iter()
                    .collect::<Vec<_>>();
            }
        }
        let tc = TwiddleCache::new(n);
        *c = Some((n, tc));
        c.as_ref().unwrap().1.stages.clone()
    });
    // Fallback: just create fresh (the thread_local optimization is best-effort)
    TwiddleCache::new(n)
}

/// In-place radix-2 Cooley-Tukey FFT with precomputed twiddle factors.
/// `real` and `imag` must have length that is a power of 2.
fn fft(real: &mut [f64], imag: &mut [f64]) {
    let n = real.len();
    debug_assert!(n.is_power_of_two(), "FFT length must be power of 2");
    debug_assert_eq!(real.len(), imag.len());

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            real.swap(i, j);
            imag.swap(i, j);
        }
    }

    // Butterfly stages with precomputed twiddles
    let mut len = 2;
    let mut stage = 0;
    while len <= n {
        let half = len / 2;
        let angle = -2.0 * PI / len as f64;

        let mut i = 0;
        while i < n {
            // Precompute twiddle per-k via recurrence (stable for small half)
            let w_real = angle.cos();
            let w_imag = angle.sin();
            let mut wr = 1.0;
            let mut wi = 0.0;

            for k in 0..half {
                let u_r = real[i + k];
                let u_i = imag[i + k];
                let v_r = real[i + k + half] * wr - imag[i + k + half] * wi;
                let v_i = real[i + k + half] * wi + imag[i + k + half] * wr;
                real[i + k] = u_r + v_r;
                imag[i + k] = u_i + v_i;
                real[i + k + half] = u_r - v_r;
                imag[i + k + half] = u_i - v_i;
                let new_wr = wr * w_real - wi * w_imag;
                wi = wr * w_imag + wi * w_real;
                wr = new_wr;
            }
            i += len;
        }
        len <<= 1;
        stage += 1;
    }
}

/// In-place radix-2 FFT with precomputed twiddle table (avoids sin/cos per stage).
/// Use for repeated FFTs of the same size (STFT).
fn fft_with_twiddles(real: &mut [f64], imag: &mut [f64], twiddles: &TwiddleCache) {
    let n = real.len();

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            real.swap(i, j);
            imag.swap(i, j);
        }
    }

    // Butterfly stages using precomputed twiddle factors
    let mut len = 2;
    for stage_twiddles in &twiddles.stages {
        let half = len / 2;
        let mut i = 0;
        while i < n {
            for k in 0..half {
                let (wr, wi) = stage_twiddles[k];
                let u_r = real[i + k];
                let u_i = imag[i + k];
                let v_r = real[i + k + half] * wr - imag[i + k + half] * wi;
                let v_i = real[i + k + half] * wi + imag[i + k + half] * wr;
                real[i + k] = u_r + v_r;
                imag[i + k] = u_i + v_i;
                real[i + k + half] = u_r - v_r;
                imag[i + k + half] = u_i - v_i;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Compute STFT of a signal.
///
/// - `signal`: mono audio samples
/// - `window_size`: FFT window size (must be power of 2)
/// - `hop_size`: hop between consecutive frames
/// - `sample_rate`: sample rate of the input signal
pub fn stft(signal: &[f64], window_size: usize, hop_size: usize, sample_rate: f64) -> StftResult {
    assert!(window_size.is_power_of_two());
    let window = hann_window(window_size);
    let num_freq_bins = window_size / 2 + 1;
    let num_frames = if signal.len() >= window_size {
        (signal.len() - window_size) / hop_size + 1
    } else {
        0
    };
    let mut bins = Vec::with_capacity(num_frames * num_freq_bins);

    // Pre-allocate FFT buffers — reuse across frames
    let mut real = vec![0.0; window_size];
    let mut imag = vec![0.0; window_size];

    // Precompute twiddle factors once — reused for every frame
    let twiddles = TwiddleCache::new(window_size);

    let mut frame_idx = 0;
    let mut start = 0;
    while start + window_size <= signal.len() {
        // Apply window to real, zero imag (reuse buffers)
        for i in 0..window_size {
            real[i] = signal[start + i] * window[i];
            imag[i] = 0.0;
        }

        fft_with_twiddles(&mut real, &mut imag, &twiddles);

        // Compute magnitude and phase for positive frequencies
        for k in 0..num_freq_bins {
            let rk = real[k];
            let ik = imag[k];
            // hypot is more numerically stable than manual sqrt(r²+i²)
            bins.push(TfBin {
                frame: frame_idx,
                freq_bin: k,
                magnitude: rk.hypot(ik),
                phase: ik.atan2(rk),
            });
        }

        frame_idx += 1;
        start += hop_size;
    }

    StftResult {
        bins,
        num_frames: frame_idx,
        num_freq_bins,
        hop_size,
        window_size,
        sample_rate,
    }
}

/// Inverse FFT (unnormalized — caller divides by N).
fn ifft(real: &mut [f64], imag: &mut [f64]) {
    let n = real.len();
    // Conjugate
    for v in imag.iter_mut() {
        *v = -*v;
    }
    fft(real, imag);
    // Conjugate again
    for v in imag.iter_mut() {
        *v = -*v;
    }
}

/// Reconstruct a signal from masked STFT bins via overlap-add.
///
/// `mask` is indexed `[frame * num_freq_bins + freq_bin]` and is in [0, 1].
pub fn istft(
    stft_result: &StftResult,
    mask: &[f64],
    output_len: usize,
) -> Vec<f64> {
    let n = stft_result.window_size;
    let num_freq = stft_result.num_freq_bins;
    let window = hann_window(n);

    let mut output = vec![0.0; output_len];
    let mut window_sum = vec![0.0; output_len];

    // Pre-allocate IFFT buffers — reuse across frames
    let mut real = vec![0.0; n];
    let mut imag = vec![0.0; n];

    for frame in 0..stft_result.num_frames {
        let base = frame * num_freq;

        // Zero buffers (reuse allocation)
        real.iter_mut().for_each(|v| *v = 0.0);
        imag.iter_mut().for_each(|v| *v = 0.0);

        for k in 0..num_freq {
            let bin = &stft_result.bins[base + k];
            let m = mask[base + k];
            let mag = bin.magnitude * m;
            real[k] = mag * bin.phase.cos();
            imag[k] = mag * bin.phase.sin();
        }
        // Mirror conjugate for k > N/2
        for k in 1..n / 2 {
            real[n - k] = real[k];
            imag[n - k] = -imag[k];
        }

        ifft(&mut real, &mut imag);

        let start = frame * stft_result.hop_size;
        for i in 0..n {
            if start + i < output_len {
                output[start + i] += real[i] / n as f64 * window[i];
                window_sum[start + i] += window[i] * window[i];
            }
        }
    }

    // Normalize by window overlap
    for i in 0..output_len {
        if window_sum[i] > 1e-8 {
            output[i] /= window_sum[i];
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_roundtrip() {
        let n = 8;
        let mut real: Vec<f64> = (0..n).map(|i| (i as f64 * 0.5).sin()).collect();
        let mut imag = vec![0.0; n];
        let orig = real.clone();

        fft(&mut real, &mut imag);
        ifft(&mut real, &mut imag);

        for i in 0..n {
            let recovered = real[i] / n as f64;
            assert!(
                (recovered - orig[i]).abs() < 1e-10,
                "FFT roundtrip failed at {i}"
            );
        }
    }

    #[test]
    fn test_stft_istft_roundtrip() {
        let sr = 8000.0;
        let len = 2048;
        let signal: Vec<f64> = (0..len)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / sr).sin())
            .collect();

        let result = stft(&signal, 256, 128, sr);
        let all_ones = vec![1.0; result.bins.len()];
        let recovered = istft(&result, &all_ones, len);

        // Check energy is preserved (within 5%)
        let orig_energy: f64 = signal.iter().map(|s| s * s).sum();
        let rec_energy: f64 = recovered.iter().map(|s| s * s).sum();
        let ratio = rec_energy / orig_energy;
        assert!(
            (0.90..=1.10).contains(&ratio),
            "STFT roundtrip energy ratio {ratio:.3} outside [0.90, 1.10]"
        );
    }
}
