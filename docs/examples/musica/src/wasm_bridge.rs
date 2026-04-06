//! WASM/C-FFI bridge for the Musica audio separation pipeline.
//!
//! Exposes the separation pipeline as `extern "C"` functions callable from
//! JavaScript via WebAssembly. The FFI surface is feature-gated behind
//! `#[cfg(feature = "wasm")]` so it does not affect the normal library build.
//!
//! # Building for WASM
//!
//! ```sh
//! cargo build --target wasm32-unknown-unknown --features wasm --release
//! ```

use crate::audio_graph::{build_audio_graph, GraphParams};
use crate::separator::{separate, SeparatorConfig};
use crate::stft;

// ---------------------------------------------------------------------------
// Internal helpers (always compiled so tests work without the `wasm` feature)
// ---------------------------------------------------------------------------

/// Run the full separation pipeline on raw audio samples and return interleaved
/// mask data: `[source0_mask..., source1_mask..., ...]`.
///
/// Each mask has length `num_frames * num_freq_bins` as produced by the STFT.
/// The total returned length is `num_sources * num_frames * num_freq_bins`.
fn run_pipeline(samples: &[f64], sample_rate: f64, num_sources: usize) -> (Vec<f64>, u64) {
    let start = std::time::Instant::now();

    let window_size = 256usize;
    let hop_size = 128usize;

    let stft_result = stft::stft(samples, window_size, hop_size, sample_rate);
    let graph = build_audio_graph(&stft_result, &GraphParams::default());

    let config = SeparatorConfig {
        num_sources,
        ..SeparatorConfig::default()
    };

    let result = separate(&graph, &config);

    // Interleave masks: [mask0..., mask1..., ...]
    let mask_len = result.masks.first().map_or(0, |m| m.len());
    let mut out = Vec::with_capacity(num_sources * mask_len);
    for mask in &result.masks {
        out.extend_from_slice(mask);
    }

    let elapsed_us = start.elapsed().as_micros() as u64;
    (out, elapsed_us)
}

// ---------------------------------------------------------------------------
// FFI surface (only compiled with `--features wasm`)
// ---------------------------------------------------------------------------

#[cfg(feature = "wasm")]
mod ffi {
    use super::run_pipeline;

    /// Length of the last result returned by `separate_audio`.
    static mut LAST_RESULT_LEN: usize = 0;
    /// Latency (microseconds) of the last `separate_audio` call.
    static mut LAST_LATENCY_US: u64 = 0;

    /// Run the audio separation pipeline.
    ///
    /// # Parameters
    /// - `ptr`  — pointer to `f64` audio samples (mono)
    /// - `len`  — number of samples
    /// - `sample_rate` — sample rate in Hz (e.g. 44100.0)
    /// - `num_sources` — number of sources to separate into (2-4)
    ///
    /// # Returns
    /// Pointer to interleaved mask data. Caller must free with `free_result`.
    /// Use `get_result_len` to discover the length.
    #[no_mangle]
    pub unsafe extern "C" fn separate_audio(
        ptr: *const f64,
        len: usize,
        sample_rate: f64,
        num_sources: usize,
    ) -> *mut f64 {
        if ptr.is_null() || len == 0 || num_sources == 0 {
            LAST_RESULT_LEN = 0;
            LAST_LATENCY_US = 0;
            return std::ptr::null_mut();
        }

        let samples = std::slice::from_raw_parts(ptr, len);
        let (result, latency) = run_pipeline(samples, sample_rate, num_sources);

        LAST_RESULT_LEN = result.len();
        LAST_LATENCY_US = latency;

        let boxed = result.into_boxed_slice();
        Box::into_raw(boxed) as *mut f64
    }

    /// Return the length (in `f64` elements) of the last result.
    #[no_mangle]
    pub extern "C" fn get_result_len() -> usize {
        unsafe { LAST_RESULT_LEN }
    }

    /// Free a result buffer previously returned by `separate_audio`.
    #[no_mangle]
    pub unsafe extern "C" fn free_result(ptr: *mut f64) {
        if ptr.is_null() {
            return;
        }
        let len = LAST_RESULT_LEN;
        if len > 0 {
            let _ = Box::from_raw(std::slice::from_raw_parts_mut(ptr, len));
        }
    }

    /// Return the wall-clock latency in microseconds of the last call.
    #[no_mangle]
    pub extern "C" fn get_latency_us() -> u64 {
        unsafe { LAST_LATENCY_US }
    }
}

// ---------------------------------------------------------------------------
// Tests (always compiled — they exercise `run_pipeline`, not FFI)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn make_tone(sr: f64, dur: f64, freq: f64) -> Vec<f64> {
        let n = (sr * dur) as usize;
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / sr).sin())
            .collect()
    }

    #[test]
    fn test_pipeline_returns_correct_shape() {
        let sr = 8000.0;
        let signal = make_tone(sr, 0.25, 440.0);
        let num_sources = 2;

        let (masks, latency_us) = run_pipeline(&signal, sr, num_sources);

        // The mask length must be a multiple of num_sources
        assert_eq!(masks.len() % num_sources, 0, "mask length not divisible by num_sources");
        // Should have non-zero output
        assert!(!masks.is_empty(), "pipeline returned empty masks");
        // Latency should be recorded
        assert!(latency_us > 0, "latency not recorded");
    }

    #[test]
    fn test_pipeline_masks_sum_to_one() {
        let sr = 8000.0;
        let signal: Vec<f64> = {
            let n = (sr * 0.25) as usize;
            (0..n)
                .map(|i| {
                    let t = i as f64 / sr;
                    (2.0 * PI * 300.0 * t).sin() + (2.0 * PI * 1800.0 * t).sin()
                })
                .collect()
        };
        let num_sources = 2;

        let (masks, _) = run_pipeline(&signal, sr, num_sources);
        let per_source = masks.len() / num_sources;

        // Check that masks sum to ~1.0 at each TF point
        for i in 0..per_source.min(200) {
            let sum: f64 = (0..num_sources).map(|s| masks[s * per_source + i]).sum();
            assert!(
                (sum - 1.0).abs() < 0.05,
                "mask sum at index {i} = {sum:.4}, expected ~1.0"
            );
        }
    }
}
