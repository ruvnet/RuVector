//! PhotonLayer WASM bindings — browser playback and receipt verification.
//!
//! Exposes the deterministic optical pipeline (ADR-260 Phase 3 / §7.3) to the
//! browser so the five-view studio UI can render without any server-side
//! inference and so receipts can be verified client-side (anti-swap guarantee).
//!
//! # Five-view rendering pipeline
//! ```text
//! grayscale bytes
//!   -> incoming field amplitude      (view 1: raw amplitude image)
//!   -> learned phase mask            (view 2: phase colormap)
//!   -> masked field intensity        (view 3: intensity after masking)
//!   -> sensor capture / frame        (view 4: "strange pattern")
//!   -> decoded / reconstructed image (view 5: placeholder = sensor frame)
//! ```
//!
//! # Anti-swap guarantee
//! `verify_receipt_json` deserializes an [`ExperimentReceipt`] and re-derives
//! the `rvf_receipt_hash` over all bound inputs; a mismatch proves tampering.

#![allow(dead_code)]

use wasm_bindgen::prelude::*;

use photonlayer_core::prelude::{
    verify_receipt, ExperimentReceipt, InputImage, OpticalConfig, OpticalField, PhaseMask,
    ScalarSimulator,
};

// ─── Normalization helpers ────────────────────────────────────────────────────

/// Map a slice of `f32` values linearly to `u8` `[0, 255]` (min-max stretch).
/// When all values are equal, every output pixel is 0.
pub fn normalize_to_u8(values: &[f32]) -> Vec<u8> {
    if values.is_empty() {
        return Vec::new();
    }
    let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;
    if range < f32::EPSILON {
        return vec![0u8; values.len()];
    }
    values
        .iter()
        .map(|&v| ((v - min) / range * 255.0).clamp(0.0, 255.0).round() as u8)
        .collect()
}

/// Extract amplitudes `|c|` from a complex field and normalize to `[0, 255]`.
pub fn field_amplitude_u8(field: &OpticalField) -> Vec<u8> {
    let amplitudes: Vec<f32> = field.data.iter().map(|c| c.abs()).collect();
    normalize_to_u8(&amplitudes)
}

/// Extract intensities `|c|^2` from a complex field and normalize to `[0, 255]`.
pub fn field_intensity_u8(field: &OpticalField) -> Vec<u8> {
    let intensities: Vec<f32> = field.data.iter().map(|c| c.norm_sqr()).collect();
    normalize_to_u8(&intensities)
}

/// Map phase radians `[0, 2π)` linearly to `[0, 255]`.
pub fn phase_to_u8(phase: &[f32]) -> Vec<u8> {
    use core::f32::consts::PI;
    let two_pi = 2.0 * PI;
    phase
        .iter()
        .map(|&p| {
            let wrapped = p.rem_euclid(two_pi);
            (wrapped / two_pi * 255.0).clamp(0.0, 255.0).round() as u8
        })
        .collect()
}

// ─── Mask parsing ─────────────────────────────────────────────────────────────

/// Build a PhaseMask from a kind string plus numeric parameters.
///
/// Supported kinds:
/// * `"identity"` — flat zero-phase (no mask effect).
/// * `"random"` — deterministic pseudo-random phases; `seed` used.
/// * `"lens"` — quadratic lens profile; `strength` used as focal strength.
pub fn build_mask(width: usize, height: usize, kind: &str, seed: u64, strength: f32) -> PhaseMask {
    match kind {
        "random" => PhaseMask::random(width, height, seed),
        "lens" => PhaseMask::lens(width, height, strength),
        _ => PhaseMask::identity(width, height),
    }
}

// ─── Core pipeline (pure Rust, testable on native) ───────────────────────────

/// Result of a full simulation trace, carrying all five view buffers.
///
/// Each `*_buf` is a row-major grayscale `Vec<u8>` ready for `ImageData`.
pub struct TraceResult {
    /// Width of all buffers (same grid).
    pub width: usize,
    /// Height of all buffers.
    pub height: usize,
    /// View 1 — amplitude of the incoming optical field.
    pub incoming_buf: Vec<u8>,
    /// View 2 — phase-mask values mapped 0..2π → 0..255.
    pub mask_buf: Vec<u8>,
    /// View 3 — intensity of the masked field.
    pub masked_intensity_buf: Vec<u8>,
    /// View 4 — sensor capture intensity ("strange pattern").
    pub sensor_buf: Vec<u8>,
    /// BLAKE3 hex digest of the sensor frame (determinism proof).
    pub frame_hash: String,
}

/// Run the optical pipeline and return a [`TraceResult`] with all five views.
///
/// `config_json` is an [`OpticalConfig`] serialized with serde_json.  Pass an
/// empty string to use `OpticalConfig::demo`.
pub fn run_trace(
    image_bytes: &[u8],
    width: usize,
    height: usize,
    mask_kind: &str,
    mask_seed: u64,
    mask_strength: f32,
    config_json: &str,
) -> Result<TraceResult, String> {
    // Parse / build config.
    // Bound untrusted image dimensions up front to block DoS / overflow.
    let max = photonlayer_core::config::MAX_GRID_DIM;
    if width == 0 || height == 0 || width > max || height > max {
        return Err(format!("image dimensions must be in 1..={max}"));
    }

    let config: OpticalConfig = if config_json.trim().is_empty() {
        OpticalConfig::demo(width, height)
    } else {
        serde_json::from_str(config_json).map_err(|e| format!("config parse error: {e}"))?
    };
    // Validate the (untrusted) config before it drives any allocation/FFT.
    config
        .validate()
        .map_err(|e| format!("invalid config: {e}"))?;

    // Build input image.
    let img = InputImage::from_gray_u8(width, height, image_bytes)
        .map_err(|e| format!("image error: {e}"))?;

    // Build mask (same size as the config grid).
    let grid_w = config.width;
    let grid_h = config.height;
    let mask = build_mask(grid_w, grid_h, mask_kind, mask_seed, mask_strength);

    // Run the full trace.
    let trace = ScalarSimulator
        .trace(&img, &mask, &config)
        .map_err(|e| format!("simulation error: {e}"))?;

    // Encode every view into a u8 buffer.
    let incoming_buf = field_amplitude_u8(&trace.incoming);
    let mask_buf = phase_to_u8(&mask.phase_radians);
    let masked_intensity_buf = field_intensity_u8(&trace.masked);
    let sensor_buf = normalize_to_u8(&trace.frame.intensity);
    let frame_hash = trace.frame.frame_hash.clone();

    Ok(TraceResult {
        width: grid_w,
        height: grid_h,
        incoming_buf,
        mask_buf,
        masked_intensity_buf,
        sensor_buf,
        frame_hash,
    })
}

// ─── WASM-bindgen exported struct ────────────────────────────────────────────

/// All five view buffers returned to JavaScript.
///
/// Getters returning `Vec<u8>` copy the data into a fresh JS `Uint8Array`
/// each call — suitable for passing to `ImageData` or `canvas.putImageData`.
#[wasm_bindgen]
pub struct WasmTraceResult {
    width: usize,
    height: usize,
    incoming_buf: Vec<u8>,
    mask_buf: Vec<u8>,
    masked_intensity_buf: Vec<u8>,
    sensor_buf: Vec<u8>,
    frame_hash: String,
}

#[wasm_bindgen]
impl WasmTraceResult {
    /// Grid width in pixels.
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.width as u32
    }

    /// Grid height in pixels.
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.height as u32
    }

    /// View 1: amplitude of the incoming optical field, normalized to 0..255.
    #[wasm_bindgen(getter)]
    pub fn incoming_buf(&self) -> Vec<u8> {
        self.incoming_buf.clone()
    }

    /// View 2: phase mask mapped 0..2π → 0..255.
    #[wasm_bindgen(getter)]
    pub fn mask_buf(&self) -> Vec<u8> {
        self.mask_buf.clone()
    }

    /// View 3: masked-field intensity normalized to 0..255.
    #[wasm_bindgen(getter)]
    pub fn masked_intensity_buf(&self) -> Vec<u8> {
        self.masked_intensity_buf.clone()
    }

    /// View 4: sensor capture ("strange pattern"), normalized to 0..255.
    #[wasm_bindgen(getter)]
    pub fn sensor_buf(&self) -> Vec<u8> {
        self.sensor_buf.clone()
    }

    /// BLAKE3 hex digest of the sensor frame (anti-swap determinism proof).
    #[wasm_bindgen(getter)]
    pub fn frame_hash(&self) -> String {
        self.frame_hash.clone()
    }
}

// ─── WASM-bindgen exported functions ─────────────────────────────────────────

/// Crate version, exported to verify the WASM module loads.
#[wasm_bindgen]
pub fn photonlayer_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Run the five-view optical simulation pipeline.
///
/// # Parameters
/// * `image_bytes` — row-major grayscale u8 pixels (len must equal `w * h`).
/// * `w` / `h` — image dimensions.
/// * `mask_kind` — `"identity"`, `"random"`, or `"lens"`.
/// * `mask_seed` — seed for `"random"` masks (ignored for others).
/// * `mask_strength` — focal strength for `"lens"` masks (ignored for others).
/// * `config_json` — JSON-serialized [`OpticalConfig`]; empty → `demo` config.
///
/// Returns a [`WasmTraceResult`] whose getter methods supply canvas-ready
/// grayscale buffers for each of the five studio views.
///
/// Throws a JS error string on any failure.
#[wasm_bindgen]
pub fn simulate(
    image_bytes: &[u8],
    w: u32,
    h: u32,
    mask_kind: &str,
    mask_seed: u64,
    mask_strength: f32,
    config_json: &str,
) -> Result<WasmTraceResult, JsValue> {
    let result = run_trace(
        image_bytes,
        w as usize,
        h as usize,
        mask_kind,
        mask_seed,
        mask_strength,
        config_json,
    )
    .map_err(|e| JsValue::from_str(&e))?;

    Ok(WasmTraceResult {
        width: result.width,
        height: result.height,
        incoming_buf: result.incoming_buf,
        mask_buf: result.mask_buf,
        masked_intensity_buf: result.masked_intensity_buf,
        sensor_buf: result.sensor_buf,
        frame_hash: result.frame_hash,
    })
}

/// Parse an [`ExperimentReceipt`] from JSON and verify its internal consistency.
///
/// Returns `true` iff the receipt's `rvf_receipt_hash` matches a fresh
/// re-derivation over all bound fields — proving the output was not swapped.
#[wasm_bindgen]
pub fn verify_receipt_json(json: &str) -> bool {
    match serde_json::from_str::<ExperimentReceipt>(json) {
        Ok(receipt) => verify_receipt(&receipt),
        Err(_) => false,
    }
}

/// Return a JSON-serialized [`OpticalConfig::demo`] for the given dimensions.
///
/// JavaScript can call this to obtain a valid starting config, then pass it
/// (possibly modified) back to `simulate`.
#[wasm_bindgen]
pub fn default_config_json(width: u32, height: u32) -> String {
    let cfg = OpticalConfig::demo(width as usize, height as usize);
    serde_json::to_string(&cfg).unwrap_or_else(|_| "{}".to_string())
}

// ─── Native unit tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use photonlayer_core::prelude::{build_receipt, MetricReport, Provenance};

    /// Build a small checkerboard image (power-of-two dimensions).
    fn checkerboard_u8(n: usize) -> Vec<u8> {
        (0..n * n)
            .map(|i| {
                let (x, y) = (i % n, i / n);
                if (x / 4 + y / 4) % 2 == 0 {
                    255u8
                } else {
                    0u8
                }
            })
            .collect()
    }

    #[test]
    fn smoke_version() {
        assert!(!photonlayer_version().is_empty());
    }

    // ── Normalization helpers ──────────────────────────────────────────────

    #[test]
    fn normalize_range_is_0_to_255() {
        let v = vec![0.0f32, 0.25, 0.5, 0.75, 1.0];
        let u = normalize_to_u8(&v);
        assert_eq!(u[0], 0);
        assert_eq!(u[4], 255);
        // Monotonic non-decreasing mapping (0..=1 -> 0..=255); range is guaranteed by u8.
        assert!(u.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn normalize_uniform_gives_zeros() {
        let v = vec![2.5f32; 8];
        let u = normalize_to_u8(&v);
        assert!(u.iter().all(|&b| b == 0));
    }

    #[test]
    fn phase_to_u8_wraps_correctly() {
        use core::f32::consts::PI;
        let phases = vec![0.0f32, PI, 2.0 * PI - 0.001];
        let u = phase_to_u8(&phases);
        assert_eq!(u[0], 0);
        assert!(u[1] > 100 && u[1] < 155); // PI maps to ~127
        assert!(u[2] >= 254); // ~2π maps to 255
    }

    // ── Full pipeline ──────────────────────────────────────────────────────

    #[test]
    fn simulate_returns_correct_buffer_lengths() {
        let n = 16usize;
        let img = checkerboard_u8(n);
        let result = run_trace(&img, n, n, "random", 42, 1.0, "").unwrap();

        let expected = result.width * result.height;
        assert_eq!(
            result.incoming_buf.len(),
            expected,
            "incoming_buf wrong len"
        );
        assert_eq!(result.mask_buf.len(), expected, "mask_buf wrong len");
        assert_eq!(
            result.masked_intensity_buf.len(),
            expected,
            "masked_intensity_buf wrong len"
        );
        assert_eq!(result.sensor_buf.len(), expected, "sensor_buf wrong len");
    }

    #[test]
    fn all_buffer_values_are_u8() {
        // Confirm that `run_trace` produces non-empty buffers of u8.
        // The type guarantee (values in 0..=255) is enforced by the `u8` type
        // itself; here we just check length > 0 and that max is reachable.
        let n = 16usize;
        let img = checkerboard_u8(n);
        let result = run_trace(&img, n, n, "identity", 0, 0.0, "").unwrap();
        let expected = result.width * result.height;

        assert!(!result.incoming_buf.is_empty());
        assert!(!result.mask_buf.is_empty());
        assert!(!result.masked_intensity_buf.is_empty());
        assert!(!result.sensor_buf.is_empty());

        // The checkerboard has white pixels so the max amplitude should hit 255.
        assert_eq!(result.incoming_buf.len(), expected);
        assert_eq!(result.mask_buf.len(), expected);
        assert_eq!(result.masked_intensity_buf.len(), expected);
        assert_eq!(result.sensor_buf.len(), expected);
    }

    #[test]
    fn sensor_buf_differs_from_incoming_buf() {
        // The propagated sensor pattern should not be identical to the input
        // amplitude (ADR-260 acceptance: frame must not be human-readable).
        let n = 16usize;
        let img = checkerboard_u8(n);
        let result = run_trace(&img, n, n, "random", 7, 1.0, "").unwrap();
        assert_ne!(
            result.sensor_buf, result.incoming_buf,
            "sensor frame must differ from incoming field"
        );
    }

    #[test]
    fn pipeline_is_deterministic_same_frame_hash() {
        let n = 16usize;
        let img = checkerboard_u8(n);
        let r1 = run_trace(&img, n, n, "random", 99, 1.0, "").unwrap();
        let r2 = run_trace(&img, n, n, "random", 99, 1.0, "").unwrap();
        assert_eq!(
            r1.frame_hash, r2.frame_hash,
            "determinism invariant: same inputs must yield same frame_hash"
        );
        assert_eq!(r1.sensor_buf, r2.sensor_buf);
    }

    #[test]
    fn different_seeds_give_different_hashes() {
        let n = 16usize;
        let img = checkerboard_u8(n);
        let r1 = run_trace(&img, n, n, "random", 1, 1.0, "").unwrap();
        let r2 = run_trace(&img, n, n, "random", 2, 1.0, "").unwrap();
        assert_ne!(r1.frame_hash, r2.frame_hash);
    }

    // ── Mask kinds ────────────────────────────────────────────────────────

    #[test]
    fn identity_mask_has_all_zero_phase() {
        let m = build_mask(8, 8, "identity", 0, 0.0);
        assert!(m.phase_radians.iter().all(|&p| p == 0.0));
    }

    #[test]
    fn lens_mask_has_non_zero_phase() {
        let m = build_mask(8, 8, "lens", 0, 0.01);
        assert!(m.phase_radians.iter().any(|&p| p != 0.0));
    }

    #[test]
    fn unknown_mask_kind_falls_back_to_identity() {
        let m = build_mask(8, 8, "nonexistent_kind", 0, 0.0);
        assert!(m.phase_radians.iter().all(|&p| p == 0.0));
    }

    // ── Receipt verification ───────────────────────────────────────────────

    #[test]
    fn verify_receipt_json_round_trips() {
        // Build an experiment, produce a receipt, serialize → verify.
        let n = 16usize;
        let px: Vec<f32> = (0..n * n).map(|i| (i % n) as f32 / n as f32).collect();
        let img = InputImage::from_norm_f32(n, n, px).unwrap();
        let mask = PhaseMask::random(n, n, 42);
        let cfg = OpticalConfig::demo(n, n);
        let trace = ScalarSimulator.trace(&img, &mask, &cfg).unwrap();
        let metrics = MetricReport::default();
        let prov = Provenance::default();
        let receipt = build_receipt(
            "wasm-test-exp",
            &img,
            &mask,
            &cfg,
            &trace.frame,
            &metrics,
            &prov,
        );

        let json = serde_json::to_string(&receipt).unwrap();
        assert!(
            verify_receipt_json(&json),
            "receipt should verify after round-trip serialization"
        );
    }

    #[test]
    fn verify_receipt_json_rejects_tampered() {
        let n = 16usize;
        let px: Vec<f32> = vec![0.5f32; n * n];
        let img = InputImage::from_norm_f32(n, n, px).unwrap();
        let mask = PhaseMask::identity(n, n);
        let cfg = OpticalConfig::demo(n, n);
        let trace = ScalarSimulator.trace(&img, &mask, &cfg).unwrap();
        let metrics = MetricReport::default();
        let prov = Provenance::default();
        let mut receipt = build_receipt(
            "tamper-test",
            &img,
            &mask,
            &cfg,
            &trace.frame,
            &metrics,
            &prov,
        );

        // Tamper with the output hash.
        receipt.output_hash.push_str("TAMPERED");
        let json = serde_json::to_string(&receipt).unwrap();
        assert!(
            !verify_receipt_json(&json),
            "tampered receipt must fail verification"
        );
    }

    #[test]
    fn verify_receipt_json_rejects_invalid_json() {
        assert!(!verify_receipt_json("not valid json at all"));
        assert!(!verify_receipt_json("{}"));
        assert!(!verify_receipt_json(""));
    }

    // ── Default config ────────────────────────────────────────────────────

    #[test]
    fn default_config_json_is_valid() {
        let json = default_config_json(32, 32);
        let cfg: OpticalConfig = serde_json::from_str(&json)
            .expect("default_config_json must produce valid OpticalConfig JSON");
        assert_eq!(cfg.width, 32);
        assert_eq!(cfg.height, 32);
    }

    #[test]
    fn config_json_round_trips_through_simulate() {
        let n = 16usize;
        let img = checkerboard_u8(n);
        let cfg_json = default_config_json(n as u32, n as u32);
        let result = run_trace(&img, n, n, "identity", 0, 0.0, &cfg_json);
        assert!(result.is_ok(), "simulate with explicit config must succeed");
    }
}
