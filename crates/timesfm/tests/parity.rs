//! Gated weight-parity integration test.
//!
//! Validates that the candle TimesFM 1.0 200M port reproduces the official
//! PyTorch reference forecast on a deterministic series, within a tight f32
//! tolerance. It is GATED on the converted artifacts existing locally:
//!
//!   * `/tmp/timesfm-parity/timesfm.safetensors` — produced by
//!     `scripts/convert_weights.py` from `google/timesfm-1.0-200m-pytorch`.
//!   * `/tmp/timesfm-parity/ref.json` — produced by the reference-generation
//!     script (official `PatchedTimeSeriesDecoder` forward).
//!
//! When either is absent (CI without the 814MB weights), the test prints a
//! skip notice and passes — it never fabricates a result. The full procedure
//! to regenerate both artifacts is documented in `examples/parity.rs`.
//!
//! Measured locally (2026-06-24, ruvultra, f32 CPU): max-abs-diff 8.58e-6,
//! MAE 3.25e-6, rel-error 5.83e-7 over a horizon of 128.

#![cfg(feature = "candle")]

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use serde_json::Value;
use timesfm::config::TimesfmConfig;
use timesfm::model::PatchedTimeSeriesDecoder;

const WEIGHTS: &str = "/tmp/timesfm-parity/timesfm.safetensors";
const REF: &str = "/tmp/timesfm-parity/ref.json";

#[test]
fn timesfm_1p0_200m_weight_parity() -> anyhow::Result<()> {
    if !std::path::Path::new(WEIGHTS).exists() || !std::path::Path::new(REF).exists() {
        eprintln!(
            "SKIP timesfm parity: artifacts missing ({WEIGHTS} / {REF}). \
             Generate via scripts/convert_weights.py + the reference script \
             (see examples/parity.rs). Not fabricating a pass."
        );
        return Ok(());
    }

    let device = Device::Cpu;
    let refj: Value = serde_json::from_str(&std::fs::read_to_string(REF)?)?;
    let context_len = refj["config"]["context_len"].as_u64().unwrap() as usize;
    let horizon = refj["config"]["horizon"].as_u64().unwrap() as usize;
    let freq_id = refj["config"]["freq_id"].as_u64().unwrap() as u32;

    let f32s = |v: &Value| -> Vec<f32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    };
    let input_series = f32s(&refj["input_series"]);
    let ref_point = f32s(&refj["point_forecast"]);
    assert_eq!(input_series.len(), context_len);
    assert_eq!(ref_point.len(), horizon);

    let cfg = TimesfmConfig::timesfm_1p0_200m();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[WEIGHTS], DType::F32, &device)? };
    let model = PatchedTimeSeriesDecoder::load(cfg, vb)?;

    let input_ts = Tensor::from_vec(input_series, (1, context_len), &device)?;
    let input_padding = Tensor::zeros((1, context_len), DType::F32, &device)?;
    let freq = Tensor::from_vec(vec![freq_id], (1, 1), &device)?;

    let (point, _full) = model.decode(&input_ts, &input_padding, &freq, horizon)?;
    let candle_point: Vec<f32> = point.i(0)?.to_vec1()?;

    // Honesty guard: any non-finite output is an outright failure.
    let n_bad = candle_point.iter().filter(|x| !x.is_finite()).count();
    assert_eq!(
        n_bad, 0,
        "candle produced {n_bad} non-finite forecast values"
    );

    let mut max_abs = 0f32;
    let mut sum_abs = 0f64;
    for (c, r) in candle_point.iter().zip(ref_point.iter()) {
        let d = (c - r).abs();
        max_abs = max_abs.max(d);
        sum_abs += d as f64;
    }
    let mae = sum_abs / horizon as f64;
    eprintln!("timesfm parity: max-abs-diff={max_abs:.3e} MAE={mae:.3e}");

    // Target was <1e-2; we measure ~1e-5. Assert the stricter 1e-3 to catch
    // any future numerical regression while allowing f32 accumulation noise.
    assert!(
        max_abs < 1e-3,
        "weight parity FAILED: max-abs-diff {max_abs:.3e} >= 1e-3"
    );
    Ok(())
}
