//! Quantized-inference bench: load TimesFM-200M at f32 / Q8_0 (int8) / Q4_0
//! (int4), and report each variant's decode latency and forecast error vs the
//! f32 model on the same input. Real weights only.
//!
//! ```ignore
//! cargo run -p timesfm --features candle --release --example quant_bench -- \
//!     /tmp/timesfm-parity/timesfm.safetensors
//! ```
//! Skips cleanly (exit 0) when weights are absent.

use std::time::Instant;

use candle_core::quantized::GgmlDType;
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use timesfm::config::TimesfmConfig;
use timesfm::model::PatchedTimeSeriesDecoder;

fn load(weights: &str, quant: Option<GgmlDType>) -> anyhow::Result<PatchedTimeSeriesDecoder> {
    let device = timesfm::select_device()?;
    let cfg = TimesfmConfig::timesfm_1p0_200m();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device)? };
    Ok(match quant {
        None => PatchedTimeSeriesDecoder::load(cfg, vb)?,
        Some(dt) => PatchedTimeSeriesDecoder::load_quantized(cfg, vb, dt)?,
    })
}

fn decode_point(
    model: &PatchedTimeSeriesDecoder,
    ctx: &Tensor,
    pad: &Tensor,
    freq: &Tensor,
    h: usize,
) -> anyhow::Result<Vec<f32>> {
    let (point, _full) = model.decode(ctx, pad, freq, h)?;
    Ok(point.i(0)?.to_vec1()?)
}

fn main() -> anyhow::Result<()> {
    let weights = std::env::args()
        .nth(1)
        .or_else(|| std::env::var("TIMESFM_WEIGHTS").ok())
        .unwrap_or_else(|| "/tmp/timesfm-parity/timesfm.safetensors".into());
    if !std::path::Path::new(&weights).exists() {
        eprintln!("SKIP quant_bench: weights missing ({weights}).");
        return Ok(());
    }

    let device = timesfm::select_device()?;
    let ctx_len = 512usize;
    let horizon = 128usize;
    // Deterministic input series.
    let series: Vec<f32> = (0..ctx_len)
        .map(|t| ((t as f32) / 24.0).sin() * 10.0 + 50.0 + 0.01 * t as f32)
        .collect();
    let ctx = Tensor::from_vec(series, (1, ctx_len), &device)?;
    let pad = Tensor::zeros((1, ctx_len), DType::F32, &device)?;
    let freq = Tensor::from_vec(vec![0u32], (1, 1), &device)?;

    // f32 reference.
    let m32 = load(&weights, None)?;
    let _ = decode_point(&m32, &ctx, &pad, &freq, horizon)?; // warm
    let t = Instant::now();
    let ref32 = decode_point(&m32, &ctx, &pad, &freq, horizon)?;
    let ms32 = t.elapsed().as_secs_f64() * 1000.0;
    let approx_f32_mb = 200.0; // 200M params × 4 bytes ≈ 800MB on disk; resident ≈ mmap.
    println!("f32     : {ms32:7.2} ms   (reference)   weights≈814 MB on disk");

    for (name, dt, bytes_per_w) in [
        ("Q8_0", GgmlDType::Q8_0, 1.06_f64),
        ("Q4_0", GgmlDType::Q4_0, 0.56_f64),
    ] {
        let m = load(&weights, Some(dt))?;
        let _ = decode_point(&m, &ctx, &pad, &freq, horizon)?; // warm
        let t = Instant::now();
        let q = decode_point(&m, &ctx, &pad, &freq, horizon)?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;

        let n_bad = q.iter().filter(|x| !x.is_finite()).count();
        let mut max_abs = 0f32;
        let mut sum_abs = 0f64;
        let mut scale = 1e-6f32;
        for (a, b) in q.iter().zip(ref32.iter()) {
            max_abs = max_abs.max((a - b).abs());
            sum_abs += (a - b).abs() as f64;
            scale = scale.max(b.abs());
        }
        let mae = sum_abs / horizon as f64;
        let approx_mb = approx_f32_mb * 4.0 * (bytes_per_w / 4.0); // vs f32 4 bytes/w
        println!(
            "{name:7}: {ms:7.2} ms   ({:.2}× vs f32)   MAE={mae:.3e}  max-abs={max_abs:.3e}  rel={:.3e}  weights≈{:.0} MB  {}",
            ms32 / ms,
            max_abs / scale,
            approx_mb,
            if n_bad == 0 { "finite-ok" } else { "NON-FINITE!" },
        );
        anyhow::ensure!(n_bad == 0, "{name} produced non-finite forecasts");
    }
    Ok(())
}
