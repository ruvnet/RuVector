//! Forward-only latency bench for TimesFM-200M (real weights). Loads once, times N decodes.
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::time::Instant;
use timesfm::config::TimesfmConfig;
use timesfm::model::PatchedTimeSeriesDecoder;
fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let weights = std::env::var("TIMESFM_WEIGHTS")
        .unwrap_or("/tmp/timesfm-parity/timesfm.safetensors".into());
    let cfg = TimesfmConfig::timesfm_1p0_200m();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device)? };
    let model = PatchedTimeSeriesDecoder::load(cfg, vb)?;
    let ctx = 512usize;
    let horizon = 128usize;
    let input_ts = Tensor::randn(0f32, 1f32, (1, ctx), &device)?;
    let pad = Tensor::zeros((1, ctx), DType::F32, &device)?;
    let freq = Tensor::from_vec(vec![0u32], (1, 1), &device)?;
    for _ in 0..3 {
        let _ = model.decode(&input_ts, &pad, &freq, horizon)?;
    }
    let n = 30;
    let t = Instant::now();
    for _ in 0..n {
        let _ = model.decode(&input_ts, &pad, &freq, horizon)?;
    }
    let per = t.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let thr = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(0);
    println!(
        "TimesFM-200M decode(ctx=512,h=128) CPU: {:.2} ms/forecast  (mean of {} iters, {} threads)",
        per, n, thr
    );
    Ok(())
}
