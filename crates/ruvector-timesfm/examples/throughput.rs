//! Batched throughput bench: forecast B series sequentially vs. in one batched
//! model call, and verify the batched path matches the per-series path.
//!
//! ```ignore
//! cargo run -p ruvector-timesfm --features candle --release --example throughput \
//!     -- /tmp/timesfm-parity/timesfm.safetensors
//! # GPU:
//! TIMESFM_DEVICE=cuda cargo run -p ruvector-timesfm --features cuda --release \
//!     --example throughput -- /tmp/timesfm-parity/timesfm.safetensors
//! ```
//! Skips cleanly (exit 0) when weights are absent.

use std::time::Instant;

use ruvector_timesfm::Forecaster;

fn synth(seed: usize, n: usize) -> Vec<f32> {
    (0..n)
        .map(|t| {
            let phase = seed as f32 * 0.7;
            50.0 + 12.0 * ((t as f32 / 16.0) + phase).sin() + 0.03 * t as f32
        })
        .collect()
}

fn main() -> anyhow::Result<()> {
    let weights = std::env::args()
        .nth(1)
        .or_else(|| std::env::var("TIMESFM_WEIGHTS").ok())
        .unwrap_or_else(|| "/tmp/timesfm-parity/timesfm.safetensors".into());
    if !std::path::Path::new(&weights).exists() {
        eprintln!("SKIP throughput: weights missing ({weights}).");
        return Ok(());
    }

    let device = timesfm::select_device()?;
    let prec = std::env::var("TIMESFM_PRECISION").unwrap_or_else(|_| "f32".into());
    let dev_label = format!(
        "{}/{prec}",
        std::env::var("TIMESFM_DEVICE").unwrap_or_else(|_| "cpu".into())
    );
    let f = if prec == "f16" {
        Forecaster::load_f16(&weights, device)?
    } else {
        Forecaster::load(&weights, device)?
    };

    let batch_size = 32usize;
    let ctx = 256usize;
    let horizon = 64usize;
    let series: Vec<Vec<f32>> = (0..batch_size).map(|s| synth(s, ctx)).collect();

    // Warm up.
    let _ = f.forecast_batch(&series, horizon, 0)?;

    // Sequential.
    let t = Instant::now();
    let seq: Vec<_> = series
        .iter()
        .map(|s| f.forecast(s, horizon))
        .collect::<Result<_, _>>()?;
    let seq_ms = t.elapsed().as_secs_f64() * 1000.0;

    // Batched.
    let t = Instant::now();
    let batched = f.forecast_batch(&series, horizon, 0)?;
    let batch_ms = t.elapsed().as_secs_f64() * 1000.0;

    // Correctness: batched must match sequential. On CPU this is bit-exact; on
    // GPU, batched vs per-row matmuls reduce in a different order, so compare
    // with a *relative* tolerance (scaled by the series magnitude) rather than a
    // tight absolute one.
    let mut max_abs = 0f32;
    let mut scale = 1e-6f32;
    for (a, b) in seq.iter().zip(batched.iter()) {
        for (x, y) in a.point.iter().zip(b.point.iter()) {
            max_abs = max_abs.max((x - y).abs());
            scale = scale.max(x.abs());
        }
    }
    let rel = max_abs / scale;

    println!(
        "throughput [{dev_label}] B={batch_size} ctx={ctx} h={horizon}:\n  \
         sequential: {seq_ms:8.2} ms total = {:7.0} forecasts/s\n  \
         batched:    {batch_ms:8.2} ms total = {:7.0} forecasts/s   ({:.2}x)\n  \
         batched-vs-sequential max-abs-diff = {max_abs:.3e} (rel {rel:.3e})",
        batch_size as f64 / (seq_ms / 1000.0),
        batch_size as f64 / (batch_ms / 1000.0),
        seq_ms / batch_ms,
    );

    // Relative tolerance: bit-exact on CPU (~0), GPU reduction-order ~1e-4 rel.
    assert!(
        rel < 1e-3,
        "batched path diverged from sequential: rel {rel:.3e} (abs {max_abs:.3e})"
    );
    Ok(())
}
