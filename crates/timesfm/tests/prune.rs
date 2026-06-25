//! Gated integration test for predictive pruning (ADR-191 §2).
//!
//! Demonstrates that [`timesfm::prune::decide_prune`] makes the correct call on
//! two synthetic optimization curves (lower = better):
//!
//!   (a) a curve that plateaus HIGH (doomed) ⇒ PRUNE,
//!   (b) a curve still improving toward a low floor ⇒ CONTINUE.
//!
//! GATED on the converted weights at `/tmp/timesfm-parity/timesfm.safetensors`.
//! When absent (CI without the 814MB weights), it prints a skip notice and
//! passes — it never fabricates a result.

#![cfg(feature = "candle")]

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use timesfm::config::TimesfmConfig;
use timesfm::model::PatchedTimeSeriesDecoder;
use timesfm::prune::decide_prune;

const WEIGHTS: &str = "/tmp/timesfm-parity/timesfm.safetensors";

fn exp_decay(k: usize, y0: f32, floor: f32, tau: f32) -> Vec<f32> {
    (0..k)
        .map(|t| floor + (y0 - floor) * (-(t as f32) / tau).exp())
        .collect()
}

#[test]
fn predictive_prune_doomed_vs_healthy() -> anyhow::Result<()> {
    if !std::path::Path::new(WEIGHTS).exists() {
        eprintln!(
            "SKIP timesfm prune: weights missing ({WEIGHTS}). \
             Generate via scripts/convert_weights.py. Not fabricating a pass."
        );
        return Ok(());
    }

    let device = Device::Cpu;
    let cfg = TimesfmConfig::timesfm_1p0_200m();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[WEIGHTS], DType::F32, &device)? };
    let model = PatchedTimeSeriesDecoder::load(cfg, vb)?;

    let k = 128usize;
    let target = 1000usize;
    let threshold = 0.05f32;

    // (a) DOOMED: plateaus HIGH (floor 0.20, best-so-far never reaches 0.05) ⇒
    //     forecast plateau lands far above threshold ⇒ PRUNE.
    let doomed = exp_decay(k, 0.95, 0.20, 16.0);
    let d_doomed = decide_prune(&model, &doomed, target, threshold, &device)?;
    eprintln!(
        "doomed:  plateau={:.4} best={:.4} conf={:.3} prune={}",
        d_doomed.forecast_plateau, d_doomed.best_so_far, d_doomed.confidence, d_doomed.prune
    );

    // (b) HEALTHY: already drove exploitability below 0.05 by iter K (floor
    //     0.005) ⇒ already-viable guard ⇒ CONTINUE.
    let healthy = exp_decay(k, 0.95, 0.005, 20.0);
    let d_healthy = decide_prune(&model, &healthy, target, threshold, &device)?;
    eprintln!(
        "healthy: plateau={:.4} best={:.4} conf={:.3} prune={}",
        d_healthy.forecast_plateau, d_healthy.best_so_far, d_healthy.confidence, d_healthy.prune
    );

    // Honesty guards: forecasts must be finite.
    assert!(
        d_doomed.forecast_plateau.is_finite(),
        "doomed forecast non-finite"
    );
    assert!(
        d_healthy.forecast_plateau.is_finite(),
        "healthy forecast non-finite"
    );

    // The decisions.
    assert!(
        d_doomed.prune,
        "doomed curve (plateau ~0.12 > 0.05) should PRUNE, got plateau={}",
        d_doomed.forecast_plateau
    );
    assert!(
        !d_healthy.prune,
        "healthy curve (floor ~0.01 < 0.05) should CONTINUE, got plateau={}",
        d_healthy.forecast_plateau
    );

    Ok(())
}
