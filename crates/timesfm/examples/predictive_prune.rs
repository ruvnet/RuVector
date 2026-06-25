//! Predictive-pruning demo (ADR-191 §2): "kill doomed runs at iter 50 instead
//! of 1000."
//!
//! Builds the real TimesFM 1.0 200M model, then runs [`timesfm::prune::decide_prune`]
//! on TWO synthetic optimization curves (lower = better, like exploitability):
//!
//!   (a) DOOMED  — descends fast then plateaus HIGH (above the viability
//!       threshold). Expected decision: PRUNE.
//!   (b) HEALTHY — still descending toward (and past) the threshold at iter K.
//!       Expected decision: CONTINUE.
//!
//! Gated on the converted weights existing locally; skips cleanly (exit 0) when
//! absent, exactly like `examples/parity.rs`. Real numbers only — never
//! fabricates a decision.
//!
//! Run with:
//! ```ignore
//! cargo run -p timesfm --features candle --example predictive_prune -- \
//!     /tmp/timesfm-parity/timesfm.safetensors
//! ```

#[cfg(not(feature = "candle"))]
fn main() {
    eprintln!("this example requires --features candle");
    std::process::exit(2);
}

#[cfg(feature = "candle")]
fn main() -> anyhow::Result<()> {
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;
    use timesfm::config::TimesfmConfig;
    use timesfm::model::PatchedTimeSeriesDecoder;
    use timesfm::prune::decide_prune;

    let weights = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/timesfm-parity/timesfm.safetensors".to_string());

    if !std::path::Path::new(&weights).exists() {
        eprintln!(
            "SKIP predictive_prune: weights missing ({weights}). \
             Generate via scripts/convert_weights.py. Not fabricating a decision."
        );
        return Ok(());
    }

    let device = Device::Cpu;
    let cfg = TimesfmConfig::timesfm_1p0_200m();
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights.clone()], DType::F32, &device)? };
    let model = PatchedTimeSeriesDecoder::load(cfg, vb)?;
    eprintln!("loaded TimesFM 1.0 200M from {weights}");

    // --- Synthetic curves. K = first 128 iterations observed; target = 1000. ---
    // Both are MONOTONE non-increasing (best-so-far / champion curves), matching
    // how poker-darwin tracks exploitability. This is the real "kill at iter 128
    // of a 1000-iter budget" scenario.
    let k = 128usize;
    let target = 1000usize;
    let threshold = 0.05f32; // ship only if exploitability is <= 5%.

    // Helper: y(t) = floor + (y0 - floor) * exp(-t / tau).
    let decay = |floor: f32, tau: f32| -> Vec<f32> {
        (0..k)
            .map(|t| floor + (0.95f32 - floor) * (-(t as f32) / tau).exp())
            .collect::<Vec<f32>>()
    };

    // (a) DOOMED: plateaus HIGH at floor ~0.20 — best-so-far NEVER reaches the
    //     0.05 viability threshold, and TimesFM forecasts a tail far above it.
    //     The PRUNE here is forecast-driven.
    let doomed = decay(0.20, 16.0);

    // (b) HEALTHY: a strong genome that has ALREADY driven exploitability below
    //     the 0.05 ship threshold by iteration K (floor ~0.005). The CONTINUE
    //     here is driven by the "already-viable" guard — you never kill a run
    //     that has empirically already proven viable.
    let healthy = decay(0.005, 20.0);

    let report = |name: &str, curve: &[f32], expect_prune: bool| -> anyhow::Result<bool> {
        let d = decide_prune(&model, curve, target, threshold, &device)?;
        println!("---- {name} ----");
        println!(
            "  observed[0]={:.4} observed[K-1]={:.4}  best_so_far={:.4}",
            curve[0],
            curve[curve.len() - 1],
            d.best_so_far
        );
        println!(
            "  forecast_plateau={:.4}  threshold={:.4}  confidence={:.3}",
            d.forecast_plateau, d.threshold, d.confidence
        );
        let verdict = if d.prune { "PRUNE" } else { "CONTINUE" };
        let want = if expect_prune { "PRUNE" } else { "CONTINUE" };
        let ok = d.prune == expect_prune;
        println!(
            "  decision = {verdict}   (expected {want})  [{}]",
            if ok { "OK" } else { "MISMATCH" }
        );
        Ok(ok)
    };

    println!("================ Predictive pruning (TimesFM) ================");
    println!("K={k} observed, target_horizon={target}, threshold={threshold}");
    println!();
    let a_ok = report("(a) DOOMED  → expect PRUNE", &doomed, true)?;
    println!();
    let b_ok = report("(b) HEALTHY → expect CONTINUE", &healthy, false)?;
    println!();

    if a_ok && b_ok {
        println!("VERDICT: PASS  (both decisions correct)");
        Ok(())
    } else {
        println!("VERDICT: FAIL  (a_ok={a_ok}, b_ok={b_ok})");
        std::process::exit(1);
    }
}
