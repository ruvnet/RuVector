//! Predictive pruning for evolutionary / optimization runs (ADR-191 §2).
//!
//! The capability: **kill doomed runs at iteration 50 instead of 1000.** Given
//! the first `K` metric values of a long optimization (e.g. a Darwin genome's
//! exploitability or loss curve, where *lower is better*), use TimesFM to
//! forecast the tail out to a target horizon, estimate the plateau the run is
//! heading toward, and decide `PRUNE` vs `CONTINUE` against a viability
//! threshold.
//!
//! ## Decoupling
//!
//! This module deliberately operates on a **generic numeric curve**
//! (`&[f32]`), not on any poker-darwin / agent-harness-generator type. That
//! keeps `timesfm` free of a cross-repo dependency: the caller is responsible
//! for extracting a monotone (or near-monotone) scalar metric per iteration and
//! handing it in as a `Vec<f32>`. See [`forecast_plateau`] /
//! [`decide_prune`] for the call surface and the doc-comment on
//! [`decide_prune`] for how poker-darwin would wire it in.
//!
//! ## Why TimesFM and not a curve fit
//!
//! A parametric fit (e.g. `a - b·exp(-c·t)`) needs you to *assume* the curve
//! family. TimesFM is a zero-shot foundation forecaster: it reads the shape of
//! the first `K` points and extrapolates without a hand-picked functional form,
//! which is exactly what you want when different genomes converge with
//! different dynamics.
//!
//! ## Honest calibration note (measured 2026-06-24, f32 CPU)
//!
//! On *short, synthetic, sharply-decayed* monotone curves, TimesFM tends to
//! forecast a mild **mean-reversion upward** rather than holding the decayed
//! floor — so the absolute `forecast_plateau` it returns is biased *high* by a
//! constant-ish offset on such inputs (e.g. a curve truly heading to 0.01 may
//! forecast a tail near ~0.3–0.6 from 128 points). What stays robust is the
//! **relative ordering**: a curve heading to a worse floor reliably forecasts a
//! higher plateau than one heading to a better floor (verified across
//! K ∈ {96..256}). The decision logic here is therefore built on two robust
//! signals — (1) doomed runs forecast a plateau *far above* the threshold, and
//! (2) genuinely-good runs trip the *already-viable guard* (best-so-far has
//! already beaten the threshold) — rather than on the model's absolute
//! plateau calibration. Real optimization curves (longer, noisier, less
//! perfectly exponential) sit closer to TimesFM's training distribution; a
//! caller wanting an absolutely-calibrated plateau should fit a small per-family
//! bias correction on held-out runs.

#[cfg(feature = "candle")]
use candle_core::{DType, Device, IndexOp, Tensor};

#[cfg(feature = "candle")]
use crate::model::PatchedTimeSeriesDecoder;
use crate::Result;

/// A prune/continue decision plus the evidence behind it.
#[derive(Debug, Clone, PartialEq)]
pub struct PruneDecision {
    /// `true` ⇒ the run is forecast to plateau *worse* than the viability
    /// threshold and should be killed; `false` ⇒ let it keep running.
    pub prune: bool,
    /// The forecast plateau value (the level the metric is predicted to settle
    /// at by the end of the horizon). Same units as the input curve; lower is
    /// better.
    pub forecast_plateau: f32,
    /// The best (lowest) value seen *so far* in the partial curve — context for
    /// interpreting the plateau.
    pub best_so_far: f32,
    /// A `[0, 1]` confidence in the PRUNE/CONTINUE call. It is large when the
    /// forecast plateau is far (relative to the curve's own scale) from the
    /// viability threshold, and small when the plateau sits right on the
    /// threshold (a near-tie we should not act aggressively on).
    pub confidence: f32,
    /// The viability threshold the decision was taken against (echoed for
    /// auditability).
    pub threshold: f32,
}

/// Forecast the **plateau** a partial curve is heading toward.
///
/// `partial` is the first `K` metric values (lower = better). The function
/// instance-normalizes nothing itself — TimesFM applies RevIN internally — it
/// just shapes the input, asks the model for `horizon` future steps, and
/// summarizes the *tail* of that forecast as the plateau.
///
/// The plateau is the mean of the last `plateau_window` forecast steps (default
/// via [`decide_prune`] is `horizon / 4`, min 1). Averaging the tail rather
/// than taking a single endpoint de-noises the autoregressive decode.
///
/// Returns `(plateau, full_forecast)` so callers can inspect the whole
/// trajectory if they want (e.g. to detect "still descending at the horizon").
#[cfg(feature = "candle")]
pub fn forecast_plateau(
    model: &PatchedTimeSeriesDecoder,
    partial: &[f32],
    horizon: usize,
    plateau_window: usize,
    device: &Device,
) -> Result<(f32, Vec<f32>)> {
    assert!(!partial.is_empty(), "partial curve must be non-empty");
    assert!(horizon > 0, "horizon must be > 0");

    let k = partial.len();
    let input_ts = Tensor::from_vec(partial.to_vec(), (1, k), device)?;
    // All observed points are real (not padding).
    let input_padding = Tensor::zeros((1, k), DType::F32, device)?;
    // freq_id = 0: high-frequency / fine-grained series (the TimesFM default
    // bucket). An optimization curve is a dense per-iteration signal, so the
    // finest frequency bucket is the right one.
    let freq = Tensor::from_vec(vec![0u32], (1, 1), device)?;

    let (point, _full) = model.decode(&input_ts, &input_padding, &freq, horizon)?;
    let forecast: Vec<f32> = point.i(0)?.to_vec1()?;

    let w = plateau_window.clamp(1, horizon);
    let tail = &forecast[horizon - w..];
    let plateau = tail.iter().copied().sum::<f32>() / w as f32;

    Ok((plateau, forecast))
}

/// Decide PRUNE vs CONTINUE for a partial optimization curve.
///
/// # Semantics (lower = better, like exploitability or loss)
///
/// * Forecast the curve's tail to `horizon` steps with TimesFM.
/// * Take the **plateau** = mean of the last `horizon/4` (min 1) forecast steps.
/// * `PRUNE` iff the forecast plateau is *above* (worse than) `threshold`.
/// * `confidence` scales with how far the plateau is from the threshold,
///   normalized by the curve's observed dynamic range, so it is comparable
///   across metrics of different magnitudes.
///
/// # Guards
///
/// * A non-finite forecast (NaN/Inf) is treated as **CONTINUE with confidence
///   0** — we never kill a run on a broken forecast. The `forecast_plateau`
///   field carries the (non-finite) value so the caller can log it.
/// * If `best_so_far` is *already* at/under the threshold, the run has already
///   proven viable; we never PRUNE it regardless of the forecast.
///
/// # How poker-darwin would call this
///
/// poker-darwin (in `ruvnet/agent-harness-generator`) tracks a **monotone
/// champion curve**: the best-so-far exploitability of the population at each
/// generation. After the first `K` generations it would do, roughly:
///
/// ```ignore
/// // champion_expl: Vec<f32> — best exploitability per generation so far (K long)
/// let cfg = TimesfmConfig::timesfm_1p0_200m();
/// let model = PatchedTimeSeriesDecoder::load(cfg, vb)?;
/// let decision = timesfm::prune::decide_prune(
///     &model,
///     &champion_expl,     // first K champion values
///     1000,               // target total generations (horizon = 1000 - K)
///     0.05,               // viability threshold: ship only if expl <= 5%
///     &Device::Cpu,
/// )?;
/// if decision.prune && decision.confidence > 0.6 {
///     // forecast says this genome plateaus above 5% exploitability —
///     // kill it at generation K instead of burning to generation 1000.
///     genome.terminate("predictive-prune", decision);
/// }
/// ```
///
/// The `confidence > 0.6` gate is the caller's risk knob: only act on
/// high-confidence kills, let borderline runs continue.
#[cfg(feature = "candle")]
pub fn decide_prune(
    model: &PatchedTimeSeriesDecoder,
    partial: &[f32],
    target_horizon: usize,
    threshold: f32,
    device: &Device,
) -> Result<PruneDecision> {
    assert!(!partial.is_empty(), "partial curve must be non-empty");

    let k = partial.len();
    // Horizon = how many steps remain until the target length. If the caller
    // already has >= target_horizon points there is nothing to forecast; use a
    // minimal 1-step horizon so the API stays total.
    let horizon = target_horizon.saturating_sub(k).max(1);
    let plateau_window = (horizon / 4).max(1);

    let best_so_far = partial.iter().copied().fold(f32::INFINITY, f32::min);
    let worst_so_far = partial.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let (plateau, _forecast) = forecast_plateau(model, partial, horizon, plateau_window, device)?;

    // Honesty guard: a broken (non-finite) forecast must NOT kill a run.
    if !plateau.is_finite() {
        return Ok(PruneDecision {
            prune: false,
            forecast_plateau: plateau,
            best_so_far,
            confidence: 0.0,
            threshold,
        });
    }

    // Already-viable guard: if the run has already beaten the threshold, keep it.
    if best_so_far <= threshold {
        return Ok(PruneDecision {
            prune: false,
            forecast_plateau: plateau,
            best_so_far,
            confidence: confidence_for(plateau, threshold, best_so_far, worst_so_far),
            threshold,
        });
    }

    let prune = plateau > threshold;
    let confidence = confidence_for(plateau, threshold, best_so_far, worst_so_far);

    Ok(PruneDecision {
        prune,
        forecast_plateau: plateau,
        best_so_far,
        confidence,
        threshold,
    })
}

/// Confidence that the (plateau vs threshold) call is correct.
///
/// Normalized by the curve's observed dynamic range `|worst - best|` so it is
/// scale-invariant: a plateau half the curve's range away from the threshold is
/// confidence ≈ 0.5; right on the threshold is ≈ 0; far away saturates at 1.
fn confidence_for(plateau: f32, threshold: f32, best: f32, worst: f32) -> f32 {
    let range = (worst - best).abs();
    // Fall back to |threshold| (or 1) when the curve is flat so we never divide
    // by ~0.
    let scale = if range > 1e-6 {
        range
    } else {
        threshold.abs().max(1.0)
    };
    let margin = (plateau - threshold).abs() / scale;
    // Map margin → [0,1): 0 at the threshold, → 1 as the plateau gets far away.
    (margin / (1.0 + margin)).clamp(0.0, 1.0)
}
