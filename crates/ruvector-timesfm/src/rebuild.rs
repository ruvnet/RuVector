//! Forecast-driven vector-index maintenance: decide *when* to rebuild an HNSW
//! index from its recall-drift history.
//!
//! RuVector ANN indexes lose recall as they accrue deletes/updates (see the
//! `ruvector-diskann` recall-trigger work). Instead of rebuilding on a fixed
//! schedule or only after recall has already dropped, forecast the recall curve
//! with TimesFM and schedule the rebuild to land *just before* recall crosses a
//! floor — fewer rebuilds at equal-or-better served recall.

use serde::{Deserialize, Serialize};

use crate::forecast_types::Forecast;

/// Advice on whether/when to rebuild a vector index.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RebuildAdvice {
    /// Rebuild now: recall is already at/under the floor, or the conservative
    /// (p10) forecast crosses the floor within `lead_steps`.
    pub rebuild_now: bool,
    /// Forecast steps until the **median (p50)** recall is predicted to drop
    /// below the floor (`None` if it never does within the horizon).
    pub steps_until_floor: Option<usize>,
    /// Same, but for the conservative **lower band (p10)** — the early-warning
    /// signal `rebuild_now` acts on.
    pub steps_until_floor_p10: Option<usize>,
    /// Recall floor the advice was taken against (echoed for auditability).
    pub floor: f32,
    /// The recall forecast (point + bands) the advice was derived from.
    pub forecast: Forecast,
}

/// Decide rebuild timing from a recall forecast.
///
/// `recall` curves are *higher = better*, so "crossing the floor" means the
/// forecast falling **below** `floor`. `lead_steps` is how many steps of
/// look-ahead trigger an immediate rebuild (e.g. enough time for a rebuild to
/// finish before recall actually degrades). Acts on the **p10** lower band so
/// the trigger is conservative.
#[must_use]
pub fn advise_from_forecast(
    forecast: Forecast,
    last_observed_recall: f32,
    floor: f32,
    lead_steps: usize,
) -> RebuildAdvice {
    let first_below = |band: &[f32]| band.iter().position(|&r| r < floor);
    let p50 = forecast.p50();
    let p10 = forecast.p10();
    let steps_until_floor = first_below(&p50);
    let steps_until_floor_p10 = first_below(&p10);

    let rebuild_now = last_observed_recall <= floor
        || matches!(steps_until_floor_p10, Some(s) if s <= lead_steps);

    RebuildAdvice {
        rebuild_now,
        steps_until_floor,
        steps_until_floor_p10,
        floor,
        forecast,
    }
}

#[cfg(feature = "candle")]
impl crate::Forecaster {
    /// Forecast `horizon` steps of recall from `recall_history` and advise on
    /// rebuild timing against `floor` (see [`advise_from_forecast`]).
    pub fn advise_rebuild(
        &self,
        recall_history: &[f32],
        floor: f32,
        horizon: usize,
        lead_steps: usize,
    ) -> crate::Result<RebuildAdvice> {
        if recall_history.is_empty() {
            return Err(crate::Error::Invalid("recall_history is empty".into()));
        }
        let forecast = self.forecast(recall_history, horizon)?;
        let last = *recall_history.last().unwrap();
        Ok(advise_from_forecast(forecast, last, floor, lead_steps))
    }
}
