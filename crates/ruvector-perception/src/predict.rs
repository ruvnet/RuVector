//! Boundary-first world model.
//!
//! Conventional world models predict the *full* next state and measure error
//! against it. That is expensive and, for a perception substrate, beside the
//! point: we do not care what every zone will read next, we care **where
//! coherence will break next**. So instead of forecasting state, this module
//! forecasts the *boundary*:
//!
//! ```text
//! boundary_{t+1} = f(boundary_t, delta_history, modality_conflict)
//! ```
//!
//! Each zone keeps a short rolling history of an *instability* sample. The
//! per-observation sample combines how cleanly a boundary recurs (its
//! `coherence`) with how much the modalities disagree about it (its
//! `contradiction`):
//!
//! ```text
//! instability = coherence * (1 + contradiction)
//! ```
//!
//! A clean boundary that keeps recurring *with* contradictions is the most
//! destabilising: it is consistent enough to be real and conflicted enough to be
//! unresolved. From the window we read a *level* (mean) and a *trend* (slope),
//! and forecast `level + trend` for the next step. The zone with the highest
//! forecast is the one most likely to break next.
//!
//! The model is deterministic, allocation-light, and uses only `std` + `serde`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// A single observed boundary event for one zone at one time step.
///
/// `coherence` is how cleanly the boundary separated (see
/// [`crate::coherence::Boundary::coherence`]); `contradiction` is how strongly
/// the modalities disagreed about it. Both are expected in `[0, 1]` but are
/// clamped defensively.
#[derive(Debug, Clone)]
pub struct BoundaryObservation {
    /// The zone this event concerns.
    pub zone: String,
    /// Cleanliness of the boundary in `[0, 1]` (high = sharp separation).
    pub coherence: f32,
    /// Modality disagreement in `[0, 1]` (high = unresolved conflict).
    pub contradiction: f32,
    /// Logical time of the observation.
    pub t: u64,
}

impl BoundaryObservation {
    /// Convenience constructor.
    pub fn new(zone: impl Into<String>, coherence: f32, contradiction: f32, t: u64) -> Self {
        Self {
            zone: zone.into(),
            coherence,
            contradiction,
            t,
        }
    }

    /// Per-observation instability sample: `coherence * (1 + contradiction)`.
    ///
    /// Inputs are clamped to `[0, 1]`, so the result lies in `[0, 2]`.
    fn instability_sample(&self) -> f32 {
        let coh = self.coherence.clamp(0.0, 1.0);
        let con = self.contradiction.clamp(0.0, 1.0);
        coh * (1.0 + con)
    }
}

/// A forecast of where coherence will break next, for a single zone.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundaryForecast {
    /// The zone this forecast concerns.
    pub zone: String,
    /// Forecast instability for the next step, `(level + trend).max(0.0)`.
    pub instability: f32,
    /// Slope of the recent window. Positive = the boundary is worsening.
    pub trend: f32,
}

/// Predicts which boundary breaks next from a rolling per-zone history.
///
/// Construct with [`BoundaryPredictor::new`], feed events with
/// [`BoundaryPredictor::observe`], then read [`BoundaryPredictor::forecast`] or
/// [`BoundaryPredictor::next_break`].
#[derive(Debug, Clone)]
pub struct BoundaryPredictor {
    /// Rolling window length kept per zone (at least 1).
    window: usize,
    /// Per-zone rolling instability samples, oldest first.
    ///
    /// `BTreeMap` keeps iteration deterministic so equal forecasts keep a stable
    /// (alphabetical) order after the instability sort.
    history: BTreeMap<String, Vec<f32>>,
}

impl BoundaryPredictor {
    /// Create a predictor keeping a rolling window of `window` samples per zone.
    ///
    /// A `window` of `0` is treated as `1` (a forecast needs at least one
    /// sample), keeping the type total and panic-free.
    pub fn new(window: usize) -> Self {
        Self {
            window: window.max(1),
            history: BTreeMap::new(),
        }
    }

    /// Record an observed boundary event for a zone at time `t`.
    ///
    /// The derived instability sample is appended to that zone's window; the
    /// oldest sample is evicted once the window is full. The `t` field is part
    /// of the public record but does not affect the rolling order, which is the
    /// order of `observe` calls (callers are expected to feed events in time
    /// order, as the rest of the pipeline does).
    pub fn observe(&mut self, obs: &BoundaryObservation) {
        let sample = obs.instability_sample();
        let win = self.window;
        let series = self.history.entry(obs.zone.clone()).or_default();
        series.push(sample);
        if series.len() > win {
            // Drop the oldest sample to keep the rolling window bounded.
            let overflow = series.len() - win;
            series.drain(0..overflow);
        }
    }

    /// Forecast per-zone instability for the next step, sorted by instability
    /// descending (ties broken by zone name for determinism).
    ///
    /// Returns an empty vector if nothing has been observed.
    pub fn forecast(&self) -> Vec<BoundaryForecast> {
        let mut out: Vec<BoundaryForecast> = self
            .history
            .iter()
            .filter(|(_, series)| !series.is_empty())
            .map(|(zone, series)| {
                let level = mean(series);
                let trend = slope(series);
                let instability = (level + trend).max(0.0);
                BoundaryForecast {
                    zone: zone.clone(),
                    instability,
                    trend,
                }
            })
            .collect();

        out.sort_by(|a, b| {
            b.instability
                .partial_cmp(&a.instability)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.zone.cmp(&b.zone))
        });
        out
    }

    /// The single most-likely zone to break next (highest forecast
    /// instability), or `None` if nothing has been observed.
    pub fn next_break(&self) -> Option<BoundaryForecast> {
        self.forecast().into_iter().next()
    }
}

/// Arithmetic mean of a non-empty slice. Returns `0.0` for an empty slice.
fn mean(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f32>() / xs.len() as f32
}

/// Least-squares slope of `xs` over indices `0..len`.
///
/// Equivalent to the trend of the rolling window: positive means the boundary
/// is worsening. Returns `0.0` for fewer than two samples (a single point has
/// no trend).
fn slope(xs: &[f32]) -> f32 {
    let n = xs.len();
    if n < 2 {
        return 0.0;
    }
    let n_f = n as f32;
    // x is the integer index 0..n; mean_x = (n-1)/2.
    let mean_x = (n_f - 1.0) / 2.0;
    let mean_y = mean(xs);
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for (i, &y) in xs.iter().enumerate() {
        let dx = i as f32 - mean_x;
        num += dx * (y - mean_y);
        den += dx * dx;
    }
    if den == 0.0 {
        0.0
    } else {
        num / den
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rising_zone_has_positive_trend_and_higher_forecast() {
        let mut p = BoundaryPredictor::new(5);

        // "kitchen" worsens: coherence and contradiction both climb.
        for (i, (coh, con)) in [(0.1, 0.0), (0.3, 0.2), (0.5, 0.4), (0.7, 0.6), (0.9, 0.8)]
            .into_iter()
            .enumerate()
        {
            p.observe(&BoundaryObservation::new("kitchen", coh, con, i as u64));
        }

        // "hallway" stays calm: low coherence, no contradiction.
        for i in 0..5 {
            p.observe(&BoundaryObservation::new("hallway", 0.1, 0.0, i as u64));
        }

        let forecast = p.forecast();
        assert_eq!(forecast.len(), 2);

        let kitchen = forecast.iter().find(|f| f.zone == "kitchen").unwrap();
        let hallway = forecast.iter().find(|f| f.zone == "hallway").unwrap();

        // The worsening zone has a clearly positive trend...
        assert!(
            kitchen.trend > 0.0,
            "rising zone should have positive trend, got {}",
            kitchen.trend
        );
        // ...the calm zone is flat...
        assert!(
            hallway.trend.abs() < 1e-6,
            "stable zone should be flat, got {}",
            hallway.trend
        );
        // ...and forecast instability is higher for the worsening zone.
        assert!(
            kitchen.instability > hallway.instability,
            "rising {} should exceed stable {}",
            kitchen.instability,
            hallway.instability
        );

        // Sorted descending: the worsening zone leads.
        assert_eq!(forecast[0].zone, "kitchen");
    }

    #[test]
    fn next_break_returns_the_rising_zone() {
        let mut p = BoundaryPredictor::new(4);
        for i in 0..4 {
            let coh = 0.2 + 0.2 * i as f32;
            let con = 0.1 * i as f32;
            p.observe(&BoundaryObservation::new("garage", coh, con, i as u64));
            p.observe(&BoundaryObservation::new("porch", 0.05, 0.0, i as u64));
        }

        let next = p.next_break().expect("a break should be predicted");
        assert_eq!(next.zone, "garage");
        assert!(next.instability > 0.0);
    }

    #[test]
    fn empty_predictor_yields_nothing() {
        let p = BoundaryPredictor::new(8);
        assert!(p.forecast().is_empty());
        assert!(p.next_break().is_none());
    }

    #[test]
    fn window_evicts_oldest_samples() {
        let mut p = BoundaryPredictor::new(2);
        // Early calm sample then two strong ones; with window 2 the calm sample
        // is evicted, so the level reflects only the recent strong activity.
        p.observe(&BoundaryObservation::new("z", 0.0, 0.0, 0));
        p.observe(&BoundaryObservation::new("z", 0.9, 0.9, 1));
        p.observe(&BoundaryObservation::new("z", 0.9, 0.9, 2));

        let f = p.next_break().unwrap();
        // Both retained samples are identical => flat trend, high level.
        assert!(f.trend.abs() < 1e-6);
        assert!(f.instability > 1.0, "got {}", f.instability);
    }

    #[test]
    fn instability_sample_is_clamped() {
        // Out-of-range inputs are clamped to [0, 1] before combining.
        let obs = BoundaryObservation::new("z", 2.0, 5.0, 0);
        // coherence -> 1.0, contradiction -> 1.0 => 1.0 * (1 + 1) = 2.0
        assert!((obs.instability_sample() - 2.0).abs() < 1e-6);

        let neg = BoundaryObservation::new("z", -1.0, -1.0, 0);
        assert!(neg.instability_sample().abs() < 1e-6);
    }

    #[test]
    fn forecast_is_deterministic_and_serializable() {
        let mut p = BoundaryPredictor::new(3);
        p.observe(&BoundaryObservation::new("a", 0.5, 0.5, 0));
        p.observe(&BoundaryObservation::new("b", 0.5, 0.5, 0));

        let f = p.forecast();
        // Equal instability => alphabetical tie-break is stable.
        assert_eq!(f[0].zone, "a");
        assert_eq!(f[1].zone, "b");

        let json = serde_json::to_string(&f[0]).unwrap();
        let back: BoundaryForecast = serde_json::from_str(&json).unwrap();
        assert_eq!(f[0], back);
    }
}
