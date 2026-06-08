//! Physical state history: per-(zone, modality) rolling baselines and how
//! *responsive* each sensor usually is in each zone (used to detect a sensor
//! that "should have reacted but didn't" — the contradiction signal).

use crate::modality::Modality;
use std::collections::HashMap;

/// A single sensor sample in one zone at one time window.
#[derive(Debug, Clone, PartialEq)]
pub struct Reading {
    /// Human-readable zone name (e.g. "table_left_zone").
    pub zone: String,
    /// Which modality produced the sample.
    pub modality: Modality,
    /// Scalar value (already feature-extracted, e.g. band energy).
    pub value: f32,
}

impl Reading {
    /// Convenience constructor.
    pub fn new(zone: impl Into<String>, modality: Modality, value: f32) -> Self {
        Self {
            zone: zone.into(),
            modality,
            value,
        }
    }
}

/// Per-(zone, modality) running statistics.
#[derive(Debug, Clone, Copy)]
struct Channel {
    /// EWMA baseline of the value.
    baseline: f32,
    /// EWMA of |delta| magnitude — the channel's typical activity.
    activity: f32,
    /// Fraction of updates with a significant delta (responsiveness in [0,1]).
    responsiveness: f32,
    /// Whether the channel has been initialised.
    seen: bool,
}

impl Default for Channel {
    fn default() -> Self {
        Self {
            baseline: 0.0,
            activity: 0.0,
            responsiveness: 0.0,
            seen: false,
        }
    }
}

/// Rolling multi-modal world state.
#[derive(Debug, Clone, Default)]
pub struct WorldState {
    channels: HashMap<(String, Modality), Channel>,
    alpha: f32,            // EWMA smoothing
    active_threshold: f32, // |delta| above this counts as "responded"
}

impl WorldState {
    /// New state. `alpha` is the EWMA factor (e.g. 0.3); `active_threshold` is
    /// the |delta| above which a channel is considered to have responded.
    pub fn new(alpha: f32, active_threshold: f32) -> Self {
        Self {
            channels: HashMap::new(),
            alpha,
            active_threshold,
        }
    }

    /// Current baseline for a channel (0 if unseen).
    pub fn baseline(&self, zone: &str, m: Modality) -> f32 {
        self.channels
            .get(&(zone.to_string(), m))
            .map(|c| c.baseline)
            .unwrap_or(0.0)
    }

    /// How responsive a channel historically is, in `[0, 1]`.
    pub fn responsiveness(&self, zone: &str, m: Modality) -> f32 {
        self.channels
            .get(&(zone.to_string(), m))
            .map(|c| c.responsiveness)
            .unwrap_or(0.0)
    }

    /// Whether a channel has any history.
    pub fn seen(&self, zone: &str, m: Modality) -> bool {
        self.channels
            .get(&(zone.to_string(), m))
            .map(|c| c.seen)
            .unwrap_or(false)
    }

    /// Threshold above which a |delta| counts as a response.
    pub fn active_threshold(&self) -> f32 {
        self.active_threshold
    }

    /// Fold a reading into the rolling state (after its delta has been read).
    pub fn update(&mut self, r: &Reading) {
        let key = (r.zone.clone(), r.modality);
        let a = self.alpha;
        let thr = self.active_threshold;
        let ch = self.channels.entry(key).or_default();
        if !ch.seen {
            ch.baseline = r.value;
            ch.seen = true;
            return;
        }
        let delta = (r.value - ch.baseline).abs();
        let responded = if delta >= thr { 1.0 } else { 0.0 };
        ch.activity = (1.0 - a) * ch.activity + a * delta;
        ch.responsiveness = (1.0 - a) * ch.responsiveness + a * responded;
        ch.baseline = (1.0 - a) * ch.baseline + a * r.value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_baseline_and_responsiveness() {
        let mut s = WorldState::new(0.5, 0.5);
        // Thermal in zone A reacts repeatedly -> high responsiveness.
        for v in [0.0, 1.0, 0.0, 1.0, 0.0, 1.0] {
            s.update(&Reading::new("A", Modality::Thermal, v));
        }
        assert!(s.seen("A", Modality::Thermal));
        assert!(s.responsiveness("A", Modality::Thermal) > 0.4);
        // An unseen channel is quiet.
        assert!(!s.seen("A", Modality::Rf));
        assert_eq!(s.responsiveness("A", Modality::Rf), 0.0);
    }
}
