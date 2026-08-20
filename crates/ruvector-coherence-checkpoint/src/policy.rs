//! Snapshot scheduling policies.
//!
//! Every policy answers one question per event: "should a new witness-chained
//! snapshot be taken right now?" The three variants trade storage (number of
//! snapshots) against worst-case replay gap (events since the nearest
//! snapshot) differently:
//!
//! - [`FixedInterval`] (baseline): snapshot every `interval` events,
//!   independent of workload content. Simple, but wastes snapshots during
//!   calm periods and under-covers bursts that happen mid-interval.
//! - [`DriftTriggered`] (candidate A): snapshot when the store's centroid has
//!   moved more than `threshold` (cosine distance) since the last snapshot.
//!   Adapts to workload content but has no worst-case bound if drift never
//!   crosses the threshold.
//! - [`DriftTriggeredCapped`] (candidate B): drift-triggered, plus a hard
//!   `max_interval` ceiling so a long calm period still gets bounded
//!   worst-case replay gap.

use crate::drift::drift;

/// Decision inputs available to a policy at each event.
pub struct PolicyContext<'a> {
    pub current_centroid: &'a [f32],
    pub last_snapshot_centroid: Option<&'a [f32]>,
    pub events_since_last_snapshot: usize,
}

pub trait CheckpointPolicy {
    /// Called once per event (after the event's insert has already been
    /// applied and the running centroid updated). Return `true` to take a
    /// snapshot at this event.
    fn should_snapshot(&mut self, ctx: &PolicyContext<'_>) -> bool;

    fn name(&self) -> &'static str;
}

/// Baseline: snapshot every `interval` events.
pub struct FixedInterval {
    pub interval: usize,
}

impl CheckpointPolicy for FixedInterval {
    fn should_snapshot(&mut self, ctx: &PolicyContext<'_>) -> bool {
        ctx.events_since_last_snapshot >= self.interval
    }

    fn name(&self) -> &'static str {
        "FixedInterval"
    }
}

/// Candidate A: snapshot when centroid drift since the last snapshot exceeds
/// `threshold`.
pub struct DriftTriggered {
    pub threshold: f32,
}

impl CheckpointPolicy for DriftTriggered {
    fn should_snapshot(&mut self, ctx: &PolicyContext<'_>) -> bool {
        match ctx.last_snapshot_centroid {
            None => false,
            Some(last) => drift(last, ctx.current_centroid) >= self.threshold,
        }
    }

    fn name(&self) -> &'static str {
        "DriftTriggered"
    }
}

/// Candidate B: drift-triggered with a hard maximum interval, bounding
/// worst-case replay gap even if drift never crosses `threshold`.
pub struct DriftTriggeredCapped {
    pub threshold: f32,
    pub max_interval: usize,
}

impl CheckpointPolicy for DriftTriggeredCapped {
    fn should_snapshot(&mut self, ctx: &PolicyContext<'_>) -> bool {
        if ctx.events_since_last_snapshot >= self.max_interval {
            return true;
        }
        match ctx.last_snapshot_centroid {
            None => false,
            Some(last) => drift(last, ctx.current_centroid) >= self.threshold,
        }
    }

    fn name(&self) -> &'static str {
        "DriftTriggeredCapped"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_interval_triggers_exactly_at_interval() {
        let mut p = FixedInterval { interval: 10 };
        let ctx_below = PolicyContext {
            current_centroid: &[0.0],
            last_snapshot_centroid: None,
            events_since_last_snapshot: 9,
        };
        assert!(!p.should_snapshot(&ctx_below));
        let ctx_at = PolicyContext {
            current_centroid: &[0.0],
            last_snapshot_centroid: None,
            events_since_last_snapshot: 10,
        };
        assert!(p.should_snapshot(&ctx_at));
    }

    #[test]
    fn drift_triggered_never_fires_without_a_prior_snapshot() {
        let mut p = DriftTriggered { threshold: 0.01 };
        let ctx = PolicyContext {
            current_centroid: &[1.0, 0.0],
            last_snapshot_centroid: None,
            events_since_last_snapshot: 1000,
        };
        assert!(!p.should_snapshot(&ctx));
    }

    #[test]
    fn drift_triggered_fires_past_threshold() {
        let mut p = DriftTriggered { threshold: 0.1 };
        let last = [1.0, 0.0];
        let ctx = PolicyContext {
            current_centroid: &[0.0, 1.0],
            last_snapshot_centroid: Some(&last),
            events_since_last_snapshot: 3,
        };
        assert!(p.should_snapshot(&ctx));
    }

    #[test]
    fn capped_forces_snapshot_at_max_interval_even_with_no_drift() {
        let mut p = DriftTriggeredCapped {
            threshold: 0.9,
            max_interval: 5,
        };
        let last = [1.0, 0.0];
        let ctx = PolicyContext {
            current_centroid: &[1.0, 0.0],
            last_snapshot_centroid: Some(&last),
            events_since_last_snapshot: 5,
        };
        assert!(p.should_snapshot(&ctx));
    }
}
