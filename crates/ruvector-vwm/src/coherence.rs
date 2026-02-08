//! Coherence gate for world-model update control.
//!
//! The [`CoherenceGate`] evaluates incoming sensor data and tile updates against
//! a tunable [`CoherencePolicy`] to produce a [`CoherenceDecision`]: accept the
//! update, defer it for later, freeze the tile, or rollback to a previous state.
//!
//! This implements the "governance loop" from the VWM architecture, ensuring that
//! updates are only applied when they are consistent with the existing world state,
//! sensor confidence is sufficient, and rendering budgets are not exceeded.

/// Coherence gate decision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CoherenceDecision {
    /// Accept the update and apply it to the world model.
    Accept,
    /// Defer the update for re-evaluation in a future cycle.
    Defer,
    /// Freeze the tile in its current state (no updates accepted).
    Freeze,
    /// Rollback the tile to a previous consistent state.
    Rollback,
}

/// Inputs to the coherence gate evaluation.
#[derive(Clone, Debug)]
pub struct CoherenceInput {
    /// Disagreement between the proposed update and the existing tile state (0.0 = agreement).
    pub tile_disagreement: f32,
    /// Continuity score of tracked entities (1.0 = perfect continuity).
    pub entity_continuity: f32,
    /// Confidence of the sensor providing the update.
    pub sensor_confidence: f32,
    /// Age of the sensor data in milliseconds.
    pub sensor_freshness_ms: u64,
    /// Current rendering budget pressure (0.0 = relaxed, 1.0 = maxed out).
    pub budget_pressure: f32,
    /// Permission level of the update source.
    pub permission_level: PermissionLevel,
}

/// Permission level controlling what updates are allowed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PermissionLevel {
    /// Read-only access; updates are always deferred.
    ReadOnly,
    /// Standard update permissions.
    Standard,
    /// Elevated permissions; can override some thresholds.
    Elevated,
    /// Full administrative access; can force updates.
    Admin,
}

/// Tunable thresholds for the coherence gate.
#[derive(Clone, Debug)]
pub struct CoherencePolicy {
    /// Minimum entity continuity score to accept an update.
    pub accept_threshold: f32,
    /// Below this continuity, defer the update.
    pub defer_threshold: f32,
    /// Tile disagreement above this triggers a freeze.
    pub freeze_disagreement: f32,
    /// Tile disagreement above this triggers a rollback.
    pub rollback_disagreement: f32,
    /// Maximum acceptable age of sensor data in milliseconds.
    pub max_staleness_ms: u64,
    /// Budget pressure above this triggers a freeze.
    pub budget_freeze_threshold: f32,
}

impl Default for CoherencePolicy {
    fn default() -> Self {
        Self {
            accept_threshold: 0.7,
            defer_threshold: 0.4,
            freeze_disagreement: 0.8,
            rollback_disagreement: 0.95,
            max_staleness_ms: 5000,
            budget_freeze_threshold: 0.9,
        }
    }
}

/// The coherence gate controller.
///
/// Evaluates [`CoherenceInput`] against a [`CoherencePolicy`] to produce a
/// [`CoherenceDecision`]. The evaluation follows a priority order:
///
/// 1. Admin-level permissions always accept.
/// 2. Read-only permissions always defer.
/// 3. Stale data is deferred.
/// 4. High disagreement triggers rollback or freeze.
/// 5. High budget pressure triggers freeze.
/// 6. Entity continuity determines accept vs defer.
pub struct CoherenceGate {
    policy: CoherencePolicy,
}

impl CoherenceGate {
    /// Create a new coherence gate with the given policy.
    pub fn new(policy: CoherencePolicy) -> Self {
        Self { policy }
    }

    /// Create a coherence gate with default policy.
    pub fn with_defaults() -> Self {
        Self {
            policy: CoherencePolicy::default(),
        }
    }

    /// Evaluate the coherence of an update.
    #[inline]
    pub fn evaluate(&self, input: &CoherenceInput) -> CoherenceDecision {
        // Admin overrides all checks
        if input.permission_level == PermissionLevel::Admin {
            return CoherenceDecision::Accept;
        }

        // Read-only sources can never write
        if input.permission_level == PermissionLevel::ReadOnly {
            return CoherenceDecision::Defer;
        }

        // Stale sensor data is deferred
        if input.sensor_freshness_ms > self.policy.max_staleness_ms {
            return CoherenceDecision::Defer;
        }

        // Very high disagreement triggers rollback
        if input.tile_disagreement >= self.policy.rollback_disagreement {
            return CoherenceDecision::Rollback;
        }

        // High disagreement triggers freeze
        if input.tile_disagreement >= self.policy.freeze_disagreement {
            return CoherenceDecision::Freeze;
        }

        // Excessive budget pressure triggers freeze
        if input.budget_pressure >= self.policy.budget_freeze_threshold {
            return CoherenceDecision::Freeze;
        }

        // Low sensor confidence reduces effective continuity
        let effective_continuity = input.entity_continuity * input.sensor_confidence;

        // Elevated permissions get a small boost to effective continuity
        let effective_continuity = if input.permission_level == PermissionLevel::Elevated {
            (effective_continuity + 0.1).min(1.0)
        } else {
            effective_continuity
        };

        // Accept if continuity is above threshold
        if effective_continuity >= self.policy.accept_threshold {
            return CoherenceDecision::Accept;
        }

        // Defer if continuity is above defer threshold but below accept
        if effective_continuity >= self.policy.defer_threshold {
            return CoherenceDecision::Defer;
        }

        // Very low continuity triggers freeze
        CoherenceDecision::Freeze
    }

    /// Replace the current policy with a new one.
    pub fn update_policy(&mut self, policy: CoherencePolicy) {
        self.policy = policy;
    }

    /// Get a reference to the current policy.
    pub fn policy(&self) -> &CoherencePolicy {
        &self.policy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn good_input() -> CoherenceInput {
        CoherenceInput {
            tile_disagreement: 0.1,
            entity_continuity: 0.9,
            sensor_confidence: 1.0,
            sensor_freshness_ms: 100,
            budget_pressure: 0.3,
            permission_level: PermissionLevel::Standard,
        }
    }

    #[test]
    fn test_accept_good_input() {
        let gate = CoherenceGate::with_defaults();
        assert_eq!(gate.evaluate(&good_input()), CoherenceDecision::Accept);
    }

    #[test]
    fn test_admin_always_accepts() {
        let gate = CoherenceGate::with_defaults();
        let mut input = good_input();
        input.tile_disagreement = 1.0;
        input.entity_continuity = 0.0;
        input.permission_level = PermissionLevel::Admin;
        assert_eq!(gate.evaluate(&input), CoherenceDecision::Accept);
    }

    #[test]
    fn test_readonly_always_defers() {
        let gate = CoherenceGate::with_defaults();
        let mut input = good_input();
        input.permission_level = PermissionLevel::ReadOnly;
        assert_eq!(gate.evaluate(&input), CoherenceDecision::Defer);
    }

    #[test]
    fn test_stale_data_defers() {
        let gate = CoherenceGate::with_defaults();
        let mut input = good_input();
        input.sensor_freshness_ms = 10_000;
        assert_eq!(gate.evaluate(&input), CoherenceDecision::Defer);
    }

    #[test]
    fn test_high_disagreement_freezes() {
        let gate = CoherenceGate::with_defaults();
        let mut input = good_input();
        input.tile_disagreement = 0.85;
        assert_eq!(gate.evaluate(&input), CoherenceDecision::Freeze);
    }

    #[test]
    fn test_very_high_disagreement_rollbacks() {
        let gate = CoherenceGate::with_defaults();
        let mut input = good_input();
        input.tile_disagreement = 0.96;
        assert_eq!(gate.evaluate(&input), CoherenceDecision::Rollback);
    }

    #[test]
    fn test_budget_pressure_freezes() {
        let gate = CoherenceGate::with_defaults();
        let mut input = good_input();
        input.budget_pressure = 0.95;
        assert_eq!(gate.evaluate(&input), CoherenceDecision::Freeze);
    }

    #[test]
    fn test_low_continuity_defers() {
        let gate = CoherenceGate::with_defaults();
        let mut input = good_input();
        input.entity_continuity = 0.5;
        // effective = 0.5 * 1.0 = 0.5, above defer (0.4) but below accept (0.7)
        assert_eq!(gate.evaluate(&input), CoherenceDecision::Defer);
    }

    #[test]
    fn test_very_low_continuity_freezes() {
        let gate = CoherenceGate::with_defaults();
        let mut input = good_input();
        input.entity_continuity = 0.2;
        input.sensor_confidence = 0.5;
        // effective = 0.2 * 0.5 = 0.1, below defer (0.4)
        assert_eq!(gate.evaluate(&input), CoherenceDecision::Freeze);
    }

    #[test]
    fn test_elevated_permission_boost() {
        let gate = CoherenceGate::with_defaults();
        let mut input = good_input();
        input.entity_continuity = 0.65;
        input.sensor_confidence = 1.0;
        input.permission_level = PermissionLevel::Standard;
        // effective = 0.65, below accept (0.7) -> Defer
        assert_eq!(gate.evaluate(&input), CoherenceDecision::Defer);

        input.permission_level = PermissionLevel::Elevated;
        // effective = 0.65 + 0.1 = 0.75, above accept (0.7) -> Accept
        assert_eq!(gate.evaluate(&input), CoherenceDecision::Accept);
    }

    #[test]
    fn test_update_policy() {
        let mut gate = CoherenceGate::with_defaults();
        let mut strict = CoherencePolicy::default();
        strict.accept_threshold = 0.99;
        gate.update_policy(strict);

        let input = good_input(); // continuity 0.9, below new threshold 0.99
        assert_eq!(gate.evaluate(&input), CoherenceDecision::Defer);
    }
}
