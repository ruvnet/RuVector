//! Priority computation for the coherence-aware scheduler.
//!
//! Implements the priority formula from ADR-087 Section 5.1:
//!
//! ```text
//! score = deadline_urgency + novelty_boost - risk_penalty
//! ```
//!
//! Where:
//! - `deadline_urgency = 1.0 / (time_until_deadline_us + 1.0)` (EDF component)
//! - `novelty_boost = pending_input_novelty` (0.0..1.0)
//! - `risk_penalty = |min(pending_coherence_delta, 0.0)| * RISK_WEIGHT`

use crate::{Instant, TaskControlBlock};
use ruvix_types::SchedulerScore;

/// Default weight for coherence risk penalty.
pub const RISK_WEIGHT: f32 = 2.0;

/// Default maximum deadline urgency (for tasks with very short deadlines).
pub const MAX_DEADLINE_URGENCY: f32 = 5.0;

/// Default deadline urgency for tasks without deadlines.
pub const DEFAULT_DEADLINE_URGENCY: f32 = 1.0;

/// Configuration for priority computation.
#[derive(Debug, Clone, Copy)]
pub struct PriorityConfig {
    /// Weight for coherence risk penalty.
    pub risk_weight: f32,

    /// Maximum deadline urgency value (clamped).
    pub max_deadline_urgency: f32,

    /// Default urgency for tasks without deadlines.
    pub default_deadline_urgency: f32,

    /// Minimum coherence delta before penalty applies.
    pub coherence_penalty_threshold: f32,
}

impl Default for PriorityConfig {
    fn default() -> Self {
        Self {
            risk_weight: RISK_WEIGHT,
            max_deadline_urgency: MAX_DEADLINE_URGENCY,
            default_deadline_urgency: DEFAULT_DEADLINE_URGENCY,
            coherence_penalty_threshold: 0.0,
        }
    }
}

impl PriorityConfig {
    /// Creates a new priority configuration with custom risk weight.
    #[inline]
    #[must_use]
    pub const fn with_risk_weight(mut self, weight: f32) -> Self {
        self.risk_weight = weight;
        self
    }

    /// Creates a new priority configuration with custom max deadline urgency.
    #[inline]
    #[must_use]
    pub const fn with_max_deadline_urgency(mut self, urgency: f32) -> Self {
        self.max_deadline_urgency = urgency;
        self
    }
}

/// Computes the scheduler priority for a task.
///
/// This implements the priority formula from ADR-087 Section 5.1.
///
/// # Arguments
///
/// * `task` - The task control block
/// * `now` - Current time instant
/// * `config` - Priority computation configuration
///
/// # Returns
///
/// A `SchedulerScore` containing the computed priority and its components.
///
/// # Example
///
/// ```
/// use ruvix_sched::{compute_priority, PriorityConfig, TaskControlBlock, Instant};
/// use ruvix_types::{TaskHandle, TaskPriority};
/// use ruvix_cap::CapRights;
///
/// let task = TaskControlBlock::new(
///     TaskHandle::new(1, 0),
///     CapRights::READ,
///     TaskPriority::Normal,
///     0,
/// )
/// .with_novelty(0.5)
/// .with_coherence_delta(-0.1);
///
/// let now = Instant::from_micros(0);
/// let config = PriorityConfig::default();
/// let score = compute_priority(&task, now, &config);
///
/// assert!(score.novelty_boost > 0.0);
/// assert!(score.risk_penalty > 0.0);
/// ```
#[inline]
#[must_use]
pub fn compute_priority(task: &TaskControlBlock, now: Instant, config: &PriorityConfig) -> SchedulerScore {
    // Compute deadline urgency (EDF component)
    let deadline_urgency = compute_deadline_urgency(task, now, config);

    // Novelty boost (0.0..1.0)
    let novelty_boost = task.pending_input_novelty;

    // Risk penalty (only for negative coherence deltas)
    let risk_penalty = compute_risk_penalty(task, config);

    SchedulerScore::new(deadline_urgency, novelty_boost, risk_penalty)
}

/// Computes the deadline urgency component.
///
/// For tasks with deadlines, this returns `1.0 / (time_until_deadline_us + 1.0)`,
/// clamped to `[0.0, max_deadline_urgency]`.
///
/// For tasks without deadlines, returns the default urgency value.
#[inline]
fn compute_deadline_urgency(task: &TaskControlBlock, now: Instant, config: &PriorityConfig) -> f32 {
    match task.deadline {
        Some(deadline) => {
            let time_until = deadline.saturating_duration_since(now);
            let time_us = time_until.as_micros() as f64;

            // Avoid division by zero and compute inverse urgency
            let urgency = (1.0 / (time_us + 1.0)) as f32;

            // Scale and clamp
            let scaled = urgency * config.max_deadline_urgency * 1_000_000.0;
            scaled.clamp(0.0, config.max_deadline_urgency)
        }
        None => {
            // No deadline: use base priority weight
            let base = match task.priority {
                ruvix_types::TaskPriority::Idle => 0.25,
                ruvix_types::TaskPriority::Normal => 1.0,
                ruvix_types::TaskPriority::High => 2.0,
                ruvix_types::TaskPriority::RealTime => 3.0,
                ruvix_types::TaskPriority::Critical => 4.0,
            };
            base * config.default_deadline_urgency
        }
    }
}

/// Computes the risk penalty for negative coherence deltas.
///
/// Only applies when `pending_coherence_delta < coherence_penalty_threshold`.
#[inline]
fn compute_risk_penalty(task: &TaskControlBlock, config: &PriorityConfig) -> f32 {
    if task.pending_coherence_delta < config.coherence_penalty_threshold {
        let delta = task.pending_coherence_delta - config.coherence_penalty_threshold;
        (delta.abs() * config.risk_weight).min(config.risk_weight * 2.0)
    } else {
        0.0
    }
}

/// Computes priority with default configuration.
///
/// Convenience function that uses `PriorityConfig::default()`.
#[inline]
#[must_use]
pub fn compute_priority_default(task: &TaskControlBlock, now: Instant) -> SchedulerScore {
    compute_priority(task, now, &PriorityConfig::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvix_cap::CapRights;
    use ruvix_types::TaskHandle;

    fn make_task() -> TaskControlBlock {
        TaskControlBlock::new(
            TaskHandle::new(1, 0),
            CapRights::READ,
            ruvix_types::TaskPriority::Normal,
            0,
        )
    }

    #[test]
    fn test_basic_priority() {
        let task = make_task();
        let now = Instant::from_micros(0);
        let config = PriorityConfig::default();

        let score = compute_priority(&task, now, &config);

        // Normal task, no deadline, no novelty, no risk
        assert!(score.deadline_urgency > 0.0);
        assert!(score.novelty_boost.abs() < 0.001);
        assert!(score.risk_penalty.abs() < 0.001);
    }

    #[test]
    fn test_deadline_urgency() {
        let now = Instant::from_micros(0);
        let config = PriorityConfig::default();

        // Task with tight deadline (1ms away)
        let tight_deadline = make_task().with_deadline(Instant::from_micros(1000));
        let tight_score = compute_priority(&tight_deadline, now, &config);

        // Task with loose deadline (1s away)
        let loose_deadline = make_task().with_deadline(Instant::from_micros(1_000_000));
        let loose_score = compute_priority(&loose_deadline, now, &config);

        // Tight deadline should have higher urgency
        assert!(
            tight_score.deadline_urgency > loose_score.deadline_urgency,
            "tight {} > loose {}",
            tight_score.deadline_urgency,
            loose_score.deadline_urgency
        );
    }

    #[test]
    fn test_novelty_boost() {
        let now = Instant::from_micros(0);
        let config = PriorityConfig::default();

        let low_novelty = make_task().with_novelty(0.1);
        let high_novelty = make_task().with_novelty(0.9);

        let low_score = compute_priority(&low_novelty, now, &config);
        let high_score = compute_priority(&high_novelty, now, &config);

        assert!(
            high_score.novelty_boost > low_score.novelty_boost,
            "high {} > low {}",
            high_score.novelty_boost,
            low_score.novelty_boost
        );
        assert!(
            high_score.score > low_score.score,
            "high {} > low {}",
            high_score.score,
            low_score.score
        );
    }

    #[test]
    fn test_risk_penalty() {
        let now = Instant::from_micros(0);
        let config = PriorityConfig::default();

        let safe = make_task().with_coherence_delta(0.1);
        let risky = make_task().with_coherence_delta(-0.5);

        let safe_score = compute_priority(&safe, now, &config);
        let risky_score = compute_priority(&risky, now, &config);

        // Safe task should have no penalty
        assert!(safe_score.risk_penalty.abs() < 0.001);

        // Risky task should have penalty
        assert!(risky_score.risk_penalty > 0.0);

        // Safe task should have higher overall score
        assert!(
            safe_score.score > risky_score.score,
            "safe {} > risky {}",
            safe_score.score,
            risky_score.score
        );
    }

    #[test]
    fn test_combined_signals() {
        let now = Instant::from_micros(0);
        let config = PriorityConfig::default();

        // Novel but risky task
        let novel_risky = make_task()
            .with_novelty(0.8)
            .with_coherence_delta(-0.3);

        // Not novel but safe task
        let safe_boring = make_task()
            .with_novelty(0.1)
            .with_coherence_delta(0.2);

        let novel_score = compute_priority(&novel_risky, now, &config);
        let safe_score = compute_priority(&safe_boring, now, &config);

        // Both signals should be present
        assert!(novel_score.novelty_boost > safe_score.novelty_boost);
        assert!(novel_score.risk_penalty > safe_score.risk_penalty);
    }

    #[test]
    fn test_priority_levels() {
        let now = Instant::from_micros(0);
        let config = PriorityConfig::default();

        let idle = TaskControlBlock::new(
            TaskHandle::new(1, 0),
            CapRights::READ,
            ruvix_types::TaskPriority::Idle,
            0,
        );

        let critical = TaskControlBlock::new(
            TaskHandle::new(2, 0),
            CapRights::READ,
            ruvix_types::TaskPriority::Critical,
            0,
        );

        let idle_score = compute_priority(&idle, now, &config);
        let critical_score = compute_priority(&critical, now, &config);

        assert!(
            critical_score.deadline_urgency > idle_score.deadline_urgency,
            "critical {} > idle {}",
            critical_score.deadline_urgency,
            idle_score.deadline_urgency
        );
    }

    #[test]
    fn test_deadline_passed() {
        let now = Instant::from_micros(1000);
        let config = PriorityConfig::default();

        // Deadline already passed
        let passed = make_task().with_deadline(Instant::from_micros(500));
        let score = compute_priority(&passed, now, &config);

        // Should have maximum urgency (clamped)
        assert!((score.deadline_urgency - config.max_deadline_urgency).abs() < 0.001);
    }

    #[test]
    fn test_custom_risk_weight() {
        let now = Instant::from_micros(0);
        let low_weight = PriorityConfig::default().with_risk_weight(1.0);
        let high_weight = PriorityConfig::default().with_risk_weight(4.0);

        let risky = make_task().with_coherence_delta(-0.5);

        let low_score = compute_priority(&risky, now, &low_weight);
        let high_score = compute_priority(&risky, now, &high_weight);

        assert!(
            high_score.risk_penalty > low_score.risk_penalty,
            "high {} > low {}",
            high_score.risk_penalty,
            low_score.risk_penalty
        );
    }
}
