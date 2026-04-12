//! Concrete policy registry implementation.

use crate::model::AxisScores;

/// Single axis constraint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AxisConstraint {
    /// Minimum acceptable axis value in `[0, 1]`.
    pub min: f32,
    /// Maximum acceptable axis value in `[0, 1]`.
    pub max: f32,
}

impl AxisConstraint {
    /// Constraint requiring `>= min`.
    pub fn min(value: f32) -> Self {
        Self {
            min: value,
            max: 1.0,
        }
    }
    /// Constraint bounded above by `<= max`.
    pub fn max(value: f32) -> Self {
        Self {
            min: 0.0,
            max: value,
        }
    }
    /// No constraint (score is always 1.0).
    pub fn any() -> Self {
        Self {
            min: 0.0,
            max: 1.0,
        }
    }
    /// Score how well `value` satisfies the constraint in `[0, 1]`.
    ///
    /// `1.0` when strictly inside `[min, max]`, linearly falling off outside.
    pub fn score(&self, value: f32) -> f32 {
        let v = value.clamp(0.0, 1.0);
        if v < self.min {
            // Distance to min, normalized by min (avoid div by zero).
            let denom = self.min.max(1e-3);
            ((v / denom).clamp(0.0, 1.0)).clamp(0.0, 1.0)
        } else if v > self.max {
            let slack = (1.0 - self.max).max(1e-3);
            (1.0 - ((v - self.max) / slack).min(1.0)).clamp(0.0, 1.0)
        } else {
            1.0
        }
    }
}

/// Constraint bundle over all four axes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AxisConstraints {
    /// Limit axis constraint.
    pub limit: AxisConstraint,
    /// Care axis constraint.
    pub care: AxisConstraint,
    /// Bridge axis constraint.
    pub bridge: AxisConstraint,
    /// Clarity axis constraint.
    pub clarity: AxisConstraint,
}

impl AxisConstraints {
    /// No-op bundle — every axis is `any()`.
    pub fn any() -> Self {
        Self {
            limit: AxisConstraint::any(),
            care: AxisConstraint::any(),
            bridge: AxisConstraint::any(),
            clarity: AxisConstraint::any(),
        }
    }
}

/// Policy definition — name, bitmask, required axes.
#[derive(Debug, Clone)]
pub struct Policy {
    /// Stable id.
    pub id: u64,
    /// Human-readable name.
    pub name: String,
    /// Bitmask — policies apply when `(node.policy_mask & policy.mask) != 0`.
    pub mask: u64,
    /// Axis constraints this policy enforces.
    pub required_axes: AxisConstraints,
}

/// Policy registry.
#[derive(Debug, Clone, Default)]
pub struct PolicyRegistry {
    policies: Vec<Policy>,
}

impl PolicyRegistry {
    /// Empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a policy.
    pub fn register(&mut self, policy: Policy) {
        self.policies.push(policy);
    }

    /// Number of registered policies.
    pub fn len(&self) -> usize {
        self.policies.len()
    }

    /// `true` if no policies have been registered.
    pub fn is_empty(&self) -> bool {
        self.policies.is_empty()
    }

    /// Product of axis-constraint satisfaction scores for every policy that
    /// matches `mask`. Policies that do not touch this node contribute `1.0`.
    pub fn policy_fit(&self, axes: &AxisScores, mask: u64) -> f32 {
        if self.policies.is_empty() {
            return 1.0;
        }
        let mut score = 1.0_f32;
        let mut matched = 0;
        for p in &self.policies {
            if p.mask & mask == 0 {
                continue;
            }
            matched += 1;
            let s = p.required_axes.limit.score(axes.limit)
                * p.required_axes.care.score(axes.care)
                * p.required_axes.bridge.score(axes.bridge)
                * p.required_axes.clarity.score(axes.clarity);
            score *= s;
        }
        if matched == 0 {
            1.0
        } else {
            score.clamp(0.0, 1.0)
        }
    }

    /// `1 - policy_fit`.
    pub fn policy_risk(&self, axes: &AxisScores, mask: u64) -> f32 {
        1.0 - self.policy_fit(axes, mask)
    }
}
