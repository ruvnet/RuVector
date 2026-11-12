//! Errors and halt reasons for the monitoring ladder (PIR WP33, ADR-337).
//!
//! [`MonitorError`] surfaces from *construction*, where a misconfigured
//! ladder must fail loudly rather than run as a silent no-op. At inspection
//! time errors do not surface at all: they become
//! [`HaltReason`](crate::monitor::HaltReason) on a
//! [`MonitorOutcome::Halt`](crate::monitor::MonitorOutcome), so a caller has
//! no fallible value to coerce into an allow.

/// Errors from configuring or running the ladder.
///
/// Note these surface from *construction*, where failing loudly is right. At
/// inspection time an error becomes [`MonitorOutcome::Halt`] instead.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum MonitorError {
    /// A value that must be finite was not. Reported before any comparison
    /// consumes it — `NaN` silently passes comparison gates in every
    /// direction, which is how such a gate fails open unnoticed.
    #[error("{field} must be finite, got {value}")]
    NonFinite {
        /// Field that failed.
        field: &'static str,
        /// The offending value, rendered.
        value: String,
    },
    /// Risk outside `[0, 1]`.
    #[error("risk must be within [0, 1], got {risk}")]
    RiskOutOfRange {
        /// The offending risk.
        risk: f64,
    },
    /// Uncertainty was zero or negative.
    #[error("uncertainty must be > 0, got {uncertainty}")]
    NonPositiveUncertainty {
        /// The offending uncertainty.
        uncertainty: f64,
    },
    /// A detector reported an uncertainty above 1 on a unit-scale risk.
    ///
    /// Refused because it is meaningless there, and because the only thing
    /// an inflated uncertainty does is raise the calibration ceiling — the
    /// guard that exists to catch a never-escalate ladder would otherwise be
    /// the lever for waving one through.
    #[error("detector uncertainty must be within (0, 1] on a unit-scale risk, got {uncertainty}")]
    UncertaintyOutOfRange {
        /// The offending uncertainty.
        uncertainty: f64,
    },
    /// A detector saturation of zero would divide by zero on every score.
    #[error("detector saturation must be > 0")]
    ZeroSaturation,
    /// A ladder with no rungs can never inspect anything, so a mandatory
    /// class could not be honoured.
    #[error("ladder must have at least one rung")]
    EmptyLadder,
    /// A rung with zero monetized cost is purchased forever: value of
    /// information decays geometrically but never reaches zero, so a free
    /// rung always has positive net value. Refused at construction.
    #[error("rung {index} ('{name}') has zero monetized cost; every rung must cost > 0")]
    FreeRung {
        /// Position in the ladder.
        index: usize,
        /// Rung name.
        name: String,
    },
    /// Rungs are not ordered cheap-to-expensive.
    #[error(
        "rung {index} ('{name}') costs {cost} but the previous rung costs {previous}; \
         rungs must be ordered cheapest first"
    )]
    UnorderedRungs {
        /// Position in the ladder.
        index: usize,
        /// Rung name.
        name: String,
        /// This rung's monetized cost.
        cost: f64,
        /// The preceding rung's monetized cost.
        previous: f64,
    },
    /// The round cap was zero, which would prevent even a mandatory
    /// inspection from running.
    #[error("max_rounds must be >= 1")]
    ZeroRounds,
    /// A threshold outside `[0, 1]`.
    #[error("{field} must be within [0, 1], got {value}")]
    ThresholdOutOfRange {
        /// Field that failed.
        field: &'static str,
        /// The offending value.
        value: f64,
    },
    /// No rung can ever be purchased at this `value_of_success`, so the
    /// ladder is a never-escalate switch that still looks configured.
    ///
    /// The ceiling on any rung's worth is `value_of_success × σ/√(2π)`
    /// (≈`0.4σ`). With a unit-scale risk and a typical `σ ≈ 0.05`, leaving
    /// `value_of_success` at a nominal `1.0` makes anything costing more than
    /// about two cents permanently unpurchasable.
    #[error(
        "miscalibrated: cheapest rung costs {cheapest_cost} but no purchase can be worth more \
         than {ceiling} (value_of_success {value_of_success} × σ {uncertainty} / √(2π)); \
         express value_of_success as the currency value of catching one violation"
    )]
    Miscalibrated {
        /// Cheapest rung's monetized cost.
        cheapest_cost: f64,
        /// Maximum attainable value of a purchase.
        ceiling: f64,
        /// Configured value of success.
        value_of_success: f64,
        /// Expected detector uncertainty used for the check.
        uncertainty: f64,
    },
    /// The underlying value-of-information primitive rejected an input.
    #[error("value-of-information error: {0}")]
    Voi(String),
}

/// Why the ladder refused an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HaltReason {
    /// Assessed risk reached the critical threshold.
    CriticalRisk,
    /// The tiny detector could not score the operation.
    DetectorFailed,
    /// An investigator rung failed.
    InvestigatorFailed,
    /// An investigator returned a non-finite or out-of-range observation.
    InvalidObservation,
    /// The value-of-information decision itself errored.
    DecisionFailed,
    /// The Bayesian belief update failed.
    BeliefUpdateFailed,
}

impl HaltReason {
    /// Stable identifier for logs and receipts.
    pub const fn as_str(self) -> &'static str {
        match self {
            HaltReason::CriticalRisk => "critical_risk",
            HaltReason::DetectorFailed => "detector_failed",
            HaltReason::InvestigatorFailed => "investigator_failed",
            HaltReason::InvalidObservation => "invalid_observation",
            HaltReason::DecisionFailed => "decision_failed",
            HaltReason::BeliefUpdateFailed => "belief_update_failed",
        }
    }
}
