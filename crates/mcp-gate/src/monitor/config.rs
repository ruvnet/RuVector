//! Ladder rungs, economic configuration, and the investigator trait
//! (PIR WP33, ADR-337).

use ruvector_voi::{EstimatorSpec, VoiConfig};

use super::detector::InspectionSubject;
use super::MonitorError;

/// One purchasable investigator.
#[derive(Debug, Clone, PartialEq)]
pub struct LadderRung {
    /// Identifier for logs and receipts.
    pub name: String,
    /// Cost, latency, and observation noise of this investigator.
    pub spec: EstimatorSpec,
}

impl LadderRung {
    /// Build a rung.
    pub fn new(name: impl Into<String>, cost: f64, latency_us: f64, noise_std: f64) -> Self {
        Self {
            name: name.into(),
            spec: EstimatorSpec {
                cost,
                latency_us,
                noise_std,
            },
        }
    }

    /// A noiseless rung is an oracle: its observation *is* the truth, so the
    /// ladder takes it as ground truth and exits rather than folding it into
    /// a posterior. The underlying primitive's `observe` refuses `noise_std
    /// == 0` for exactly this reason.
    pub fn is_oracle(&self) -> bool {
        self.spec.noise_std == 0.0
    }
}

/// Economic and threshold configuration.
///
/// Deliberately contains no per-class enable/disable flag, allowlist, or
/// bypass token: the mandatory classes are not configurable, so there is
/// nothing here that can narrow them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MonitorConfig {
    /// Value of catching one violation, and the price of latency.
    pub voi: VoiConfig,
    /// Risk level at which the allow/inspect decision is most ambiguous;
    /// used as the outside option in the value-of-information evaluation, so
    /// information is worth most when risk sits near it.
    pub escalation_threshold: f64,
    /// Risk at or above which the operation is halted outright.
    pub halt_risk: f64,
    /// Hard cap on investigator purchases per operation. Bounds the loop
    /// independently of the economics.
    ///
    /// With [`VoiConfig::latency_price`] at zero this is the **only** thing
    /// bounding an operation's wall-clock monitoring cost, because a rung
    /// declaring ten seconds of latency then costs exactly what an instant
    /// rung costs. Price latency if the overhead budget is meant to be
    /// enforced economically rather than merely capped.
    pub max_rounds: usize,
}

impl Default for MonitorConfig {
    /// Defaults calibrated against the reference two-rung ladder (a $0.01 /
    /// 5 ms verifier at σ=0.15 and a $0.50 / 200 ms investigator at σ=0.05)
    /// so that both calibration failure modes are avoided.
    ///
    /// `value_of_success` is bounded on *both* sides, and the window is
    /// narrower than it looks. Too low and nothing is ever purchasable
    /// ([`MonitorError::Miscalibrated`] catches that at construction). Too
    /// high and every operation is worth investigating, including zero-risk
    /// ones — which silently blows the overhead budget while looking
    /// perfectly configured, and which construction *cannot* catch, because
    /// such a ladder is well-formed. At 100 a zero-risk operation is not
    /// worth investigating while one sitting at the escalation threshold is.
    ///
    /// `latency_price` defaults to `1e-6`, i.e. **one dollar per second** of
    /// added latency. A zero default would make the wall-clock cost of a rung
    /// economically invisible: a ten-second investigator would price
    /// identically to an instant one, and only [`Self::max_rounds`] would
    /// bound the damage. Pricing latency is what lets the ladder trade
    /// against an overhead budget at all.
    ///
    /// Re-derive these when the rungs or the detector's uncertainty change;
    /// [`rung_is_purchasable`](crate::monitor::rung_is_purchasable) answers
    /// the question directly for a given belief.
    fn default() -> Self {
        Self {
            voi: VoiConfig {
                // Currency value of catching one violation.
                value_of_success: 100.0,
                // $1 per second of added latency.
                latency_price: 1e-6,
            },
            escalation_threshold: 0.5,
            halt_risk: 0.95,
            max_rounds: 4,
        }
    }
}

/// Supplies an investigator's verdict: an observed violation probability.
pub trait Investigator {
    /// Investigate `subject` at `rung`, returning an observed violation
    /// probability in `[0, 1]`. An `Err` halts the operation — investigators
    /// fail closed.
    fn investigate(
        &self,
        subject: &InspectionSubject,
        rung: &LadderRung,
    ) -> Result<f64, MonitorError>;
}
