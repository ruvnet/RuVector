//! Adaptive runtime monitoring via a value-of-information escalation ladder
//! (PIR WP33, ADR-337).
//!
//! A cheap detector scores every operation; progressively more expensive
//! investigators are purchased only when the purchase pays for itself:
//!
//! ```text
//! stream → tiny detector → risk score
//!   low risk      → continue
//!   medium risk   → local verifier
//!   high risk     → strong investigator
//!   critical      → halt
//! ```
//!
//! The routing rule is *investigate when `P(violation) × damage > verification
//! cost`*. That is exactly the purchase inequality of the Wave-4
//! value-of-information primitive
//! ([`ruvector_voi`]), with `value_of_success` carrying the
//! damage term — so this module **reuses** that primitive rather than
//! restating its mathematics. Each [`LadderRung`] is one
//! [`EstimatorSpec`]: cheap and noisy first, expensive and sharp last.
//!
//! # Two things that are not economic
//!
//! **Mandatory classes.** Privilege escalation, network access, credential
//! use, runtime mutation, and destructive operations are inspected at 100%
//! regardless of what the economics say. Classification runs *before* the
//! value-of-information decision and cannot be disabled: see
//! [`classes`] for why that is structural rather than conventional.
//!
//! **Failure.** [`EscalationLadder::inspect`] does not return a `Result`. A
//! detector error, an investigator error, a non-finite observation, or a
//! failed belief update all produce [`MonitorOutcome::Halt`]. This is
//! deliberate: a fallible signature invites `.unwrap_or(allow)` at the call
//! site, and a monitoring gate that fails open under error is worse than no
//! gate, because it looks like one.
//!
//! # Design target, and what is actually enforced
//!
//! The target is under 5% average monitoring overhead with 100%
//! mandatory-class coverage. [`OverheadAccount`] measures the first and
//! [`OverheadAccount::mandatory_coverage`] reports the second against a
//! denominator, so both are checkable rather than asserted.
//!
//! Only the coverage half is *enforced*. Overhead is bounded by
//! [`MonitorConfig::max_rounds`] and priced by [`VoiConfig::latency_price`],
//! but nothing refuses a ladder whose rungs are simply slow. Measured on the
//! reference ladder (5 ms / 200 ms rungs, 200 operations, one in ten
//! mandatory, 20 ms attributed per operation):
//!
//! | Case | Overhead | Rungs |
//! |---|---|---|
//! | Investigator resolves the ambiguity (reports 0.2) | **2.6%** | 20 |
//! | Investigator leaves it ambiguous (reports 0.5), latency priced | **108%** | 80 |
//! | Same, latency unpriced | **205%** | 80 |
//!
//! So the target holds only when the first investigator actually settles the
//! question. When it does not, the belief stays near the escalation
//! threshold, every further rung still looks worth buying, and the ladder
//! spends [`MonitorConfig::max_rounds`] rungs on every operation it
//! *inspects* — four
//! rounds of a 200 ms investigator against a 20 ms workload is 40× the
//! budget on its own.
//!
//! Pricing latency halves that (205% → 108%) by shifting the *mix* toward
//! cheaper rungs, but it does not reduce the *count*: at
//! `value_of_success = 100` even a priced 200 ms rung is worth buying at
//! maximum ambiguity. **`max_rounds` is the only hard bound on wall-clock
//! cost.** Treat 5% as a budget to configure toward — chiefly by keeping
//! `max_rounds` small and the top rung fast — not as a property this module
//! guarantees.
//!
//! # Not wired into the request path
//!
//! This is a library module with **no call site** in the gate's request
//! handling: `server.rs`, `tools.rs`, `types.rs`, and `main.rs` do not
//! reference it. Nothing here is protecting production traffic yet. Wiring
//! it into [`crate::tools::McpGateTools`] is separate work.

pub mod classes;
pub mod config;
pub mod detector;
pub mod error;
pub mod overhead;

use std::time::Instant;

use ruvector_voi::{
    decide, observe, voi_upper_bound, Belief, EstimatorSpec, VoiConfig, VoiDecision,
};

pub use classes::{Classification, MandatoryClass};
pub use config::{Investigator, LadderRung, MonitorConfig};
pub use detector::{InspectionSubject, KeywordDetector, RiskSignal, TinyDetector};
pub use error::{HaltReason, MonitorError};
pub use overhead::OverheadAccount;

/// Result of monitoring one operation.
///
/// Two variants only, and no `Default`: there is no neutral outcome to fall
/// back to. Branch with [`MonitorOutcome::permits_execution`].
#[derive(Debug, Clone, PartialEq)]
#[must_use = "an unexamined monitoring outcome permits an unmonitored operation"]
pub enum MonitorOutcome {
    /// The operation may proceed.
    Allow {
        /// Final assessed risk.
        risk: f64,
        /// Investigator rungs purchased.
        rungs_purchased: usize,
        /// Mandatory classes this operation belonged to.
        mandatory: Vec<MandatoryClass>,
    },
    /// The operation is refused. Terminal.
    Halt {
        /// Why.
        reason: HaltReason,
        /// Risk at the point of refusal, when one was established.
        risk: Option<f64>,
        /// Mandatory classes this operation belonged to.
        mandatory: Vec<MandatoryClass>,
    },
}

impl MonitorOutcome {
    /// Whether the operation may proceed. The only sanctioned way to branch.
    pub fn permits_execution(&self) -> bool {
        matches!(self, MonitorOutcome::Allow { .. })
    }

    /// Mandatory classes recorded for the operation, whatever the outcome.
    pub fn mandatory(&self) -> &[MandatoryClass] {
        match self {
            MonitorOutcome::Allow { mandatory, .. } => mandatory,
            MonitorOutcome::Halt { mandatory, .. } => mandatory,
        }
    }
}

/// The escalation ladder.
pub struct EscalationLadder<D: TinyDetector, I: Investigator> {
    rungs: Vec<LadderRung>,
    specs: Vec<EstimatorSpec>,
    config: MonitorConfig,
    detector: D,
    investigator: I,
    account: OverheadAccount,
}

/// Rendered without the detector or investigator, which need not be `Debug`.
impl<D: TinyDetector, I: Investigator> std::fmt::Debug for EscalationLadder<D, I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EscalationLadder")
            .field("rungs", &self.rungs)
            .field("config", &self.config)
            .field("account", &self.account)
            .finish_non_exhaustive()
    }
}

impl<D: TinyDetector, I: Investigator> EscalationLadder<D, I> {
    /// Build a ladder, refusing every configuration that would make it
    /// silently useless.
    ///
    /// Rejects: an empty ladder; any rung with zero monetized cost (which
    /// would be purchased forever); rungs not ordered cheapest-first; a zero
    /// round cap; thresholds outside `[0, 1]`; and a `value_of_success` so
    /// low that no rung is ever purchasable.
    pub fn new(
        rungs: Vec<LadderRung>,
        config: MonitorConfig,
        detector: D,
        investigator: I,
    ) -> Result<Self, MonitorError> {
        if rungs.is_empty() {
            return Err(MonitorError::EmptyLadder);
        }
        if config.max_rounds == 0 {
            return Err(MonitorError::ZeroRounds);
        }
        for (field, value) in [
            ("escalation_threshold", config.escalation_threshold),
            ("halt_risk", config.halt_risk),
        ] {
            if !value.is_finite() {
                return Err(MonitorError::NonFinite {
                    field,
                    value: format!("{value}"),
                });
            }
            if !(0.0..=1.0).contains(&value) {
                return Err(MonitorError::ThresholdOutOfRange { field, value });
            }
        }
        config.voi.validate().map_err(voi_err)?;
        // The uncertainty comes from the detector rather than configuration,
        // so it cannot diverge from what `score` actually reports. Bounded to
        // (0, 1] like risk itself: an uncertainty above 1 is meaningless on a
        // unit-scale risk, and would only serve to inflate the ceiling below.
        let detector_uncertainty = detector.expected_uncertainty();
        RiskSignal::new(0.0, detector_uncertainty)?;
        if detector_uncertainty > 1.0 {
            return Err(MonitorError::UncertaintyOutOfRange {
                uncertainty: detector_uncertainty,
            });
        }

        let mut previous = f64::NEG_INFINITY;
        for (index, rung) in rungs.iter().enumerate() {
            rung.spec.validate().map_err(voi_err)?;
            let cost = monetized_cost(&rung.spec, &config.voi);
            if cost <= 0.0 {
                return Err(MonitorError::FreeRung {
                    index,
                    name: rung.name.clone(),
                });
            }
            if cost < previous {
                return Err(MonitorError::UnorderedRungs {
                    index,
                    name: rung.name.clone(),
                    cost,
                    previous,
                });
            }
            previous = cost;
        }

        // Calibration: if the cheapest rung costs more than the most any
        // purchase could ever be worth, the ladder can never escalate on
        // economics alone.
        //
        // The ceiling is computed from the *predictive* standard deviation
        // `s = σ/√(1+(τ/σ)²)`, not from σ itself. A noisy rung reveals
        // strictly less than the full prior uncertainty, and this design
        // deliberately makes the cheapest rung the noisiest — so using σ
        // would overstate the ceiling by up to ~5× at realistic noise levels
        // and wave through ladders that then purchase nothing. `voi_upper_bound`
        // is exactly this quantity, and is the same call `rung_is_purchasable`
        // makes.
        let cheapest_cost = monetized_cost(&rungs[0].spec, &config.voi);
        let calibration_belief =
            Belief::new(config.escalation_threshold, detector_uncertainty).map_err(voi_err)?;
        let bound =
            voi_upper_bound(calibration_belief, rungs[0].spec.noise_std).map_err(voi_err)?;
        let ceiling = config.voi.value_of_success * bound;
        if cheapest_cost > ceiling {
            return Err(MonitorError::Miscalibrated {
                cheapest_cost,
                ceiling,
                value_of_success: config.voi.value_of_success,
                uncertainty: detector_uncertainty,
            });
        }

        let specs = rungs.iter().map(|r| r.spec).collect();
        Ok(Self {
            rungs,
            specs,
            config,
            detector,
            investigator,
            account: OverheadAccount::new(),
        })
    }

    /// Monitoring totals so far.
    pub fn account(&self) -> &OverheadAccount {
        &self.account
    }

    /// The configured rungs.
    pub fn rungs(&self) -> &[LadderRung] {
        &self.rungs
    }

    /// The investigator, so a caller can cross-check the ladder's counters
    /// against the investigator's own tally rather than trusting one side.
    pub fn investigator(&self) -> &I {
        &self.investigator
    }

    /// Record the cost of an operation the caller executed, so
    /// [`OverheadAccount::fraction`] has a denominator.
    pub fn record_workload_us(&mut self, micros: u128) {
        self.account.record_workload(micros);
    }

    /// Monitor one operation.
    ///
    /// Infallible by design: every internal error becomes
    /// [`MonitorOutcome::Halt`], so there is no fallible result a caller can
    /// coerce into an allow.
    pub fn inspect(&mut self, subject: &InspectionSubject) -> MonitorOutcome {
        let started = Instant::now();
        let outcome = self.run(subject);
        self.account
            .record_monitoring(started.elapsed().as_micros());
        if let MonitorOutcome::Halt { .. } = outcome {
            self.account.record_halt();
        }
        outcome
    }

    fn run(&mut self, subject: &InspectionSubject) -> MonitorOutcome {
        // Classification is computed here, from the subject the ladder was
        // given. It is never accepted as a parameter, so it cannot be forged.
        let classification = classes::classify(subject);
        let mandatory: Vec<MandatoryClass> = classification.matched().to_vec();
        let is_mandatory = classification.is_mandatory();
        if is_mandatory {
            // Recorded here, before any decision can divert the operation, so
            // `mandatory_coverage` has an honest denominator.
            self.account.record_mandatory_seen();
        }

        let signal = match self.detector.score(subject) {
            Ok(s) => s,
            Err(_) => return halt(HaltReason::DetectorFailed, None, mandatory),
        };
        if signal.risk() >= self.config.halt_risk {
            return halt(HaltReason::CriticalRisk, Some(signal.risk()), mandatory);
        }

        let mut belief = match Belief::new(signal.risk(), signal.uncertainty()) {
            Ok(b) => b,
            Err(_) => return halt(HaltReason::DetectorFailed, Some(signal.risk()), mandatory),
        };

        // A mandatory class forces the first purchase, bypassing the
        // economics entirely. The economics still govern how far *beyond*
        // rung 0 the ladder climbs.
        let mut force_purchase = is_mandatory;
        let mut rungs_purchased = 0usize;
        let mut inspected = false;

        while rungs_purchased < self.config.max_rounds {
            let index = if force_purchase {
                force_purchase = false;
                0
            } else {
                match decide(
                    belief,
                    self.config.escalation_threshold,
                    &self.specs,
                    &self.config.voi,
                ) {
                    Ok(VoiDecision::Buy(i)) => i,
                    Ok(VoiDecision::Route) => break,
                    Err(_) => {
                        return halt(HaltReason::DecisionFailed, Some(belief.mean()), mandatory)
                    }
                }
            };

            let rung = &self.rungs[index];
            let observation = match self.investigator.investigate(subject, rung) {
                Ok(v) => v,
                Err(_) => {
                    return halt(
                        HaltReason::InvestigatorFailed,
                        Some(belief.mean()),
                        mandatory,
                    )
                }
            };
            // The rung ran and was paid for, whatever it returned. Count it
            // before validating the observation, or an investigator that
            // fails validation is billed but never recorded.
            rungs_purchased += 1;
            inspected = true;
            self.account.record_rung();
            // Charge the rung's *declared* latency, not the wall-clock of
            // whatever ran. A fixture investigator returns instantly, so
            // wall-clock alone would report an overhead figure that says more
            // about the test double than about the ladder as configured.
            // Declared latency is also what the economics priced, so the
            // accounting and the purchase decision agree on what a rung costs.
            self.account
                .record_monitoring(rung.spec.latency_us.max(0.0) as u128);

            // Validate before any comparison: a non-finite observation would
            // otherwise pass the halt check silently.
            if !observation.is_finite() || !(0.0..=1.0).contains(&observation) {
                return halt(
                    HaltReason::InvalidObservation,
                    Some(belief.mean()),
                    mandatory,
                );
            }

            if rung.is_oracle() {
                // Ground truth: take it and exit without calling `observe`,
                // which refuses noiseless specs.
                self.account.record_inspection(is_mandatory);
                return if observation >= self.config.halt_risk {
                    halt(HaltReason::CriticalRisk, Some(observation), mandatory)
                } else {
                    MonitorOutcome::Allow {
                        risk: observation,
                        rungs_purchased,
                        mandatory,
                    }
                };
            }

            belief = match observe(belief, &rung.spec, observation) {
                Ok(b) => b,
                Err(_) => {
                    return halt(
                        HaltReason::BeliefUpdateFailed,
                        Some(belief.mean()),
                        mandatory,
                    )
                }
            };

            if belief.mean() >= self.config.halt_risk {
                self.account.record_inspection(is_mandatory);
                return halt(HaltReason::CriticalRisk, Some(belief.mean()), mandatory);
            }
        }

        if inspected {
            self.account.record_inspection(is_mandatory);
        }
        MonitorOutcome::Allow {
            risk: belief.mean(),
            rungs_purchased,
            mandatory,
        }
    }
}

fn halt(reason: HaltReason, risk: Option<f64>, mandatory: Vec<MandatoryClass>) -> MonitorOutcome {
    MonitorOutcome::Halt {
        reason,
        risk,
        mandatory,
    }
}

fn monetized_cost(spec: &EstimatorSpec, config: &VoiConfig) -> f64 {
    spec.cost + spec.latency_us * config.latency_price
}

fn voi_err(e: impl std::fmt::Display) -> MonitorError {
    MonitorError::Voi(e.to_string())
}

/// Sanity check that a rung is purchasable in principle for a given belief.
///
/// Exposed so an operator can answer "is this ladder actually able to
/// escalate?" without running traffic through it.
///
/// Validates its inputs through the same choke point the ladder uses. Without
/// that, a `value_of_success` of infinity reports `true` — a diagnostic that
/// answers "yes, escalation is possible" for a configuration the ladder
/// itself would refuse is worse than no diagnostic.
pub fn rung_is_purchasable(
    belief: Belief,
    rung: &LadderRung,
    config: &MonitorConfig,
) -> Result<bool, MonitorError> {
    config.voi.validate().map_err(voi_err)?;
    rung.spec.validate().map_err(voi_err)?;
    let bound = voi_upper_bound(belief, rung.spec.noise_std).map_err(voi_err)?;
    Ok(config.voi.value_of_success * bound > monetized_cost(&rung.spec, &config.voi))
}
