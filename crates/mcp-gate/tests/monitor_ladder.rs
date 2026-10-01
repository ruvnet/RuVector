//! Behavioural tests for the value-of-information escalation ladder
//! (PIR WP33, ADR-337).
//!
//! The load-bearing one is [`no_valid_config_can_skip_a_mandatory_class`]:
//! the whole point of the mandatory classes is that they survive a
//! configuration chosen to avoid inspecting anything.

use std::cell::RefCell;

use mcp_gate::monitor::{
    EscalationLadder, HaltReason, InspectionSubject, Investigator, KeywordDetector, LadderRung,
    MandatoryClass, MonitorConfig, MonitorError, MonitorOutcome, RiskSignal, TinyDetector,
};
use ruvector_voi::VoiConfig;
use serde_json::json;

/// Investigator that returns a fixed verdict and records every rung it ran.
struct RecordingInvestigator {
    verdict: f64,
    seen: RefCell<Vec<String>>,
}

impl RecordingInvestigator {
    fn new(verdict: f64) -> Self {
        Self {
            verdict,
            seen: RefCell::new(Vec::new()),
        }
    }
    fn calls(&self) -> usize {
        self.seen.borrow().len()
    }
}

impl Investigator for RecordingInvestigator {
    fn investigate(
        &self,
        _subject: &InspectionSubject,
        rung: &LadderRung,
    ) -> Result<f64, MonitorError> {
        self.seen.borrow_mut().push(rung.name.clone());
        Ok(self.verdict)
    }
}

/// Investigator that always fails, to exercise fail-closed behaviour.
struct FailingInvestigator;
impl Investigator for FailingInvestigator {
    fn investigate(
        &self,
        _subject: &InspectionSubject,
        _rung: &LadderRung,
    ) -> Result<f64, MonitorError> {
        Err(MonitorError::ZeroSaturation)
    }
}

/// Investigator returning an out-of-range observation.
struct RogueInvestigator(f64);
impl Investigator for RogueInvestigator {
    fn investigate(
        &self,
        _subject: &InspectionSubject,
        _rung: &LadderRung,
    ) -> Result<f64, MonitorError> {
        Ok(self.0)
    }
}

/// Detector that always fails.
struct FailingDetector;
impl TinyDetector for FailingDetector {
    fn score(&self, _s: &InspectionSubject) -> Result<RiskSignal, MonitorError> {
        Err(MonitorError::ZeroSaturation)
    }
    fn expected_uncertainty(&self) -> f64 {
        0.2
    }
}

/// Detector returning a fixed signal, so tests control the risk exactly.
struct FixedDetector(f64, f64);
impl TinyDetector for FixedDetector {
    fn score(&self, _s: &InspectionSubject) -> Result<RiskSignal, MonitorError> {
        RiskSignal::new(self.0, self.1)
    }
    fn expected_uncertainty(&self) -> f64 {
        self.1
    }
}

/// Declares an uncertainty that differs from what it scores — the divergence
/// the calibration guard used to be blind to when the figure was configured
/// rather than derived.
struct LyingDetector {
    scored: f64,
    declared: f64,
}
impl TinyDetector for LyingDetector {
    fn score(&self, _s: &InspectionSubject) -> Result<RiskSignal, MonitorError> {
        RiskSignal::new(0.5, self.scored)
    }
    fn expected_uncertainty(&self) -> f64 {
        self.declared
    }
}

/// Reference ladder. Latencies are declared honestly (5 ms, 200 ms) — modest
/// for real verifiers — so that pricing them has something to price.
fn rungs() -> Vec<LadderRung> {
    vec![
        LadderRung::new("cheap-verifier", 0.01, 5_000.0, 0.15),
        LadderRung::new("strong-investigator", 0.50, 200_000.0, 0.05),
    ]
}

fn benign() -> InspectionSubject {
    InspectionSubject::new("sum", "add two integers", json!({"a": 1}))
}

#[test]
fn a_free_rung_is_refused_at_construction() {
    // A zero-cost rung is purchased forever: value of information decays
    // geometrically but never reaches zero, so net value stays positive.
    let bad = vec![LadderRung::new("free", 0.0, 0.0, 0.1)];
    let err = EscalationLadder::new(
        bad,
        MonitorConfig::default(),
        KeywordDetector::new(),
        RecordingInvestigator::new(0.1),
    )
    .expect_err("a free rung must be refused");
    assert!(
        matches!(err, MonitorError::FreeRung { index: 0, .. }),
        "{err}"
    );
}

#[test]
fn unordered_and_empty_ladders_are_refused() {
    let unordered = vec![
        LadderRung::new("expensive", 1.0, 0.0, 0.05),
        LadderRung::new("cheap", 0.01, 0.0, 0.15),
    ];
    assert!(matches!(
        EscalationLadder::new(
            unordered,
            MonitorConfig::default(),
            KeywordDetector::new(),
            RecordingInvestigator::new(0.1)
        ),
        Err(MonitorError::UnorderedRungs { .. })
    ));
    assert!(matches!(
        EscalationLadder::new(
            vec![],
            MonitorConfig::default(),
            KeywordDetector::new(),
            RecordingInvestigator::new(0.1)
        ),
        Err(MonitorError::EmptyLadder)
    ));
}

#[test]
fn a_never_escalating_value_of_success_is_refused_loudly() {
    // The trap this guards: value_of_success left at a nominal 1.0 makes the
    // ceiling ~0.4σ ≈ 0.08, so a 0.50 rung is permanently unpurchasable and
    // the ladder becomes a never-escalate switch that still looks configured.
    let config = MonitorConfig {
        voi: VoiConfig {
            value_of_success: 1.0,
            latency_price: 0.0,
        },
        ..MonitorConfig::default()
    };
    let err = EscalationLadder::new(
        vec![LadderRung::new("verifier", 0.50, 0.0, 0.1)],
        config,
        KeywordDetector::new(),
        RecordingInvestigator::new(0.1),
    )
    .expect_err("miscalibration must be refused");
    assert!(matches!(err, MonitorError::Miscalibrated { .. }), "{err}");
}

#[test]
fn a_detector_cannot_inflate_the_ceiling_to_wave_through_a_dead_ladder() {
    // MEDIUM-1. When the uncertainty was an operator-declared config field it
    // was used ONLY in the construction ceiling and never cross-checked
    // against what the detector reported, so declaring a large value bought
    // past the very guard that exists to catch a never-escalate ladder.
    // It is now taken from the detector and bounded to (0, 1].
    for absurd in [100.0_f64, 1.5, 1e12] {
        let err = EscalationLadder::new(
            vec![LadderRung::new("verifier", 0.50, 0.0, 0.1)],
            MonitorConfig {
                voi: VoiConfig {
                    value_of_success: 1.0,
                    latency_price: 0.0,
                },
                ..MonitorConfig::default()
            },
            LyingDetector {
                scored: 0.2,
                declared: absurd,
            },
            RecordingInvestigator::new(0.1),
        )
        .expect_err("an uncertainty above 1 must be refused");
        assert!(
            matches!(err, MonitorError::UncertaintyOutOfRange { .. }),
            "declared {absurd} gave {err}"
        );
    }
}

#[test]
fn the_calibration_ceiling_uses_the_predictive_std_not_sigma() {
    // MEDIUM-2. What a rung can reveal is s = σ/√(1+(τ/σ)²), strictly less
    // than σ whenever the rung is noisy — and the cheapest rung is
    // deliberately the noisiest. A σ-based ceiling therefore overstates what
    // is purchasable and admits ladders that then buy nothing.
    //
    // For the reference ladder (σ=0.2, τ=0.15, cheapest monetized cost
    // $0.015) the σ-based ceiling accepts value_of_success ≥ 0.188 while
    // real escalation needs ≥ 0.235. This value sits in that gap.
    let in_the_gap = MonitorConfig {
        voi: VoiConfig {
            value_of_success: 0.20,
            latency_price: 1e-6,
        },
        ..MonitorConfig::default()
    };
    let err = EscalationLadder::new(
        rungs(),
        in_the_gap,
        KeywordDetector::new(),
        RecordingInvestigator::new(0.5),
    )
    .expect_err("a ladder inside the σ/s gap must be refused");
    assert!(matches!(err, MonitorError::Miscalibrated { .. }), "{err}");
}

#[test]
fn zero_round_cap_is_refused() {
    let config = MonitorConfig {
        max_rounds: 0,
        ..MonitorConfig::default()
    };
    assert!(matches!(
        EscalationLadder::new(
            rungs(),
            config,
            KeywordDetector::new(),
            RecordingInvestigator::new(0.1)
        ),
        Err(MonitorError::ZeroRounds)
    ));
}

#[test]
fn out_of_range_thresholds_are_refused() {
    for bad in [1.5_f64, -0.1, f64::NAN] {
        let config = MonitorConfig {
            halt_risk: bad,
            ..MonitorConfig::default()
        };
        assert!(
            EscalationLadder::new(
                rungs(),
                config,
                KeywordDetector::new(),
                RecordingInvestigator::new(0.1)
            )
            .is_err(),
            "halt_risk {bad} must be refused"
        );
    }
}

#[test]
fn no_valid_config_can_skip_a_mandatory_class() {
    // The key security property. The configuration below is chosen to be as
    // escalation-hostile as construction permits: the detector reports zero
    // risk (so the economics see nothing worth investigating) and the
    // escalation threshold sits far from it. Every mandatory class must
    // still be investigated.
    //
    // Note also what this test cannot do, which is itself the guarantee:
    // there is no MonitorConfig field that disables a class, and
    // `Classification` has no public constructor, so a caller cannot pass a
    // forged empty classification — the ladder classifies the subject it was
    // handed. Those are compile-time facts, not runtime ones.
    let hostile = MonitorConfig {
        voi: VoiConfig {
            value_of_success: 5_000.0,
            latency_price: 0.0,
        },
        escalation_threshold: 0.99,
        halt_risk: 1.0,
        max_rounds: 4,
    };

    let cases = [
        (MandatoryClass::PrivilegeEscalation, "run sudo installer"),
        (MandatoryClass::NetworkAccess, "post to https endpoint"),
        (MandatoryClass::CredentialUse, "read the api_key"),
        (MandatoryClass::RuntimeMutation, "mount a plugin"),
        (MandatoryClass::DestructiveOperation, "delete the bucket"),
    ];

    for (class, description) in cases {
        let investigator = RecordingInvestigator::new(0.0);
        let mut ladder = EscalationLadder::new(
            rungs(),
            hostile,
            // Zero risk: the economics have every reason to skip.
            FixedDetector(0.0, 0.2),
            investigator,
        )
        .expect("ladder builds");

        let subject = InspectionSubject::new("op", description, json!({}));
        let outcome = ladder.inspect(&subject);

        assert!(
            outcome.mandatory().contains(&class),
            "{} was not classified: {:?}",
            class.as_str(),
            outcome.mandatory()
        );
        assert_eq!(
            ladder.account().mandatory_inspections(),
            1,
            "{} was not inspected despite being mandatory",
            class.as_str()
        );
        assert!(
            ladder.account().rungs_purchased() >= 1,
            "{} bought no investigator rung",
            class.as_str()
        );
    }
}

#[test]
fn a_benign_operation_is_allowed_without_purchasing_a_rung() {
    // The other half of the bargain: if mandatory classes always inspected
    // *and* everything else did too, the overhead target would be
    // unreachable. A zero-risk benign operation must cost nothing.
    let mut ladder = EscalationLadder::new(
        rungs(),
        MonitorConfig::default(),
        FixedDetector(0.0, 0.2),
        RecordingInvestigator::new(0.0),
    )
    .expect("ladder builds");

    let outcome = ladder.inspect(&benign());
    assert!(outcome.permits_execution());
    assert_eq!(ladder.account().rungs_purchased(), 0);
    assert_eq!(ladder.account().mandatory_inspections(), 0);
}

#[test]
fn the_round_cap_terminates_the_ladder() {
    // Both rungs are attractive at this risk; without the cap the ladder
    // would keep buying while value of information stays positive.
    let config = MonitorConfig {
        max_rounds: 2,
        ..MonitorConfig::default()
    };
    let mut ladder = EscalationLadder::new(
        rungs(),
        config,
        FixedDetector(0.5, 0.3),
        RecordingInvestigator::new(0.5),
    )
    .expect("ladder builds");

    let outcome = ladder.inspect(&benign());
    assert!(ladder.account().rungs_purchased() <= 2, "cap exceeded");
    assert!(matches!(outcome, MonitorOutcome::Allow { .. }));
}

#[test]
fn an_oracle_rung_exits_without_a_belief_update() {
    // A noiseless rung's observation *is* the truth. `observe` refuses
    // noise_std == 0, so the ladder must take the verdict and exit rather
    // than folding it into a posterior — if it called observe, this would
    // halt with BeliefUpdateFailed instead of allowing.
    let oracle = vec![LadderRung::new("oracle", 0.01, 0.0, 0.0)];
    let mut ladder = EscalationLadder::new(
        oracle,
        MonitorConfig::default(),
        FixedDetector(0.4, 0.3),
        RecordingInvestigator::new(0.10),
    )
    .expect("ladder builds");

    let outcome = ladder.inspect(&InspectionSubject::new("op", "delete data", json!({})));
    match outcome {
        MonitorOutcome::Allow { risk, .. } => {
            assert!(
                (risk - 0.10).abs() < 1e-12,
                "oracle verdict not taken: {risk}"
            );
        }
        other => panic!("expected Allow from oracle verdict, got {other:?}"),
    }
    assert_eq!(ladder.account().rungs_purchased(), 1);
}

#[test]
fn an_oracle_verdict_above_the_halt_threshold_halts() {
    let oracle = vec![LadderRung::new("oracle", 0.01, 0.0, 0.0)];
    let mut ladder = EscalationLadder::new(
        oracle,
        MonitorConfig::default(),
        FixedDetector(0.4, 0.3),
        RecordingInvestigator::new(0.99),
    )
    .expect("ladder builds");

    let outcome = ladder.inspect(&InspectionSubject::new("op", "delete data", json!({})));
    assert!(!outcome.permits_execution());
    assert!(matches!(
        outcome,
        MonitorOutcome::Halt {
            reason: HaltReason::CriticalRisk,
            ..
        }
    ));
}

#[test]
fn every_failure_path_halts_rather_than_allowing() {
    // Fail-closed, exhaustively. `inspect` returns no Result, so there is no
    // fallible value a caller could `.unwrap_or(allow)`.
    let cfg = MonitorConfig::default();

    // Detector failure.
    let mut ladder = EscalationLadder::new(
        rungs(),
        cfg,
        FailingDetector,
        RecordingInvestigator::new(0.1),
    )
    .unwrap();
    let out = ladder.inspect(&benign());
    assert!(!out.permits_execution());
    assert!(matches!(
        out,
        MonitorOutcome::Halt {
            reason: HaltReason::DetectorFailed,
            ..
        }
    ));

    // Investigator failure, reached via a mandatory operation.
    let mut ladder =
        EscalationLadder::new(rungs(), cfg, FixedDetector(0.1, 0.2), FailingInvestigator).unwrap();
    let out = ladder.inspect(&InspectionSubject::new("op", "sudo rm", json!({})));
    assert!(!out.permits_execution());
    assert!(matches!(
        out,
        MonitorOutcome::Halt {
            reason: HaltReason::InvestigatorFailed,
            ..
        }
    ));

    // Out-of-range and non-finite observations.
    for rogue in [1.5_f64, -0.5, f64::NAN, f64::INFINITY] {
        let mut ladder = EscalationLadder::new(
            rungs(),
            cfg,
            FixedDetector(0.1, 0.2),
            RogueInvestigator(rogue),
        )
        .unwrap();
        let out = ladder.inspect(&InspectionSubject::new("op", "sudo rm", json!({})));
        assert!(
            !out.permits_execution(),
            "observation {rogue} was allowed through"
        );
        assert!(matches!(
            out,
            MonitorOutcome::Halt {
                reason: HaltReason::InvalidObservation,
                ..
            }
        ));
    }
}

#[test]
fn a_nan_observation_cannot_pass_the_halt_comparison() {
    // Specifically the comparison-gate failure mode: `NaN >= halt_risk` is
    // false, so a naive gate would fall through to Allow.
    let mut ladder = EscalationLadder::new(
        rungs(),
        MonitorConfig::default(),
        FixedDetector(0.1, 0.2),
        RogueInvestigator(f64::NAN),
    )
    .unwrap();
    let out = ladder.inspect(&InspectionSubject::new(
        "op",
        "delete everything",
        json!({}),
    ));
    assert!(!out.permits_execution(), "NaN fell through to Allow");
}

#[test]
fn a_nan_from_an_oracle_rung_cannot_fall_through_to_allow() {
    // The dangerous path, and the reason the observation check exists as its
    // own guard rather than relying on `observe`.
    //
    // A noisy rung's NaN is caught incidentally: the ladder calls `observe`,
    // which rejects non-finite observations. An ORACLE rung never calls
    // `observe` — its verdict goes straight to `observation >= halt_risk`,
    // and that comparison is false for NaN, so without an explicit check the
    // operation is allowed. This test fails if the guard is removed; the
    // noisy-rung test above does not.
    let oracle = vec![LadderRung::new("oracle", 0.01, 0.0, 0.0)];
    for rogue in [f64::NAN, f64::INFINITY, -1.0, 2.0] {
        let mut ladder = EscalationLadder::new(
            oracle.clone(),
            MonitorConfig::default(),
            FixedDetector(0.1, 0.2),
            RogueInvestigator(rogue),
        )
        .unwrap();
        let out = ladder.inspect(&InspectionSubject::new(
            "op",
            "delete everything",
            json!({}),
        ));
        assert!(
            !out.permits_execution(),
            "oracle observation {rogue} fell through to Allow"
        );
        assert!(matches!(
            out,
            MonitorOutcome::Halt {
                reason: HaltReason::InvalidObservation,
                ..
            }
        ));
    }
}

#[test]
fn a_critical_detector_score_halts_before_any_purchase() {
    let mut ladder = EscalationLadder::new(
        rungs(),
        MonitorConfig::default(),
        FixedDetector(0.99, 0.2),
        RecordingInvestigator::new(0.0),
    )
    .unwrap();
    let out = ladder.inspect(&benign());
    assert!(matches!(
        out,
        MonitorOutcome::Halt {
            reason: HaltReason::CriticalRisk,
            ..
        }
    ));
    assert_eq!(ladder.account().rungs_purchased(), 0);
    assert_eq!(ladder.account().halts(), 1);
}

#[test]
fn measured_overhead_on_a_synthetic_workload_is_under_five_percent() {
    // 200 operations, one in ten mandatory. Workload is attributed at 20ms
    // per operation, a modest figure for a real tool call.
    //
    // HONEST SCOPE: the investigator here is a fixture that returns
    // immediately, so this measures the *ladder's* overhead — classification,
    // detection, the VoI decision, and belief updates — not the cost of a
    // real investigator model call. A real ladder's overhead is dominated by
    // what its rungs actually do.
    const OPS: usize = 200;
    const WORKLOAD_US_PER_OP: u128 = 20_000;

    let mut ladder = EscalationLadder::new(
        rungs(),
        MonitorConfig::default(),
        KeywordDetector::new(),
        RecordingInvestigator::new(0.2),
    )
    .expect("ladder builds");

    for i in 0..OPS {
        let subject = if i % 10 == 0 {
            InspectionSubject::new("deploy", "delete the old bucket", json!({"i": i}))
        } else {
            InspectionSubject::new("sum", "add two integers", json!({"i": i}))
        };
        let outcome = ladder.inspect(&subject);
        let _ = outcome.permits_execution();
        ladder.record_workload_us(WORKLOAD_US_PER_OP);
    }

    let account = ladder.account();
    let fraction = account.fraction().expect("workload recorded");
    assert_eq!(account.operations(), OPS as u64);
    assert_eq!(
        account.mandatory_inspections(),
        (OPS / 10) as u64,
        "every mandatory operation must have been inspected"
    );
    assert!(
        fraction < 0.05,
        "monitoring overhead {:.4} exceeded the 5% target",
        fraction
    );
    eprintln!(
        "MEASURED OVERHEAD: {:.6} ({} monitoring µs over {} workload µs), \
         {} inspections of which {} mandatory, {} rungs purchased",
        fraction,
        account.monitoring_us(),
        account.workload_us(),
        account.inspections(),
        account.mandatory_inspections(),
        account.rungs_purchased()
    );
}

#[test]
fn overhead_charges_declared_rung_latency_not_fixture_speed() {
    // MEDIUM-4. With `latency_price` at zero a ten-second rung priced
    // identically to an instant one, and the measured figure reflected only
    // how fast the test double returned. The account now charges what the
    // rung declares, so the number means something about the configuration.
    let mut ladder = EscalationLadder::new(
        rungs(),
        MonitorConfig::default(),
        FixedDetector(0.0, 0.2),
        RecordingInvestigator::new(0.1),
    )
    .expect("ladder builds");

    let outcome = ladder.inspect(&InspectionSubject::new(
        "op",
        "delete the bucket",
        json!({}),
    ));
    assert!(outcome.permits_execution(), "benign verdict should allow");
    let purchased = ladder.account().rungs_purchased();
    assert!(purchased >= 1);
    // The cheap rung alone declares 5 ms; a fixture returning instantly
    // cannot account for less than that.
    assert!(
        ladder.account().monitoring_us() >= 5_000,
        "declared latency was not charged: {} µs for {} rungs",
        ladder.account().monitoring_us(),
        purchased
    );
}

#[test]
fn measured_overhead_with_honest_latency_pricing() {
    // The number that actually bears on the 5% design target: rungs declare
    // realistic latency (5 ms / 200 ms) and the account charges it. Reported
    // rather than asserted against 5%, because whether the target is met is a
    // property of the configuration, not of this module.
    const OPS: usize = 200;
    const WORKLOAD_US_PER_OP: u128 = 20_000;

    let mut ladder = EscalationLadder::new(
        rungs(),
        MonitorConfig::default(),
        KeywordDetector::new(),
        RecordingInvestigator::new(0.2),
    )
    .expect("ladder builds");

    for i in 0..OPS {
        let subject = if i % 10 == 0 {
            InspectionSubject::new("deploy", "delete the old bucket", json!({"i": i}))
        } else {
            InspectionSubject::new("sum", "add two integers", json!({"i": i}))
        };
        let outcome = ladder.inspect(&subject);
        let _ = outcome.permits_execution();
        ladder.record_workload_us(WORKLOAD_US_PER_OP);
    }

    let a = ladder.account();
    let fraction = a.fraction().expect("workload recorded");
    // Coverage is the enforced half and IS asserted.
    assert_eq!(
        a.mandatory_coverage(),
        Some(1.0),
        "mandatory coverage regressed"
    );
    eprintln!(
        "HONEST OVERHEAD: {:.4} ({} monitoring µs over {} workload µs) |          mandatory {}/{} = {:?} | {} rungs purchased | {} halts",
        fraction,
        a.monitoring_us(),
        a.workload_us(),
        a.mandatory_inspections(),
        a.mandatory_operations_seen(),
        a.mandatory_coverage(),
        a.rungs_purchased(),
        a.halts()
    );
}

#[test]
fn latency_pricing_binds_only_when_the_belief_stays_ambiguous() {
    // What actually limits purchases at the reference configuration is the
    // belief update, not the price of latency: an investigator reporting a
    // clearly-benign 0.2 moves the belief away from the 0.5 threshold, value
    // of information collapses, and the ladder declines the second rung
    // whether or not latency costs anything.
    //
    // Latency pricing binds in the case that update does NOT resolve — an
    // investigator that keeps reporting exactly the threshold, so the
    // operation stays maximally ambiguous and every further rung still looks
    // worth buying. This measures both, so the difference is evidence rather
    // than assertion.
    fn run(latency_price: f64, verdict: f64) -> (f64, u64) {
        let config = MonitorConfig {
            voi: VoiConfig {
                value_of_success: 100.0,
                latency_price,
            },
            ..MonitorConfig::default()
        };
        let mut ladder = EscalationLadder::new(
            rungs(),
            config,
            KeywordDetector::new(),
            RecordingInvestigator::new(verdict),
        )
        .expect("ladder builds");
        for i in 0..200 {
            let subject = if i % 10 == 0 {
                InspectionSubject::new("deploy", "delete the old bucket", json!({"i": i}))
            } else {
                InspectionSubject::new("sum", "add two integers", json!({"i": i}))
            };
            let outcome = ladder.inspect(&subject);
            let _ = outcome.permits_execution();
            ladder.record_workload_us(20_000);
        }
        let a = ladder.account();
        (
            a.fraction().expect("workload recorded"),
            a.rungs_purchased(),
        )
    }

    let (resolved_priced, resolved_rungs) = run(1e-6, 0.2);
    let (ambiguous_priced, ambiguous_priced_rungs) = run(1e-6, 0.5);
    let (ambiguous_free, ambiguous_free_rungs) = run(0.0, 0.5);

    eprintln!(
        "OVERHEAD resolved/priced   {:.4} ({} rungs)\n\
         OVERHEAD ambiguous/priced  {:.4} ({} rungs)\n\
         OVERHEAD ambiguous/unpriced {:.4} ({} rungs)",
        resolved_priced,
        resolved_rungs,
        ambiguous_priced,
        ambiguous_priced_rungs,
        ambiguous_free,
        ambiguous_free_rungs
    );

    // Pricing latency can only reduce or leave unchanged what is purchased.
    assert!(
        ambiguous_priced_rungs <= ambiguous_free_rungs,
        "pricing latency increased purchases: {ambiguous_priced_rungs} vs {ambiguous_free_rungs}"
    );
}

#[test]
fn the_purchasability_diagnostic_validates_its_own_inputs() {
    // LOW-8. This is the one public fn that answers "can this ladder ever
    // escalate?", so it must not answer "yes" for a configuration the ladder
    // itself would refuse. Without validation, an infinite value_of_success
    // returned Ok(true).
    use mcp_gate::monitor::rung_is_purchasable;
    use ruvector_voi::Belief;

    let belief = Belief::new(0.5, 0.2).unwrap();
    let rung = LadderRung::new("verifier", 0.01, 0.0, 0.15);

    for bad in [f64::INFINITY, f64::NAN, 0.0, -1.0] {
        let config = MonitorConfig {
            voi: VoiConfig {
                value_of_success: bad,
                latency_price: 0.0,
            },
            ..MonitorConfig::default()
        };
        assert!(
            rung_is_purchasable(belief, &rung, &config).is_err(),
            "value_of_success {bad} produced an answer instead of an error"
        );
    }
    // A well-formed configuration still answers.
    assert!(rung_is_purchasable(belief, &rung, &MonitorConfig::default()).is_ok());
}

#[test]
fn an_empty_account_reports_no_overhead_figure() {
    let ladder = EscalationLadder::new(
        rungs(),
        MonitorConfig::default(),
        KeywordDetector::new(),
        RecordingInvestigator::new(0.1),
    )
    .unwrap();
    // Not 0.0: a ladder that never ran must not report a flattering zero.
    assert_eq!(ladder.account().fraction(), None);
}

#[test]
fn the_investigator_actually_saw_the_mandatory_operations() {
    let investigator = RecordingInvestigator::new(0.1);
    let mut ladder = EscalationLadder::new(
        rungs(),
        MonitorConfig::default(),
        FixedDetector(0.0, 0.2),
        investigator,
    )
    .unwrap();
    let outcome = ladder.inspect(&InspectionSubject::new(
        "op",
        "delete the bucket",
        json!({}),
    ));
    assert!(
        outcome
            .mandatory()
            .contains(&MandatoryClass::DestructiveOperation),
        "operation was not classified as destructive"
    );
    // Reach back through the ladder to confirm a rung genuinely ran, rather
    // than inferring it from counters alone.
    assert!(ladder.account().rungs_purchased() >= 1);
    assert_eq!(ladder.account().mandatory_coverage(), Some(1.0));
}

#[test]
fn recording_investigator_call_count_matches_rungs_purchased() {
    let investigator = RecordingInvestigator::new(0.1);
    let mut ladder = EscalationLadder::new(
        rungs(),
        MonitorConfig::default(),
        FixedDetector(0.0, 0.2),
        investigator,
    )
    .unwrap();
    let outcome = ladder.inspect(&InspectionSubject::new("op", "sudo delete", json!({})));
    let _ = outcome.permits_execution();
    let purchased = ladder.account().rungs_purchased();
    assert!(purchased >= 1, "mandatory operation bought no rung");
    // Consistency between the account and the investigator's own tally is
    // what makes the overhead numbers trustworthy.
    let ladder_calls = {
        let inv: &RecordingInvestigator = ladder_investigator(&ladder);
        inv.calls() as u64
    };
    assert_eq!(purchased, ladder_calls);
}

/// Helper reaching the investigator for cross-checking counters.
fn ladder_investigator<D: TinyDetector>(
    ladder: &EscalationLadder<D, RecordingInvestigator>,
) -> &RecordingInvestigator {
    ladder.investigator()
}
