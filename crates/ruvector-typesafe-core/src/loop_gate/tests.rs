//! Unit tests for the promotion gate — pure, no engine/embedder/bank.

use super::*;

/// Tiny xorshift64 RNG (no `rand` dependency); deterministic per seed.
struct XorShift64(u64);
impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self(seed | 1) // never zero
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn bern(&mut self, p: f64) -> bool {
        ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64) < p
    }
}

fn proposal() -> Proposal {
    Proposal {
        id: 1,
        parent: None,
        kind: ProposalKind::BankGrowth,
        description_hash: 0,
    }
}

fn evidence(paired: Vec<(bool, bool)>, base_t: f32, champ_t: f32) -> Evidence {
    Evidence {
        paired,
        baseline_transfer_acc: base_t,
        champion_transfer_acc: champ_t,
        transfer_n: 100,
        model_id: "hash-bow-64@test-double".into(),
        head: Head::NearestPrototype,
        temperature: 1.0,
        created_seq: 0,
        created: None,
    }
}

#[test]
fn true_improvement_rejects_within_a_few_hundred_items() {
    // Champion wins 70% of discordant pairs; feed all-discordant pairs.
    let mut rng = XorShift64::new(0x1234_5678);
    let mut test = PairedSequentialTest::standard();
    let mut n = 0;
    for _ in 0..400 {
        n += 1;
        let champ_wins = rng.bern(0.7);
        // discordant: exactly one correct
        test.update(!champ_wins, champ_wins);
        if test.rejected() {
            break;
        }
    }
    assert!(test.rejected(), "should reject a true 0.7 improvement");
    assert!(n <= 400, "rejected after {n} discordant pairs");
}

#[test]
fn no_difference_controls_type_one_error() {
    // Ville: P(ever reject | null) <= alpha = 0.05. Over 40 seeds, expect
    // at most ~2 rejections; assert <= 2 (deterministic given fixed seeds).
    let mut rejections = 0;
    for seed in 0..40u64 {
        let mut rng = XorShift64::new(0x9E37_79B9 ^ seed.wrapping_mul(2654435761));
        let mut test = PairedSequentialTest::standard();
        for _ in 0..2000 {
            // null: champion wins a discordant pair with prob 0.5
            let champ_wins = rng.bern(0.5);
            test.update(!champ_wins, champ_wins);
            if test.rejected() {
                break;
            }
        }
        if test.rejected() {
            rejections += 1;
        }
    }
    assert!(rejections <= 2, "type-I too high: {rejections}/40 rejected");
}

#[test]
fn concordant_pairs_do_not_move_wealth() {
    let mut test = PairedSequentialTest::standard();
    test.update(true, true);
    test.update(false, false);
    let (cw, bw) = test.discordant_counts();
    assert_eq!((cw, bw), (0, 0));
    assert!((test.statistic().wealth - 1.0).abs() < 1e-12);
}

#[test]
fn rejection_latches_even_if_wealth_drifts_back() {
    let mut test = PairedSequentialTest::new(0.5, 1.9); // threshold = 2.0
                                                        // Two champion wins push wealth to 1.95*1.95 = 3.80 > 2.0 -> latch.
    test.update(false, true);
    test.update(false, true);
    assert!(test.rejected());
    // Baseline wins now; wealth falls, but the latch holds.
    for _ in 0..50 {
        test.update(true, false);
    }
    assert!(test.rejected());
    assert!(test.statistic().wealth < 2.0);
}

#[test]
fn transfer_regression_rejects_even_when_validation_passes() {
    // Strong validation win: champion right, baseline wrong, 200 times.
    let paired = vec![(false, true); 200];
    // Transfer regresses 0.90 -> 0.85, well beyond the 0.01 tolerance.
    let ev = evidence(paired, 0.90, 0.85);
    let mut gate = Gate::new(Budget::new(100, "day-1"));
    let out = gate.evaluate(proposal(), &ev, "day-1");
    assert!(matches!(out.decision, GateDecision::Reject(_)));
    // The reason is the transfer regression, not the sequential test.
    if let GateDecision::Reject(r) = &out.decision {
        assert!(r.contains("transfer"), "reason was: {r}");
    }
}

#[test]
fn clean_win_with_stable_transfer_promotes() {
    let paired = vec![(false, true); 200];
    let ev = evidence(paired, 0.90, 0.905);
    let mut gate = Gate::new(Budget::new(100, "day-1"));
    let out = gate.evaluate(proposal(), &ev, "day-1");
    assert_eq!(out.decision, GateDecision::Promote);
    assert!(out.receipt.statistic.rejected);
}

#[test]
fn budget_exhaustion_pauses_not_lowers_the_bar() {
    let mut gate = Gate::new(Budget::new(1, "day-1"));
    let paired = vec![(false, true); 200];
    let ev = evidence(paired, 0.9, 0.91);
    // First evaluation spends the single unit and promotes.
    let first = gate.evaluate(proposal(), &ev, "day-1");
    assert_eq!(first.decision, GateDecision::Promote);
    // Second on the same day is paused, regardless of how strong it is.
    let second = gate.evaluate(proposal(), &ev, "day-1");
    assert!(matches!(second.decision, GateDecision::Paused(_)));
    // A new day rolls the budget over.
    let third = gate.evaluate(proposal(), &ev, "day-2");
    assert_eq!(third.decision, GateDecision::Promote);
}

#[test]
fn control_arm_alarms_on_champion_drift() {
    let mut arm = ControlArm::new(0.05, 100);
    for _ in 0..200 {
        arm.observe(true, false); // control right, champion wrong
    }
    assert!(arm.alarm());
    assert!(arm.drift() < 0.0);
    // Below the sample floor, no alarm even with drift.
    let mut young = ControlArm::new(0.05, 100);
    young.observe(true, false);
    assert!(!young.alarm());
}

#[test]
fn campaign_mints_exactly_two_test_tokens() {
    let mut campaign = Campaign::new();
    assert!(campaign.test_token().is_some()); // baseline
    assert!(campaign.test_token().is_some()); // champion
    assert!(campaign.test_token().is_none()); // loop cannot score test again
}
