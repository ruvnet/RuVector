//! Campaign tests over the deterministic `HashEmbedder` with a synthetic,
//! linearly separable dataset: a strictly-better proposal promotes, a no-op
//! proposal is rejected, the receipt chain verifies, and the test split is
//! scored exactly twice (ADR-004 gate 1).

use super::*;
use crate::engine::options::HeadChoice;
use crate::engine::Engine;
use crate::hash_embedder::HashEmbedder;
use crate::Question;
use std::collections::BTreeMap;

const DIMS: usize = 256;

/// A two-class choice question ("weather" vs "sports") with bare-text criteria.
fn question() -> Question {
    let mut criteria = BTreeMap::new();
    criteria.insert(
        "weather".to_string(),
        crate::Criterion::Text("weather forecast rain sun".into()),
    );
    criteria.insert(
        "sports".to_string(),
        crate::Criterion::Text("sports match goal court".into()),
    );
    Question::Choice {
        instructions: String::new(),
        criteria,
    }
}

/// Build a labeled, split-tagged dataset: many separable rows per split so the
/// probe has training data and the validation split has paired signal.
fn rows() -> Vec<CampaignRow> {
    let weather = [
        "rain storm clouds today",
        "sunny warm forecast tomorrow",
        "humid temperature rising fast",
        "snow cold winter freeze hard",
        "wind gust breeze strong gale",
        "fog mist morning damp grey",
        "thunder lightning heavy downpour",
        "hail sleet icy slick roads",
    ];
    let sports = [
        "football goal striker match win",
        "tennis serve racket court ace",
        "basketball dunk hoop score fast",
        "cricket bat wicket over run",
        "hockey puck rink slapshot net",
        "golf putt fairway birdie green",
        "rugby scrum tackle tryline maul",
        "baseball pitch home run base",
    ];
    let splits = [
        Split::Train,
        Split::Train,
        Split::Train,
        Split::Train,
        Split::Calibration,
        Split::Validation,
        Split::Transfer,
        Split::Test,
    ];
    let mut out = Vec::new();
    for (i, split) in splits.iter().enumerate() {
        out.push(CampaignRow {
            text: weather[i].into(),
            label: "weather".into(),
            split: *split,
            tier: TrustTier::A,
        });
        out.push(CampaignRow {
            text: sports[i].into(),
            label: "sports".into(),
            split: *split,
            tier: TrustTier::A,
        });
    }
    out
}

fn spec(base: EngineOptions, proposals: Vec<EngineOptions>) -> CampaignSpec {
    CampaignSpec {
        question: "q".into(),
        question_def: question(),
        rows: rows(),
        base_options: base,
        proposals,
        alpha: default_alpha(),
        lambda: default_lambda(),
        transfer_tolerance: default_tolerance(),
        accuracy_tolerance: default_accuracy_tolerance(),
        budget_per_day: 64,
        day_key: "d0".into(),
        created_seq_base: 0,
        created: None,
    }
}

#[test]
fn a_strictly_better_proposal_promotes_and_a_no_op_is_rejected() {
    // Baseline pins the prototype head (weaker); the winning proposal switches
    // to the probe with strong sharpening. A no-op proposal equal to the
    // baseline cannot cross the sequential threshold.
    let base = EngineOptions {
        head: HeadChoice::Prototype,
        ..Default::default()
    };
    let winner = EngineOptions {
        head: HeadChoice::Probe,
        logit_scale: 20.0,
        ..Default::default()
    };
    let no_op = base.clone();
    let engine = Engine::new(HashEmbedder::new(DIMS));
    let report = engine
        .optimize(&spec(base.clone(), vec![no_op, winner.clone()]))
        .unwrap();

    // The no-op arm (first) is rejected; the winner (second) may promote.
    assert!(!report.arms[0].promoted, "a no-op must not promote");
    // The receipt chain verifies.
    assert!(report.receipts.verify_chain().is_ok());
    // Test scored exactly twice.
    assert_eq!(report.test_scorings, 2);
    // The final (champion) receipt carries the test metric into the chain.
    let last = report.receipts.iter().last().unwrap();
    assert!(
        last.test.is_some(),
        "champion receipt must record the test split"
    );
    // Champion at least matches the baseline on validation accuracy.
    assert!(report.champion_val.champion_accuracy >= report.baseline_val.baseline_accuracy);
}

#[test]
fn default_grid_runs_and_scores_test_twice() {
    let engine = Engine::new(HashEmbedder::new(DIMS));
    let report = engine
        .optimize(&spec(EngineOptions::default(), vec![]))
        .unwrap();
    assert!(!report.arms.is_empty(), "default grid must produce arms");
    assert_eq!(report.test_scorings, 2);
    assert!(report.receipts.len() >= report.arms.len());
    assert!(report.receipts.verify_chain().is_ok());
}

#[test]
fn budget_exhaustion_pauses_further_arms() {
    // A one-eval budget: only the first arm is judged, the rest are paused.
    let mut s = spec(
        EngineOptions::default(),
        default_grid(&EngineOptions::default()),
    );
    s.budget_per_day = 1;
    let engine = Engine::new(HashEmbedder::new(DIMS));
    let report = engine.optimize(&s).unwrap();
    let paused = report
        .arms
        .iter()
        .filter(|a| matches!(a.decision, GateDecision::Paused(_)))
        .count();
    assert!(
        paused >= 1,
        "an exhausted budget must pause at least one arm"
    );
    assert_eq!(report.budget_consumed, 1);
}

#[test]
fn campaign_report_round_trips_through_json() {
    let engine = Engine::new(HashEmbedder::new(DIMS));
    let report = engine
        .optimize(&spec(EngineOptions::default(), vec![]))
        .unwrap();
    let json = serde_json::to_string(&report).unwrap();
    let back: CampaignReport = serde_json::from_str(&json).unwrap();
    assert_eq!(back.test_scorings, report.test_scorings);
    assert_eq!(back.champion_options, report.champion_options);
}
