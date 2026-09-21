//! Integration tests for the decision engine (ADR-003), over the deterministic
//! `HashEmbedder`; one test renames it to exercise the calibration path (gated
//! off for `@test-double` ids).

use super::*;
use crate::embedder::Embedder;
use crate::hash_embedder::HashEmbedder;
use crate::{Answer, Criterion, DecisionRequest, Head, Question, TypesafeError};
use std::collections::BTreeMap;

const DIMS: usize = 256;

fn structured(what: &str, examples: &[&str], not_for: Option<&str>) -> Criterion {
    Criterion::Structured {
        what: what.into(),
        not_for: not_for.map(str::to_string),
        examples: examples.iter().map(|s| s.to_string()).collect(),
    }
}

fn choice_request(state: &str, criteria: Vec<(&str, Criterion)>) -> DecisionRequest {
    let mut map = BTreeMap::new();
    for (k, c) in criteria {
        map.insert(k.to_string(), c);
    }
    let mut questions = BTreeMap::new();
    questions.insert(
        "q".to_string(),
        Question::Choice {
            instructions: String::new(),
            criteria: map,
        },
    );
    DecisionRequest {
        state: state.into(),
        questions,
    }
}

fn as_choice(a: &Answer) -> (&str, &BTreeMap<String, f32>, &crate::AnswerMeta) {
    match a {
        Answer::Choice {
            choice,
            probabilities,
            meta,
        } => (choice, probabilities, meta),
        other => panic!("expected choice, got {other:?}"),
    }
}

fn topic_request(state: &str) -> DecisionRequest {
    choice_request(
        state,
        vec![
            (
                "weather",
                structured(
                    "weather forecast",
                    &["sunny rain clouds storm", "temperature tomorrow humid"],
                    None,
                ),
            ),
            (
                "sports",
                structured(
                    "sports match scores",
                    &["football goal player match", "tennis racket serve court"],
                    None,
                ),
            ),
            (
                "cooking",
                structured(
                    "cooking recipe kitchen",
                    &["bake oven flour dough", "simmer sauce garlic onion"],
                    None,
                ),
            ),
        ],
    )
}

#[test]
fn engine_accepts_a_boxed_trait_object_embedder() {
    let boxed: Box<dyn Embedder> = Box::new(HashEmbedder::new(DIMS));
    let engine = Engine::new(boxed);
    let resp = engine
        .decide(&topic_request("rain clouds storm forecast"))
        .unwrap();
    let (choice, _, _) = as_choice(&resp.answers["q"]);
    assert_eq!(choice, "weather");
}

#[test]
fn three_option_choice_picks_the_matching_option() {
    let engine = Engine::new(HashEmbedder::new(DIMS));
    let req = topic_request("will it rain tomorrow with clouds and storm");
    let resp = engine.decide(&req).unwrap();
    let (choice, probs, meta) = as_choice(&resp.answers["q"]);
    assert_eq!(choice, "weather");
    let sum: f32 = probs.values().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "probabilities sum to 1, got {sum}"
    );
    assert!((0.0..=1.0).contains(&meta.abstain));
    assert_eq!(meta.head, Head::NearestPrototype);
    assert_eq!(resp.usage.embed_calls, 1);
}

#[test]
fn out_of_scope_state_lowers_confidence_and_raises_abstain() {
    let engine = Engine::new(HashEmbedder::new(DIMS));
    let in_scope = engine
        .decide(&topic_request(
            "football goal player and tennis serve match",
        ))
        .unwrap();
    let out_scope = engine
        .decide(&topic_request(
            "quantum helicopter velvet umbrella xylophone",
        ))
        .unwrap();
    let (_, _, in_meta) = as_choice(&in_scope.answers["q"]);
    let (_, _, out_meta) = as_choice(&out_scope.answers["q"]);
    assert!(
        out_meta.confidence < in_meta.confidence,
        "out={} in={}",
        out_meta.confidence,
        in_meta.confidence
    );
    assert!(
        out_meta.abstain > in_meta.abstain,
        "out={} in={}",
        out_meta.abstain,
        in_meta.abstain
    );
}

#[test]
fn not_for_flips_a_near_tie() {
    // alpha shares three tokens with the state, beta two, so alpha wins; a
    // `not_for` on alpha matching the state subtracts enough to hand it to beta.
    let state = "apple banana cherry";
    let engine = Engine::new(HashEmbedder::new(DIMS));

    let base = choice_request(
        state,
        vec![
            (
                "alpha",
                structured("alpha topic", &["apple banana cherry"], None),
            ),
            (
                "beta",
                structured("beta topic", &["apple banana melon"], None),
            ),
        ],
    );
    let base_choice = {
        let r = engine.decide(&base).unwrap();
        let (c, _, _) = as_choice(&r.answers["q"]);
        c.to_string()
    };

    let flipped = choice_request(
        state,
        vec![
            (
                "alpha",
                structured("alpha topic", &["apple banana cherry"], Some(state)),
            ),
            (
                "beta",
                structured("beta topic", &["apple banana melon"], None),
            ),
        ],
    );
    let flip_choice = {
        let r = engine.decide(&flipped).unwrap();
        let (c, _, _) = as_choice(&r.answers["q"]);
        c.to_string()
    };

    assert_eq!(base_choice, "alpha", "alpha should win without not_for");
    assert_eq!(
        flip_choice, "beta",
        "not_for on alpha should hand it to beta"
    );
}

fn training_examples() -> Vec<(&'static str, &'static [&'static str])> {
    vec![
        (
            "weather",
            &[
                "rain clouds storm today",
                "sunny warm forecast tomorrow",
                "humid temperature rising",
                "snow cold winter freeze",
                "wind gust breeze strong",
                "fog mist morning damp",
                "thunder lightning downpour",
                "hail sleet icy roads",
                "heatwave scorching dry summer",
                "drizzle overcast grey sky",
            ],
        ),
        (
            "sports",
            &[
                "football goal striker match",
                "tennis serve racket court",
                "basketball dunk hoop score",
                "cricket bat wicket over",
                "hockey puck rink slapshot",
                "golf putt fairway birdie",
                "rugby scrum tackle tryline",
                "baseball pitch home run",
                "boxing punch jab knockout",
                "swimming lap freestyle relay",
            ],
        ),
    ]
}

fn build_probe_training() -> Vec<LabeledExample> {
    let mut ex = Vec::new();
    for (label, samples) in training_examples() {
        for s in samples {
            ex.push(LabeledExample {
                text: (*s).into(),
                label: label.into(),
            });
        }
    }
    ex
}

fn two_topic_request(state: &str) -> DecisionRequest {
    choice_request(
        state,
        vec![
            (
                "weather",
                structured("weather forecast", &["rain clouds sunny storm"], None),
            ),
            (
                "sports",
                structured("sports match", &["football tennis goal court"], None),
            ),
        ],
    )
}

#[test]
fn linear_probe_beats_prototype_after_training() {
    let held_out: Vec<(&str, &str)> = vec![
        ("storm rain freeze icy", "weather"),
        ("scorching heatwave dry", "weather"),
        ("overcast drizzle fog", "weather"),
        ("striker goal match tackle", "sports"),
        ("racket court serve birdie", "sports"),
        ("dunk hoop wicket over", "sports"),
    ];

    let proto_engine = Engine::new(HashEmbedder::new(DIMS));
    let mut probe_engine = Engine::new(HashEmbedder::new(DIMS));
    let report = probe_engine.train("q", &build_probe_training()).unwrap();
    assert_eq!(report.head, Head::LinearProbe);
    assert_eq!(report.accepted, 20);

    let mut proto_ok = 0;
    let mut probe_ok = 0;
    let mut probe_head_seen = false;
    for (state, want) in &held_out {
        let req = two_topic_request(state);
        let (pc, _, _) = {
            let r = proto_engine.decide(&req).unwrap();
            let (c, _, _) = as_choice(&r.answers["q"]);
            (c.to_string(), (), ())
        };
        if pc == *want {
            proto_ok += 1;
        }
        let r = probe_engine.decide(&req).unwrap();
        let (c, _, meta) = as_choice(&r.answers["q"]);
        if meta.head == Head::LinearProbe {
            probe_head_seen = true;
        }
        if c == *want {
            probe_ok += 1;
        }
    }
    assert!(probe_head_seen, "trained question must use the probe head");
    assert!(
        probe_ok >= proto_ok,
        "probe {probe_ok} should not trail prototype {proto_ok}"
    );
}

#[test]
fn score_returns_the_expected_bucket() {
    // `score` is the rounded expected index; a middle-bucket state centres it.
    let mut questions = BTreeMap::new();
    questions.insert(
        "mood".to_string(),
        Question::Score {
            instructions: String::new(),
            legend: vec!["calm".into(), "irritated".into(), "furious".into()],
        },
    );
    let req = DecisionRequest {
        state: "irritated".into(),
        questions,
    };
    let engine = Engine::new(HashEmbedder::new(DIMS));
    let resp = engine.decide(&req).unwrap();
    match &resp.answers["mood"] {
        Answer::Score {
            score,
            legend,
            probabilities,
            ..
        } => {
            assert_eq!(*score, 1);
            assert_eq!(legend, "irritated");
            assert_eq!(probabilities.len(), 3);
            let sum: f32 = probabilities.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
        other => panic!("expected score, got {other:?}"),
    }
}

#[test]
fn noul_untrained_is_similarity_uncalibrated() {
    let mut questions = BTreeMap::new();
    questions.insert(
        "urgent".to_string(),
        Question::Noul {
            instructions: "the sender needs a response soon urgent".into(),
        },
    );
    let req = DecisionRequest {
        state: "please respond soon this is urgent".into(),
        questions,
    };
    let engine = Engine::new(HashEmbedder::new(DIMS));
    let resp = engine.decide(&req).unwrap();
    match &resp.answers["urgent"] {
        Answer::Noul { noul, meta } => {
            assert!((0.0..=1.0).contains(noul));
            assert!(!meta.calibrated);
            assert_eq!(meta.head, Head::SimilarityUncalibrated);
        }
        other => panic!("expected noul, got {other:?}"),
    }
}

#[test]
fn noul_trained_uses_the_logistic_head() {
    let mut engine = Engine::new(HashEmbedder::new(DIMS));
    let ex: Vec<LabeledExample> = [
        ("respond immediately please urgent", "yes"),
        ("need this answered right away", "yes"),
        ("asap deadline today critical", "yes"),
        ("emergency time sensitive now", "yes"),
        ("no rush whenever convenient", "no"),
        ("just for your records later", "no"),
        ("fyi background reading", "no"),
        ("someday maybe eventually", "no"),
    ]
    .iter()
    .map(|(t, l)| LabeledExample {
        text: (*t).into(),
        label: (*l).into(),
    })
    .collect();
    engine.train("urgent", &ex).unwrap();

    let mut questions = BTreeMap::new();
    questions.insert(
        "urgent".to_string(),
        Question::Noul {
            instructions: "the sender needs a response soon".into(),
        },
    );
    let req = DecisionRequest {
        state: "respond immediately please this is urgent asap".into(),
        questions,
    };
    let resp = engine.decide(&req).unwrap();
    match &resp.answers["urgent"] {
        Answer::Noul { noul, meta } => {
            assert_eq!(meta.head, Head::Logistic);
            assert!(
                *noul > 0.5,
                "urgent-like state should score high, got {noul}"
            );
        }
        other => panic!("expected noul, got {other:?}"),
    }
}

#[test]
fn two_engines_are_bit_for_bit_identical() {
    let build = || {
        let mut e = Engine::new(HashEmbedder::new(DIMS));
        e.train("q", &build_probe_training()).unwrap();
        e
    };
    let e1 = build();
    let e2 = build();
    let req = two_topic_request("storm rain and football goal");
    let r1 = serde_json::to_string(&e1.decide(&req).unwrap()).unwrap();
    let r2 = serde_json::to_string(&e2.decide(&req).unwrap()).unwrap();
    assert_eq!(r1, r2);
}

#[test]
fn a_limits_violation_surfaces_as_a_limit_error() {
    let engine = Engine::new(HashEmbedder::new(DIMS));
    let big = "x ".repeat(crate::limits::MAX_STATE_BYTES);
    let req = topic_request(&big);
    assert!(matches!(engine.decide(&req), Err(TypesafeError::Limit(_))));
}

/// Delegates to `HashEmbedder` but presents a non-`@test-double` id so the
/// calibration path is allowed to run.
struct ProdLike {
    inner: HashEmbedder,
    id: String,
}

impl ProdLike {
    fn new(dims: usize) -> Self {
        Self {
            inner: HashEmbedder::new(dims),
            id: format!("hash-bow-{dims}@prod"),
        }
    }
}

impl Embedder for ProdLike {
    fn embed(&self, texts: &[&str]) -> crate::Result<Vec<Vec<f32>>> {
        self.inner.embed(texts)
    }
    fn dims(&self) -> usize {
        self.inner.dims()
    }
    fn id(&self) -> &str {
        &self.id
    }
}

fn many_examples() -> Vec<LabeledExample> {
    // 120 examples so the every-5th calibration slice clears the 20-example
    // floor. Texts must be UNIQUE: the bank is append-only and dedupes by
    // content, so a repeated `(question,label,text)` never adds a second row.
    // The trailing `number {i}` keeps every example distinct.
    let weather = ["rain", "storm", "sunny", "cloud", "snow", "wind"];
    let sports = ["goal", "match", "court", "score", "tackle", "serve"];
    let mut ex = Vec::new();
    for i in 0..60 {
        let w = weather[i % weather.len()];
        let s = sports[i % sports.len()];
        ex.push(LabeledExample {
            text: format!("{w} weather forecast today number {i}"),
            label: "weather".into(),
        });
        ex.push(LabeledExample {
            text: format!("{s} sports match play number {i}"),
            label: "sports".into(),
        });
    }
    ex
}

#[test]
fn calibration_runs_for_a_non_test_double_embedder() {
    let mut prod = Engine::new(ProdLike::new(DIMS));
    prod.train("q", &many_examples()).unwrap();
    let mut baseline = Engine::new(HashEmbedder::new(DIMS));
    baseline.train("q", &many_examples()).unwrap();

    let req = two_topic_request("storm rain clouds sunny forecast");
    let prod_resp = prod.decide(&req).unwrap();
    let base_resp = baseline.decide(&req).unwrap();
    let (prod_choice, _, prod_meta) = as_choice(&prod_resp.answers["q"]);
    let (base_choice, _, base_meta) = as_choice(&base_resp.answers["q"]);

    assert!(prod_meta.calibrated, "prod-like embedder should calibrate");
    assert!(
        (prod_meta.temperature - 1.0).abs() > 1e-6,
        "a fitted temperature should differ from 1.0, got {}",
        prod_meta.temperature
    );
    assert!(!base_meta.calibrated, "test double must stay uncalibrated");
    // Temperature never changes the argmax.
    assert_eq!(prod_choice, base_choice);
}

#[test]
fn bank_backed_train_keeps_splits_frozen_across_two_calls() {
    // Training the same examples in one call vs two calls (same order) must
    // yield the identical decision: the bank appends in insertion order and the
    // Train/Calibration carve is positional over that order, so a call boundary
    // does not move any example between splits.
    let all = build_probe_training();
    let (first, second) = all.split_at(all.len() / 2);

    let mut one = Engine::new(HashEmbedder::new(DIMS));
    one.train("q", &all).unwrap();

    let mut two = Engine::new(HashEmbedder::new(DIMS));
    two.train("q", first).unwrap();
    let report2 = two.train("q", second).unwrap();
    // The second call's provisional head matches the whole-bank state.
    assert_eq!(report2.head, Head::LinearProbe);

    let req = two_topic_request("storm rain and football goal");
    let r_one = serde_json::to_string(&one.decide(&req).unwrap()).unwrap();
    let r_two = serde_json::to_string(&two.decide(&req).unwrap()).unwrap();
    assert_eq!(
        r_one, r_two,
        "split assignment must be call-boundary independent"
    );

    // Both banks hold every (deduplicated) example.
    assert_eq!(one.bank_summary().total, all.len());
    assert_eq!(two.bank_summary().total, all.len());
}

#[test]
fn export_import_bank_round_trips_the_decision() {
    let mut src = Engine::new(HashEmbedder::new(DIMS));
    src.train("q", &build_probe_training()).unwrap();
    let json = src.export_bank().unwrap();

    let mut dst = Engine::new(HashEmbedder::new(DIMS));
    dst.import_bank(&json).unwrap();

    let req = two_topic_request("storm rain and football goal");
    let a = serde_json::to_string(&src.decide(&req).unwrap()).unwrap();
    let b = serde_json::to_string(&dst.decide(&req).unwrap()).unwrap();
    assert_eq!(a, b, "an imported bank reproduces the exporter's decision");
    assert_eq!(dst.bank_summary().total, src.bank_summary().total);
}
