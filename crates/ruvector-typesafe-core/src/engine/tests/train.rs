use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

struct CountingEmbedder {
    inner: HashEmbedder,
    texts: Arc<AtomicUsize>,
}

impl Embedder for CountingEmbedder {
    fn embed(&self, texts: &[&str]) -> crate::Result<Vec<Vec<f32>>> {
        self.texts.fetch_add(texts.len(), Ordering::Relaxed);
        self.inner.embed(texts)
    }

    fn dims(&self) -> usize {
        self.inner.dims()
    }

    fn id(&self) -> &str {
        self.inner.id()
    }
}

#[test]
fn training_embeds_each_new_identity_only_once() {
    let count = Arc::new(AtomicUsize::new(0));
    let mut engine = Engine::new(CountingEmbedder {
        inner: HashEmbedder::new(DIMS),
        texts: count.clone(),
    });
    let one = LabeledExample {
        text: "rain storm".into(),
        label: "weather".into(),
    };
    let two = LabeledExample {
        text: "football goal".into(),
        label: "sports".into(),
    };

    let first = engine.train("q", &[one.clone(), one.clone(), two.clone()]).unwrap();
    assert_eq!(first.accepted, 2);
    assert_eq!(count.load(Ordering::Relaxed), 2);
    let generation = *engine.train_gen.read().unwrap().get("q").unwrap();

    let repeat = engine.train("q", &[one.clone(), two.clone()]).unwrap();
    assert_eq!(repeat.accepted, 0);
    assert_eq!(count.load(Ordering::Relaxed), 2);
    assert_eq!(*engine.train_gen.read().unwrap().get("q").unwrap(), generation);

    let three = LabeledExample {
        text: "fog tomorrow".into(),
        label: "weather".into(),
    };
    let last = engine.train("q", &[one, three]).unwrap();
    assert_eq!(last.accepted, 1);
    assert_eq!(count.load(Ordering::Relaxed), 3);
    assert_eq!(engine.bank_summary().total, 3);
}

#[test]
fn dynamic_question_caches_remain_bounded() {
    let engine = Engine::new(HashEmbedder::new(DIMS));
    for i in 0..(MAX_COMPILED_QUESTIONS + 16) {
        let mut req = two_topic_request("rain clouds storm");
        let question = req.questions.remove("q").unwrap();
        req.questions.insert(format!("q-{i}"), question);
        let response = engine.decide(&req).unwrap();
        assert!(response.answers.contains_key(&format!("q-{i}")));
    }
    assert!(engine.compiled_cache.read().unwrap().len() <= MAX_COMPILED_QUESTIONS);
    assert!(engine.artifact_cache.read().unwrap().len() <= MAX_FITTED_ARTIFACTS);
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
