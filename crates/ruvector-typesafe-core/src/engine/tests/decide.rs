use super::*;

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
fn a_limits_violation_surfaces_as_a_limit_error() {
    let engine = Engine::new(HashEmbedder::new(DIMS));
    let big = "x ".repeat(crate::limits::MAX_STATE_BYTES);
    let req = topic_request(&big);
    assert!(matches!(engine.decide(&req), Err(TypesafeError::Limit(_))));
}
