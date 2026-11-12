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

fn instructed_choice(instructions: &str) -> DecisionRequest {
    let mut req = choice_request(
        "the customer wants to sell their shares",
        vec![
            (
                "shares",
                structured("company shares equity stock", &[], None),
            ),
            (
                "bonds",
                structured("government bonds fixed income", &[], None),
            ),
        ],
    );
    if let Some(Question::Choice {
        instructions: i, ..
    }) = req.questions.get_mut("q")
    {
        *i = instructions.to_string();
    }
    req
}

fn choice_probs(engine: &Engine<HashEmbedder>, req: &DecisionRequest) -> BTreeMap<String, f32> {
    let r = engine.decide(req).unwrap();
    as_choice(&r.answers["q"]).1.clone()
}

#[test]
fn choice_instructions_are_ignored_by_default() {
    // Original behaviour: `choice` embeds only the criteria.
    let engine = Engine::new(HashEmbedder::new(DIMS));
    let own = choice_probs(&engine, &instructed_choice("which asset do they own"));
    let avoid = choice_probs(&engine, &instructed_choice("which asset do they avoid"));
    assert_eq!(own, avoid);
}

#[test]
fn choice_instructions_opt_in_changes_the_prototypes() {
    let opts = EngineOptions {
        choice_instructions: true,
        ..EngineOptions::default()
    };
    let on = Engine::with_options(HashEmbedder::new(DIMS), opts);
    let off = Engine::new(HashEmbedder::new(DIMS));

    // With instructions present, the opt-in embeds "instructions. option", so
    // the same request scores differently than on the default path.
    let asked = instructed_choice("which asset does the customer want to sell");
    assert_ne!(
        choice_probs(&on, &asked),
        choice_probs(&off, &asked),
        "instructions should now reach the prototypes"
    );

    // Empty instructions: bit-identical to the default path.
    assert_eq!(
        choice_probs(&on, &instructed_choice("")),
        choice_probs(&off, &instructed_choice(""))
    );
}

fn with_other(state: &str) -> DecisionRequest {
    choice_request(
        state,
        vec![
            (
                "weather",
                structured("weather forecast", &["sunny rain clouds storm"], None),
            ),
            (
                "sports",
                structured("sports results", &["football goal match score"], None),
            ),
            ("other", structured("anything else", &[], None)),
        ],
    )
}

#[test]
fn catch_all_is_an_ordinary_option_by_default() {
    let engine = Engine::new(HashEmbedder::new(DIMS));
    let r = engine.decide(&with_other("anything else at all")).unwrap();
    let (choice, probs, _) = as_choice(&r.answers["q"]);
    // Its own text matches, so as an ordinary option it wins here.
    assert_eq!(choice, "other");
    assert_eq!(probs.len(), 3);
}

#[test]
fn catch_all_takes_off_topic_states_and_leaves_in_scope_ones() {
    let opts = EngineOptions {
        catch_all: Some("other".into()),
        ..EngineOptions::default()
    };
    let engine = Engine::with_options(HashEmbedder::new(DIMS), opts);

    let off = engine
        .decide(&with_other("banana bread recipe with walnuts"))
        .unwrap();
    let (c, p, meta) = as_choice(&off.answers["q"]);
    assert_eq!(c, "other");
    assert!((meta.confidence - p["other"]).abs() < 1e-6);
    let sum: f32 = p.values().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "probabilities sum to 1, got {sum}"
    );

    let on = engine
        .decide(&with_other("storm and rain in the weather forecast"))
        .unwrap();
    let (c, p, _) = as_choice(&on.answers["q"]);
    assert_eq!(c, "weather");
    assert!(p["other"] < 0.5);

    // The catch-all's own text is not matched as a prototype.
    let own = engine.decide(&with_other("anything else")).unwrap();
    let (_, p_own, _) = as_choice(&own.answers["q"]);
    let (_, p_off, _) = as_choice(&off.answers["q"]);
    assert!((p_own["other"] - p_off["other"]).abs() < 1e-6);
}
