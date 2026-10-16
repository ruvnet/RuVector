use ruvector_graph::{EdgeBuilder, GraphDB, NodeBuilder, PropertyValue};
use ruvector_kge::Tables;
use ruvector_language::{parser, runtime, types::*};
use ruvector_language_native::{
    artifact::{ModelArtifact, ModelBundle, NativePredictor, Origin, ScorerKind},
    capture_quiescent, content_hash, load_models,
    training::*,
};

fn known(scorer: ScorerKind) -> ModelArtifact {
    ModelArtifact {
        version: 1,
        name: "incident_v1".into(),
        revision: "fixed_weights_1".into(),
        scorer,
        dims: 4,
        entities: vec!["api".into(), "cache".into(), "db".into()],
        relations: vec!["Cause".into()],
        entity_weights: vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.5, 0.5, 0.5, 0.5],
        relation_weights: vec![1.0, 0.0, 0.0, 1.0],
        origin: Origin::Imported,
    }
}

fn prediction(model: &ModelArtifact, work: u64) -> PredictionRequest {
    PredictionRequest {
        model: model.name.clone(),
        relation: "Cause".into(),
        subjects: vec!["api".into()],
        k: 3,
        max_work: work,
    }
}

fn context() -> Context {
    Context {
        now: 100,
        capabilities: [
            Effect::GraphRead,
            Effect::ModelPredict,
            Effect::DecisionEmit,
        ]
        .into_iter()
        .collect(),
        max_steps: 1_000_000,
        max_results: 100,
    }
}

#[test]
fn native_hole_and_rotate_match_direct_score_for_every_candidate() {
    for kind in [ScorerKind::Hole, ScorerKind::Rotate] {
        let model = known(kind);
        let tables = model.tables().unwrap();
        let scorer = model.scorer().unwrap();
        let predictor = NativePredictor::new(ModelBundle {
            version: 1,
            models: vec![model.clone()],
        })
        .unwrap();
        let batch = predictor.predict(&prediction(&model, 1_000_000)).unwrap();
        assert_eq!(batch.model_hash, content_hash(&model).unwrap());
        assert_eq!(batch.revision, "fixed_weights_1");
        assert!(batch.exact);
        for candidate in batch.candidates {
            let object = model
                .entities
                .iter()
                .position(|s| *s == candidate.object)
                .unwrap() as u32;
            let direct = scorer.score(
                tables.entity(0).unwrap(),
                tables.relation(0).unwrap(),
                tables.entity(object).unwrap(),
            );
            assert_eq!(candidate.score.to_bits(), direct.to_bits());
        }
        assert_eq!(
            predictor.predict(&prediction(&model, 1)).unwrap_err().code,
            "BUDGET"
        );
        let mut request = prediction(&model, 1_000_000);
        request.subjects.push("unseen".into());
        assert!(predictor.predict(&request).is_err());
    }
}

#[test]
fn native_graph_capture_execute_and_recompute_after_deletion() {
    let db = GraphDB::new();
    db.create_node(
        NodeBuilder::new()
            .id("api")
            .label("Service")
            .property("embedding", PropertyValue::FloatArray(vec![1.0, 0.0]))
            .build(),
    )
    .unwrap();
    db.create_node(NodeBuilder::new().id("db").label("Database").build())
        .unwrap();
    db.create_edge(
        EdgeBuilder::new("api".into(), "db".into(), "DEPENDS_ON")
            .id("dep")
            .build(),
    )
    .unwrap();
    let snapshot = capture_quiescent(
        &db,
        "r1",
        vec![Observation {
            id: "o1".into(),
            subject: "db".into(),
            predicate: "Failing".into(),
            value: true,
            source: "probe".into(),
            observed_at: 90,
            expires_at: 110,
        }],
        vec![],
    )
    .unwrap();
    assert_eq!(snapshot.nodes[0].vector, Some(vec![1.0, 0.0]));
    assert_eq!(snapshot.edges[0].target, db.get_edge("dep").unwrap().to);
    let program = parser::parse("program diagnosis { effect graph.read; effect decision.emit; budget 100000; let s = select [\"api\"]; let d = traverse s \"DEPENDS_ON\" depth 1; let e = observe d \"Failing\" within 20; let a = assess e; emit a; }").unwrap();
    let result = runtime::execute(&program, &snapshot, &context(), None).unwrap();
    match result.output {
        Output::Decisions(d) => assert!(d
            .iter()
            .any(|d| d.subject == "db" && d.status == Status::Supported)),
        _ => panic!("expected decisions"),
    }
    let mut expired = context();
    expired.now = 110;
    let expired_result = runtime::execute(&program, &snapshot, &expired, None).unwrap();
    match expired_result.output {
        Output::Decisions(d) => assert!(d
            .iter()
            .any(|d| d.subject == "db" && d.status == Status::Unknown)),
        _ => panic!("expected decisions"),
    }
    db.delete_edge("dep").unwrap();
    let fresh = capture_quiescent(&db, "r2", snapshot.observations, vec![]).unwrap();
    let after = runtime::execute(&program, &fresh, &context(), None).unwrap();
    assert_ne!(result.snapshot_hash, after.snapshot_hash);
}

#[test]
fn native_prediction_executes_from_real_graph_and_replays_identically() {
    let db = GraphDB::new();
    db.create_node(NodeBuilder::new().id("api").label("Service").build())
        .unwrap();
    let snapshot = capture_quiescent(&db, "r1", vec![], vec![]).unwrap();
    let model = known(ScorerKind::Hole);
    let predictor = NativePredictor::new(ModelBundle {
        version: 1,
        models: vec![model.clone()],
    })
    .unwrap();
    let program = parser::parse("program rank { effect graph.read; effect model.predict; effect decision.emit; budget 100000; let s = nodes \"Service\"; let p = predict s \"Cause\" model \"incident_v1\" top 3; emit p; }").unwrap();
    let receipt = runtime::execute(&program, &snapshot, &context(), Some(&predictor)).unwrap();
    assert_eq!(
        receipt,
        runtime::execute(&program, &snapshot, &context(), Some(&predictor)).unwrap()
    );
    match receipt.output {
        Output::Predictions(p) => assert_eq!(
            p,
            predictor.predict(&prediction(&model, 1_000_000)).unwrap()
        ),
        _ => panic!("expected predictions"),
    }
}

#[test]
fn rejects_malformed_models_and_unsupported_capture_without_panicking() {
    let model = known(ScorerKind::Hole);
    for dims in [0, 1, 3, 2048] {
        let mut invalid = model.clone();
        invalid.dims = dims;
        assert!(invalid.tables().is_err());
    }
    let mut invalid = model.clone();
    invalid.entity_weights[0] = f32::NAN;
    assert!(invalid.validate().is_err());
    let mut invalid = model.clone();
    invalid.entities[1] = "api".into();
    assert!(invalid.validate().is_err());
    let json = serde_json::to_string(&ModelBundle {
        version: 1,
        models: vec![model],
    })
    .unwrap();
    assert!(
        load_models(&json.replacen("\"version\":1", "\"version\":1,\"version\":1", 1)).is_err()
    );
    assert!(load_models(&json.replacen("\"dims\":4", "\"dims\":4,\"hidden\":1", 1)).is_err());
    let db = GraphDB::new();
    db.create_node(NodeBuilder::new().id("x").label("A").label("B").build())
        .unwrap();
    assert!(capture_quiescent(&db, "r", vec![], vec![]).is_err());
}

#[test]
fn borrowed_capture_rejects_oversized_payload_and_visitors_stop_on_error() {
    let db = GraphDB::new();
    db.create_node(
        NodeBuilder::new()
            .id("large")
            .label("Entity")
            .property("payload", "x".repeat(100_001))
            .build(),
    )
    .unwrap();
    db.create_node(NodeBuilder::new().id("small").label("Entity").build())
        .unwrap();
    assert_eq!(
        capture_quiescent(&db, "r1", vec![], vec![])
            .unwrap_err()
            .code,
        "CAPTURE"
    );
    let mut visited = 0;
    let stopped: Result<(), &str> = db.try_for_each_node(|_| {
        visited += 1;
        Err("stop")
    });
    assert_eq!(stopped, Err("stop"));
    assert_eq!(visited, 1);
    db.create_edge(
        EdgeBuilder::new("large".into(), "small".into(), "related")
            .id("large_edge")
            .property("payload", "x".repeat(100_001))
            .build(),
    )
    .unwrap();
    let mut edge_visited = 0;
    let stopped: Result<(), &str> = db.try_for_each_edge(|edge| {
        edge_visited += 1;
        assert_eq!(edge.id, "large_edge");
        Err("stop")
    });
    assert_eq!(stopped, Err("stop"));
    assert_eq!(edge_visited, 1);
}

pub fn training_request(kind: ScorerKind) -> TrainingRequest {
    let fact = |s: &str, o: &str| Fact {
        subject: s.into(),
        relation: "Cause".into(),
        object: o.into(),
    };
    TrainingRequest {
        version: 1,
        name: "incident_v1".into(),
        revision: "trained_1".into(),
        scorer: kind,
        dims: 8,
        epochs: 5,
        seed: 42,
        learning_rate: 0.05,
        max_work: 100_000_000,
        data: TrainingData {
            train: vec![fact("api", "cache"), fact("cache", "db"), fact("db", "api")],
            evaluation: vec![fact("api", "db")],
        },
    }
}

#[test]
fn actual_training_changes_weights_is_reproducible_and_matches_native_evaluation() {
    for kind in [ScorerKind::Hole, ScorerKind::Rotate] {
        let request = training_request(kind);
        let candidate = train_candidate(&request).unwrap();
        assert_eq!(candidate, train_candidate(&request).unwrap());
        let initial = Tables::new(3, 1, 8, 42);
        assert_ne!(candidate.model.entity_weights, initial.entities_raw());
        assert_eq!(candidate.report.losses.len(), 5);
        assert_eq!(candidate.report.trained.combined.count, 2);
        let evaluation = evaluate_model(
            &candidate.model,
            &EvaluationRequest {
                version: 1,
                data: request.data,
                seed: 42,
                max_work: 100_000_000,
            },
        )
        .unwrap();
        assert_eq!(candidate.report.trained, evaluation.metrics);
        let bundle = promote(&candidate, 0.0).unwrap();
        assert_eq!(bundle.models.len(), 1);
        let tables = candidate.model.tables().unwrap();
        let scorer = candidate.model.scorer().unwrap();
        let prediction = NativePredictor::new(bundle)
            .unwrap()
            .predict(&prediction(&candidate.model, 1_000_000))
            .unwrap();
        let subject = candidate
            .model
            .entities
            .iter()
            .position(|s| s == "api")
            .unwrap() as u32;
        for p in prediction.candidates {
            let object = candidate
                .model
                .entities
                .iter()
                .position(|s| s == &p.object)
                .unwrap() as u32;
            assert_eq!(
                p.score.to_bits(),
                scorer
                    .score(
                        tables.entity(subject).unwrap(),
                        tables.relation(0).unwrap(),
                        tables.entity(object).unwrap()
                    )
                    .to_bits()
            );
        }
        let mut damaged = candidate;
        damaged.model.entity_weights[0] += 0.01;
        assert!(promote(&damaged, 0.0).is_err());
    }
}

#[test]
fn rejects_leakage_unseen_entities_and_excessive_training_work() {
    let mut request = training_request(ScorerKind::Hole);
    request.data.evaluation.push(request.data.train[0].clone());
    assert!(train_candidate(&request).is_err());
    request = training_request(ScorerKind::Hole);
    request.data.evaluation[0].object = "unseen".into();
    assert!(train_candidate(&request).is_err());
    request = training_request(ScorerKind::Hole);
    request.max_work = 1;
    assert_eq!(train_candidate(&request).unwrap_err().code, "BUDGET");
}

#[test]
fn malformed_promotion_metrics_are_rejected_without_overflow() {
    let candidate = train_candidate(&training_request(ScorerKind::Hole)).unwrap();
    let mut invalid = candidate.clone();
    invalid.report.evaluation_count = usize::MAX;
    assert!(promote(&invalid, 0.0).is_err());
    let mut invalid = candidate.clone();
    invalid.report.trained.combined.mrr = 1.0;
    invalid.report.trained.head.mrr = 0.0;
    invalid.report.trained.tail.mrr = 0.0;
    assert!(promote(&invalid, 0.0).is_err());
    let mut invalid = candidate;
    invalid.report.trained.head.hits1 = f32::NAN;
    assert!(promote(&invalid, 0.0).is_err());
}

#[test]
fn native_prediction_bounds_long_names_before_output_allocation() {
    let entities: Vec<String> = (0..1000)
        .map(|i| format!("{i:04}{}", "x".repeat(508)))
        .collect();
    let model = ModelArtifact {
        version: 1,
        name: "large".into(),
        revision: "one".into(),
        scorer: ScorerKind::Rotate,
        dims: 2,
        entities: entities.clone(),
        relations: vec!["r".repeat(512)],
        entity_weights: vec![0.0; 2000],
        relation_weights: vec![1.0, 0.0],
        origin: Origin::Imported,
    };
    let predictor = NativePredictor::new(ModelBundle {
        version: 1,
        models: vec![model],
    })
    .unwrap();
    let request = PredictionRequest {
        model: "large".into(),
        relation: "r".repeat(512),
        subjects: entities[..50].to_vec(),
        k: 1000,
        max_work: 1_000_000_000,
    };
    let error = predictor.predict(&request).unwrap_err();
    assert_eq!(error.code, "LIMIT");
    assert!(error.message.contains("16 MiB"));
    // Every score ties. The bounded result must retain lexical tie order and
    // copy only the requested rows, not full strings for the whole vocabulary.
    let short = PredictionRequest {
        subjects: vec![entities[900].clone()],
        k: 3,
        ..request
    };
    let result = predictor.predict(&short).unwrap();
    assert_eq!(
        result
            .candidates
            .iter()
            .map(|p| &p.object)
            .collect::<Vec<_>>(),
        entities[..3].iter().collect::<Vec<_>>()
    );
    assert!(
        serde_json::to_vec(&result).unwrap().len()
            < ruvector_language_native::artifact::MAX_PREDICTION_BYTES
    );
}
