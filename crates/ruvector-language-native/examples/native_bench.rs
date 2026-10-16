//! cargo run --release --example native_bench
//! Synthetic initialized weights measure execution, not prediction quality.
use ruvector_graph::{GraphDB, NodeBuilder};
use ruvector_kge::{Scorer, Tables};
use ruvector_language::{parser, runtime, types::*};
use ruvector_language_native::{
    artifact::{ModelArtifact, ModelBundle, NativePredictor, Origin, ScorerKind},
    capture_quiescent,
};
use std::{hint::black_box, time::Instant};

fn direct(
    tables: &Tables,
    scorer: &dyn Scorer,
    model: &ModelArtifact,
    request: &PredictionRequest,
) -> Vec<Prediction> {
    let mut result = Vec::with_capacity(request.subjects.len() * request.k);
    for subject in &request.subjects {
        let sid = model.entities.iter().position(|x| x == subject).unwrap();
        let mut row = Vec::with_capacity(model.entities.len());
        for oid in 0..model.entities.len() {
            row.push((
                scorer.score(
                    tables.entity(sid as u32).unwrap(),
                    tables.relation(0).unwrap(),
                    tables.entity(oid as u32).unwrap(),
                ),
                oid,
            ));
        }
        row.sort_unstable_by(|(a, ai), (b, bi)| {
            b.total_cmp(a)
                .then(model.entities[*ai].cmp(&model.entities[*bi]))
        });
        for (score, index) in row.into_iter().take(request.k) {
            result.push(Prediction {
                subject: subject.clone(),
                relation: request.relation.clone(),
                object: model.entities[index].clone(),
                score,
            });
        }
    }
    result
}

fn micros<T>(mut f: impl FnMut() -> T) -> Vec<f64> {
    for _ in 0..5 {
        black_box(f());
    }
    let mut measurements = Vec::new();
    for _ in 0..30 {
        let start = Instant::now();
        black_box(f());
        measurements.push(start.elapsed().as_secs_f64() * 1e6);
    }
    measurements.sort_by(f64::total_cmp);
    measurements
}

fn main() {
    let tables = Tables::new(256, 1, 32, 42);
    let model = ModelArtifact {
        version: 1,
        name: "benchmark".into(),
        revision: "initialization_only".into(),
        scorer: ScorerKind::Hole,
        dims: 32,
        entities: (0..256).map(|i| format!("entity{i:03}")).collect(),
        relations: vec!["related".into()],
        entity_weights: tables.entities_raw().to_vec(),
        relation_weights: tables.relations_raw().to_vec(),
        origin: Origin::Imported,
    };
    let request = PredictionRequest {
        model: model.name.clone(),
        relation: "related".into(),
        subjects: model.entities[..8].to_vec(),
        k: 5,
        max_work: 100_000_000,
    };
    let scorer = model.scorer().unwrap();
    let adapter = NativePredictor::new(ModelBundle {
        version: 1,
        models: vec![model.clone()],
    })
    .unwrap();
    let graph = GraphDB::new();
    for id in &request.subjects {
        graph
            .create_node(NodeBuilder::new().id(id.clone()).label("Entity").build())
            .unwrap();
    }
    let snapshot = capture_quiescent(&graph, "bench1", vec![], vec![]).unwrap();
    let program=parser::parse("program benchmark { effect graph.read; effect model.predict; effect decision.emit; budget 100000000; let s = nodes \"Entity\"; let p = predict s \"related\" model \"benchmark\" top 5; emit p; }").unwrap();
    let context = Context {
        now: 0,
        capabilities: [
            Effect::GraphRead,
            Effect::ModelPredict,
            Effect::DecisionEmit,
        ]
        .into_iter()
        .collect(),
        max_steps: 100_000_000,
        max_results: 100,
    };
    let expected = direct(&tables, &*scorer, &model, &request);
    assert_eq!(expected, adapter.predict(&request).unwrap().candidates);
    match runtime::execute(&program, &snapshot, &context, Some(&adapter))
        .unwrap()
        .output
    {
        Output::Predictions(batch) => assert_eq!(expected, batch.candidates),
        _ => panic!("prediction output expected"),
    }
    let baseline = micros(|| direct(&tables, &*scorer, &model, &request));
    let native = micros(|| adapter.predict(&request).unwrap());
    let language =
        micros(|| runtime::execute(&program, &snapshot, &context, Some(&adapter)).unwrap());
    println!(
        "{}",
        serde_json::json!({"schema":1,"scorer":"hole","dimensions":32,"entities":256,"subjects":8,"top_k":5,
        "samples":30,"warmups":5,"equivalent_rankings":true,"weights":"seeded initialization, not quality evidence",
        "direct_median_us":baseline[15],"direct_p95_us":baseline[28],"adapter_median_us":native[15],
        "adapter_p95_us":native[28],"language_median_us":language[15],"language_p95_us":language[28],
        "language_overhead_percent":100.0*(language[15]/baseline[15]-1.0),
        "limitations":"single process synthetic microbenchmark; direct baseline omits validation, permissions and receipt hashing"})
    );
}
