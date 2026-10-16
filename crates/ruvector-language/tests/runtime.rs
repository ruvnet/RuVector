use ruvector_language::{
    parser,
    runtime::{execute, validate_snapshot, verify_receipt},
    *,
};
use std::collections::{BTreeMap, BTreeSet};

fn node(id: &str) -> Node {
    Node {
        id: id.into(),
        label: "Service".into(),
        properties: BTreeMap::new(),
        vector: None,
    }
}
fn observation(id: &str, predicate: &str, value: bool, expires_at: u64) -> Observation {
    Observation {
        id: id.into(),
        subject: "api".into(),
        predicate: predicate.into(),
        value,
        source: format!("source_{id}"),
        observed_at: 90,
        expires_at,
    }
}
fn snapshot() -> Snapshot {
    Snapshot {
        version: 1,
        revision: "revision1".into(),
        nodes: vec![node("api"), node("db")],
        edges: vec![Edge {
            id: "depends".into(),
            source: "api".into(),
            target: "db".into(),
            relation: "DEPENDS_ON".into(),
        }],
        observations: vec![observation("o1", "Failing", true, 200)],
        rules: vec![],
    }
}
fn context() -> Context {
    Context {
        now: 100,
        capabilities: BTreeSet::from([
            Effect::GraphRead,
            Effect::DecisionEmit,
            Effect::ModelPredict,
        ]),
        max_steps: 100_000,
        max_results: 1000,
    }
}
fn program(body: &str) -> Program {
    parser::parse(&format!("program test {{ effect graph.read; effect model.predict; effect decision.emit; budget 100000; {body} }}")).unwrap()
}
fn observing() -> Program {
    program("let n = select [\"api\"]; let e = observe n \"Failing\" within 300; let d = assess e; emit d;")
}
fn decisions(receipt: &Receipt) -> &[Decision] {
    match &receipt.output {
        Output::Decisions(d) => d,
        _ => panic!("expected decisions"),
    }
}
fn condition(predicate: &str, value: bool) -> Condition {
    Condition {
        predicate: predicate.into(),
        value,
    }
}
fn rule(id: &str, requires: &[(&str, bool)], predicate: &str, value: bool) -> Rule {
    Rule {
        id: id.into(),
        requires: requires.iter().map(|(p, v)| condition(p, *v)).collect(),
        conclusion: condition(predicate, value),
    }
}

#[test]
fn observed_supported_unknown_refuted_and_conflicting() {
    let mut graph = snapshot();
    let p = observing();
    assert_eq!(
        decisions(&execute(&p, &graph, &context(), None).unwrap())[0].status,
        Status::Supported
    );
    graph
        .observations
        .push(observation("o2", "Failing", false, 200));
    let receipt = execute(&p, &graph, &context(), None).unwrap();
    assert_eq!(decisions(&receipt)[0].status, Status::Conflicting);
    assert_eq!(decisions(&receipt)[0].evidence.len(), 2);
    graph.observations.remove(0);
    assert_eq!(
        decisions(&execute(&p, &graph, &context(), None).unwrap())[0].status,
        Status::Refuted
    );
    graph.observations.clear();
    let absent = execute(&p, &graph, &context(), None).unwrap();
    assert_eq!(decisions(&absent)[0].status, Status::Unknown);
    assert!(decisions(&absent)[0].evidence.is_empty());
}

#[test]
fn empty_observation_domain_keeps_all_selected_subjects() {
    let mut graph = snapshot();
    graph.observations.clear();
    let p = program("let n = nodes \"Service\"; let e = observe n \"Missing\" within 60; let d = assess e; emit d;");
    let receipt = execute(&p, &graph, &context(), None).unwrap();
    assert_eq!(decisions(&receipt).len(), 2);
    assert!(decisions(&receipt)
        .iter()
        .all(|d| d.status == Status::Unknown));
}

#[test]
fn expiry_freshness_and_future_evidence() {
    let mut graph = snapshot();
    let p = program("let n = select [\"api\"]; let e = observe n \"Failing\" within 10; let d = assess e; emit d;");
    let mut c = context();
    let receipt = execute(&p, &graph, &c, None).unwrap();
    assert_eq!(decisions(&receipt)[0].evidence[0].expires_at, 101);
    c.now = 101;
    assert_eq!(
        decisions(&execute(&p, &graph, &c, None).unwrap())[0].status,
        Status::Unknown
    );
    c.now = 89;
    assert_eq!(
        decisions(&execute(&p, &graph, &c, None).unwrap())[0].status,
        Status::Unknown
    );
    c.now = 100;
    graph.observations[0].expires_at = 100;
    assert_eq!(
        decisions(&execute(&p, &graph, &c, None).unwrap())[0].status,
        Status::Unknown
    );
}

fn inference() -> Program {
    program("let n = select [\"api\"]; let e = observe n \"Failing\" within 300; let f = observe n \"Healthy\" within 300; let i = infer [e,f] using [\"cause\",\"page\"]; let d = assess i; emit d;")
}
fn inferred_graph() -> Snapshot {
    let mut graph = snapshot();
    graph
        .observations
        .push(observation("o2", "Healthy", false, 150));
    graph.rules = vec![
        rule(
            "cause",
            &[("Failing", true), ("Healthy", false)],
            "Cause",
            true,
        ),
        rule("page", &[("Cause", true)], "Page", true),
    ];
    graph
}
#[test]
fn signed_conjunction_infers_and_preserves_sources_and_minimum_expiry() {
    let receipt = execute(&inference(), &inferred_graph(), &context(), None).unwrap();
    let page = decisions(&receipt)
        .iter()
        .find(|d| d.predicate == "Page")
        .unwrap();
    assert_eq!(page.status, Status::Supported);
    let evidence = &page.evidence[0];
    assert_eq!(evidence.kind, EvidenceKind::Derived);
    assert_eq!(evidence.expires_at, 150);
    assert_eq!(
        evidence.observation_ids,
        BTreeSet::from(["o1".into(), "o2".into()])
    );
    assert_eq!(
        evidence.sources,
        BTreeSet::from(["source_o1".into(), "source_o2".into()])
    );
    assert_eq!(
        evidence.rule_ids,
        BTreeSet::from(["cause".into(), "page".into()])
    );
}
#[test]
fn deleting_or_expiring_a_premise_retracts_transitive_conclusions() {
    let mut graph = inferred_graph();
    let p = inference();
    let before = execute(&p, &graph, &context(), None).unwrap();
    graph.observations.retain(|o| o.id != "o2");
    let after = execute(&p, &graph, &context(), None).unwrap();
    assert_ne!(before.snapshot_hash, after.snapshot_hash);
    assert_eq!(before.snapshot_revision, after.snapshot_revision);
    for predicate in ["Cause", "Page"] {
        assert_eq!(
            decisions(&after)
                .iter()
                .find(|d| d.predicate == predicate)
                .unwrap()
                .status,
            Status::Unknown
        );
    }
    let mut c = context();
    c.now = 150;
    let expired = execute(&p, &inferred_graph(), &c, None).unwrap();
    assert_eq!(
        decisions(&expired)
            .iter()
            .find(|d| d.predicate == "Page")
            .unwrap()
            .status,
        Status::Unknown
    );
}
#[test]
fn unsupported_cycles_do_not_self_support_and_supported_cycles_terminate() {
    let mut graph = snapshot();
    graph.observations.clear();
    graph.rules = vec![
        rule("ab", &[("A", true)], "B", true),
        rule("ba", &[("B", true)], "A", true),
    ];
    let p = program("let n = select [\"api\"]; let e = observe n \"A\" within 300; let i = infer [e] using [\"ba\",\"ab\"]; let d = assess i; emit d;");
    let unknown = execute(&p, &graph, &context(), None).unwrap();
    assert!(decisions(&unknown)
        .iter()
        .all(|d| d.status == Status::Unknown));
    graph.observations.push(observation("o1", "A", true, 200));
    let supported = execute(&p, &graph, &context(), None).unwrap();
    assert!(decisions(&supported)
        .iter()
        .all(|d| d.status == Status::Supported));
    assert_eq!(
        decisions(&supported)
            .iter()
            .find(|d| d.predicate == "B")
            .unwrap()
            .evidence[0]
            .rule_ids,
        BTreeSet::from(["ab".into()])
    );
}
#[test]
fn canonical_witness_changes_when_the_first_source_expires() {
    let mut graph = snapshot();
    graph.observations[0].expires_at = 101;
    graph
        .observations
        .push(observation("o2", "Failing", true, 200));
    graph
        .rules
        .push(rule("cause", &[("Failing", true)], "Cause", true));
    let p=program("let n = select [\"api\"]; let e = observe n \"Failing\" within 300; let i = infer [e] using [\"cause\"]; let d = assess i; emit d;");
    let before = execute(&p, &graph, &context(), None).unwrap();
    assert_eq!(
        decisions(&before)
            .iter()
            .find(|d| d.predicate == "Failing")
            .unwrap()
            .evidence
            .len(),
        2
    );
    assert_eq!(
        decisions(&before)
            .iter()
            .find(|d| d.predicate == "Cause")
            .unwrap()
            .evidence[0]
            .observation_ids,
        BTreeSet::from(["o1".into()])
    );
    let mut c = context();
    c.now = 101;
    let after = execute(&p, &graph, &c, None).unwrap();
    assert_eq!(
        decisions(&after)
            .iter()
            .find(|d| d.predicate == "Cause")
            .unwrap()
            .evidence[0]
            .observation_ids,
        BTreeSet::from(["o2".into()])
    );
}
#[test]
fn explicit_negative_premises_cannot_be_inferred_from_absence() {
    let mut graph = inferred_graph();
    graph.observations.retain(|o| o.predicate != "Healthy");
    let receipt = execute(&inference(), &graph, &context(), None).unwrap();
    assert_eq!(
        decisions(&receipt)
            .iter()
            .find(|d| d.predicate == "Cause")
            .unwrap()
            .status,
        Status::Unknown
    );
}
#[test]
fn receipts_replay_bit_identically_and_detect_tampering() {
    let p = observing();
    let graph = snapshot();
    let c = context();
    let first = execute(&p, &graph, &c, None).unwrap();
    assert_eq!(first, execute(&p, &graph, &c, None).unwrap());
    verify_receipt(&first).unwrap();
    let mut tampered = first.clone();
    tampered.now += 1;
    assert!(verify_receipt(&tampered).is_err());
    let mut reordered = graph.clone();
    reordered.nodes.reverse();
    assert_eq!(first, execute(&p, &reordered, &c, None).unwrap());
}
#[test]
fn graph_traversal_is_bounded_cyclic_and_includes_seed() {
    let mut graph = snapshot();
    graph.edges.push(Edge {
        id: "cycle".into(),
        source: "db".into(),
        target: "api".into(),
        relation: "DEPENDS_ON".into(),
    });
    let p=program("let n = select [\"api\"]; let t = traverse n \"DEPENDS_ON\" depth 64; let e = observe t \"Failing\" within 300; let d = assess e; emit d;");
    let receipt = execute(&p, &graph, &context(), None).unwrap();
    assert_eq!(
        decisions(&receipt)
            .iter()
            .map(|d| d.subject.as_str())
            .collect::<Vec<_>>(),
        vec!["api", "db"]
    );
    let mut c = context();
    c.max_results = 1;
    assert_eq!(
        execute(&p, &graph, &c, None).unwrap_err().code,
        "RESULT_LIMIT"
    );
}
#[test]
fn exact_cosine_similarity_uses_stable_id_ties() {
    let mut graph = snapshot();
    graph.nodes[0].vector = Some(vec![1., 0.]);
    graph.nodes[1].vector = Some(vec![1., 0.]);
    let p=program("let n = nodes \"Service\"; let t = similar n [1,0] top 1; let e = observe t \"Failing\" within 300; let d = assess e; emit d;");
    assert_eq!(
        decisions(&execute(&p, &graph, &context(), None).unwrap())[0].subject,
        "api"
    );
    graph.nodes[1].vector = Some(vec![1., 0., 0.]);
    assert_eq!(
        execute(&p, &graph, &context(), None).unwrap_err().code,
        "INVALID_INPUT"
    );
}
#[test]
fn permissions_fail_before_provider_or_graph_work_and_budgets_fail_closed() {
    let mut c = context();
    c.capabilities.remove(&Effect::GraphRead);
    let mut graph = snapshot();
    graph.version = 99;
    assert_eq!(
        execute(&observing(), &graph, &c, None).unwrap_err().code,
        "UNAUTHORIZED"
    );
    let mut c = context();
    c.max_steps = 1;
    assert_eq!(
        execute(&observing(), &snapshot(), &c, None)
            .unwrap_err()
            .code,
        "BUDGET_EXCEEDED"
    );
    let mut p = observing();
    p.budget = 1;
    assert_eq!(
        execute(&p, &snapshot(), &context(), None).unwrap_err().code,
        "BUDGET_EXCEEDED"
    );
}
#[test]
fn invalid_snapshots_are_rejected() {
    let base = snapshot();
    let mut variants = Vec::new();
    let mut v = base.clone();
    v.nodes.push(v.nodes[0].clone());
    variants.push(v);
    let mut v = base.clone();
    v.edges[0].target = "missing".into();
    variants.push(v);
    let mut v = base.clone();
    v.observations[0].expires_at = 90;
    variants.push(v);
    let mut v = base.clone();
    v.nodes[0].vector = Some(vec![f32::NAN]);
    variants.push(v);
    let mut v = base.clone();
    v.rules.push(rule("empty", &[], "True", true));
    variants.push(v);
    for graph in variants {
        assert!(validate_snapshot(&graph).is_err());
    }
}

struct Provider {
    bad: bool,
}
impl Predictor for Provider {
    fn predict(&self, r: &PredictionRequest) -> Result<PredictionBatch> {
        Ok(PredictionBatch {
            model: r.model.clone(),
            revision: "v1".into(),
            model_hash: "a".repeat(64),
            exact: true,
            work: 10,
            candidates: vec![Prediction {
                subject: r.subjects[0].clone(),
                relation: r.relation.clone(),
                object: "outside_graph".into(),
                score: if self.bad { f32::NAN } else { 0.8 },
            }],
        })
    }
}
fn predicting() -> Program {
    program("let n = select [\"api\"]; let p = predict n \"Cause\" model \"m1\" top 1; emit p;")
}
#[test]
fn predictions_remain_typed_and_provider_output_is_validated() {
    let receipt = execute(
        &predicting(),
        &snapshot(),
        &context(),
        Some(&Provider { bad: false }),
    )
    .unwrap();
    assert!(matches!(receipt.output, Output::Predictions(_)));
    assert_eq!(
        execute(
            &predicting(),
            &snapshot(),
            &context(),
            Some(&Provider { bad: true })
        )
        .unwrap_err()
        .code,
        "PROVIDER_ERROR"
    );
    assert_eq!(
        execute(&predicting(), &snapshot(), &context(), None)
            .unwrap_err()
            .code,
        "PROVIDER_UNAVAILABLE"
    );
    let mut p = predicting();
    p.steps.push(Step {
        bind: "d".into(),
        op: Op::Assess { input: "p".into() },
    });
    p.emit = "d".into();
    assert!(execute(&p, &snapshot(), &context(), Some(&Provider { bad: false })).is_err());
}
#[test]
fn provenance_growth_is_bounded() {
    let mut c = context();
    c.max_results = 3;
    let error = execute(&inference(), &inferred_graph(), &c, None).unwrap_err();
    assert_eq!(error.code, "RESULT_LIMIT");
}
