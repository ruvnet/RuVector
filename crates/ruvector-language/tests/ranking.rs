use ruvector_language::*;
use std::collections::BTreeMap;

#[test]
fn partial_selection_matches_full_sort_for_ties_zeros_missing_and_extreme_vectors() {
    for seed in 0..16 {
        let nodes: Vec<_> = (0..64)
            .map(|i| {
                let vector = match (i + seed) % 8 {
                    0 => None,
                    1 => Some(vec![0.0, 0.0]),
                    2 => Some(vec![f32::MAX, -f32::MAX]),
                    3 => Some(vec![-1.0, 0.0]),
                    4 => Some(vec![1.0, 0.0]),
                    _ => Some(vec![((i * 7 + seed) % 13) as f32 - 6.0, 1.0]),
                };
                Node {
                    id: format!("n{i:03}"),
                    label: "N".into(),
                    properties: BTreeMap::new(),
                    vector,
                }
            })
            .collect();
        let mut expected: Vec<_> = nodes
            .iter()
            .filter_map(|n| {
                n.vector.as_ref().and_then(|v| {
                    let norm = (f64::from(v[0]).powi(2) + f64::from(v[1]).powi(2)).sqrt();
                    (norm > 0.0).then(|| (n.id.clone(), f64::from(v[0]) / norm))
                })
            })
            .collect();
        expected.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let snapshot = Snapshot {
            version: 1,
            revision: "rank".into(),
            nodes,
            edges: vec![],
            observations: vec![],
            rules: vec![],
        };
        let context = Context {
            now: 1000,
            capabilities: [Effect::GraphRead, Effect::DecisionEmit].into(),
            max_steps: 100000,
            max_results: 100,
        };
        for k in [1, 4, 32, 64] {
            let p=parser::parse(&format!(r#"program rank {{ effect graph.read; effect decision.emit; budget 100000; let n = nodes "N"; let s = similar n [1.0,0.0] top {k}; let e = observe s "P" within 100; let d = assess e; emit d; }}"#)).unwrap();
            let a = runtime::execute(&p, &snapshot, &context, None).unwrap();
            let Output::Decisions(d) = &a.output else {
                panic!()
            };
            let mut ids: Vec<_> = expected.iter().take(k).map(|x| x.0.clone()).collect();
            ids.sort();
            assert_eq!(d.iter().map(|d| d.subject.clone()).collect::<Vec<_>>(), ids);
            assert_eq!(
                a,
                runtime::PreparedSnapshot::new(&snapshot)
                    .unwrap()
                    .execute(&p, &context, None)
                    .unwrap()
            );
        }
    }
}
#[test]
fn selection_still_validates_candidates_outside_top_k() {
    let snapshot = Snapshot {
        version: 1,
        revision: "bad-dimension".into(),
        nodes: vec![
            Node {
                id: "a".into(),
                label: "N".into(),
                properties: BTreeMap::new(),
                vector: Some(vec![1.0, 0.0]),
            },
            Node {
                id: "z".into(),
                label: "N".into(),
                properties: BTreeMap::new(),
                vector: Some(vec![0.0]),
            },
        ],
        edges: vec![],
        observations: vec![],
        rules: vec![],
    };
    let p=parser::parse(r#"program p { effect graph.read; effect decision.emit; budget 100000; let n = nodes "N"; let s = similar n [1.0,0.0] top 1; let e = observe s "P" within 100; let d = assess e; emit d; }"#).unwrap();
    let c = Context {
        now: 1000,
        capabilities: [Effect::GraphRead, Effect::DecisionEmit].into(),
        max_steps: 100000,
        max_results: 100,
    };
    assert_eq!(
        runtime::execute(&p, &snapshot, &c, None).unwrap_err().code,
        "INVALID_INPUT"
    );
}
