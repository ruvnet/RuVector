//! Synthetic agent memory fixture and an independent, status-only evaluator.
#![allow(dead_code)]
use ruvector_language::*;
use std::collections::{BTreeMap, BTreeSet};
pub type Answers = BTreeMap<(String, String), Status>;
pub fn request(size: usize, seed: u64) -> RunRequest {
    let source = r#"program memory {
      effect graph.read; effect decision.emit; budget 10000000;
      let memories = nodes "Memory";
      let nearest = similar memories [1.0, 0.0] top 4;
      let related = traverse nearest "RELATED" depth 2;
      let evidence = observe related "Reliable" within 100;
      let usable = infer [evidence] using ["allow", "deny"];
      let decisions = assess usable; emit decisions;
    }"#;
    let nodes = (0..size)
        .map(|i| Node {
            id: format!("m{i:05}"),
            label: "Memory".into(),
            properties: BTreeMap::new(),
            vector: Some(vec![
                1.0,
                ((i * 17 + seed as usize % size) % size) as f32 / size as f32,
            ]),
        })
        .collect();
    let edges = (0..size)
        .map(|i| Edge {
            id: format!("e{i}"),
            source: format!("m{i:05}"),
            target: format!("m{:05}", (i + 7) % size),
            relation: "RELATED".into(),
        })
        .collect();
    let mut observations = vec![];
    for i in 0..size {
        for (suffix, value) in [("positive", true), ("negative", false)] {
            let state = (i as u64 + seed) % 4;
            if state == 3 || (state == 0 && !value) || (state == 1 && value) {
                continue;
            }
            observations.push(Observation {
                id: format!("o{i}-{suffix}"),
                subject: format!("m{i:05}"),
                predicate: "Reliable".into(),
                value,
                source: if value { "primary" } else { "review" }.into(),
                observed_at: 950 + (seed % 7) * 10,
                expires_at: 1100,
            });
        }
    }
    let rules = [("allow", true), ("deny", false)]
        .into_iter()
        .map(|(id, value)| Rule {
            id: id.into(),
            requires: vec![Condition {
                predicate: "Reliable".into(),
                value,
            }],
            conclusion: Condition {
                predicate: "Usable".into(),
                value,
            },
        })
        .collect();
    RunRequest {
        program: parser::parse(source).unwrap(),
        snapshot: Snapshot {
            version: 1,
            revision: format!("synthetic-memory-{size}-{seed}"),
            nodes,
            edges,
            observations,
            rules,
        },
        context: Context {
            now: 1000,
            capabilities: BTreeSet::from([Effect::GraphRead, Effect::DecisionEmit]),
            max_steps: 10_000_000,
            max_results: 100_000,
        },
    }
}
/// Independent direct implementation of this fixed application, not a language
/// interpreter. Omits validation, grants, work limits, provenance and receipt hashes.
pub fn reference(request: &RunRequest) -> Answers {
    let s = &request.snapshot;
    let mut ranked: Vec<_> = s
        .nodes
        .iter()
        .filter(|n| n.label == "Memory")
        .filter_map(|n| {
            n.vector.as_ref().map(|v| {
                let score =
                    f64::from(v[0]) / (f64::from(v[0]).powi(2) + f64::from(v[1]).powi(2)).sqrt();
                (n.id.clone(), score)
            })
        })
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut selected: BTreeSet<String> = ranked.into_iter().take(4).map(|p| p.0).collect();
    for _ in 0..2 {
        let extra: Vec<_> = s
            .edges
            .iter()
            .filter(|e| e.relation == "RELATED" && selected.contains(&e.source))
            .map(|e| e.target.clone())
            .collect();
        selected.extend(extra);
    }
    let mut answers = BTreeMap::new();
    for subject in selected {
        let mut positive = false;
        let mut negative = false;
        for o in &s.observations {
            if o.subject == subject
                && o.predicate == "Reliable"
                && o.observed_at <= request.context.now
                && request.context.now < o.expires_at
                && request.context.now - o.observed_at <= 100
            {
                if o.value {
                    positive = true;
                } else {
                    negative = true;
                }
            }
        }
        let status = match (positive, negative) {
            (true, true) => Status::Conflicting,
            (true, false) => Status::Supported,
            (false, true) => Status::Refuted,
            _ => Status::Unknown,
        };
        answers.insert((subject.clone(), "Reliable".into()), status.clone());
        answers.insert((subject, "Usable".into()), status);
    }
    answers
}
pub fn actual(receipt: &Receipt) -> Answers {
    let Output::Decisions(d) = &receipt.output else {
        panic!("expected decisions")
    };
    d.iter()
        .map(|d| ((d.subject.clone(), d.predicate.clone()), d.status.clone()))
        .collect()
}
