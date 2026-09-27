//! Synthetic numeric feature fixtures, not a trained sensor or language model.
use ruvector_typesafe_core::{
    embedder::l2_normalize,
    engine::{Engine, EngineOptions, LabeledExample},
    *,
};
use serde_json::json;
use std::{collections::BTreeMap, fs, path::PathBuf};

struct Numeric {
    dims: usize,
    prototypes: Vec<Vec<f32>>,
    negative: Vec<f32>,
}
impl Embedder for Numeric {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts
            .iter()
            .map(|t| {
                if *t == "negative" {
                    return Ok(self.negative.clone());
                }
                if let Some(index) = t.strip_prefix("prototype") {
                    return Ok(self.prototypes[index.parse::<usize>().unwrap()].clone());
                }
                let mut v: Vec<f32> = t.split(',').map(|s| s.parse::<f32>().unwrap()).collect();
                assert_eq!(v.len(), self.dims);
                l2_normalize(&mut v);
                Ok(v)
            })
            .collect()
    }
    fn dims(&self) -> usize {
        self.dims
    }
    fn id(&self) -> &str {
        "synthetic-numeric-features-v1"
    }
}
fn text(x: &[f32]) -> String {
    x.iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(",")
}
fn next(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
}
fn main() {
    let out = PathBuf::from(std::env::args().nth(1).expect("output directory"));
    let test_seed: u64 = std::env::args()
        .nth(2)
        .map(|s| s.parse().unwrap())
        .unwrap_or(987654321);
    fs::create_dir_all(&out).unwrap();
    for (name, kind, trained, dims) in [
        ("prototype", "choice", false, 32),
        ("probe", "choice", true, 32),
        ("score", "score", true, 32),
        ("logistic", "noul", true, 32),
        ("similarity", "noul", false, 32),
        ("wide", "choice", true, 384),
    ] {
        let mut seed = 47u64;
        let mut centers = Vec::new();
        for c in 0..3 {
            let mut row: Vec<f32> = (0..dims).map(|_| next(&mut seed) * 0.15).collect();
            row[c] += 1.0;
            l2_normalize(&mut row);
            centers.push(row);
        }
        let mut negative = vec![0.0; dims];
        negative[3] = 1.0;
        let q = match kind {
            "choice" => Question::Choice {
                instructions: String::new(),
                criteria: (0..3)
                    .map(|c| {
                        (
                            format!("class{c}"),
                            Criterion::Structured {
                                what: format!("prototype{c}"),
                                not_for: if c == 1 {
                                    None
                                } else {
                                    Some("negative".into())
                                },
                                examples: vec![],
                            },
                        )
                    })
                    .collect(),
            },
            "score" => Question::Score {
                instructions: String::new(),
                legend: (0..3).map(|c| format!("prototype{c}")).collect(),
            },
            _ => Question::Noul {
                instructions: "prototype0".into(),
            },
        };
        let mut engine = Engine::with_options(
            Numeric {
                dims,
                prototypes: centers.clone(),
                negative,
            },
            EngineOptions {
                logit_scale: 5.0,
                ..Default::default()
            },
        );
        if trained {
            let samples: Vec<_> = (0..180)
                .map(|i| {
                    let c = i % 3;
                    let v: Vec<f32> = centers[c]
                        .iter()
                        .map(|x| x + next(&mut seed) * 0.10)
                        .collect();
                    let label = match kind {
                        "choice" => format!("class{c}"),
                        "score" => format!("prototype{c}"),
                        _ => if c == 0 { "yes" } else { "no" }.into(),
                    };
                    LabeledExample {
                        text: text(&v),
                        label,
                    }
                })
                .collect();
            engine.train("decision", &samples).unwrap();
        }
        let snapshot = engine.export_embedded("decision", &q).unwrap();
        fs::write(
            out.join(format!("{name}.json")),
            serde_json::to_string_pretty(&snapshot).unwrap(),
        )
        .unwrap();
        let mut rows = Vec::new();
        // Separate RNG stream; none of these rows enter training or calibration.
        let mut heldout = test_seed;
        for i in 0..1000 {
            let mut v: Vec<f32> = if i < 750 {
                centers[i % 3]
                    .iter()
                    .map(|x| x + next(&mut heldout) * 0.12)
                    .collect()
            } else {
                (0..dims).map(|_| next(&mut heldout)).collect()
            };
            l2_normalize(&mut v);
            let req = DecisionRequest {
                state: text(&v),
                questions: BTreeMap::from([("decision".into(), q.clone())]),
            };
            let answer = engine
                .decide(&req)
                .unwrap()
                .answers
                .remove("decision")
                .unwrap();
            rows.push(json!({"features": v, "answer": answer}));
        }
        fs::write(
            out.join(format!("{name}.vectors.json")),
            serde_json::to_string(&rows).unwrap(),
        )
        .unwrap();
    }
}
