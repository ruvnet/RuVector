//! Train the actual RuVector engine on externally prepared, isolated sensor rows.
use ruvector_typesafe_core::{
    embedder::l2_normalize,
    engine::{Engine, EngineOptions, LabeledExample},
    *,
};
use serde::Deserialize;
use serde_json::json;
use std::{collections::BTreeMap, fs, path::PathBuf};

#[derive(Deserialize)]
struct Row {
    features: Vec<f32>,
    raw: Vec<f32>,
    label: usize,
    date: String,
}
#[derive(Deserialize)]
struct Input {
    train: Vec<Row>,
    #[serde(default)]
    calibration: Vec<Row>,
    validation: Vec<Row>,
    test: Vec<Row>,
    preprocessing: serde_json::Value,
    #[serde(default)]
    engine_options: Option<EngineOptions>,
}
struct Numeric {
    prototypes: Vec<Vec<f32>>,
}
impl Embedder for Numeric {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts
            .iter()
            .map(|s| {
                if let Some(c) = s.strip_prefix("prototype") {
                    return Ok(self.prototypes[c.parse::<usize>().unwrap()].clone());
                }
                let mut v: Vec<f32> = s.split(',').map(|x| x.parse().unwrap()).collect();
                assert_eq!(v.len(), self.dims());
                l2_normalize(&mut v);
                Ok(v)
            })
            .collect()
    }
    fn dims(&self) -> usize {
        self.prototypes[0].len()
    }
    fn id(&self) -> &str {
        "uci-occupancy-standardize-f32-v1"
    }
}
fn text(v: &[f32]) -> String {
    v.iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(",")
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let input: Input = serde_json::from_slice(&fs::read(&args[1]).unwrap()).unwrap();
    let out = PathBuf::from(&args[2]);
    fs::create_dir_all(&out).unwrap();
    let dims = input.train[0].features.len();
    let labels = ["empty", "occupied"];
    let mut prototypes = vec![vec![0.0; dims]; 2];
    for row in &input.train {
        let mut v = row.features.clone();
        l2_normalize(&mut v);
        for (p, x) in prototypes[row.label].iter_mut().zip(v) {
            *p += x;
        }
    }
    for p in &mut prototypes {
        l2_normalize(p);
    }
    let mut engine = Engine::with_options(
        Numeric { prototypes },
        input.engine_options.unwrap_or(EngineOptions {
            logit_scale: 5.0,
            ..Default::default()
        }),
    );
    let examples: Vec<_> = input
        .train
        .iter()
        .map(|r| LabeledExample {
            text: text(&r.features),
            label: labels[r.label].into(),
        })
        .collect();
    let training = engine.train("occupancy", &examples).unwrap();
    let question = Question::Choice {
        instructions: String::new(),
        criteria: labels
            .iter()
            .enumerate()
            .map(|(i, label)| {
                (
                    label.to_string(),
                    Criterion::Structured {
                        what: format!("prototype{i}"),
                        not_for: None,
                        examples: vec![],
                    },
                )
            })
            .collect(),
    };
    let mut snapshot =
        serde_json::to_value(engine.export_embedded("occupancy", &question).unwrap()).unwrap();
    snapshot["preprocessing"] = input.preprocessing;
    fs::write(
        out.join("snapshot.json"),
        serde_json::to_vec_pretty(&snapshot).unwrap(),
    )
    .unwrap();
    fs::write(
        out.join("training.json"),
        serde_json::to_vec_pretty(&training).unwrap(),
    )
    .unwrap();
    for (name, rows) in [
        ("calibration", input.calibration),
        ("validation", input.validation),
        ("test", input.test),
    ] {
        let output: Vec<_> = rows.iter().map(|row| {
            let answer = engine.decide(&DecisionRequest {
                state: text(&row.features), questions: BTreeMap::from([("occupancy".into(), question.clone())]),
            }).unwrap().answers.remove("occupancy").unwrap();
            json!({"features":row.features,"raw":row.raw,"label":row.label,"date":row.date,"answer":answer})
        }).collect();
        fs::write(
            out.join(format!("{name}.json")),
            serde_json::to_vec(&output).unwrap(),
        )
        .unwrap();
    }
    println!("{}", serde_json::to_string(&training).unwrap());
}
