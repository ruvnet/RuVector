use ruvector_language::{parser, types::*};
use ruvector_language_native::training::{Candidate, EvaluationReport};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::{Command, Output as ProcessOutput},
};

fn run(binary: &str, directory: &Path, args: &[&str]) -> ProcessOutput {
    Command::new(binary)
        .current_dir(directory)
        .args(args)
        .output()
        .unwrap()
}
fn success(output: ProcessOutput) -> ProcessOutput {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}
fn write<T: serde::Serialize>(path: PathBuf, value: &T) {
    fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
}
struct Directory(PathBuf);
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn process_train_promote_evaluate_run_replay_and_no_overwrite() {
    let directory =
        Directory(std::env::temp_dir().join(format!("rvg-native-cli-{}", std::process::id())));
    fs::create_dir(&directory.0).unwrap();
    fs::write(
        directory.0.join("training.json"),
        include_str!("../examples/training.json"),
    )
    .unwrap();
    fs::write(
        directory.0.join("evaluation.json"),
        include_str!("../examples/evaluation.json"),
    )
    .unwrap();
    let model_cli = env!("CARGO_BIN_EXE_rvg-model");
    let rvg = env!("CARGO_BIN_EXE_rvg");
    success(run(
        model_cli,
        &directory.0,
        &["train", "training.json", "candidate.json"],
    ));
    let candidate: Candidate =
        serde_json::from_slice(&fs::read(directory.0.join("candidate.json")).unwrap()).unwrap();
    assert_eq!(candidate.report.losses.len(), 20);
    assert_eq!(candidate.report.evaluation_count, 1);
    success(run(
        model_cli,
        &directory.0,
        &["promote", "candidate.json", "0", "models.json"],
    ));
    let original = fs::read(directory.0.join("models.json")).unwrap();
    assert!(!run(
        model_cli,
        &directory.0,
        &["promote", "candidate.json", "0", "models.json"]
    )
    .status
    .success());
    assert_eq!(original, fs::read(directory.0.join("models.json")).unwrap());
    success(run(
        model_cli,
        &directory.0,
        &[
            "evaluate",
            "models.json",
            "incident_v1",
            "evaluation.json",
            "evaluation_report.json",
        ],
    ));
    let evaluated: EvaluationReport =
        serde_json::from_slice(&fs::read(directory.0.join("evaluation_report.json")).unwrap())
            .unwrap();
    assert_eq!(candidate.report.trained, evaluated.metrics);
    let program=parser::parse("program prediction { effect graph.read; effect model.predict; effect decision.emit; budget 100000; let s = nodes \"Service\"; let p = predict s \"Cause\" model \"incident_v1\" top 3; emit p; }").unwrap();
    let request = RunRequest {
        program,
        snapshot: Snapshot {
            version: 1,
            revision: "toy".into(),
            nodes: vec![Node {
                id: "api".into(),
                label: "Service".into(),
                properties: BTreeMap::new(),
                vector: None,
            }],
            edges: vec![],
            observations: vec![],
            rules: vec![],
        },
        context: Context {
            now: 100,
            capabilities: [
                Effect::GraphRead,
                Effect::ModelPredict,
                Effect::DecisionEmit,
            ]
            .into_iter()
            .collect(),
            max_steps: 100000,
            max_results: 100,
        },
    };
    write(directory.0.join("request.json"), &request);
    let output = success(run(
        rvg,
        &directory.0,
        &[
            "run",
            "--request",
            "request.json",
            "--models",
            "models.json",
        ],
    ));
    let receipt: Receipt = serde_json::from_slice(&output.stdout).unwrap();
    match &receipt.output {
        Output::Predictions(batch) => assert_eq!(batch.candidates.len(), 3),
        _ => panic!("prediction expected"),
    }
    write(directory.0.join("receipt.json"), &receipt);
    success(run(
        rvg,
        &directory.0,
        &[
            "replay",
            "--request",
            "request.json",
            "--receipt",
            "receipt.json",
            "--models",
            "models.json",
        ],
    ));
    assert!(
        !run(rvg, &directory.0, &["run", "--request", "request.json"])
            .status
            .success()
    );
}
