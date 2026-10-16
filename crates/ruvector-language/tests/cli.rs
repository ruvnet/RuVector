use ruvector_language::{cli::run_cli, Error, Predictor, Result};
use std::io::Write;
use std::process::{Command, Output, Stdio};

fn forbidden_loader(_: &str) -> Result<Box<dyn Predictor>> {
    panic!("invalid CLI usage must not load models")
}
fn args(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| value.to_string()).collect()
}
fn process(args: &[&str], stdin: &[u8]) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_rvg-core"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(stdin).unwrap();
    child.wait_with_output().unwrap()
}
fn process_error(output: &Output) -> Error {
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    serde_json::from_slice(&output.stderr).unwrap()
}

#[test]
fn replay_required_flags_are_checked_before_io_or_model_loading() {
    for input in [
        vec![
            "replay",
            "--request",
            "/missing/request",
            "--models",
            "/missing/model",
        ],
        vec!["replay", "--receipt", "/missing/receipt"],
    ] {
        let error = run_cli(&args(&input), forbidden_loader).unwrap_err();
        assert_eq!(error.code, "USAGE");
    }
}

#[test]
fn ambiguous_cli_inputs_are_rejected_before_io() {
    for input in [
        vec!["run", "--request", "-", "--models", "-"],
        vec!["run", "--request", "/missing", "--context", "/missing"],
        vec!["run", "--request", "/missing", "--request", "/missing"],
        vec!["run", "--receipt", "/missing", "--request", "/missing"],
        vec!["check", "/missing", "--unknown", "x"],
    ] {
        assert_eq!(
            run_cli(&args(&input), forbidden_loader).unwrap_err().code,
            "USAGE"
        );
    }
}

#[test]
fn real_process_emits_structured_errors_and_rejects_invalid_utf8() {
    let output = process(&["check", "-"], &[0xff]);
    assert_eq!(process_error(&output).code, "INVALID_INPUT");
    let output = process(
        &["run", "--request", "-"],
        br#"{"program":{},"program":{}}"#,
    );
    assert!(process_error(&output).message.contains("duplicate"));
}

#[test]
fn real_process_bounds_stdin_before_parsing() {
    let oversized = vec![b' '; 16 * 1024 * 1024 + 1];
    let output = process(&["check", "-"], &oversized);
    assert_eq!(process_error(&output).code, "INPUT_LIMIT");
}

#[test]
fn real_process_compiles_the_bundled_source_without_install_side_effects() {
    let source = include_str!("../examples/incident.rvg");
    let output = process(&["compile", "-"], source.as_bytes());
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    let program: ruvector_language::Program = serde_json::from_slice(&output.stdout).unwrap();
    ruvector_language::compiler::check(&program).unwrap();
}
