//! `ruvllm serve` exits non-zero when the model fails to load; placeholder
//! (mock) responses need an explicit `--allow-mock` and are labeled.
//!
//! No network: every model is a local `.gguf` file that fails to parse or
//! is rejected by architecture, so the backend never falls back to the Hub.

use assert_cmd::Command;
use predicates::prelude::*;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

/// A GGUF v3 file with no tensors whose only metadata is
/// `general.architecture = arch`.
fn gguf_with_arch(dir: &Path, arch: &str) -> PathBuf {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes()); // version
    bytes.extend_from_slice(&0u64.to_le_bytes()); // tensor count
    bytes.extend_from_slice(&1u64.to_le_bytes()); // metadata entries
    let key = b"general.architecture";
    bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
    bytes.extend_from_slice(key);
    bytes.extend_from_slice(&8u32.to_le_bytes()); // value type: string
    bytes.extend_from_slice(&(arch.len() as u64).to_le_bytes());
    bytes.extend_from_slice(arch.as_bytes());
    let path = dir.join(format!("{arch}.gguf"));
    std::fs::write(&path, bytes).unwrap();
    path
}

fn garbage_gguf(dir: &Path) -> PathBuf {
    let path = dir.join("bad.gguf");
    std::fs::write(&path, b"this is not a gguf file").unwrap();
    path
}

fn serve_args(cache: &Path, model: &Path, port: u16) -> Vec<String> {
    vec![
        "--no-color".into(),
        "--cache-dir".into(),
        cache.display().to_string(),
        "serve".into(),
        model.display().to_string(),
        "--host".into(),
        "127.0.0.1".into(),
        "--port".into(),
        port.to_string(),
    ]
}

/// `ruvllm serve <model>` with a clean environment.
fn serve(cache: &Path, model: &Path) -> Command {
    let mut cmd = Command::cargo_bin("ruvllm").unwrap();
    cmd.env_remove("RUVLLM_STRICT")
        .env_remove("RUVLLM_ALLOW_MOCK")
        .env_remove("RUVLLM_CACHE_DIR")
        .args(serve_args(cache, model, 0))
        .timeout(Duration::from_secs(120));
    cmd
}

#[test]
fn unloadable_model_exits_nonzero_by_default() {
    let dir = tempfile::tempdir().unwrap();
    let model = garbage_gguf(dir.path());
    serve(dir.path(), &model)
        .assert()
        .code(1)
        .stdout(predicate::str::contains("Server ready").not())
        .stderr(predicate::str::contains("failed to load"))
        .stderr(predicate::str::contains(model.display().to_string()))
        .stderr(predicate::str::contains("--allow-mock"));
}

#[test]
fn unsupported_architecture_is_named_and_exits() {
    let dir = tempfile::tempdir().unwrap();
    for arch in ["phi2", "qwen35"] {
        let model = gguf_with_arch(dir.path(), arch);
        serve(dir.path(), &model)
            .assert()
            .code(1)
            .stderr(predicate::str::contains(format!(
                "GGUF architecture '{arch}' is not supported"
            )));
    }
}

#[test]
fn ruvllm_strict_zero_no_longer_enables_mock_mode() {
    let dir = tempfile::tempdir().unwrap();
    let model = garbage_gguf(dir.path());
    serve(dir.path(), &model)
        .env("RUVLLM_STRICT", "0")
        .assert()
        .code(1)
        .stdout(predicate::str::contains("Server ready").not())
        .stderr(predicate::str::contains(
            "RUVLLM_STRICT=0 no longer enables mock mode",
        ));
}

#[test]
fn strict_conflicts_with_allow_mock() {
    let dir = tempfile::tempdir().unwrap();
    let model = garbage_gguf(dir.path());
    serve(dir.path(), &model)
        .args(["--strict", "--mock"])
        .assert()
        .code(1)
        .stderr(predicate::str::contains(
            "--allow-mock conflicts with --strict",
        ));
    serve(dir.path(), &model)
        .env("RUVLLM_STRICT", "1")
        .env("RUVLLM_ALLOW_MOCK", "1")
        .assert()
        .code(1)
        .stderr(predicate::str::contains(
            "--allow-mock conflicts with --strict",
        ));
}

/// Kills the server when the test ends, pass or fail.
struct Server(std::process::Child);

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn http(port: u16, request: &str) -> String {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .unwrap();
    stream.write_all(request.as_bytes()).unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).unwrap();
    response
}

#[test]
fn allow_mock_serves_labeled_placeholder_responses() {
    let dir = tempfile::tempdir().unwrap();
    let model = garbage_gguf(dir.path());
    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();

    let child = std::process::Command::new(assert_cmd::cargo::cargo_bin("ruvllm"))
        .env_remove("RUVLLM_STRICT")
        .env_remove("RUVLLM_ALLOW_MOCK")
        .env_remove("RUVLLM_CACHE_DIR")
        .args(serve_args(dir.path(), &model, port))
        .arg("--allow-mock")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut server = Server(child);

    let stdout = BufReader::new(server.0.stdout.take().unwrap());
    let mut banner = String::new();
    for line in stdout.lines() {
        let line = line.unwrap();
        banner.push_str(&line);
        banner.push('\n');
        if line.contains("Press Ctrl+C") {
            break;
        }
    }
    assert!(banner.contains("Running in MOCK MODE"), "{banner}");
    assert!(
        banner.contains("Server ready (MOCK MODE: placeholder responses, not model output)"),
        "{banner}"
    );

    let body = r#"{"model":"m","messages":[{"role":"user","content":"Hello"}]}"#;
    let response = http(
        port,
        &format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n\
             Content-Type: application/json\r\nContent-Length: {}\r\n\
             Connection: close\r\n\r\n{}",
            body.len(),
            body
        ),
    );
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    assert!(response.contains("x-ruvllm-mode: mock"), "{response}");
    assert!(
        response.contains(r#""system_fingerprint":"ruvllm-mock""#),
        "{response}"
    );
    assert!(response.contains("[ruvllm mock mode]"), "{response}");
}

/// `ruvllm chat <model>` with a clean environment and no input (EOF).
fn chat(cache: &Path, model: &Path) -> Command {
    let mut cmd = Command::cargo_bin("ruvllm").unwrap();
    // chat saves its readline history under the user cache dir: keep it in `cache`.
    cmd.env_remove("RUVLLM_ALLOW_MOCK")
        .env_remove("RUVLLM_CACHE_DIR")
        .env("HOME", cache)
        .env("XDG_CACHE_HOME", cache)
        .args(["--no-color", "--cache-dir"])
        .arg(cache)
        .arg("chat")
        .arg(model)
        .write_stdin("")
        .timeout(Duration::from_secs(120));
    cmd
}

#[test]
fn chat_exits_on_a_failed_load_unless_mock_is_allowed() {
    let dir = tempfile::tempdir().unwrap();
    let model = garbage_gguf(dir.path());
    chat(dir.path(), &model)
        .assert()
        .failure()
        .stdout(predicate::str::contains("Type your message").not())
        .stderr(predicate::str::contains("failed to load"))
        .stderr(predicate::str::contains("--allow-mock"));
    chat(dir.path(), &model)
        .arg("--allow-mock")
        .assert()
        .success()
        .stdout(predicate::str::contains("MOCK MODE"));
}
