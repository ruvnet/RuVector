//! End-to-end CLI tests for `ruvllm-bridge` (iter 125, ADR-173 seam).
//!
//! Mirrors iter-118's mmwave-bridge pattern: spawn the actual built
//! binary, drive it through stdin/stdout JSONL, assert on the
//! response shape and the cluster-post behaviour against a fakeworker.

use std::io::Write;
use std::process::{Command, Stdio};

mod common;
use common::{free_port, spawn_fakeworker};

const BRIDGE: &str = env!("CARGO_BIN_EXE_ruvllm-bridge");

#[test]
fn ruvllm_bridge_single_request_returns_vector_response() {
    let port = free_port();
    let mut worker = spawn_fakeworker(port, 4, "fp:llm-cli");

    let mut child = Command::new(BRIDGE)
        .args([
            "--workers",
            &format!("127.0.0.1:{}", port),
            "--dim",
            "4",
            "--fingerprint",
            "fp:llm-cli",
            "--quiet",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn ruvllm-bridge");

    {
        let stdin = child.stdin.as_mut().expect("stdin");
        stdin
            .write_all(b"{\"text\":\"hello world from ruvllm\"}\n")
            .unwrap();
    }
    drop(child.stdin.take()); // close stdin → bridge exits clean
    let out = child.wait_with_output().expect("wait bridge");
    let _ = worker.kill();
    let _ = worker.wait();

    assert!(
        out.status.success(),
        "expected exit 0, got {:?}",
        out.status
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout.trim();
    assert!(line.contains("\"dim\":4"), "missing dim: {}", line);
    assert!(line.contains("\"vector\":["), "missing vector: {}", line);
    assert!(
        line.contains("\"latency_us\":"),
        "missing latency: {}",
        line
    );
}

#[test]
fn ruvllm_bridge_multi_line_with_request_id_propagates() {
    let port = free_port();
    let mut worker = spawn_fakeworker(port, 4, "fp:llm-cli");

    let mut child = Command::new(BRIDGE)
        .args([
            "--workers",
            &format!("127.0.0.1:{}", port),
            "--dim",
            "4",
            "--fingerprint",
            "fp:llm-cli",
            "--quiet",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn bridge");

    {
        let stdin = child.stdin.as_mut().unwrap();
        stdin.write_all(b"{\"text\":\"first request\"}\n").unwrap();
        stdin
            .write_all(b"{\"text\":\"second\",\"request_id\":\"01HRZK6S6NABCDEF12345678AB\"}\n")
            .unwrap();
        stdin.write_all(b"{\"text\":\"third no id\"}\n").unwrap();
    }
    drop(child.stdin.take());
    let out = child.wait_with_output().expect("wait bridge");
    let _ = worker.kill();
    let _ = worker.wait();

    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(
        lines.len(),
        3,
        "expected 3 response lines, got: {:?}",
        lines
    );
    // First and third have no request_id; middle does.
    assert!(!lines[0].contains("request_id"));
    assert!(
        lines[1].contains("\"request_id\":\"01HRZK6S6NABCDEF12345678AB\""),
        "request_id should be propagated: {}",
        lines[1]
    );
    assert!(!lines[2].contains("request_id"));
}

#[test]
fn ruvllm_bridge_blank_stdin_lines_are_ignored() {
    let port = free_port();
    let mut worker = spawn_fakeworker(port, 4, "fp:llm-cli");

    let mut child = Command::new(BRIDGE)
        .args([
            "--workers",
            &format!("127.0.0.1:{}", port),
            "--dim",
            "4",
            "--fingerprint",
            "fp:llm-cli",
            "--quiet",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn bridge");

    {
        let stdin = child.stdin.as_mut().unwrap();
        stdin.write_all(b"\n\n").unwrap(); // blank lines
        stdin.write_all(b"{\"text\":\"x\"}\n").unwrap();
        stdin.write_all(b"\n").unwrap();
    }
    drop(child.stdin.take());
    let out = child.wait_with_output().expect("wait bridge");
    let _ = worker.kill();
    let _ = worker.wait();

    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 1, "blanks should be ignored, got {:?}", lines);
}

#[test]
fn ruvllm_bridge_malformed_request_emits_error_line_continues() {
    let port = free_port();
    let mut worker = spawn_fakeworker(port, 4, "fp:llm-cli");

    let mut child = Command::new(BRIDGE)
        .args([
            "--workers",
            &format!("127.0.0.1:{}", port),
            "--dim",
            "4",
            "--fingerprint",
            "fp:llm-cli",
            "--quiet",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn bridge");

    {
        let stdin = child.stdin.as_mut().unwrap();
        // Missing `text` field.
        stdin.write_all(b"{\"not_text\":\"foo\"}\n").unwrap();
        // Valid request next — bridge must still process it.
        stdin.write_all(b"{\"text\":\"recovered\"}\n").unwrap();
    }
    drop(child.stdin.take());
    let out = child.wait_with_output().expect("wait bridge");
    let _ = worker.kill();
    let _ = worker.wait();

    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 2, "expected 1 error + 1 ok, got {:?}", lines);
    assert!(
        lines[0].contains("\"error\""),
        "first line should be error: {}",
        lines[0]
    );
    assert!(
        lines[1].contains("\"vector\":["),
        "second line should embed: {}",
        lines[1]
    );
}

#[test]
fn ruvllm_bridge_no_workers_flag_errors_immediately() {
    let out = Command::new(BRIDGE)
        .output()
        .expect("run bridge with no flags");
    assert!(!out.status.success(), "expected non-zero exit");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--workers"),
        "stderr should require --workers: {}",
        stderr
    );
}

#[test]
fn ruvllm_bridge_workers_without_fingerprint_refused() {
    let out = Command::new(BRIDGE)
        .args(["--workers", "127.0.0.1:1", "--dim", "4"])
        .output()
        .expect("run bridge");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("ADR-172 §2a") || stderr.contains("empty --fingerprint"),
        "stderr should reference §2a gate: {}",
        stderr
    );
}

#[test]
fn ruvllm_bridge_help_prints_synopsis() {
    let out = Command::new(BRIDGE).arg("--help").output().expect("--help");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("JSONL"));
    assert!(stdout.contains("--workers"));
    assert!(stdout.contains("--fingerprint"));
    // Iter 252 — lock that iter-238/243/245 flags stay documented.
    // A future refactor that drops one of these from the help text
    // (e.g. someone forgets to update print_help when adding the
    // next bridge knob) should fail loud.
    assert!(stdout.contains("--cache"));
    assert!(stdout.contains("--cache-ttl"));
    assert!(stdout.contains("--health-check"));
}

/// Iter 252 — `--cache N` without fingerprint must be refused by
/// the ADR-172 §2a gate, mirroring the embed.rs / bench.rs gates.
/// Lock the wire-up so a future regression (e.g. cache flag added
/// but gate skipped) fails CI.
#[test]
fn ruvllm_bridge_cache_without_fingerprint_refused() {
    // No --fingerprint AND no --allow-empty-fingerprint ⇒ gate fires
    // on either the workers-empty-fp gate or the cache-empty-fp gate
    // (whichever is checked first; both reference §2a in the message).
    let out = Command::new(BRIDGE)
        .args([
            "--workers",
            "127.0.0.1:1",
            "--dim",
            "4",
            "--cache",
            "1024",
        ])
        .output()
        .expect("run bridge");
    assert!(
        !out.status.success(),
        "bridge must refuse --cache without --fingerprint"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("§2a") || stderr.contains("empty --fingerprint"),
        "stderr should reference the §2a cache+fp gate: {}",
        stderr
    );
}

/// Iter 252 — `--cache N` paired with `--fingerprint` must succeed
/// (gate satisfied). Bridge produces correct response shape.
#[test]
fn ruvllm_bridge_cache_with_fingerprint_accepted() {
    let port = free_port();
    let mut worker = spawn_fakeworker(port, 4, "fp:cache-test");

    let mut child = Command::new(BRIDGE)
        .args([
            "--workers",
            &format!("127.0.0.1:{}", port),
            "--dim",
            "4",
            "--fingerprint",
            "fp:cache-test",
            "--cache",
            "256",
            "--cache-ttl",
            "60",
            "--quiet",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn ruvllm-bridge with --cache + --cache-ttl");

    {
        let stdin = child.stdin.as_mut().expect("stdin");
        // Same text twice → second is a cache hit but the bridge's
        // JSONL contract is opaque to that; both must produce a
        // vector response with dim=4 + the same vector contents.
        stdin
            .write_all(b"{\"text\":\"cached query\"}\n")
            .unwrap();
        stdin
            .write_all(b"{\"text\":\"cached query\"}\n")
            .unwrap();
    }
    drop(child.stdin.take());
    let out = child.wait_with_output().expect("wait bridge");
    let _ = worker.kill();
    let _ = worker.wait();

    assert!(
        out.status.success(),
        "expected exit 0 with --cache + --cache-ttl + --fingerprint, got {:?}\nstderr: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = stdout.lines().filter(|l| !l.is_empty()).collect();
    assert_eq!(lines.len(), 2, "expected 2 response lines, got {:?}", lines);
    for l in &lines {
        assert!(l.contains("\"dim\":4"), "missing dim=4 in: {}", l);
        assert!(l.contains("\"vector\":["), "missing vector in: {}", l);
    }
}

#[test]
fn ruvllm_bridge_version_prints_pkg_name_and_version() {
    let out = Command::new(BRIDGE)
        .arg("--version")
        .output()
        .expect("--version");
    assert!(out.status.success());
    let line = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let parts: Vec<&str> = line.split_whitespace().collect();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0], "ruvector-hailo-cluster");
}
