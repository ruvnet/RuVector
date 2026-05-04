//! End-to-end integration test that spawns the actual built binaries.
//!
//! Pre-iter-70, refactors of `src/bin/embed.rs` (lots of them) had no
//! CLI-level safety net — only the library code was tested. This test
//! spawns the real `ruvector-hailo-fakeworker` and `ruvector-hailo-embed`
//! binaries via `std::process::Command`, drives one embedding through
//! `--text`, parses the stdout JSON, and asserts the contract.
//!
//! Cargo provides the `CARGO_BIN_EXE_<name>` env var for each `[[bin]]`
//! defined in Cargo.toml — no path probing needed.
//!
//! `cargo test` builds binaries before running tests, so these env vars
//! always point at fresh artifacts.

use std::io::{Read as _, Write as _};
use std::process::{Command, Stdio};

mod common;
use common::{free_port, spawn_fakeworker};

const EMBED: &str = env!("CARGO_BIN_EXE_ruvector-hailo-embed");

#[test]
fn embed_cli_text_flag_emits_json_line() {
    let port = free_port();
    let mut worker = spawn_fakeworker(port, 4, "");

    let out = Command::new(EMBED)
        .args([
            "--workers",
            &format!("127.0.0.1:{}", port),
            "--dim",
            "4",
            "--text",
            "hello",
            "--quiet",
        ])
        .output()
        .expect("run embed");

    let _ = worker.kill();
    let _ = worker.wait();

    assert!(
        out.status.success(),
        "embed exited {:?}, stderr: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout.trim();
    // Verify the JSON shape we promise downstream tools.
    assert!(
        line.starts_with("{") && line.ends_with("}"),
        "expected JSON line, got {:?}",
        line
    );
    assert!(
        line.contains("\"text\":\"hello\""),
        "missing text field: {:?}",
        line
    );
    assert!(line.contains("\"dim\":4"), "missing dim field: {:?}", line);
    assert!(
        line.contains("\"vec_head\":["),
        "missing vec_head field (default --output mode): {:?}",
        line
    );
}

#[test]
fn embed_cli_output_full_emits_full_vector() {
    let port = free_port();
    let mut worker = spawn_fakeworker(port, 8, "");

    let out = Command::new(EMBED)
        .args([
            "--workers",
            &format!("127.0.0.1:{}", port),
            "--dim",
            "8",
            "--text",
            "world",
            "--output",
            "full",
            "--quiet",
        ])
        .output()
        .expect("run embed");

    let _ = worker.kill();
    let _ = worker.wait();

    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout.trim();
    assert!(
        line.contains("\"vector\":["),
        "expected full vector field, got {:?}",
        line
    );
    // Count commas inside the vector array — should have 7 separators
    // for 8 floats (dim=8).
    let vec_segment = line.split("\"vector\":[").nth(1).expect("vector field");
    let vec_segment = vec_segment.split(']').next().unwrap();
    let comma_count = vec_segment.matches(',').count();
    assert_eq!(
        comma_count, 7,
        "expected 7 commas for dim=8, got {} in {:?}",
        comma_count, vec_segment
    );
}

#[test]
fn embed_cli_repeated_text_flags_embed_in_order() {
    let port = free_port();
    let mut worker = spawn_fakeworker(port, 4, "");

    let out = Command::new(EMBED)
        .args([
            "--workers",
            &format!("127.0.0.1:{}", port),
            "--dim",
            "4",
            "--text",
            "first",
            "--text",
            "second",
            "--text",
            "third",
            "--quiet",
        ])
        .output()
        .expect("run embed");

    let _ = worker.kill();
    let _ = worker.wait();
    assert!(out.status.success());

    let stdout = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 3, "expected 3 JSON lines, got: {}", stdout);
    assert!(lines[0].contains("\"text\":\"first\""));
    assert!(lines[1].contains("\"text\":\"second\""));
    assert!(lines[2].contains("\"text\":\"third\""));
}

#[test]
fn embed_cli_stdin_pipe_works_when_no_text_flag() {
    let port = free_port();
    let mut worker = spawn_fakeworker(port, 4, "");

    let mut child = Command::new(EMBED)
        .args([
            "--workers",
            &format!("127.0.0.1:{}", port),
            "--dim",
            "4",
            "--quiet",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn embed");

    {
        let stdin = child.stdin.as_mut().expect("stdin");
        stdin.write_all(b"alpha\nbeta\n").unwrap();
        // Drop stdin → EOF → embed exits its read loop.
    }
    child.stdin.take();

    let mut stdout = String::new();
    child
        .stdout
        .as_mut()
        .unwrap()
        .read_to_string(&mut stdout)
        .unwrap();
    let status = child.wait().expect("embed wait");

    let _ = worker.kill();
    let _ = worker.wait();

    assert!(status.success(), "embed exited {:?}", status);
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 2);
    assert!(lines[0].contains("\"text\":\"alpha\""));
    assert!(lines[1].contains("\"text\":\"beta\""));
}

#[test]
fn embed_cli_version_flag_prints_pkg_name_and_version() {
    // Standard CLI convention: `--version` and `-V` both print and exit 0.
    // Output format `<pkg-name> <semver>` so shell scripts can parse with
    // `awk '{print $2}'`.
    for arg in &["--version", "-V"] {
        let out = Command::new(EMBED).arg(arg).output().expect("run embed");
        assert!(
            out.status.success(),
            "embed {} exited {:?}",
            arg,
            out.status
        );
        let stdout = String::from_utf8_lossy(&out.stdout);
        let line = stdout.trim();
        assert!(
            line.starts_with("ruvector-hailo-cluster"),
            "expected pkg-name prefix, got: {:?}",
            line
        );
        let parts: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(
            parts.len(),
            2,
            "expected `<name> <version>`, got: {:?}",
            line
        );
        // Version should be a semver-ish string (digits + dots).
        assert!(
            parts[1]
                .chars()
                .all(|c| c.is_ascii_digit() || c == '.' || c == '-' || c.is_ascii_alphabetic()),
            "version field looks malformed: {:?}",
            parts[1]
        );
    }
}

#[test]
fn embed_cli_cache_with_empty_fingerprint_refuses_without_opt_in() {
    // ADR-172 §2a iter-101 gate: --cache > 0 with no fingerprint and
    // no --allow-empty-fingerprint must fail loud, before any RPC fires.
    let port = free_port();
    let mut worker = spawn_fakeworker(port, 4, "");

    let out = Command::new(EMBED)
        .args([
            "--workers",
            &format!("127.0.0.1:{}", port),
            "--dim",
            "4",
            "--cache",
            "16",
            "--text",
            "hello",
            "--quiet",
        ])
        .output()
        .expect("run embed");

    let _ = worker.kill();
    let _ = worker.wait();

    assert!(!out.status.success(), "expected non-zero exit");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("ADR-172 §2a") || stderr.contains("empty fingerprint"),
        "stderr should reference the ADR-172 §2a gate, got: {}",
        stderr
    );
    // No vector should have been emitted on stdout.
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.trim().is_empty(),
        "stdout should be empty, got: {}",
        stdout
    );
}

#[test]
fn embed_cli_cache_with_explicit_opt_in_runs() {
    // The escape hatch: --allow-empty-fingerprint lets legacy fleets
    // keep their old behavior (cache enabled, fingerprint empty).
    let port = free_port();
    let mut worker = spawn_fakeworker(port, 4, "");

    let out = Command::new(EMBED)
        .args([
            "--workers",
            &format!("127.0.0.1:{}", port),
            "--dim",
            "4",
            "--cache",
            "16",
            "--allow-empty-fingerprint",
            "--text",
            "hello",
            "--quiet",
        ])
        .output()
        .expect("run embed");

    let _ = worker.kill();
    let _ = worker.wait();

    assert!(
        out.status.success(),
        "expected success with --allow-empty-fingerprint, got {:?}, stderr: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr),
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("\"text\":\"hello\""),
        "expected JSON line, got: {}",
        stdout
    );
}

#[test]
fn embed_cli_cache_with_fingerprint_passes_gate() {
    // The intended path: pass --fingerprint <hex> and the gate is happy.
    let port = free_port();
    let mut worker = spawn_fakeworker(port, 4, "fp:test");

    let out = Command::new(EMBED)
        .args([
            "--workers",
            &format!("127.0.0.1:{}", port),
            "--dim",
            "4",
            "--cache",
            "16",
            "--fingerprint",
            "fp:test",
            "--text",
            "hello",
            "--quiet",
        ])
        .output()
        .expect("run embed");

    let _ = worker.kill();
    let _ = worker.wait();

    assert!(
        out.status.success(),
        "expected success with --fingerprint set, got {:?}, stderr: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr),
    );
}

#[test]
fn embed_cli_validate_fleet_with_wrong_fingerprint_exits_nonzero() {
    let port = free_port();
    let mut worker = spawn_fakeworker(port, 4, "");

    let out = Command::new(EMBED)
        .args([
            "--workers",
            &format!("127.0.0.1:{}", port),
            "--dim",
            "4",
            "--fingerprint",
            "fp:not-the-fakeworker-fp",
            "--validate-fleet",
            "--validate-only",
            "--quiet",
        ])
        .output()
        .expect("run embed");

    let _ = worker.kill();
    let _ = worker.wait();

    // validate_fleet detects 0 healthy workers → exit 2.
    assert!(!out.status.success(), "expected non-zero exit");
    assert_eq!(out.status.code(), Some(2), "expected exit 2");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("FAILED"),
        "stderr should explain why, got: {}",
        stderr
    );
}
