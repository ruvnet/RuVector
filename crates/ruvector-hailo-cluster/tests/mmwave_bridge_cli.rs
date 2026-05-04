//! End-to-end CLI tests for the `ruvector-mmwave-bridge` binary
//! (iter 118 — production-readiness pass).
//!
//! Verifies that the bridge actually composes with the cluster the way
//! the manual live-test in iter 116 demonstrated, but committed and
//! re-runnable in CI. Three cases:
//!
//!   1. `--simulator` mode without `--workers` produces the expected
//!      cycle of JSONL events on stdout.
//!   2. `--simulator --workers` posts decoded events to a fakeworker
//!      via the embed RPC; assert successful posts on stderr.
//!   3. `--workers` without `--fingerprint` is refused (ADR-172 §2a
//!      gate parity with embed/bench).

use std::process::{Command, Stdio};
use std::time::Duration;

mod common;
use common::{free_port, spawn_fakeworker};

const BRIDGE: &str = env!("CARGO_BIN_EXE_ruvector-mmwave-bridge");

#[test]
fn bridge_simulator_emits_cycle_of_jsonl_events() {
    // 5 Hz × 1.5s = 7-8 events. Cycle is breathing → heart_rate →
    // distance → presence; assert at least one of each kind in the
    // window so a future state-machine bug that drops a frame type
    // surfaces.
    let mut child = Command::new(BRIDGE)
        .args(["--simulator", "--rate", "10", "--quiet"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn bridge");

    std::thread::sleep(Duration::from_millis(700));
    let _ = child.kill();
    let out = child.wait_with_output().expect("wait bridge");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let kinds: std::collections::HashSet<&str> = stdout
        .lines()
        .filter_map(|l| {
            // Crude but sufficient: extract the "kind":"X" value.
            l.split("\"kind\":\"")
                .nth(1)
                .and_then(|s| s.split('"').next())
        })
        .collect();
    assert!(
        kinds.contains("breathing"),
        "no breathing event in {:?}",
        kinds
    );
    assert!(
        kinds.contains("heart_rate"),
        "no heart_rate event in {:?}",
        kinds
    );
    assert!(
        kinds.contains("distance"),
        "no distance event in {:?}",
        kinds
    );
    assert!(
        kinds.contains("presence"),
        "no presence event in {:?}",
        kinds
    );
}

#[test]
fn bridge_simulator_with_workers_posts_to_cluster() {
    let port = free_port();
    let mut worker = spawn_fakeworker(port, 4, "fp:bridge-test");

    let mut child = Command::new(BRIDGE)
        .args([
            "--simulator",
            "--rate",
            "10",
            "--workers",
            &format!("127.0.0.1:{}", port),
            "--dim",
            "4",
            "--fingerprint",
            "fp:bridge-test",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn bridge with cluster sink");

    // Let it pump for a moment — needs to dial the worker, send a few
    // RPCs, see results come back.
    std::thread::sleep(Duration::from_millis(900));
    let _ = child.kill();
    let out = child.wait_with_output().expect("wait bridge");
    let _ = worker.kill();
    let _ = worker.wait();

    let stderr = String::from_utf8_lossy(&out.stderr);
    let post_count = stderr.matches("posted text=").count();
    assert!(
        post_count >= 3,
        "expected ≥ 3 cluster posts in window, saw {}: {}",
        post_count,
        stderr
    );
    // None of them should have failed — fakeworker is local, latency
    // budget is generous.
    assert!(
        !stderr.contains("cluster post failed"),
        "saw post failures: {}",
        stderr
    );
}

#[test]
fn bridge_workers_without_fingerprint_refused_by_default() {
    // ADR-172 §2a parity: --workers + empty --fingerprint must fail
    // before any RPC is attempted, just like embed/bench.
    let out = Command::new(BRIDGE)
        .args([
            "--simulator",
            "--workers",
            "127.0.0.1:1", // never dialed; gate fires first
            "--dim",
            "4",
            // intentionally no --fingerprint
        ])
        .output()
        .expect("run bridge");

    assert!(!out.status.success(), "expected non-zero exit");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("ADR-172 §2a") || stderr.contains("empty --fingerprint"),
        "stderr should reference the §2a gate, got: {}",
        stderr
    );
}

#[test]
fn bridge_workers_without_fingerprint_succeeds_with_opt_in() {
    let port = free_port();
    let mut worker = spawn_fakeworker(port, 4, ""); // fakeworker default fp

    let mut child = Command::new(BRIDGE)
        .args([
            "--simulator",
            "--rate",
            "5",
            "--workers",
            &format!("127.0.0.1:{}", port),
            "--dim",
            "4",
            "--allow-empty-fingerprint",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn bridge");

    std::thread::sleep(Duration::from_millis(900));
    let _ = child.kill();
    let out = child.wait_with_output().expect("wait bridge");
    let _ = worker.kill();
    let _ = worker.wait();

    let stderr = String::from_utf8_lossy(&out.stderr);
    let post_count = stderr.matches("posted text=").count();
    assert!(
        post_count >= 1,
        "with --allow-empty-fingerprint, expected ≥ 1 post, saw {}: {}",
        post_count,
        stderr
    );
}

#[test]
fn bridge_no_mode_flag_errors_cleanly() {
    let out = Command::new(BRIDGE)
        .output()
        .expect("run bridge with no args");
    assert!(!out.status.success(), "expected non-zero exit");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--device") || stderr.contains("--simulator") || stderr.contains("--auto"),
        "error should name the missing mode flags, got: {}",
        stderr
    );
}

#[test]
fn bridge_help_prints_synopsis() {
    let out = Command::new(BRIDGE)
        .arg("--help")
        .output()
        .expect("run bridge --help");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("--simulator"));
    assert!(stdout.contains("--workers"));
    assert!(stdout.contains("--fingerprint"));
    // Iter 253 — lock that iter-242/243/245 flags stay in --help.
    assert!(stdout.contains("--cache"));
    assert!(stdout.contains("--cache-ttl"));
    assert!(stdout.contains("--health-check"));
}

/// Iter 253 — `--cache N` without fingerprint must be refused per
/// the ADR-172 §2a gate, mirroring the iter-252 ruvllm-bridge gate
/// test and iter-253's csi-bridge gate test.
#[test]
fn bridge_cache_without_fingerprint_refused() {
    let out = Command::new(BRIDGE)
        .args([
            "--simulator",
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

#[test]
fn bridge_version_prints_pkg_name_and_version() {
    let out = Command::new(BRIDGE)
        .arg("--version")
        .output()
        .expect("run bridge --version");
    assert!(out.status.success());
    let line = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let parts: Vec<&str> = line.split_whitespace().collect();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0], "ruvector-hailo-cluster");
}
