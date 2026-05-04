//! End-to-end integration tests for `ruvector-hailo-cluster-bench`.
//!
//! Mirrors `embed_cli.rs` (iter 70) and `stats_cli.rs` (iter 71). Spawns
//! the real built binary, drives it via `Command`, asserts on stdout /
//! exit code / Prom artifact.

use std::process::Command;
use std::time::{Duration, Instant};

mod common;
use common::{free_port, spawn_fakeworker};

const BENCH: &str = env!("CARGO_BIN_EXE_ruvector-hailo-cluster-bench");

#[test]
fn bench_cli_default_stdout_includes_percentile_block() {
    let port = free_port();
    let mut worker = spawn_fakeworker(port, 4, "");

    let out = Command::new(BENCH)
        .args([
            "--workers",
            &format!("127.0.0.1:{}", port),
            "--concurrency",
            "2",
            "--duration-secs",
            "1",
            "--dim",
            "4",
        ])
        .output()
        .expect("run bench");

    let _ = worker.kill();
    let _ = worker.wait();

    assert!(
        out.status.success(),
        "bench exited {:?}, stderr: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    // Verify the human-readable summary lines we promise.
    for needle in &[
        "wall_seconds",
        "requests_total",
        "requests_ok",
        "throughput_per_s",
        "p50",
        "p99",
    ] {
        assert!(
            stdout.contains(needle),
            "stdout missing {:?}, got: {}",
            needle,
            stdout
        );
    }
}

#[test]
fn bench_cli_quiet_silences_stdout_but_still_runs() {
    let port = free_port();
    let mut worker = spawn_fakeworker(port, 4, "");

    let out = Command::new(BENCH)
        .args([
            "--workers",
            &format!("127.0.0.1:{}", port),
            "--concurrency",
            "2",
            "--duration-secs",
            "1",
            "--dim",
            "4",
            "--quiet",
        ])
        .output()
        .expect("run bench");

    let _ = worker.kill();
    let _ = worker.wait();

    assert!(out.status.success(), "bench exited {:?}", out.status);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(
        stdout.trim(),
        "",
        "--quiet should produce empty stdout, got: {:?}",
        stdout
    );
}

#[test]
fn bench_cli_prom_file_contains_throughput_metric() {
    let port = free_port();
    let mut worker = spawn_fakeworker(port, 4, "");
    let prom_path = std::env::temp_dir().join(format!("bench-cli-iter72-{}.prom", port));
    let prom_str = prom_path.to_string_lossy().to_string();

    let out = Command::new(BENCH)
        .args([
            "--workers",
            &format!("127.0.0.1:{}", port),
            "--concurrency",
            "2",
            "--duration-secs",
            "1",
            "--dim",
            "4",
            "--quiet",
            "--prom",
            &prom_str,
        ])
        .output()
        .expect("run bench");

    let _ = worker.kill();
    let _ = worker.wait();

    assert!(out.status.success());
    let prom_body = std::fs::read_to_string(&prom_path).expect("prom file should exist");
    let _ = std::fs::remove_file(&prom_path);

    // Required textfile-collector preamble + the throughput metric.
    assert!(
        prom_body.contains("# HELP ruvector_hailo_bench_throughput_per_second"),
        "missing HELP, got: {}",
        prom_body
    );
    assert!(
        prom_body.contains("ruvector_hailo_bench_throughput_per_second{concurrency=\"2\"}"),
        "missing throughput metric with concurrency label, got: {}",
        prom_body
    );
}

#[test]
fn bench_cli_validate_fleet_with_wrong_fingerprint_exits_two() {
    let port = free_port();
    let mut worker = spawn_fakeworker(port, 4, "fp:current");

    let out = Command::new(BENCH)
        .args([
            "--workers",
            &format!("127.0.0.1:{}", port),
            "--concurrency",
            "1",
            "--duration-secs",
            "1",
            "--dim",
            "4",
            "--fingerprint",
            "fp:not-the-fakeworker",
            "--validate-fleet",
            "--quiet",
        ])
        .output()
        .expect("run bench");

    let _ = worker.kill();
    let _ = worker.wait();

    assert_eq!(
        out.status.code(),
        Some(2),
        "validate_fleet on drifted fp should exit 2, got {:?}",
        out.status
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("FAILED"),
        "stderr should explain the failure, got: {}",
        stderr
    );
}

#[test]
fn bench_cli_version_flag_prints_pkg_name_and_version() {
    for arg in &["--version", "-V"] {
        let out = Command::new(BENCH).arg(arg).output().expect("run bench");
        assert!(out.status.success());
        let line = String::from_utf8_lossy(&out.stdout).trim().to_string();
        assert!(line.starts_with("ruvector-hailo-cluster"), "got: {}", line);
        assert_eq!(line.split_whitespace().count(), 2);
    }
}

#[test]
fn bench_cli_duration_secs_actually_bounds_runtime() {
    // 1-second bench should complete in roughly 1s — bound by 2s to
    // leave slack for spin-up + CI noise. Catches regressions where
    // the stop signal isn't honored.
    let port = free_port();
    let mut worker = spawn_fakeworker(port, 4, "");

    let start = Instant::now();
    let out = Command::new(BENCH)
        .args([
            "--workers",
            &format!("127.0.0.1:{}", port),
            "--concurrency",
            "1",
            "--duration-secs",
            "1",
            "--dim",
            "4",
            "--quiet",
        ])
        .output()
        .expect("run bench");
    let elapsed = start.elapsed();

    let _ = worker.kill();
    let _ = worker.wait();

    assert!(out.status.success());
    assert!(
        elapsed >= Duration::from_millis(900),
        "bench should run for ~1s, got {:?}",
        elapsed
    );
    assert!(
        elapsed < Duration::from_secs(3),
        "bench should NOT exceed 3s for --duration-secs 1, got {:?}",
        elapsed
    );
}
