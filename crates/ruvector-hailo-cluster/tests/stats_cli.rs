//! End-to-end integration tests for the `ruvector-hailo-stats` binary.
//!
//! Mirrors `embed_cli.rs` (iter 70) — spawn the real binary, drive it
//! via `std::process::Command`, assert on stdout / exit code / stderr.
//! Catches CLI-level regressions when refactoring `src/bin/stats.rs`.

use std::process::Command;

mod common;
use common::{free_port, spawn_fakeworker};

const STATS: &str = env!("CARGO_BIN_EXE_ruvector-hailo-stats");

#[test]
fn stats_cli_list_workers_does_not_require_live_workers() {
    // --list-workers short-circuits before any RPC, so it works against
    // arbitrary addresses with no actual server. Verifies the discovery
    // → print path doesn't accidentally regress to needing live workers.
    let out = Command::new(STATS)
        .args([
            "--workers",
            "10.255.255.1:50051,10.255.255.2:50051",
            "--list-workers",
        ])
        .output()
        .expect("run stats");
    assert!(
        out.status.success(),
        "stats --list-workers exited {:?}",
        out.status
    );

    let stdout = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 3, "header + 2 workers, got: {}", stdout);
    assert!(lines[0].starts_with("worker\taddress"));
    assert!(lines[1].contains("10.255.255.1:50051"));
    assert!(lines[2].contains("10.255.255.2:50051"));
}

#[test]
fn stats_cli_default_tsv_against_live_worker() {
    let port = free_port();
    let mut worker = spawn_fakeworker(port, 384, "fp:test");

    let out = Command::new(STATS)
        .args(["--workers", &format!("127.0.0.1:{}", port)])
        .output()
        .expect("run stats");

    let _ = worker.kill();
    let _ = worker.wait();

    assert!(
        out.status.success(),
        "stats exited {:?}, stderr: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    // Header + 1 worker row.
    assert_eq!(lines.len(), 2, "expected header+1 row, got: {}", stdout);
    assert!(
        lines[0].starts_with("worker\taddress\tfingerprint"),
        "unexpected header: {}",
        lines[0]
    );
    assert!(
        lines[1].contains("fp:test"),
        "fingerprint should appear in row: {}",
        lines[1]
    );
}

#[test]
fn stats_cli_json_output_includes_fingerprint_field() {
    let port = free_port();
    let mut worker = spawn_fakeworker(port, 384, "fp:json-test");

    let out = Command::new(STATS)
        .args(["--workers", &format!("127.0.0.1:{}", port), "--json"])
        .output()
        .expect("run stats");

    let _ = worker.kill();
    let _ = worker.wait();

    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout.trim();
    assert!(
        line.contains("\"fingerprint\":\"fp:json-test\""),
        "JSON should include fingerprint, got: {}",
        line
    );
    assert!(line.contains("\"stats\":"), "JSON should include stats");
}

#[test]
fn stats_cli_tsv_includes_rate_limit_columns() {
    // Iter-105 (ADR-172 §3b follow-up): rl_denials + rl_peers must
    // surface in the default TSV. Fakeworker reports 0 for both
    // (limiter not exercised), but the columns are present.
    let port = free_port();
    let mut worker = spawn_fakeworker(port, 4, "fp:rl");

    let out = Command::new(STATS)
        .args(["--workers", &format!("127.0.0.1:{}", port)])
        .output()
        .expect("run stats");

    let _ = worker.kill();
    let _ = worker.wait();

    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    assert!(
        lines[0].contains("rl_denials"),
        "header should include rl_denials: {}",
        lines[0]
    );
    assert!(
        lines[0].contains("rl_peers"),
        "header should include rl_peers: {}",
        lines[0]
    );
    let cols: Vec<&str> = lines[1].split('\t').collect();
    assert_eq!(
        cols.len(),
        12,
        "expected 12 columns in TSV row, got {}: {:?}",
        cols.len(),
        cols
    );
    // Last two columns are u64 — must parse cleanly.
    assert!(
        cols[10].parse::<u64>().is_ok(),
        "rl_denials should be u64: {:?}",
        cols[10]
    );
    assert!(
        cols[11].parse::<u64>().is_ok(),
        "rl_peers should be u64: {:?}",
        cols[11]
    );
}

#[test]
fn stats_cli_prom_output_includes_rate_limit_metrics() {
    let port = free_port();
    let mut worker = spawn_fakeworker(port, 4, "fp:rl-prom");

    let out = Command::new(STATS)
        .args(["--workers", &format!("127.0.0.1:{}", port), "--prom"])
        .output()
        .expect("run stats");

    let _ = worker.kill();
    let _ = worker.wait();

    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("ruvector_rate_limit_denials_total"),
        "prom output missing denials metric: {}",
        stdout
    );
    assert!(
        stdout.contains("ruvector_rate_limit_tracked_peers"),
        "prom output missing tracked_peers metric: {}",
        stdout
    );
    // HELP/TYPE sections should also document them.
    assert!(stdout.contains("# HELP ruvector_rate_limit_denials_total"));
    assert!(stdout.contains("# TYPE ruvector_rate_limit_tracked_peers gauge"));
}

#[test]
fn stats_cli_strict_homogeneous_with_drift_exits_three() {
    // Two workers, different fingerprints — drift detected.
    // --strict-homogeneous turns drift into exit 3.
    let port_a = free_port();
    let port_b = free_port();
    let mut wa = spawn_fakeworker(port_a, 384, "fp:current");
    let mut wb = spawn_fakeworker(port_b, 384, "fp:stale");

    let out = Command::new(STATS)
        .args([
            "--workers",
            &format!("127.0.0.1:{},127.0.0.1:{}", port_a, port_b),
            "--strict-homogeneous",
        ])
        .output()
        .expect("run stats");

    let _ = wa.kill();
    let _ = wa.wait();
    let _ = wb.kill();
    let _ = wb.wait();

    assert_eq!(
        out.status.code(),
        Some(3),
        "drift + --strict-homogeneous should exit 3, got {:?}",
        out.status
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("DRIFT"),
        "stderr should mention DRIFT, got: {}",
        stderr
    );
}

#[test]
fn stats_cli_version_flag_prints_pkg_name_and_version() {
    for arg in &["--version", "-V"] {
        let out = Command::new(STATS).arg(arg).output().expect("run stats");
        assert!(out.status.success());
        let line = String::from_utf8_lossy(&out.stdout).trim().to_string();
        assert!(line.starts_with("ruvector-hailo-cluster"), "got: {}", line);
        assert_eq!(line.split_whitespace().count(), 2);
    }
}

#[test]
fn stats_cli_strict_homogeneous_with_no_drift_exits_zero() {
    // Same fingerprint on both workers → no drift → exit 0.
    let port_a = free_port();
    let port_b = free_port();
    let mut wa = spawn_fakeworker(port_a, 384, "fp:same");
    let mut wb = spawn_fakeworker(port_b, 384, "fp:same");

    let out = Command::new(STATS)
        .args([
            "--workers",
            &format!("127.0.0.1:{},127.0.0.1:{}", port_a, port_b),
            "--strict-homogeneous",
        ])
        .output()
        .expect("run stats");

    let _ = wa.kill();
    let _ = wa.wait();
    let _ = wb.kill();
    let _ = wb.wait();

    assert!(
        out.status.success(),
        "homogeneous fleet should exit 0, got {:?}",
        out.status
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("DRIFT"),
        "stderr must NOT mention DRIFT for homogeneous fleet, got: {}",
        stderr
    );
}

// ---- ADR-172 §1c iter-110 end-to-end CLI coverage ----

/// Build a deterministic ed25519 keypair for the test, format both
/// pubkey and (later) signature as the lowercase hex the CLI expects.
fn fixture_signing_key() -> ed25519_dalek::SigningKey {
    let seed: [u8; 32] = [
        0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
        0xa8, 0xb1, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6,
        0xc7, 0xc8,
    ];
    ed25519_dalek::SigningKey::from_bytes(&seed)
}

fn hex_lower(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        write!(&mut s, "{:02x}", b).unwrap();
    }
    s
}

/// Stage a temp dir with manifest + signature + pubkey files. Returns
/// the dir path so callers can drop it (TempDir-like, but stdlib only
/// to avoid a tempfile dev-dep). Caller is expected to clean up.
fn write_manifest_fixture(manifest_body: &str) -> std::path::PathBuf {
    use std::io::Write as _;
    let dir = std::env::temp_dir().join(format!(
        "ruvector-stats-mfsig-{}-{}",
        std::process::id(),
        // Distinguish parallel tests from each other.
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();

    // Write manifest
    let mut mf = std::fs::File::create(dir.join("workers.txt")).unwrap();
    mf.write_all(manifest_body.as_bytes()).unwrap();
    drop(mf);

    let sk = fixture_signing_key();
    let pk_hex = hex_lower(sk.verifying_key().as_bytes());
    use ed25519_dalek::Signer;
    let sig_hex = hex_lower(&sk.sign(manifest_body.as_bytes()).to_bytes());

    std::fs::write(dir.join("workers.sig"), sig_hex).unwrap();
    std::fs::write(dir.join("pubkey.hex"), pk_hex).unwrap();
    dir
}

#[test]
fn stats_cli_signed_workers_file_succeeds_with_matching_sig() {
    let port = free_port();
    let mut worker = spawn_fakeworker(port, 4, "fp:signed");

    let body = format!("pi-0 = 127.0.0.1:{}\n", port);
    let dir = write_manifest_fixture(&body);
    let manifest = dir.join("workers.txt");
    let sig = dir.join("workers.sig");
    let pk = dir.join("pubkey.hex");

    let out = Command::new(STATS)
        .args([
            "--workers-file",
            manifest.to_str().unwrap(),
            "--workers-file-sig",
            sig.to_str().unwrap(),
            "--workers-file-pubkey",
            pk.to_str().unwrap(),
        ])
        .output()
        .expect("run stats");

    let _ = worker.kill();
    let _ = worker.wait();
    let _ = std::fs::remove_dir_all(&dir);

    assert!(
        out.status.success(),
        "signed manifest should succeed, exit={:?} stderr={}",
        out.status,
        String::from_utf8_lossy(&out.stderr),
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("fp:signed"),
        "expected worker fp in TSV: {}",
        stdout
    );
}

#[test]
fn stats_cli_tampered_workers_file_fails_signature_check() {
    let port = free_port();
    let mut worker = spawn_fakeworker(port, 4, "fp:tamper");

    // Sign one body, then tamper before invoking stats.
    let original = format!("pi-0 = 127.0.0.1:{}\n", port);
    let dir = write_manifest_fixture(&original);
    let manifest = dir.join("workers.txt");
    // Overwrite manifest with a different body — sig no longer matches.
    let tampered = format!("pi-0 = 127.0.0.1:{}\npi-rogue = 10.0.0.99:50051\n", port);
    std::fs::write(&manifest, &tampered).unwrap();

    let sig = dir.join("workers.sig");
    let pk = dir.join("pubkey.hex");

    let out = Command::new(STATS)
        .args([
            "--workers-file",
            manifest.to_str().unwrap(),
            "--workers-file-sig",
            sig.to_str().unwrap(),
            "--workers-file-pubkey",
            pk.to_str().unwrap(),
        ])
        .output()
        .expect("run stats");

    let _ = worker.kill();
    let _ = worker.wait();
    let _ = std::fs::remove_dir_all(&dir);

    assert!(!out.status.success(), "tampered manifest must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("signature verification failed") || stderr.contains("manifest_sig"),
        "stderr should reference the verification failure: {}",
        stderr
    );
}

#[test]
fn stats_cli_partial_signature_config_is_refused() {
    // Only --workers-file-sig set, --workers-file-pubkey omitted.
    // Must fail before any RPC fires (the gate happens at flag-parse
    // time in the bin's discovery construction).
    let body = "pi-0 = 127.0.0.1:65535\n";
    let dir = write_manifest_fixture(body);
    let manifest = dir.join("workers.txt");
    let sig = dir.join("workers.sig");

    let out = Command::new(STATS)
        .args([
            "--workers-file",
            manifest.to_str().unwrap(),
            "--workers-file-sig",
            sig.to_str().unwrap(),
            // intentionally no --workers-file-pubkey
        ])
        .output()
        .expect("run stats");

    let _ = std::fs::remove_dir_all(&dir);

    assert!(!out.status.success(), "partial config must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("ADR-172 §1c") || stderr.contains("must both be set"),
        "stderr should reference the gate: {}",
        stderr
    );
}
