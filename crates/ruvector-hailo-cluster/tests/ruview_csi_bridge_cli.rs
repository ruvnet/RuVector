//! End-to-end CLI tests for `ruview-csi-bridge` (iter 125 —
//! production-readiness coverage for iter-123's bridge).
//!
//! Mirrors the iter-118 pattern for `mmwave-bridge`: spawn the actual
//! built binary via `std::process::Command`, drive it with synthesized
//! ADR-018 UDP packets, assert on stdout/stderr/exit-code.

use std::net::UdpSocket;
use std::process::{Command, Stdio};
use std::time::Duration;

mod common;
use common::{free_port, spawn_fakeworker};

const BRIDGE: &str = env!("CARGO_BIN_EXE_ruview-csi-bridge");

/// Synthesize an ADR-018 v6 (feature-state) CSI packet header.
/// Returns 20-byte header + zeroed I/Q payload of the requested size.
fn synth_csi_v6_packet(node_id: u8, channel: u8, rssi: i8, n_subc: u16) -> Vec<u8> {
    let mut p = Vec::with_capacity(20 + n_subc as usize * 2 * 2);
    p.extend_from_slice(&0xC511_0006u32.to_le_bytes());
    p.push(node_id);
    p.push(2); // n_antennas
    p.extend_from_slice(&n_subc.to_le_bytes());
    p.push(channel);
    p.push(rssi as u8);
    p.push(0xA6); // noise_floor = -90 dBm as i8
    p.extend_from_slice(&[0u8; 5]); // reserved bytes 11..16
    p.extend_from_slice(&12345u32.to_le_bytes()); // timestamp_us at 16..20
                                                  // I/Q payload (zeros are fine for header-parse coverage)
    p.extend_from_slice(&vec![0u8; n_subc as usize * 2 * 2]);
    p
}

#[test]
fn ruview_bridge_emits_jsonl_for_synthetic_csi_packet() {
    let udp_port = free_port();
    let mut child = Command::new(BRIDGE)
        .args(["--listen", &format!("127.0.0.1:{}", udp_port), "--quiet"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn bridge");

    // Give it a moment to bind the UDP port.
    std::thread::sleep(Duration::from_millis(200));

    // Fire 4 synthetic packets at the bridge.
    let sender = UdpSocket::bind("127.0.0.1:0").expect("ephemeral sender");
    let target = format!("127.0.0.1:{}", udp_port);
    let pkt = synth_csi_v6_packet(7, 6, -42, 64);
    for _ in 0..4 {
        sender.send_to(&pkt, &target).expect("send");
    }

    std::thread::sleep(Duration::from_millis(400));
    let _ = child.kill();
    let out = child.wait_with_output().expect("wait bridge");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    assert!(
        lines.len() >= 3,
        "expected ≥ 3 JSONL lines, got {}: {:?}",
        lines.len(),
        stdout
    );
    for line in &lines {
        assert!(
            line.contains("\"kind\":\"csi_feature_state\""),
            "wrong kind: {}",
            line
        );
        assert!(line.contains("\"node_id\":7"), "missing node_id: {}", line);
        assert!(line.contains("\"channel\":6"), "missing channel: {}", line);
        assert!(line.contains("\"rssi_dbm\":-42"), "missing rssi: {}", line);
    }
}

#[test]
fn ruview_bridge_posts_to_cluster_when_workers_set() {
    let cluster_port = free_port();
    let mut worker = spawn_fakeworker(cluster_port, 4, "fp:csi-cluster");

    let udp_port = free_port();
    let mut child = Command::new(BRIDGE)
        .args([
            "--listen",
            &format!("127.0.0.1:{}", udp_port),
            "--workers",
            &format!("127.0.0.1:{}", cluster_port),
            "--dim",
            "4",
            "--fingerprint",
            "fp:csi-cluster",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn bridge");

    std::thread::sleep(Duration::from_millis(250));

    let sender = UdpSocket::bind("127.0.0.1:0").unwrap();
    let target = format!("127.0.0.1:{}", udp_port);
    let pkt = synth_csi_v6_packet(3, 11, -55, 32);
    for _ in 0..3 {
        sender.send_to(&pkt, &target).expect("send");
    }

    std::thread::sleep(Duration::from_millis(700));
    let _ = child.kill();
    let out = child.wait_with_output().expect("wait bridge");
    let _ = worker.kill();
    let _ = worker.wait();

    let stderr = String::from_utf8_lossy(&out.stderr);
    let post_count = stderr.matches("posted text=").count();
    assert!(
        post_count >= 2,
        "expected ≥ 2 cluster posts, saw {}: {}",
        post_count,
        stderr
    );
    assert!(
        !stderr.contains("cluster post failed"),
        "no posts should fail: {}",
        stderr
    );
}

#[test]
fn ruview_bridge_rejects_workers_without_fingerprint() {
    let out = Command::new(BRIDGE)
        .args([
            "--listen",
            "127.0.0.1:1",
            "--workers",
            "127.0.0.1:1",
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
        "stderr should reference §2a gate: {}",
        stderr
    );
}

#[test]
fn ruview_bridge_drops_malformed_packets_silently() {
    // Bridge must not crash on garbage UDP. Send 3 random packets,
    // then 1 valid one; assert exactly 1 JSONL line on stdout.
    let udp_port = free_port();
    let mut child = Command::new(BRIDGE)
        .args(["--listen", &format!("127.0.0.1:{}", udp_port), "--quiet"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn bridge");
    std::thread::sleep(Duration::from_millis(200));

    let sender = UdpSocket::bind("127.0.0.1:0").unwrap();
    let target = format!("127.0.0.1:{}", udp_port);
    sender.send_to(&[0xAA, 0xBB], &target).unwrap(); // too short
    sender.send_to(&[0u8; 32], &target).unwrap(); // wrong magic, padded
    sender
        .send_to(b"hello world this is not a csi frame", &target)
        .unwrap();
    sender
        .send_to(&synth_csi_v6_packet(1, 1, -50, 16), &target)
        .unwrap();

    std::thread::sleep(Duration::from_millis(400));
    let _ = child.kill();
    let out = child.wait_with_output().expect("wait bridge");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(
        lines.len(),
        1,
        "expected exactly 1 JSONL line (1 valid + 3 rejected), got: {:?}",
        stdout
    );
    assert!(lines[0].contains("\"node_id\":1"));
}

#[test]
fn ruview_bridge_help_prints_synopsis() {
    let out = Command::new(BRIDGE)
        .arg("--help")
        .output()
        .expect("run bridge --help");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("--listen"));
    assert!(stdout.contains("--workers"));
    assert!(stdout.contains("ADR-018"));
}

#[test]
fn ruview_bridge_version_prints_pkg_name_and_version() {
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
