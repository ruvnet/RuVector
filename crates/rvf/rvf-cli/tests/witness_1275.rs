//! Regression tests for #1275: `rvf verify-witness` on runtime-written stores.
use std::path::Path;
use std::process::{Command, Output};

fn rvf(args: &[&str], cwd: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rvf"))
        .args(args)
        .current_dir(cwd)
        .output()
        .unwrap()
}

fn text(o: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    )
}

fn write_vectors(dir: &Path, base: usize, n: usize) {
    let recs: Vec<String> = (0..n)
        .map(|i| format!("{{\"id\":{},\"vector\":[{},1,2,3]}}", base + i + 1, i))
        .collect();
    std::fs::write(dir.join("v.json"), format!("[{}]", recs.join(","))).unwrap();
}

/// A store with several witness entries over several sessions (reopen).
fn store() -> tempfile::TempDir {
    let d = tempfile::tempdir().unwrap();
    assert!(rvf(&["create", "-d", "4", "s.rvf"], d.path())
        .status
        .success());
    for round in 0..3 {
        write_vectors(d.path(), round * 10, 5);
        assert!(rvf(&["ingest", "-i", "v.json", "s.rvf"], d.path())
            .status
            .success());
    }
    d
}

/// Offsets of every witness segment payload in the file.
fn witness_payload_offsets(bytes: &[u8]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos + 64 <= bytes.len() && bytes[pos..pos + 4] == 0x5256_4653u32.to_le_bytes() {
        let plen = u64::from_le_bytes(bytes[pos + 0x10..pos + 0x18].try_into().unwrap()) as usize;
        if bytes[pos + 5] == 0x0A {
            out.push((pos + 64, plen));
        }
        pos = (pos + 64 + plen + 63) & !63;
    }
    out
}

#[test]
fn runtime_written_witness_chain_verifies_across_sessions() {
    let d = store();
    let o = rvf(&["verify-witness", "s.rvf"], d.path());
    assert!(o.status.success(), "{}", text(&o));
    let t = text(&o);
    assert!(t.contains("verified successfully"), "{t}");
    assert!(t.contains("Total entries:") && t.contains('3'), "{t}");
    let bytes = std::fs::read(d.path().join("s.rvf")).unwrap();
    for (_, len) in witness_payload_offsets(&bytes) {
        assert_eq!(len, 73, "witness payload is one fixed 73-byte chain entry");
    }
}

#[test]
fn chain_survives_delete_and_compact() {
    let d = store();
    let o = rvf(&["delete", "s.rvf", "--ids", "1,2"], d.path());
    assert!(o.status.success(), "{}", text(&o));
    // Absolute path: `rvf compact s.rvf` (bare file name) fails to fsync the
    // parent directory "" -- a separate pre-existing bug, not part of #1275.
    let abs = d.path().join("s.rvf");
    let o = rvf(&["compact", abs.to_str().unwrap()], d.path());
    assert!(o.status.success(), "{}", text(&o));
    let o = rvf(&["verify-witness", "s.rvf"], d.path());
    assert!(o.status.success(), "{}", text(&o));
}

/// Flip one byte in the middle witness entry's action hash (segment hash
/// recomputed by the attacker is not needed: either layer must catch it).
#[test]
fn tampered_witness_is_detected_and_exits_nonzero() {
    for which in 0..3 {
        let d = store();
        let p = d.path().join("s.rvf");
        let mut bytes = std::fs::read(&p).unwrap();
        let offs = witness_payload_offsets(&bytes);
        assert_eq!(offs.len(), 3);
        let (start, _) = offs[which];
        bytes[start + 40] ^= 0x01; // inside action_hash
        std::fs::write(&p, &bytes).unwrap();
        let o = rvf(&["verify-witness", "s.rvf"], d.path());
        assert!(
            !o.status.success(),
            "entry {which} tamper not detected: {}",
            text(&o)
        );
        assert_eq!(o.status.code(), Some(1), "{}", text(&o));
    }
}

/// An attacker who ALSO fixes the segment content hash is still caught by the
/// hash chain, except for the newest entry (documented limit).
#[test]
fn chain_catches_edit_with_repaired_segment_hash() {
    let d = store();
    let p = d.path().join("s.rvf");
    let mut bytes = std::fs::read(&p).unwrap();
    let offs = witness_payload_offsets(&bytes);
    let (start, len) = offs[0];
    bytes[start + 40] ^= 0x01;
    // Recompute the xxh3-128 content hash the header carries (algo 1).
    let hdr = start - 64;
    assert_eq!(bytes[hdr + 0x20], 1, "xxh3 content hash expected");
    let h = rvf_wire::hash::compute_content_hash(1, &bytes[start..start + len]);
    bytes[hdr + 0x28..hdr + 0x38].copy_from_slice(&h);
    std::fs::write(&p, &bytes).unwrap();
    let o = rvf(&["verify-witness", "s.rvf"], d.path());
    assert_eq!(o.status.code(), Some(1), "{}", text(&o));
    assert!(text(&o).contains("chain broken"), "{}", text(&o));
}

#[test]
fn removing_a_middle_entry_breaks_the_chain() {
    let d = store();
    let p = d.path().join("s.rvf");
    let mut bytes = std::fs::read(&p).unwrap();
    let offs = witness_payload_offsets(&bytes);
    // Retype the middle witness segment to padding (type 0x00 is unknown).
    bytes[offs[1].0 - 64 + 5] = 0x7F;
    std::fs::write(&p, &bytes).unwrap();
    let o = rvf(&["verify-witness", "s.rvf"], d.path());
    assert_eq!(o.status.code(), Some(1), "{}", text(&o));
}

/// A published rvf-runtime 0.4.0 store: 70-byte variable record. Reported as
/// legacy / not chain-verifiable, never as valid.
#[test]
fn legacy_70_byte_records_are_reported_not_valid() {
    let d = store();
    let p = d.path().join("s.rvf");
    let mut bytes = std::fs::read(&p).unwrap();
    // Rewrite every witness segment's payload into the legacy layout
    // (73 -> 70 bytes: the header length shrinks, content hash is recomputed).
    let offs = witness_payload_offsets(&bytes);
    for (start, _) in offs {
        let action = b"ingest:count=1000,epoch=2"; // 25 bytes -> 70 total
        let mut legacy = vec![0x02u8];
        legacy.extend_from_slice(&1u64.to_le_bytes());
        legacy.extend_from_slice(&(action.len() as u32).to_le_bytes());
        legacy.extend_from_slice(action);
        legacy.extend_from_slice(&[0u8; 32]);
        assert_eq!(legacy.len(), 70);
        let hdr = start - 64;
        bytes[hdr + 0x10..hdr + 0x18].copy_from_slice(&70u64.to_le_bytes());
        bytes[start..start + 70].copy_from_slice(&legacy);
        bytes[start + 70..start + 73].fill(0);
        let h = rvf_wire::hash::compute_content_hash(1, &legacy);
        bytes[hdr + 0x28..hdr + 0x38].copy_from_slice(&h);
    }
    std::fs::write(&p, &bytes).unwrap();

    let o = rvf(&["verify-witness", "s.rvf"], d.path());
    let t = text(&o);
    assert_eq!(o.status.code(), Some(2), "{t}");
    assert!(t.contains("legacy") || t.contains("LEGACY"), "{t}");
    assert!(!t.contains("verified successfully"), "{t}");
    let o = rvf(&["verify-witness", "--json", "s.rvf"], d.path());
    assert!(text(&o).contains("legacy_unverifiable"), "{}", text(&o));
    let o = rvf(&["verify-witness", "--allow-legacy", "s.rvf"], d.path());
    assert!(o.status.success(), "{}", text(&o));
}
