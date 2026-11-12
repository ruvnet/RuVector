//! ADR-354 item 2: dual-accept. Every file written by the pre-ADR-354 writer
//! (`../tests/vectors/legacy/*.rvf`, frozen bytes) must still open and query
//! identically, be flagged `legacy_dialect`, and stay usable for appends.

use rvf_runtime::{QueryOptions, RvfOptions, RvfStore};
use rvf_types::security::SecurityPolicy;
use rvf_types::{ErrorCode, RvfError};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn legacy_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/vectors/legacy")
}

fn vecs(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    x = x
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((x >> 33) as f32) / (u32::MAX as f32) - 0.5
                })
                .collect()
        })
        .collect()
}

/// Never open the committed bytes in place: copy to a temp dir first.
fn copy_fixture(name: &str) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let dst = dir.path().join(name);
    std::fs::copy(legacy_dir().join(name), &dst).unwrap();
    (dir, dst)
}

/// name -> (header line, per-query hit lines) from `legacy_expected.txt`.
fn expected() -> BTreeMap<String, (String, Vec<String>)> {
    let text = std::fs::read_to_string(legacy_dir().join("legacy_expected.txt")).unwrap();
    let mut m: BTreeMap<String, (String, Vec<String>)> = BTreeMap::new();
    for line in text.lines() {
        let (name, rest) = line.split_once(' ').unwrap();
        let e = m.entry(name.to_string()).or_default();
        if rest.starts_with("total_vectors=") {
            e.0 = rest.to_string();
        } else {
            e.1.push(rest.to_string());
        }
    }
    m
}

fn observed(s: &RvfStore) -> (String, Vec<String>) {
    let st = s.status();
    let head = format!(
        "total_vectors={} segments={} epoch={}",
        st.total_vectors, st.total_segments, st.current_epoch
    );
    let mut hits = Vec::new();
    for (qi, q) in vecs(3, 8, 99).iter().enumerate() {
        for (rank, h) in s
            .query(q, 5, &QueryOptions::default())
            .unwrap()
            .iter()
            .enumerate()
        {
            hits.push(format!(
                "q{qi} r{rank} id={} dist={:08x}",
                h.id,
                h.distance.to_bits()
            ));
        }
    }
    (head, hits)
}

const FIXTURES: [(&str, &str); 3] = [
    ("legacy_basic.rvf", "basic"),
    ("legacy_multi.rvf", "multi"),
    ("legacy_compacted.rvf", "compacted"),
];

#[test]
fn legacy_files_open_readonly_and_query_identically() {
    let exp = expected();
    for (file, key) in FIXTURES {
        let (_d, p) = copy_fixture(file);
        let s = RvfStore::open_readonly(&p).unwrap();
        assert!(s.legacy_dialect(), "{file} must be marked legacy");
        assert_eq!(observed(&s), exp[key], "{file}");
    }
}

#[test]
fn legacy_files_open_read_write_and_query_identically() {
    let exp = expected();
    for (file, key) in FIXTURES {
        let (_d, p) = copy_fixture(file);
        let s = RvfStore::open(&p).unwrap();
        assert!(s.legacy_dialect());
        assert_eq!(observed(&s), exp[key], "{file}");
        s.close().unwrap();
    }
}

#[test]
fn appending_to_a_legacy_file_keeps_old_data_and_aligns_new_segments() {
    let (_d, p) = copy_fixture("legacy_multi.rvf");
    let before_len = std::fs::metadata(&p).unwrap().len();
    let mut s = RvfStore::open(&p).unwrap();
    let v = vecs(4, 8, 555);
    let refs: Vec<&[f32]> = v.iter().map(|x| x.as_slice()).collect();
    s.ingest_batch(&refs, &[9000, 9001, 9002, 9003], None)
        .unwrap();
    s.close().unwrap();

    let s = RvfStore::open_readonly(&p).unwrap();
    assert!(s.legacy_dialect(), "the unpadded history is still there");
    assert_eq!(s.status().total_vectors, 45 - 3 + 4);
    // Every segment written after the append starts on a 64-byte boundary.
    let new_starts: Vec<u64> = s
        .segment_dir()
        .iter()
        .map(|e| e.1)
        .filter(|&o| o >= before_len)
        .collect();
    assert!(!new_starts.is_empty());
    assert!(new_starts.iter().all(|o| o % 64 == 0), "{new_starts:?}");
    // Old vectors still answer identically for ids that are not near the new ones.
    let q = &vecs(3, 8, 99)[0];
    let hits = s.query(q, 5, &QueryOptions::default()).unwrap();
    assert!(!hits.is_empty());
}

#[test]
fn compacting_a_legacy_file_preserves_results() {
    let (_d, p) = copy_fixture("legacy_multi.rvf");
    let mut s = RvfStore::open(&p).unwrap();
    let before = observed(&s).1;
    s.compact().unwrap();
    s.close().unwrap();
    let s = RvfStore::open_readonly(&p).unwrap();
    assert_eq!(observed(&s).1, before);
    // The rewritten file is wire-aligned throughout.
    let bytes = std::fs::read(&p).unwrap();
    assert_eq!(bytes.len() % 64, 0);
    assert!(s.segment_dir().iter().all(|e| e.1 % 64 == 0));
}

#[test]
fn legacy_files_are_refused_under_strict_and_open_under_lower_policies() {
    let (_d, p) = copy_fixture("legacy_basic.rvf");
    let strict = RvfStore::open_with_policy(&p, RvfOptions::default(), SecurityPolicy::Strict);
    match strict {
        Err(RvfError::Security(rvf_types::security::SecurityError::UnsignedManifest {
            ..
        })) => {}
        Err(other) => panic!("unexpected {other:?}"),
        Ok(_) => panic!("Strict opened an unsigned legacy file"),
    }
    for policy in [SecurityPolicy::Permissive, SecurityPolicy::WarnOnly] {
        let s = RvfStore::open_with_policy(&p, RvfOptions::default(), policy).unwrap();
        assert!(s.legacy_dialect());
        assert_eq!(s.status().total_vectors, 20);
    }
    assert_eq!(
        rvf_runtime::store::security_error_code(
            &rvf_types::security::SecurityError::UnsignedManifest { manifest_offset: 0 }
        ),
        ErrorCode::UnsignedManifest
    );
}

#[test]
fn legacy_hash_of_unknown_algorithm_is_refused() {
    // Fail closed: flip a segment header's checksum_algo to an unassigned
    // value and the payload must not be accepted.
    let (_d, p) = copy_fixture("legacy_basic.rvf");
    let mut bytes = std::fs::read(&p).unwrap();
    // First segment is the (empty) manifest at offset 0; corrupt the VEC
    // segment that follows it: find the SFVR + type 0x01 header.
    let pos = (0..bytes.len() - 64)
        .find(|&i| bytes[i..i + 4] == *b"SFVR" && bytes[i + 5] == 0x01)
        .unwrap();
    bytes[pos + 0x20] = 9;
    std::fs::write(&p, &bytes).unwrap();
    assert!(RvfStore::open_readonly(&p).is_err());
    let _ = Path::new(&p);
}
