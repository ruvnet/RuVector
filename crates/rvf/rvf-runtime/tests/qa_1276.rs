//! Regression tests for issue #1276 (items 1-4): truncated stores (strict open),
//! unsupported segment versions, non-finite query vectors and rejected-only
//! ingests.

use rvf_runtime::{QueryOptions, RvfOptions, RvfStore};
use rvf_types::{ErrorCode, RvfError};
use std::path::Path;

const DIM: u16 = 4;

fn opts() -> RvfOptions {
    RvfOptions {
        dimension: DIM,
        ..Default::default()
    }
}

/// Build a store with `gens` ingests of `per` vectors each; returns the path.
fn build(dir: &Path, gens: u64, per: u64) -> std::path::PathBuf {
    let path = dir.join("s.rvf");
    let mut s = RvfStore::create(&path, opts()).unwrap();
    for g in 0..gens {
        let vs: Vec<Vec<f32>> = (0..per)
            .map(|i| vec![(g * per + i) as f32, 1.0, 2.0, 3.0])
            .collect();
        let refs: Vec<&[f32]> = vs.iter().map(|v| v.as_slice()).collect();
        let ids: Vec<u64> = (0..per).map(|i| g * per + i + 1).collect();
        let r = s.ingest_batch(&refs, &ids, None).unwrap();
        assert_eq!(r.accepted, per);
    }
    s.close().unwrap();
    path
}

fn code(e: RvfError) -> ErrorCode {
    match e {
        RvfError::Code(c) => c,
        other => panic!("expected a plain error code, got {other:?}"),
    }
}

/// Offsets and (type, padded end) of each segment, walking from byte 0.
fn segments(bytes: &[u8]) -> Vec<(usize, u8, usize)> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos + 64 <= bytes.len() && bytes[pos..pos + 4] == 0x5256_4653u32.to_le_bytes() {
        let plen = u64::from_le_bytes(bytes[pos + 0x10..pos + 0x18].try_into().unwrap()) as usize;
        let end = (pos + 64 + plen + 63) & !63;
        out.push((pos, bytes[pos + 5], end));
        pos = end;
    }
    out
}

const MANIFEST: u8 = rvf_types::SegmentType::Manifest as u8;

/// The QA repro: a store cut inside its newest manifest (data written, commit
/// record lost) used to open as an older, here EMPTY, store.
#[test]
fn truncated_store_does_not_open_as_empty() {
    let d = tempfile::tempdir().unwrap();
    let path = build(d.path(), 2, 50);
    let full = std::fs::read(&path).unwrap();
    let segs = segments(&full);
    let last_manifest = segs.iter().rev().find(|s| s.1 == MANIFEST).unwrap().0;
    // Cut the newest manifest in half.
    let cut = &full[..last_manifest + 64 + 5];
    let cp = d.path().join("cut.rvf");
    std::fs::write(&cp, cut).unwrap();
    // The strict open refuses it.
    let e = RvfStore::open_readonly_strict(&cp)
        .err()
        .expect("strict open must refuse a truncated file");
    assert_eq!(code(e), ErrorCode::TruncatedSegment);
    // The recovering open (ADR-280 crash rule) still serves the last manifest
    // but now REPORTS that it ignored a torn tail.
    let s = RvfStore::open_readonly(&cp).unwrap();
    assert!(s.tail_recovery().is_torn(), "{:?}", s.tail_recovery());
    assert!(s.tail_recovery().uncommitted_segments >= 1);
    // The intact file opens strictly with everything and no torn tail.
    let s = RvfStore::open_readonly_strict(&path).unwrap();
    assert_eq!(s.status().total_vectors, 100);
    assert!(!s.tail_recovery().is_torn());
}

/// Crash recovery is unchanged (ADR-280 criterion 6, see
/// `metadata_crash_safety.rs`): a partially written tail after a valid manifest
/// is ignored and the store opens at that manifest.
#[test]
fn torn_tail_append_is_still_recovered() {
    let d = tempfile::tempdir().unwrap();
    let path = build(d.path(), 2, 50);
    let full = std::fs::read(&path).unwrap();
    let segs = segments(&full);
    // First manifest after creation is segs[0]; the epoch-1 manifest is the
    // second Manifest; the epoch-2 ingest's first segment follows it.
    let manifests: Vec<_> = segs.iter().filter(|s| s.1 == MANIFEST).collect();
    let after_gen1 = manifests[1].2;
    let next_seg = segs.iter().find(|s| s.0 == after_gen1).unwrap().0;
    let cut = &full[..next_seg + 64 + 100]; // mid-payload of the next Vec
    let cp = d.path().join("torn.rvf");
    std::fs::write(&cp, cut).unwrap();
    let s = RvfStore::open_readonly(&cp).expect("torn tail append must still open");
    assert_eq!(s.status().total_vectors, 50, "opens at the last manifest");
    assert!(s.tail_recovery().is_torn());
    assert_eq!(
        code(RvfStore::open_readonly_strict(&cp).err().unwrap()),
        ErrorCode::TruncatedSegment
    );
}

/// Sweep every cut length: a STRICT open either fails or yields exactly a whole
/// generation (0, 50 or 100 vectors) with no torn tail -- never a partial state
/// or a panic. The recovering open never panics and also serves whole generations.
#[test]
fn every_cut_is_an_error_or_a_whole_generation() {
    let d = tempfile::tempdir().unwrap();
    let path = build(d.path(), 2, 50);
    let full = std::fs::read(&path).unwrap();
    let cp = d.path().join("sweep.rvf");
    let mut opened = 0;
    let mut rejected = 0;
    for len in (0..full.len()).step_by(7) {
        std::fs::write(&cp, &full[..len]).unwrap();
        if let Ok(s) = RvfStore::open_readonly(&cp) {
            let n = s.status().total_vectors;
            assert!(n == 0 || n == 50 || n == 100, "recover len {len}: {n}");
        }
        match RvfStore::open_readonly_strict(&cp) {
            Ok(s) => {
                assert!(!s.tail_recovery().is_torn(), "len {len}");
                opened += 1;
                let n = s.status().total_vectors;
                assert!(n == 0 || n == 50 || n == 100, "len {len}: {n} vectors");
            }
            Err(_) => rejected += 1,
        }
    }
    assert!(
        rejected > 0 && opened > 0,
        "opened {opened}, rejected {rejected}"
    );
}

/// A file in a segment-format version this reader does not support is
/// reported as such, not as a missing manifest.
#[test]
fn unsupported_segment_version_is_named() {
    let d = tempfile::tempdir().unwrap();
    let path = build(d.path(), 1, 10);
    let mut bytes = std::fs::read(&path).unwrap();
    // Re-stamp every segment header with version 9.
    for (pos, _, _) in segments(&bytes) {
        bytes[pos + 4] = 9;
    }
    std::fs::write(&path, &bytes).unwrap();
    let e = RvfStore::open_readonly(&path).err().expect("must not open");
    assert_eq!(code(e), ErrorCode::InvalidVersion);
}

#[test]
fn non_finite_query_vectors_are_rejected() {
    let d = tempfile::tempdir().unwrap();
    let path = build(d.path(), 1, 10);
    let s = RvfStore::open_readonly(&path).unwrap();
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let q = [bad, 0.0, 0.0, 0.0];
        let e = s.query(&q, 3, &QueryOptions::default()).unwrap_err();
        assert_eq!(code(e), ErrorCode::NonFiniteQuery);
        let e = s
            .query_with_envelope(&q, 3, &QueryOptions::default())
            .err()
            .unwrap();
        assert_eq!(code(e), ErrorCode::NonFiniteQuery);
    }
    // Finite queries still work.
    assert_eq!(
        s.query(&[1.0, 0.0, 0.0, 0.0], 3, &QueryOptions::default())
            .unwrap()
            .len(),
        3
    );
}

#[test]
fn fully_rejected_ingest_does_not_advance_epoch() {
    let d = tempfile::tempdir().unwrap();
    let path = build(d.path(), 2, 10);
    let mut s = RvfStore::open(&path).unwrap();
    let before = s.epoch();
    let wrong_dim = [1.0f32, 2.0];
    let r = s.ingest_batch(&[&wrong_dim[..]], &[999], None).unwrap();
    assert_eq!((r.accepted, r.rejected), (0, 1));
    assert_eq!(r.epoch, before, "reported epoch");
    assert_eq!(s.epoch(), before, "store epoch");
    s.close().unwrap();
    let s = RvfStore::open_readonly(&path).unwrap();
    assert_eq!(s.epoch(), before, "persisted epoch");
}

#[test]
fn non_finite_vectors_are_rejected_at_ingest() {
    let d = tempfile::tempdir().unwrap();
    let path = build(d.path(), 1, 5);
    let mut s = RvfStore::open(&path).unwrap();
    let before = s.epoch();
    let nan = [f32::NAN, 0.0, 0.0, 0.0];
    let inf = [0.0, f32::INFINITY, 0.0, 0.0];
    let r = s
        .ingest_batch(&[&nan[..], &inf[..]], &[100, 101], None)
        .unwrap();
    assert_eq!((r.accepted, r.rejected), (0, 2));
    assert_eq!(s.epoch(), before, "all rejected => epoch unchanged");
    // Mixed batch: only the finite row lands.
    let ok = [1.0f32, 2.0, 3.0, 4.0];
    let r = s
        .ingest_batch(&[&nan[..], &ok[..]], &[102, 103], None)
        .unwrap();
    assert_eq!((r.accepted, r.rejected), (1, 1));
    assert_eq!(s.epoch(), before + 1);
}

#[test]
fn overflowing_distances_are_an_error_not_an_arbitrary_order() {
    let d = tempfile::tempdir().unwrap();
    let path = build(d.path(), 1, 5);
    let s = RvfStore::open_readonly(&path).unwrap();
    let q = [f32::MAX; 4]; // finite, but squared distances overflow to inf
    let e = s.query(&q, 3, &QueryOptions::default()).unwrap_err();
    assert_eq!(code(e), ErrorCode::NonFiniteQuery);
}

/// A writable open removes a torn tail so later appends stay readable.
#[test]
fn writable_open_repairs_torn_tail_before_appending() {
    let d = tempfile::tempdir().unwrap();
    let path = build(d.path(), 2, 10);
    let full = std::fs::read(&path).unwrap();
    // Torn witness append: valid header claiming 73 bytes, 10 present.
    let mut seg = vec![0u8; 64];
    seg[..4].copy_from_slice(&0x5256_4653u32.to_le_bytes());
    seg[4] = 1;
    seg[5] = 0x0A;
    seg[0x10..0x18].copy_from_slice(&73u64.to_le_bytes());
    let mut torn = full.clone();
    torn.extend_from_slice(&seg);
    torn.extend_from_slice(&[7u8; 10]);
    std::fs::write(&path, &torn).unwrap();
    assert!(RvfStore::open_readonly_strict(&path).is_err());

    let mut s = RvfStore::open(&path).unwrap();
    assert!(s.tail_recovery().repaired);
    s.ingest_batch(&[&[9.0f32, 9.0, 9.0, 9.0][..]], &[999], None)
        .unwrap();
    s.close().unwrap();
    let s = RvfStore::open_readonly_strict(&path).expect("clean after repair");
    assert_eq!(s.status().total_vectors, 21);
}
