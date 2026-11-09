#![cfg(feature = "ed25519")]
//! ADR-354 item 5: segment authentication and signed-store writes under
//! `Strict`/`Paranoid` (continuation of `strict_policy.rs`).

use rvf_runtime::options::WitnessConfig;
use rvf_runtime::{QueryOptions, RvfOptions, RvfStore};
use rvf_types::security::{SecurityError, SecurityPolicy};
use rvf_types::{RvfError, SegmentType};
use rvf_wire::hash::compute_shake256_128;
use rvf_wire::tail_scan::split_root_page;
use std::path::Path;

#[path = "support/signed.rs"]
mod signed;
use signed::*;

// ---- segment authentication --------------------------------------------------------

fn swap_vec_payload(path: &Path, recompute_header_hash: bool) {
    let mut bytes = std::fs::read(path).unwrap();
    let segs = segments(&bytes);
    let (off, _, len) = *segs.iter().find(|s| s.1 == SegmentType::Vec as u8).unwrap();
    bytes[off + 64 + len - 1] ^= 0x40; // alter a vector value
    if recompute_header_hash {
        // Keyless algo-2 hash: the attacker can recompute it.
        let h = compute_shake256_128(&bytes[off + 64..off + 64 + len]);
        bytes[off + 0x28..off + 0x38].copy_from_slice(&h);
    }
    std::fs::write(path, &bytes).unwrap();
}

#[test]
fn vector_segment_with_recomputed_header_hash_is_refused_by_the_committed_hash() {
    let d = tempfile::tempdir().unwrap();
    let p = signed_store(d.path());
    swap_vec_payload(&p, true);
    for policy in [
        SecurityPolicy::Strict,
        SecurityPolicy::Paranoid,
        SecurityPolicy::WarnOnly,
    ] {
        let e = security(open_err(&p, &[public(&SECRET)], policy));
        assert!(
            matches!(
                e,
                SecurityError::ContentHashMismatch {
                    pointer_name: "segment_content_hash",
                    ..
                }
            ),
            "{policy:?}: {e:?}"
        );
    }
    // Permissive has no committed-hash enforcement: wrong data is served.
    assert!(RvfStore::open_with_policy(&p, opts(&[]), SecurityPolicy::Permissive).is_ok());
}

#[test]
fn corrupt_vector_segment_without_hash_fix_is_refused_everywhere() {
    let d = tempfile::tempdir().unwrap();
    let p = signed_store(d.path());
    swap_vec_payload(&p, false);
    for policy in [
        SecurityPolicy::Strict,
        SecurityPolicy::Paranoid,
        SecurityPolicy::WarnOnly,
    ] {
        assert!(RvfStore::open_with_policy(&p, opts(&[public(&SECRET)]), policy).is_err());
    }
}

#[test]
fn index_segment_must_be_authentic_not_silently_rebuilt() {
    let d = tempfile::tempdir().unwrap();
    let path = d.path().join("idx.rvf");
    let mut o = opts(&[]);
    o.witness = WitnessConfig {
        witness_ingest: false,
        witness_delete: false,
        witness_compact: false,
        audit_queries: false,
    };
    let mut s = RvfStore::create_signed(&path, o, &SECRET).unwrap();
    let v = rows(1100, 0.0);
    let refs: Vec<&[f32]> = v.iter().map(Vec::as_slice).collect();
    s.ingest_batch(&refs, &(0..1100).collect::<Vec<u64>>(), None)
        .unwrap();
    // Force an index build so close() persists an INDEX_SEG.
    let _ = s
        .query(&[1.0, 1.0, 2.0, 3.0], 5, &QueryOptions::default())
        .unwrap();
    s.close().unwrap();

    let mut bytes = std::fs::read(&path).unwrap();
    let segs = segments(&bytes);
    let &(off, _, len) = segs
        .iter()
        .rev()
        .find(|s| s.1 == SegmentType::Index as u8)
        .expect("an INDEX_SEG must have been persisted (>= 1024 vectors + a query)");
    bytes[off + 64 + len / 2] ^= 0x01;
    let h = compute_shake256_128(&bytes[off + 64..off + 64 + len]);
    bytes[off + 0x28..off + 0x38].copy_from_slice(&h);
    std::fs::write(&path, &bytes).unwrap();
    let e = security(open_err(&path, &[public(&SECRET)], SecurityPolicy::Strict));
    assert!(
        matches!(e, SecurityError::ContentHashMismatch { .. }),
        "{e:?}"
    );
}

#[test]
fn paranoid_reads_every_segment_strict_defers_unread_ones() {
    let d = tempfile::tempdir().unwrap();
    let path = d.path().join("k.rvf");
    let mut s = RvfStore::create_signed(&path, opts(&[]), &SECRET).unwrap();
    s.ingest_batch(&[&[1.0, 2.0, 3.0, 4.0]], &[1], None)
        .unwrap();
    s.embed_kernel(0, 0, 0, &[0x7f; 256], 0, None).unwrap();
    s.close().unwrap();

    let mut bytes = std::fs::read(&path).unwrap();
    let segs = segments(&bytes);
    let (off, _, len) = *segs
        .iter()
        .find(|s| s.1 == SegmentType::Kernel as u8)
        .unwrap();
    bytes[off + 64 + len - 1] ^= 0x01;
    let h = compute_shake256_128(&bytes[off + 64..off + 64 + len]);
    bytes[off + 0x28..off + 0x38].copy_from_slice(&h);
    std::fs::write(&path, &bytes).unwrap();

    let trusted = [public(&SECRET)];
    let strict = RvfStore::open_with_policy(&path, opts(&trusted), SecurityPolicy::Strict).unwrap();
    let e = strict.extract_kernel().unwrap_err();
    // Executable bytes are refused at the point of use.
    assert!(
        matches!(e, RvfError::Code(_) | RvfError::Security(_)),
        "{e:?}"
    );
    assert!(strict.extract_kernel().is_err());
    let e = security(open_err(&path, &trusted, SecurityPolicy::Paranoid));
    assert!(
        matches!(e, SecurityError::ContentHashMismatch { .. }),
        "{e:?}"
    );
}

// ---- writing signed stores ------------------------------------------------------------

#[test]
fn signed_store_stays_strict_valid_across_append_delete_compact_and_audit() {
    let d = tempfile::tempdir().unwrap();
    let path = d.path().join("w.rvf");
    let o = RvfOptions {
        witness: WitnessConfig {
            audit_queries: true,
            ..Default::default()
        },
        ..opts(&[])
    };
    let mut s = RvfStore::create_signed(&path, o, &SECRET).unwrap();
    let trusted = [public(&SECRET)];
    let check = |expect: u64| {
        let r =
            RvfStore::open_with_policy(&path, opts(&trusted), SecurityPolicy::Paranoid).unwrap();
        assert_eq!(r.status().total_vectors, expect);
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len() % 64, 0);
        assert_eq!(split_root_page(&bytes).1.map(<[u8]>::len), Some(4096));
    };

    let v = rows(8, 0.0);
    let refs: Vec<&[f32]> = v.iter().map(Vec::as_slice).collect();
    s.ingest_batch(&refs, &(0..8).collect::<Vec<u64>>(), None)
        .unwrap();
    // Readers can open a live writer's file (no lock for read-only).
    check(8);
    s.delete(&[1, 2]).unwrap();
    check(6);
    s.query_audited(&[0.0, 1.0, 2.0, 3.0], 2, &QueryOptions::default())
        .unwrap();
    check(6);
    s.compact().unwrap();
    check(6);
    s.close().unwrap();

    // Reopen for writing under the same key, append, close.
    let mut w =
        RvfStore::open_signed(&path, opts(&trusted), SecurityPolicy::Strict, &SECRET).unwrap();
    w.ingest_batch(&[&[9.0, 9.0, 9.0, 9.0]], &[77], None)
        .unwrap();
    w.close().unwrap();
    check(7);
}

#[test]
fn read_only_policy_open_cannot_write_and_signed_open_needs_a_signing_policy() {
    let d = tempfile::tempdir().unwrap();
    let p = signed_store(d.path());
    let mut s =
        RvfStore::open_with_policy(&p, opts(&[public(&SECRET)]), SecurityPolicy::Strict).unwrap();
    assert!(s
        .ingest_batch(&[&[1.0, 2.0, 3.0, 4.0]], &[500], None)
        .is_err());
    assert!(RvfStore::open_signed(
        &p,
        opts(&[public(&SECRET)]),
        SecurityPolicy::Permissive,
        &SECRET
    )
    .is_err());
    // open_signed itself verifies first: an untrusted key list is refused.
    assert!(RvfStore::open_signed(&p, opts(&[]), SecurityPolicy::Strict, &SECRET).is_err());
}

#[test]
fn plain_open_of_a_signed_file_refuses_to_write_unless_unsigning_is_allowed() {
    let d = tempfile::tempdir().unwrap();
    let p = signed_store(d.path());
    let before = std::fs::read(&p).unwrap();
    let mut s = RvfStore::open(&p).unwrap(); // Permissive: no key, nothing verified
    for r in [
        s.ingest_batch(&[&[5.0, 5.0, 5.0, 5.0]], &[900], None)
            .map(|_| ()),
        s.delete(&[1]).map(|_| ()),
        s.compact().map(|_| ()),
    ] {
        assert!(
            matches!(r, Err(RvfError::Code(rvf_types::ErrorCode::ReadOnly))),
            "{r:?}"
        );
    }
    s.close().unwrap();
    assert_eq!(
        std::fs::read(&p).unwrap(),
        before,
        "refused writes leave the signed file untouched"
    );
    assert!(
        RvfStore::open_with_policy(&p, opts(&[public(&SECRET)]), SecurityPolicy::Strict).is_ok()
    );

    // Explicit consent: the write succeeds and the file is unsigned afterwards.
    let mut s = RvfStore::open(&p).unwrap();
    s.allow_unsigning(true);
    s.ingest_batch(&[&[5.0, 5.0, 5.0, 5.0]], &[900], None)
        .unwrap();
    s.close().unwrap();
    let bytes = std::fs::read(&p).unwrap();
    // No stale RVM0 page is left behind in the middle of the file.
    assert!(split_root_page(&bytes).1.is_none());
    assert!(!bytes
        .chunks(64)
        .any(|c| c[..4] == rvf_types::ROOT_MANIFEST_MAGIC.to_le_bytes()));
    let e = security(open_err(&p, &[public(&SECRET)], SecurityPolicy::Strict));
    assert!(matches!(e, SecurityError::UnsignedManifest { .. }));
    assert_eq!(
        RvfStore::open_readonly(&p).unwrap().status().total_vectors,
        13
    );
}

#[test]
fn segments_written_in_an_open_signed_session_read_back_under_enforcement() {
    // `open_signed` enforces committed hashes from the manifest it opened;
    // segments added (and moved by compaction) in the same session must be
    // readable too, not refused as uncommitted.
    let d = tempfile::tempdir().unwrap();
    let p = signed_store(d.path());
    let trusted = [public(&SECRET)];
    let mut w = RvfStore::open_signed(&p, opts(&trusted), SecurityPolicy::Strict, &SECRET).unwrap();

    w.embed_wasm(0, 0, 0, b"\0asm\x01\0\0\0session module", 1, 0, 0)
        .unwrap();
    assert!(
        w.extract_wasm().unwrap().is_some(),
        "extract after embed, same session"
    );

    w.embed_kernel(0, 0, 0, &[0x7f; 128], 0, None).unwrap();
    w.delete(&[1]).unwrap();
    w.compact().unwrap();
    assert!(
        w.extract_kernel().unwrap().is_some(),
        "extract after compact, same session"
    );
    assert!(w.extract_wasm().unwrap().is_some());
    w.close().unwrap();

    let r = RvfStore::open_with_policy(&p, opts(&trusted), SecurityPolicy::Paranoid).unwrap();
    assert!(r.extract_kernel().unwrap().is_some());
    assert_eq!(r.status().total_vectors, 11);
}
