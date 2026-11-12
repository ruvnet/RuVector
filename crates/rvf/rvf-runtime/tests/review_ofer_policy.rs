#![cfg(feature = "ed25519")]
//! Adversarial review repros for the Strict policy and tail scan (F-12, F-13).

use rvf_runtime::{RvfOptions, RvfStore};
use rvf_types::security::{SecurityError, SecurityPolicy};
use rvf_types::{RvfError, SegmentFlags, SegmentType};
use rvf_wire::writer::write_segment;

#[path = "support/signed.rs"]
mod signed;
use signed::*;

/// F-12 (fixed): a plain `RvfStore::open` of a signed store can no longer
/// destroy the signature by accident. It cannot re-sign, so its writes are
/// refused (`ReadOnly`) and the file is untouched, unless the caller opts in
/// with `allow_unsigning(true)`.
#[test]
fn f12_plain_open_append_cannot_silently_unsign_a_signed_store() {
    let d = tempfile::tempdir().unwrap();
    let p = signed_store(d.path());
    assert!(
        RvfStore::open_with_policy(&p, opts(&[public(&SECRET)]), SecurityPolicy::Strict).is_ok()
    );
    let mut s = RvfStore::open(&p).unwrap();
    let v = rows(1, 500.0);
    let refs: Vec<&[f32]> = v.iter().map(Vec::as_slice).collect();
    let r = s.ingest_batch(&refs, &[900], None);
    assert!(
        matches!(r, Err(RvfError::Code(rvf_types::ErrorCode::ReadOnly))),
        "{r:?}"
    );
    s.close().unwrap();
    assert!(
        RvfStore::open_with_policy(&p, opts(&[public(&SECRET)]), SecurityPolicy::Strict).is_ok(),
        "still signed"
    );
}

/// F-13: under Permissive (and WarnOnly) a fake MANIFEST header planted at a
/// 64-byte aligned position *inside the payload of a later VEC segment*
/// (attacker-chosen vector floats, e.g. the leftover of a crashed append)
/// beats the real manifest. The store opens with the attacker's directory.
#[test]
fn f13_fake_manifest_inside_trailing_vec_payload_wins_under_permissive() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("plain.rvf");
    {
        let mut s = RvfStore::create(
            &p,
            RvfOptions {
                dimension: 4,
                ..Default::default()
            },
        )
        .unwrap();
        for b in 0..2u64 {
            let v = rows(6, b as f32 * 100.0);
            let refs: Vec<&[f32]> = v.iter().map(Vec::as_slice).collect();
            let ids: Vec<u64> = (0..6).map(|i| b * 10 + i).collect();
            s.ingest_batch(&refs, &ids, None).unwrap();
        }
        s.close().unwrap();
    }
    let real = RvfStore::open_readonly(&p).unwrap().status().total_vectors;
    assert_eq!(real, 12);

    // Fake manifest: epoch 99, dim 4, 0 vectors, empty directory.
    let mut fm = Vec::new();
    fm.extend_from_slice(&99u32.to_le_bytes());
    fm.extend_from_slice(&4u16.to_le_bytes());
    fm.extend_from_slice(&0u64.to_le_bytes());
    fm.extend_from_slice(&0u32.to_le_bytes());
    fm.extend_from_slice(&[0u8; 4]); // profile + metric + reserved
    fm.extend_from_slice(&0u32.to_le_bytes()); // deleted count
    let fake = write_segment(SegmentType::Manifest as u8, &fm, SegmentFlags::empty(), 777);
    // Carrier: an ordinary VEC segment whose payload begins with the fake.
    let carrier = write_segment(SegmentType::Vec as u8, &fake, SegmentFlags::empty(), 778);
    let mut bytes = std::fs::read(&p).unwrap();
    assert_eq!(bytes.len() % 64, 0);
    bytes.extend_from_slice(&carrier);
    std::fs::write(&p, &bytes).unwrap();

    let got = RvfStore::open_readonly(&p).unwrap().status().total_vectors;
    assert_eq!(
        got, 0,
        "attacker manifest hidden inside a payload won the tail scan"
    );
}

fn fake_manifest_segment() -> Vec<u8> {
    let mut fm = Vec::new();
    fm.extend_from_slice(&99u32.to_le_bytes());
    fm.extend_from_slice(&4u16.to_le_bytes());
    fm.extend_from_slice(&0u64.to_le_bytes());
    fm.extend_from_slice(&0u32.to_le_bytes());
    fm.extend_from_slice(&[0u8; 4]);
    fm.extend_from_slice(&0u32.to_le_bytes());
    write_segment(SegmentType::Manifest as u8, &fm, SegmentFlags::empty(), 777)
}

/// F-14: legacy (byte-granular) scanning is selected by ONE attacker-writable
/// byte, the first segment header's `checksum_algo`, which no hash covers. On a
/// wire-dialect file this makes the runtime (Permissive/WarnOnly) pick an
/// UNALIGNED fake manifest hidden in a payload, while the rvf-wire aligned
/// scanner (what rvm-rvf and other consumers use) still sees the real one: a
/// parser differential on identical bytes, the class ADR-354 item 4 targets.
#[test]
fn f14_first_header_algo_byte_no_longer_selects_the_weaker_scan() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("plain.rvf");
    {
        let mut s = RvfStore::create(
            &p,
            RvfOptions {
                dimension: 4,
                ..Default::default()
            },
        )
        .unwrap();
        let v = rows(6, 0.0);
        let refs: Vec<&[f32]> = v.iter().map(Vec::as_slice).collect();
        let ids: Vec<u64> = (0..6).collect();
        s.ingest_batch(&refs, &ids, None).unwrap();
        s.close().unwrap();
    }
    let mut bytes = std::fs::read(&p).unwrap();
    let (real_off, _) = rvf_wire::tail_scan::find_latest_manifest(&bytes).unwrap();
    // carrier: VEC segment, fake manifest at payload offset 8 (not 64-aligned)
    let mut payload = vec![0u8; 8];
    payload.extend_from_slice(&fake_manifest_segment());
    let carrier = write_segment(SegmentType::Vec as u8, &payload, SegmentFlags::empty(), 778);
    bytes.extend_from_slice(&carrier);
    std::fs::write(&p, &bytes).unwrap();

    // Untouched: both agree on the real manifest.
    let (a, _) = rvf_wire::tail_scan::find_latest_manifest(&bytes).unwrap();
    assert_eq!(a, real_off);
    assert_eq!(
        RvfStore::open_readonly(&p).unwrap().status().total_vectors,
        6
    );

    // Flip the (unhashed) first-segment algo byte to the legacy marker.
    bytes[0x20] = 0;
    std::fs::write(&p, &bytes).unwrap();
    let (b, _) = rvf_wire::tail_scan::find_latest_manifest(&bytes).unwrap();
    assert_eq!(b, real_off, "aligned scanner unaffected");
    let runtime_view = RvfStore::open_readonly(&p).unwrap().status().total_vectors;
    // F-14 fixed: the unhashed byte no longer selects the byte-granular scan
    // on a file whose layout is plainly wire dialect.
    assert_eq!(
        runtime_view, 6,
        "runtime ignores the unaligned fake manifest, agreeing with rvf-wire"
    );
}

/// Checked and sound: an empty trusted set never means accept-all.
#[test]
fn strict_with_empty_trusted_set_refuses_a_validly_signed_file() {
    let d = tempfile::tempdir().unwrap();
    let p = signed_store(d.path());
    for pol in [SecurityPolicy::Strict, SecurityPolicy::Paranoid] {
        match RvfStore::open_with_policy(&p, opts(&[]), pol) {
            Err(RvfError::Security(SecurityError::UnknownSigner { .. })) => {}
            other => panic!("{pol:?}: {:?}", other.map(|_| ())),
        }
    }
}

/// F-16 (fixed): every header field of a committed segment is authenticated.
/// The hash covers the payload; the directory pins `segment_id`, `seg_type`
/// and `payload_length`; the writer's constants pin `flags`, `compression`,
/// the reserved words, `uncompressed_len`, `timestamp_ns`, `alignment_pad`,
/// the version and the algorithm. Rewriting any one is refused by Strict and
/// Paranoid with a ContentHashMismatch.
#[test]
fn f16_strict_authenticates_every_segment_header_field() {
    let d = tempfile::tempdir().unwrap();
    let p = signed_store(d.path());
    let orig = std::fs::read(&p).unwrap();
    let segs = segments(&orig);
    let (off, _, _) = *segs.iter().find(|s| s.1 == SegmentType::Vec as u8).unwrap();
    // (byte offset in the header, xor mask)
    let edits: [(usize, u8, &str); 10] = [
        (0x05, 0x04, "seg_type"),
        (0x06, 0xFF, "flags"),
        (0x08, 0x55, "segment_id"),
        (0x18, 0x55, "timestamp_ns"),
        (0x21, 0x03, "compression"),
        (0x22, 0x01, "reserved_0"),
        (0x24, 0x01, "reserved_1"),
        (0x38, 0x01, "uncompressed_len"),
        (0x3C, 0x40, "alignment_pad"),
        (0x04, 0x01, "version"),
    ];
    for (at, mask, name) in edits {
        let mut bytes = orig.clone();
        bytes[off + at] ^= mask;
        std::fs::write(&p, &bytes).unwrap();
        for pol in [SecurityPolicy::Strict, SecurityPolicy::Paranoid] {
            let r = RvfStore::open_with_policy(&p, opts(&[public(&SECRET)]), pol);
            assert!(r.is_err(), "{pol:?} accepted a rewritten {name}");
        }
    }
    std::fs::write(&p, &orig).unwrap();
    assert!(
        RvfStore::open_with_policy(&p, opts(&[public(&SECRET)]), SecurityPolicy::Paranoid).is_ok()
    );
}
