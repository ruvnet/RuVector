#![cfg(feature = "ed25519")]
//! Adversarial review repros (finding F-01 / F-02): compaction of a signed
//! store re-signs bytes it never authenticated.

use rvf_runtime::RvfStore;
use rvf_types::security::SecurityPolicy;
use rvf_types::SegmentType;
use rvf_wire::hash::compute_shake256_128;
use rvf_wire::tail_scan::split_root_page;

#[path = "support/signed.rs"]
mod signed;
use signed::*;

fn embed_and_close(path: &std::path::Path, wasm: &[u8]) {
    let mut s = RvfStore::open_signed(
        path,
        opts_prefix(&[public(&SECRET)]),
        SecurityPolicy::Strict,
        &SECRET,
    )
    .unwrap();
    s.embed_wasm(0, 0, 0, wasm, 1, 0, 0).unwrap();
    s.close().unwrap();
}

/// F-01: a tampered, manifest-listed WASM segment (header hash recomputed by
/// the attacker, who needs no key) is not read by a Strict open, and
/// `compact()` copies it verbatim and records a fresh committed hash over the
/// tampered bytes. Afterwards the Strict reader accepts it.
#[test]
fn f01_compact_refuses_a_tampered_listed_segment() {
    let d = tempfile::tempdir().unwrap();
    let p = signed_store(d.path());
    embed_and_close(&p, b"\0asm-GOOD-GOOD-GOOD-GOOD-GOOD");

    // Attacker (no key) flips a bytecode byte and fixes the header hash.
    let mut bytes = std::fs::read(&p).unwrap();
    let segs = segments(&bytes);
    let (off, _, len) = *segs
        .iter()
        .find(|s| s.1 == SegmentType::Wasm as u8)
        .unwrap();
    bytes[off + 64 + len - 1] ^= 0x01;
    let h = compute_shake256_128(&bytes[off + 64..off + 64 + len]);
    bytes[off + 0x28..off + 0x38].copy_from_slice(&h);
    std::fs::write(&p, &bytes).unwrap();

    // Strict open succeeds (wasm is lazy), but reading it is refused.
    let s = RvfStore::open_with_policy(&p, opts_prefix(&[public(&SECRET)]), SecurityPolicy::Strict)
        .expect("strict open does not touch the wasm segment");
    assert!(
        s.extract_wasm().is_err(),
        "tampered wasm refused before compaction"
    );
    drop(s);

    // The legitimate owner compacts (maintenance). F-01 fixed: the tampered
    // segment fails its committed hash, so compaction refuses and nothing
    // changes; it is not laundered into the signature.
    let mut s = RvfStore::open_signed(
        &p,
        opts_prefix(&[public(&SECRET)]),
        SecurityPolicy::Strict,
        &SECRET,
    )
    .unwrap();
    let before = std::fs::read(&p).unwrap();
    match s.compact() {
        Err(rvf_types::RvfError::Security(
            rvf_types::security::SecurityError::ContentHashMismatch { pointer_name, .. },
        )) => assert_eq!(pointer_name, "compaction_source"),
        other => panic!("compaction must refuse: {:?}", other.map(|_| ())),
    }
    assert_eq!(
        s.status().total_vectors,
        12,
        "refused before any state change"
    );
    s.close().unwrap();
    assert_eq!(std::fs::read(&p).unwrap(), before, "file untouched");

    let s = RvfStore::open_with_policy(&p, opts_prefix(&[public(&SECRET)]), SecurityPolicy::Strict)
        .unwrap();
    assert!(s.extract_wasm().is_err(), "tampered wasm is still refused");
}

/// F-01 in a signing session that created the store: the handle's own
/// recorded hashes (not the file) are the reference, so tampering done while
/// the handle is open is caught too.
#[test]
fn f01_midsession_tamper_is_caught_by_the_handles_own_hashes() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("m.rvf");
    let mut s = RvfStore::create_signed(&p, opts(&[]), &SECRET).unwrap();
    s.ingest_batch(&[&[1.0, 2.0, 3.0, 4.0]], &[1], None)
        .unwrap();
    s.embed_wasm(0, 0, 0, b"\0asm-GOOD-GOOD-GOOD-GOOD-GOOD", 1, 0, 0)
        .unwrap();
    let mut bytes = std::fs::read(&p).unwrap();
    let segs = segments(&bytes);
    let (off, _, len) = *segs
        .iter()
        .find(|s| s.1 == SegmentType::Wasm as u8)
        .unwrap();
    bytes[off + 64 + len - 1] ^= 1;
    let h = compute_shake256_128(&bytes[off + 64..off + 64 + len]);
    bytes[off + 0x28..off + 0x38].copy_from_slice(&h);
    std::fs::write(&p, &bytes).unwrap();
    assert!(
        s.extract_wasm().is_err(),
        "reads are enforced in a signing session"
    );
    assert!(s.compact().is_err());
}

/// F-02: a segment that is NOT in the signed manifest directory, injected
/// between the manifest and the (re-appended, still valid) root page, is
/// invisible to a Strict open but is adopted and signed by `compact()`.
#[test]
fn f02_compact_drops_unindexed_injected_segment() {
    let d = tempfile::tempdir().unwrap();
    let p = signed_store(d.path());
    let bytes = std::fs::read(&p).unwrap();
    let (body, page) = split_root_page(&bytes);
    let page = page.unwrap().to_vec();
    // Forge an unsigned, header-valid WASM segment (keyless).
    let payload = vec![0xEEu8; 128];
    let forged = rvf_wire::writer::write_segment(
        SegmentType::Wasm as u8,
        &payload,
        rvf_types::SegmentFlags::empty(),
        9999,
    );
    let mut forged = forged;
    forged[0x20] = 2; // SHAKE-256-128, keyless
    let h = compute_shake256_128(&payload);
    forged[0x28..0x38].copy_from_slice(&h);
    let mut evil = body.to_vec();
    evil.extend_from_slice(&forged);
    evil.extend_from_slice(&page);
    std::fs::write(&p, &evil).unwrap();

    // Strict open still succeeds: newest manifest unchanged, page valid.
    let s = RvfStore::open_with_policy(&p, opts_prefix(&[public(&SECRET)]), SecurityPolicy::Strict)
        .expect("strict open succeeds");
    assert!(s.extract_wasm().unwrap().is_none());
    drop(s);

    let mut s = RvfStore::open_signed(
        &p,
        opts_prefix(&[public(&SECRET)]),
        SecurityPolicy::Strict,
        &SECRET,
    )
    .unwrap();
    s.compact().unwrap();
    s.close().unwrap();

    // F-02 fixed: only directory-listed segments are carried, so the
    // injected segment is dropped, not adopted into the signature.
    let s = RvfStore::open_with_policy(
        &p,
        opts_prefix(&[public(&SECRET)]),
        SecurityPolicy::Paranoid,
    )
    .unwrap();
    assert!(
        s.extract_wasm().unwrap().is_none(),
        "injected segment was not adopted"
    );
    let bytes = std::fs::read(&p).unwrap();
    assert!(
        !segments(&bytes)
            .iter()
            .any(|s| s.1 == SegmentType::Wasm as u8),
        "and it is gone from the file"
    );
}
