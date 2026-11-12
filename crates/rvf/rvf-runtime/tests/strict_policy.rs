#![cfg(feature = "ed25519")]
//! ADR-354 item 5 / ADR-033 sections 3.5 and 4: `Strict` is real.
//!
//! Runs against the real `RvfStore::create_signed` / `open_with_policy`, not a
//! model of them.

use rvf_runtime::root_page::{self, PageContent};
use rvf_runtime::{QueryOptions, RvfStore};
use rvf_types::security::{SecurityError, SecurityPolicy};
use rvf_types::{FileIdentity, SegmentType};
use rvf_wire::hash::compute_shake256_128;
use rvf_wire::tail_scan::split_root_page;
use std::path::{Path, PathBuf};

#[path = "support/signed.rs"]
mod signed;
use signed::*;

#[test]
fn signed_store_opens_under_strict_and_paranoid_and_queries() {
    let d = tempfile::tempdir().unwrap();
    let p = signed_store(d.path());
    let bytes = std::fs::read(&p).unwrap();
    assert_eq!(bytes.len() % 64, 0);
    assert!(split_root_page(&bytes).1.is_some(), "ends in an RVM0 page");
    for policy in [
        SecurityPolicy::Strict,
        SecurityPolicy::Paranoid,
        SecurityPolicy::WarnOnly,
    ] {
        let s = RvfStore::open_with_policy(&p, opts(&[public(&SECRET)]), policy).unwrap();
        assert_eq!(s.security_policy(), policy);
        assert!(!s.legacy_dialect());
        assert_eq!(s.status().total_vectors, 12);
        let r = s
            .query(&[0.0, 1.0, 2.0, 3.0], 3, &QueryOptions::default())
            .unwrap();
        assert_eq!(r[0].id, 0);
    }
    // open_with_options reads RvfOptions.security_policy (default Strict).
    assert!(RvfStore::open_with_options(&p, opts(&[public(&SECRET)])).is_ok());
    assert!(RvfStore::open_with_options(&p, opts(&[])).is_err());
}

#[test]
fn unsigned_manifest_is_refused_under_strict_and_accepted_under_lower_policies() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("plain.rvf");
    let mut s = RvfStore::create(&p, opts(&[])).unwrap();
    s.ingest_batch(&[&[1.0, 2.0, 3.0, 4.0]], &[1], None)
        .unwrap();
    s.close().unwrap();

    for policy in [SecurityPolicy::Strict, SecurityPolicy::Paranoid] {
        let e = security(open_err(&p, &[public(&SECRET)], policy));
        assert!(matches!(e, SecurityError::UnsignedManifest { .. }), "{e:?}");
        assert_eq!(
            rvf_runtime::store::security_error_code(&e) as u16,
            0x0800,
            "stable code"
        );
    }
    for policy in [SecurityPolicy::WarnOnly, SecurityPolicy::Permissive] {
        let s = RvfStore::open_with_policy(&p, opts(&[]), policy).unwrap();
        assert_eq!(s.status().total_vectors, 1);
    }
    // The explicit lower policy that existing callers get implicitly.
    assert!(RvfStore::open(&p).is_ok());
    assert!(RvfStore::open_readonly(&p).is_ok());
}

// ---- ADR-033 section 3.5: Malicious Tail Manifest ---------------------------------

/// Rewrite pointer fields in the page, FIX the CRC32C, do NOT re-sign.
fn tampered_pointer_file(dir: &Path) -> (PathBuf, u64) {
    let p = signed_store(dir);
    let mut bytes = std::fs::read(&p).unwrap();
    let segs = segments(&bytes);
    let manifest_off = segs
        .iter()
        .rev()
        .find(|s| s.1 == SegmentType::Manifest as u8)
        .unwrap()
        .0;
    let other = segs
        .iter()
        .find(|s| s.1 == SegmentType::Vec as u8)
        .unwrap()
        .0;
    let mut page = page_of(&bytes);
    page[0x008..0x010].copy_from_slice(&(other as u64).to_le_bytes()); // redirect L1
    root_page::reseal_crc(&mut page);
    assert!(root_page::view(&page).crc_ok, "attacker kept the CRC valid");
    set_page(&mut bytes, &page);
    std::fs::write(&p, &bytes).unwrap();
    (p, manifest_off as u64)
}

#[test]
fn malicious_tail_manifest_fails_strict_and_paranoid_deterministically() {
    let d = tempfile::tempdir().unwrap();
    let (p, manifest_off) = tampered_pointer_file(d.path());
    for policy in [SecurityPolicy::Strict, SecurityPolicy::Paranoid] {
        let e = security(open_err(&p, &[public(&SECRET)], policy));
        match e {
            SecurityError::InvalidSignature {
                manifest_offset,
                rejection_phase,
            } => {
                assert_eq!(manifest_offset, manifest_off);
                assert_eq!(
                    rejection_phase, "signature_verification",
                    "not content_hash"
                );
            }
            other => panic!("{other:?}"),
        }
        // Stable code, same on every run.
        assert_eq!(
            rvf_runtime::store::security_error_code(&security(open_err(
                &p,
                &[public(&SECRET)],
                policy
            ))) as u16,
            0x0103
        );
    }
}

#[test]
fn malicious_tail_manifest_warnonly_opens_and_serves_committed_data() {
    // The page's L1 pointer is advisory here (the manifest is located by the
    // single tail scanner), so redirecting it cannot redirect reads. WarnOnly
    // warns about the signature and serves data that still matches the
    // manifest's committed hashes.
    let d = tempfile::tempdir().unwrap();
    let (p, _) = tampered_pointer_file(d.path());
    let s =
        RvfStore::open_with_policy(&p, opts(&[public(&SECRET)]), SecurityPolicy::WarnOnly).unwrap();
    assert_eq!(s.status().total_vectors, 12);
}

#[test]
fn permissive_ignores_everything_by_design() {
    let d = tempfile::tempdir().unwrap();
    let (p, _) = tampered_pointer_file(d.path());
    assert!(RvfStore::open_with_policy(&p, opts(&[]), SecurityPolicy::Permissive).is_ok());
}

#[test]
fn resigned_with_a_different_key_is_unknown_signer_not_invalid_signature() {
    let d = tempfile::tempdir().unwrap();
    let p = signed_store(d.path());
    let mut bytes = std::fs::read(&p).unwrap();
    let old = root_page::view(&page_of(&bytes));
    // Attacker rebuilds the page byte-for-byte with their own key.
    let id = FileIdentity::zeroed();
    let forged = root_page::build_signed_page(
        &PageContent {
            l1_offset: old.l1_offset,
            l1_length: old.l1_length,
            l1_content_hash: old.l1_content_hash,
            l1_segment_id: old.l1_segment_id,
            total_vectors: 12,
            dimension: 4,
            profile: 0,
            epoch: 1,
            identity: &id,
        },
        &OTHER_SECRET,
    );
    set_page(&mut bytes, &forged);
    std::fs::write(&p, &bytes).unwrap();

    let e = security(open_err(&p, &[public(&SECRET)], SecurityPolicy::Strict));
    match e {
        SecurityError::UnknownSigner {
            actual_signer,
            expected_signer,
            ..
        } => {
            assert_eq!(
                actual_signer,
                root_page::signer_fingerprint(&public(&OTHER_SECRET))
            );
            assert_eq!(
                expected_signer,
                Some(root_page::signer_fingerprint(&public(&SECRET)))
            );
        }
        other => panic!("{other:?}"),
    }
    // Distinct from "no signature" and from "bad signature".
    assert_eq!(rvf_runtime::store::security_error_code(&e) as u16, 0x0802);
    // An empty trust store cannot vouch for anyone.
    let e = security(open_err(&p, &[], SecurityPolicy::Strict));
    assert!(matches!(
        e,
        SecurityError::UnknownSigner {
            expected_signer: None,
            ..
        }
    ));
}

#[test]
fn forged_fingerprint_claiming_the_trusted_signer_is_invalid_signature() {
    let d = tempfile::tempdir().unwrap();
    let p = signed_store(d.path());
    let mut bytes = std::fs::read(&p).unwrap();
    let mut page = page_of(&bytes);
    // Sign with the attacker's key, then overwrite the fingerprint with the
    // trusted one (before re-signing so the page is internally consistent
    // except for the key).
    page[root_page::OFF_SIGNER_FINGERPRINT..root_page::OFF_SIGNER_FINGERPRINT + 16]
        .copy_from_slice(&root_page::signer_fingerprint(&public(&SECRET)));
    let sig = rvf_types::ed25519::ed25519_sign(&OTHER_SECRET, &root_page::signing_bytes(&page));
    page[0x09C..0x09C + 64].copy_from_slice(&sig);
    root_page::reseal_crc(&mut page);
    set_page(&mut bytes, &page);
    std::fs::write(&p, &bytes).unwrap();
    let e = security(open_err(&p, &[public(&SECRET)], SecurityPolicy::Strict));
    assert!(
        matches!(
            e,
            SecurityError::InvalidSignature {
                rejection_phase: "signature_verification",
                ..
            }
        ),
        "{e:?}"
    );
}

#[test]
fn planted_newer_manifest_beyond_a_signed_page_is_refused_unless_the_prefix_is_opted_in() {
    // Attacker keeps the genuine page (valid signature!) but plants a newer
    // manifest segment in front of it. The tail scan finds the plant; the
    // signed page commits to a different offset and hash.
    let d = tempfile::tempdir().unwrap();
    let p = signed_store(d.path());
    let bytes = std::fs::read(&p).unwrap();
    let (body, page) = split_root_page(&bytes);
    let segs = segments(&bytes);
    let (m_off, _, m_len) = *segs
        .iter()
        .rev()
        .find(|s| s.1 == SegmentType::Manifest as u8)
        .unwrap();
    // Copy the real manifest payload with one directory byte flipped.
    let mut payload = body[m_off + 64..m_off + 64 + m_len].to_vec();
    payload[22 + 8] ^= 0x01; // a directory offset
                             // Same algorithm as a genuine signed manifest (SHAKE-256-128), so the
                             // only thing wrong with the plant is that the page does not commit to it.
    let planted = rvf_wire::writer::write_segment_with_algo(
        SegmentType::Manifest as u8,
        &payload,
        rvf_types::SegmentFlags::empty(),
        500,
        2,
    );
    let mut out = body.to_vec();
    let planted_off = out.len();
    out.extend_from_slice(&planted);
    out.extend_from_slice(page.unwrap());
    std::fs::write(&p, &out).unwrap();

    // A newer parseable manifest beyond the signed page is an unsigned tail:
    // Strict refuses by default (F3-02) ...
    let e = security(open_err(&p, &[public(&SECRET)], SecurityPolicy::Strict));
    assert!(
        matches!(&e, SecurityError::ContentHashMismatch { pointer_name: "newer_manifest_beyond_signed_page", seg_offset, .. } if *seg_offset == planted_off as u64),
        "{e:?}"
    );
    // ... and serves the signed prefix only when the caller opts in; the
    // ignored manifest is reported, never used (F2-07: it also cannot hide
    // or replace the signed one).
    let mut o = opts(&[public(&SECRET)]);
    o.serve_signed_prefix = true;
    let s = RvfStore::open_with_policy(&p, o, SecurityPolicy::Strict).unwrap();
    assert_eq!(s.status().total_vectors, 12);
    assert_eq!(
        s.newer_manifest_ignored().map(|x| x.0),
        Some(planted_off as u64)
    );

    // Freshness pin: a caller that last accepted a newer epoch refuses this
    // state outright (the file's own manifest epoch is the monotone counter).
    let mut o = opts(&[public(&SECRET)]);
    o.serve_signed_prefix = true;
    o.freshness = Some(rvf_runtime::Freshness::MinEpoch(u32::MAX));
    assert!(matches!(
        RvfStore::open_with_policy(&p, o, SecurityPolicy::Strict),
        Err(rvf_types::RvfError::Code(
            rvf_types::ErrorCode::StreamStaleEpoch
        ))
    ));
}

#[test]
fn rewritten_manifest_under_an_intact_page_is_refused() {
    // Same-offset, same-length manifest rewrite with a recomputed header
    // hash: the signature still verifies, the committed L1 hash does not.
    let d = tempfile::tempdir().unwrap();
    let p = signed_store(d.path());
    let mut bytes = std::fs::read(&p).unwrap();
    let segs = segments(&bytes);
    let (m_off, _, m_len) = *segs
        .iter()
        .rev()
        .find(|s| s.1 == SegmentType::Manifest as u8)
        .unwrap();
    bytes[m_off + 64 + 22 + 8] ^= 0x01; // flip a directory offset byte
    let new_hash = compute_shake256_128(&bytes[m_off + 64..m_off + 64 + m_len]);
    bytes[m_off + 0x28..m_off + 0x38].copy_from_slice(&new_hash);
    std::fs::write(&p, &bytes).unwrap();
    let e = security(open_err(&p, &[public(&SECRET)], SecurityPolicy::Strict));
    assert!(
        matches!(
            e,
            SecurityError::ContentHashMismatch {
                pointer_name: "l1_manifest",
                ..
            }
        ),
        "{e:?}"
    );
}

#[test]
fn freshness_pin_accepts_the_pinned_epoch_and_refuses_older_state() {
    let d = tempfile::tempdir().unwrap();
    let p = signed_store(d.path());
    let epoch = RvfStore::open_with_policy(&p, opts(&[public(&SECRET)]), SecurityPolicy::Strict)
        .unwrap()
        .epoch();
    let mut o = opts(&[public(&SECRET)]);
    o.freshness = Some(rvf_runtime::Freshness::MinEpoch(epoch));
    assert!(RvfStore::open_with_policy(&p, o.clone(), SecurityPolicy::Strict).is_ok());
    o.freshness = Some(rvf_runtime::Freshness::MinEpoch(epoch + 1));
    for pol in [SecurityPolicy::Strict, SecurityPolicy::Paranoid] {
        assert!(matches!(
            RvfStore::open_with_policy(&p, o.clone(), pol),
            Err(rvf_types::RvfError::Code(
                rvf_types::ErrorCode::StreamStaleEpoch
            ))
        ));
    }
}
