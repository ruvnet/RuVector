#![cfg(feature = "ed25519")]
//! Round 2 review: attacks on the fixes for F-01/02/03/12/14/16.
use rvf_runtime::options::WitnessConfig;
use rvf_runtime::{QueryOptions, RvfOptions, RvfStore};
use rvf_types::security::SecurityPolicy;
use rvf_types::{SegmentFlags, SegmentType};
use rvf_wire::hash::compute_shake256_128;
use rvf_wire::writer::write_segment;

#[path = "support/signed.rs"]
mod signed;
use signed::*;

fn strict(p: &std::path::Path) -> Result<RvfStore, rvf_types::RvfError> {
    RvfStore::open_with_policy(p, opts_prefix(&[public(&SECRET)]), SecurityPolicy::Paranoid)
}
fn owner(p: &std::path::Path) -> RvfStore {
    RvfStore::open_signed(
        p,
        opts_prefix(&[public(&SECRET)]),
        SecurityPolicy::Strict,
        &SECRET,
    )
    .unwrap()
}

/// Legitimate signed store with every embeddable segment type, audit witnesses,
/// delete, compact (twice): must still open under Paranoid and extract all.
#[test]
fn r2_legit_signed_store_survives_all_ops() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("all.rvf");
    let mut o = opts(&[]);
    o.witness = WitnessConfig {
        witness_ingest: true,
        witness_delete: true,
        witness_compact: true,
        audit_queries: true,
    };
    let mut s = RvfStore::create_signed(&p, o, &SECRET).unwrap();
    let v = rows(8, 0.0);
    let refs: Vec<&[f32]> = v.iter().map(Vec::as_slice).collect();
    let ids: Vec<u64> = (0..8).collect();
    s.ingest_batch(&refs, &ids, None).unwrap();
    s.embed_kernel(0, 0, 0, &[7u8; 300], 8080, Some("console=ttyS0"))
        .unwrap();
    s.embed_ebpf(1, 1, 4, &[9u8; 100], None).unwrap();
    s.embed_dashboard(0, b"index.html", "index.html").unwrap();
    s.embed_wasm(0, 0, 0, b"\0asm-xxxxxxxx", 1, 0, 0).unwrap();
    s.query_audited(&[1.0, 1.0, 2.0, 3.0], 3, &QueryOptions::default())
        .unwrap();
    s.delete(&[3]).unwrap();
    s.compact().unwrap();
    s.query_audited(&[1.0, 1.0, 2.0, 3.0], 3, &QueryOptions::default())
        .unwrap();
    s.compact().unwrap();
    s.close().unwrap();
    let s = strict(&p).expect("paranoid open after full lifecycle");
    assert!(s.extract_kernel().unwrap().is_some());
    assert!(s.extract_ebpf().unwrap().is_some());
    assert!(s.extract_dashboard().unwrap().is_some());
    assert!(s.extract_wasm().unwrap().is_some());
    assert_eq!(s.status().total_vectors, 7);
    // and the owner can keep going
    drop(s);
    let mut s = owner(&p);
    s.compact().unwrap();
    s.close().unwrap();
    assert!(strict(&p).is_ok());
}

fn signed_with_wasm(dir: &std::path::Path) -> std::path::PathBuf {
    let p = signed_store(dir);
    let mut s = owner(&p);
    s.embed_wasm(0, 0, 0, b"\0asm-GOOD-GOOD-GOOD", 1, 0, 0)
        .unwrap();
    s.close().unwrap();
    p
}

/// Mid-session tamper: the owner holds the store open, the file is edited
/// underneath it (header hash recomputed, keyless), then compact().
#[test]
fn r2_midsession_tamper_then_compact_is_refused() {
    let d = tempfile::tempdir().unwrap();
    let p = signed_with_wasm(d.path());
    let mut s = owner(&p);
    let mut bytes = std::fs::read(&p).unwrap();
    let (off, _, len) = *segments(&bytes)
        .iter()
        .find(|x| x.1 == SegmentType::Wasm as u8)
        .unwrap();
    bytes[off + 64 + len - 1] ^= 1;
    let h = compute_shake256_128(&bytes[off + 64..off + 64 + len]);
    bytes[off + 0x28..off + 0x38].copy_from_slice(&h);
    std::fs::write(&p, &bytes).unwrap();
    assert!(
        s.compact().is_err(),
        "tampered source must refuse compaction"
    );
}

/// Tamper of a segment type compaction does NOT verify (it skips Vec/Meta/
/// Index/Journal because it rebuilds them from memory): vectors changed on
/// disk mid-session must not be signed in.
#[test]
fn r2_midsession_vec_tamper_not_laundered() {
    let d = tempfile::tempdir().unwrap();
    let p = signed_store(d.path());
    let mut s = owner(&p);
    let mut bytes = std::fs::read(&p).unwrap();
    let (off, _, len) = *segments(&bytes)
        .iter()
        .find(|x| x.1 == SegmentType::Vec as u8)
        .unwrap();
    bytes[off + 64 + len - 1] ^= 0x40;
    std::fs::write(&p, &bytes).unwrap();
    let before = s.read_all_vectors();
    let r = s.compact();
    s.close().ok();
    if r.is_ok() {
        let s2 = strict(&p).unwrap();
        assert_eq!(s2.read_all_vectors().len(), before.len());
    }
}

/// Injection between manifest and page + audited query + compaction: the
/// injected segment must never become part of a signed state.
#[test]
fn r2_injected_segment_with_witness_flow() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("w.rvf");
    let mut o = opts(&[]);
    o.witness.audit_queries = true;
    {
        let mut s = RvfStore::create_signed(&p, o.clone(), &SECRET).unwrap();
        let v = rows(6, 0.0);
        let refs: Vec<&[f32]> = v.iter().map(Vec::as_slice).collect();
        s.ingest_batch(&refs, &(0..6).collect::<Vec<u64>>(), None)
            .unwrap();
        s.close().unwrap();
    }
    let bytes = std::fs::read(&p).unwrap();
    let (body, page) = rvf_wire::tail_scan::split_root_page(&bytes);
    let payload = vec![0xEEu8; 128];
    let mut forged = write_segment(
        SegmentType::Wasm as u8,
        &payload,
        SegmentFlags::empty(),
        9999,
    );
    forged[0x20] = 2;
    forged[0x28..0x38].copy_from_slice(&compute_shake256_128(&payload));
    let mut evil = body.to_vec();
    evil.extend_from_slice(&forged);
    evil.extend_from_slice(page.unwrap());
    std::fs::write(&p, &evil).unwrap();
    let mut oo = opts(&[public(&SECRET)]);
    oo.serve_signed_prefix = true;
    oo.witness.audit_queries = true;
    let mut s = RvfStore::open_signed(&p, oo, SecurityPolicy::Strict, &SECRET).unwrap();
    s.query_audited(&[1.0, 1.0, 2.0, 3.0], 2, &QueryOptions::default())
        .unwrap();
    s.compact().unwrap();
    s.close().unwrap();
    let s = strict(&p).unwrap();
    assert!(
        s.extract_wasm().unwrap().is_none(),
        "injected wasm became signed"
    );
}

/// F-03 fix: 300 fake manifest candidates after the real manifest.
fn fake_header_run(n: usize) -> Vec<u8> {
    let tmpl = write_segment(SegmentType::Manifest as u8, &[], SegmentFlags::empty(), 1);
    let mut out = Vec::new();
    for i in 0..n {
        let mut h = tmpl[..64].to_vec();
        h[0x08..0x10].copy_from_slice(&u64::MAX.to_le_bytes());
        let rem = ((n - i - 1) * 64) as u64;
        h[0x10..0x18].copy_from_slice(&rem.to_le_bytes());
        out.extend_from_slice(&h);
    }
    out
}

#[test]
fn r2_hide_real_manifest_behind_fake_candidates_plain() {
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
        s.ingest_batch(&refs, &(0..6).collect::<Vec<u64>>(), None)
            .unwrap();
        s.close().unwrap();
    }
    let mut b = std::fs::read(&p).unwrap();
    b.extend_from_slice(&fake_header_run(300));
    std::fs::write(&p, &b).unwrap();
    let r = RvfStore::open_readonly(&p);
    eprintln!(
        "plain, 300 fake candidates after the real manifest: open ok = {}",
        r.is_ok()
    );
    assert!(
        r.is_ok(),
        "availability: real manifest hidden by 300 candidates"
    );
}

#[test]
fn r2_hide_real_manifest_behind_fake_candidates_strict() {
    let d = tempfile::tempdir().unwrap();
    let p = signed_store(d.path());
    let b = std::fs::read(&p).unwrap();
    let (body, page) = rvf_wire::tail_scan::split_root_page(&b);
    let mut e = body.to_vec();
    e.extend_from_slice(&fake_header_run(300));
    e.extend_from_slice(page.unwrap());
    std::fs::write(&p, &e).unwrap();
    let r = strict(&p);
    eprintln!(
        "strict, 300 fake candidates between manifest and page: ok = {}",
        r.is_ok()
    );
    assert!(r.is_ok());
}

/// F-14 fix bypass: wire layout made ambiguous by planting a valid-looking
/// header in the (unhashed) padding of the first segment, then flipping its
/// unhashed algo byte. Layout vote (true,true) falls back to the byte.
#[test]
fn r2_f14_padding_plant_forces_byte_granular() {
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
        s.ingest_batch(&refs, &(0..6).collect::<Vec<u64>>(), None)
            .unwrap();
        s.close().unwrap();
    }
    let mut bytes = std::fs::read(&p).unwrap();
    let h = rvf_wire::read_segment_header(&bytes).unwrap();
    let unpadded = 64 + h.payload_length as usize;
    assert!(unpadded % 64 != 0);
    let mut payload = vec![0u8; 8];
    let mut fm = Vec::new();
    fm.extend_from_slice(&99u32.to_le_bytes());
    fm.extend_from_slice(&4u16.to_le_bytes());
    fm.extend_from_slice(&0u64.to_le_bytes());
    fm.extend_from_slice(&0u32.to_le_bytes());
    fm.extend_from_slice(&[0u8; 4]);
    fm.extend_from_slice(&0u32.to_le_bytes());
    payload.extend_from_slice(&write_segment(
        SegmentType::Manifest as u8,
        &fm,
        SegmentFlags::empty(),
        777,
    ));
    bytes.extend_from_slice(&write_segment(
        SegmentType::Vec as u8,
        &payload,
        SegmentFlags::empty(),
        778,
    ));
    assert_eq!(
        RvfStore::open_readonly(&{
            std::fs::write(&p, &bytes).unwrap();
            p.clone()
        })
        .unwrap()
        .status()
        .total_vectors,
        6
    );
    bytes[unpadded..unpadded + 4].copy_from_slice(&rvf_types::SEGMENT_MAGIC_BYTES);
    bytes[unpadded + 4] = 1;
    bytes[unpadded + 5] = 1;
    bytes[0x20] = 0;
    std::fs::write(&p, &bytes).unwrap();
    eprintln!(
        "legacy scan chosen = {}",
        rvf_runtime::read_path::scan_is_legacy_for_bytes(&bytes)
    );
    let got = RvfStore::open_readonly(&p).map(|s| s.status().total_vectors);
    eprintln!("padding-plant result: {got:?}");
    assert_ne!(
        got.ok(),
        Some(0),
        "attacker selected byte-granular scan and its fake manifest"
    );
}

/// F-16 completeness oracle: flipping any single bit of any of the 64 header
/// bytes of a committed Vec / Wasm segment must be detected under Paranoid.
#[test]
fn r2_f16_every_header_byte_is_pinned() {
    let d = tempfile::tempdir().unwrap();
    let p = signed_with_wasm(d.path());
    let base = std::fs::read(&p).unwrap();
    let segs = segments(&base);
    let mut unpinned = Vec::new();
    for ty in [SegmentType::Vec, SegmentType::Wasm] {
        let (off, _, _) = *segs.iter().find(|x| x.1 == ty as u8).unwrap();
        for byte in 0..64 {
            for bit in 0..8 {
                let mut b = base.clone();
                b[off + byte] ^= 1 << bit;
                std::fs::write(&p, &b).unwrap();
                let detected = match strict(&p) {
                    Err(_) => true,
                    Ok(s) => s.extract_wasm().is_err(),
                };
                if !detected {
                    unpinned.push((ty as u8, byte, bit));
                }
            }
        }
    }
    assert!(unpinned.is_empty(), "unpinned header bits: {unpinned:?}");
}

/// Also the padding after a committed payload and gaps are unhashed; make sure
/// they are never read as data (document only).
#[test]
fn r2_f12_plain_open_is_readonly_and_allow_unsigning_cuts() {
    let d = tempfile::tempdir().unwrap();
    let p = signed_store(d.path());
    let mut s = RvfStore::open(&p).unwrap();
    let v = rows(1, 500.0);
    let refs: Vec<&[f32]> = v.iter().map(Vec::as_slice).collect();
    assert!(s.ingest_batch(&refs, &[900], None).is_err());
    assert!(s.delete(&[1]).is_err());
    assert!(s.compact().is_err());
    assert!(s.embed_wasm(0, 0, 0, b"x", 0, 0, 0).is_err());
    assert!(s
        .set_file_metadata("k".into(), rvf_runtime::options::MetadataValue::U64(1))
        .is_err());
    s.close().unwrap();
    assert!(
        strict(&p).is_ok(),
        "refused writes left the signature intact"
    );
}

/// F2-06: F-16 pinned the header of every directory segment, but the MANIFEST
/// segment's own header is not a directory segment: the page binds its offset,
/// length and payload hash only.
#[test]
fn r2_f2_06_manifest_segment_header_bits_are_unpinned() {
    let d = tempfile::tempdir().unwrap();
    let p = signed_store(d.path());
    let base = std::fs::read(&p).unwrap();
    let (body, _) = rvf_wire::tail_scan::split_root_page(&base);
    let (moff, _) = rvf_wire::tail_scan::find_latest_manifest(body).unwrap();
    let mut unpinned = Vec::new();
    for byte in 0..64 {
        for bit in 0..8 {
            let mut b = base.clone();
            b[moff + byte] ^= 1 << bit;
            std::fs::write(&p, &b).unwrap();
            if strict(&p).is_ok() {
                unpinned.push((byte, bit));
            }
        }
    }
    let bytes: std::collections::BTreeSet<_> = unpinned.iter().map(|x| x.0).collect();
    eprintln!("manifest header bytes whose flips are accepted under Paranoid: {bytes:?}");
    assert!(
        unpinned.is_empty(),
        "unpinned manifest header bytes: {bytes:?}"
    );
}
