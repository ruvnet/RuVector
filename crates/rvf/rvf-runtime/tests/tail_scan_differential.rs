//! ADR-354 item 4: one tail scanner. `rvf-runtime` locates the manifest with
//! `rvf-wire`'s scanner; these tests assert the two code paths agree on the
//! same bytes, including adversarial ones (a fake `RVFS` + MANIFEST header
//! embedded in an earlier payload, or in a payload that trails the real
//! manifest).

#[path = "support/differential.rs"]
mod differential;

use rvf_runtime::{RvfOptions, RvfStore};
use rvf_types::{SegmentFlags, SegmentType};
use rvf_wire::tail_scan::{find_latest_manifest, find_latest_manifest_with_mode, ScanMode};
use rvf_wire::write_segment;

/// A runtime-parseable manifest payload (dim 4, no vectors, no directory).
fn manifest_payload(epoch: u32, dim: u16) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&epoch.to_le_bytes());
    p.extend_from_slice(&dim.to_le_bytes());
    p.extend_from_slice(&0u64.to_le_bytes()); // total vectors
    p.extend_from_slice(&0u32.to_le_bytes()); // seg_count
    p.extend_from_slice(&[0, 0, 0, 0]); // profile, metric, reserved
    p.extend_from_slice(&0u32.to_le_bytes()); // deleted count
    p
}

fn seg(ty: SegmentType, id: u64, payload: &[u8]) -> Vec<u8> {
    write_segment(ty as u8, payload, SegmentFlags::empty(), id)
}

/// A real wire-dialect store file, as RvfStore writes it today.
fn real_store_bytes() -> Vec<u8> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("diff.rvf");
    let o = RvfOptions {
        dimension: 4,
        ..Default::default()
    };
    let mut s = RvfStore::create(&path, o).unwrap();
    let v: Vec<Vec<f32>> = (0..8).map(|i| vec![i as f32; 4]).collect();
    let refs: Vec<&[f32]> = v.iter().map(|x| x.as_slice()).collect();
    s.ingest_batch(&refs, &(0..8).collect::<Vec<u64>>(), None)
        .unwrap();
    s.close().unwrap();
    std::fs::read(&path).unwrap()
}

#[test]
fn real_runtime_file_agrees_with_wire_scanner() {
    let bytes = real_store_bytes();
    let got = differential::check(&bytes).unwrap().unwrap();
    let (w_off, _) = find_latest_manifest(&bytes).unwrap();
    assert_eq!(got, (w_off as u64, false), "aligned, not legacy");
}

#[test]
fn fake_manifest_inside_an_earlier_payload_is_never_chosen() {
    // The fake sits at an UNALIGNED offset inside a VEC payload, before the
    // real manifest, and carries a parseable payload with a different epoch.
    let fake = seg(SegmentType::Manifest, 99, &manifest_payload(77, 9));
    let mut vec_payload = vec![0xAB; 21];
    vec_payload.extend_from_slice(&fake);
    let mut file = seg(SegmentType::Vec, 1, &vec_payload);
    let real_off = file.len();
    file.extend_from_slice(&seg(SegmentType::Manifest, 2, &manifest_payload(5, 4)));

    let got = differential::check(&file).unwrap().unwrap();
    assert_eq!(got, (real_off as u64, false));
}

#[test]
fn unaligned_fake_in_a_trailing_payload_loses_to_the_real_manifest() {
    // compact() appends a witness AFTER the manifest, so a payload that
    // trails the real manifest is the realistic attacker position.
    let mut file = seg(SegmentType::Vec, 1, &[1u8; 40]);
    let real_off = file.len();
    file.extend_from_slice(&seg(SegmentType::Manifest, 2, &manifest_payload(5, 4)));
    let fake = seg(SegmentType::Manifest, 99, &manifest_payload(77, 9));
    let mut witness_payload = vec![0xCD; 13]; // makes the fake unaligned
    witness_payload.extend_from_slice(&fake);
    file.extend_from_slice(&seg(SegmentType::Witness, 3, &witness_payload));

    // The old byte-granular scan (what rvf-runtime did before ADR-354) would
    // pick the fake: it is newer and parses.
    let (byte_off, _) = find_latest_manifest_with_mode(&file, ScanMode::ByteGranular).unwrap();
    assert!(
        byte_off > real_off,
        "legacy scan prefers the planted header"
    );
    // The wire scanner and the runtime both ignore it.
    let got = differential::check(&file).unwrap().unwrap();
    assert_eq!(got, (real_off as u64, false));
    assert_eq!(find_latest_manifest(&file).unwrap().0, real_off);
}

#[test]
fn aligned_fake_in_a_trailing_payload_is_a_known_limit_both_paths_agree() {
    // An attacker who controls a trailing payload can plant an ALIGNED header
    // (payload starts at +64, so a multiple of 64 is reachable). Scanning
    // cannot authenticate; both paths pick it. `Strict` refuses it because
    // the signed root page does not commit to it (see strict_policy.rs).
    let mut file = seg(SegmentType::Vec, 1, &[1u8; 40]);
    file.extend_from_slice(&seg(SegmentType::Manifest, 2, &manifest_payload(5, 4)));
    let fake = seg(SegmentType::Manifest, 99, &manifest_payload(77, 9));
    let witness_payload = fake.clone(); // payload begins at a 64-aligned offset
    let trailer_off = file.len();
    file.extend_from_slice(&seg(SegmentType::Witness, 3, &witness_payload));

    let got = differential::check(&file).unwrap().unwrap();
    assert_eq!(got, ((trailer_off + 64) as u64, false));
    assert_eq!(find_latest_manifest(&file).unwrap().0, trailer_off + 64);
}

#[test]
fn legacy_byte_granular_mode_finds_unpadded_manifests() {
    // Unpadded legacy-style packing, first segment stamped algo 0.
    let mut a = seg(SegmentType::Vec, 1, &[1u8; 40]);
    a[0x20] = 0; // checksum_algo = 0 (legacy label)
    a.truncate(64 + 40); // no tail padding
    let mut file = a;
    let m_off = file.len();
    let mut m = seg(SegmentType::Manifest, 2, &manifest_payload(1, 4));
    m.truncate(64 + manifest_payload(1, 4).len());
    file.extend_from_slice(&m);
    assert_ne!(m_off % 64, 0);
    let got = differential::check(&file).unwrap().unwrap();
    assert_eq!(got, (m_off as u64, true));
}

#[test]
fn trailing_root_page_does_not_hide_or_replace_the_manifest() {
    let mut file = seg(SegmentType::Vec, 1, &[1u8; 40]);
    let m_off = file.len();
    file.extend_from_slice(&seg(SegmentType::Manifest, 2, &manifest_payload(5, 4)));
    let mut page = vec![0u8; 4096];
    page[..4].copy_from_slice(&rvf_types::ROOT_MANIFEST_MAGIC.to_le_bytes());
    // plant an aligned fake manifest inside the page
    let fake = seg(SegmentType::Manifest, 99, &manifest_payload(77, 9));
    page[128..128 + fake.len()].copy_from_slice(&fake);
    file.extend_from_slice(&page);
    let got = differential::check(&file).unwrap().unwrap();
    assert_eq!(got, (m_off as u64, false));
}

#[test]
fn deterministic_mutation_sweep_never_disagrees() {
    // Cheap stand-in for the fuzz target: flip/insert bytes around a real
    // file and require the invariants to hold every time.
    let base = real_store_bytes();
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    for _ in 0..3000 {
        let mut d = base.clone();
        for _ in 0..(1 + next() % 4) {
            let i = (next() as usize) % d.len();
            match next() % 3 {
                0 => d[i] ^= 1 << (next() % 8),
                1 => d[i] = next() as u8,
                _ => {
                    d.insert(i, next() as u8);
                }
            }
        }
        if next() % 5 == 0 {
            d.truncate((next() as usize) % d.len());
        }
        differential::check(&d).unwrap_or_else(|e| panic!("{e}"));
    }
}

/// Found by the cargo-fuzz target: a first header whose payload_length is near
/// u64::MAX overflowed the layout probe's arithmetic (panic with overflow
/// checks, silent wrap in release). Must neither panic nor disagree.
#[test]
fn fuzz_regression_huge_payload_length_in_the_layout_probe() {
    let bytes: [u8; 64] = [
        0x53, 0x46, 0x56, 0x52, 0x01, 0x05, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xff, 0xed, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x37, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xbf, 0xff, 0xff, 0xff, 0xbf, 0xff, 0xff, 0xff, 0xff, 0xff,
        0x00, 0x00, 0x3b, 0x00, 0xfb, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x32, 0x12,
        0x00, 0xdc, 0x00, 0xae,
    ];
    differential::check(&bytes).unwrap();
}
