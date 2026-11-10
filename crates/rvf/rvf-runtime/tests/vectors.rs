#![cfg(feature = "ed25519")]
//! ADR-354 item 6 (ruvector side): shared byte fixtures in
//! `crates/rvf/tests/vectors/`, consumed by the rvm repo.
//!
//! * `legacy/*.rvf` were written by the pre-ADR-354 writer and are frozen.
//! * `runtime_wire_dialect.rvf` / `runtime_signed_wire.rvf` are written by
//!   `RvfStore` today. `RvfStore` is not byte-deterministic (the file id mixes
//!   the path and the clock), so regeneration is compared STRUCTURALLY:
//!   segment sequence (type, id, algo, payload length, padding), every
//!   non-manifest payload byte-for-byte, manifest payloads with the 68-byte
//!   FileIdentity masked.
//!
//! Set `OFER_REGEN_VECTORS=1` to rewrite the two RvfStore fixtures.

use rvf_runtime::options::{DistanceMetric, WitnessConfig};
use rvf_runtime::{QueryOptions, RvfOptions, RvfStore};
use rvf_types::security::SecurityPolicy;
use rvf_types::{SegmentHeader, SEGMENT_HEADER_SIZE};
use rvf_wire::tail_scan::split_root_page;
use rvf_wire::writer::calculate_padded_size;
use std::path::PathBuf;

const SIGNING_SECRET: [u8; 32] = [0x42; 32];

fn vectors_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/vectors")
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

fn options() -> RvfOptions {
    RvfOptions {
        dimension: 8,
        metric: DistanceMetric::L2,
        // No witness segments: they embed the wall clock.
        witness: WitnessConfig {
            witness_ingest: false,
            witness_delete: false,
            witness_compact: false,
            audit_queries: false,
        },
        ..Default::default()
    }
}

fn fill(s: &mut RvfStore) {
    for b in 0..2u64 {
        let v = vecs(10, 8, 100 + b);
        let refs: Vec<&[f32]> = v.iter().map(|x| x.as_slice()).collect();
        let ids: Vec<u64> = (0..10).map(|i| b * 10 + i).collect();
        s.ingest_batch(&refs, &ids, None).unwrap();
    }
}

fn gen_wire(path: &std::path::Path) {
    let _ = std::fs::remove_file(path);
    let mut s = RvfStore::create(path, options()).unwrap();
    fill(&mut s);
    s.close().unwrap();
}

fn gen_signed(path: &std::path::Path) {
    let _ = std::fs::remove_file(path);
    let mut s = RvfStore::create_signed(path, options(), &SIGNING_SECRET).unwrap();
    fill(&mut s);
    s.delete(&[3]).unwrap();
    s.close().unwrap();
}

fn public_key() -> [u8; 32] {
    rvf_types::ed25519::Ed25519Keypair::from_secret(&SIGNING_SECRET).public_key()
}

struct Seg {
    header: SegmentHeader,
    payload: Vec<u8>,
}

/// Walk a wire-dialect file at padded strides (what rvm-rvf does).
fn walk(bytes: &[u8]) -> Vec<Seg> {
    let (body, _page) = split_root_page(bytes);
    let mut off = 0;
    let mut out = Vec::new();
    while off < body.len() {
        let header = rvf_wire::read_segment_header(&body[off..]).expect("header");
        let plen = header.payload_length as usize;
        let payload = body[off + SEGMENT_HEADER_SIZE..off + SEGMENT_HEADER_SIZE + plen].to_vec();
        rvf_wire::validate_segment(&header, &payload).expect("hash");
        off += calculate_padded_size(SEGMENT_HEADER_SIZE, plen);
        out.push(Seg { header, payload });
    }
    assert_eq!(off, body.len(), "segments tile the file exactly");
    out
}

/// Mask what depends on the (random) file id: the 68-byte FileIdentity after
/// the "FIDI" marker, and, in a signed manifest's hash trailer, the hashes of
/// earlier MANIFEST segments (their payloads embed the file id).
fn mask_manifest(mut payload: Vec<u8>, manifest_ids: &[u64]) -> Vec<u8> {
    let marker = 0x4649_4449u32.to_le_bytes();
    let n = u32::from_le_bytes(payload[14..18].try_into().unwrap()) as usize;
    let del = 22 + n * 25;
    let ndel = u32::from_le_bytes(payload[del..del + 4].try_into().unwrap()) as usize;
    let at = del + 4 + ndel * 8;
    assert_eq!(payload[at..at + 4], marker, "FIDI marker");
    payload[at + 4..at + 4 + 68].fill(0);
    let t = at + 72;
    if payload.len() > t {
        assert_eq!(
            payload[t..t + 4],
            0x4744_4853u32.to_le_bytes(),
            "SHDG trailer"
        );
        let count = u32::from_le_bytes(payload[t + 4..t + 8].try_into().unwrap()) as usize;
        for i in 0..count {
            let e = t + 8 + i * 24;
            let id = u64::from_le_bytes(payload[e..e + 8].try_into().unwrap());
            if manifest_ids.contains(&id) {
                payload[e + 8..e + 24].fill(0);
            }
        }
    }
    payload
}

fn assert_structurally_equal(committed: &[u8], fresh: &[u8]) {
    assert_eq!(committed.len(), fresh.len(), "file length");
    let (a, b) = (walk(committed), walk(fresh));
    assert_eq!(a.len(), b.len());
    let ids = |segs: &[Seg]| -> Vec<u64> {
        segs.iter()
            .filter(|s| s.header.seg_type == rvf_types::SegmentType::Manifest as u8)
            .map(|s| s.header.segment_id)
            .collect()
    };
    let (ids_a, ids_b) = (ids(&a), ids(&b));
    assert_eq!(ids_a, ids_b);
    for (x, y) in a.iter().zip(&b) {
        let (hx, hy) = (&x.header, &y.header);
        assert_eq!(
            (
                hx.seg_type,
                hx.segment_id,
                hx.checksum_algo,
                hx.payload_length,
                hx.alignment_pad
            ),
            (
                hy.seg_type,
                hy.segment_id,
                hy.checksum_algo,
                hy.payload_length,
                hy.alignment_pad
            )
        );
        if hx.seg_type == rvf_types::SegmentType::Manifest as u8 {
            // The manifest hash covers the (random) file id: masked compare.
            assert_eq!(
                mask_manifest(x.payload.clone(), &ids_a),
                mask_manifest(y.payload.clone(), &ids_b)
            );
        } else {
            assert_eq!(x.payload, y.payload);
            assert_eq!(hx.content_hash, hy.content_hash);
        }
    }
}

fn maybe_regen(name: &str, fresh: &std::path::Path) {
    if std::env::var_os("OFER_REGEN_VECTORS").is_some() {
        std::fs::copy(fresh, vectors_dir().join(name)).unwrap();
    }
}

#[test]
fn runtime_wire_dialect_fixture_regenerates_structurally() {
    let dir = tempfile::tempdir().unwrap();
    let fresh = dir.path().join("w.rvf");
    gen_wire(&fresh);
    maybe_regen("runtime_wire_dialect.rvf", &fresh);
    let committed = std::fs::read(vectors_dir().join("runtime_wire_dialect.rvf")).unwrap();
    assert_structurally_equal(&committed, &std::fs::read(&fresh).unwrap());
    assert_eq!(committed.len() % 64, 0);
    assert!(walk(&committed).iter().all(|s| s.header.checksum_algo == 1));
    // Opens and queries like any store, and is not flagged legacy.
    let s = RvfStore::open_readonly(&vectors_dir().join("runtime_wire_dialect.rvf")).unwrap();
    assert!(!s.legacy_dialect());
    assert_eq!(s.status().total_vectors, 20);
    let q = vecs(1, 8, 7).remove(0);
    assert_eq!(s.query(&q, 3, &QueryOptions::default()).unwrap().len(), 3);
}

#[test]
fn runtime_signed_fixture_regenerates_and_opens_under_strict() {
    let dir = tempfile::tempdir().unwrap();
    let fresh = dir.path().join("s.rvf");
    gen_signed(&fresh);
    maybe_regen("runtime_signed_wire.rvf", &fresh);
    let committed = std::fs::read(vectors_dir().join("runtime_signed_wire.rvf")).unwrap();
    assert_structurally_equal(&committed, &std::fs::read(&fresh).unwrap());
    let (_, page) = split_root_page(&committed);
    assert_eq!(
        page.map(<[u8]>::len),
        Some(4096),
        "ends in the RVM0 root page"
    );
    assert!(walk(&committed).iter().all(|s| s.header.checksum_algo == 2));

    let o = RvfOptions {
        trusted_signers: vec![public_key()],
        freshness: Some(rvf_runtime::Freshness::Unchecked),
        ..Default::default()
    };
    let s = RvfStore::open_with_policy(
        &vectors_dir().join("runtime_signed_wire.rvf"),
        o,
        SecurityPolicy::Strict,
    )
    .unwrap();
    assert_eq!(s.status().total_vectors, 19);
}

#[test]
fn committed_fixtures_are_never_written_to_by_opening_them() {
    // Guard for the tests above: they open fixtures in place read-only.
    let before = std::fs::read(vectors_dir().join("runtime_wire_dialect.rvf")).unwrap();
    let _ = RvfStore::open_readonly(&vectors_dir().join("runtime_wire_dialect.rvf")).unwrap();
    let after = std::fs::read(vectors_dir().join("runtime_wire_dialect.rvf")).unwrap();
    assert_eq!(before, after);
}
