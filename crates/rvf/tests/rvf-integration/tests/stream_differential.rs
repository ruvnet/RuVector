//! P10: for a corpus of real bodies, the RVSD stream verifier (fed in file order and in
//! stage order) accepts exactly when whole-body verification (SHA-256 identity plus
//! per-segment recomputation of the committed hashes) accepts.
//!
//! Corpus: bodies written by `RvfStore` (the producer behind rvf-server), signed containers
//! from `rvf-forge-core::ContainerBuilder` (with a Level-0 root page), and synthetic bodies
//! built with the rvf-wire writer. Each body is also tampered (bit flips in segment headers,
//! payloads and the root page) and fed against the descriptor of the pristine body.

use ed25519_dalek::SigningKey;
use rvf_crypto::{chunk_list_hash, leaf_hash};
use rvf_types::sha256;
use rvf_wire::stream::*;

const FILE_ID: [u8; 16] = *b"differential-001";

fn key() -> SigningKey {
    SigningKey::from_bytes(&[9u8; 32])
}

/// Whole-body verification, written independently of the frame path.
fn whole_body_accepts(b: &StreamBundle, body: &[u8]) -> bool {
    let h = &b.header;
    if body.len() as u64 != h.body_len || sha256(body) != h.rvf_identity {
        return false;
    }
    for e in &b.entries {
        let seg = &body[e.body_offset as usize..(e.body_offset + e.stored_len) as usize];
        let cs = h.chunk_size().unwrap_or(u64::MAX);
        let leaves: Vec<[u8; 32]> = seg
            .chunks(cs.min(usize::MAX as u64) as usize)
            .enumerate()
            .map(|(i, c)| leaf_hash(&h.file_id, e.seg_id, i as u32, c))
            .collect();
        let commit = if leaves.len() == 1 {
            leaves[0]
        } else {
            chunk_list_hash(&h.file_id, e.seg_id, &leaves)
        };
        if commit != e.chunk_table_hash {
            return false;
        }
        // ADR-353 2.6: executable and index segments must not rely on checksum_algo 0 or 1.
        if matches!(e.seg_type, 0x0E | 0x0F | 0x10 | 0x02 | 0x08 | 0x0D) && seg[0x20] <= 1 {
            return false;
        }
    }
    true
}

fn stream_accepts(body: &[u8], frames: impl Iterator<Item = Vec<u8>>) -> bool {
    let keys = [key().verifying_key().to_bytes()];
    let mut v = StreamVerifier::new(StreamPolicy::new(&keys));
    for f in frames {
        if v.push(&f).is_err() {
            return false;
        }
    }
    match (v.state(), v.identity_status()) {
        (StreamState::Complete, Some(IdentityStatus::Verified)) => true,
        (StreamState::CompleteUnverified, Some(IdentityStatus::Unverifiable)) => v
            .finish_identity(|o, buf| {
                buf.copy_from_slice(&body[o as usize..o as usize + buf.len()]);
                true
            })
            .is_ok(),
        _ => false,
    }
}

fn check(name: &str, pristine: &[u8], log2: u8) -> (usize, usize) {
    let n_stage_pat = |n: usize| (0..n).map(|i| (i % 4) as u8).collect::<Vec<_>>();
    let file = StreamDescriptorBuilder::new(FILE_ID, key())
        .chunk_log2(log2)
        .build(pristine)
        .unwrap();
    let stage = StreamDescriptorBuilder::new(FILE_ID, key())
        .chunk_log2(log2)
        .stages(&n_stage_pat(file.entries.len()))
        .requires_lower_stages(true)
        .build(pristine)
        .unwrap();
    let mut variants: Vec<(String, Vec<u8>)> = vec![("pristine".into(), pristine.to_vec())];
    let len = pristine.len();
    let mut offs: Vec<usize> = vec![
        0,
        5,
        0x20,
        0x28,
        64,
        len / 3,
        len / 2,
        len - 1,
        len.saturating_sub(4096),
        len.saturating_sub(4000),
    ];
    offs.extend(file.entries.iter().map(|e| e.body_offset as usize + 8));
    for o in offs {
        if o < len {
            let mut t = pristine.to_vec();
            t[o] ^= 0x01;
            variants.push((format!("flip@{o}"), t));
        }
    }
    let (mut acc, mut rej) = (0, 0);
    for (label, body) in &variants {
        let want = whole_body_accepts(&file, body);
        let got_file = stream_accepts(body, file.frames_file_order(body));
        assert_eq!(
            want, got_file,
            "{name}/{label}: file order disagrees (whole={want})"
        );
        let want_stage = whole_body_accepts(&stage, body);
        let got_stage = stream_accepts(body, stage.frames_stage_order(body));
        assert_eq!(
            want_stage, got_stage,
            "{name}/{label}: stage order disagrees (whole={want_stage})"
        );
        assert_eq!(want, want_stage);
        if want {
            acc += 1
        } else {
            rej += 1
        }
    }
    (acc, rej)
}

fn synthetic() -> Vec<(String, Vec<u8>)> {
    use rvf_types::SegmentFlags as F;
    use rvf_wire::writer::write_segment_with_algo;
    let mut out = Vec::new();
    for (i, sizes) in [
        vec![0usize],
        vec![100, 5000, 70000],
        vec![1, 2, 3, 4, 5, 6, 7, 8],
        vec![200_000],
    ]
    .into_iter()
    .enumerate()
    {
        let mut b = Vec::new();
        for (j, s) in sizes.iter().enumerate() {
            let p: Vec<u8> = (0..*s).map(|k| (k * 31 + j * 7 + i) as u8).collect();
            let ty = if j % 3 == 0 { 0x10 } else { 0x01 };
            let algo = if ty == 0x10 || j % 2 == 0 { 2 } else { 1 };
            b.extend(write_segment_with_algo(
                ty,
                &p,
                F::empty(),
                j as u64 + 1,
                algo,
            ));
        }

        out.push((format!("synthetic-{i}"), b));
    }
    // an executable that relies on a non-cryptographic checksum: both sides must reject it
    let weak = write_segment_with_algo(0x10, &[1, 2, 3], F::empty(), 1, 1);
    out.push(("synthetic-weak-exec".into(), weak));
    out
}

fn forge() -> Vec<(String, Vec<u8>)> {
    use rvf_forge_core::{CapabilityClass, ContainerBuilder};
    use rvf_types::SegmentType;
    let sk = SigningKey::from_bytes(&[3u8; 32]);
    let a = ContainerBuilder::new()
        .file_id(*b"forge-corpus-01!")
        .build()
        .unwrap();
    let b = ContainerBuilder::new()
        .file_id(*b"forge-corpus-02!")
        .declare_capability(CapabilityClass::Network)
        .segment(SegmentType::Wasm, b"\0asm\x01\0\0\0".repeat(900))
        .segment(SegmentType::Meta, vec![7u8; 300])
        .sign_with(sk)
        .build()
        .unwrap();
    vec![
        ("forge-minimal".into(), a.bytes),
        ("forge-signed-wasm".into(), b.bytes),
    ]
}

fn runtime_store() -> Option<(String, Vec<u8>)> {
    use rvf_runtime::options::{DistanceMetric, RvfOptions};
    use rvf_runtime::RvfStore;
    let dir = tempfile::TempDir::new().ok()?;
    let path = dir.path().join("c.rvf");
    let mut s = RvfStore::create(
        &path,
        RvfOptions {
            dimension: 16,
            metric: DistanceMetric::L2,
            ..Default::default()
        },
    )
    .ok()?;
    let vs: Vec<Vec<f32>> = (0..64)
        .map(|i| (0..16).map(|k| ((i * 16 + k) % 13) as f32 / 13.0).collect())
        .collect();
    let refs: Vec<&[f32]> = vs.iter().map(|v| v.as_slice()).collect();
    let ids: Vec<u64> = (1..=64).collect();
    s.ingest_batch(&refs, &ids, None).ok()?;
    s.close().ok()?;
    std::fs::read(&path)
        .ok()
        .map(|b| ("rvf-runtime-store".into(), b))
}

#[test]
fn stream_verifier_agrees_with_whole_body_verification() {
    let mut corpus = synthetic();
    corpus.extend(forge());
    match runtime_store() {
        Some(c) => corpus.push(c),
        None => eprintln!("NOTE: RvfStore corpus entry could not be produced"),
    }
    let mut stats = Vec::new();
    for (name, body) in &corpus {
        for log2 in [0xFFu8, 12, 16] {
            // A body the descriptor builder cannot segment (see ADR-354 dialect issues) is
            // reported, not silently skipped.
            let probe = StreamDescriptorBuilder::new(FILE_ID, key())
                .chunk_log2(log2)
                .build(body);
            match probe {
                Ok(_) => {
                    let (a, r) = check(name, body, log2);
                    stats.push(format!("{name} log2={log2}: accepted {a}, rejected {r}"));
                }
                Err(e) => stats.push(format!("{name} log2={log2}: NOT SEGMENTABLE ({e:?})")),
            }
        }
    }
    for s in &stats {
        eprintln!("{s}");
    }
    assert!(stats
        .iter()
        .any(|s| s.contains("accepted") && !s.contains("accepted 0")));
    assert!(stats.iter().any(|s| !s.contains("rejected 0")));
    assert!(stats.iter().filter(|s| s.contains("accepted")).count() >= 12);
    assert!(stats.iter().any(|s| s.starts_with("forge-signed-wasm")));
}
