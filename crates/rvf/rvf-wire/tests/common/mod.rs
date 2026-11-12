#![allow(dead_code)]
//! Shared helpers for the RVSD tests.

pub mod json;

use ed25519_dalek::SigningKey;
use rvf_types::{SegmentFlags, ROOT_MANIFEST_MAGIC_BYTES, ROOT_MANIFEST_SIZE};
use rvf_wire::stream::*;
use rvf_wire::writer::{write_segment, write_segment_with_algo};

pub const FILE_ID: [u8; 16] = *b"ofer-test-file-1";

pub fn key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}

pub fn pk(seed: u8) -> [u8; 32] {
    key(seed).verifying_key().to_bytes()
}

/// Deterministic pseudo-random payload.
pub fn payload(seed: u64, len: usize) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 24) as u8
        })
        .collect()
}

pub fn root_page() -> Vec<u8> {
    let mut p = vec![0u8; ROOT_MANIFEST_SIZE];
    p[..4].copy_from_slice(&ROOT_MANIFEST_MAGIC_BYTES);
    p[4..16].copy_from_slice(b"root-page-v1");
    p
}

/// A synthetic body built with the rvf-wire writer: Meta, Manifest, Wasm
/// (SHAKE-256 checksum), Index (SHAKE-256), Vec, Quant, then a root page.
pub fn sample_body(scale: usize) -> Vec<u8> {
    let f = SegmentFlags::empty();
    let mut b = Vec::new();
    b.extend(write_segment(0x07, &payload(1, 300), f, 1));
    b.extend(write_segment(0x05, &payload(2, 120), f, 2));
    b.extend(write_segment_with_algo(
        0x10,
        &payload(3, 9000 * scale),
        f,
        3,
        2,
    ));
    b.extend(write_segment_with_algo(
        0x02,
        &payload(4, 5000 * scale),
        f,
        4,
        2,
    ));
    b.extend(write_segment(0x01, &payload(5, 20000 * scale), f, 5));
    b.extend(write_segment(0x06, &payload(6, 1500), f, 6));
    b.extend(root_page());
    b
}

pub fn policy(keys: &[[u8; 32]]) -> StreamPolicy<'_> {
    StreamPolicy::new(keys)
}

pub struct Run {
    pub released: Vec<(u64, Vec<u8>)>,
    pub errors: Vec<ErrorCodeAt>,
    pub state: StreamState,
    pub complete: Option<IdentityStatus>,
}
pub type ErrorCodeAt = (usize, rvf_types::ErrorCode);

/// Feed whole frames; collect releases and errors.
pub fn run_frames<I: IntoIterator<Item = Vec<u8>>>(frames: I, pol: StreamPolicy<'_>) -> Run {
    let mut v = StreamVerifier::new(pol);
    let mut r = Run {
        released: vec![],
        errors: vec![],
        state: v.state(),
        complete: None,
    };
    for (i, f) in frames.into_iter().enumerate() {
        match v.push(&f) {
            Ok(Event::Chunk {
                chunk, complete, ..
            }) => {
                r.released.push((chunk.body_offset, chunk.bytes.to_vec()));
                if complete.is_some() {
                    r.complete = complete;
                }
            }
            Ok(_) => {}
            Err(e) => r.errors.push((i, e)),
        }
    }
    r.state = v.state();
    r
}

/// Every released chunk must be a byte-identical slice of the body.
pub fn assert_subset(r: &Run, body: &[u8]) {
    for (off, bytes) in &r.released {
        let o = *off as usize;
        assert!(o + bytes.len() <= body.len(), "released past the body");
        assert_eq!(
            &body[o..o + bytes.len()],
            &bytes[..],
            "released bytes differ at {o}"
        );
    }
}

/// Re-sign a mutated copy of a bundle's header and table with `key`.
pub fn resign(
    b: &StreamBundle,
    key: &SigningKey,
    f: impl FnOnce(&mut Vec<rvf_types::StreamEntry>, &mut rvf_types::StreamHeader),
) -> Vec<u8> {
    let (mut es, mut h) = (b.entries.clone(), b.header);
    f(&mut es, &mut h);
    let mut d = h.to_bytes().to_vec();
    for e in &es {
        d.extend_from_slice(&e.to_bytes());
    }
    d.extend_from_slice(&64u16.to_le_bytes());
    let sig = rvf_crypto::sign_descriptor(&d, key);
    d.extend_from_slice(&sig);
    d
}
