//! Shared helpers for the signed-store tests (`strict_policy.rs`,
//! `strict_segments.rs`), included with `#[path]`.
#![allow(dead_code)]

use rvf_runtime::{RvfOptions, RvfStore};
use rvf_types::ed25519::Ed25519Keypair;
use rvf_types::security::{SecurityError, SecurityPolicy};
use rvf_types::{RvfError, SEGMENT_HEADER_SIZE};
use rvf_wire::tail_scan::split_root_page;
use rvf_wire::writer::calculate_padded_size;
use std::path::{Path, PathBuf};

pub const SECRET: [u8; 32] = [0x11; 32];
pub const OTHER_SECRET: [u8; 32] = [0x22; 32];

pub fn public(secret: &[u8; 32]) -> [u8; 32] {
    Ed25519Keypair::from_secret(secret).public_key()
}

pub fn opts(trusted: &[[u8; 32]]) -> RvfOptions {
    RvfOptions {
        dimension: 4,
        trusted_signers: trusted.to_vec(),
        freshness: Some(rvf_runtime::Freshness::Unchecked),
        ..Default::default()
    }
}

pub fn rows(n: usize, seed: f32) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| vec![seed + i as f32, 1.0, 2.0, 3.0])
        .collect()
}

pub fn signed_store(dir: &Path) -> PathBuf {
    let path = dir.join("signed.rvf");
    let mut s = RvfStore::create_signed(&path, opts(&[]), &SECRET).unwrap();
    for b in 0..2u64 {
        let v = rows(6, b as f32 * 100.0);
        let refs: Vec<&[f32]> = v.iter().map(Vec::as_slice).collect();
        let ids: Vec<u64> = (0..6).map(|i| b * 10 + i).collect();
        s.ingest_batch(&refs, &ids, None).unwrap();
    }
    s.close().unwrap();
    path
}

/// (offset, type, payload_len) of every segment, wire-dialect walk.
pub fn segments(bytes: &[u8]) -> Vec<(usize, u8, usize)> {
    let (body, _) = split_root_page(bytes);
    let mut off = 0;
    let mut out = Vec::new();
    while off < body.len() {
        let h = rvf_wire::read_segment_header(&body[off..]).unwrap();
        out.push((off, h.seg_type, h.payload_length as usize));
        off += calculate_padded_size(SEGMENT_HEADER_SIZE, h.payload_length as usize);
    }
    out
}

pub fn page_of(bytes: &[u8]) -> [u8; 4096] {
    bytes[bytes.len() - 4096..].try_into().unwrap()
}

pub fn set_page(bytes: &mut [u8], page: &[u8; 4096]) {
    let n = bytes.len();
    bytes[n - 4096..].copy_from_slice(page);
}

pub fn security(e: RvfError) -> SecurityError {
    match e {
        RvfError::Security(s) => s,
        other => panic!("expected a Security error, got {other:?}"),
    }
}

pub fn open_err(path: &Path, trusted: &[[u8; 32]], policy: SecurityPolicy) -> RvfError {
    match RvfStore::open_with_policy(path, opts(trusted), policy) {
        Err(e) => e,
        Ok(_) => panic!("{policy:?} opened a file it must refuse"),
    }
}

/// [`opts`] that serve the signed prefix when unsigned manifest-shaped bytes
/// follow the signed manifest (F4-04 opt-in), for tests that inject such bytes.
pub fn opts_prefix(trusted: &[[u8; 32]]) -> RvfOptions {
    RvfOptions {
        serve_signed_prefix: true,
        ..opts(trusted)
    }
}
