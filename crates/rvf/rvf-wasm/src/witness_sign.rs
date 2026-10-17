//! Ed25519 witness signing/verification, feature-gated (`witness-sign`).
//!
//! Nightly research candidate ("Candidate A" — in-place cost). Exercises
//! the real production `rvf_crypto::sign::{sign_segment, verify_segment}`
//! path (the same canonicalization used for signed RVF segments) to
//! measure the actual size and latency cost of adding Ed25519 witness
//! signing to the existing Cognitum edge tile microkernel, which the
//! module doc for this crate targets at "< 8 KB after wasm-opt" without
//! signing.
//!
//! This module is entirely opt-in: it is compiled only when the
//! `witness-sign` feature is enabled, so the default build of this crate
//! (and its size) is unaffected.

use ed25519_dalek::{SigningKey, VerifyingKey};
use rvf_crypto::{sign_segment, verify_segment};
use rvf_types::SegmentHeader;

/// Segment type tag used for synthetic witness-signing benchmark headers.
/// Not a real `SegmentType` variant; chosen to be unambiguous in traces.
const WITNESS_BENCH_SEG_TYPE: u8 = 0xFE;

/// Maximum payload size accepted by this module's fixed scratch buffer.
const MAX_PAYLOAD_SIZE: usize = 256;

static mut PAYLOAD_SCRATCH: [u8; MAX_PAYLOAD_SIZE] = [0u8; MAX_PAYLOAD_SIZE];

/// Return a pointer to the payload scratch buffer for JS to write into.
#[no_mangle]
pub extern "C" fn rvf_witness_sign_payload_ptr() -> i32 {
    core::ptr::addr_of_mut!(PAYLOAD_SCRATCH) as i32
}

/// Capacity, in bytes, of the payload scratch buffer.
#[no_mangle]
pub extern "C" fn rvf_witness_sign_payload_capacity() -> i32 {
    MAX_PAYLOAD_SIZE as i32
}

/// Sign the first `payload_len` bytes of the payload scratch buffer, using
/// a synthetic `SegmentHeader` built from `segment_id`, via the real
/// `rvf_crypto::sign_segment` production path. Writes the 64-byte Ed25519
/// signature to `sig_out_ptr`.
///
/// Returns 0 on success, or a negative error code:
///   -1: `payload_len` out of range
///   -2: malformed 32-byte secret key at `secret_ptr`
#[no_mangle]
pub extern "C" fn rvf_witness_sign_segment(
    segment_id: i32,
    payload_len: i32,
    secret_ptr: i32,
    sig_out_ptr: i32,
) -> i32 {
    if payload_len < 0 || payload_len as usize > MAX_PAYLOAD_SIZE {
        return -1;
    }
    let secret: &[u8; 32] = unsafe { &*(secret_ptr as *const [u8; 32]) };
    let key = SigningKey::from_bytes(secret);
    let header = SegmentHeader::new(WITNESS_BENCH_SEG_TYPE, segment_id as u64);
    let payload: &[u8] = unsafe {
        core::slice::from_raw_parts(
            core::ptr::addr_of!(PAYLOAD_SCRATCH) as *const u8,
            payload_len as usize,
        )
    };
    let footer = sign_segment(&header, payload, &key);
    unsafe {
        core::ptr::copy_nonoverlapping(footer.signature.as_ptr(), sig_out_ptr as *mut u8, 64);
    }
    0
}

/// Verify a 64-byte Ed25519 signature at `sig_ptr` against the first
/// `payload_len` bytes of the payload scratch buffer and a synthetic
/// `SegmentHeader` built from `segment_id`, via the real
/// `rvf_crypto::verify_segment` production path.
///
/// Returns 1 if valid, 0 if invalid, or a negative error code:
///   -1: `payload_len` out of range
///   -2: malformed 32-byte public key at `pub_ptr`
#[no_mangle]
pub extern "C" fn rvf_witness_verify_segment(
    segment_id: i32,
    payload_len: i32,
    pub_ptr: i32,
    sig_ptr: i32,
) -> i32 {
    if payload_len < 0 || payload_len as usize > MAX_PAYLOAD_SIZE {
        return -1;
    }
    let public: &[u8; 32] = unsafe { &*(pub_ptr as *const [u8; 32]) };
    let pubkey = match VerifyingKey::from_bytes(public) {
        Ok(k) => k,
        Err(_) => return -2,
    };
    let header = SegmentHeader::new(WITNESS_BENCH_SEG_TYPE, segment_id as u64);
    let payload: &[u8] = unsafe {
        core::slice::from_raw_parts(
            core::ptr::addr_of!(PAYLOAD_SCRATCH) as *const u8,
            payload_len as usize,
        )
    };
    let sig_bytes: [u8; 64] = unsafe { *(sig_ptr as *const [u8; 64]) };
    let mut signature = [0u8; rvf_types::SignatureFooter::MAX_SIG_LEN];
    signature[..64].copy_from_slice(&sig_bytes);
    let footer = rvf_types::SignatureFooter {
        sig_algo: 0,
        sig_length: 64,
        signature,
        footer_length: rvf_types::SignatureFooter::compute_footer_length(64),
    };
    if verify_segment(&header, payload, &footer, &pubkey) {
        1
    } else {
        0
    }
}
