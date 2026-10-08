//! Minimal Ed25519 witness signing/verification WASM module.
//!
//! Nightly research candidate ("Candidate B" — measurement floor). Wraps
//! only `rvf_types::ed25519` (generic message sign/verify, no
//! `SegmentHeader`/`SignatureFooter` canonicalization) to measure the
//! smallest possible WASM footprint and latency for the exact signing
//! primitive `ruvector-agent-memory`'s `SignedWitnessSink` depends on.
//!
//! Target: wasm32-unknown-unknown, no_std + alloc.
//! No dynamic allocation is exposed to JS: all buffers are fixed-size
//! static scratch space in WASM linear memory, matching the sibling
//! `rvf-wasm` / `rvf-solver-wasm` edge-tile convention.

#![no_std]

extern crate alloc;

use dlmalloc::GlobalDlmalloc;
use rvf_types::ed25519::{
    ed25519_sign, ed25519_verify, PUBLIC_KEY_SIZE, SECRET_KEY_SIZE, SIGNATURE_SIZE,
};

#[global_allocator]
static ALLOC: GlobalDlmalloc = GlobalDlmalloc;

/// Maximum witness message size this module accepts (256 bytes covers the
/// fixed-size witness records used elsewhere in the ecosystem, e.g. the
/// 73-byte SHAKE-256 chain entry and the 64-byte `LedgerWitnessRecord`,
/// with headroom).
const MAX_MESSAGE_SIZE: usize = 256;

/// Scratch buffer for the message being signed/verified.
static mut MESSAGE_SCRATCH: [u8; MAX_MESSAGE_SIZE] = [0u8; MAX_MESSAGE_SIZE];

/// Return a pointer to the message scratch buffer for JS to write into.
/// Capacity is `rvf_ws_message_capacity()` bytes.
#[no_mangle]
pub extern "C" fn rvf_ws_message_ptr() -> i32 {
    core::ptr::addr_of_mut!(MESSAGE_SCRATCH) as i32
}

/// Capacity, in bytes, of the message scratch buffer.
#[no_mangle]
pub extern "C" fn rvf_ws_message_capacity() -> i32 {
    MAX_MESSAGE_SIZE as i32
}

/// Sign the first `msg_len` bytes of the message scratch buffer with the
/// 32-byte secret key at `secret_ptr`, writing the 64-byte signature to
/// `sig_out_ptr`.
///
/// Returns 0 on success, or a negative error code:
///   -1: `msg_len` out of range (negative or larger than the scratch buffer)
#[no_mangle]
pub extern "C" fn rvf_ws_sign(secret_ptr: i32, msg_len: i32, sig_out_ptr: i32) -> i32 {
    if msg_len < 0 || msg_len as usize > MAX_MESSAGE_SIZE {
        return -1;
    }
    let secret: &[u8; SECRET_KEY_SIZE] = unsafe { &*(secret_ptr as *const [u8; SECRET_KEY_SIZE]) };
    let msg: &[u8] = unsafe {
        core::slice::from_raw_parts(
            core::ptr::addr_of!(MESSAGE_SCRATCH) as *const u8,
            msg_len as usize,
        )
    };
    let sig = ed25519_sign(secret, msg);
    unsafe {
        core::ptr::copy_nonoverlapping(sig.as_ptr(), sig_out_ptr as *mut u8, SIGNATURE_SIZE);
    }
    0
}

/// Verify a 64-byte signature at `sig_ptr` against the first `msg_len`
/// bytes of the message scratch buffer and the 32-byte public key at
/// `pub_ptr`.
///
/// Returns 1 if valid, 0 if invalid, or a negative error code:
///   -1: `msg_len` out of range (negative or larger than the scratch buffer)
#[no_mangle]
pub extern "C" fn rvf_ws_verify(pub_ptr: i32, msg_len: i32, sig_ptr: i32) -> i32 {
    if msg_len < 0 || msg_len as usize > MAX_MESSAGE_SIZE {
        return -1;
    }
    let public: &[u8; PUBLIC_KEY_SIZE] = unsafe { &*(pub_ptr as *const [u8; PUBLIC_KEY_SIZE]) };
    let msg: &[u8] = unsafe {
        core::slice::from_raw_parts(
            core::ptr::addr_of!(MESSAGE_SCRATCH) as *const u8,
            msg_len as usize,
        )
    };
    let signature: &[u8; SIGNATURE_SIZE] = unsafe { &*(sig_ptr as *const [u8; SIGNATURE_SIZE]) };
    if ed25519_verify(public, msg, signature) {
        1
    } else {
        0
    }
}

/// Panic handler for no_std WASM (matches `rvf-wasm`'s convention).
#[cfg(not(test))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}
