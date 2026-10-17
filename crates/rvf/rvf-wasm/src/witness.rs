//! Optional Ed25519 witness-record signing for Cognitum tiles (ADR-361).
//!
//! Gated behind the `witness-signing` feature (default: off). Reuses the
//! existing Message I/O buffer region (`memory::MESSAGE_IO_OFFSET`) instead
//! of growing `DATA_MEMORY`, so a tile built without this feature is
//! byte-for-byte unaffected and the 8 KB data-memory layout in
//! `memory.rs` does not change.
//!
//! Buffer layout while this feature's exports are in use (within the
//! 2048-byte Message I/O buffer):
//!
//! ```text
//! [0..32)              key: 32-byte secret key (sign) or public key (verify)
//! [32..32+msg_len)      message bytes (caller-supplied length)
//! [tail-64..tail)       64-byte Ed25519 signature (output of sign, input to verify)
//! ```
//!
//! This mirrors `rvf_types::ed25519`'s raw sign/verify primitive directly
//! (not `rvf-crypto`'s segment-specific wrapper), since that primitive is
//! the shared signing surface `ruvector-retrieval-receipt` and
//! `ruvector-agent-memory` both already depend on.

use crate::memory::{DATA_MEMORY, MESSAGE_IO_OFFSET, MESSAGE_IO_SIZE};
use rvf_types::ed25519::{
    ed25519_sign, ed25519_verify, PUBLIC_KEY_SIZE, SECRET_KEY_SIZE, SIGNATURE_SIZE,
};

const KEY_OFFSET: usize = 0;
const MSG_OFFSET: usize = 32;
const SIG_OFFSET: usize = MESSAGE_IO_SIZE - SIGNATURE_SIZE;
/// Largest message this buffer layout can hold alongside the 32-byte key
/// and the 64-byte signature tail.
pub const MAX_MSG_LEN: usize = SIG_OFFSET - MSG_OFFSET;

const _: () = assert!(PUBLIC_KEY_SIZE == SECRET_KEY_SIZE);

/// Sign `msg_len` bytes of message data from the Message I/O buffer using
/// the 32-byte secret key at the buffer's start. Writes the 64-byte
/// signature to the buffer's last 64 bytes.
///
/// Returns 0 on success, -1 if `msg_len` does not fit the buffer.
#[no_mangle]
pub extern "C" fn rvf_witness_ed25519_sign(msg_len: i32) -> i32 {
    if msg_len < 0 {
        return -1;
    }
    let msg_len = msg_len as usize;
    if msg_len == 0 || msg_len > MAX_MSG_LEN {
        return -1;
    }
    unsafe {
        let base = DATA_MEMORY.as_ptr().add(MESSAGE_IO_OFFSET);
        let mut secret = [0u8; SECRET_KEY_SIZE];
        core::ptr::copy_nonoverlapping(base.add(KEY_OFFSET), secret.as_mut_ptr(), SECRET_KEY_SIZE);
        let msg_slice = core::slice::from_raw_parts(base.add(MSG_OFFSET), msg_len);
        let sig = ed25519_sign(&secret, msg_slice);
        let out = DATA_MEMORY.as_mut_ptr().add(MESSAGE_IO_OFFSET + SIG_OFFSET);
        core::ptr::copy_nonoverlapping(sig.as_ptr(), out, SIGNATURE_SIZE);
    }
    0
}

/// Verify `msg_len` bytes of message data in the Message I/O buffer
/// against the 32-byte public key at the buffer's start and the 64-byte
/// signature in the buffer's last 64 bytes.
///
/// Returns 1 if the signature is valid, 0 if invalid, -1 if `msg_len` does
/// not fit the buffer.
#[no_mangle]
pub extern "C" fn rvf_witness_ed25519_verify(msg_len: i32) -> i32 {
    if msg_len < 0 {
        return -1;
    }
    let msg_len = msg_len as usize;
    if msg_len == 0 || msg_len > MAX_MSG_LEN {
        return -1;
    }
    unsafe {
        let base = DATA_MEMORY.as_ptr().add(MESSAGE_IO_OFFSET);
        let mut public = [0u8; PUBLIC_KEY_SIZE];
        core::ptr::copy_nonoverlapping(base.add(KEY_OFFSET), public.as_mut_ptr(), PUBLIC_KEY_SIZE);
        let msg_slice = core::slice::from_raw_parts(base.add(MSG_OFFSET), msg_len);
        let mut sig = [0u8; SIGNATURE_SIZE];
        core::ptr::copy_nonoverlapping(base.add(SIG_OFFSET), sig.as_mut_ptr(), SIGNATURE_SIZE);
        if ed25519_verify(&public, msg_slice, &sig) {
            1
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestRng(u64);
    impl rand_core::RngCore for TestRng {
        fn next_u32(&mut self) -> u32 {
            self.next_u64() as u32
        }
        fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn fill_bytes(&mut self, dest: &mut [u8]) {
            let mut i = 0;
            while i < dest.len() {
                let val = self.next_u64().to_le_bytes();
                let take = (dest.len() - i).min(8);
                dest[i..i + take].copy_from_slice(&val[..take]);
                i += take;
            }
        }
        fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
            self.fill_bytes(dest);
            Ok(())
        }
    }
    impl rand_core::CryptoRng for TestRng {}

    fn write_buf(offset_in_io: usize, data: &[u8]) {
        unsafe {
            let dst = DATA_MEMORY
                .as_mut_ptr()
                .add(MESSAGE_IO_OFFSET + offset_in_io);
            core::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
    }

    fn read_buf(offset_in_io: usize, len: usize) -> alloc::vec::Vec<u8> {
        unsafe {
            let src = DATA_MEMORY.as_ptr().add(MESSAGE_IO_OFFSET + offset_in_io);
            core::slice::from_raw_parts(src, len).to_vec()
        }
    }

    #[test]
    fn sign_then_verify_round_trips_through_shared_buffer() {
        let mut rng = TestRng(0xC0FFEE_1234_5678);
        let kp = rvf_types::ed25519::Ed25519Keypair::generate(&mut rng);
        let message = b"rvf-wasm witness-signing round trip";

        write_buf(KEY_OFFSET, &kp.secret_key());
        write_buf(MSG_OFFSET, message);
        assert_eq!(rvf_witness_ed25519_sign(message.len() as i32), 0);

        let sig = read_buf(SIG_OFFSET, SIGNATURE_SIZE);

        write_buf(KEY_OFFSET, &kp.public_key());
        assert_eq!(rvf_witness_ed25519_verify(message.len() as i32), 1);

        // Tamper with one byte of the signature: verification must fail.
        let mut bad_sig = sig.clone();
        bad_sig[0] ^= 0x01;
        write_buf(SIG_OFFSET, &bad_sig);
        assert_eq!(rvf_witness_ed25519_verify(message.len() as i32), 0);
    }

    #[test]
    fn oversized_message_is_rejected() {
        assert_eq!(rvf_witness_ed25519_sign((MAX_MSG_LEN + 1) as i32), -1);
        assert_eq!(rvf_witness_ed25519_verify((MAX_MSG_LEN + 1) as i32), -1);
        assert_eq!(rvf_witness_ed25519_sign(0), -1);
        assert_eq!(rvf_witness_ed25519_sign(-1), -1);
    }
}
