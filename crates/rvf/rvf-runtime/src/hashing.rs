//! Segment content hashing for the runtime's on-disk formats.
//!
//! Two dialects exist (ADR-354):
//!
//! * **Wire dialect** (current writer): `checksum_algo` follows the
//!   `rvf-wire` registry (1 = XXH3-128, 2 = SHAKE-256-128) and segments are
//!   padded to 64 bytes.
//! * **Legacy dialect** (every file written before ADR-354): a 16-byte hash
//!   built from four byte-rotations of an IEEE CRC32 (poly 0xEDB88320) over
//!   the payload, stored with `checksum_algo = 0`, in unpadded segments.
//!
//! `rvf-wire` reads algo 0 as XXH3-128, so the two are not byte-compatible.
//! The reader therefore dual-accepts: algo 0 verifies as the legacy CRC
//! rotation first, then as XXH3-128 (what `rvf-wire` means by algo 0); algo 1
//! and 2 use the registry; anything else is refused (fail closed). The writer
//! never emits algo 0 any more.

use rvf_wire::hash::compute_content_hash;

/// Checksum algorithm the wire-dialect writer uses by default (XXH3-128).
pub(crate) const ALGO_XXH3_128: u8 = 1;
/// Checksum algorithm for anything that will be signed (SHAKE-256-128).
pub(crate) const ALGO_SHAKE256_128: u8 = 2;

/// Verify `payload` against a header's `(algo, expected)` pair.
///
/// Returns `false` for an unknown algorithm. `true` means only that the
/// payload matches the header hash; whether that hash is trustworthy is a
/// policy question (see `Strict`).
pub(crate) fn verify_payload_hash(algo: u8, payload: &[u8], expected: &[u8; 16]) -> bool {
    match algo {
        0 => {
            legacy_content_hash(payload) == *expected
                || compute_content_hash(1, payload) == *expected
        }
        1 | 2 => compute_content_hash(algo, payload) == *expected,
        _ => false,
    }
}

/// Compute the legacy 16-byte content hash: IEEE CRC32 over the payload
/// (via `crc32fast`), rotated left by 0/8/16/24 bits to fill four distinct
/// 4-byte little-endian lanes.
pub(crate) fn legacy_content_hash(data: &[u8]) -> [u8; 16] {
    let crc = crc32fast::hash(data);
    let mut hash = [0u8; 16];
    for i in 0..4 {
        let rotated = crc.rotate_left(i as u32 * 8);
        hash[i * 4..(i + 1) * 4].copy_from_slice(&rotated.to_le_bytes());
    }
    hash
}

/// The 32-byte add-and-rotate fold that earlier versions stored under the
/// name `simple_shake256_256`. It is not a cryptographic hash and is kept only
/// to verify COW state version 1 (`CowMapHeader.version == 1`); nothing new is
/// written with it except back into such a file.
pub(crate) fn legacy_fold_256(data: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, &b) in data.iter().enumerate() {
        out[i % 32] = out[i % 32].wrapping_add(b);
        let j = (i + 13) % 32;
        out[j] = out[j].wrapping_add(out[i % 32].rotate_left(3));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference: the original per-bit IEEE CRC32 (poly 0xEDB88320,
    /// init 0xFFFFFFFF, final XOR) that `crc32fast` replaced.
    fn crc32_reference(data: &[u8]) -> u32 {
        let mut crc: u32 = 0xFFFF_FFFF;
        for &byte in data {
            crc ^= byte as u32;
            for _ in 0..8 {
                if crc & 1 != 0 {
                    crc = (crc >> 1) ^ 0xEDB8_8320;
                } else {
                    crc >>= 1;
                }
            }
        }
        !crc
    }

    fn reference_hash(data: &[u8]) -> [u8; 16] {
        let crc = crc32_reference(data);
        let mut hash = [0u8; 16];
        for i in 0..4 {
            let rotated = crc.rotate_left(i as u32 * 8);
            hash[i * 4..(i + 1) * 4].copy_from_slice(&rotated.to_le_bytes());
        }
        hash
    }

    #[test]
    fn matches_reference_bitwise_implementation() {
        let inputs: [&[u8]; 5] = [
            b"",
            b"a",
            b"123456789",
            b"hello rvf segment payload",
            &[0xFFu8; 1024],
        ];
        for input in inputs {
            assert_eq!(legacy_content_hash(input), reference_hash(input));
        }
        // Standard IEEE CRC32 check value survives in the first lane.
        assert_eq!(
            &legacy_content_hash(b"123456789")[0..4],
            &0xCBF4_3926u32.to_le_bytes()
        );
    }

    #[test]
    fn deterministic_and_input_sensitive() {
        let a = legacy_content_hash(b"payload-a");
        assert_eq!(a, legacy_content_hash(b"payload-a"));
        assert_ne!(a, legacy_content_hash(b"payload-b"));
    }
}
