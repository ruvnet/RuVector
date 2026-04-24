//! Hash computation and verification for RVF segments.
//!
//! The segment header stores a 128-bit content hash. The algorithm is
//! identified by the `checksum_algo` field:
//!
//! - 0 = deprecated CRC32C (transparently upgraded to XXH3-128)
//! - 1 = XXH3-128 (default, ~50 GB/s)
//! - 2 = SHAKE-256 first 128 bits (cryptographic, ~300 MB/s)
//! - 3 = HMAC-SHAKE-256 (reserved, not yet implemented)

use rvf_types::SegmentHeader;
use sha3::{
    digest::{ExtendableOutput, Update},
    Shake256,
};
use subtle::ConstantTimeEq;

/// Compute the XXH3-128 hash of `data`, returning a 16-byte array.
pub fn compute_xxh3_128(data: &[u8]) -> [u8; 16] {
    let h = xxhash_rust::xxh3::xxh3_128(data);
    h.to_le_bytes()
}

/// Compute the CRC32C checksum of `data`.
///
/// Used internally for block-level checksums in manifest and vec_seg codecs.
/// NOT used for segment-level content hashing (see `compute_content_hash`).
pub fn compute_crc32c(data: &[u8]) -> u32 {
    crc32c::crc32c(data)
}

/// Compute SHAKE-256 of `data`, truncated to the first 128 bits (16 bytes).
///
/// SHAKE-256 is a cryptographic extendable-output function from the SHA-3
/// family. Truncating to 128 bits provides 128-bit collision resistance and
/// post-quantum preimage resistance, at ~300 MB/s throughput.
pub fn compute_shake256_128(data: &[u8]) -> [u8; 16] {
    let mut hasher = Shake256::default();
    hasher.update(data);
    let mut out = [0u8; 16];
    hasher.finalize_xof_into(&mut out);
    out
}

/// Compute the content hash for a payload using the algorithm specified
/// by `algo` (the `checksum_algo` field from the segment header).
///
/// - 0 = DEPRECATED CRC32C — upgraded to XXH3-128 for full 128-bit security.
/// - 1 = XXH3-128 (16 bytes)
/// - 2 = SHAKE-256 (first 128 bits, cryptographic)
/// - 3 = HMAC-SHAKE-256 — reserved, falls back to XXH3-128 until key
///   management is implemented.
/// - Other values fall back to XXH3-128.
pub fn compute_content_hash(algo: u8, data: &[u8]) -> [u8; 16] {
    match algo {
        2 => compute_shake256_128(data),
        // 0 (deprecated CRC32C), 1 (XXH3-128), 3 (reserved HMAC), and
        // unknown values all use XXH3-128.
        _ => compute_xxh3_128(data),
    }
}

/// Verify the content hash stored in a segment header against the actual
/// payload bytes.
///
/// Uses constant-time comparison via `subtle::ConstantTimeEq` to prevent
/// timing side-channel attacks that could reveal partial hash values.
///
/// Returns `true` if the computed hash matches `header.content_hash`.
pub fn verify_content_hash(header: &SegmentHeader, payload: &[u8]) -> bool {
    let expected = compute_content_hash(header.checksum_algo, payload);
    expected.ct_eq(&header.content_hash).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xxh3_128_deterministic() {
        let data = b"hello world";
        let h1 = compute_xxh3_128(data);
        let h2 = compute_xxh3_128(data);
        assert_eq!(h1, h2);
        assert_ne!(h1, [0u8; 16]);
    }

    #[test]
    fn shake256_128_deterministic() {
        let data = b"hello world";
        let h1 = compute_shake256_128(data);
        let h2 = compute_shake256_128(data);
        assert_eq!(h1, h2);
        assert_ne!(h1, [0u8; 16]);
    }

    #[test]
    fn shake256_differs_from_xxh3() {
        let data = b"test payload for differentiation";
        let xxh3 = compute_xxh3_128(data);
        let shake = compute_shake256_128(data);
        assert_ne!(xxh3, shake);
    }

    #[test]
    fn compute_content_hash_dispatches_algo_0_to_xxh3() {
        let data = b"algo zero";
        assert_eq!(compute_content_hash(0, data), compute_xxh3_128(data));
    }

    #[test]
    fn compute_content_hash_dispatches_algo_1_to_xxh3() {
        let data = b"algo one";
        assert_eq!(compute_content_hash(1, data), compute_xxh3_128(data));
    }

    #[test]
    fn compute_content_hash_dispatches_algo_2_to_shake256() {
        let data = b"algo two";
        assert_eq!(compute_content_hash(2, data), compute_shake256_128(data));
    }

    #[test]
    fn compute_content_hash_unknown_algo_falls_back_to_xxh3() {
        let data = b"unknown algo";
        assert_eq!(compute_content_hash(255, data), compute_xxh3_128(data));
    }

    #[test]
    fn verify_content_hash_xxh3() {
        let payload = b"some vector data";
        let hash = compute_xxh3_128(payload);
        let header = SegmentHeader {
            magic: rvf_types::SEGMENT_MAGIC,
            version: 1,
            seg_type: 0x01,
            flags: 0,
            segment_id: 1,
            payload_length: payload.len() as u64,
            timestamp_ns: 0,
            checksum_algo: 1, // XXH3-128
            compression: 0,
            reserved_0: 0,
            reserved_1: 0,
            content_hash: hash,
            uncompressed_len: 0,
            alignment_pad: 0,
        };
        assert!(verify_content_hash(&header, payload));
        assert!(!verify_content_hash(&header, b"wrong data"));
    }

    #[test]
    fn verify_content_hash_algo_zero_uses_xxh3() {
        // algo=0 (formerly CRC32C) is upgraded to XXH3-128.
        let payload = b"crc payload";
        let hash = compute_xxh3_128(payload);
        let header = SegmentHeader {
            magic: rvf_types::SEGMENT_MAGIC,
            version: 1,
            seg_type: 0x01,
            flags: 0,
            segment_id: 2,
            payload_length: payload.len() as u64,
            timestamp_ns: 0,
            checksum_algo: 0,
            compression: 0,
            reserved_0: 0,
            reserved_1: 0,
            content_hash: hash,
            uncompressed_len: 0,
            alignment_pad: 0,
        };
        assert!(verify_content_hash(&header, payload));
    }

    #[test]
    fn verify_content_hash_shake256() {
        let payload = b"cryptographic segment data";
        let hash = compute_shake256_128(payload);
        let header = SegmentHeader {
            magic: rvf_types::SEGMENT_MAGIC,
            version: 1,
            seg_type: 0x01,
            flags: 0,
            segment_id: 3,
            payload_length: payload.len() as u64,
            timestamp_ns: 0,
            checksum_algo: 2, // SHAKE-256
            compression: 0,
            reserved_0: 0,
            reserved_1: 0,
            content_hash: hash,
            uncompressed_len: 0,
            alignment_pad: 0,
        };
        assert!(verify_content_hash(&header, payload));
        assert!(!verify_content_hash(&header, b"tampered data"));
    }

    #[test]
    fn verify_rejects_algo_mismatch() {
        // Write with algo=1 (XXH3), but header claims algo=2 (SHAKE-256).
        // The hashes differ, so verification must fail.
        let payload = b"mismatch test";
        let xxh3_hash = compute_xxh3_128(payload);
        let header = SegmentHeader {
            magic: rvf_types::SEGMENT_MAGIC,
            version: 1,
            seg_type: 0x01,
            flags: 0,
            segment_id: 4,
            payload_length: payload.len() as u64,
            timestamp_ns: 0,
            checksum_algo: 2, // claims SHAKE-256
            compression: 0,
            reserved_0: 0,
            reserved_1: 0,
            content_hash: xxh3_hash, // but hash is XXH3
            uncompressed_len: 0,
            alignment_pad: 0,
        };
        // Verifier computes SHAKE-256 for algo=2, which won't match the XXH3 hash.
        assert!(!verify_content_hash(&header, payload));
    }

    #[test]
    fn constant_time_comparison_rejects_wrong_data() {
        // Ensure the constant-time comparison correctly rejects mismatches.
        let payload = b"correct payload";
        let hash = compute_content_hash(1, payload);
        let header = SegmentHeader {
            magic: rvf_types::SEGMENT_MAGIC,
            version: 1,
            seg_type: 0x01,
            flags: 0,
            segment_id: 5,
            payload_length: payload.len() as u64,
            timestamp_ns: 0,
            checksum_algo: 1,
            compression: 0,
            reserved_0: 0,
            reserved_1: 0,
            content_hash: hash,
            uncompressed_len: 0,
            alignment_pad: 0,
        };
        // Flip one byte in payload
        let mut corrupted = payload.to_vec();
        corrupted[0] ^= 0xFF;
        assert!(!verify_content_hash(&header, &corrupted));
    }
}
