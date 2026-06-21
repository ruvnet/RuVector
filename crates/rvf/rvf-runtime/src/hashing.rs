//! Segment content hashing for the runtime's on-disk format — the single
//! source of truth.
//!
//! The runtime's 0.2.0 format stores a 16-byte content hash built from four
//! byte-rotations of an IEEE CRC32 (poly 0xEDB88320) over the payload, with
//! `checksum_algo = 0` in the segment header. This module is the ONE
//! implementation of that algorithm; both `write_path` (computing hashes)
//! and `read_path` (verifying them) delegate here. Previously each file
//! carried its own copy of the same function.
//!
//! Relationship to `rvf-wire`: the wire crate defines the standard
//! algorithm registry (0 = legacy CRC32C upgraded to XXH3-128,
//! 1 = XXH3-128, 2 = SHAKE-256/128) which is NOT byte-compatible with this
//! legacy hash — the runtime historically labelled its CRC-rotation hash
//! with `checksum_algo = 0`, which rvf-wire interprets as XXH3-128.
//! Migrating the runtime onto the rvf-wire registry therefore requires a
//! format-version bump plus a dual-accept reader (verify XXH3 first, fall
//! back to this legacy hash) so existing .rvf files keep opening. That
//! migration is tracked as follow-up work; until then this module keeps the
//! 0.2.0 on-disk contract byte-for-byte intact.

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
