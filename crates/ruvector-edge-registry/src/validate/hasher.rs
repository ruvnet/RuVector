//! Streaming segment content hashes.
//!
//! The `checksum_algo` registry is rvf-wire's (`rvf_wire::hash`): 1 =
//! XXH3-128, 2 = SHAKE-256 truncated to 128 bits. Algorithm 0 is ambiguous on
//! disk: rvf-wire reads it as XXH3-128, while the rvf runtime (the engine the
//! `rvf` CLI exports with) writes an IEEE CRC32 rotated into four lanes under
//! the same byte (`rvf-runtime/src/hashing.rs`). Algorithm 0 therefore
//! accepts either, exactly the dual-accept reader that module prescribes.
//! Algorithm 3 (reserved HMAC) and unknown values are refused rather than
//! silently falling back as rvf-wire does.
//!
//! The one-shot equivalents in `rvf_wire::hash` are the oracle for the unit
//! tests below.

use sha3::digest::{ExtendableOutput, Update};
use sha3::Shake256;
use xxhash_rust::xxh3::Xxh3;

/// Incremental content hasher for one segment payload.
pub(crate) enum ContentHasher {
    /// Algorithm 0: XXH3-128 or the legacy CRC32 rotation.
    Legacy(Box<Xxh3>, crc32fast::Hasher),
    /// Algorithm 1.
    Xxh3(Box<Xxh3>),
    /// Algorithm 2.
    Shake(Box<Shake256>),
}

impl ContentHasher {
    /// Hasher for `algo`, or `None` if the algorithm is not accepted.
    pub(crate) fn for_algo(algo: u8) -> Option<Self> {
        match algo {
            0 => Some(ContentHasher::Legacy(
                Box::new(Xxh3::new()),
                crc32fast::Hasher::new(),
            )),
            1 => Some(ContentHasher::Xxh3(Box::new(Xxh3::new()))),
            2 => Some(ContentHasher::Shake(Box::default())),
            _ => None,
        }
    }

    /// Feed payload bytes.
    pub(crate) fn update(&mut self, data: &[u8]) {
        match self {
            ContentHasher::Legacy(x, c) => {
                x.update(data);
                c.update(data);
            }
            ContentHasher::Xxh3(x) => x.update(data),
            ContentHasher::Shake(s) => Update::update(s.as_mut(), data),
        }
    }

    /// `true` if the payload hashes to `expected`.
    pub(crate) fn matches(self, expected: &[u8; 16]) -> bool {
        match self {
            ContentHasher::Legacy(x, c) => {
                x.digest128().to_le_bytes() == *expected || legacy_lanes(c.finalize()) == *expected
            }
            ContentHasher::Xxh3(x) => x.digest128().to_le_bytes() == *expected,
            ContentHasher::Shake(s) => {
                let mut out = [0u8; 16];
                s.finalize_xof_into(&mut out);
                out == *expected
            }
        }
    }
}

/// The rvf runtime's 16-byte layout of a CRC32: rotations by 0/8/16/24 bits.
pub(crate) fn legacy_lanes(crc: u32) -> [u8; 16] {
    let mut out = [0u8; 16];
    for (i, lane) in out.chunks_exact_mut(4).enumerate() {
        lane.copy_from_slice(&crc.rotate_left(i as u32 * 8).to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn streamed(algo: u8, data: &[u8], split: usize) -> ContentHasher {
        let mut h = ContentHasher::for_algo(algo).unwrap();
        let (a, b) = data.split_at(split.min(data.len()));
        h.update(a);
        h.update(b);
        h
    }

    #[test]
    fn matches_rvf_wire_oracle_for_every_accepted_algo() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 7 + 3) as u8).collect();
        for algo in 0..=2u8 {
            let want = rvf_wire::hash::compute_content_hash(algo, &data);
            for split in [0, 1, 17, 500, 1000] {
                assert!(streamed(algo, &data, split).matches(&want), "algo {algo}");
            }
        }
    }

    #[test]
    fn algo_zero_also_accepts_runtime_legacy_crc() {
        let data = b"runtime payload";
        let legacy = legacy_lanes(crc32fast::hash(data));
        assert!(streamed(0, data, 3).matches(&legacy));
        assert!(!streamed(1, data, 3).matches(&legacy));
    }

    #[test]
    fn reserved_and_unknown_algos_are_refused() {
        for algo in 3..=255u8 {
            assert!(ContentHasher::for_algo(algo).is_none());
        }
    }

    #[test]
    fn wrong_digest_is_rejected() {
        assert!(!streamed(2, b"abc", 1).matches(&[0u8; 16]));
    }
}
