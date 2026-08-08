//! Synthetic RVF construction, for tests and fixtures.
//!
//! Production authoring lives in [`crate::author`]; this module is the
//! tamper-friendly counterpart. It builds *partial* containers — a lone
//! segment, a segment stream with no root manifest page — so that a test can
//! corrupt one field and confirm verification refuses the result. `author`
//! deliberately cannot produce those shapes, which is why both exist.
//!
//! The segment encoder is shared rather than reimplemented: everything here
//! calls [`crate::author::write_segment_bytes`], so a change to the wire layout
//! cannot leave fixtures describing a format nothing else writes.
//!
//! The segment layout is the one [`crate::container`] reads:
//!
//! ```text
//! [ 64-byte header ][ payload ][ signature footer, if SIGNED ][ zero pad to 64 ]
//! ```

use crate::author::write_segment_bytes;
use ed25519_dalek::SigningKey;
use rvf_types::SegmentType;

/// Content-hash algorithm the fixtures use: XXH3-128, the format default.
///
/// Authored containers use SHAKE-256 instead (see
/// [`crate::author::DEFAULT_CHECKSUM_ALGO`]). Fixtures keep the default so
/// that the reader is exercised against both algorithms.
const FIXTURE_CHECKSUM_ALGO: u8 = 1;

/// An Ed25519 keypair derived deterministically from a seed, so that a test
/// that signs a fixture produces the same bytes on every run.
pub struct TestKeypair {
    /// The signing key.
    pub signing: SigningKey,
    /// The 32-byte public key, as [`crate::VerifyOptions::trusted_keys`] takes it.
    pub public: [u8; 32],
}

impl TestKeypair {
    /// Derive a keypair from `seed`. The same seed always yields the same key.
    pub fn deterministic(seed: u8) -> Self {
        let secret = rvf_types::sha256(&[seed; 32]);
        let signing = SigningKey::from_bytes(&secret);
        let public = signing.verifying_key().to_bytes();
        Self { signing, public }
    }
}

/// Build an unsigned segment.
pub fn unsigned_segment(seg_type: SegmentType, payload: &[u8], segment_id: u64) -> Vec<u8> {
    write_segment_bytes(seg_type, payload, segment_id, FIXTURE_CHECKSUM_ALGO, None)
}

/// Build a segment signed with `keypair`, carrying an Ed25519 signature footer.
pub fn signed_segment(
    seg_type: SegmentType,
    payload: &[u8],
    segment_id: u64,
    keypair: &TestKeypair,
) -> Vec<u8> {
    write_segment_bytes(
        seg_type,
        payload,
        segment_id,
        FIXTURE_CHECKSUM_ALGO,
        Some(&keypair.signing),
    )
}

/// Build a minimal but complete container: a `META` segment carrying the given
/// capability declaration line, then a `MANIFEST` segment as the root.
pub fn minimal_container(capabilities: &str) -> Vec<u8> {
    let meta = format!("{}={}", crate::CAPABILITY_DECLARATION_KEY, capabilities);
    let mut out = unsigned_segment(SegmentType::Meta, meta.as_bytes(), 1);
    out.extend(unsigned_segment(SegmentType::Manifest, b"root", 2));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvf_types::SEGMENT_ALIGNMENT;

    #[test]
    fn unsigned_segment_is_aligned_and_parses() {
        let data = unsigned_segment(SegmentType::Meta, b"hello", 3);
        assert_eq!(data.len() % SEGMENT_ALIGNMENT, 0);
        let got = crate::inspect_bytes(&data).unwrap();
        assert_eq!(got.segments.len(), 1);
        assert_eq!(got.segments[0].segment_id, 3);
        assert!(!got.segments[0].signed);
    }

    #[test]
    fn signed_segment_is_aligned_and_carries_a_footer() {
        let kp = TestKeypair::deterministic(1);
        let data = signed_segment(SegmentType::Wasm, b"\0asm\x01\0\0\0", 4, &kp);
        assert_eq!(data.len() % SEGMENT_ALIGNMENT, 0);
        let got = crate::inspect_bytes(&data).unwrap();
        assert!(got.segments[0].signed);
        assert_eq!(
            got.segments[0].signature_algo.as_ref().unwrap().name,
            "ed25519"
        );
        assert_eq!(
            got.segments[0]
                .signature_algo
                .as_ref()
                .unwrap()
                .signature_length,
            64
        );
    }

    #[test]
    fn fixtures_are_byte_reproducible() {
        let kp = TestKeypair::deterministic(2);
        assert_eq!(
            signed_segment(SegmentType::Wasm, b"code", 1, &kp),
            signed_segment(SegmentType::Wasm, b"code", 1, &kp)
        );
        assert_eq!(minimal_container("network"), minimal_container("network"));
    }

    #[test]
    fn the_same_seed_yields_the_same_key() {
        assert_eq!(
            TestKeypair::deterministic(9).public,
            TestKeypair::deterministic(9).public
        );
        assert_ne!(
            TestKeypair::deterministic(9).public,
            TestKeypair::deterministic(10).public
        );
    }
}
