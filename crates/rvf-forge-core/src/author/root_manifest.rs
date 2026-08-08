//! The Level-0 root manifest page that closes every authored container.
//!
//! An RVF file is discovered from its tail: the last 4096 bytes are a fixed
//! Level-0 root manifest (`rvf-manifest/src/level0.rs`), carrying `RVM0` rather
//! than the segment magic. It is not a segment, and [`crate::container`] trims
//! it before walking the segment stream.
//!
//! Writing that page is what makes authored output a *file* rather than a bare
//! segment stream, and it is where the container-level signature lives.

use super::{AuthoredContainer, ED25519_SIGNATURE_LENGTH};
use ed25519_dalek::SigningKey;
use rvf_types::{ROOT_MANIFEST_MAGIC, ROOT_MANIFEST_SIZE};

/// Ed25519, as the Level-0 root manifest's `sig_algo` field names it.
///
/// Not the same value a *segment* signature footer uses for Ed25519, which is
/// 0 (`rvf-crypto/src/sign.rs`). The two structures carry different algorithm
/// enumerations; each writer has to use its own.
pub(super) const SIG_ALGO_ED25519: u16 = 1;

/// Level-0 root manifest field offsets (`rvf-manifest/src/level0.rs`).
mod l0 {
    pub(super) const VERSION: usize = 0x004;
    pub(super) const L1_OFFSET: usize = 0x008;
    pub(super) const L1_LENGTH: usize = 0x010;
    pub(super) const DTYPE: usize = 0x022;
    pub(super) const EPOCH: usize = 0x024;
    pub(super) const SIG_ALGO: usize = 0x098;
    pub(super) const SIG_LEN: usize = 0x09A;
    pub(super) const SIGNATURE: usize = 0x09C;
    pub(super) const FILE_ID: usize = 0xF00;
    pub(super) const CHECKSUM: usize = 0xFFC;
}

/// Build the 4096-byte Level-0 root manifest page.
pub(super) fn root_manifest_page(
    file_id: &[u8; 16],
    l1_offset: u64,
    l1_length: u64,
    key: Option<&SigningKey>,
) -> [u8; ROOT_MANIFEST_SIZE] {
    let mut page = [0u8; ROOT_MANIFEST_SIZE];
    page[0x00..0x04].copy_from_slice(&ROOT_MANIFEST_MAGIC.to_le_bytes());
    page[l0::VERSION..l0::VERSION + 2].copy_from_slice(&1u16.to_le_bytes());
    page[l0::L1_OFFSET..l0::L1_OFFSET + 8].copy_from_slice(&l1_offset.to_le_bytes());
    page[l0::L1_LENGTH..l0::L1_LENGTH + 8].copy_from_slice(&l1_length.to_le_bytes());
    // total_vector_count and dimension stay zero: an agent container declares
    // no vectors, and a zero dimension is only invalid alongside a non-zero
    // vector count.
    page[l0::DTYPE] = 1; // f32
    page[l0::EPOCH..l0::EPOCH + 4].copy_from_slice(&1u32.to_le_bytes());
    // created_ns and modified_ns stay zero, for reproducibility.
    page[l0::FILE_ID..l0::FILE_ID + 16].copy_from_slice(file_id);

    if let Some(key) = key {
        use ed25519_dalek::Signer;
        // The algorithm and length fields are written *before* signing so the
        // signature covers them. Leaving them out would let an attacker
        // rewrite `sig_algo` to a weaker scheme without invalidating anything.
        page[l0::SIG_ALGO..l0::SIG_ALGO + 2].copy_from_slice(&SIG_ALGO_ED25519.to_le_bytes());
        page[l0::SIG_LEN..l0::SIG_LEN + 2]
            .copy_from_slice(&(ED25519_SIGNATURE_LENGTH as u16).to_le_bytes());
        let signature = key.sign(&root_manifest_signing_bytes(&page)).to_bytes();
        page[l0::SIGNATURE..l0::SIGNATURE + ED25519_SIGNATURE_LENGTH].copy_from_slice(&signature);
    }

    let checksum = rvf_wire::hash::compute_crc32c(&page[..l0::CHECKSUM]);
    page[l0::CHECKSUM..].copy_from_slice(&checksum.to_le_bytes());
    page
}

/// The bytes a root manifest page's signature covers: the whole page with the
/// signature region and the trailing CRC32C zeroed.
///
/// Both regions have to be excluded because both are written *after* signing.
/// Zeroing rather than skipping keeps the signed length fixed at 4096, so a
/// verifier reconstructs the input without knowing any field order.
pub fn root_manifest_signing_bytes(page: &[u8; ROOT_MANIFEST_SIZE]) -> [u8; ROOT_MANIFEST_SIZE] {
    let mut signable = *page;
    signable[l0::SIGNATURE..l0::SIGNATURE + ED25519_SIGNATURE_LENGTH].fill(0);
    signable[l0::CHECKSUM..].fill(0);
    signable
}

/// Verify the Ed25519 signature on a container's root manifest page.
///
/// Returns `false` when the page is unsigned, names an algorithm other than
/// Ed25519, or does not verify against `public_key`.
pub fn verify_root_manifest_signature(
    page: &[u8; ROOT_MANIFEST_SIZE],
    public_key: &[u8; 32],
) -> bool {
    use ed25519_dalek::{Signature, Verifier, VerifyingKey};

    let algo = u16::from_le_bytes([page[l0::SIG_ALGO], page[l0::SIG_ALGO + 1]]);
    let len = u16::from_le_bytes([page[l0::SIG_LEN], page[l0::SIG_LEN + 1]]);
    if algo != SIG_ALGO_ED25519 || len as usize != ED25519_SIGNATURE_LENGTH {
        return false;
    }
    let Ok(verifying) = VerifyingKey::from_bytes(public_key) else {
        return false;
    };
    let mut raw = [0u8; ED25519_SIGNATURE_LENGTH];
    raw.copy_from_slice(&page[l0::SIGNATURE..l0::SIGNATURE + ED25519_SIGNATURE_LENGTH]);
    verifying
        .verify(
            &root_manifest_signing_bytes(page),
            &Signature::from_bytes(&raw),
        )
        .is_ok()
}

/// Borrow the trailing Level-0 root manifest page of an authored container.
///
/// Returns `None` when `data` does not end in one.
pub fn root_manifest_page_of(data: &[u8]) -> Option<&[u8; ROOT_MANIFEST_SIZE]> {
    let (_, page) = crate::container::split_root_manifest(data);
    page.map(|range| {
        data[range]
            .try_into()
            .expect("split_root_manifest returns a 4096-byte range")
    })
}

impl AuthoredContainer {
    /// The container's Level-0 root manifest page.
    pub fn root_manifest_page(&self) -> &[u8; ROOT_MANIFEST_SIZE] {
        root_manifest_page_of(&self.bytes).expect("an authored container always ends in a page")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::author::ContainerBuilder;
    use crate::inspect_bytes;
    use crate::testkit::TestKeypair;
    use crate::SegmentType;
    use rvf_types::SEGMENT_ALIGNMENT;

    #[test]
    fn the_page_closes_the_file_and_checksums_itself() {
        let built = ContainerBuilder::new().build().unwrap();
        assert_eq!(built.bytes.len() % SEGMENT_ALIGNMENT, 0);

        let page = built.root_manifest_page();
        assert_eq!(page[..4], ROOT_MANIFEST_MAGIC.to_le_bytes());
        let stored = u32::from_le_bytes(page[l0::CHECKSUM..].try_into().unwrap());
        assert_eq!(
            stored,
            rvf_wire::hash::compute_crc32c(&page[..l0::CHECKSUM])
        );
    }

    #[test]
    fn the_signature_verifies_and_only_against_the_signing_key() {
        let kp = TestKeypair::deterministic(12);
        let built = ContainerBuilder::new()
            .sign_with(kp.signing.clone())
            .build()
            .unwrap();

        assert!(verify_root_manifest_signature(
            built.root_manifest_page(),
            &kp.public
        ));
        assert!(!verify_root_manifest_signature(
            built.root_manifest_page(),
            &TestKeypair::deterministic(13).public
        ));
    }

    #[test]
    fn rewriting_a_signed_field_breaks_the_signature() {
        let kp = TestKeypair::deterministic(15);
        let built = ContainerBuilder::new()
            .sign_with(kp.signing.clone())
            .build()
            .unwrap();

        let mut page = *built.root_manifest_page();
        page[l0::VERSION] = 7;
        assert!(!verify_root_manifest_signature(&page, &kp.public));

        // The algorithm field is covered too, so downgrading it is caught.
        let mut downgraded = *built.root_manifest_page();
        downgraded[l0::SIG_ALGO] = 9;
        assert!(!verify_root_manifest_signature(&downgraded, &kp.public));
    }

    #[test]
    fn an_unsigned_page_carries_no_signature() {
        let built = ContainerBuilder::new().build().unwrap();
        assert!(!built.signed);
        assert!(!verify_root_manifest_signature(
            built.root_manifest_page(),
            &TestKeypair::deterministic(1).public
        ));
    }

    #[test]
    fn the_file_id_reaches_the_page() {
        let built = ContainerBuilder::new().file_id([0xab; 16]).build().unwrap();
        assert_eq!(
            &built.root_manifest_page()[l0::FILE_ID..l0::FILE_ID + 16],
            &[0xab; 16]
        );
    }

    #[test]
    fn the_l1_pointer_locates_the_manifest_segment() {
        let built = ContainerBuilder::new()
            .segment(SegmentType::Vec, vec![0u8; 100])
            .build()
            .unwrap();
        let page = built.root_manifest_page();
        let offset = u64::from_le_bytes(page[l0::L1_OFFSET..l0::L1_OFFSET + 8].try_into().unwrap());
        let length = u64::from_le_bytes(page[l0::L1_LENGTH..l0::L1_LENGTH + 8].try_into().unwrap());

        assert!(length > 0, "an empty L1 pointer would warn on validation");
        assert!(offset + length <= built.bytes.len() as u64);
        assert_eq!(
            inspect_bytes(&built.bytes).unwrap().root_manifest_offset,
            Some(offset)
        );
    }

    #[test]
    fn a_bare_segment_stream_has_no_page() {
        let stream = crate::testkit::minimal_container("network");
        assert!(root_manifest_page_of(&stream).is_none());
    }
}
