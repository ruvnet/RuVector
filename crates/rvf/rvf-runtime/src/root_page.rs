//! Level-0 root manifest page: build, sign, and verify (ADR-033 section 4,
//! ADR-354 item 5).
//!
//! The 4096-byte page (`RVM0`) is the *last* 4096 bytes of a signed store. Its
//! layout follows `rvf-manifest`'s Level-0 codec, and the signature scheme is
//! byte-for-byte the one `rvf-forge-core` uses, so one verifier accepts both:
//!
//! * `sig_algo` at `0x098` is `1` for Ed25519 (the Level-0 enumeration, which
//!   is not the segment-footer enumeration where Ed25519 is `0`);
//! * the signature covers the whole page with the signature region
//!   (`0x09C..0x0DC`) and the trailing CRC32C (`0xFFC..`) zeroed;
//! * CRC32C of bytes `0x000..0xFFC` is stored at `0xFFC`.
//!
//! Two fields are new in the reserved area and are covered by the signature:
//!
//! * `0xFD0`, `l1_content_hash`: SHAKE-256-128 of the MANIFEST segment
//!   payload that `l1_manifest_offset/length` point at. Without it the page
//!   signs *where* the manifest is but not *what* it says, so a rewritten
//!   manifest at the same offset would still verify.
//! * `0xFE0`, `signer_fingerprint`: SHAKE-256-128 of the signing public key,
//!   which lets a verifier tell "unknown signer" from "invalid signature".
//!
//! Without the `ed25519` feature nothing can be signed and verification
//! always fails closed.

use rvf_types::{FileIdentity, ROOT_MANIFEST_MAGIC, ROOT_MANIFEST_SIZE};
use rvf_wire::hash::{compute_crc32c, compute_shake256_128};

/// Page size in bytes.
pub const PAGE_SIZE: usize = ROOT_MANIFEST_SIZE;
/// `sig_algo` value for Ed25519 in the Level-0 page.
pub const SIG_ALGO_ED25519: u16 = 1;
/// Ed25519 signature length.
pub const ED25519_SIGNATURE_LEN: usize = 64;

const OFF_VERSION: usize = 0x004;
const OFF_L1_OFFSET: usize = 0x008;
const OFF_L1_LENGTH: usize = 0x010;
const OFF_TOTAL_VEC: usize = 0x018;
const OFF_DIM: usize = 0x020;
const OFF_DTYPE: usize = 0x022;
const OFF_PROFILE: usize = 0x023;
const OFF_EPOCH: usize = 0x024;
const OFF_SIG_ALGO: usize = 0x098;
const OFF_SIG_LEN: usize = 0x09A;
const OFF_SIGNATURE: usize = 0x09C;
/// FileIdentity (68 bytes) starts the reserved area.
const OFF_IDENTITY: usize = 0xF00;
/// Offset of the L1 manifest payload hash (reserved area).
pub const OFF_L1_CONTENT_HASH: usize = 0xFD0;
/// Offset of the signer fingerprint (reserved area).
pub const OFF_SIGNER_FINGERPRINT: usize = 0xFE0;
/// Segment id of the L1 manifest segment (u64, signed). Zero in pages written
/// before F2-06; verifiers then skip that one pin.
pub const OFF_L1_SEGMENT_ID: usize = 0xFF0;
const OFF_CHECKSUM: usize = 0xFFC;

/// What a freshly written page commits to.
pub struct PageContent<'a> {
    /// Offset of the MANIFEST segment header.
    pub l1_offset: u64,
    /// Length of the MANIFEST segment (header + payload).
    pub l1_length: u64,
    /// SHAKE-256-128 of the MANIFEST payload.
    pub l1_content_hash: [u8; 16],
    /// Segment id of the MANIFEST segment.
    pub l1_segment_id: u64,
    /// Live vector count.
    pub total_vectors: u64,
    /// Vector dimension.
    pub dimension: u16,
    /// Hardware profile id.
    pub profile: u8,
    /// Manifest epoch.
    pub epoch: u32,
    /// File lineage identity.
    pub identity: &'a FileIdentity,
}

/// Fingerprint of an Ed25519 public key: SHAKE-256-128 of its 32 bytes.
pub fn signer_fingerprint(public_key: &[u8; 32]) -> [u8; 16] {
    compute_shake256_128(public_key)
}

/// The bytes the page signature covers: the page with the signature region
/// and the trailing CRC32C zeroed. Identical to `rvf-forge-core`'s
/// `root_manifest_signing_bytes`.
pub fn signing_bytes(page: &[u8; PAGE_SIZE]) -> [u8; PAGE_SIZE] {
    let mut signable = *page;
    signable[OFF_SIGNATURE..OFF_SIGNATURE + ED25519_SIGNATURE_LEN].fill(0);
    signable[OFF_CHECKSUM..].fill(0);
    signable
}

/// Build and sign a root page with an Ed25519 secret key.
#[cfg(feature = "ed25519")]
pub fn build_signed_page(content: &PageContent<'_>, secret: &[u8; 32]) -> [u8; PAGE_SIZE] {
    use rvf_types::ed25519::{ed25519_sign, Ed25519Keypair};

    let mut page = [0u8; PAGE_SIZE];
    page[..4].copy_from_slice(&ROOT_MANIFEST_MAGIC.to_le_bytes());
    page[OFF_VERSION..OFF_VERSION + 2].copy_from_slice(&1u16.to_le_bytes());
    page[OFF_L1_OFFSET..OFF_L1_OFFSET + 8].copy_from_slice(&content.l1_offset.to_le_bytes());
    page[OFF_L1_LENGTH..OFF_L1_LENGTH + 8].copy_from_slice(&content.l1_length.to_le_bytes());
    page[OFF_TOTAL_VEC..OFF_TOTAL_VEC + 8].copy_from_slice(&content.total_vectors.to_le_bytes());
    page[OFF_DIM..OFF_DIM + 2].copy_from_slice(&content.dimension.to_le_bytes());
    page[OFF_DTYPE] = 1; // f32
    page[OFF_PROFILE] = content.profile;
    page[OFF_EPOCH..OFF_EPOCH + 4].copy_from_slice(&content.epoch.to_le_bytes());
    // created_ns / modified_ns stay zero: pages are reproducible.
    page[OFF_IDENTITY..OFF_IDENTITY + 68].copy_from_slice(&content.identity.to_bytes());
    page[OFF_L1_CONTENT_HASH..OFF_L1_CONTENT_HASH + 16].copy_from_slice(&content.l1_content_hash);
    page[OFF_L1_SEGMENT_ID..OFF_L1_SEGMENT_ID + 8]
        .copy_from_slice(&content.l1_segment_id.to_le_bytes());

    let public = Ed25519Keypair::from_secret(secret).public_key();
    page[OFF_SIGNER_FINGERPRINT..OFF_SIGNER_FINGERPRINT + 16]
        .copy_from_slice(&signer_fingerprint(&public));
    // Algorithm and length are written before signing so the signature covers
    // them; a downgrade of `sig_algo` then invalidates the page.
    page[OFF_SIG_ALGO..OFF_SIG_ALGO + 2].copy_from_slice(&SIG_ALGO_ED25519.to_le_bytes());
    page[OFF_SIG_LEN..OFF_SIG_LEN + 2]
        .copy_from_slice(&(ED25519_SIGNATURE_LEN as u16).to_le_bytes());
    let sig = ed25519_sign(secret, &signing_bytes(&page));
    page[OFF_SIGNATURE..OFF_SIGNATURE + ED25519_SIGNATURE_LEN].copy_from_slice(&sig);
    reseal_crc(&mut page);
    page
}

/// Recompute the trailing CRC32C after editing a page (used by tests that
/// model an attacker who fixes the checksum but cannot re-sign).
pub fn reseal_crc(page: &mut [u8; PAGE_SIZE]) {
    let crc = compute_crc32c(&page[..OFF_CHECKSUM]);
    page[OFF_CHECKSUM..].copy_from_slice(&crc.to_le_bytes());
}

/// Parsed, **unverified** view of a page.
#[derive(Clone, Copy, Debug)]
pub struct PageView {
    /// `RVM0` magic present.
    pub magic_ok: bool,
    /// CRC32C matches.
    pub crc_ok: bool,
    /// `sig_algo` (0 = unsigned).
    pub sig_algo: u16,
    /// `sig_length`.
    pub sig_len: u16,
    /// L1 manifest segment offset.
    pub l1_offset: u64,
    /// L1 manifest segment length (header + payload).
    pub l1_length: u64,
    /// Claimed SHAKE-256-128 of the L1 payload.
    pub l1_content_hash: [u8; 16],
    /// Claimed signer fingerprint.
    pub signer_fingerprint: [u8; 16],
    /// Claimed manifest segment id (0 = not recorded).
    pub l1_segment_id: u64,
}

/// Parse the fixed fields of a page. Verifies nothing cryptographic.
pub fn view(page: &[u8; PAGE_SIZE]) -> PageView {
    let u64_at = |o: usize| u64::from_le_bytes(page[o..o + 8].try_into().expect("8 bytes"));
    let u16_at = |o: usize| u16::from_le_bytes(page[o..o + 2].try_into().expect("2 bytes"));
    let crc = u32::from_le_bytes(page[OFF_CHECKSUM..].try_into().expect("4 bytes"));
    let mut l1h = [0u8; 16];
    l1h.copy_from_slice(&page[OFF_L1_CONTENT_HASH..OFF_L1_CONTENT_HASH + 16]);
    let mut fp = [0u8; 16];
    fp.copy_from_slice(&page[OFF_SIGNER_FINGERPRINT..OFF_SIGNER_FINGERPRINT + 16]);
    PageView {
        magic_ok: page[..4] == ROOT_MANIFEST_MAGIC.to_le_bytes(),
        crc_ok: crc == compute_crc32c(&page[..OFF_CHECKSUM]),
        sig_algo: u16_at(OFF_SIG_ALGO),
        sig_len: u16_at(OFF_SIG_LEN),
        l1_offset: u64_at(OFF_L1_OFFSET),
        l1_length: u64_at(OFF_L1_LENGTH),
        l1_content_hash: l1h,
        signer_fingerprint: fp,
        l1_segment_id: u64_at(OFF_L1_SEGMENT_ID),
    }
}

/// Verify the Ed25519 signature on `page` against `public_key`.
///
/// `false` for an unsigned page, a different algorithm, a bad length, or a
/// signature that does not verify. Always `false` without the `ed25519`
/// feature (fail closed).
pub fn verify_signature(page: &[u8; PAGE_SIZE], public_key: &[u8; 32]) -> bool {
    #[cfg(feature = "ed25519")]
    {
        let v = view(page);
        if v.sig_algo != SIG_ALGO_ED25519 || v.sig_len as usize != ED25519_SIGNATURE_LEN {
            return false;
        }
        let mut sig = [0u8; ED25519_SIGNATURE_LEN];
        sig.copy_from_slice(&page[OFF_SIGNATURE..OFF_SIGNATURE + ED25519_SIGNATURE_LEN]);
        // `verify_strict` (rejects small-order keys and non-canonical
        // signatures), as the stream descriptor does.
        use ed25519_dalek::{Signature, VerifyingKey};
        let Ok(vk) = VerifyingKey::from_bytes(public_key) else {
            return false;
        };
        vk.verify_strict(&signing_bytes(page), &Signature::from_bytes(&sig))
            .is_ok()
    }
    #[cfg(not(feature = "ed25519"))]
    {
        let _ = (page, public_key);
        false
    }
}

#[cfg(all(test, feature = "ed25519"))]
mod tests {
    use super::*;

    fn content(identity: &FileIdentity) -> PageContent<'_> {
        PageContent {
            l1_offset: 4096,
            l1_length: 200,
            l1_content_hash: [9; 16],
            l1_segment_id: 5,
            total_vectors: 7,
            dimension: 4,
            profile: 0,
            epoch: 3,
            identity,
        }
    }

    #[test]
    fn build_verify_round_trip_and_wrong_key() {
        let id = FileIdentity::new_root([1; 16]);
        let page = build_signed_page(&content(&id), &[5u8; 32]);
        let public = rvf_types::ed25519::Ed25519Keypair::from_secret(&[5u8; 32]).public_key();
        let other = rvf_types::ed25519::Ed25519Keypair::from_secret(&[6u8; 32]).public_key();
        assert!(view(&page).magic_ok && view(&page).crc_ok);
        assert!(verify_signature(&page, &public));
        assert!(!verify_signature(&page, &other));
        assert_eq!(view(&page).signer_fingerprint, signer_fingerprint(&public));
    }

    #[test]
    fn any_signed_field_edit_breaks_the_signature_even_with_a_fixed_crc() {
        let id = FileIdentity::new_root([1; 16]);
        let page = build_signed_page(&content(&id), &[5u8; 32]);
        let public = rvf_types::ed25519::Ed25519Keypair::from_secret(&[5u8; 32]).public_key();
        for off in [
            OFF_L1_OFFSET,
            OFF_L1_LENGTH,
            OFF_L1_CONTENT_HASH,
            OFF_SIG_ALGO,
            OFF_EPOCH,
        ] {
            let mut p = page;
            p[off] ^= 1;
            reseal_crc(&mut p);
            assert!(view(&p).crc_ok);
            assert!(!verify_signature(&p, &public), "offset {off:#x}");
        }
    }
}
