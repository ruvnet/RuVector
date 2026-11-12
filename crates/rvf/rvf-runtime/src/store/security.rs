//! Security policy state and the policy-checked open path.
//!
//! ADR-033 section 4 / ADR-354 item 5. Plain `create`, `open` and
//! `open_readonly` keep their historical behaviour, which is exactly
//! [`SecurityPolicy::Permissive`]: nothing is verified. The policy-aware
//! entry points are [`RvfStore::create_signed`], [`RvfStore::open_with_policy`],
//! [`RvfStore::open_with_options`] and [`RvfStore::open_signed`].
//!
//! Chain of trust under `Strict`/`Paranoid`:
//!
//! 1. the file ends in a Level-0 root page (`RVM0`); absent or `sig_algo = 0`
//!    is `UnsignedManifest`;
//! 2. the page CRC32C holds, and its Ed25519 signature verifies under one of
//!    the caller's trusted signers (`UnknownSigner` when the page names a
//!    signer that is not trusted, `InvalidSignature` when it verifies under
//!    none of the matching keys);
//! 3. the page commits to the manifest the tail scan found: same offset,
//!    same length, and SHAKE-256-128 of the manifest payload equals the
//!    signed `l1_content_hash` (`ContentHashMismatch` otherwise);
//! 4. the signed manifest carries a per-segment hash trailer; every segment
//!    read afterwards (vectors, index, kernels, wasm, ...) must have the
//!    committed SHAKE-256-128 hash, so a swapped segment is refused even if
//!    its own header hash was recomputed by the attacker.
//!
//! Which directory did we use? The runtime's own manifest directory, extended
//! with an optional trailer, not `rvf-manifest`'s `SegmentDirEntry`: that
//! codec has 64-byte entries in a different TLV container that nothing in the
//! runtime reads or writes, and adopting it would be a format change, not a
//! policy change.

use super::*;
use rvf_types::security::{SecurityError, SecurityPolicy};
use rvf_wire::hash::compute_shake256_128;

use crate::root_page;

/// What a signed manifest commits to for one directory segment. The hash
/// covers the payload; the header fields are pinned to the directory entry
/// (identity, type, length) or to the constants the writer always emits, so
/// every header byte of a committed segment is authenticated (ADR-354, F-16).
#[derive(Clone, Copy, Debug)]
pub(super) struct Committed {
    pub hash: [u8; 16],
    pub seg_id: u64,
    pub seg_type: u8,
    pub payload_len: u64,
}

impl Committed {
    /// True when `h` is exactly the header this writer would have produced.
    pub(super) fn header_ok(&self, h: &rvf_types::SegmentHeader) -> bool {
        let pad =
            rvf_wire::writer::calculate_padded_size(SEGMENT_HEADER_SIZE, h.payload_length as usize)
                - SEGMENT_HEADER_SIZE
                - h.payload_length as usize;
        h.checksum_algo == crate::hashing::ALGO_SHAKE256_128
            && h.content_hash == self.hash
            && h.segment_id == self.seg_id
            && h.seg_type == self.seg_type
            && h.payload_length == self.payload_len
            && h.version == rvf_types::SEGMENT_VERSION
            && h.flags == 0
            && h.compression == 0
            && h.reserved_0 == 0
            && h.reserved_1 == 0
            && h.uncompressed_len == 0
            && h.timestamp_ns == 0
            && h.alignment_pad as usize == pad
    }
}

/// Manifest facts the signed page must be re-bound to at the next commit.
#[derive(Clone, Copy)]
pub(super) struct LastManifest {
    pub seg_id: u64,
    pub offset: u64,
    /// Header + payload length of the MANIFEST segment.
    pub len: u64,
    /// SHAKE-256-128 of its payload.
    pub hash: [u8; 16],
}

/// Per-store security state.
pub(super) struct SecurityState {
    pub policy: SecurityPolicy,
    pub trusted_signers: Vec<[u8; 32]>,
    /// Secret key; present only for stores that sign (never printed).
    pub signing_key: Option<[u8; 32]>,
    /// True while the file ends in a root page this store wrote or verified.
    pub root_page_present: AtomicBool,
    /// Committed content hash by segment offset, when enforcement is on.
    pub committed: Option<HashMap<u64, Committed>>,
    /// Set when any legacy-dialect byte layout or hash was encountered.
    pub legacy_dialect: AtomicBool,
    pub last_manifest: Option<LastManifest>,
    /// Explicit consent for a handle that cannot re-sign to cut a valid
    /// signed root page by writing (F-12). Default false.
    pub allow_unsigning: bool,
    /// Freshness pin and prefix opt-in from `RvfOptions` (F3-02).
    pub freshness: Option<crate::options::Freshness>,
    pub serve_signed_prefix: bool,
    /// Newest parseable manifest found beyond the signed page, if any.
    pub newer_ignored: Option<(u64, u32)>,
    /// Offset of unsigned bytes following the signed manifest, if any.
    pub unsigned_tail: Option<u64>,
}

impl SecurityState {
    /// Historical behaviour of `create`/`open`/`open_readonly`.
    pub(super) fn permissive() -> Self {
        Self {
            policy: SecurityPolicy::Permissive,
            trusted_signers: Vec::new(),
            signing_key: None,
            root_page_present: AtomicBool::new(false),
            committed: None,
            legacy_dialect: AtomicBool::new(false),
            last_manifest: None,
            allow_unsigning: false,
            freshness: None,
            serve_signed_prefix: false,
            newer_ignored: None,
            unsigned_tail: None,
        }
    }

    /// A store that signs every commit with `key` (created by `create_signed`).
    pub(super) fn signer(key: [u8; 32]) -> Self {
        Self::with(SecurityPolicy::Strict, Vec::new(), Some(key))
    }

    pub(super) fn with(
        policy: SecurityPolicy,
        trusted: Vec<[u8; 32]>,
        key: Option<[u8; 32]>,
    ) -> Self {
        Self {
            policy,
            trusted_signers: trusted,
            signing_key: key,
            ..Self::permissive()
        }
    }
}

/// A committed-hash check failed on read.
#[derive(Debug)]
pub(super) struct CommitMismatch {
    pub offset: u64,
    pub expected: Option<[u8; 16]>,
    pub actual: [u8; 16],
}

impl std::fmt::Display for CommitMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "segment at 0x{:X} does not match its committed hash",
            self.offset
        )
    }
}

impl std::error::Error for CommitMismatch {}

/// Map a Security error to its stable wire code (ADR-033 section 3.5: the
/// code, not the prose, is the contract).
pub fn security_error_code(e: &SecurityError) -> ErrorCode {
    match e {
        SecurityError::UnsignedManifest { .. } => ErrorCode::UnsignedManifest,
        SecurityError::InvalidSignature { .. } => ErrorCode::InvalidSignature,
        SecurityError::UnknownSigner { .. } => ErrorCode::UnknownSigner,
        SecurityError::ContentHashMismatch { .. } => ErrorCode::ContentHashMismatch,
        SecurityError::EpochDriftExceeded { .. } => ErrorCode::EpochDriftExceeded,
        SecurityError::Level1InvalidSignature { .. } => ErrorCode::Level1InvalidSignature,
        // `SecurityError` is `#[non_exhaustive]`: an unknown future failure is
        // still a refusal, mapped to the generic signature failure.
        _ => ErrorCode::InvalidSignature,
    }
}

pub(super) fn read_trailing_page(file: &File) -> Option<[u8; root_page::PAGE_SIZE]> {
    let len = file.metadata().ok()?.len();
    if len < (root_page::PAGE_SIZE + SEGMENT_HEADER_SIZE) as u64 {
        return None;
    }
    let mut reader = BufReader::new(file);
    reader
        .seek(SeekFrom::Start(len - root_page::PAGE_SIZE as u64))
        .ok()?;
    let mut page = [0u8; root_page::PAGE_SIZE];
    reader.read_exact(&mut page).ok()?;
    root_page::view(&page).magic_ok.then_some(page)
}

/// The manifest offset a CRC-valid trailing root page names, if any.
pub(super) fn signed_page_manifest_offset(file: &File) -> Option<u64> {
    let page = read_trailing_page(file)?;
    let v = root_page::view(&page);
    (v.crc_ok && v.sig_algo != 0).then_some(v.l1_offset)
}

/// The signature-level checks (steps 1-3 plus the dialect/trailer rules).
pub(super) fn check_signed_root(
    file: &File,
    trusted: &[[u8; 32]],
    located: &read_path::LocatedManifest,
) -> Result<(), SecurityError> {
    let off = located.offset;
    let unsigned = SecurityError::UnsignedManifest {
        manifest_offset: off,
    };
    let invalid = |phase: &'static str| SecurityError::InvalidSignature {
        manifest_offset: off,
        rejection_phase: phase,
    };
    let page = read_trailing_page(file).ok_or_else(|| unsigned.clone())?;
    let v = root_page::view(&page);
    if v.sig_algo == 0 {
        return Err(unsigned);
    }
    if !v.crc_ok {
        return Err(invalid("level0_crc32c"));
    }
    if v.sig_algo != root_page::SIG_ALGO_ED25519
        || v.sig_len as usize != root_page::ED25519_SIGNATURE_LEN
    {
        return Err(invalid("signature_algorithm"));
    }
    let matching: Vec<&[u8; 32]> = trusted
        .iter()
        .filter(|k| root_page::signer_fingerprint(k) == v.signer_fingerprint)
        .collect();
    if matching.is_empty() {
        return Err(SecurityError::UnknownSigner {
            manifest_offset: off,
            actual_signer: v.signer_fingerprint,
            expected_signer: trusted.first().map(root_page::signer_fingerprint),
        });
    }
    if !matching
        .iter()
        .any(|k| root_page::verify_signature(&page, k))
    {
        return Err(invalid("signature_verification"));
    }
    // The page is authentic. Now bind it to the manifest we are about to use.
    if located.mode != rvf_wire::tail_scan::ScanMode::Aligned
        || located.header.checksum_algo != crate::hashing::ALGO_SHAKE256_128
    {
        return Err(invalid("legacy_dialect"));
    }
    let actual = compute_shake256_128(&located.payload);
    let l1_len = SEGMENT_HEADER_SIZE as u64 + located.payload.len() as u64;
    if v.l1_offset != off || v.l1_length != l1_len || v.l1_content_hash != actual {
        return Err(SecurityError::ContentHashMismatch {
            pointer_name: "l1_manifest",
            expected_hash: v.l1_content_hash,
            actual_hash: actual,
            seg_offset: off,
        });
    }
    // The manifest segment's own header is pinned too (F2-06): the page
    // commits to its payload hash and, for pages that record it, its id; every
    // other header field must be what the writer emits.
    let pad = rvf_wire::writer::calculate_padded_size(SEGMENT_HEADER_SIZE, located.payload.len())
        - SEGMENT_HEADER_SIZE
        - located.payload.len();
    let h = &located.header;
    let header_ok = h.version == rvf_types::SEGMENT_VERSION
        && h.content_hash == actual
        && h.flags == 0
        && h.compression == 0
        && h.reserved_0 == 0
        && h.reserved_1 == 0
        && h.uncompressed_len == 0
        && h.timestamp_ns == 0
        && h.alignment_pad as usize == pad
        && (v.l1_segment_id == 0 || h.segment_id == v.l1_segment_id);
    if !header_ok {
        return Err(SecurityError::ContentHashMismatch {
            pointer_name: "l1_manifest_header",
            expected_hash: v.l1_content_hash,
            actual_hash: h.content_hash,
            seg_offset: off,
        });
    }
    if located.manifest.seg_hashes.is_none() {
        return Err(invalid("segment_hashes_missing"));
    }
    Ok(())
}
