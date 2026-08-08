//! The individual verification checks, one record each.
//!
//! Split out of the parent module so that adding a check does not grow the
//! file that defines the report types. Every function here returns exactly one
//! [`VerificationRecord`] and never panics: an unreadable footer or an
//! unrecognized algorithm becomes a `Fail` or `Skip` record, because a check
//! that cannot run is itself a result worth witnessing.

use super::{CheckKind, Outcome, VerificationRecord, VerifyOptions};
use crate::canonical::hex_lower;
use crate::container::{self, ParsedSegment};
use ed25519_dalek::VerifyingKey;
use rvf_types::{SegmentType, SignatureFooter};

pub(super) fn record(
    check: CheckKind,
    outcome: Outcome,
    seg: Option<&ParsedSegment>,
    detail: String,
) -> VerificationRecord {
    VerificationRecord {
        check,
        outcome,
        segment_index: seg.map(|s| s.index),
        segment_id: seg.map(|s| s.header.segment_id),
        segment_type: seg.map(|s| container::segment_type_name(s.header.seg_type)),
        detail,
    }
}

/// ADR-291 §3 step 1: the root manifest is verified before anything else.
pub(super) fn root_manifest_record(segments: &[ParsedSegment]) -> VerificationRecord {
    match segments
        .iter()
        .rev()
        .find(|s| s.header.seg_type == SegmentType::Manifest as u8)
    {
        Some(seg) => record(
            CheckKind::RootManifestPresent,
            Outcome::Pass,
            Some(seg),
            format!("root manifest segment found at offset {}", seg.offset),
        ),
        None => record(
            CheckKind::RootManifestPresent,
            Outcome::Fail,
            None,
            "container has no MANIFEST segment; there is no root manifest to verify".to_string(),
        ),
    }
}

pub(super) fn content_hash_record(data: &[u8], seg: &ParsedSegment) -> VerificationRecord {
    let payload = &data[seg.payload.clone()];
    match rvf_wire::validate_segment(&seg.header, payload) {
        Ok(()) => record(
            CheckKind::ContentHash,
            Outcome::Pass,
            Some(seg),
            format!(
                "payload matches header content hash {}",
                hex_lower(&seg.header.content_hash)
            ),
        ),
        Err(e) => record(
            CheckKind::ContentHash,
            Outcome::Fail,
            Some(seg),
            format!(
                "payload does not match header content hash {}: {e:?}",
                hex_lower(&seg.header.content_hash)
            ),
        ),
    }
}

pub(super) fn executable_signed_record(
    seg: &ParsedSegment,
    opts: &VerifyOptions,
) -> VerificationRecord {
    let name = container::segment_type_name(seg.header.seg_type);
    if seg.is_signed() {
        return record(
            CheckKind::ExecutableSigned,
            Outcome::Pass,
            Some(seg),
            format!("executable {name} segment carries a signature footer"),
        );
    }
    if opts.allow_unsigned_executable {
        return record(
            CheckKind::ExecutableSigned,
            Outcome::Skip,
            Some(seg),
            format!(
                "executable {name} segment is unsigned; permitted because this is an \
                 unsigned development build"
            ),
        );
    }
    record(
        CheckKind::ExecutableSigned,
        Outcome::Fail,
        Some(seg),
        format!("executable {name} segment is unsigned and unsigned executables are refused"),
    )
}

pub(super) fn signature_record(
    data: &[u8],
    seg: &ParsedSegment,
    opts: &VerifyOptions,
) -> VerificationRecord {
    let Some(footer_range) = seg.footer.clone() else {
        return record(
            CheckKind::Signature,
            Outcome::Fail,
            Some(seg),
            "segment sets the SIGNED flag but carries no footer".to_string(),
        );
    };

    let footer = match rvf_crypto::decode_signature_footer(&data[footer_range]) {
        Ok(f) => f,
        Err(e) => {
            return record(
                CheckKind::Signature,
                Outcome::Fail,
                Some(seg),
                format!("signature footer is malformed: {e:?}"),
            )
        }
    };

    if footer.sig_algo != rvf_types::SignatureAlgo::Ed25519 as u16 {
        return record(
            CheckKind::Signature,
            Outcome::Skip,
            Some(seg),
            format!(
                "signature algorithm {} is not Ed25519; this crate verifies Ed25519 only",
                footer.sig_algo
            ),
        );
    }

    if opts.trusted_keys.is_empty() {
        return record(
            CheckKind::Signature,
            Outcome::Skip,
            Some(seg),
            "no trusted key supplied, so the signature could not be checked".to_string(),
        );
    }

    let payload = &data[seg.payload.clone()];
    match verifying_key_that_signed(&seg.header, payload, &footer, &opts.trusted_keys) {
        Some(key) => record(
            CheckKind::Signature,
            Outcome::Pass,
            Some(seg),
            format!(
                "Ed25519 signature verifies against trusted key {}",
                hex_lower(&key)
            ),
        ),
        None => record(
            CheckKind::Signature,
            Outcome::Fail,
            Some(seg),
            format!(
                "Ed25519 signature does not verify against any of the {} trusted keys",
                opts.trusted_keys.len()
            ),
        ),
    }
}

/// The first trusted key whose signature over this segment verifies.
fn verifying_key_that_signed(
    header: &rvf_types::SegmentHeader,
    payload: &[u8],
    footer: &SignatureFooter,
    trusted: &[[u8; 32]],
) -> Option<[u8; 32]> {
    trusted.iter().copied().find(|raw| {
        VerifyingKey::from_bytes(raw)
            .is_ok_and(|key| rvf_crypto::verify_segment(header, payload, footer, &key))
    })
}
