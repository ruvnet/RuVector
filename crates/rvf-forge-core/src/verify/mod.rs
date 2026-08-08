//! Verification: root-manifest signature, per-segment hashes, and the
//! unsigned-executable rule, each producing a witness record.
//!
//! Two design points are worth stating up front, because both are required
//! rather than stylistic.
//!
//! **Failure returns a report, not an error.** RVM loading requirement §4.9
//! calls for a witness record for *every* verification result, and a refusal
//! is a result. Returning `Err` on the first bad hash would discard exactly
//! the evidence the refusal needs to be auditable. So [`verify`] returns
//! `Ok(report)` whenever the container parses, with `report.ok == false` and a
//! record per check. `Err` is reserved for I/O and for bytes that are not an
//! RVF at all. Callers that want the failure as an error call
//! [`VerificationReport::into_result`].
//!
//! **Unsigned executables are refused by default.** RVM loading requirement
//! §4.3 and ADR-283 invariant 3 both require it.
//! [`VerifyOptions::allow_unsigned_executable`] exists for the unsigned
//! development builds of ADR-290 §5, which are labeled as such in provenance
//! and are not eligible for distribution.

mod checks;

use crate::canonical::hex_lower;
use crate::container;
use crate::error::{ForgeError, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Schema identifier carried by every [`VerificationReport`].
pub const VERIFICATION_SCHEMA: &str = "rvforge.verification/1";

/// Which check a record describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CheckKind {
    /// A root manifest segment is present.
    RootManifestPresent,
    /// A segment's payload matches the content hash in its header.
    ContentHash,
    /// A segment's Ed25519 signature verifies against a trusted key.
    Signature,
    /// An executable segment carries a signature at all.
    ExecutableSigned,
    /// The container's SHA-256 matches the caller's expected identity.
    Identity,
}

/// The result of one check.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Outcome {
    /// The check passed.
    Pass,
    /// The check failed. Any failure makes the whole report `ok == false`.
    Fail,
    /// The check did not apply, or could not run because no trusted key was
    /// supplied. A skip never makes a report fail on its own.
    Skip,
}

/// One witness-style record. Every check performed produces exactly one,
/// whether it passed, failed, or was skipped.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VerificationRecord {
    /// Which check this records.
    pub check: CheckKind,
    /// Its result.
    pub outcome: Outcome,
    /// Stream position of the segment checked, when the check is per-segment.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segment_index: Option<usize>,
    /// Ordinal of the segment checked, when the check is per-segment.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segment_id: Option<u64>,
    /// Human-readable type name of the segment checked.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segment_type: Option<String>,
    /// Why the check produced this outcome.
    pub detail: String,
}

/// The full result of verifying one artifact.
///
/// Deliberately carries no timestamp: two verifications of the same bytes
/// under the same options produce byte-identical reports, which is what makes
/// a report comparable across the build workers of a single build.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VerificationReport {
    /// Schema identifier; always [`VERIFICATION_SCHEMA`].
    pub schema: String,
    /// SHA-256 of the whole container, lowercase hex.
    pub rvf_identity: String,
    /// True when no record failed.
    pub ok: bool,
    /// How many segments were examined.
    pub segment_count: usize,
    /// One record per check performed, in check order.
    pub records: Vec<VerificationRecord>,
}

impl VerificationReport {
    /// The failing records.
    pub fn failures(&self) -> Vec<&VerificationRecord> {
        self.records
            .iter()
            .filter(|r| r.outcome == Outcome::Fail)
            .collect()
    }

    /// Convert a failed report into the matching [`ForgeError`].
    ///
    /// An unsigned executable segment maps to
    /// [`ForgeError::UnsignedSegment`], since that is its own CLI error code;
    /// every other failure maps to [`ForgeError::VerifyFailed`].
    ///
    /// # Errors
    ///
    /// The error described above whenever `self.ok` is false.
    pub fn into_result(self) -> Result<Self> {
        if self.ok {
            return Ok(self);
        }
        let unsigned = self
            .records
            .iter()
            .find(|r| r.outcome == Outcome::Fail && r.check == CheckKind::ExecutableSigned);
        if let Some(rec) = unsigned {
            return Err(ForgeError::UnsignedSegment {
                segment_id: rec.segment_id.unwrap_or_default(),
                segment_type: rec.segment_type.clone().unwrap_or_else(|| "unknown".into()),
            });
        }
        let detail = self
            .failures()
            .iter()
            .map(|r| r.detail.clone())
            .collect::<Vec<_>>()
            .join("; ");
        Err(ForgeError::VerifyFailed { detail })
    }
}

/// How to verify.
///
/// [`VerifyOptions::default`] is the strict posture: unsigned executables are
/// refused, and no key is trusted until one is supplied.
#[derive(Clone, Debug, Default)]
pub struct VerifyOptions {
    /// Ed25519 public keys whose signatures are accepted. With none supplied,
    /// signature checks are recorded as skipped rather than passed — an
    /// unverifiable signature is never treated as a valid one.
    pub trusted_keys: Vec<[u8; 32]>,
    /// Permit executable segments with no signature. Off by default; see the
    /// module docs.
    pub allow_unsigned_executable: bool,
    /// When set, the container's SHA-256 must equal this value.
    pub expected_identity: Option<String>,
}

impl VerifyOptions {
    /// Strict defaults with the given trusted keys.
    pub fn with_trusted_keys(keys: Vec<[u8; 32]>) -> Self {
        Self {
            trusted_keys: keys,
            ..Self::default()
        }
    }

    /// Require the container to hash to `identity`.
    pub fn expect_identity(mut self, identity: impl Into<String>) -> Self {
        self.expected_identity = Some(identity.into());
        self
    }
}

/// Verify the RVF at `path`.
///
/// # Errors
///
/// [`ForgeError::Io`] if the file cannot be read, [`ForgeError::InvalidRvf`]
/// if it is not a well-formed container. Verification *failures* are reported
/// in the returned value, not raised; see the module docs.
pub fn verify(path: impl AsRef<Path>, opts: &VerifyOptions) -> Result<VerificationReport> {
    let path = path.as_ref();
    let data = std::fs::read(path).map_err(|e| ForgeError::io(path, &e))?;
    verify_bytes(&data, opts)
}

/// Verify an in-memory RVF container.
///
/// # Errors
///
/// [`ForgeError::InvalidRvf`] if `data` is not a well-formed container.
pub fn verify_bytes(data: &[u8], opts: &VerifyOptions) -> Result<VerificationReport> {
    let segments = container::walk(data)?;
    let identity = hex_lower(&rvf_types::sha256(data));
    let mut records = Vec::new();

    if let Some(expected) = &opts.expected_identity {
        let matches = expected.eq_ignore_ascii_case(&identity);
        records.push(checks::record(
            CheckKind::Identity,
            if matches {
                Outcome::Pass
            } else {
                Outcome::Fail
            },
            None,
            if matches {
                format!("container hashes to the expected identity {identity}")
            } else {
                format!("expected identity {expected}, container hashes to {identity}")
            },
        ));
    }

    records.push(checks::root_manifest_record(&segments));

    for seg in &segments {
        records.push(checks::content_hash_record(data, seg));
        if seg.is_executable() {
            records.push(checks::executable_signed_record(seg, opts));
        }
        if seg.is_signed() {
            records.push(checks::signature_record(data, seg, opts));
        }
    }

    let ok = !records.iter().any(|r| r.outcome == Outcome::Fail);
    Ok(VerificationReport {
        schema: VERIFICATION_SCHEMA.to_string(),
        rvf_identity: identity,
        ok,
        segment_count: segments.len(),
        records,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{signed_segment, unsigned_segment, TestKeypair};
    use rvf_types::SegmentFlags;
    use rvf_types::SegmentType;
    use rvf_wire::write_segment;

    fn manifest_only() -> Vec<u8> {
        write_segment(
            SegmentType::Manifest as u8,
            b"root",
            SegmentFlags::empty(),
            1,
        )
    }

    #[test]
    fn a_clean_container_verifies() {
        let report = verify_bytes(&manifest_only(), &VerifyOptions::default()).unwrap();
        assert!(report.ok, "{:?}", report.failures());
        assert_eq!(report.segment_count, 1);
    }

    #[test]
    fn every_segment_produces_a_content_hash_record() {
        let mut data = manifest_only();
        data.extend(unsigned_segment(SegmentType::Vec, b"vectors", 2));
        let report = verify_bytes(&data, &VerifyOptions::default()).unwrap();
        let hash_records: Vec<_> = report
            .records
            .iter()
            .filter(|r| r.check == CheckKind::ContentHash)
            .collect();
        assert_eq!(hash_records.len(), 2);
        assert!(hash_records.iter().all(|r| r.outcome == Outcome::Pass));
    }

    #[test]
    fn tampered_payload_fails_its_content_hash() {
        let mut data = manifest_only();
        data.extend(unsigned_segment(SegmentType::Vec, b"vectors", 2));
        // Segment 1 occupies bytes 0..128 after alignment, so segment 2's
        // header is at 128 and its payload begins at 192. Flip a payload byte,
        // leaving both headers intact so the walk still succeeds and only the
        // content-hash check fails.
        data[192] ^= 0xff;

        let report = verify_bytes(&data, &VerifyOptions::default()).unwrap();
        assert!(!report.ok);
        let failures = report.failures();
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].check, CheckKind::ContentHash);
        assert_eq!(failures[0].segment_id, Some(2));

        let err = report.into_result().unwrap_err();
        assert_eq!(err.code(), crate::ErrorCode::VerifyFailed);
    }

    #[test]
    fn unsigned_executable_is_refused_by_default() {
        let mut data = manifest_only();
        data.extend(unsigned_segment(SegmentType::Wasm, b"\0asm", 5));
        let report = verify_bytes(&data, &VerifyOptions::default()).unwrap();
        assert!(!report.ok);
        let err = report.into_result().unwrap_err();
        assert_eq!(err.code(), crate::ErrorCode::UnsignedSegment);
        assert!(err.to_string().contains("wasm"), "{err}");
    }

    #[test]
    fn unsigned_executable_is_permitted_for_development_builds() {
        let mut data = manifest_only();
        data.extend(unsigned_segment(SegmentType::Wasm, b"\0asm", 5));
        let opts = VerifyOptions {
            allow_unsigned_executable: true,
            ..VerifyOptions::default()
        };
        let report = verify_bytes(&data, &opts).unwrap();
        assert!(report.ok, "{:?}", report.failures());
        let rec = report
            .records
            .iter()
            .find(|r| r.check == CheckKind::ExecutableSigned)
            .unwrap();
        assert_eq!(rec.outcome, Outcome::Skip);
    }

    #[test]
    fn signed_executable_verifies_against_its_key() {
        let kp = TestKeypair::deterministic(7);
        let mut data = manifest_only();
        data.extend(signed_segment(SegmentType::Wasm, b"\0asm", 5, &kp));

        let opts = VerifyOptions::with_trusted_keys(vec![kp.public]);
        let report = verify_bytes(&data, &opts).unwrap();
        assert!(report.ok, "{:?}", report.failures());
        let sig = report
            .records
            .iter()
            .find(|r| r.check == CheckKind::Signature)
            .unwrap();
        assert_eq!(sig.outcome, Outcome::Pass);
    }

    #[test]
    fn signature_fails_against_an_untrusted_key() {
        let signer = TestKeypair::deterministic(7);
        let stranger = TestKeypair::deterministic(8);
        let mut data = manifest_only();
        data.extend(signed_segment(SegmentType::Wasm, b"\0asm", 5, &signer));

        let opts = VerifyOptions::with_trusted_keys(vec![stranger.public]);
        let report = verify_bytes(&data, &opts).unwrap();
        assert!(!report.ok);
        assert_eq!(
            report.into_result().unwrap_err().code(),
            crate::ErrorCode::VerifyFailed
        );
    }

    #[test]
    fn signature_is_skipped_not_passed_when_no_key_is_trusted() {
        let kp = TestKeypair::deterministic(7);
        let mut data = manifest_only();
        data.extend(signed_segment(SegmentType::Wasm, b"\0asm", 5, &kp));

        let report = verify_bytes(&data, &VerifyOptions::default()).unwrap();
        let sig = report
            .records
            .iter()
            .find(|r| r.check == CheckKind::Signature)
            .unwrap();
        assert_eq!(sig.outcome, Outcome::Skip);
        // A skip is not a failure, and the executable is signed, so this passes.
        assert!(report.ok, "{:?}", report.failures());
    }

    #[test]
    fn missing_root_manifest_fails() {
        let data = unsigned_segment(SegmentType::Vec, b"vectors", 1);
        let report = verify_bytes(&data, &VerifyOptions::default()).unwrap();
        assert!(!report.ok);
        assert_eq!(report.failures()[0].check, CheckKind::RootManifestPresent);
    }

    #[test]
    fn identity_mismatch_fails() {
        let data = manifest_only();
        let opts = VerifyOptions::default().expect_identity("00".repeat(32));
        let report = verify_bytes(&data, &opts).unwrap();
        assert!(!report.ok);
        assert_eq!(report.failures()[0].check, CheckKind::Identity);
    }

    #[test]
    fn identity_match_passes() {
        let data = manifest_only();
        let identity = hex_lower(&rvf_types::sha256(&data));
        let opts = VerifyOptions::default().expect_identity(identity);
        assert!(verify_bytes(&data, &opts).unwrap().ok);
    }

    #[test]
    fn malformed_container_is_an_error_not_a_report() {
        let err = verify_bytes(b"not an rvf", &VerifyOptions::default()).unwrap_err();
        assert_eq!(err.code(), crate::ErrorCode::InvalidRvf);
    }

    #[test]
    fn report_is_deterministic_and_serializable() {
        let data = manifest_only();
        let a = verify_bytes(&data, &VerifyOptions::default()).unwrap();
        let b = verify_bytes(&data, &VerifyOptions::default()).unwrap();
        assert_eq!(a, b);
        let json = serde_json::to_string(&a).unwrap();
        assert!(json.contains("\"rvfIdentity\""), "{json}");
        assert!(json.contains("\"root-manifest-present\""), "{json}");
    }
}
