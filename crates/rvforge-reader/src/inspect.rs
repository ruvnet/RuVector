//! Inspection and verification of an `.rvf` package, via `rvf-forge-core`.
//!
//! Two operations, deliberately distinct (ADR-284 §2):
//!
//! - [`inspect`] reads identity, the segment inventory, and the declared
//!   capability classes. It checks nothing. Its verification status is always
//!   [`VerificationStatus::Unverified`].
//! - [`verify`] runs the root-manifest, per-segment-hash, and
//!   unsigned-executable checks, and writes one witness receipt per result —
//!   pass or fail alike (ADR-284 §1 requirement 9).
//!
//! Three invariants:
//!
//! - **Neither operation executes RVF content** (ADR-284 §1 requirement 7).
//!   `rvf-forge-core` reads segment headers and hashes payloads; nothing maps,
//!   links, or interprets an executable segment.
//! - **"Unverified" means no check ran**, not "the check was inconclusive". A
//!   skipped signature check reports [`SignatureStatus::NotChecked`], never
//!   [`SignatureStatus::Verified`].
//! - **Verification and inspection see the same bytes.** Both operate on a
//!   single read of the file, so a package cannot be swapped between the two.

use serde::{Deserialize, Serialize};
use std::path::Path;

use rvf_forge_core::{
    inspect_bytes, verify_bytes, CheckKind, ForgeInspection, Outcome, VerificationReport,
    VerifyOptions,
};

use crate::capability::CapabilityCard;
use crate::receipts::{Receipt, ReceiptLog};

/// Result of the hash and signature checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum VerificationStatus {
    /// No verification has been performed. Execution must not proceed.
    Unverified,
    /// Every check that ran passed and none failed.
    Verified,
    /// At least one check failed, or the container did not parse.
    Failed,
}

/// What is known about the publisher's signatures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SignatureStatus {
    /// No signature has been checked; the publisher is unknown.
    Unverified,
    /// The container carries no signature at all.
    Unsigned,
    /// A signature is present but could not be checked, because the reader has
    /// no trusted publisher key. Not a pass.
    NotChecked,
    /// A signature verified against a trusted key.
    Verified,
    /// A signature did not verify.
    Invalid,
}

/// One segment, as observed from its header.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentInfo {
    /// Position in the segment stream.
    pub index: usize,
    /// The segment's ordinal.
    pub segment_id: u64,
    /// Type name, e.g. `wasm`, `manifest`, `meta`.
    pub segment_type: String,
    /// Declared payload length in bytes.
    pub payload_length: u64,
    /// A signature footer follows the payload.
    pub signed: bool,
    /// The payload is declared encrypted.
    pub encrypted: bool,
    /// The payload is code a runtime would execute.
    pub executable: bool,
    /// The header's content hash, lowercase hex.
    pub content_hash: String,
}

/// What the reader can say about a package without running it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct InspectionSummary {
    pub path: String,
    pub file_name: String,
    /// The path ends in `.rvf`. A filename check, not a format check.
    pub extension_ok: bool,
    /// `false` when the path does not exist or could not be read.
    pub exists: bool,
    pub size_bytes: u64,
    /// SHA-256 of the whole container — the canonical `rvfIdentity`. `None`
    /// when the bytes are not a well-formed container.
    pub identity: Option<String>,
    /// Always `None`: naming a publisher needs a trust store mapping keys to
    /// identities, and the reader has none. See the module docs.
    pub publisher: Option<String>,
    pub verification: VerificationStatus,
    pub signature: SignatureStatus,
    /// Every segment, in stream order.
    pub segments: Vec<SegmentInfo>,
    /// Capability classes the container declares. Empty means default-deny.
    pub declared_capabilities: Vec<String>,
    pub executable_segment_count: usize,
    /// Ordinals of executable segments carrying no signature. A non-empty list
    /// means verification refuses the package under default options.
    pub unsigned_executable_segments: Vec<u64>,
    pub root_manifest_present: bool,
    /// Always `false`. Neither inspection nor verification executes content.
    pub executed: bool,
    /// Why inspection could not read the container, when it could not.
    pub error: Option<String>,
    pub notes: Vec<String>,
}

/// A verification result together with the inspection it applies to.
///
/// The reader's own fields are snake_case, matching the rest of this crate's
/// command surface; the embedded `report` keeps `rvf-forge-core`'s camelCase
/// schema, because that is the schema receipts are written in.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct VerificationOutcome {
    /// The inspection, with [`InspectionSummary::verification`] and
    /// [`InspectionSummary::signature`] filled in from the checks.
    pub summary: InspectionSummary,
    /// The witness records, one per check. `None` when the bytes were not a
    /// container, so no check could run.
    pub report: Option<VerificationReport>,
    /// Where the receipt was appended, when it could be written.
    pub receipt_path: Option<String>,
}

/// Inspect a package without executing or verifying it.
pub fn inspect(path: &Path) -> InspectionSummary {
    let (data, read_error) = match std::fs::read(path) {
        Ok(d) => (Some(d), None),
        Err(e) => (None, Some(e.to_string())),
    };
    let mut summary = skeleton(path, data.as_deref().map(<[u8]>::len).unwrap_or(0), data.is_some());

    let Some(data) = data else {
        summary.error = read_error;
        summary
            .notes
            .push("No readable file at this path.".to_string());
        finish_notes(&mut summary);
        return summary;
    };

    match inspect_bytes(&data) {
        Ok(inspection) => apply_inspection(&mut summary, &inspection),
        Err(e) => {
            summary.error = Some(e.to_string());
            summary
                .notes
                .push("This file is not a well-formed RVF container.".to_string());
        }
    }
    finish_notes(&mut summary);
    summary
}

/// Verify a package and record the result, writing to the default receipt log.
pub fn verify(path: &Path) -> VerificationOutcome {
    verify_with_receipts(path, &ReceiptLog::default_log())
}

/// Verify a package, appending the witness receipt to `log`.
///
/// Both a pass and a failure produce a receipt: a refused package is an
/// auditable event, not a silent error (ADR-284 §1 requirement 9).
pub fn verify_with_receipts(path: &Path, log: &ReceiptLog) -> VerificationOutcome {
    let mut summary = inspect(path);

    let Ok(data) = std::fs::read(path) else {
        summary.verification = VerificationStatus::Failed;
        let detail = summary
            .error
            .clone()
            .unwrap_or_else(|| "the package could not be read".to_string());
        summary.notes.push(format!("Verification failed: {detail}"));
        return record(summary, None, detail, log, path);
    };

    // Bind the check to the identity inspection computed, so a report can never
    // describe bytes other than the ones inspected.
    let mut opts = VerifyOptions::default();
    if let Some(identity) = &summary.identity {
        opts = opts.expect_identity(identity.clone());
    }

    match verify_bytes(&data, &opts) {
        Ok(report) => {
            apply_report(&mut summary, &report);
            let detail = failure_detail(&report);
            record(summary, Some(report), detail, log, path)
        }
        Err(e) => {
            summary.verification = VerificationStatus::Failed;
            summary.signature = SignatureStatus::Unverified;
            let detail = e.to_string();
            summary.notes.push(format!("Verification failed: {detail}"));
            record(summary, None, detail, log, path)
        }
    }
}

/// The P6 capability card for a package.
///
/// Verification runs first and the card is refused if it does not pass: a card
/// is a statement about what a package is permitted to do, and there is nothing
/// to permit until the package is known to be intact.
pub fn capability_card(path: &Path) -> CapabilityCard {
    capability_card_with_receipts(path, &ReceiptLog::default_log())
}

/// [`capability_card`], with the receipt log named explicitly.
pub fn capability_card_with_receipts(path: &Path, log: &ReceiptLog) -> CapabilityCard {
    card_for(&verify_with_receipts(path, log), path)
}

/// The card for an outcome whose verification has already run.
///
/// Exists so a caller that has just verified a package — the install flow —
/// derives the card the user was shown without verifying the same bytes twice
/// and writing a second receipt for one user action. The refusal rule is
/// unchanged and is applied here, not by the caller.
pub fn card_for(outcome: &VerificationOutcome, path: &Path) -> CapabilityCard {
    if outcome.summary.verification != VerificationStatus::Verified {
        let reason = outcome
            .summary
            .error
            .clone()
            .or_else(|| outcome.report.as_ref().map(failure_detail))
            .filter(|d| !d.is_empty())
            .unwrap_or_else(|| "the package did not verify".to_string());
        return CapabilityCard::manifest_unavailable(&format!(
            "No capability card is shown for an unverified package: {reason}"
        ));
    }

    let declared = &outcome.summary.declared_capabilities;
    let sidecar = sidecar_manifest_path(path).and_then(|p| {
        std::fs::read_to_string(&p)
            .ok()
            .map(|json| (p.display().to_string(), json))
    });

    let derived = match &sidecar {
        Some((_, json)) => CapabilityCard::refined_with_manifest_json(declared, json),
        None => CapabilityCard::from_declared_classes(declared),
    };

    match derived {
        Ok(mut card) => {
            if let Some((path, _)) = sidecar {
                card.notes.push(format!(
                    "Scopes came from the unsigned development sidecar {path}; \
                     the granted classes came from the verified container."
                ));
            }
            card
        }
        Err(e) => CapabilityCard::manifest_unavailable(&format!(
            "The capability declaration was rejected: {e}"
        )),
    }
}

/// `<path>.manifest.json`, when it exists.
fn sidecar_manifest_path(path: &Path) -> Option<std::path::PathBuf> {
    let mut name = path.as_os_str().to_os_string();
    name.push(".manifest.json");
    let candidate = std::path::PathBuf::from(name);
    candidate.is_file().then_some(candidate)
}

fn skeleton(path: &Path, size_bytes: usize, exists: bool) -> InspectionSummary {
    InspectionSummary {
        path: path.to_string_lossy().into_owned(),
        file_name: path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
        extension_ok: path
            .extension()
            .map(|e| e.eq_ignore_ascii_case("rvf"))
            .unwrap_or(false),
        exists,
        size_bytes: size_bytes as u64,
        identity: None,
        publisher: None,
        verification: VerificationStatus::Unverified,
        signature: SignatureStatus::Unverified,
        segments: Vec::new(),
        declared_capabilities: Vec::new(),
        executable_segment_count: 0,
        unsigned_executable_segments: Vec::new(),
        root_manifest_present: false,
        executed: false,
        error: None,
        notes: vec!["Package content was not executed.".to_string()],
    }
}

fn apply_inspection(summary: &mut InspectionSummary, inspection: &ForgeInspection) {
    summary.identity = Some(inspection.rvf_identity.clone());
    summary.executable_segment_count = inspection.executable_segment_count;
    summary.unsigned_executable_segments = inspection.unsigned_executable_segments.clone();
    summary.root_manifest_present = inspection.root_manifest().is_some();
    summary.declared_capabilities = inspection
        .declared_capabilities
        .iter()
        .map(|c| c.to_string())
        .collect();
    summary.segments = inspection
        .segments
        .iter()
        .map(|s| SegmentInfo {
            index: s.index,
            segment_id: s.segment_id,
            segment_type: s.segment_type_name.clone(),
            payload_length: s.payload_length,
            signed: s.signed,
            encrypted: s.encrypted,
            executable: s.executable,
            content_hash: s.content_hash.clone(),
        })
        .collect();

    if !summary.unsigned_executable_segments.is_empty() {
        summary.notes.push(format!(
            "{} executable segment(s) carry no signature; verification refuses them.",
            summary.unsigned_executable_segments.len()
        ));
    }
    if !summary.root_manifest_present {
        summary
            .notes
            .push("This container has no root manifest segment.".to_string());
    }
}

fn apply_report(summary: &mut InspectionSummary, report: &VerificationReport) {
    summary.verification = if report.ok {
        VerificationStatus::Verified
    } else {
        VerificationStatus::Failed
    };
    summary.signature = signature_status(report, summary);

    for failure in report.failures() {
        summary.notes.push(format!("Check failed: {}", failure.detail));
    }
    if summary.signature == SignatureStatus::NotChecked {
        summary.notes.push(
            "Signatures are present but were not checked: no trusted publisher key is \
             configured, so the publisher remains unknown."
                .to_string(),
        );
    }
}

/// Signature status from the records, biased toward the least reassuring answer
/// the evidence supports.
fn signature_status(report: &VerificationReport, summary: &InspectionSummary) -> SignatureStatus {
    let sig_records: Vec<_> = report
        .records
        .iter()
        .filter(|r| r.check == CheckKind::Signature)
        .collect();

    if sig_records.iter().any(|r| r.outcome == Outcome::Fail) {
        return SignatureStatus::Invalid;
    }
    if sig_records.is_empty() {
        return if summary.segments.iter().any(|s| s.signed) {
            SignatureStatus::NotChecked
        } else {
            SignatureStatus::Unsigned
        };
    }
    if sig_records.iter().any(|r| r.outcome == Outcome::Skip) {
        return SignatureStatus::NotChecked;
    }
    SignatureStatus::Verified
}

fn failure_detail(report: &VerificationReport) -> String {
    report
        .failures()
        .iter()
        .map(|r| r.detail.clone())
        .collect::<Vec<_>>()
        .join("; ")
}

fn record(
    summary: InspectionSummary,
    report: Option<VerificationReport>,
    detail: String,
    log: &ReceiptLog,
    path: &Path,
) -> VerificationOutcome {
    let receipt = Receipt::new(
        path,
        summary.identity.clone(),
        summary.verification == VerificationStatus::Verified,
        detail,
        report.clone(),
    );
    let receipt_path = match log.append(&receipt) {
        Ok(()) => Some(log.path().display().to_string()),
        Err(_) => None,
    };
    VerificationOutcome {
        summary,
        report,
        receipt_path,
    }
}

fn finish_notes(summary: &mut InspectionSummary) {
    if !summary.extension_ok {
        summary
            .notes
            .push("Filename does not end in .rvf.".to_string());
    }
    summary.notes.push(
        "Identity and segments were read; no signature has been checked. Run verification \
         before installing."
            .to_string(),
    );
}
