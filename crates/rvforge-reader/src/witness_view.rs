//! The witness viewer (requirements P15.11 and P8, ADR-295 §2 element 8).
//!
//! A witness record is only worth showing if the viewer re-derives it. This
//! module reads a receipts file and, per subject, checks two things for every
//! record:
//!
//! 1. **Content-address recomputation.** `receiptId` is the SHA-256 of the
//!    canonical JSON of the receipt with `signatures` and `receiptId` removed
//!    (`registry-model.md`, `WitnessReceipt`). Editing any other field changes
//!    the recomputed id, so an edited receipt cannot keep its id.
//! 2. **`prevReceipt` continuity.** The first receipt for a subject carries
//!    `prevReceipt: null`; every later one carries the previous receipt's id.
//!    Reordering a chain therefore breaks it even when every individual record
//!    is intact.
//!
//! The first record that fails either check is where the chain stops being
//! evidence, and it is reported by index rather than as a bare "invalid" — the
//! records before it verified, and hiding that would overstate the damage as
//! much as reporting "valid" would understate it.
//!
//! # Honest statuses
//!
//! The reader's own [`crate::receipts::Receipt`] log is an audit trail, not a
//! hash chain: those records carry no `receiptId` and no `prevReceipt`, so
//! there is nothing to verify. They are rendered as
//! [`ChainStatus::Unchained`], never as [`ChainStatus::Valid`], and the dock
//! reports [`WitnessStatus::Unavailable`] for them. Inventing ids so that a
//! local log could display as a valid chain is exactly the failure this viewer
//! exists to make visible.

use serde::Serialize;
use std::path::Path;

use crate::dock::WitnessStatus;
use crate::receipts::{Receipt, ReceiptLog, RECEIPT_SCHEMA};

pub use crate::witness_receipt::{
    recompute_id, short_id, Actor, Evidence, WitnessReceipt, DIGEST_PREFIX, WITNESS_RECEIPT_TYPE,
    WITNESS_SCHEMA_VERSION,
};

/// How a chain came out.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum ChainStatus {
    /// Every receipt recomputed to its declared id and every link matched.
    Valid,
    /// The chain verified up to `index` and stopped there.
    BrokenAt { index: usize, reason: String },
    /// There were no records at all.
    Empty,
    /// Records exist but carry no chain to verify — the reader's own local
    /// receipt log. Never reported as valid.
    Unchained,
}

impl ChainStatus {
    /// The compact label the UI and the acceptance tests use:
    /// `valid`, `broken-at-N`, `empty`, `unchained`.
    pub fn label(&self) -> String {
        match self {
            Self::Valid => "valid".to_string(),
            Self::BrokenAt { index, .. } => format!("broken-at-{index}"),
            Self::Empty => "empty".to_string(),
            Self::Unchained => "unchained".to_string(),
        }
    }

    /// What the dock's witness-status element shows for this chain. Anything
    /// short of a fully verified chain is `Broken` or `Unavailable`; there is
    /// no neutral rendering.
    pub fn dock_status(&self) -> WitnessStatus {
        match self {
            Self::Valid => WitnessStatus::Valid,
            Self::BrokenAt { .. } => WitnessStatus::Broken,
            Self::Empty | Self::Unchained => WitnessStatus::Unavailable,
        }
    }

    /// Ordering used to fold per-subject statuses into one: the worst answer
    /// wins, so one broken subject is never hidden by a valid one.
    fn severity(&self) -> u8 {
        match self {
            Self::BrokenAt { .. } => 3,
            Self::Unchained => 2,
            Self::Valid => 1,
            Self::Empty => 0,
        }
    }
}

/// One receipt, as rendered.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReceiptRow {
    /// Position within this subject's chain, zero-based.
    pub index: usize,
    pub event: String,
    pub outcome: String,
    /// `kind:id`, from the receipt's actor.
    pub actor: String,
    pub timestamp: String,
    pub id_short: String,
    /// The receipt's own evidence line, or the reader receipt's detail.
    pub detail: String,
    /// False for the reader's local receipts, which carry no chain.
    pub chained: bool,
    /// Set on the one receipt where verification stopped.
    pub break_reason: Option<String>,
}

/// Every receipt recorded for one subject, oldest first.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SubjectChain {
    pub subject: String,
    pub subject_short: String,
    pub status: ChainStatus,
    pub label: String,
    pub receipts: Vec<ReceiptRow>,
}

/// A line that was not a witness record. Kept rather than dropped: an
/// unreadable line is a gap in the evidence, and a silently skipped gap reads
/// as no gap at all.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MalformedLine {
    /// One-based line number in the source file.
    pub line: usize,
    pub reason: String,
}

/// The whole viewer model for one receipts file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WitnessChainView {
    /// Where the records were read from.
    pub source: String,
    pub status: ChainStatus,
    pub label: String,
    /// What the dock's witness-status element reports.
    pub dock_status: WitnessStatus,
    /// One line, suitable for the dock and for the runtime screen.
    pub summary: String,
    pub receipt_count: usize,
    pub subjects: Vec<SubjectChain>,
    pub malformed: Vec<MalformedLine>,
}

/// The chain in the reader's own receipt log.
///
/// A log that does not exist yet reads as empty, which is the honest answer
/// before any package has been verified.
pub fn chain_from_default_log() -> WitnessChainView {
    let log = ReceiptLog::default_log();
    chain_at(log.path()).unwrap_or_else(|e| {
        let mut view = build_view(
            log.path().to_string_lossy().as_ref(),
            Vec::new(),
            Vec::new(),
        );
        view.malformed.push(MalformedLine {
            line: 0,
            reason: format!("the receipt log could not be read: {e}"),
        });
        view
    })
}

/// The chain in an explicit receipts file.
///
/// # Errors
///
/// Any I/O error other than a missing file, which reads as empty.
pub fn chain_at(path: &Path) -> std::io::Result<WitnessChainView> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(e),
    };
    Ok(chain_from_text(path.to_string_lossy().as_ref(), &text))
}

/// One parsed line.
enum Record {
    Chained(Box<WitnessReceipt>),
    Local(Box<Receipt>),
}

/// Parse and verify JSON Lines without touching the filesystem.
pub fn chain_from_text(source: &str, text: &str) -> WitnessChainView {
    let mut records: Vec<Record> = Vec::new();
    let mut malformed: Vec<MalformedLine> = Vec::new();

    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        match parse_line(line) {
            Ok(record) => records.push(record),
            Err(reason) => malformed.push(MalformedLine {
                line: i + 1,
                reason,
            }),
        }
    }

    build_view(source, records, malformed)
}

fn parse_line(line: &str) -> Result<Record, String> {
    let value: serde_json::Value =
        serde_json::from_str(line).map_err(|e| format!("not JSON: {e}"))?;
    let object = value.as_object().ok_or("not a JSON object")?;

    if object.get("type").and_then(|v| v.as_str()) == Some(WITNESS_RECEIPT_TYPE) {
        let receipt: WitnessReceipt = serde_json::from_value(value.clone())
            .map_err(|e| format!("not a readable witness receipt: {e}"))?;
        if receipt.schema_version != WITNESS_SCHEMA_VERSION {
            return Err(format!(
                "witness receipt declares schemaVersion {}, this reader understands {WITNESS_SCHEMA_VERSION}",
                receipt.schema_version
            ));
        }
        return Ok(Record::Chained(Box::new(receipt)));
    }

    if object.get("schema").and_then(|v| v.as_str()) == Some(RECEIPT_SCHEMA) {
        let receipt: Receipt = serde_json::from_value(value)
            .map_err(|e| format!("not a readable reader receipt: {e}"))?;
        return Ok(Record::Local(Box::new(receipt)));
    }

    Err("neither a witness receipt nor a reader receipt".to_string())
}

/// Group by subject, preserving first-appearance order, then verify.
fn build_view(
    source: &str,
    records: Vec<Record>,
    malformed: Vec<MalformedLine>,
) -> WitnessChainView {
    let mut order: Vec<String> = Vec::new();
    let mut grouped: Vec<Vec<&Record>> = Vec::new();

    for record in &records {
        let subject = match record {
            Record::Chained(r) => r.subject.clone(),
            Record::Local(r) => r.subject_path.clone(),
        };
        match order.iter().position(|s| *s == subject) {
            Some(i) => grouped[i].push(record),
            None => {
                order.push(subject);
                grouped.push(vec![record]);
            }
        }
    }

    let subjects: Vec<SubjectChain> = order
        .into_iter()
        .zip(grouped)
        .map(|(subject, records)| verify_subject(subject, &records))
        .collect();

    let receipt_count = records.len();
    let status = fold_status(&subjects, receipt_count, &malformed);
    let summary = summarize(&status, &subjects, receipt_count, &malformed);

    WitnessChainView {
        source: source.to_string(),
        label: status.label(),
        dock_status: status.dock_status(),
        summary,
        receipt_count,
        subjects,
        malformed,
        status,
    }
}

/// Verify one subject's records in file order.
fn verify_subject(subject: String, records: &[&Record]) -> SubjectChain {
    let mut rows: Vec<ReceiptRow> = Vec::new();
    let mut status: Option<ChainStatus> = None;
    let mut previous_id: Option<String> = None;
    let mut any_chained = false;
    let mut all_chained = true;

    for (index, record) in records.iter().enumerate() {
        match record {
            Record::Local(receipt) => {
                all_chained = false;
                rows.push(local_row(index, receipt));
            }
            Record::Chained(receipt) => {
                any_chained = true;
                let failure = check_link(receipt, index, previous_id.as_deref());
                let mut row = chained_row(index, receipt);
                if let Some(reason) = failure {
                    // The first failure is where the chain stops being
                    // evidence. Later records are still listed — they exist —
                    // but the chain is not re-anchored on them.
                    if status.is_none() {
                        row.break_reason = Some(reason.clone());
                        status = Some(ChainStatus::BrokenAt { index, reason });
                    }
                }
                previous_id = Some(receipt.receipt_id.clone());
                rows.push(row);
            }
        }
    }

    let status = status.unwrap_or(if rows.is_empty() {
        ChainStatus::Empty
    } else if any_chained && all_chained {
        ChainStatus::Valid
    } else {
        ChainStatus::Unchained
    });

    SubjectChain {
        subject_short: short_id(&subject),
        subject,
        label: status.label(),
        status,
        receipts: rows,
    }
}

/// Both checks for one link. `None` means it verified.
fn check_link(receipt: &WitnessReceipt, index: usize, previous_id: Option<&str>) -> Option<String> {
    match recompute_id(receipt) {
        Some(computed) if computed == receipt.receipt_id => {}
        Some(computed) => {
            return Some(format!(
                "content does not match receiptId: this record hashes to {}, not {}",
                short_id(&computed),
                short_id(&receipt.receipt_id)
            ));
        }
        None => return Some("the receipt could not be re-encoded to recompute its id".to_string()),
    }

    match (index, previous_id, receipt.prev_receipt.as_deref()) {
        (0, _, None) => None,
        (0, _, Some(prev)) => Some(format!(
            "the first receipt for this subject declares a previous receipt {}",
            short_id(prev)
        )),
        (_, Some(expected), Some(prev)) if prev == expected => None,
        (_, Some(expected), Some(prev)) => Some(format!(
            "prevReceipt is {}, but the receipt before it is {}",
            short_id(prev),
            short_id(expected)
        )),
        (_, Some(expected), None) => Some(format!(
            "prevReceipt is null, but the receipt before it is {}",
            short_id(expected)
        )),
        // No chained predecessor to link to: the records before this one in
        // this subject were unchained local receipts.
        (_, None, _) => Some(
            "this receipt follows records that carry no content address, so the link \
             cannot be checked"
                .to_string(),
        ),
    }
}

fn chained_row(index: usize, receipt: &WitnessReceipt) -> ReceiptRow {
    ReceiptRow {
        index,
        event: receipt.event.clone(),
        outcome: receipt.outcome.clone(),
        actor: format!("{}:{}", receipt.actor.kind, receipt.actor.id),
        timestamp: receipt.timestamp.clone(),
        id_short: short_id(&receipt.receipt_id),
        detail: receipt.evidence.details.clone(),
        chained: true,
        break_reason: None,
    }
}

/// A reader receipt has no event vocabulary of its own: it is always a local
/// verification, and its outcome is the pass/fail it recorded.
fn local_row(index: usize, receipt: &Receipt) -> ReceiptRow {
    ReceiptRow {
        index,
        event: "verify".to_string(),
        outcome: if receipt.ok { "pass" } else { "fail" }.to_string(),
        actor: "reader:local".to_string(),
        timestamp: format!("{} (unix seconds)", receipt.recorded_at_unix),
        id_short: receipt
            .rvf_identity
            .as_deref()
            .map(short_id)
            .unwrap_or_else(|| "—".to_string()),
        detail: if receipt.detail.is_empty() {
            "verification passed".to_string()
        } else {
            receipt.detail.clone()
        },
        chained: false,
        break_reason: None,
    }
}

/// The worst per-subject answer, with two adjustments: nothing at all is
/// `Empty`, and an unreadable line prevents an otherwise-valid file from
/// reporting `Valid`.
fn fold_status(
    subjects: &[SubjectChain],
    receipt_count: usize,
    malformed: &[MalformedLine],
) -> ChainStatus {
    if receipt_count == 0 && malformed.is_empty() {
        return ChainStatus::Empty;
    }
    let worst = subjects
        .iter()
        .map(|s| s.status.clone())
        .max_by_key(|s| s.severity())
        .unwrap_or(ChainStatus::Empty);

    if !malformed.is_empty() && worst.severity() <= ChainStatus::Valid.severity() {
        return ChainStatus::Unchained;
    }
    worst
}

fn summarize(
    status: &ChainStatus,
    subjects: &[SubjectChain],
    receipt_count: usize,
    malformed: &[MalformedLine],
) -> String {
    let unreadable = match malformed.len() {
        0 => String::new(),
        1 => " · 1 unreadable record".to_string(),
        n => format!(" · {n} unreadable records"),
    };
    let plural = |n: usize, word: &str| {
        if n == 1 {
            format!("{n} {word}")
        } else {
            format!("{n} {word}s")
        }
    };

    match status {
        ChainStatus::Empty => format!("No witness records{unreadable}"),
        ChainStatus::Valid => format!(
            "Witness chain valid — {} across {}{unreadable}",
            plural(receipt_count, "receipt"),
            plural(subjects.len(), "subject")
        ),
        ChainStatus::Unchained => format!(
            "{} recorded, no hash chain to verify{unreadable}",
            plural(receipt_count, "receipt")
        ),
        ChainStatus::BrokenAt { index, reason } => {
            let broken = subjects
                .iter()
                .find(|s| matches!(s.status, ChainStatus::BrokenAt { .. }));
            let (subject, total) = broken
                .map(|s| (s.subject_short.clone(), s.receipts.len()))
                .unwrap_or_else(|| ("unknown subject".to_string(), receipt_count));
            format!(
                "Witness chain broken at receipt {} of {total} for {subject}: {reason}{unreadable}",
                index + 1
            )
        }
    }
}
