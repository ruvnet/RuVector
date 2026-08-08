//! The `WitnessReceipt` wire shape and its content address.
//!
//! Field-for-field the `registry-model.md` shape, so re-serializing a receipt
//! reproduces the bytes its id was computed over. That is the whole point: the
//! reader recomputes ids rather than trusting the ones in the file, and a
//! recomputation that used a different encoding would disagree with the
//! registry about intact receipts and agree with it about nothing.
//!
//! [`crate::witness_view`] is where the chain is verified; this module only
//! defines what a receipt is and how its address is derived.

use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use rvf_forge_core::canonical::canonical_sha256_hex;

/// The `type` value every chained receipt carries.
pub const WITNESS_RECEIPT_TYPE: &str = "witness-receipt";
/// The prefix on every content address in the registry model.
pub const DIGEST_PREFIX: &str = "sha256:";
/// The schema version this viewer understands.
pub const WITNESS_SCHEMA_VERSION: u32 = 1;
/// Hex digits kept when shortening a content address for display.
const ID_SHORT_HEX: usize = 12;

/// One link in a subject's witness chain, as written by the registry or the
/// CLI.
///
/// `event` and `outcome` are strings rather than enums: a viewer that refused
/// to display a receipt whose event it did not recognise would hide exactly the
/// records an operator most needs to see, and an unknown value hashes the same
/// either way. `deny_unknown_fields` still holds, because a field this reader
/// does not model is a field it cannot hash, and silently dropping it would
/// make a tampered receipt recompute to its declared id.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WitnessReceipt {
    pub schema_version: u32,
    #[serde(rename = "type")]
    pub object_type: String,
    pub receipt_id: String,
    pub subject: String,
    pub event: String,
    pub outcome: String,
    pub actor: Actor,
    pub evidence: Evidence,
    pub timestamp: String,
    pub prev_receipt: Option<String>,
    /// Not verified here — this viewer checks the chain, not the signers — and
    /// removed before the id is recomputed, so its shape is irrelevant.
    #[serde(default)]
    pub signatures: Vec<serde_json::Value>,
}

/// Who performed the witnessed operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Actor {
    pub kind: String,
    pub id: String,
}

/// What the actor observed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Evidence {
    pub details: String,
}

/// Recompute a receipt's content address.
///
/// `signatures` and `receiptId` are removed first: a receipt cannot commit to
/// its own id, and a countersignature added later must not change it. Everything
/// else is covered, so editing any other field moves the address.
pub fn recompute_id(receipt: &WitnessReceipt) -> Option<String> {
    let mut value = serde_json::to_value(receipt).ok()?;
    let map = value.as_object_mut()?;
    map.remove("signatures");
    map.remove("receiptId");
    canonical_sha256_hex(&value)
        .ok()
        .map(|hex| format!("{DIGEST_PREFIX}{hex}"))
}

impl WitnessReceipt {
    /// Build a receipt that chains onto `prev_receipt`, computing its own
    /// content address.
    ///
    /// The id is derived rather than supplied, so a caller cannot mint a
    /// receipt whose declared address disagrees with its contents — that is
    /// exactly the tampering [`crate::witness_view`] exists to catch, and it
    /// would be pointless for the writer to be able to produce it by accident.
    pub fn chained(
        subject: impl Into<String>,
        event: impl Into<String>,
        outcome: impl Into<String>,
        actor: Actor,
        details: impl Into<String>,
        prev_receipt: Option<String>,
    ) -> Self {
        let mut receipt = Self {
            schema_version: WITNESS_SCHEMA_VERSION,
            object_type: WITNESS_RECEIPT_TYPE.to_string(),
            receipt_id: String::new(),
            subject: subject.into(),
            event: event.into(),
            outcome: outcome.into(),
            actor,
            evidence: Evidence {
                details: details.into(),
            },
            timestamp: now_rfc3339(),
            prev_receipt,
            signatures: Vec::new(),
        };
        receipt.receipt_id = recompute_id(&receipt).unwrap_or_default();
        receipt
    }
}

/// The actor RVForge itself records for a lifecycle operation the user asked
/// for. Not a publisher and not the agent: the reader is what observed it.
pub fn reader_actor() -> Actor {
    Actor {
        kind: "reader".to_string(),
        id: "io.ruv.rvforge.reader".to_string(),
    }
}

/// An append-only log of chained witness receipts, one JSON object per line.
///
/// This is where the *lifecycle* is witnessed — install, update, rollback,
/// uninstall, state export — as opposed to [`crate::receipts::ReceiptLog`],
/// which records verification results in the reader's own local shape. Both
/// files are read by the same viewer, and receipts written here are chained per
/// subject, so a missing or reordered lifecycle event is detectable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WitnessLog {
    path: PathBuf,
}

impl WitnessLog {
    pub fn at(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Every receipt in the log, oldest first. A line that does not parse is
    /// skipped here; [`crate::witness_view`] is what reports it as a gap.
    ///
    /// # Errors
    ///
    /// Any I/O error from reading the file. A log that does not exist yet reads
    /// as empty.
    pub fn read_all(&self) -> std::io::Result<Vec<WitnessReceipt>> {
        let text = match std::fs::read_to_string(&self.path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };
        Ok(text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str::<WitnessReceipt>(l).ok())
            .filter(|r| r.object_type == WITNESS_RECEIPT_TYPE)
            .collect())
    }

    /// The most recent receipt id recorded for `subject`, which becomes the
    /// next receipt's `prevReceipt`.
    ///
    /// # Errors
    ///
    /// Any I/O error from reading the log.
    pub fn head_for(&self, subject: &str) -> std::io::Result<Option<String>> {
        Ok(self
            .read_all()?
            .into_iter()
            .rfind(|r| r.subject == subject)
            .map(|r| r.receipt_id))
    }

    /// Append one receipt for `subject`, chained onto that subject's head.
    ///
    /// # Errors
    ///
    /// Any I/O error from reading the existing chain, creating the parent
    /// directory, or writing the line. The receipt is returned so the caller can
    /// report the id it actually recorded.
    pub fn append(
        &self,
        subject: &str,
        event: &str,
        outcome: &str,
        details: &str,
    ) -> std::io::Result<WitnessReceipt> {
        let prev = self.head_for(subject)?;
        let receipt =
            WitnessReceipt::chained(subject, event, outcome, reader_actor(), details, prev);

        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let line = serde_json::to_string(&receipt)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let mut opts = std::fs::OpenOptions::new();
        opts.create(true).append(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let mut file = opts.open(&self.path)?;
        writeln!(file, "{line}")?;
        file.flush()?;
        Ok(receipt)
    }
}

/// The current time as an RFC 3339 UTC timestamp.
///
/// Written out rather than pulled from a date crate: the receipt schema wants
/// one specific string shape, this is the only place the reader produces one,
/// and the civil-date conversion is a closed-form calculation.
fn now_rfc3339() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format_rfc3339(secs)
}

/// Seconds since the Unix epoch, as RFC 3339 UTC.
pub fn format_rfc3339(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let time_of_day = secs % 86_400;
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        time_of_day / 3600,
        (time_of_day % 3600) / 60,
        time_of_day % 60
    )
}

/// Days since 1970-01-01 to a civil date, by Howard Hinnant's `civil_from_days`.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m as u32, d as u32)
}

/// A content address shortened for display, prefix kept so it stays readable
/// as a digest rather than as an opaque token.
pub fn short_id(id: &str) -> String {
    match id.strip_prefix(DIGEST_PREFIX) {
        Some(hex) if hex.len() > ID_SHORT_HEX => {
            format!("{DIGEST_PREFIX}{}…", &hex[..ID_SHORT_HEX])
        }
        _ if id.is_empty() => "—".to_string(),
        _ => id.to_string(),
    }
}
