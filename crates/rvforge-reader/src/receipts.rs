//! The local witness receipt log (ADR-284 §1 requirement 9).
//!
//! Every verification result is appended here, whether it passed or failed. A
//! refused package is the case the log exists for: without a record, a refusal
//! is indistinguishable from a verification that was never run.
//!
//! The log is append-only JSON Lines under the reader's state directory, one
//! [`Receipt`] per line, each embedding the full `rvf-forge-core`
//! [`VerificationReport`] so the individual check records survive. It records
//! filesystem paths, so it is created `0600` on Unix.

use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};

use rvf_forge_core::VerificationReport;

/// Schema identifier carried by every [`Receipt`].
pub const RECEIPT_SCHEMA: &str = "rvforge.reader.receipt/1";

/// One verification result, as written to the log.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Receipt {
    /// Always [`RECEIPT_SCHEMA`].
    pub schema: String,
    /// Seconds since the Unix epoch. The report itself is deliberately
    /// timestamp-free so it stays reproducible; the *receipt* is an audit
    /// entry, and when a check ran is part of the audit.
    pub recorded_at_unix: u64,
    /// The package this result is about.
    pub subject_path: String,
    /// The container's SHA-256, when it parsed.
    pub rvf_identity: Option<String>,
    /// Whether verification passed.
    pub ok: bool,
    /// Why, when it did not. Empty on success.
    pub detail: String,
    /// The per-check witness records. `None` when the bytes were not a
    /// container, so no check could run.
    pub report: Option<VerificationReport>,
}

impl Receipt {
    /// Build a receipt, stamping it with the current wall-clock time.
    pub fn new(
        subject_path: &Path,
        rvf_identity: Option<String>,
        ok: bool,
        detail: String,
        report: Option<VerificationReport>,
    ) -> Self {
        Self {
            schema: RECEIPT_SCHEMA.to_string(),
            recorded_at_unix: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            subject_path: subject_path.to_string_lossy().into_owned(),
            rvf_identity,
            ok,
            detail,
            report,
        }
    }
}

/// An append-only log of verification receipts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceiptLog {
    path: PathBuf,
}

impl ReceiptLog {
    /// The log under the reader's per-user state root.
    pub fn default_log() -> Self {
        Self::at(crate::state::default_state_root().join("receipts.jsonl"))
    }

    /// A log at an explicit path.
    pub fn at(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// Where entries are written.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append one receipt.
    ///
    /// # Errors
    ///
    /// Any I/O error from creating the parent directory, opening the file, or
    /// writing the line. Callers surface the failure rather than treating an
    /// unwritten receipt as a written one.
    pub fn append(&self, receipt: &Receipt) -> std::io::Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let line = serde_json::to_string(receipt)
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
        file.flush()
    }

    /// Every receipt in the log, oldest first.
    ///
    /// A line that does not parse is skipped rather than failing the read: a
    /// corrupt entry must not hide the entries around it.
    ///
    /// # Errors
    ///
    /// Any I/O error from reading the file. A log that does not exist yet reads
    /// as empty.
    pub fn read_all(&self) -> std::io::Result<Vec<Receipt>> {
        let text = match std::fs::read_to_string(&self.path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };
        Ok(text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect())
    }
}
