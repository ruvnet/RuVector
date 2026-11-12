//! Where installations and their state live on disk.
//!
//! One layout per reader: an install root holding one directory per installed
//! agent, and a state root holding the capsules and both receipt logs. They are
//! separate roots because ADR-288 §2 requires state to be deletable without
//! touching the immutable base artifact, and [`crate::state::StateCapsule`]
//! refuses a state root nested inside an install root.

use std::path::{Path, PathBuf};

use crate::receipts::ReceiptLog;
use crate::witness_receipt::WitnessLog;

use super::{io, InstallError, InstallRecord};

/// Where installations and their state live. One layout per reader.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstallLayout {
    install_root: PathBuf,
    state_root: PathBuf,
}

impl InstallLayout {
    pub fn new(install_root: impl Into<PathBuf>, state_root: impl Into<PathBuf>) -> Self {
        Self {
            install_root: install_root.into(),
            state_root: state_root.into(),
        }
    }

    /// The per-user layout: installs beside the state root, never inside it.
    pub fn default_roots() -> Self {
        let state_root = crate::state::default_state_root();
        let install_root = state_root
            .parent()
            .map(|p| p.join("installed"))
            .unwrap_or_else(|| PathBuf::from("installed"));
        Self::new(install_root, state_root)
    }

    pub fn install_root(&self) -> &Path {
        &self.install_root
    }

    pub fn state_root(&self) -> &Path {
        &self.state_root
    }

    pub fn install_dir(&self, install_id: &str) -> PathBuf {
        self.install_root.join(install_id)
    }

    pub fn record_path(&self, install_id: &str) -> PathBuf {
        self.install_dir(install_id).join("install.json")
    }

    /// The verification receipt log.
    pub fn receipt_log(&self) -> ReceiptLog {
        ReceiptLog::at(self.state_root.join("receipts.jsonl"))
    }

    /// The chained lifecycle witness log.
    pub fn lifecycle_log(&self) -> WitnessLog {
        WitnessLog::at(self.state_root.join("lifecycle.jsonl"))
    }

    /// Read one installation record.
    ///
    /// # Errors
    ///
    /// [`InstallError::Io`] if the record is missing or unreadable.
    pub fn read_record(&self, install_id: &str) -> Result<InstallRecord, InstallError> {
        let path = self.record_path(install_id);
        let text = std::fs::read_to_string(&path).map_err(|e| io(&path, &e))?;
        serde_json::from_str(&text).map_err(|e| InstallError::Io {
            path: path.display().to_string(),
            message: format!("not a readable install record: {e}"),
        })
    }

    /// Every install id present under the install root, sorted.
    ///
    /// # Errors
    ///
    /// [`InstallError::Io`] if the install root exists but cannot be listed. A
    /// root that does not exist yet lists as empty.
    pub fn install_ids(&self) -> Result<Vec<String>, InstallError> {
        let entries = match std::fs::read_dir(&self.install_root) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(io(&self.install_root, &e)),
        };
        let mut ids: Vec<String> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().join("install.json").is_file())
            .filter_map(|e| e.file_name().into_string().ok())
            .collect();
        ids.sort();
        Ok(ids)
    }

    /// Write a record, replacing any previous one. Used by the update flow,
    /// which rebinds an install id to a new base identity.
    ///
    /// # Errors
    ///
    /// [`InstallError::Io`] for any filesystem or serialization failure.
    pub(crate) fn write_record(&self, record: &InstallRecord) -> Result<(), InstallError> {
        let path = self.record_path(&record.install_id);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| io(parent, &e))?;
        }
        let json = serde_json::to_string_pretty(record).map_err(|e| InstallError::Io {
            path: path.display().to_string(),
            message: e.to_string(),
        })?;
        std::fs::write(&path, json).map_err(|e| io(&path, &e))
    }
}

impl Default for InstallLayout {
    fn default() -> Self {
        Self::default_roots()
    }
}
