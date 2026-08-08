//! The registry itself: the operations P7 lists, over one [`InstallLayout`].

use std::path::{Path, PathBuf};

use crate::install::{self, InstallLayout, InstallRecord, InstallRequest};
use crate::inspect::{self, VerificationStatus};
use crate::state::recorded_identity;

use super::fsops::{clear_dir, copy_dir, dir_bytes, sealed_files, unlock_dir};
use super::{
    io, LibraryEntry, LibraryEntryState, LibraryError, LibraryState, StateTransfer,
    UninstallOutcome, UninstallPolicy,
};

/// The installed-agent registry over one [`InstallLayout`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Library {
    layout: InstallLayout,
}

impl Library {
    pub fn new(layout: InstallLayout) -> Self {
        Self { layout }
    }

    pub fn at(install_root: impl Into<PathBuf>, state_root: impl Into<PathBuf>) -> Self {
        Self::new(InstallLayout::new(install_root, state_root))
    }

    pub fn layout(&self) -> &InstallLayout {
        &self.layout
    }

    /// Verify a package and derive the contract, without installing it.
    pub fn offer(&self, rvf_path: &Path) -> install::InstallOffer {
        install::offer(&self.layout, rvf_path)
    }

    /// Install a package into this library.
    ///
    /// # Errors
    ///
    /// Every [`InstallError`], surfaced as [`LibraryError::Install`].
    pub fn install(&self, request: &InstallRequest) -> Result<LibraryEntry, LibraryError> {
        let record = install::install(&self.layout, request)?;
        self.write_state(
            &record.install_id,
            &LibraryEntryState::at(LibraryState::Installed, None),
        )?;
        self.entry_for(record)
    }

    /// Every installed agent, by install id.
    ///
    /// # Errors
    ///
    /// [`LibraryError::Io`] if the install root cannot be listed. An unreadable
    /// individual record is skipped rather than failing the whole listing: one
    /// corrupt entry must not hide the rest of the library.
    pub fn list(&self) -> Result<Vec<LibraryEntry>, LibraryError> {
        let mut entries = Vec::new();
        for id in self.layout.install_ids()? {
            if let Ok(record) = self.layout.read_record(&id) {
                entries.push(self.entry_for(record)?);
            }
        }
        Ok(entries)
    }

    /// One installed agent.
    ///
    /// # Errors
    ///
    /// [`LibraryError::NotInstalled`] when there is no such record.
    pub fn get(&self, install_id: &str) -> Result<LibraryEntry, LibraryError> {
        let record = self
            .layout
            .read_record(install_id)
            .map_err(|_| LibraryError::NotInstalled {
                install_id: install_id.to_string(),
            })?;
        self.entry_for(record)
    }

    pub fn state_of(&self, install_id: &str) -> LibraryState {
        self.read_state(install_id)
            .map(|s| s.state)
            .unwrap_or(LibraryState::Installed)
    }

    /// Re-verify the installed artifact, witnessing the result.
    ///
    /// A failure quarantines the entry: an artifact that stopped verifying is
    /// not something to keep offering as runnable.
    ///
    /// # Errors
    ///
    /// [`LibraryError::NotInstalled`], or [`LibraryError::VerificationFailed`]
    /// when the installed bytes no longer pass.
    pub fn verify(&self, install_id: &str) -> Result<VerificationStatus, LibraryError> {
        let entry = self.get(install_id)?;
        let outcome = inspect::verify_with_receipts(
            &entry.record.base_artifact(),
            &self.layout.receipt_log(),
        );
        if outcome.summary.verification == VerificationStatus::Verified {
            self.witness(install_id, "verify", "passed", "installed artifact verified");
            return Ok(VerificationStatus::Verified);
        }
        let detail = outcome
            .summary
            .notes
            .last()
            .cloned()
            .unwrap_or_else(|| "verification failed".to_string());
        self.force_state(install_id, LibraryState::Quarantined, Some(detail.clone()))?;
        self.witness(install_id, "verify", "failed", &detail);
        Err(LibraryError::VerificationFailed {
            install_id: install_id.to_string(),
            detail,
        })
    }

    /// Open an agent: verify first, then move to [`LibraryState::Running`].
    ///
    /// # Errors
    ///
    /// [`LibraryError::VerificationFailed`] if the artifact no longer verifies,
    /// and [`LibraryError::IllegalTransition`] from a state that is not
    /// runnable.
    pub fn open(&self, install_id: &str) -> Result<LibraryState, LibraryError> {
        // The transition is checked before verification so an archived or
        // quarantined agent is refused for the reason the user needs to hear,
        // rather than passing verification and then being refused anyway.
        self.require_transition(install_id, LibraryState::Running)?;
        self.verify(install_id)?;
        self.move_to(install_id, LibraryState::Running, None)
    }

    pub fn pause(&self, install_id: &str) -> Result<LibraryState, LibraryError> {
        self.move_to(install_id, LibraryState::Paused, None)
    }

    /// Stop an agent. It returns to [`LibraryState::Installed`], not to a
    /// terminal state: the installation outlives the run.
    pub fn terminate(&self, install_id: &str) -> Result<LibraryState, LibraryError> {
        self.move_to(install_id, LibraryState::Installed, None)
    }

    pub fn quarantine(&self, install_id: &str, reason: &str) -> Result<LibraryState, LibraryError> {
        self.move_to(
            install_id,
            LibraryState::Quarantined,
            Some(reason.to_string()),
        )
    }

    pub fn archive(&self, install_id: &str) -> Result<LibraryState, LibraryError> {
        self.move_to(install_id, LibraryState::Archived, None)
    }

    /// Bring a quarantined or archived agent back to installed.
    pub fn restore(&self, install_id: &str) -> Result<LibraryState, LibraryError> {
        self.move_to(install_id, LibraryState::Installed, None)
    }

    /// Flag that a newer release exists (set by [`crate::update::plan`]).
    pub fn mark_update_available(
        &self,
        install_id: &str,
        note: &str,
    ) -> Result<LibraryState, LibraryError> {
        self.move_to(install_id, LibraryState::Updates, Some(note.to_string()))
    }

    /// Discard every delta and checkpoint, returning the agent to the base
    /// artifact's initial state (ADR-288 §5 `reset`). The base is untouched.
    ///
    /// # Errors
    ///
    /// [`LibraryError::Io`] if the capsule directory cannot be cleared.
    pub fn reset(&self, install_id: &str) -> Result<usize, LibraryError> {
        let entry = self.get(install_id)?;
        let dir = entry.record.capsule()?.capsule_dir();
        let removed = clear_dir(&dir)?;
        self.witness(
            install_id,
            "reset",
            "completed",
            &format!("discarded {removed} state record(s); base artifact untouched"),
        );
        Ok(removed)
    }

    /// Copy the sealed state capsule out. The bytes stay encrypted: an export
    /// is a copy of the capsule, not a decryption of it.
    ///
    /// # Errors
    ///
    /// [`LibraryError::Io`] for any copy failure.
    pub fn export_state(
        &self,
        install_id: &str,
        dest: &Path,
    ) -> Result<StateTransfer, LibraryError> {
        let entry = self.get(install_id)?;
        let capsule = entry.record.capsule()?;
        let transfer = copy_dir(&capsule.capsule_dir(), dest, &entry.record.base_identity)?;
        self.witness(
            install_id,
            "state-export",
            "completed",
            &format!(
                "exported {} sealed record(s) ({} bytes) to {}",
                transfer.files, transfer.bytes, transfer.path
            ),
        );
        Ok(transfer)
    }

    /// Copy a sealed capsule in, refusing state from another lineage.
    ///
    /// Every sealed file is checked against the installed base identity before
    /// anything is written, so a lineage rejection leaves the existing capsule
    /// exactly as it was (ADR-288 §4).
    ///
    /// # Errors
    ///
    /// [`LibraryError::LineageRejected`] for foreign state, [`LibraryError::Io`]
    /// for a copy failure.
    pub fn import_state(
        &self,
        install_id: &str,
        source: &Path,
    ) -> Result<StateTransfer, LibraryError> {
        let entry = self.get(install_id)?;
        let capsule = entry.record.capsule()?;

        for path in sealed_files(source)? {
            let bytes = std::fs::read(&path).map_err(|e| io(&path, &e))?;
            let recorded = recorded_identity(&bytes)?;
            capsule.check_lineage(&recorded)?;
        }

        let transfer = copy_dir(source, &capsule.capsule_dir(), &entry.record.base_identity)?;
        self.witness(
            install_id,
            "state-import",
            "completed",
            &format!(
                "imported {} sealed record(s) from {}",
                transfer.files,
                source.display()
            ),
        );
        Ok(transfer)
    }

    /// Remove an agent under `policy`.
    ///
    /// # Errors
    ///
    /// [`LibraryError::StillActive`] while it may be executing,
    /// [`LibraryError::StateWouldBeLost`] if removal would destroy state that
    /// was not explicitly given up, and [`LibraryError::Io`] for a failed
    /// export — in which case nothing is removed.
    pub fn uninstall(
        &self,
        install_id: &str,
        policy: &UninstallPolicy,
    ) -> Result<UninstallOutcome, LibraryError> {
        let entry = self.get(install_id)?;
        if entry.state.is_active() {
            return Err(LibraryError::StillActive {
                install_id: install_id.to_string(),
                state: entry.state,
            });
        }

        let capsule_dir = entry.record.capsule()?.capsule_dir();
        let has_state = !sealed_files(&capsule_dir)?.is_empty();

        // Export first. A failure here aborts before anything is removed.
        let mut exported = None;
        if let Some(dest) = &policy.export_state_to {
            exported = Some(self.export_state(install_id, dest)?);
        }
        if policy.remove_state && has_state && exported.is_none() && !policy.explicit_discard() {
            return Err(LibraryError::StateWouldBeLost {
                install_id: install_id.to_string(),
            });
        }

        let install_dir = entry.record.install_dir();
        if install_dir.exists() {
            unlock_dir(&install_dir);
            std::fs::remove_dir_all(&install_dir).map_err(|e| io(&install_dir, &e))?;
        }
        if policy.remove_state {
            let _ = std::fs::remove_dir_all(&capsule_dir);
        }

        let outcome = UninstallOutcome {
            install_id: install_id.to_string(),
            runtime_removed: true,
            state_removed: policy.remove_state,
            exported_to: exported.as_ref().map(|t| t.path.clone()),
            exported_files: exported.as_ref().map(|t| t.files).unwrap_or(0),
            state_retained_at: (!policy.remove_state)
                .then(|| capsule_dir.display().to_string()),
        };
        self.witness(
            install_id,
            "uninstall",
            "completed",
            &format!(
                "runtime removed; state {}",
                match (&outcome.state_removed, &outcome.exported_to) {
                    (true, Some(to)) => format!("exported to {to} then deleted"),
                    (true, None) => "deleted at the user's request".to_string(),
                    (false, _) => format!("retained at {}", capsule_dir.display()),
                }
            ),
        );
        Ok(outcome)
    }

    /// Record a lifecycle event. Best effort: a receipt that could not be
    /// written must not roll back an operation that already happened.
    pub(crate) fn witness(&self, subject: &str, event: &str, outcome: &str, details: &str) {
        let _ = self
            .layout
            .lifecycle_log()
            .append(subject, event, outcome, details);
    }

    /// Check a move without performing it, returning the current state.
    fn require_transition(
        &self,
        install_id: &str,
        to: LibraryState,
    ) -> Result<LibraryState, LibraryError> {
        let from = self.get(install_id)?.state;
        if !from.may_move_to(to) {
            return Err(LibraryError::IllegalTransition {
                install_id: install_id.to_string(),
                from,
                to,
            });
        }
        Ok(from)
    }

    pub(crate) fn move_to(
        &self,
        install_id: &str,
        to: LibraryState,
        note: Option<String>,
    ) -> Result<LibraryState, LibraryError> {
        let from = self.require_transition(install_id, to)?;
        self.write_state(install_id, &LibraryEntryState::at(to, note))?;
        if from != to {
            self.witness(
                install_id,
                "library-state",
                to.label(),
                &format!("{} → {}", from.label(), to.label()),
            );
        }
        Ok(to)
    }

    /// Move regardless of the table. Only the verification failure path uses
    /// it: quarantine is asserted by the system, and refusing to quarantine
    /// because of a table gap would leave a failing artifact runnable.
    fn force_state(
        &self,
        install_id: &str,
        to: LibraryState,
        note: Option<String>,
    ) -> Result<(), LibraryError> {
        self.write_state(install_id, &LibraryEntryState::at(to, note))
    }

    fn state_path(&self, install_id: &str) -> PathBuf {
        self.layout.install_dir(install_id).join("library.json")
    }

    fn read_state(&self, install_id: &str) -> Result<LibraryEntryState, LibraryError> {
        let path = self.state_path(install_id);
        let text = std::fs::read_to_string(&path).map_err(|e| io(&path, &e))?;
        serde_json::from_str(&text).map_err(|e| LibraryError::Io {
            path: path.display().to_string(),
            message: format!("not a readable library state: {e}"),
        })
    }

    fn write_state(
        &self,
        install_id: &str,
        state: &LibraryEntryState,
    ) -> Result<(), LibraryError> {
        let path = self.state_path(install_id);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| io(parent, &e))?;
        }
        let json = serde_json::to_string_pretty(state).map_err(|e| LibraryError::Io {
            path: path.display().to_string(),
            message: e.to_string(),
        })?;
        std::fs::write(&path, json).map_err(|e| io(&path, &e))
    }

    fn entry_for(&self, record: InstallRecord) -> Result<LibraryEntry, LibraryError> {
        let stored = self.read_state(&record.install_id).ok();
        let capsule_dir = record.capsule().map(|c| c.capsule_dir()).ok();
        let state_bytes = capsule_dir
            .as_deref()
            .map(dir_bytes)
            .transpose()?
            .unwrap_or(0);
        let state = stored.as_ref().map(|s| s.state).unwrap_or(LibraryState::Installed);
        Ok(LibraryEntry {
            record,
            state,
            state_label: state.label(),
            note: stored.and_then(|s| s.note),
            state_bytes,
        })
    }
}

impl UninstallPolicy {
    /// `remove_state` with no export destination is the explicit discard: the
    /// caller named the destructive option and named no rescue.
    fn explicit_discard(&self) -> bool {
        self.remove_state && self.export_state_to.is_none()
    }
}

