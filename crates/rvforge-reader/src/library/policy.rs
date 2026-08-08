//! What an uninstall may remove, and what it reports having removed.
//!
//! ADR-294's rule is that revocation and removal preserve export and forensic
//! access, so the default here keeps state and deleting it has to be asked for
//! by name.

use std::path::PathBuf;

use serde::Serialize;

/// What an uninstall is allowed to remove.
///
/// The default removes the runtime and keeps the state, which is the ADR-294
/// rule: revocation and removal preserve export and forensic access unless the
/// user asked for the state to go too.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct UninstallPolicy {
    /// Delete the state capsule. Only ever true because someone asked for it.
    pub remove_state: bool,
    /// Copy the sealed state here first. An export that fails aborts the
    /// uninstall, so state is never lost to a half-completed removal.
    pub export_state_to: Option<PathBuf>,
}

impl UninstallPolicy {
    /// Remove the runtime, keep the state.
    pub fn keep_state() -> Self {
        Self::default()
    }

    /// Export the state, then remove both.
    pub fn export_then_remove(dest: impl Into<PathBuf>) -> Self {
        Self {
            remove_state: true,
            export_state_to: Some(dest.into()),
        }
    }
}

/// What an uninstall actually did.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UninstallOutcome {
    pub install_id: String,
    pub runtime_removed: bool,
    pub state_removed: bool,
    pub exported_to: Option<String>,
    pub exported_files: usize,
    /// Where state was left, when it was left.
    pub state_retained_at: Option<String>,
}

/// A copied state capsule.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StateTransfer {
    pub base_identity: String,
    pub files: usize,
    pub bytes: u64,
    pub path: String,
}

