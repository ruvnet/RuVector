//! The installed-agent library (requirements P7, ADR-294 §Consequences).
//!
//! The library is the list of things this machine has agreed to run, and the
//! operations that change that agreement. Three rules shape it:
//!
//! - **Verification precedes every load.** [`Library::open`] re-verifies the
//!   installed `base.rvf` before it will move an agent to
//!   [`LibraryState::Running`]. A package that stops verifying — because the
//!   file was altered after installation — is moved to
//!   [`LibraryState::Quarantined`] rather than opened, and the refusal is
//!   witnessed.
//! - **State transitions are checked, not assumed.** [`LibraryState::may_move_to`]
//!   is the whole table. An archived agent cannot start running and a
//!   quarantined one cannot start running; both must be brought back to
//!   [`LibraryState::Installed`] first, which is a deliberate act.
//! - **Removal never silently destroys user state.** ADR-294 requires that
//!   revocation and removal preserve export and forensic access.
//!   [`UninstallPolicy`] therefore defaults to keeping state, and the only way
//!   to delete it is to ask for that specifically — optionally through
//!   [`UninstallPolicy::export_state_to`], which exports first and aborts the
//!   whole uninstall if the export fails.
//!
//! Every operation appends a chained lifecycle receipt, so the library's
//! history is auditable through the same witness viewer as everything else.
//!
//! `Organization Managed`, the seventh P7 state, is not implemented: it is a
//! property of an organizational policy source the reader has no channel to.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::install::{InstallError, InstallRecord};
use crate::state::StateError;

mod fsops;
mod policy;
mod registry;

pub use policy::{StateTransfer, UninstallOutcome, UninstallPolicy};
pub use registry::Library;

/// Schema identifier carried by every [`LibraryEntryState`].
pub const LIBRARY_STATE_SCHEMA: &str = "rvforge.reader.library/1";

/// The P7 library states the reader implements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LibraryState {
    Installed,
    Running,
    Paused,
    /// P7's "Updates": a newer release is available for this agent.
    Updates,
    /// Execution is blocked by policy. State is preserved (ADR-294).
    Quarantined,
    /// Kept, not runnable, not deleted.
    Archived,
}

impl LibraryState {
    pub fn label(self) -> &'static str {
        match self {
            LibraryState::Installed => "Installed",
            LibraryState::Running => "Running",
            LibraryState::Paused => "Paused",
            LibraryState::Updates => "Update available",
            LibraryState::Quarantined => "Quarantined",
            LibraryState::Archived => "Archived",
        }
    }

    /// Whether an agent in this state may be executing.
    pub fn is_active(self) -> bool {
        matches!(self, LibraryState::Running | LibraryState::Paused)
    }

    /// The transition table.
    ///
    /// `Quarantined` and `Archived` release only to `Installed`: coming back
    /// from either is a decision, and a decision that lands directly in
    /// `Running` would hide it.
    pub fn may_move_to(self, to: LibraryState) -> bool {
        use LibraryState::*;
        if self == to {
            return true;
        }
        match self {
            Installed => matches!(to, Running | Updates | Quarantined | Archived),
            Running => matches!(to, Paused | Installed | Updates | Quarantined),
            Paused => matches!(to, Running | Installed | Updates | Quarantined),
            Updates => matches!(to, Running | Installed | Quarantined | Archived),
            Quarantined => matches!(to, Installed | Archived),
            Archived => matches!(to, Installed),
        }
    }
}

/// The mutable half of an installation, written beside the immutable record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LibraryEntryState {
    pub schema: String,
    pub state: LibraryState,
    pub changed_at_unix: u64,
    /// Why the entry is where it is, when that needs saying.
    pub note: Option<String>,
}

impl LibraryEntryState {
    fn at(state: LibraryState, note: Option<String>) -> Self {
        Self {
            schema: LIBRARY_STATE_SCHEMA.to_string(),
            state,
            changed_at_unix: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            note,
        }
    }
}

/// One row of the library.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LibraryEntry {
    pub record: InstallRecord,
    pub state: LibraryState,
    pub state_label: &'static str,
    pub note: Option<String>,
    /// Bytes of sealed state currently held for this agent.
    pub state_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case", tag = "kind")]
pub enum LibraryError {
    NotInstalled {
        install_id: String,
    },
    /// The move is not in the transition table.
    IllegalTransition {
        install_id: String,
        from: LibraryState,
        to: LibraryState,
    },
    /// The installed artifact no longer verifies. The entry was quarantined.
    VerificationFailed {
        install_id: String,
        detail: String,
    },
    /// State recorded against a different base identity was offered for import.
    LineageRejected {
        detail: String,
    },
    /// Uninstalling would have destroyed state that was not explicitly given up.
    StateWouldBeLost {
        install_id: String,
    },
    /// An agent that may be executing cannot be removed under it.
    StillActive {
        install_id: String,
        state: LibraryState,
    },
    Install {
        detail: String,
    },
    Io {
        path: String,
        message: String,
    },
}

impl std::fmt::Display for LibraryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LibraryError::NotInstalled { install_id } => {
                write!(f, "no installed agent '{install_id}'")
            }
            LibraryError::IllegalTransition {
                install_id,
                from,
                to,
            } => write!(
                f,
                "'{install_id}' cannot move from {} to {}",
                from.label(),
                to.label()
            ),
            LibraryError::VerificationFailed { install_id, detail } => write!(
                f,
                "'{install_id}' no longer verifies and was quarantined: {detail}"
            ),
            LibraryError::LineageRejected { detail } => write!(f, "{detail}"),
            LibraryError::StateWouldBeLost { install_id } => write!(
                f,
                "uninstalling '{install_id}' would delete its state; export it or ask for \
                 removal explicitly"
            ),
            LibraryError::StillActive { install_id, state } => write!(
                f,
                "'{install_id}' is {}; terminate it before uninstalling",
                state.label()
            ),
            LibraryError::Install { detail } => write!(f, "{detail}"),
            LibraryError::Io { path, message } => write!(f, "{path}: {message}"),
        }
    }
}

impl std::error::Error for LibraryError {}

impl From<InstallError> for LibraryError {
    fn from(e: InstallError) -> Self {
        LibraryError::Install {
            detail: e.to_string(),
        }
    }
}

impl From<StateError> for LibraryError {
    fn from(e: StateError) -> Self {
        LibraryError::LineageRejected {
            detail: e.to_string(),
        }
    }
}

fn io(path: &Path, e: &std::io::Error) -> LibraryError {
    LibraryError::Io {
        path: path.display().to_string(),
        message: e.to_string(),
    }
}

