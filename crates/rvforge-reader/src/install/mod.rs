//! Installation (requirements P6, ADR-288 §1–§2, ADR-294 §Consequences).
//!
//! Installing is the moment a package stops being a file the user is looking at
//! and becomes something the machine is prepared to run, so three refusals are
//! structural here rather than advisory:
//!
//! 1. **Verification precedes installation.** [`install`] verifies the bytes it
//!    is about to copy and refuses anything short of
//!    [`VerificationStatus::Verified`]. The refusal is witnessed twice — once by
//!    the verification receipt, once by a lifecycle receipt — because a package
//!    that was refused at install time is exactly the event an audit is looking
//!    for.
//! 2. **A grant the container did not declare is not installable.** The user
//!    accepts a set of capability classes, and every accepted class must appear
//!    in the card derived from the verified container. Accepting more than was
//!    declared is [`InstallError::GrantNotDeclared`], not a silent narrowing:
//!    if the two disagree, the user was shown one contract and asked to consent
//!    to another.
//! 3. **The base artifact is immutable and state lives elsewhere.** The
//!    verified bytes are copied to `<install_dir>/base.rvf` and made read-only;
//!    the state capsule directory is under a state root that
//!    [`StateCapsule::new`] rejects if it is inside the install root, so
//!    deleting state can never touch the signed artifact (ADR-288 §2).
//!
//! Nothing here executes package content. The installed `base.rvf` is a copy of
//! bytes, and no code path in this crate maps, links, or interprets it.
//!
//! ```text
//! <install_root>/<install-id>/
//!     install.json     immutable install record — what was consented to
//!     library.json     mutable library state (owned by [`crate::library`])
//!     base.rvf         the verified container, read-only
//! <state_root>/
//!     receipts.jsonl   verification receipts
//!     lifecycle.jsonl  chained lifecycle witness receipts
//!     <base-identity>/ state capsule (ADR-288)
//! ```

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::inspect::SignatureStatus;
use crate::state::{StateCapsule, StateError};

mod flow;
mod layout;

pub use flow::{derive_install_id, install, install_for_host, offer, offer_for_host, InstallOffer};
pub use layout::InstallLayout;

/// Schema identifier carried by every [`InstallRecord`].
pub const INSTALL_RECORD_SCHEMA: &str = "rvforge.reader.install/1";

/// What an installation declares about its state before it happens.
///
/// P9 requires that "rollback is unsafe" be declared *before* installation, so
/// it is part of the request rather than something discovered at rollback time.
/// A release that reserves the right to make rollback unsafe later would make
/// the guarantee meaningless.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StateContract {
    /// The `stateSchemaVersion` this release reads and writes (ADR-288 §3).
    pub state_schema_version: u32,
    /// This release's state migration makes returning to the predecessor
    /// unsafe. Declared here, enforced by [`crate::update::rollback`].
    pub rollback_unsafe: bool,
}

impl Default for StateContract {
    fn default() -> Self {
        Self {
            state_schema_version: 1,
            rollback_unsafe: false,
        }
    }
}

/// One installation request: what to install, where, and what the user accepted.
#[derive(Debug, Clone)]
pub struct InstallRequest {
    pub rvf_path: PathBuf,
    /// A display name for the library and the dock. System-owned: it comes from
    /// the installer, never from the running agent.
    pub agent_name: String,
    /// The capability classes the user accepted from the card. Every one must
    /// have been declared by the verified container.
    pub accepted_classes: Vec<String>,
    pub state_contract: StateContract,
    /// Set by [`crate::update`] to keep an agent's identity stable across a new
    /// base RVF. `None` derives a fresh id from the base identity.
    pub install_id: Option<String>,
    /// The base identity this release supersedes, forming the ADR-288 §7
    /// lineage chain. `None` for a first install.
    pub predecessor_identity: Option<String>,
}

impl InstallRequest {
    /// A first install accepting exactly the classes the card granted.
    pub fn new(
        rvf_path: impl Into<PathBuf>,
        agent_name: impl Into<String>,
        accepted_classes: Vec<String>,
    ) -> Self {
        Self {
            rvf_path: rvf_path.into(),
            agent_name: agent_name.into(),
            accepted_classes,
            state_contract: StateContract::default(),
            install_id: None,
            predecessor_identity: None,
        }
    }
}

/// The immutable record of one installation. Written once and never edited;
/// mutable library state lives in `library.json` beside it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InstallRecord {
    /// Always [`INSTALL_RECORD_SCHEMA`].
    pub schema: String,
    pub install_id: String,
    pub agent_name: String,
    /// SHA-256 of the installed container, as `sha256:<hex>`.
    pub base_identity: String,
    /// Where the package was installed from. Provenance, not a live reference.
    pub source_path: String,
    pub installed_at_unix: u64,
    pub install_dir: String,
    pub state_root: String,
    /// The classes the user accepted, which are a subset of the declared ones.
    pub granted_classes: Vec<String>,
    /// The exact sentences the user was shown when they accepted.
    pub granted_text: Vec<String>,
    pub state_contract: StateContract,
    pub predecessor_identity: Option<String>,
    /// The runtime the FR004 ladder chose on this host at install time.
    pub runtime_profile: Option<String>,
    pub isolation_claim: Option<String>,
    /// What was known about the publisher's signature when it was installed.
    pub signature: SignatureStatus,
    /// The lifecycle receipt this installation was recorded as.
    pub receipt_id: Option<String>,
}

impl InstallRecord {
    pub fn install_dir(&self) -> PathBuf {
        PathBuf::from(&self.install_dir)
    }

    /// The immutable installed artifact.
    pub fn base_artifact(&self) -> PathBuf {
        self.install_dir().join("base.rvf")
    }

    /// The capsule this install's state belongs to.
    ///
    /// # Errors
    ///
    /// [`StateError`] if the recorded roots no longer satisfy the ADR-288 §2
    /// separation, or the base identity is not a content address.
    pub fn capsule(&self) -> Result<StateCapsule, StateError> {
        StateCapsule::new(&self.install_dir, &self.state_root, &self.base_identity)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case", tag = "kind")]
pub enum InstallError {
    /// The package did not verify. Nothing was written.
    NotVerified { reason: String },
    /// Verification passed but no capability card could be derived, so there is
    /// no contract to consent to.
    NoContract { reason: String },
    /// An accepted class was not declared by the verified container.
    GrantNotDeclared { class: String },
    /// No runtime on this host can run the package (ADR-289 §2 is terminal).
    UnsupportedRuntime { order: Vec<String> },
    /// This install id already exists. Updating goes through
    /// [`crate::update`], which preserves lineage.
    AlreadyInstalled { install_id: String },
    Layout { detail: String },
    Io { path: String, message: String },
}

impl std::fmt::Display for InstallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InstallError::NotVerified { reason } => {
                write!(f, "refusing to install an unverified package: {reason}")
            }
            InstallError::NoContract { reason } => {
                write!(f, "no capability contract to accept: {reason}")
            }
            InstallError::GrantNotDeclared { class } => write!(
                f,
                "'{class}' was accepted but the verified container does not declare it"
            ),
            InstallError::UnsupportedRuntime { order } => write!(
                f,
                "no compatible runtime on this host; the selection order is {}",
                order.join(" → ")
            ),
            InstallError::AlreadyInstalled { install_id } => {
                write!(f, "'{install_id}' is already installed")
            }
            InstallError::Layout { detail } => write!(f, "{detail}"),
            InstallError::Io { path, message } => write!(f, "{path}: {message}"),
        }
    }
}

impl std::error::Error for InstallError {}

impl From<StateError> for InstallError {
    fn from(e: StateError) -> Self {
        InstallError::Layout {
            detail: e.to_string(),
        }
    }
}

fn io(path: &Path, e: &std::io::Error) -> InstallError {
    InstallError::Io {
        path: path.display().to_string(),
        message: e.to_string(),
    }
}

