//! Updates and the semantic permission difference (requirements P9, ADR-288 §7).
//!
//! An update is a **new base RVF**, never a mutation of the installed one, and
//! four rules govern applying it:
//!
//! - **Lineage is checked before anything else that costs the user a decision.**
//!   The release declares the base identity it supersedes; if that is not the
//!   identity currently installed under this id, the update is rejected
//!   ([`UpdateError::LineageMismatch`]). This is what stops an unrelated
//!   package being installed over an agent the user already trusts.
//! - **The difference is semantic, not textual.** [`diff_cards`] reports added
//!   classes, removed classes, and scope changes, and it is computed from the
//!   two verified capability cards rather than from anything the release says
//!   about itself.
//! - **Any expansion needs explicit approval.** [`UpdatePlan::requires_approval`]
//!   is true whenever a class is added or a scope widens, and [`apply`] refuses
//!   unless the approval names every expanding class. An update that only
//!   narrows or leaves the contract alone can be applied without one, which is
//!   the P9 "code fixes within the existing contract" case.
//! - **Rollback stays available until it was declared unsafe.** The
//!   [`StateContract::rollback_unsafe`] flag is part of the install request, so
//!   it is declared *before* the release is installed. [`rollback`] refuses on
//!   a release that declared it, rather than discovering the problem while
//!   rolling back.
//!
//! What is **not** implemented: ADR-288 §5 `migrate`. State is keyed by base
//! identity, so an updated agent starts from the new base's empty capsule and
//! the predecessor's capsule is retained untouched. The plan reports this as
//! [`StateDisposition::RetainedUnderPredecessor`]; carrying state across the
//! boundary is `rvm-state` scope and the reader does not fake it.

use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::install::{InstallError, StateContract};
use crate::library::LibraryError;

mod diff;
mod flow;

pub use diff::{
    diff_cards, CapabilityChange, ChangeKind, PermissionDiff, SchemaChange, StateDisposition,
    UpdatePlan,
};
pub use flow::{apply, plan, rollback};

/// One release offered as an update to an installed agent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseDescriptor {
    pub rvf_path: PathBuf,
    /// What the publisher calls it. Display only; nothing is decided from it.
    pub version_label: String,
    /// The base identity this release supersedes (ADR-288 §7). Checked against
    /// what is actually installed.
    pub predecessor_identity: String,
    /// Declared before this release is installed, per P9.
    pub state_contract: StateContract,
}

impl ReleaseDescriptor {
    pub fn new(
        rvf_path: impl Into<PathBuf>,
        version_label: impl Into<String>,
        predecessor_identity: impl Into<String>,
    ) -> Self {
        Self {
            rvf_path: rvf_path.into(),
            version_label: version_label.into(),
            predecessor_identity: predecessor_identity.into(),
            state_contract: StateContract::default(),
        }
    }
}


/// The user's answer to an [`UpdatePlan`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct UpdateApproval {
    /// Every expanding class, named. An approval that omits one is not an
    /// approval of it.
    pub approved_classes: Vec<String>,
}

impl UpdateApproval {
    /// Approve exactly what a plan asks for.
    pub fn granting(plan: &UpdatePlan) -> Self {
        Self {
            approved_classes: plan.approval_classes.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case", tag = "kind")]
pub enum UpdateError {
    /// The release does not supersede what is installed.
    LineageMismatch {
        install_id: String,
        installed: String,
        declared: String,
    },
    /// The new release did not verify. Nothing was touched.
    NotVerified { reason: String },
    /// The release is the same base identity that is already installed.
    NotAnUpdate { identity: String },
    /// An expansion was not approved by name.
    ApprovalRequired { classes: Vec<String> },
    /// The installed release declared, before it was installed, that its state
    /// migration makes rollback unsafe.
    RollbackUnsafe { install_id: String },
    /// There is no retained predecessor to roll back to.
    NoPredecessor { install_id: String },
    Library { detail: String },
    Install { detail: String },
    Io { path: String, message: String },
}

impl std::fmt::Display for UpdateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UpdateError::LineageMismatch {
                install_id,
                installed,
                declared,
            } => write!(
                f,
                "this release supersedes {declared}, but '{install_id}' has {installed} installed"
            ),
            UpdateError::NotVerified { reason } => {
                write!(f, "refusing to update from an unverified package: {reason}")
            }
            UpdateError::NotAnUpdate { identity } => {
                write!(f, "{identity} is already the installed release")
            }
            UpdateError::ApprovalRequired { classes } => write!(
                f,
                "this update expands {} and needs explicit approval",
                classes.join(", ")
            ),
            UpdateError::RollbackUnsafe { install_id } => write!(
                f,
                "'{install_id}' declared before installation that its state migration makes \
                 rollback unsafe"
            ),
            UpdateError::NoPredecessor { install_id } => {
                write!(f, "'{install_id}' has no retained predecessor to roll back to")
            }
            UpdateError::Library { detail } | UpdateError::Install { detail } => {
                write!(f, "{detail}")
            }
            UpdateError::Io { path, message } => write!(f, "{path}: {message}"),
        }
    }
}

impl std::error::Error for UpdateError {}

impl From<LibraryError> for UpdateError {
    fn from(e: LibraryError) -> Self {
        UpdateError::Library {
            detail: e.to_string(),
        }
    }
}

impl From<InstallError> for UpdateError {
    fn from(e: InstallError) -> Self {
        UpdateError::Install {
            detail: e.to_string(),
        }
    }
}

fn io(path: &Path, e: &std::io::Error) -> UpdateError {
    UpdateError::Io {
        path: path.display().to_string(),
        message: e.to_string(),
    }
}

