//! The dock's roster, built from the library rather than from sample data.
//!
//! This is the seam where ADR-295 §7 has to hold against real data: everything
//! the dock renders about an installed agent comes from the install record, the
//! library state, and the verified witness chain — three sources the agent does
//! not write to. The agent's own channel remains
//! [`DockRoster::apply_agent_update`], and nothing in this module passes an
//! agent-supplied value into [`SystemOwnedStatus::measured`].
//!
//! Two things are reported as unknown rather than invented, because nothing in
//! the reader measures them yet: [`ResourceUsage`] is zero and the model is
//! `None`. A dock that displayed plausible-looking CPU figures for an agent
//! nobody is metering would be worse than one that displays zero.

use crate::dock::{
    AgentDockEntry, IsolationClass, NetworkIndicator, PermissionSummary, ResourceUsage,
    SystemOwnedStatus, TrustBadge, WitnessStatus,
};
use crate::dock_roster::DockRoster;
use crate::dock_state::DockState;
use crate::inspect::SignatureStatus;
use crate::install::InstallRecord;
use crate::library::{Library, LibraryEntry, LibraryError, LibraryState};
use crate::witness_view;

/// The icon the dock draws for an installed agent. System chrome: it is not
/// read from the package, so a publisher cannot pick a glyph that imitates a
/// dock control.
pub const INSTALLED_AGENT_ICON: &str = "◆";

/// Build a roster reflecting every installed agent.
///
/// # Errors
///
/// Any [`LibraryError`] from listing the library.
pub fn roster(library: &Library) -> Result<DockRoster, LibraryError> {
    let witness = lifecycle_witness(library);
    let mut roster = DockRoster::new();
    for entry in library.list()? {
        roster.insert(dock_entry(&entry, witness));
    }
    Ok(roster)
}

/// The witness status the dock shows for this library: the verdict over the
/// lifecycle chain, recomputed by the viewer rather than reported by the file.
pub fn lifecycle_witness(library: &Library) -> WitnessStatus {
    witness_view::chain_at(library.layout().lifecycle_log().path())
        .map(|view| view.dock_status)
        .unwrap_or(WitnessStatus::Unavailable)
}

/// One library row as a dock entry.
pub fn dock_entry(entry: &LibraryEntry, witness: WitnessStatus) -> AgentDockEntry {
    let system = SystemOwnedStatus::measured(
        dock_state(entry.state),
        trust_badge(entry.record.signature),
        network_indicator(&entry.record),
        permissions(&entry.record),
        witness,
        // Nothing meters an installed agent yet, so nothing is claimed.
        ResourceUsage::ZERO,
        0,
        None,
    );
    AgentDockEntry::new(
        entry.record.install_id.clone(),
        entry.record.agent_name.clone(),
        INSTALLED_AGENT_ICON,
        system,
    )
}

/// Library state to dock state.
///
/// `Updates` maps to `Idle`: an update being available says nothing about
/// whether the agent is running, and the dock's job is to show what it is
/// doing. `Archived` maps to `Completed`, the dock's terminal state.
pub fn dock_state(state: LibraryState) -> DockState {
    match state {
        LibraryState::Installed | LibraryState::Updates => DockState::Idle,
        LibraryState::Running => DockState::Running,
        LibraryState::Paused => DockState::Paused,
        LibraryState::Quarantined => DockState::Quarantined,
        LibraryState::Archived => DockState::Completed,
    }
}

/// The trust badge for what was actually checked at install time. A signature
/// nobody could check is `Unverified`, never neutral.
pub fn trust_badge(signature: SignatureStatus) -> TrustBadge {
    match signature {
        SignatureStatus::Verified => TrustBadge::VerifiedPublisher,
        SignatureStatus::NotChecked => TrustBadge::SignedUnknownPublisher,
        SignatureStatus::Unverified | SignatureStatus::Unsigned | SignatureStatus::Invalid => {
            TrustBadge::Unverified
        }
    }
}

/// Network is `AllowedIdle` at most: the reader observes no connections, so it
/// never reports `Active`.
fn network_indicator(record: &InstallRecord) -> NetworkIndicator {
    if granted(record, "network") {
        NetworkIndicator::AllowedIdle
    } else {
        NetworkIndicator::Disabled
    }
}

fn permissions(record: &InstallRecord) -> PermissionSummary {
    PermissionSummary {
        isolation: isolation_class(record),
        local_model: granted(record, "model"),
        network_allowed: granted(record, "network"),
        selected_folder_access: granted(record, "filesystem"),
        encrypted_memory: granted(record, "persistent-state"),
        grants: record.granted_text.clone(),
        // Approval prompts come from a running policy engine, which is not
        // wired here. An empty list is the honest answer, not a claim that
        // nothing was ever asked.
        pending_approvals: Vec::new(),
    }
}

/// The isolation the selected runtime profile claims. An unrecognised or
/// missing profile is `None` — reported, never assumed.
fn isolation_class(record: &InstallRecord) -> IsolationClass {
    match record.runtime_profile.as_deref() {
        Some("rvm-native") => IsolationClass::Rvm,
        Some("wasm") | Some("os-isolation+wasm") | Some("linux-microvm") => IsolationClass::Wasm,
        _ => IsolationClass::None,
    }
}

fn granted(record: &InstallRecord, class: &str) -> bool {
    record.granted_classes.iter().any(|c| c == class)
}
