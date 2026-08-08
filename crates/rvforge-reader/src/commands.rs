//! Tauri command surface.
//!
//! Thin wrappers only: every decision lives in [`crate::inspect`],
//! [`crate::capability`], and [`crate::runtime`] so it can be tested without a
//! webview. None of these commands executes package content.

use std::path::PathBuf;
use std::sync::Mutex;

use crate::capability::CapabilityCard;
use crate::dock::{
    AgentDockEntry, IsolationClass, NetworkIndicator, PermissionSummary, ResourceUsage,
    SystemOwnedStatus, TrustBadge,
};
use crate::dock_events::Notification;
use crate::dock_roster::{AgentPill, DockRoster, ExpandedAgentView, RosterView};
use crate::dock_state::DockState;
use crate::dock_bridge;
use crate::inspect::{self, InspectionSummary, VerificationOutcome, VerificationStatus};
use crate::install::{InstallOffer, InstallRecord, InstallRequest, StateContract};
use crate::library::{Library, LibraryEntry, LibraryState, StateTransfer, UninstallOutcome, UninstallPolicy};
use crate::runtime::{self, HostProfile, RuntimeChoice};
use crate::update::{self, ReleaseDescriptor, UpdateApproval, UpdatePlan};
use crate::witness_view::{self, WitnessChainView};

/// Read identity, segments, and declared capabilities. Checks nothing, executes
/// nothing: the returned verification status is always `unverified`.
#[tauri::command]
pub fn inspect_rvf(path: String) -> InspectionSummary {
    inspect::inspect(&PathBuf::from(path))
}

/// Run the root-manifest, per-segment-hash, and unsigned-executable checks, and
/// append the witness receipt. Still executes nothing.
#[tauri::command]
pub fn verify_rvf(path: String) -> VerificationOutcome {
    inspect::verify(&PathBuf::from(path))
}

/// The P6 install-time capability contract for a package. Verifies first, and
/// returns the everything-denied card if verification does not pass.
#[tauri::command]
pub fn capability_card(path: String) -> CapabilityCard {
    inspect::capability_card(&PathBuf::from(path))
}

/// The FR004 runtime selection for this host.
///
/// The choice is a property of the host and the vendored compatibility matrix;
/// `path` only labels which package the answer was reported for.
#[tauri::command]
pub fn runtime_selection(path: String) -> Result<RuntimeChoice, String> {
    let mut choice = runtime::select_for_host(&HostProfile::detect()).map_err(|e| e.to_string())?;
    choice.subject = Some(path);
    Ok(choice)
}

/// The witness chain, verified.
///
/// With no `path`, the reader's own receipt log; with one, an external
/// `receipts.jsonl` such as the CLI writes. Each receipt's content address is
/// recomputed and each `prevReceipt` link is checked, so the returned status is
/// derived here rather than reported by whoever wrote the file.
#[tauri::command]
pub fn witness_chain(path: Option<String>) -> Result<WitnessChainView, String> {
    match path {
        Some(p) => witness_view::chain_at(&PathBuf::from(p)).map_err(|e| e.to_string()),
        None => Ok(witness_view::chain_from_default_log()),
    }
}

// ---------------------------------------------------------------------------
// Install, library, update (requirements P6, P7, P9).
//
// Each of these is a thin wrapper too. The refusals — unverified package,
// undeclared grant, illegal library transition, lineage mismatch, unapproved
// expansion, unsafe rollback — are decided in `install`, `library`, and
// `update`, so the webview cannot reach a decision by calling a different
// command.
// ---------------------------------------------------------------------------

/// Verify a package and derive the contract the user will be asked to accept.
/// Writes a verification receipt; installs nothing.
#[tauri::command]
pub fn install_offer(library: tauri::State<'_, Library>, path: String) -> InstallOffer {
    library.offer(&PathBuf::from(path))
}

/// Install a verified package with the classes the user accepted.
///
/// `accepted_classes` must be a subset of what
/// [`install_offer`] reported as grantable; anything else is refused.
#[tauri::command]
pub fn install_agent(
    library: tauri::State<'_, Library>,
    path: String,
    name: String,
    accepted_classes: Vec<String>,
    state_schema_version: Option<u32>,
    rollback_unsafe: Option<bool>,
) -> Result<LibraryEntry, String> {
    let mut request = InstallRequest::new(PathBuf::from(path), name, accepted_classes);
    request.state_contract = StateContract {
        state_schema_version: state_schema_version.unwrap_or(1),
        rollback_unsafe: rollback_unsafe.unwrap_or(false),
    };
    library.install(&request).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn library_list(library: tauri::State<'_, Library>) -> Result<Vec<LibraryEntry>, String> {
    library.list().map_err(|e| e.to_string())
}

/// Open an agent. Re-verifies the installed artifact first; a package that no
/// longer verifies is quarantined instead of opened.
#[tauri::command]
pub fn library_open(
    library: tauri::State<'_, Library>,
    install_id: String,
) -> Result<LibraryState, String> {
    library.open(&install_id).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn library_pause(
    library: tauri::State<'_, Library>,
    install_id: String,
) -> Result<LibraryState, String> {
    library.pause(&install_id).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn library_terminate(
    library: tauri::State<'_, Library>,
    install_id: String,
) -> Result<LibraryState, String> {
    library.terminate(&install_id).map_err(|e| e.to_string())
}

/// Discard every delta and checkpoint. The base artifact is untouched.
#[tauri::command]
pub fn library_reset(
    library: tauri::State<'_, Library>,
    install_id: String,
) -> Result<usize, String> {
    library.reset(&install_id).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn library_verify(
    library: tauri::State<'_, Library>,
    install_id: String,
) -> Result<VerificationStatus, String> {
    library.verify(&install_id).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn library_archive(
    library: tauri::State<'_, Library>,
    install_id: String,
) -> Result<LibraryState, String> {
    library.archive(&install_id).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn library_restore(
    library: tauri::State<'_, Library>,
    install_id: String,
) -> Result<LibraryState, String> {
    library.restore(&install_id).map_err(|e| e.to_string())
}

/// Copy the sealed state capsule out. The bytes stay encrypted.
#[tauri::command]
pub fn library_export_state(
    library: tauri::State<'_, Library>,
    install_id: String,
    dest: String,
) -> Result<StateTransfer, String> {
    library
        .export_state(&install_id, &PathBuf::from(dest))
        .map_err(|e| e.to_string())
}

/// Copy a sealed capsule in, refusing state recorded against another lineage.
#[tauri::command]
pub fn library_import_state(
    library: tauri::State<'_, Library>,
    install_id: String,
    source: String,
) -> Result<StateTransfer, String> {
    library
        .import_state(&install_id, &PathBuf::from(source))
        .map_err(|e| e.to_string())
}

/// Remove an agent. `remove_state` defaults to `false`, and asking for it
/// without an `export_to` is the explicit discard.
#[tauri::command]
pub fn library_uninstall(
    library: tauri::State<'_, Library>,
    install_id: String,
    remove_state: Option<bool>,
    export_to: Option<String>,
) -> Result<UninstallOutcome, String> {
    let policy = UninstallPolicy {
        remove_state: remove_state.unwrap_or(false),
        export_state_to: export_to.map(PathBuf::from),
    };
    library
        .uninstall(&install_id, &policy)
        .map_err(|e| e.to_string())
}

/// The P9 permission difference for a candidate release. Decides nothing and
/// installs nothing.
#[tauri::command]
pub fn update_plan(
    library: tauri::State<'_, Library>,
    install_id: String,
    path: String,
    version_label: String,
    predecessor_identity: String,
    state_schema_version: Option<u32>,
    rollback_unsafe: Option<bool>,
) -> Result<UpdatePlan, String> {
    let release = release_of(
        path,
        version_label,
        predecessor_identity,
        state_schema_version,
        rollback_unsafe,
    );
    update::plan(&library, &install_id, &release).map_err(|e| e.to_string())
}

/// Apply an update. `approved_classes` must name every expanding class the plan
/// listed, or the update is refused.
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub fn update_apply(
    library: tauri::State<'_, Library>,
    install_id: String,
    path: String,
    version_label: String,
    predecessor_identity: String,
    approved_classes: Vec<String>,
    state_schema_version: Option<u32>,
    rollback_unsafe: Option<bool>,
) -> Result<InstallRecord, String> {
    let release = release_of(
        path,
        version_label,
        predecessor_identity,
        state_schema_version,
        rollback_unsafe,
    );
    let plan = update::plan(&library, &install_id, &release).map_err(|e| e.to_string())?;
    let approval = UpdateApproval { approved_classes };
    update::apply(&library, &release, &plan, &approval).map_err(|e| e.to_string())
}

/// Return to the retained predecessor, unless this release declared rollback
/// unsafe before it was installed.
#[tauri::command]
pub fn update_rollback(
    library: tauri::State<'_, Library>,
    install_id: String,
) -> Result<InstallRecord, String> {
    update::rollback(&library, &install_id).map_err(|e| e.to_string())
}

fn release_of(
    path: String,
    version_label: String,
    predecessor_identity: String,
    state_schema_version: Option<u32>,
    rollback_unsafe: Option<bool>,
) -> ReleaseDescriptor {
    let mut release =
        ReleaseDescriptor::new(PathBuf::from(path), version_label, predecessor_identity);
    release.state_contract = StateContract {
        state_schema_version: state_schema_version.unwrap_or(1),
        rollback_unsafe: rollback_unsafe.unwrap_or(false),
    };
    release
}

/// The dock's roster, managed by the Tauri app.
///
/// Chrome state lives on the Rust side of the process boundary, which is what
/// makes ADR-295 §7 structural here: the webview can read it and can invoke
/// user controls, but the agent-facing command is
/// [`dock_agent_report`] and it carries only task text and progress.
#[derive(Default)]
pub struct DockSurface(pub Mutex<DockRoster>);

impl DockSurface {
    fn with<R>(&self, f: impl FnOnce(&mut DockRoster) -> R) -> Result<R, String> {
        let mut roster = self
            .0
            .lock()
            .map_err(|_| "dock roster lock is poisoned".to_string())?;
        Ok(f(&mut roster))
    }
}

/// The collapsed bar: active agent, two secondary slots, overflow count,
/// aggregate usage, approval count.
#[tauri::command]
pub fn dock_view(dock: tauri::State<'_, DockSurface>) -> Result<RosterView, String> {
    dock.with(|r| r.view())
}

/// Rebuild the roster from the library, so the dock shows real installations
/// rather than whatever was registered by hand.
///
/// The roster is replaced wholesale: an agent that was uninstalled leaves the
/// dock, and one whose library state moved is redrawn from the record rather
/// than from what the dock last believed. Agent-reported task text does not
/// survive the rebuild, which is the correct direction to lose data in — the
/// dock re-derives system chrome and waits for the agent to report again.
#[tauri::command]
pub fn dock_sync(
    dock: tauri::State<'_, DockSurface>,
    library: tauri::State<'_, Library>,
) -> Result<RosterView, String> {
    let rebuilt = dock_bridge::roster(&library).map_err(|e| e.to_string())?;
    dock.with(|r| {
        *r = rebuilt;
        r.view()
    })
}

/// Interaction one of the acceptance test.
#[tauri::command]
pub fn dock_expand(
    dock: tauri::State<'_, DockSurface>,
    agent_id: String,
) -> Result<ExpandedAgentView, String> {
    dock.with(|r| r.expand(&agent_id))?
        .map_err(|e| e.to_string())
}

/// One action, from any non-terminal state.
#[tauri::command]
pub fn dock_pause(
    dock: tauri::State<'_, DockSurface>,
    agent_id: String,
) -> Result<DockState, String> {
    dock.with(|r| r.pause(&agent_id))?
        .map_err(|e| e.to_string())
}

/// Interaction two of the acceptance test. No confirmation chain.
#[tauri::command]
pub fn dock_terminate(
    dock: tauri::State<'_, DockSurface>,
    agent_id: String,
) -> Result<DockState, String> {
    dock.with(|r| r.terminate(&agent_id))?
        .map_err(|e| e.to_string())
}

/// Behind the overflow count.
#[tauri::command]
pub fn dock_swarm_view(dock: tauri::State<'_, DockSurface>) -> Result<Vec<AgentPill>, String> {
    dock.with(|r| r.swarm_view())
}

/// Threshold events only (ADR-295 §9).
#[tauri::command]
pub fn dock_notifications(
    dock: tauri::State<'_, DockSurface>,
) -> Result<Vec<Notification>, String> {
    dock.with(|r| r.event_log().notifications().to_vec())
}

/// **The agent channel.** The whole agent-supplied surface is these two
/// arguments, and both are sanitized and clamped before they are stored. There
/// is no command through which an agent sets its own state, trust badge,
/// network indicator, permissions, witness status, cost, or resource usage.
#[tauri::command]
pub fn dock_agent_report(
    dock: tauri::State<'_, DockSurface>,
    agent_id: String,
    task: String,
    progress: i64,
) -> Result<(), String> {
    dock.with(|r| r.apply_agent_update(&agent_id, &task, progress))?
        .map_err(|e| e.to_string())
}

/// Register a placeholder agent so the dock can be exercised before `rvm-ffi`
/// is wired in.
///
/// Every security-bearing field reports its unknown value — unverified
/// publisher, no confinement engaged — because the scaffold has measured none
/// of them. It does not fabricate a trusted-looking agent.
///
/// The witness status is the exception, and it is not a placeholder: it is the
/// verdict [`crate::witness_view`] reached over the reader's own receipt log.
/// An empty log still reports `Unavailable`, but a log the viewer found broken
/// reports `Broken` here too.
#[tauri::command]
pub fn dock_add_scaffold_agent(
    dock: tauri::State<'_, DockSurface>,
    agent_id: String,
    name: String,
    icon: String,
) -> Result<RosterView, String> {
    let system = SystemOwnedStatus::measured(
        DockState::Idle,
        TrustBadge::Unverified,
        NetworkIndicator::Disabled,
        PermissionSummary {
            isolation: IsolationClass::None,
            local_model: false,
            network_allowed: false,
            selected_folder_access: false,
            encrypted_memory: false,
            grants: Vec::new(),
            pending_approvals: Vec::new(),
        },
        witness_view::chain_from_default_log().dock_status,
        ResourceUsage::ZERO,
        0,
        None,
    );
    dock.with(|r| {
        r.insert(AgentDockEntry::new(agent_id, name, icon, system));
        r.view()
    })
}
