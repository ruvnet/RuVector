//! RVForge Reader — desktop reader for signed `.rvf` agent packages.
//!
//! This crate is the Tauri v2 host described in ADR-289. It is split so that
//! every decision the user sees is made in a plain library module that can be
//! tested without a webview:
//!
//! - [`inspect`] — read and verify an `.rvf` through `rvf-forge-core`, without
//!   executing it (ADR-289 §3).
//! - [`receipts`] — the local witness log, one record per verification result.
//! - [`capability`] — derive the install-time capability contract (requirements P6).
//! - [`runtime`] — apply the FR004 runtime selection order (ADR-289 §2).
//! - [`state`] — encrypted state-capsule layout and sealing (ADR-288).
//! - [`install`] — verify, present the contract, and materialize an
//!   installation (requirements P6).
//! - [`library`] — the installed-agent registry and its operations
//!   (requirements P7).
//! - [`update`] — the semantic permission diff, the approval gate, and rollback
//!   (requirements P9, ADR-288 §7).
//! - [`dock`], [`dock_state`], [`dock_roster`], [`dock_events`] — the Agent
//!   Dock: what a running agent shows and how it is stopped (ADR-295).
//! - [`dock_bridge`] — the dock roster, built from the library.
//! - [`witness_receipt`], [`witness_view`] — the witness viewer: recompute each
//!   receipt's content address, check `prevReceipt` continuity per subject, and
//!   report where a chain stops verifying (requirements P15.11, P8).
//!
//! # Security invariants
//!
//! These hold for every module here:
//!
//! 1. **RVF content is never executed.** Inspection reads segment headers and
//!    hashes payloads. There is no code path in this crate that loads, links,
//!    or interprets a segment (ADR-284 §1 requirement 7).
//! 2. **Verification precedes any load, and every result is witnessed.**
//!    [`inspect::inspect`] alone reports
//!    [`inspect::VerificationStatus::Unverified`]; only [`inspect::verify`]
//!    reports a real outcome, and it appends a receipt for a refusal exactly as
//!    for a pass (ADR-284 §1 requirement 9). A signature that could not be
//!    checked reports [`inspect::SignatureStatus::NotChecked`], never
//!    `Verified`.
//! 3. **Capability rendering is default-deny.** A class that was not requested
//!    is rendered in the "cannot" list. A missing manifest yields a card where
//!    everything is denied, not an empty card.
//! 4. **No vague permission prose.** Broad scopes and phrases such as "access
//!    your computer" are rejected at derivation time (requirements P6).
//! 5. **No network calls.** Nothing in this crate opens a socket; the packaged
//!    application's CSP forbids remote content.
//! 6. **Runtime order is not configurable.** The selection order comes from the
//!    vendored compatibility matrix. No environment variable, CLI flag, or
//!    setting reorders it; only signed policy may, and signed policy is not yet
//!    implemented (ADR-289 §2).
//! 7. **Installation is gated on verification, and grants cannot exceed the
//!    declaration.** [`install::install`] refuses an unverified package and
//!    refuses an accepted class the verified container did not declare. Every
//!    lifecycle decision — install, update, rollback, uninstall, state export
//!    and import — appends a chained witness receipt, so a refusal is a record
//!    rather than an absence.
//! 8. **Removal does not silently destroy user state.** [`library::UninstallPolicy`]
//!    keeps state by default; deleting it requires asking, and the
//!    export-before-remove path aborts the whole uninstall if the export fails
//!    (ADR-294).
//! 9. **Agent input cannot reach dock chrome.** An agent supplies task text and
//!    progress and nothing else, through
//!    [`dock::AgentProvidedStatus::from_agent`]. Trust badge, network
//!    indicator, permission summary, witness status, resource usage, cost, and
//!    the runtime state live in [`dock::SystemOwnedStatus`], which has no
//!    constructor or setter that takes an agent-derived value (ADR-295 §7).

pub mod capability;
pub mod dock;
pub mod dock_bridge;
pub mod dock_events;
pub mod dock_roster;
pub mod dock_state;
pub mod dock_text;
pub mod inspect;
pub mod install;
pub mod library;
pub mod receipts;
pub mod runtime;
pub mod state;
pub mod update;
pub mod witness_receipt;
pub mod witness_view;

#[cfg(feature = "desktop")]
pub mod commands;

/// Launch the desktop shell.
#[cfg(feature = "desktop")]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        // Dock chrome state lives here, on the Rust side of the process
        // boundary, not in the webview that renders agent content.
        .manage(commands::DockSurface::default())
        // The library is the reader's own state, on the Rust side of the
        // process boundary for the same reason the dock roster is.
        .manage(library::Library::default())
        .invoke_handler(tauri::generate_handler![
            commands::inspect_rvf,
            commands::verify_rvf,
            commands::capability_card,
            commands::runtime_selection,
            commands::install_offer,
            commands::install_agent,
            commands::library_list,
            commands::library_open,
            commands::library_pause,
            commands::library_terminate,
            commands::library_reset,
            commands::library_verify,
            commands::library_archive,
            commands::library_restore,
            commands::library_export_state,
            commands::library_import_state,
            commands::library_uninstall,
            commands::update_plan,
            commands::update_apply,
            commands::update_rollback,
            commands::dock_sync,
            commands::dock_view,
            commands::dock_expand,
            commands::dock_pause,
            commands::dock_terminate,
            commands::dock_swarm_view,
            commands::dock_notifications,
            commands::dock_agent_report,
            commands::dock_add_scaffold_agent,
            commands::witness_chain,
        ])
        .run(tauri::generate_context!())
        .expect("error while running RVForge Reader");
}
