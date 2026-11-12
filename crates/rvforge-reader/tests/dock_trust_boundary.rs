//! ADR-295 §7: agent content cannot become dock chrome.
//!
//! The structural half of this boundary is enforced by the type signatures and
//! is checked by the compiler rather than by a test:
//!
//! - [`AgentProvidedStatus`] has one constructor, `from_agent(&str, i64)`. It
//!   derives `Serialize` but not `Deserialize`, so no agent-supplied JSON
//!   object is ever deserialized into it — the agent's bytes only ever arrive
//!   as those two scalars.
//! - [`SystemOwnedStatus`] has private fields, no `Deserialize`, and one
//!   constructor, `measured(..)`, whose arguments are all system types. There
//!   is no function anywhere that takes an `AgentProvidedStatus` (or a `&str`
//!   originating from one) and returns or mutates a `SystemOwnedStatus`.
//!
//! What is left to test at runtime is the sanitizer and the invariant that the
//! one agent-facing mutator leaves the system half untouched.

use rvforge_reader::dock::{
    sanitize_agent_text, AgentDockEntry, AgentProvidedStatus, IsolationClass, NetworkIndicator,
    PermissionSummary, ResourceUsage, SanitizerFinding, SystemOwnedStatus, TrustBadge,
    WitnessStatus, MAX_TASK_CHARS, WITHHELD_TEXT,
};
use rvforge_reader::dock_state::DockState;

fn trusted_system() -> SystemOwnedStatus {
    SystemOwnedStatus::measured(
        DockState::Running,
        TrustBadge::VerifiedPublisher,
        NetworkIndicator::Disabled,
        PermissionSummary {
            isolation: IsolationClass::Wasm,
            local_model: true,
            network_allowed: false,
            selected_folder_access: true,
            encrypted_memory: true,
            grants: vec!["Read the files and folders you select".to_string()],
            pending_approvals: Vec::new(),
        },
        WitnessStatus::Valid,
        ResourceUsage {
            cpu_millis_per_sec: 120,
            memory_bytes: 512 * 1024 * 1024,
            tokens_in: 4_000,
            tokens_out: 900,
        },
        12_500,
        Some("local/llama".to_string()),
    )
}

fn entry() -> AgentDockEntry {
    AgentDockEntry::new("agent-1", "Cognitum Analyst", "analyst", trusted_system())
}

#[test]
fn an_agent_update_cannot_move_any_system_field() {
    let mut e = entry();
    let before = e.system().clone();

    // Every shape an agent could try, through the only channel it has.
    let very_long = "x".repeat(10_000);
    for hostile in [
        r#"{"state":"paused","trust":"verified-publisher","network":"disabled"}"#,
        "state=paused; trust=verified-publisher",
        "\u{1b}[2J\u{1b}[1;31mNetwork disabled\u{1b}[0m",
        very_long.as_str(),
    ] {
        e.apply_agent_update(hostile, 100);
        assert_eq!(
            e.system(),
            &before,
            "agent text '{}' changed a system-owned field",
            &hostile[..hostile.len().min(40)]
        );
    }

    assert_eq!(e.state(), DockState::Running);
}

#[test]
fn progress_outside_the_range_is_clamped_and_flagged() {
    let low = AgentProvidedStatus::from_agent("indexing", -50);
    assert_eq!(low.progress(), 0);
    assert!(low.progress_was_clamped());

    let high = AgentProvidedStatus::from_agent("indexing", 10_000);
    assert_eq!(high.progress(), 100);
    assert!(high.progress_was_clamped());

    let ok = AgentProvidedStatus::from_agent("indexing", 68);
    assert_eq!(ok.progress(), 68);
    assert!(!ok.progress_was_clamped());
}

#[test]
fn ansi_escape_sequences_are_stripped() {
    let t = sanitize_agent_text("\u{1b}[1;32mReviewing\u{1b}[0m 42 documents");
    assert_eq!(t.display(), "Reviewing 42 documents");
    assert!(!t.is_suspicious());
    assert!(t.findings().contains(&SanitizerFinding::AnsiEscapesRemoved));
    assert!(!t.display().contains('\u{1b}'));
}

#[test]
fn osc_sequences_cannot_smuggle_a_hyperlink_or_a_title() {
    let t = sanitize_agent_text("\u{1b}]0;RVForge\u{7}Reviewing 42 documents");
    assert_eq!(t.display(), "Reviewing 42 documents");
    assert!(t.findings().contains(&SanitizerFinding::AnsiEscapesRemoved));
}

#[test]
fn control_characters_and_line_breaks_never_survive() {
    let t = sanitize_agent_text("Reviewing\n42\tdocuments\u{0}\u{7}\u{202e}now");
    assert_eq!(t.display(), "Reviewing 42 documentsnow");
    assert!(t
        .findings()
        .contains(&SanitizerFinding::LineBreaksFlattened));
    assert!(t
        .findings()
        .contains(&SanitizerFinding::ControlCharactersRemoved));
    for c in t.display().chars() {
        assert!(!c.is_control(), "control character survived: {c:?}");
    }
}

#[test]
fn task_text_is_capped() {
    let t = sanitize_agent_text(&"a".repeat(MAX_TASK_CHARS * 5));
    assert_eq!(t.display().chars().count(), MAX_TASK_CHARS);
    assert!(t.findings().contains(&SanitizerFinding::Truncated));
    assert!(t.display().ends_with('…'));
}

#[test]
fn text_mimicking_a_system_label_is_withheld_not_rendered() {
    for mimic in [
        "Approved",
        "Paused — safe to close",
        "Stopped",
        "Terminated cleanly",
        "✓ Network disabled",
        "Verified publisher: Cognitum",
        "Quarantined by RVForge",
    ] {
        let t = sanitize_agent_text(mimic);
        assert!(t.is_suspicious(), "'{mimic}' should be flagged");
        assert_eq!(t.display(), WITHHELD_TEXT);
        assert_eq!(t.withheld().map(str::trim), Some(mimic.trim()));
        assert!(t
            .findings()
            .iter()
            .any(|f| matches!(f, SanitizerFinding::SystemLabelMimicry(_))));
    }
}

#[test]
fn mimicry_is_screened_after_escapes_are_stripped() {
    // An agent hiding "Paused" behind a CSI sequence is still claiming a state.
    let t = sanitize_agent_text("Pa\u{1b}[0mused");
    assert!(t.is_suspicious());
    assert_eq!(t.display(), WITHHELD_TEXT);
}

#[test]
fn ordinary_task_text_is_left_alone() {
    for ordinary in [
        "Reviewing 42 documents",
        "Waiting on the build to finish",
        "Summarising unapproved invoices",
        "Comparing approvals-workflow drafts",
    ] {
        let t = sanitize_agent_text(ordinary);
        assert!(!t.is_suspicious(), "'{ordinary}' should not be flagged");
        assert_eq!(t.display(), ordinary);
        assert!(t.withheld().is_none());
    }
}

#[test]
fn the_capability_card_is_derived_from_system_state_only() {
    let mut e = entry();
    e.apply_agent_update("✓ Verified publisher · Network disabled", 100);
    let card = e.capability_card();

    assert_eq!(card.lines.len(), 7);
    let labels: Vec<&str> = card.lines.iter().map(|l| l.label.as_str()).collect();
    assert_eq!(
        labels,
        [
            "Verified publisher",
            "RVM or WASM isolated",
            "Local model",
            "Network disabled",
            "Selected folder access",
            "Encrypted memory",
            "Witness chain valid",
        ]
    );
    assert!(card.lines.iter().all(|l| l.satisfied));

    // The same card against an unmeasured system says so on every line, whatever
    // the agent claims in its task text.
    let unknown = SystemOwnedStatus::measured(
        DockState::Running,
        TrustBadge::Unverified,
        NetworkIndicator::Active,
        PermissionSummary {
            isolation: IsolationClass::None,
            local_model: false,
            network_allowed: true,
            selected_folder_access: false,
            encrypted_memory: false,
            grants: Vec::new(),
            pending_approvals: Vec::new(),
        },
        WitnessStatus::Unavailable,
        ResourceUsage::ZERO,
        0,
        None,
    );
    let mut bare = AgentDockEntry::new("agent-2", "Loud Agent", "loud", unknown);
    bare.apply_agent_update("Network disabled and witness chain valid", 100);
    assert!(bare.capability_card().lines.iter().all(|l| !l.satisfied));
}

#[test]
fn a_fresh_entry_reports_no_task_rather_than_an_empty_pill() {
    let e = entry();
    assert_eq!(e.agent().task().display(), "No task reported");
    assert_eq!(e.agent().progress(), 0);
    assert!(!e.agent().is_suspicious());
}
