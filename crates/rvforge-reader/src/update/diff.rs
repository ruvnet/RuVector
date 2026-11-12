//! The semantic permission difference (requirements P9).
//!
//! Computed from two verified capability cards, never from what a release says
//! about itself, and biased toward calling an ambiguous change an expansion:
//! the failure mode of guessing the other way is a contract that grows without
//! anyone approving it.

use serde::{Deserialize, Serialize};

use crate::capability::{CapabilityCard, CardLine};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ChangeKind {
    Added,
    Removed,
    ScopeChanged,
}

/// One line of the P9 permission difference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CapabilityChange {
    pub class: String,
    pub kind: ChangeKind,
    pub from_scope: Option<String>,
    pub to_scope: Option<String>,
    /// The change gives the agent more than it had. Only expansions need
    /// approval.
    pub expansion: bool,
    /// The exact sentence shown to the user.
    pub text: String,
}

/// A `stateSchemaVersion` move across an update.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct SchemaChange {
    pub from: u32,
    pub to: u32,
}

/// What changes between two releases' contracts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PermissionDiff {
    pub changes: Vec<CapabilityChange>,
    pub state_schema_change: Option<SchemaChange>,
}

impl PermissionDiff {
    /// The changes that give the agent more than it had.
    pub fn expansions(&self) -> Vec<&CapabilityChange> {
        self.changes.iter().filter(|c| c.expansion).collect()
    }

    /// Classes the user must approve by name.
    pub fn expanding_classes(&self) -> Vec<String> {
        self.expansions().iter().map(|c| c.class.clone()).collect()
    }

    /// P9: users must approve any capability expansion.
    pub fn requires_approval(&self) -> bool {
        !self.expansions().is_empty()
    }

    /// The P9 rendering, one line per change.
    pub fn lines(&self) -> Vec<String> {
        let mut lines: Vec<String> = self.changes.iter().map(|c| c.text.clone()).collect();
        if let Some(s) = self.state_schema_change {
            lines.push(format!(
                "Changes memory schema from {} to {}",
                s.from, s.to
            ));
        }
        if lines.is_empty() {
            lines.push("No capability change; this release stays within the existing contract."
                .to_string());
        }
        lines
    }
}

/// What happens to accumulated state when the update is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum StateDisposition {
    /// The predecessor's capsule is kept and the new base starts empty.
    /// Migration is not implemented here (see the module docs).
    RetainedUnderPredecessor,
}

/// Everything the user needs to decide, computed before anything is written.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UpdatePlan {
    pub install_id: String,
    pub version_label: String,
    pub from_identity: String,
    pub to_identity: String,
    pub diff: PermissionDiff,
    pub requires_approval: bool,
    /// The classes an approval must name.
    pub approval_classes: Vec<String>,
    /// Whether rolling back off the *new* release will be possible.
    pub rollback_available_after: bool,
    pub state_disposition: StateDisposition,
    /// The full contract of the new release, so the P6 card can be re-shown.
    pub new_card: CapabilityCard,
}

/// The semantic difference between two capability cards.
///
/// A scope that becomes unstated is a widening; a scope that becomes stated is
/// a narrowing. Two different stated scopes are treated as an expansion,
/// because the reader cannot prove the new one is contained in the old one and
/// guessing in the permissive direction is how contracts quietly grow.
pub fn diff_cards(before: &CapabilityCard, after: &CapabilityCard) -> PermissionDiff {
    let mut changes = Vec::new();

    for line in &after.requests {
        match find(&before.requests, &line.class) {
            None => changes.push(CapabilityChange {
                class: line.class.clone(),
                kind: ChangeKind::Added,
                from_scope: None,
                to_scope: line.scope.clone(),
                expansion: true,
                text: format!("Adds {}", line.text.trim_start_matches("This agent ")),
            }),
            Some(old) if old.scope != line.scope => {
                let widened = line.scope.is_none() || old.scope.is_some();
                changes.push(CapabilityChange {
                    class: line.class.clone(),
                    kind: ChangeKind::ScopeChanged,
                    from_scope: old.scope.clone(),
                    to_scope: line.scope.clone(),
                    expansion: widened,
                    text: format!(
                        "Changes {} scope from {} to {}",
                        line.class,
                        scope_text(&old.scope),
                        scope_text(&line.scope)
                    ),
                });
            }
            Some(_) => {}
        }
    }

    for line in &before.requests {
        if find(&after.requests, &line.class).is_none() {
            changes.push(CapabilityChange {
                class: line.class.clone(),
                kind: ChangeKind::Removed,
                from_scope: line.scope.clone(),
                to_scope: None,
                expansion: false,
                text: format!("Removes {} access", line.class),
            });
        }
    }

    changes.sort_by(|a, b| a.class.cmp(&b.class));
    PermissionDiff {
        changes,
        state_schema_change: None,
    }
}

fn find<'a>(lines: &'a [CardLine], class: &str) -> Option<&'a CardLine> {
    lines.iter().find(|l| l.class == class)
}

fn scope_text(scope: &Option<String>) -> String {
    match scope {
        Some(s) => format!("'{s}'"),
        None => "unstated".to_string(),
    }
}

