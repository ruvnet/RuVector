//! Staged views: parsed/derived state bound to the exact artifact version it
//! came from (ADR-318 §Decision 2).

use serde::{Deserialize, Serialize};

use crate::artifact::{ArtifactRef, ContentHash};

/// The five downstream-reference kinds ADR-318 enumerates. Each one is a
/// piece of derived state that reads or acts on an artifact and must record
/// the content hash (and revision id) of the exact version it used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ViewKind {
    /// A parser's structured output over the artifact's native bytes
    /// (StagedWorkspace's "parsed records").
    ParserResult,
    /// A tool call whose input read the artifact.
    ToolCall,
    /// A review diff computed against the artifact
    /// (StagedWorkspace's "review diffs").
    Diff,
    /// An approval decision made while looking at the artifact.
    Approval,
    /// A generated output derived from the artifact.
    Output,
}

impl ViewKind {
    /// All five kinds, for exhaustive iteration in tests and audits.
    pub const ALL: [ViewKind; 5] = [
        ViewKind::ParserResult,
        ViewKind::ToolCall,
        ViewKind::Diff,
        ViewKind::Approval,
        ViewKind::Output,
    ];
}

/// A parsed/derived view bound to the exact artifact version it came from.
///
/// A `StagedView` is valid only while the workspace's live head for
/// `artifact.artifact_id` still has `artifact.content_hash`. The moment the
/// artifact changes, every view bound to the old hash is stale by
/// construction and is rejected fail-closed by
/// [`crate::WorkspaceState::validate`] / [`crate::WorkspaceState::execute`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StagedView {
    /// Workspace-scoped monotonic view id (audit handle; not the binding —
    /// the binding is `artifact`).
    pub view_id: u64,
    /// Which of ADR-318's five reference kinds this view is.
    pub kind: ViewKind,
    /// The exact artifact version (hash + revision) this view was derived
    /// from. This is the ADR-318 binding.
    pub artifact: ArtifactRef,
    /// SHA-256 of the view's own derived payload (parser output, diff text,
    /// approval record, ...), so the view's content is itself
    /// content-addressed and receipt-referenceable.
    pub payload_hash: ContentHash,
}
